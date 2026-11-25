// Sampler (1.10.0): a token from each row of FP32 logits, on the GPU, into tokens[row] (uint32): greedy (argmax),
// or with temperature T, top-k (k <= 1024) and top-p. The logits never leave the GPU; the token buffer can feed the
// next step's embedding gather directly. One group of 1024 threads per row.
//
// Order of operations (Sampler.reference computes the same with Math.exp):
// - max: the largest logit, ties to the lowest index; NaN counts as -infinity. Greedy (T <= 0 or k == 1) stops here.
// - weights w = exp((l - max) * (1 / T)), the reciprocal from the host.
// - without top-k and top-p: the weights in index order (thread t a contiguous chunk, then an exclusive scan of the
//   threads' sums); with: the k largest logits (radix select on order-preserving keys; ties at the k-th taken in
//   index order), sorted by logit descending then index, their weights scanned in that order, cut at the first whose
//   running sum reaches p times the total.
// - u = rand * total, rand from a counter-based hash of (seed, row, step) with 24 bits: the token is the first whose
//   running sum exceeds u.
// The sums are in fixed orders, so a device gives the same token for the same seed and step every time. exp is the
// device's, so two devices agree unless u falls within their exp error of a boundary.
#include "dynamis/kit.slang"

struct SampleParams
{
    uint rows;
    uint vocab;
    uint ldl;
    float invT;      // 1 / temperature, 0 for greedy
    uint topK;       // 0: none (all the vocabulary); else 1 .. 1024
    float topP;      // >= 1: none
    uint seedLo;
    uint seedHi;
    uint step;
    uint gridX;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> logits;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> tokens;
[[vk::push_constant]] ConstantBuffer<SampleParams> p;

static const uint THREADS = 1024;
static const uint KMAX = 1024;

groupshared float sv[THREADS];      // reductions, scans of weights; the sort's logits
groupshared uint si[THREADS];       // the max's index; count scans; the candidates' indices
groupshared uint hist[256];
groupshared float gMax;
groupshared uint gIdx, gPrefix, gRemain, gFound;
groupshared float gTotal, gU;

// order-preserving key: a larger logit has a larger key; NaN the smallest
uint key(float l) { uint f = asuint(l); return dg_is_nan(l) ? 0u : ((f & 0x80000000u) != 0 ? ~f : f | 0x80000000u); }

// murmur3's finalizer
uint fmix(uint h) { h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16; return h; }
// uniform in [0, 1) with 24 bits, from (seed, row, step)
float rand01(uint row)
{
    uint h = fmix(p.seedLo ^ fmix(p.seedHi ^ fmix(row * 0x9E3779B9u ^ fmix(p.step + 0x632BE5ABu))));
    return float(h >> 8) * asfloat(0x33800000u);   // 2^-24
}

// Is (va, ia) before (vb, ib): larger value first, then lower index
bool before(float va, uint ia, float vb, uint ib) { return va > vb || (va == vb && ia < ib); }

[shader("compute")]
[numthreads(1024, 1, 1)]
void sample_rows(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint row = dg_linear_group(group, p.gridX);
    if (row >= p.rows) return;   // uniform
    uint t = local.x, base = row * p.ldl, n = p.vocab;

    // the max, ties to the lowest index, NaN as -infinity
    float bv = dg_negative_infinity(); uint bi = 0xFFFFFFFFu;
    for (uint j = t; j < n; j += THREADS)
    {
        float l = logits[base + j];
        if (!dg_is_nan(l) && (bi == 0xFFFFFFFFu || l > bv)) { bv = l; bi = j; }
    }
    sv[t] = bv; si[t] = bi;
    [ForceUnroll]
    for (uint w = THREADS / 2; w > 0; w /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        if (t < w)
        {
            float ov = sv[t + w]; uint oi = si[t + w];
            if (oi != 0xFFFFFFFFu && (si[t] == 0xFFFFFFFFu || before(ov, oi, sv[t], si[t]))) { sv[t] = ov; si[t] = oi; }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) { gMax = sv[0]; gIdx = si[0] == 0xFFFFFFFFu ? 0u : si[0]; }
    GroupMemoryBarrierWithGroupSync();
    float mx = gMax;
    uint argmax = gIdx;
    if (p.invT <= 0.0f || p.topK == 1 || mx == dg_negative_infinity())
    {
        if (t == 0) tokens[row] = argmax;
        return;
    }

    bool restricted = p.topK != 0 || p.topP < 1.0f;
    if (!restricted)
    {
        // the whole vocabulary in index order: contiguous chunks, then an exclusive scan of their sums
        uint chunk = (n + THREADS - 1) / THREADS, j0 = t * chunk, j1 = min(n, j0 + chunk);
        float part = 0.0f;
        for (uint j = j0; j < j1; j++) part = part + exp((logits[base + j] - mx) * p.invT);
        sv[t] = part;
        DG_GROUP_SCAN_EXCLUSIVE(sv, t, THREADS, dg_op_add, 0.0f);
        if (t == THREADS - 1) gTotal = sv[t] + part;
        if (t == 0) gFound = 0xFFFFFFFFu;
        GroupMemoryBarrierWithGroupSync();
        float u = rand01(row) * gTotal, run = sv[t];
        if (part > 0.0f && u < run + part)
        {
            for (uint j = j0; j < j1; j++)
            {
                run = run + exp((logits[base + j] - mx) * p.invT);
                if (run > u) { InterlockedMin(gFound, j); break; }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (t == 0) tokens[row] = gFound == 0xFFFFFFFFu ? argmax : gFound;
        return;
    }

    // the k largest: radix select of the k-th largest key, 8 bits at a time
    uint k = p.topK == 0 ? min(n, KMAX) : min(p.topK, n);
    if (t == 0) { gPrefix = 0; gRemain = k; }
    for (uint shift = 24; ; shift -= 8)
    {
        for (uint b = t; b < 256; b += THREADS) hist[b] = 0;
        GroupMemoryBarrierWithGroupSync();
        uint prefix = gPrefix;
        uint mask = shift == 24 ? 0u : ~0u << (shift + 8);
        for (uint j = t; j < n; j += THREADS)
        {
            uint kv = key(logits[base + j]);
            if ((kv & mask) == (prefix & mask)) InterlockedAdd(hist[(kv >> shift) & 255u], 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (t == 0)
        {
            uint remain = gRemain, b = 255;
            for (;; b--)
            {
                uint c = hist[b];
                if (c >= remain || b == 0) break;
                remain -= c;
            }
            gPrefix = prefix | (b << shift);
            gRemain = remain;   // of the keys in this bin, how many are taken (all keys above it are)
        }
        GroupMemoryBarrierWithGroupSync();
        if (shift == 0) break;
    }
    uint threshold = gPrefix, ties = gRemain;   // keys > threshold, plus the first `ties` keys == threshold

    // compact the candidates in index order: contiguous chunks; scan the ties, then what each chunk takes
    uint chunk = (n + THREADS - 1) / THREADS, j0 = t * chunk, j1 = min(n, j0 + chunk);
    uint greater = 0, equal = 0;
    for (uint j = j0; j < j1; j++)
    {
        uint kv = key(logits[base + j]);
        greater += kv > threshold ? 1u : 0u;
        equal += kv == threshold ? 1u : 0u;
    }
    si[t] = equal;
    DG_GROUP_SCAN_EXCLUSIVE(si, t, THREADS, dg_op_addu, 0u);
    uint tiesBefore = si[t];
    GroupMemoryBarrierWithGroupSync();
    uint takeTies = ties > tiesBefore ? min(equal, ties - tiesBefore) : 0u;
    si[t] = greater + takeTies;
    DG_GROUP_SCAN_EXCLUSIVE(si, t, THREADS, dg_op_addu, 0u);
    uint at = si[t];
    GroupMemoryBarrierWithGroupSync();
    // empty slots sort last
    sv[t] = dg_negative_infinity(); si[t] = 0xFFFFFFFFu;
    GroupMemoryBarrierWithGroupSync();
    uint tieSeen = 0;
    for (uint j = j0; j < j1; j++)
    {
        float l = logits[base + j];
        uint kv = key(l);
        bool take = kv > threshold || (kv == threshold && tieSeen++ < takeTies);
        if (take) { sv[at] = dg_is_nan(l) ? dg_negative_infinity() : l; si[at] = j; at++; }
    }
    GroupMemoryBarrierWithGroupSync();

    // bitonic sort of the 1024 slots: larger logit first, then lower index
    for (uint size = 2; size <= THREADS; size *= 2)
    {
        for (uint stride = size / 2; stride > 0; stride /= 2)
        {
            uint partner = t ^ stride;
            if (partner > t)
            {
                bool descending = (t & size) == 0;
                float va = sv[t], vb = sv[partner]; uint ia = si[t], ib = si[partner];
                bool swap = descending ? before(vb, ib, va, ia) : before(va, ia, vb, ib);
                if (swap) { sv[t] = vb; sv[partner] = va; si[t] = ib; si[partner] = ia; }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // the candidates' weights in sorted order, an inclusive scan, the top-p cut, and the draw
    uint myIndex = si[t];
    float wgt = t < k ? exp((sv[t] - mx) * p.invT) : 0.0f;
    GroupMemoryBarrierWithGroupSync();
    sv[t] = wgt;
    DG_GROUP_SCAN_EXCLUSIVE(sv, t, THREADS, dg_op_add, 0.0f);
    float incl = sv[t] + wgt;   // the running sum through this candidate
    if (t == THREADS - 1) gTotal = incl;
    if (t == 0) gFound = 0xFFFFFFFFu;
    GroupMemoryBarrierWithGroupSync();
    float total = gTotal;
    // the cut: the first candidate whose running sum reaches p * total
    if (t < k && incl >= p.topP * total && (t == 0 || sv[t] < p.topP * total)) gFound = t;
    if (p.topP >= 1.0f && t == 0) gFound = k - 1;
    GroupMemoryBarrierWithGroupSync();
    uint last = gFound == 0xFFFFFFFFu ? k - 1 : gFound;
    GroupMemoryBarrierWithGroupSync();
    if (t == last) gU = rand01(row) * incl;   // the kept candidates' total
    if (t == 0) gFound = 0xFFFFFFFFu;
    GroupMemoryBarrierWithGroupSync();
    float u = gU;
    if (t <= last && incl > u && (t == 0 || sv[t] <= u)) InterlockedMin(gFound, t);
    GroupMemoryBarrierWithGroupSync();
    uint pick = gFound == 0xFFFFFFFFu ? last : gFound;   // u rounded up to the total: the last kept candidate
    if (t == pick) tokens[row] = myIndex;
}
