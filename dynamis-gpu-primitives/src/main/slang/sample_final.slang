// Sampler, stage 2 (1.10.0): one group of 1024 threads per row combines stage 1's chunks (sample_chunks.slang) into a
// token. Sampler.reference computes the same with Math.exp.
// - max: the chunks' maxima in a fixed tree (ties to the lower chunk, so the lowest index). Greedy stops here.
// - without top-k and top-p: chunk g's weight w_g = s_g exp((m_g - max) / T), an exclusive scan over the chunks,
//   u = rand * total; the chunk holding u, then within it the weights exp((l - max) / T) scanned, and the first whose
//   running sum exceeds u's place in the chunk, (u - before) / w_g times the chunk's sum.
// - with them: the k largest, merged from the lists the merge rounds left (sample_merge.slang, sample_tree.slang), their
//   weights scanned, cut at top-p, and drawn as before. (Selecting them here instead, by a radix select whose group-memory atomics all hit
//   the few bins the largest keys share, took 67 of this stage's 84 us at a vocabulary of 128256.)
#include "sample_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> logits;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> sortedV;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> sortedI;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint4> info;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> tokens;
[[vk::push_constant]] ConstantBuffer<SampleParams> p;

static const uint SLOTS = THREADS;
groupshared float sv[SLOTS];
groupshared uint si[SLOTS];
#include "sample_tree.slang"
groupshared float gMax, gTotal, gU;
groupshared uint gIdx, gFound, gLast, gPrefix, gCount;

uint pow2Above(uint x) { uint s = 1; while (s < x) s *= 2; return s; }

// The kit's Blelloch exclusive scan over the first `size` slots (a power of two): the same bits as over all 1024 with
// zeros beyond, in fewer steps. Every thread calls it; it ends with a barrier.
void scanF(uint t, uint size)
{
    for (uint d = 1; d < size; d *= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        uint i = (t + 1) * d * 2 - 1;
        if (i < size) sv[i] = sv[i - d] + sv[i];
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) sv[size - 1] = 0.0f;
    for (uint d2 = size / 2; d2 > 0; d2 /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        uint i = (t + 1) * d2 * 2 - 1;
        if (i < size) { uint l = i - d2; let sum = sv[i] + sv[l]; sv[l] = sv[i]; sv[i] = sum; }
    }
    GroupMemoryBarrierWithGroupSync();
}



[shader("compute")]
[numthreads(1024, 1, 1)]
void sample_final(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint row = group.x, t = local.x, G = p.chunks, n = p.vocab;
    uint4 ci = t < G ? info[row * G + t] : uint4(asuint(dg_negative_infinity()), NONE, 0u, 0u);
    float mg = asfloat(ci.x), sg = asfloat(ci.z);

    // the row's max over the chunks
    uint G2 = pow2Above(G);
    sv[t] = mg; si[t] = ci.y;
    for (uint w = G2 / 2; w > 0; w /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        if (t < w && before(sv[t + w], si[t + w], sv[t], si[t])) { sv[t] = sv[t + w]; si[t] = si[t + w]; }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) { gMax = sv[0]; gIdx = si[0] == NONE ? 0u : si[0]; }
    GroupMemoryBarrierWithGroupSync();
    float mx = gMax;
    uint argmax = gIdx;
    if (p.invT <= 0.0f || p.topK == 1 || mx == dg_negative_infinity())
    {
        if (t == 0) tokens[row] = argmax;
        return;
    }
    float rnd = rand01(p.seedLo, p.seedHi, row, p.step);
    GroupMemoryBarrierWithGroupSync();

    if (p.restricted == 0)
    {
        // the chunk holding the draw
        float wg = t < G && sg > 0.0f ? sg * exp((mg - mx) * p.invT) : 0.0f;
        sv[t] = wg;
        scanF(t, G2);
        float before_g = sv[t];
        if (t == G2 - 1) gTotal = before_g + wg;
        if (t == 0) { gFound = NONE; gLast = 0; }
        GroupMemoryBarrierWithGroupSync();
        float u = rnd * gTotal;
        if (wg > 0.0f) InterlockedMax(gLast, t);
        if (wg > 0.0f && u >= before_g && u < before_g + wg) InterlockedMin(gFound, t);
        GroupMemoryBarrierWithGroupSync();
        uint g = gFound == NONE ? gLast : gFound;
        if (t == g) { gU = (u - before_g) / wg; }   // the draw's place in the chunk, in [0, 1)
        GroupMemoryBarrierWithGroupSync();
        float frac = gU;
        GroupMemoryBarrierWithGroupSync();
        // within the chunk: its weights against the row's max, in index order
        uint j = g * CHUNK + t;
        float e = j < n ? exp((sanitize(logits[row * p.ldl + j]) - mx) * p.invT) : 0.0f;
        sv[t] = e;
        DG_GROUP_SCAN_EXCLUSIVE(sv, t, THREADS, dg_op_add, 0.0f);
        float ex = sv[t];
        if (t == THREADS - 1) gTotal = ex + e;
        if (t == 0) { gFound = NONE; gLast = 0; }
        GroupMemoryBarrierWithGroupSync();
        float target = frac * gTotal;
        if (e > 0.0f) InterlockedMax(gLast, t);
        if (e > 0.0f && ex + e > target && ex <= target) InterlockedMin(gFound, t);
        GroupMemoryBarrierWithGroupSync();
        if (t == 0) tokens[row] = g * CHUNK + (gFound == NONE ? gLast : gFound);
        return;
    }

    // the k largest, sorted: the row's last lists after the merge rounds (sample_merge.slang), merged here (a power of
    // two of lists of c entries within 1024 slots)
    uint k = p.topK == 0 ? min(n, THREADS) : min(p.topK, n);
    uint K2 = pow2Above(k);
    uint c = min(k, p.stride), P = pow2Above(p.lists), slots = P * c;
    LOAD_LISTS(sortedV, sortedI, row, p.lists, 0u, p.lists, p.stride, c, slots, t)
    mergeTree(t, c, slots);
    float cv = t < k ? sv[t] : dg_negative_infinity(); uint cix = t < k ? si[t] : NONE;
    GroupMemoryBarrierWithGroupSync();
    sv[t] = cv; si[t] = cix;
    GroupMemoryBarrierWithGroupSync();

    // their weights in order, an inclusive scan, the top-p cut, and the draw
    uint myIndex = si[t];
    float wgt = t < k ? exp((sv[t] - mx) * p.invT) : 0.0f;
    GroupMemoryBarrierWithGroupSync();
    sv[t] = wgt;
    scanF(t, K2);
    float incl = sv[t] + wgt;
    if (t == K2 - 1) gTotal = incl;
    if (t == 0) gFound = NONE;
    GroupMemoryBarrierWithGroupSync();
    float cut = p.topP * gTotal;
    if (p.topP < 1.0f && t < k && incl >= cut && (t == 0 || sv[t] < cut)) gFound = t;
    GroupMemoryBarrierWithGroupSync();
    uint last = gFound == NONE ? k - 1 : gFound;
    GroupMemoryBarrierWithGroupSync();
    if (t == last) gU = rnd * incl;
    if (t == 0) gFound = NONE;
    GroupMemoryBarrierWithGroupSync();
    float u = gU;
    if (t <= last && incl > u && (t == 0 || sv[t] <= u)) InterlockedMin(gFound, t);
    GroupMemoryBarrierWithGroupSync();
    uint pick = gFound == NONE ? last : gFound;
    if (t == pick) tokens[row] = myIndex;
}
