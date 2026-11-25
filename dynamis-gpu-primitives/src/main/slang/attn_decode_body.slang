// Fused attention for few rows per KV head (1.11.0): decoding, where a KV head's rows (its query heads' queries, at
// most RMAX) are too few to fill attn_flash_body's 16-row groups. A group of 128 threads takes one range of
// splitKeys keys of one KV head (at most SK / rows) and keeps all their scores in group memory, so there is no
// rescaling within it:
//   - scores: thread t takes keys t, t + 128, .. and computes every row's dot product with it (K read once);
//   - per row, the max and the sum of P = exp(s - max) in the kit's fixed trees, P kept;
//   - P V: thread t takes column pair t % D2 and keys t / D2, t / D2 + TPP, .. (V rows read once, coalesced), for
//     every row; the TPP parts are added in order.
// It writes (m, l, accumulators) for attn_combine when the keys are split, else O = acc / l. The entry file defines
// DMAX, ENTRY and OUT_F32.
#include "attn_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> q;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> k;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> v;
#if OUT_F32
[[vk::binding(3, 0)]] RWStructuredBuffer<float> o;
#else
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> o;
#endif
[[vk::binding(4, 0)]] RWStructuredBuffer<float> partialML;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> partialAcc;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> table;        // paged K and V: the sequence's pages
[[vk::push_constant]] ConstantBuffer<AttnParams> p;

static const uint THREADS = 128;
static const uint RMAX = 8;
static const uint SK = 1024;              // scores kept: rows * splitKeys <= SK
static const uint D2 = DMAX / 2;
static const uint TPP = THREADS / D2;     // threads per column pair, each over its share of the keys
static const uint VB = 8;                // V loads in flight per thread

groupshared uint qs[RMAX * D2];
groupshared float ss[SK];
groupshared float red[THREADS];
groupshared float rowM[RMAX];
groupshared float rowL[RMAX];

[shader("compute")]
[numthreads(128, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = local.x, kvHead = group.y, split = group.z;
    uint hpk = p.heads / p.kvHeads, rows = hpk * p.queries;   // at most RMAX
    uint d2 = p.headDim / 2, offset = p.keys - p.queries;
    uint kStart = split * p.splitKeys, kEnd = min(p.keys, kStart + p.splitKeys);
    uint n = kEnd > kStart ? kEnd - kStart : 0u;
    uint sk = p.splitKeys;   // the stride of a row's scores

    for (uint e = t; e < RMAX * D2; e += THREADS)
    {
        uint r = e / D2, pr = e % D2, w = 0;
        if (r < rows && pr < d2)
        {
            uint h = kvHead * hpk + r / p.queries, i = r % p.queries;
            w = q[(h * p.queries + i) * d2 + pr];
        }
        qs[e] = w;
    }
    GroupMemoryBarrierWithGroupSync();

    // the scores
    for (uint j = t; j < n; j += THREADS)
    {
        uint key = kStart + j, kb = ATTN_KV_WORD(p, table, kvHead, key, 0u, d2);
        float s[RMAX];
        [ForceUnroll] for (uint r = 0; r < RMAX; r++) s[r] = 0.0f;
        for (uint p0 = 0; p0 < D2; p0 += 8)
        {
            [ForceUnroll]
            for (uint pi = 0; pi < 8; pi++)
            {
                uint pr = p0 + pi;
                uint kw = pr < d2 ? k[kb + pr] : 0u;
                float klo = attnLo(kw), khi = attnHi(kw);
                [ForceUnroll]
                for (uint r = 0; r < RMAX; r++)
                {
                    uint qw = qs[r * D2 + pr];
                    s[r] = fma(attnLo(qw), klo, s[r]);   // FP16 products are exact
                    s[r] = fma(attnHi(qw), khi, s[r]);
                }
            }
        }
        [ForceUnroll]
        for (uint r = 0; r < RMAX; r++)
        {
            if (r >= rows) break;
            uint i = r % p.queries;
            bool visible = p.causal == 0 || key <= i + offset;
            ss[r * sk + j] = visible ? s[r] * p.scale : dg_negative_infinity();
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // each row's max and sum, P in place: a thread's keys in order, then the kit's fixed tree (one tree for all rows,
    // with RMAX times the group memory, was slower: 87 against 57 us for 4096 keys)
    for (uint r = 0; r < rows; r++)
    {
        float acc = dg_negative_infinity();
        for (uint j = t; j < n; j += THREADS) acc = max(acc, ss[r * sk + j]);
        red[t] = acc;
        DG_GROUP_REDUCE(red, t, THREADS, dg_op_max)
        if (t == 0) rowM[r] = red[0];
        GroupMemoryBarrierWithGroupSync();
        float mx = rowM[r];
        acc = 0.0f;
        for (uint j = t; j < n; j += THREADS)
        {
            float sv = ss[r * sk + j];
            float pj = sv == dg_negative_infinity() ? 0.0f : exp(sv - mx);
            ss[r * sk + j] = pj;
            acc = acc + pj;
        }
        GroupMemoryBarrierWithGroupSync();
        red[t] = acc;
        DG_GROUP_REDUCE(red, t, THREADS, dg_op_add)
        if (t == 0) rowL[r] = red[0];
        GroupMemoryBarrierWithGroupSync();
    }

    // P V: column pair t % D2 over keys t / D2, t / D2 + TPP, ..
    uint pair = t % D2, part = t / D2;
    float lo[RMAX], hi[RMAX];
    [ForceUnroll] for (uint r = 0; r < RMAX; r++) { lo[r] = 0.0f; hi[r] = 0.0f; }
    if (pair < d2)
    {
        // eight keys' V words loaded before any is used: one load at a time left the loop waiting on memory
        for (uint j0 = part; j0 < n; j0 += VB * TPP)
        {
            uint vw[VB];
            [ForceUnroll] for (uint u = 0; u < VB; u++) { uint j = j0 + u * TPP; vw[u] = j < n ? v[ATTN_KV_WORD(p, table, kvHead, kStart + j, pair, d2)] : 0u; }
            [ForceUnroll]
            for (uint u = 0; u < VB; u++)
            {
                uint j = j0 + u * TPP;
                if (j >= n) break;
                float vlo = attnLo(vw[u]), vhi = attnHi(vw[u]);
                [ForceUnroll]
                for (uint r = 0; r < RMAX; r++)
                {
                    if (r >= rows) break;
                    float pj = ss[r * sk + j];
                    lo[r] = fma(pj, vlo, lo[r]);
                    hi[r] = fma(pj, vhi, hi[r]);
                }
            }
        }
    }
    // the TPP parts of each row added in order, one row at a time through the score memory (TPP * DMAX = 256 floats)
    for (uint r = 0; r < rows; r++)
    {
        GroupMemoryBarrierWithGroupSync();   // the scores, or the previous row's parts, are no longer read
        float a = 0.0f, b = 0.0f;
        [ForceUnroll] for (uint rr = 0; rr < RMAX; rr++) if (rr == r) { a = lo[rr]; b = hi[rr]; }
        ss[part * DMAX + 2 * pair] = a;
        ss[part * DMAX + 2 * pair + 1] = b;
        GroupMemoryBarrierWithGroupSync();
        if (part != 0 || pair >= d2) continue;
        a = 0.0f; b = 0.0f;
        [ForceUnroll]
        for (uint pt = 0; pt < TPP; pt++) { a = a + ss[pt * DMAX + 2 * pair]; b = b + ss[pt * DMAX + 2 * pair + 1]; }
        uint h = kvHead * hpk + r / p.queries, i = r % p.queries, row = h * p.queries + i;
        if (p.splits > 1)
        {
            uint pr = split * p.heads * p.queries + row;
            if (pair == 0) { partialML[2 * pr] = n == 0 ? dg_negative_infinity() : rowM[r]; partialML[2 * pr + 1] = n == 0 ? 0.0f : rowL[r]; }
            partialAcc[pr * p.headDim + 2 * pair] = a;
            partialAcc[pr * p.headDim + 2 * pair + 1] = b;
            continue;
        }
        float l = rowL[r];
#if OUT_F32
        o[row * p.headDim + 2 * pair] = a / l;
        o[row * p.headDim + 2 * pair + 1] = b / l;
#else
        o[row * d2 + pair] = attnPack(a / l, b / l);
#endif
    }
}
