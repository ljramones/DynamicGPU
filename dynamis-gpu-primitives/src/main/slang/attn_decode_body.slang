// Fused attention for few rows per KV head (1.11.0): decoding, where a KV head's rows (its query heads' queries, at
// most RMAX) are too few to fill attn_flash_body's 16-row groups. A group of 128 threads takes one range of
// splitKeys keys of one KV head (at most SK / rows) and keeps all their scores in group memory, so there is no
// rescaling within it:
//   - scores: thread t takes keys t, t + 128, .. and computes every row's dot product with it (K read once);
//   - per row, the max and the sum of P = exp(s - max) in the kit's fixed trees, P kept;
//   - P V: thread t takes column pair t % D2 and keys t / D2, t / D2 + TPP, .. (V rows read once, coalesced), for
//     every row; the TPP parts are added in order.
// It writes (m, l, accumulators) for attn_combine when the keys are split, else O = acc / l. The entry file defines
// DMAX, ENTRY and OUT_F32. ROW_TIER (1, 4 or 8) removes unused rows at pipeline compilation. Batched (1.11.1): group
// x is the sequence, whose tokens and pages come from the table's header (attn_common.slang); a sequence with fewer
// keys than a range leaves that range empty (m = -infinity, l = 0) for the combine.
// M4 Max Metal, 32/8 heads, D128, 512 / 4096 / 32768 keys: 64.8 / 94.6 / 410.4 us with room for eight
// rows reduced to 51.4 / 71.8 / 345.2 us by specializing to four. Active rows use the same arithmetic.
// WAVE_ROWS assigns rows to subgroups and reduces each FP32 softmax there, measuring 41.4 / 65.3 / 329.3 us.
// On MoltenVK (M4 Max, same shapes), release / specialized / subgroup totals were respectively
// 50.6 / 78.7 / 432.9 us, 46.4 / 63.7 / 359.4 us and 39.9 / 58.0 / 335.6 us.
// The FP32 weight sum rounds in subgroup order, within the scalar fused bound. Devices without subgroup
// arithmetic use the fixed tree below.
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
[vk::constant_id(0)] const uint ROW_TIER = 8;   // maximum rows processed by the specialized pipeline
static const uint SK = 1024;              // scores kept: rows * splitKeys <= SK
static const uint D2 = DMAX / 2;
static const uint TPP = THREADS / D2;     // threads per column pair, each over its share of the keys
static const uint VB = 8;                // V loads in flight per thread

groupshared uint qs[RMAX * D2];
groupshared float ss[SK];
#ifdef WAVE_ROWS
groupshared uint waves;
#else
groupshared float red[THREADS];
#endif
groupshared float rowM[RMAX];    // running max and sum of each row over the range's chunks
groupshared float rowL[RMAX];
groupshared float rowA[RMAX];    // this chunk's rescale of the row: exp(old max - new max)

[shader("compute")]
[numthreads(128, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = local.x, kvHead = group.y, split = group.z, seq = group.x;
    uint hpk = p.heads / p.kvHeads, rows = min(hpk * p.queries, ROW_TIER);   // at most RMAX
    // one sequence, or sequence `seq` of a batch: its keys, its pages in the table, its rows of Q and O
    uint keys = p.batch == 0 ? p.keys : table[2 * seq];
    uint pageBase = p.batch == 0 ? 0u : 2 * p.batch + table[2 * seq + 1];
    uint rowBase = seq * p.heads * p.queries, totalRows = max(p.batch, 1u) * p.heads * p.queries;
    uint d2 = p.headDim / 2, offset = keys - p.queries;
    uint kStart = split * p.splitKeys, kEnd = min(keys, kStart + p.splitKeys);
    uint n = kEnd > kStart ? kEnd - kStart : 0u;
    // a range is taken in chunks of the scores the group keeps; one chunk unless the range is longer (batched decode)
    uint chunk = max(128u, SK / max(rows, 1u) / 128u * 128u);
    uint sk = chunk;   // the stride of a row's scores

#ifdef WAVE_ROWS
    if (t == 0) waves = 0;
#endif
    if (t < RMAX) { rowM[t] = dg_negative_infinity(); rowL[t] = 0.0f; }
    for (uint e = t; e < ROW_TIER * D2; e += THREADS)
    {
        uint r = e / D2, pr = e % D2, w = 0;
        if (r < rows && pr < d2)
        {
            uint h = kvHead * hpk + r / p.queries, i = r % p.queries;
            w = q[(rowBase + h * p.queries + i) * d2 + pr];
        }
        qs[e] = w;
    }
    GroupMemoryBarrierWithGroupSync();

#ifdef WAVE_ROWS
    // Each subgroup owns its rows and can reduce without a workgroup barrier. Elect dense IDs
    // once, independently of the mapping between subgroup lanes and local invocation indices.
    uint wid = 0;
    uint width = WaveActiveSum(1u), lane = WavePrefixSum(1u);
    if (lane == 0) InterlockedAdd(waves, 1u, wid);
    wid = WaveActiveSum(wid);
#endif
    // the P V accumulators: column pair t % D2 over keys t / D2, t / D2 + TPP, ..
    uint pair = t % D2, part = t / D2;
    float lo[RMAX], hi[RMAX];
    [ForceUnroll] for (uint r = 0; r < RMAX; r++) { lo[r] = 0.0f; hi[r] = 0.0f; }
    for (uint c0 = kStart; c0 < kEnd; c0 += chunk)
    {
        uint n = min(chunk, kEnd - c0);   // this chunk's keys (the range's are counted outside)
        // the scores
        for (uint j = t; j < n; j += THREADS)
        {
            uint key = c0 + j, kb = ATTN_KV_WORD(p, table, pageBase, kvHead, key, 0u, d2);
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
                        if (r < ROW_TIER)
                        {
                            uint qw = qs[r * D2 + pr];
                            s[r] = fma(attnLo(qw), klo, s[r]);   // FP16 products are exact
                            s[r] = fma(attnHi(qw), khi, s[r]);
                        }
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
    
    #ifdef WAVE_ROWS
        for (uint r = wid; r < rows; r += waves)
        {
            float m = dg_negative_infinity();
            for (uint j = lane; j < n; j += width) m = max(m, ss[r * sk + j]);
            m = WaveActiveMax(m);
            float mOld = rowM[r], mNew = max(mOld, m);
            m = mNew;
            float sum = 0.0f;
            for (uint j = lane; j < n; j += width)
            {
                float sv = ss[r * sk + j];
                float w = sv == dg_negative_infinity() ? 0.0f : exp(sv - m);
                ss[r * sk + j] = w;
                sum += w;
            }
            sum = WaveActiveSum(sum);
            float alpha = mOld == dg_negative_infinity() ? 0.0f : exp(mOld - mNew);
            if (lane == 0) { rowA[r] = alpha; rowL[r] = rowL[r] * alpha + sum; rowM[r] = mNew; }
        }
        GroupMemoryBarrierWithGroupSync();
    #else
        // each row's max and sum, P in place: a thread's keys in order, then the kit's fixed tree (one tree for all rows,
        // with RMAX times the group memory, was slower: 87 against 57 us for 4096 keys)
        for (uint r = 0; r < rows; r++)
        {
            float acc = dg_negative_infinity();
            for (uint j = t; j < n; j += THREADS) acc = max(acc, ss[r * sk + j]);
            red[t] = acc;
            DG_GROUP_REDUCE(red, t, THREADS, dg_op_max)
            if (t == 0)
            {
                float mOld = rowM[r], mNew = max(mOld, red[0]);
                rowA[r] = mOld == dg_negative_infinity() ? 0.0f : exp(mOld - mNew);
                rowM[r] = mNew;
            }
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
            if (t == 0) rowL[r] = rowL[r] * rowA[r] + red[0];
            GroupMemoryBarrierWithGroupSync();
        }
    
    #endif
    
        // P V: column pair t % D2 over keys t / D2, t / D2 + TPP, .., the accumulators rescaled to the new max first
        [ForceUnroll] for (uint r = 0; r < RMAX; r++) { if (r < rows) { float a = rowA[r]; lo[r] = lo[r] * a; hi[r] = hi[r] * a; } }
        if (pair < d2)
        {
            // eight keys' V words loaded before any is used: one load at a time left the loop waiting on memory
            for (uint j0 = part; j0 < n; j0 += VB * TPP)
            {
                uint vw[VB];
                [ForceUnroll] for (uint u = 0; u < VB; u++) { uint j = j0 + u * TPP; vw[u] = j < n ? v[ATTN_KV_WORD(p, table, pageBase, kvHead, c0 + j, pair, d2)] : 0u; }
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
        GroupMemoryBarrierWithGroupSync();   // the chunk's P is no longer read
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
        uint h = kvHead * hpk + r / p.queries, i = r % p.queries, row = rowBase + h * p.queries + i;
        if (p.splits > 1)
        {
            uint pr = split * totalRows + row;
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
