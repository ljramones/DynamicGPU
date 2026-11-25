// Fused attention, one pass over the keys (1.11.0): O = softmax(scale Q K^T + mask) V without the scores in memory
// (FlashAttention-2's online softmax). Q and O are [heads][queries][headDim], K and V [kvHeads][cacheKeys][headDim],
// FP16 (packed in pairs). The entry file defines DMAX (64, 128 or 256: the largest head dimension), BC (keys per
// block), ENTRY, and OUT_F32 for FP32 output.
//
// A group takes BR = 16 rows of one KV head: the rows are its query heads' queries (row r is query r % queries of
// head kvHead * heads / kvHeads + r / queries), so grouped-query attention reads each K and V block once for all of
// them. Each row has 8 threads, or 16 in the wide Vulkan kernel. Per block of BC keys:
//   - K and V to group memory; each thread computes the scores of keys k, k + 8, .. of its row: a dot product in
//     FP32 (FP16 products are exact), times scale; masked keys are -infinity;
//   - each thread takes the block's max m' of its row's scores, rescales its running sum l and accumulators by
//     exp(m - m'), and its keys' P = exp(s - m') go to group memory;
//   - each thread adds P V over the block to its columns (pairs k, k + 8, ..), and P to l, in key order.
// Every thread of a row computes m and l the same way, so they agree. Key blocks that no row of the group sees
// (causal) are skipped. With splits > 1 the group covers one range of the keys and writes its (m, l, accumulators)
// for attn_combine; otherwise O = acc / l.
// On the M4 Max through MoltenVK, D128 causal prefill at 512/1024/2048 tokens (ms):
// 8 lanes, sequential dots 0.921/3.373/12.822; interleaved dots 0.796/2.937/11.346.
// 16 lanes with sequential dots 0.701/2.624/10.227; interleaved 0.646/2.412/9.385.
// Eight rows sharing K/V with 16 lanes and interleaved dots took 1.111/4.331/17.921.
// Interleaving reuses each FP32 query pair across two independent scores in the original FMA order.
#include "attn_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> q;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> k;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> v;
#if OUT_F32
[[vk::binding(3, 0)]] RWStructuredBuffer<float> o;
#else
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> o;
#endif
[[vk::binding(4, 0)]] RWStructuredBuffer<float> partialML;    // per split and row: m, l (splits > 1)
[[vk::binding(5, 0)]] RWStructuredBuffer<float> partialAcc;   // per split and row: headDim accumulators
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> table;        // paged K and V: the sequence's pages
[[vk::push_constant]] ConstantBuffer<AttnParams> p;

static const uint BR = 16;
#ifdef WIDE
static const uint TPR = 16;
#else
static const uint TPR = 8;
#endif
static const uint THREADS = BR * TPR;
static const uint D2 = DMAX / 2;          // pairs
static const uint STRIDE = D2 + 1;        // padded rows: no bank conflicts between rows
static const uint PPT = D2 / TPR;         // pairs per thread
static const uint KPT = BC / TPR;         // keys per thread

groupshared uint qs[BR * STRIDE];
groupshared uint ks[BC * STRIDE];
groupshared uint vs[BC * STRIDE];
groupshared float ss[BR * BC];
groupshared float ps[BR * BC];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = local.x, lr = t / TPR, kk = t % TPR;
    uint kvHead = group.y, split = group.z;
    uint hpk = p.heads / p.kvHeads, rowsKv = hpk * p.queries;
    uint r0 = group.x * BR, r = r0 + lr;
    bool live = r < rowsKv;
    uint head = kvHead * hpk + (live ? r / p.queries : 0), qi = live ? r % p.queries : 0;
    uint d2 = p.headDim / 2;
    uint offset = p.keys - p.queries;   // causal: query i sees keys <= i + offset

    // Q of the group's rows, zero beyond headDim
    for (uint e = t; e < BR * D2; e += THREADS)
    {
        uint row = r0 + e / D2, pr = e % D2;
        uint w = 0;
        if (row < rowsKv && pr < d2)
        {
            uint h = kvHead * hpk + row / p.queries, i = row % p.queries;
            w = q[(h * p.queries + i) * d2 + pr];
        }
        qs[(e / D2) * STRIDE + pr] = w;
    }

    // the keys this group covers: its range, and with the causal mask no further than its last row sees
    uint kStart = split * p.splitKeys, kEnd = min(p.keys, kStart + p.splitKeys);
    if (p.causal != 0)
    {
        uint rLast = min(r0 + BR, rowsKv) - 1;
        uint iMax = (r0 / p.queries != rLast / p.queries) ? p.queries - 1 : rLast % p.queries;
        kEnd = min(kEnd, iMax + offset + 1);
    }
    uint limit = p.causal != 0 ? qi + offset + 1 : p.keys;   // this row's keys end

    float m = dg_negative_infinity(), l = 0.0f;
    float accLo[PPT], accHi[PPT];
    [ForceUnroll] for (uint u = 0; u < PPT; u++) { accLo[u] = 0.0f; accHi[u] = 0.0f; }

    for (uint b = kStart; b < kEnd; b += BC)
    {
        GroupMemoryBarrierWithGroupSync();   // the previous block is no longer read
        for (uint e = t; e < BC * D2; e += THREADS)
        {
            uint j = e / D2, pr = e % D2, key = b + j;
            bool ok = key < kEnd && pr < d2;
            uint at = ok ? ATTN_KV_WORD(p, table, 0u, kvHead, key, pr, d2) : 0u;
            ks[j * STRIDE + pr] = ok ? k[at] : 0u;
            vs[j * STRIDE + pr] = ok ? v[at] : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        // the scores of keys kk, kk + TPR, ..
#ifdef WIDE
        float scores[KPT];
        [ForceUnroll] for (uint u = 0; u < KPT; u++) scores[u] = 0.0f;
        for (uint p0 = 0; p0 < D2; p0 += 8)
        {
            [ForceUnroll] for (uint pi = 0; pi < 8; pi++)
            {
                uint pr = p0 + pi;
                uint qw = qs[lr * STRIDE + pr];
                float qlo = attnLo(qw), qhi = attnHi(qw);
                [ForceUnroll] for (uint u = 0; u < KPT; u++)
                {
                    uint kw = ks[(kk + u * TPR) * STRIDE + pr];
                    scores[u] = fma(qlo, attnLo(kw), scores[u]);
                    scores[u] = fma(qhi, attnHi(kw), scores[u]);
                }
            }
        }
        [ForceUnroll] for (uint u = 0; u < KPT; u++)
        {
            uint j = kk + u * TPR, key = b + j;
            bool visible = live && key < kEnd && key < limit;
            ss[lr * BC + j] = visible ? scores[u] * p.scale : dg_negative_infinity();
        }
#else
        [ForceUnroll]
        for (uint u = 0; u < KPT; u++)
        {
            uint j = kk + u * TPR, key = b + j;
            float s = 0.0f;
            // blocks of 8 pairs: a full unroll nests one expression past Metal's 256 brackets at DMAX 256
            for (uint p0 = 0; p0 < D2; p0 += 8)
            {
                [ForceUnroll]
                for (uint pi = 0; pi < 8; pi++)
                {
                    uint pr = p0 + pi;
                    uint qw = qs[lr * STRIDE + pr], kw = ks[j * STRIDE + pr];
                    s = fma(attnLo(qw), attnLo(kw), s);   // FP16 products are exact: fused or not, the same
                    s = fma(attnHi(qw), attnHi(kw), s);
                }
            }
            bool visible = live && key < kEnd && key < limit;
            ss[lr * BC + j] = visible ? s * p.scale : dg_negative_infinity();
        }
#endif
        GroupMemoryBarrierWithGroupSync();
        // the row's new max, the rescale, and P of this thread's keys
        float mb = dg_negative_infinity();
        [ForceUnroll] for (uint j = 0; j < BC; j++) mb = max(mb, ss[lr * BC + j]);
        float mNew = max(m, mb);
        float alpha = m == dg_negative_infinity() ? 0.0f : exp(m - mNew);
        [ForceUnroll]
        for (uint u = 0; u < KPT; u++)
        {
            uint j = kk + u * TPR;
            float s = ss[lr * BC + j];
            ps[lr * BC + j] = s == dg_negative_infinity() ? 0.0f : exp(s - mNew);
        }
        m = mNew;
        GroupMemoryBarrierWithGroupSync();
        // l and the accumulators, in key order
        l = l * alpha;
        [ForceUnroll] for (uint u = 0; u < PPT; u++) { accLo[u] = accLo[u] * alpha; accHi[u] = accHi[u] * alpha; }
        [ForceUnroll]
        for (uint j = 0; j < BC; j++)
        {
            float pj = ps[lr * BC + j];
            l = l + pj;
            [ForceUnroll]
            for (uint u = 0; u < PPT; u++)
            {
                uint vw = vs[j * STRIDE + kk + u * TPR];
                accLo[u] = fma(pj, attnLo(vw), accLo[u]);
                accHi[u] = fma(pj, attnHi(vw), accHi[u]);
            }
        }
    }

    if (!live) return;
    uint row = head * p.queries + qi;
    if (p.splits > 1)
    {
        uint pr = split * p.heads * p.queries + row;
        if (kk == 0) { partialML[2 * pr] = m; partialML[2 * pr + 1] = l; }
        [ForceUnroll]
        for (uint u = 0; u < PPT; u++)
        {
            uint pair = kk + u * TPR;
            if (pair < d2) { partialAcc[pr * p.headDim + 2 * pair] = accLo[u]; partialAcc[pr * p.headDim + 2 * pair + 1] = accHi[u]; }
        }
        return;
    }
    [ForceUnroll]
    for (uint u = 0; u < PPT; u++)
    {
        uint pair = kk + u * TPR;
        if (pair >= d2) continue;
        float lo = accLo[u] / l, hi = accHi[u] / l;
#if OUT_F32
        o[row * p.headDim + 2 * pair] = lo;
        o[row * p.headDim + 2 * pair + 1] = hi;
#else
        o[row * d2 + pair] = attnPack(lo, hi);
#endif
    }
}
