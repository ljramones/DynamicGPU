// Fused attention, the combine of key ranges (1.11.0): one group of 128 threads per row. M = max of the ranges' m
// and L = sum l exp(m - M) in the kit's fixed trees (each thread takes ranges t, t + 128, .. in order); then each
// thread takes column pairs t, t + 128, .. and adds acc exp(m - M) over the ranges in order (eight ranges' loads at a
// time), and O = that / L. The entry file defines ENTRY and, for FP32 output, OUT_F32.
#include "attn_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> partialML;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> partialAcc;
#if OUT_F32
[[vk::binding(2, 0)]] RWStructuredBuffer<float> o;
#else
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> o;
#endif
[[vk::push_constant]] ConstantBuffer<AttnParams> p;

static const uint THREADS = 128;
static const uint MAX_SPLITS = 1024;
groupshared float red[THREADS];
groupshared float ws[MAX_SPLITS];
groupshared float rowM;
groupshared float rowL;

[shader("compute")]
[numthreads(128, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = local.x, rows = p.heads * p.queries, d2 = p.headDim / 2;
    uint row = dg_linear_group(group, p.rowBlocks);   // rowBlocks: the grid width here
    if (row >= rows) return;   // uniform across the group
    float acc = dg_negative_infinity();
    for (uint s = t; s < p.splits; s += THREADS) acc = max(acc, partialML[2 * (s * rows + row)]);
    red[t] = acc;
    DG_GROUP_REDUCE(red, t, THREADS, dg_op_max)
    DG_GROUP_BROADCAST(red, t, rowM)
    float mx = rowM;
    GroupMemoryBarrierWithGroupSync();
    acc = 0.0f;
    for (uint s = t; s < p.splits; s += THREADS)
    {
        float ms = partialML[2 * (s * rows + row)];
        float w = ms == dg_negative_infinity() ? 0.0f : exp(ms - mx);
        ws[s] = w;
        acc = fma(partialML[2 * (s * rows + row) + 1], w, acc);
    }
    red[t] = acc;
    DG_GROUP_REDUCE(red, t, THREADS, dg_op_add)
    DG_GROUP_BROADCAST(red, t, rowL)
    float l = rowL;
    for (uint pair = t; pair < d2; pair += THREADS)
    {
        float lo = 0.0f, hi = 0.0f;
        for (uint s0 = 0; s0 < p.splits; s0 += 8)
        {
            float2 a[8];
            [ForceUnroll]
            for (uint u = 0; u < 8; u++)
            {
                uint s = s0 + u, at = ((s * rows + row) * p.headDim) + 2 * pair;
                a[u] = s < p.splits ? float2(partialAcc[at], partialAcc[at + 1]) : float2(0.0f, 0.0f);
            }
            [ForceUnroll]
            for (uint u = 0; u < 8; u++)
            {
                if (s0 + u < p.splits)
                {
                    float w = ws[s0 + u];
                    lo = fma(a[u].x, w, lo);
                    hi = fma(a[u].y, w, hi);
                }
            }
        }
        lo = lo / l; hi = hi / l;
#if OUT_F32
        o[row * p.headDim + 2 * pair] = lo;
        o[row * p.headDim + 2 * pair + 1] = hi;
#else
        o[row * d2 + pair] = attnPack(lo, hi);
#endif
    }
}
