// Reduce.logSumExp, the combine (1.10.1): one group of 256 threads per row over its chunks' (max m, sum s) from
// reduce_lse.slang: M = max m, then y = M + log(sum s exp(m - M)), both in the kit's fixed trees (each thread takes
// chunks t, t + 256, .. in order). A chunk of -infinity adds nothing; M of -infinity or NaN is the result.
#include "dynamis/kit.slang"

struct Params
{
    uint rows;
    uint gridX;
    uint yLength;   // for debug builds
    uint chunks;
    DG_VIEW_FIELDS4(y)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> partial;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;
groupshared float red[THREADS];
groupshared float rowMax;

[shader("compute")]
[numthreads(256, 1, 1)]
void reduce_lse_combine(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint row = dg_linear_group(group, p.gridX);
    if (row >= p.rows) return;   // uniform across the group
    uint t = local.x, base = row * p.chunks;
    float acc = dg_negative_infinity();
    for (uint g = t; g < p.chunks; g += THREADS) acc = dg_max(acc, partial[2 * (base + g)]);
    red[t] = acc;
    DG_GROUP_REDUCE(red, t, THREADS, dg_op_max)
    DG_GROUP_BROADCAST(red, t, rowMax)
    float mx = rowMax;
    GroupMemoryBarrierWithGroupSync();
    acc = 0.0f;
    if (mx != dg_negative_infinity() && !dg_is_nan(mx))
    {
        for (uint g = t; g < p.chunks; g += THREADS)
        {
            float m = partial[2 * (base + g)];
            if (m != dg_negative_infinity()) acc = acc + partial[2 * (base + g) + 1] * exp(m - mx);
        }
    }
    red[t] = acc;
    DG_GROUP_REDUCE(red, t, THREADS, dg_op_add)
    DgView yv = DG_VIEW4(p, y);
    uint yi = dg_at(yv, uint4(dg_row(yv, row), 0));
    if (t == 0 && dg_in_range(errors, yi, p.yLength, row)) y[yi] = mx == dg_negative_infinity() || dg_is_nan(mx) ? mx : mx + log(red[0]);
}
