// Reduce.logSumExp (1.10.0): y[r] = max + log(sum exp(x - max)) along the last dimension of a rank-4 view, one group
// of 256 threads per row. The max and the sum are combined in the kit's fixed tree, so a device's result does not
// depend on scheduling; exp and log are the device's, so results are bounded. A row of -infinity gives -infinity;
// a NaN gives NaN.
#include "dynamis/kit.slang"

struct Params
{
    uint rows;
    uint gridX;
    uint xLength;   // buffer lengths in elements, for debug builds
    uint yLength;
    DG_VIEW_FIELDS4(x)
    DG_VIEW_FIELDS4(y)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;
groupshared float red[THREADS];
groupshared float rowMax;

[shader("compute")]
[numthreads(256, 1, 1)]
void reduce_lse(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint row = dg_linear_group(group, p.gridX);
    if (row >= p.rows) return;   // uniform across the group
    uint t = local.x;
    DgView xv = DG_VIEW4(p, x);
    uint n = xv.extent.w;
    uint3 c = dg_row(xv, row);
    float acc = dg_negative_infinity();
    for (uint j = t; j < n; j += THREADS)
    {
        uint xi = dg_at(xv, uint4(c, j));
        if (dg_in_range(errors, xi, p.xLength, row)) acc = dg_max(acc, x[xi]);
    }
    red[t] = acc;
    DG_GROUP_REDUCE(red, t, THREADS, dg_op_max)
    DG_GROUP_BROADCAST(red, t, rowMax)
    float mx = rowMax;
    GroupMemoryBarrierWithGroupSync();
    acc = 0.0f;
    if (mx != dg_negative_infinity())
    {
        for (uint j = t; j < n; j += THREADS)
        {
            uint xi = dg_at(xv, uint4(c, j));
            if (dg_in_range(errors, xi, p.xLength, row)) acc = acc + exp(x[xi] - mx);
        }
    }
    red[t] = acc;
    DG_GROUP_REDUCE(red, t, THREADS, dg_op_add)
    uint yi = dg_at(DG_VIEW4(p, y), uint4(c, 0));
    if (t == 0 && dg_in_range(errors, yi, p.yLength, row)) y[yi] = mx == dg_negative_infinity() || dg_is_nan(mx) ? mx : mx + log(red[0]);
}
