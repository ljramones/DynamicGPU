// Reduce.logSumExp (1.10.0): y[r] = max + log(sum exp(x - max)) along the last dimension of a rank-4 view. Since
// 1.10.1 each row is split into chunks of chunkLen, one group of 256 threads each: a chunk's max m and its sum
// s = sum exp(x - m), combined in the kit's fixed trees, go to `partial`, and reduce_lse_combine.slang gives
// M + log(sum s exp(m - M)) (one chunk writes y here). One group per row took 124 us for a row of 128256 on an M4 Max.
// The order is fixed, so a device's result does not depend on scheduling; exp and log are the device's, so results are
// bounded. A row of -infinity gives -infinity; a NaN gives NaN.
#include "dynamis/kit.slang"

struct Params
{
    uint rows;
    uint gridX;
    uint xLength;   // buffer lengths in elements, for debug builds
    uint yLength;
    uint chunks;    // per row
    uint chunkLen;
    DG_VIEW_FIELDS4(x)
    DG_VIEW_FIELDS4(y)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::binding(3, 0)]] RWStructuredBuffer<float> partial; // per row and chunk: max, sum (chunks > 1)
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;
groupshared float red[THREADS];
groupshared float rowMax;

[shader("compute")]
[numthreads(256, 1, 1)]
void reduce_lse(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint id = dg_linear_group(group, p.gridX);
    uint row = id / p.chunks, g = id % p.chunks;
    if (row >= p.rows) return;   // uniform across the group
    uint t = local.x;
    DgView xv = DG_VIEW4(p, x);
    uint start = g * p.chunkLen, n = min(xv.extent.w, start + p.chunkLen);
    uint3 c = dg_row(xv, row);
    float acc = dg_negative_infinity();
    for (uint j = start + t; j < n; j += THREADS)
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
        for (uint j = start + t; j < n; j += THREADS)
        {
            uint xi = dg_at(xv, uint4(c, j));
            if (dg_in_range(errors, xi, p.xLength, row)) acc = acc + exp(x[xi] - mx);
        }
    }
    red[t] = acc;
    DG_GROUP_REDUCE(red, t, THREADS, dg_op_add)
    if (t != 0) return;
    if (p.chunks > 1)
    {
        partial[2 * id] = mx;
        partial[2 * id + 1] = red[0];
        return;
    }
    uint yi = dg_at(DG_VIEW4(p, y), uint4(c, 0));
    if (dg_in_range(errors, yi, p.yLength, row)) y[yi] = mx == dg_negative_infinity() || dg_is_nan(mx) ? mx : mx + log(red[0]);
}
