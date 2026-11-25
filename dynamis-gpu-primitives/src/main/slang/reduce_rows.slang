// Reduce.rows: y[r] = x[r, 0] OP x[r, 1] OP ... along the last dimension of a rank-4 view, one group of 256 threads per
// row (a kit.Grid over the rows). Thread t folds columns t, t + 256, ... in order from the identity, then the group
// combines the 256 partials in DG_GROUP_REDUCE's fixed tree: the same bits on every device (Reduce.reference).
// OP (specialization constant): 0 sum, 1 max, 2 min (Java Math.max and Math.min semantics), 3 mean (1.14.0: the sum, divided
// by the row length; an empty row gives NaN).
#include "dynamis/kit.slang"

[vk::constant_id(0)] const uint OP = 0;

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

float identity()
{
    return (OP == 0 || OP == 3) ? 0.0f : OP == 1 ? dg_negative_infinity() : dg_positive_infinity();
}

float combine(float a, float b)
{
    return (OP == 0 || OP == 3) ? a + b : OP == 1 ? dg_max(a, b) : dg_min(a, b);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void reduce_rows(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint row = dg_linear_group(group, p.gridX);
    if (row >= p.rows) return;   // uniform across the group
    uint t = local.x;
    DgView xv = DG_VIEW4(p, x);
    uint n = xv.extent.w;
    uint3 c = dg_row(xv, row);
    float acc = identity();
    for (uint j = t; j < n; j += THREADS)
    {
        uint xi = dg_at(xv, uint4(c, j));
        if (dg_in_range(errors, xi, p.xLength, row)) acc = combine(acc, x[xi]);
    }
    red[t] = acc;
    DG_GROUP_REDUCE(red, t, THREADS, combine)
    uint yi = dg_at(DG_VIEW4(p, y), uint4(c, 0));
    if (t == 0 && dg_in_range(errors, yi, p.yLength, row)) y[yi] = OP == 3 ? red[0] / float(n) : red[0];
}
