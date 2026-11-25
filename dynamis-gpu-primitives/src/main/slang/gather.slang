// Gather.gather (1.14.0): y[.., j] = x[.., indices[j]] along one dimension, moved last in both views on the host, so y
// has x's shape with that dimension's extent replaced by the number of indices. 32-bit elements of any type, copied as
// words. One thread per y element. An index outside x's dimension is reported (DG_ERROR_INDEX, index = j; the
// submission fails) and its elements are left untouched.
#include "dynamis/kit.slang"

struct Params
{
    uint count;
    uint gridX;
    uint extent;      // x's extent along the gathered dimension
    uint xLength;     // buffer lengths in elements, for debug builds
    uint yLength;
    uint iLength;
    uint iOffset;     // the indices: a rank-1 view
    uint iStride;
    DG_VIEW_FIELDS4(x)
    DG_VIEW_FIELDS4(y)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> y;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> errors;
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void gather(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint id = dg_linear_thread(group, local, p.gridX, THREADS);
    if (id >= p.count) return;
    DgView yv = DG_VIEW4(p, y);
    uint4 c = dg_coords(yv, id);
    uint ii = p.iOffset + c.w * p.iStride;
    if (!dg_in_range(errors, ii, p.iLength, id)) return;
    uint k = indices[ii];
    if (k >= p.extent)
    {
        dg_report_error(errors, DG_ERROR_INDEX, c.w);
        return;
    }
    uint xi = dg_at(DG_VIEW4(p, x), uint4(c.xyz, k)), yi = dg_at(yv, c);
    if (!dg_in_range(errors, xi, p.xLength, id) || !dg_in_range(errors, yi, p.yLength, id)) return;
    y[yi] = x[xi];
}
