// Elementwise.Binary: y[i] = op(a[i], b[i]) over rank-4 views, a and b broadcasting along their unit dimensions.
#include "elementwise_ops.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::push_constant]] ConstantBuffer<BinaryParams> p;

[shader("compute")]
[numthreads(256, 1, 1)]
void ew_binary(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint i = dg_linear_thread(group, local, p.gridX, THREADS);
    if (i >= p.count) return;
    DgView yv = DG_VIEW4(p, y);
    uint4 c = dg_coords(yv, i);
    uint ai = dg_at_broadcast(DG_VIEW4(p, a), c), bi = dg_at_broadcast(DG_VIEW4(p, b), c), yi = dg_at(yv, c);
    if (!dg_in_range(errors, ai, p.aLength, i) || !dg_in_range(errors, bi, p.bLength, i) || !dg_in_range(errors, yi, p.yLength, i)) return;
    y[yi] = binary(a[ai], b[bi]);
}
