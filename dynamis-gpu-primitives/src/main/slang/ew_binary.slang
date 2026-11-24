// Elementwise.Binary: y[i] = op(a[i], b[i]) over rank-4 views, a and b broadcasting along their unit dimensions.
#include "elementwise_ops.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> y;
[[vk::push_constant]] ConstantBuffer<BinaryParams> p;

[shader("compute")]
[numthreads(256, 1, 1)]
void ew_binary(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint i = dg_linear_thread(group, local, p.gridX, THREADS);
    if (i >= p.count) return;
    DgView yv = DG_VIEW4(p, y);
    uint4 c = dg_coords(yv, i);
    y[dg_at(yv, c)] = binary(a[dg_at_broadcast(DG_VIEW4(p, a), c)], b[dg_at_broadcast(DG_VIEW4(p, b), c)]);
}
