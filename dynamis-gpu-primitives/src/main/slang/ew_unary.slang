// Elementwise.Unary: y[i] = op(x[i]) over rank-4 views, x broadcasting along its unit dimensions.
#include "elementwise_ops.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;
[[vk::push_constant]] ConstantBuffer<UnaryParams> p;

[shader("compute")]
[numthreads(256, 1, 1)]
void ew_unary(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint i = dg_linear_thread(group, local, p.gridX, THREADS);
    if (i >= p.count) return;
    DgView yv = DG_VIEW4(p, y);
    uint4 c = dg_coords(yv, i);
    y[dg_at(yv, c)] = unary(x[dg_at_broadcast(DG_VIEW4(p, x), c)]);
}
