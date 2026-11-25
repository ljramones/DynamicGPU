// Logic binary: y[i] = op(a[i], b[i]) over 32-bit words (logic_ops.slang).
#include "logic_ops.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> y;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::push_constant]] ConstantBuffer<LogicBinaryParams> p;

[shader("compute")]
[numthreads(256, 1, 1)]
void logic_binary(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint i = dg_linear_thread(group, local, p.gridX, THREADS);
    if (i >= p.count) return;
    DgView yv = DG_VIEW4(p, y);
    uint4 c = dg_coords(yv, i);
    uint ai = dg_at_broadcast(DG_VIEW4(p, a), c), bi = dg_at_broadcast(DG_VIEW4(p, b), c), yi = dg_at(yv, c);
    if (!dg_in_range(errors, ai, p.aLength, i) || !dg_in_range(errors, bi, p.bLength, i) || !dg_in_range(errors, yi, p.yLength, i)) return;
    y[yi] = lgBinary(a[ai], b[bi]);
}
