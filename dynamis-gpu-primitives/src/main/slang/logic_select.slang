// Logic select: y[i] = m[i] != 0 ? a[i] : b[i], 32-bit words of any type (logic_ops.slang).
#include "logic_ops.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> m;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> a;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> b;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> y;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::push_constant]] ConstantBuffer<LogicSelectParams> p;

[shader("compute")]
[numthreads(256, 1, 1)]
void logic_select(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint i = dg_linear_thread(group, local, p.gridX, THREADS);
    if (i >= p.count) return;
    DgView yv = DG_VIEW4(p, y);
    uint4 c = dg_coords(yv, i);
    uint mi = LG_BROADCAST_AT(p, m, c), ai = LG_BROADCAST_AT(p, a, c), bi = LG_BROADCAST_AT(p, b, c);
    uint yi = dg_at(yv, c);
    if (!dg_in_range(errors, mi, p.mLength, i) || !dg_in_range(errors, ai, p.aLength, i) || !dg_in_range(errors, bi, p.bLength, i)
        || !dg_in_range(errors, yi, p.yLength, i)) return;
    y[yi] = m[mi] != 0u ? a[ai] : b[bi];
}
