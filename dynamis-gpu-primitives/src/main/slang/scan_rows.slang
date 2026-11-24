// Scan.rows: an inclusive scan along the last dimension of a rank-4 view, y[r, j] = x[r, 0] OP ... OP x[r, j], one group
// of 256 threads per row. The row goes in blocks of 256: each block is scanned in DG_GROUP_SCAN_EXCLUSIVE's fixed
// tree and combined with the carry of the blocks before it, so the result is the same bits on every device
// (Scan.reference). OP (specialization constant): 0 sum, 1 max, 2 min.
#include "dynamis/kit.slang"

[vk::constant_id(0)] const uint OP = 0;

struct Params
{
    uint rows;
    uint gridX;
    DG_VIEW_FIELDS4(x)
    DG_VIEW_FIELDS4(y)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;
groupshared float buf[THREADS];
// The carry alternates between two slots, so a slot is never rewritten while a thread may still read it.
groupshared float carries[2];

float identity()
{
    return OP == 0 ? 0.0f : OP == 1 ? dg_negative_infinity() : dg_positive_infinity();
}

float combine(float a, float b)
{
    return OP == 0 ? a + b : OP == 1 ? dg_max(a, b) : dg_min(a, b);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void scan_rows(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint row = dg_linear_group(group, p.gridX);
    if (row >= p.rows) return;
    uint t = local.x;
    DgView xv = DG_VIEW4(p, x), yv = DG_VIEW4(p, y);
    uint n = xv.extent.w;
    uint3 c = dg_row(xv, row);
    float carry = identity();
    uint slot = 0;
    for (uint base = 0; base < n; base += THREADS)
    {
        uint j = base + t;
        float v = j < n ? x[dg_at(xv, uint4(c, j))] : identity();
        buf[t] = v;
        DG_GROUP_SCAN_EXCLUSIVE(buf, t, THREADS, combine, identity())
        float out = combine(carry, combine(buf[t], v));
        if (j < n) y[dg_at(yv, uint4(c, j))] = out;
        if (t == THREADS - 1) carries[slot] = out;
        GroupMemoryBarrierWithGroupSync();
        carry = carries[slot];
        slot ^= 1;
    }
}
