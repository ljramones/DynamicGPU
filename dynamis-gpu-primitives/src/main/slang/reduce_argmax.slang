// Reduce.argmax: the index of the first largest element of each row of a rank-4 view, as uint32; NaN counts as the
// largest (the first NaN wins), and among equal values the smallest index wins. Exact, in any order; the fold
// follows reduce_rows. One group of 256 threads per row.
#include "dynamis/kit.slang"

struct Params
{
    uint rows;
    uint gridX;
    DG_VIEW_FIELDS4(x)
    DG_VIEW_FIELDS4(y)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> y;
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;
groupshared float values[THREADS];
groupshared uint indices[THREADS];

// Whether (vb, ib) beats (va, ia).
bool beats(float vb, uint ib, float va, uint ia)
{
    bool nb = dg_is_nan(vb), na = dg_is_nan(va);
    if (nb != na) return nb;
    if (!nb && vb != va) return vb > va;
    return ib < ia;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void reduce_argmax(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint row = dg_linear_group(group, p.gridX);
    if (row >= p.rows) return;
    uint t = local.x;
    DgView xv = DG_VIEW4(p, x);
    uint n = xv.extent.w;   // at least 1 (the host checks)
    uint3 c = dg_row(xv, row);
    float best = dg_negative_infinity();
    uint at = 0xFFFFFFFFu;
    for (uint j = t; j < n; j += THREADS)
    {
        float v = x[dg_at(xv, uint4(c, j))];
        if (at == 0xFFFFFFFFu || beats(v, j, best, at)) { best = v; at = j; }
    }
    values[t] = best;
    indices[t] = at;
    [ForceUnroll]
    for (uint width = THREADS / 2; width > 0; width /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        if (t < width && indices[t + width] != 0xFFFFFFFFu
                && (indices[t] == 0xFFFFFFFFu || beats(values[t + width], indices[t + width], values[t], indices[t])))
        {
            values[t] = values[t + width];
            indices[t] = indices[t + width];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) y[dg_at(DG_VIEW4(p, y), uint4(c, 0))] = indices[0];
}
