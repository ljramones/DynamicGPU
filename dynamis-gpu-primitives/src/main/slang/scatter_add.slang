// Gather.scatterAdd, pass 2 (1.14.0): dx[.., k] (+)= sum of dy[.., j] over the j with indices[j] == k, added in
// increasing j, FP32, along one dimension moved last on the host. No float atomics: the result does not depend on
// scheduling (Gather.referenceScatterAdd, bit for bit).
//
// One group per k. Without accumulate, the group first writes +0 over its slice; a k no index names stops there (its
// count, from scatter_count, is 0). Otherwise the group walks the indices in chunks of 256, in order, until it has seen
// all count[k] matches: a fixed exclusive scan of the chunk's match flags lists the matching j in increasing order,
// and each thread adds those rows of dy to the slice elements it owns (e = t, t + 256, ...), in list order. A thread
// always owns the same elements, so its reads and writes of dx need no barrier.
#include "dynamis/kit.slang"

struct Params
{
    uint extent;      // groups: dx's extent along the dimension
    uint gridX;
    uint n;           // indices
    uint accumulate;  // 1: add into dx; 0: overwrite it
    uint dxLength;    // for debug builds
    uint dyLength;
    uint iOffset;
    uint iStride;
    DG_VIEW_FIELDS4(dy)
    DG_VIEW_FIELDS4(dx)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> dy;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> counts;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dx;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;
groupshared uint scan[THREADS];
groupshared uint list[THREADS];
groupshared uint chunkTotal;

[shader("compute")]
[numthreads(256, 1, 1)]
void scatter_add(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint k = dg_linear_group(group, p.gridX);
    if (k >= p.extent) return;   // uniform across the group
    uint t = local.x;
    DgView xv = DG_VIEW4(p, dx);
    DgView gv = DG_VIEW4(p, dy);
    uint slice = xv.extent.x * xv.extent.y * xv.extent.z;   // elements per k
    if (p.accumulate == 0u)
    {
        for (uint e = t; e < slice; e += THREADS)
        {
            uint xi = dg_at(xv, uint4(dg_row(xv, e), k));
            if (dg_in_range(errors, xi, p.dxLength, k)) dx[xi] = 0.0f;
        }
    }
    uint want = counts[k];
    uint found = 0;
    for (uint base = 0; base < p.n && found < want; base += THREADS)   // uniform: want and found are the group's
    {
        uint j = base + t;
        uint match = (j < p.n && indices[p.iOffset + j * p.iStride] == k) ? 1u : 0u;
        scan[t] = match;
        DG_GROUP_SCAN_EXCLUSIVE(scan, t, THREADS, dg_op_addu, 0u)
        if (match != 0u) list[scan[t]] = j;
        if (t == THREADS - 1u) chunkTotal = scan[t] + match;
        GroupMemoryBarrierWithGroupSync();
        uint m = chunkTotal;
        if (m != 0u)
        {
            for (uint e = t; e < slice; e += THREADS)
            {
                uint3 r = dg_row(xv, e);
                uint xi = dg_at(xv, uint4(r, k));
                if (!dg_in_range(errors, xi, p.dxLength, k)) continue;
                float acc = dx[xi];
                for (uint q = 0; q < m; q++)
                {
                    uint gi = dg_at(gv, uint4(r, list[q]));
                    if (dg_in_range(errors, gi, p.dyLength, k)) acc = acc + dy[gi];
                }
                dx[xi] = acc;
            }
        }
        found += m;
        GroupMemoryBarrierWithGroupSync();   // before the next chunk overwrites scan, list and chunkTotal
    }
}
