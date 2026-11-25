// Gather.scatterAdd, pass 1 (1.14.0): counts[k] = the number of indices equal to k, for k < extent (counts zeroed by the
// host). Integer atomics: the counts are exact whatever the order. An index outside the dimension is reported
// (DG_ERROR_INDEX, index = j; the submission fails).
#include "dynamis/kit.slang"

struct Params
{
    uint n;           // indices
    uint gridX;
    uint extent;
    uint iLength;     // for debug builds
    uint iOffset;
    uint iStride;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> counts;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> errors;
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void scatter_count(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint j = dg_linear_thread(group, local, p.gridX, THREADS);
    if (j >= p.n) return;
    uint ii = p.iOffset + j * p.iStride;
    if (!dg_in_range(errors, ii, p.iLength, j)) return;
    uint k = indices[ii];
    if (k >= p.extent)
    {
        dg_report_error(errors, DG_ERROR_INDEX, j);
        return;
    }
    InterlockedAdd(counts[k], 1u);
}
