// On an M4 Max with 128256 varied logits, this reduced the full k=1024 encode from
// 121.85 to 111.86 us. Tied logits took 108.70 us (122.05 us with group-memory sorting).
// Top-k chunk sort on devices with shuffle and arithmetic. Steps within a subgroup exchange
// register pairs. Steps spanning subgroups exchange pairs in group memory. Check the mapping
// once: Vulkan does not promise local IDs are assigned in increasing lane order.
#include "sample_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> logits;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> sortedV;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> sortedI;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint4> info;
[[vk::push_constant]] ConstantBuffer<SampleParams> p;

groupshared float sv[THREADS];
groupshared uint si[THREADS];
groupshared uint irregular;

[shader("compute")]
[numthreads(1024, 1, 1)]
void sample_shuffle(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = local.x, chunk = group.y * p.chunks + group.x, j = group.x * CHUNK + t;
    float v = j < p.vocab ? sanitize(logits[group.y * p.ldl + j]) : dg_negative_infinity();
    uint idx = j < p.vocab ? j : NONE;
    uint width = WaveGetLaneCount(), lane = WaveGetLaneIndex();
    if (t == 0) irregular = 0;
    GroupMemoryBarrierWithGroupSync();
    uint first = WaveShuffle(t, 0u), active = WaveActiveSum(1u);
    if (active != width || t != first + lane || lane != t % width) InterlockedOr(irregular, 1u);
    GroupMemoryBarrierWithGroupSync();
    uint shuffleWidth = irregular == 0 ? width : 1u;
    for (uint size = 2; size <= RUN; size *= 2)
    {
        for (uint stride = size / 2; stride > 0; stride /= 2)
        {
            float other;
            uint otherIdx;
            if (stride < shuffleWidth)
            {
                other = WaveShuffle(v, lane ^ stride);
                otherIdx = WaveShuffle(idx, lane ^ stride);
            }
            else
            {
                GroupMemoryBarrierWithGroupSync();
                sv[t] = v; si[t] = idx;
                GroupMemoryBarrierWithGroupSync();
                other = sv[t ^ stride]; otherIdx = si[t ^ stride];
            }
            bool firstHalf = (t & stride) == 0;
            bool descending = ((t % RUN) & size) == 0;
            bool swap = firstHalf == descending ? before(other, otherIdx, v, idx) : before(v, idx, other, otherIdx);
            if (swap) { v = other; idx = otherIdx; }
        }
    }
    sortedV[chunk * CHUNK + t] = v;
    sortedI[chunk * CHUNK + t] = idx;
    GroupMemoryBarrierWithGroupSync();
    if (t % RUN == 0) { sv[t / RUN] = v; si[t / RUN] = idx; }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0)
    {
        [ForceUnroll]
        for (uint r = 1; r < CHUNK / RUN; r++)
            if (before(sv[r], si[r], v, idx)) { v = sv[r]; idx = si[r]; }
        info[chunk] = uint4(asuint(v), idx, 0u, 0u);
    }
}
