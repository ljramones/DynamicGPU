// With serial bucket ranks, this window bound reduced ascending top-k on M4 Max from
// 80.45 to 57.72 us for k=40 of 128256. The complete optimized encode takes 41.22 us.
// Collect disjoint chunk winners and window minima for sample_filter in one reduction.
// Incomplete windows include a negative-infinity padding token.
#include "sample_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> logits;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> info;      // rows x chunks: (max, index, strongest window minimum, 0)
[[vk::push_constant]] ConstantBuffer<SampleParams> p;

groupshared float sv[THREADS];
groupshared uint si[THREADS];
// Each complete window of 64 witnesses at least k <= 64 logits at its minimum.
groupshared float windowMin[THREADS];

[shader("compute")]
[numthreads(1024, 1, 1)]
void sample_bounds(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint c = group.x, row = group.y, t = local.x;
    uint j = c * CHUNK + t;
    bool valid = j < p.vocab;
    float v = valid ? sanitize(logits[row * p.ldl + j]) : dg_negative_infinity();
    uint idx = valid ? j : NONE;

    // the chunk's max, ties to the lowest index
    sv[t] = v; si[t] = idx; windowMin[t] = v;
    [ForceUnroll]
    for (uint w = THREADS / 2; w > 0; w /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        if (w >= 16)
        {
            uint stride = w / 16;
            if (t % 64 < stride) windowMin[t] = min(windowMin[t], windowMin[t + stride]);
        }
        else if (t < w)
            windowMin[t * 64] = max(windowMin[t * 64], windowMin[(t + w) * 64]);
        if (t < w && before(sv[t + w], si[t + w], sv[t], si[t])) { sv[t] = sv[t + w]; si[t] = si[t + w]; }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) info[row * p.chunks + c] = uint4(asuint(sv[0]), si[0], asuint(windowMin[0]), 0u);
}
