// Sampler, stage 1 (1.10.0): one group of 1024 threads per chunk of 1024 logits (group (c, row)). It writes the
// chunk's max, its first index and the sum of exp((l - max) / T) over the chunk (a fixed tree), and with top-k or top-p
// the chunk as sorted runs of RUN (value descending, then index), for the merge rounds and stage 2 to combine.
#include "sample_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> logits;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> sortedV;   // rows x chunks x 1024
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> sortedI;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint4> info;      // rows x chunks: (max, index, sum, 0)
[[vk::push_constant]] ConstantBuffer<SampleParams> p;

groupshared float sv[THREADS];
groupshared uint si[THREADS];
groupshared float gMax;
groupshared uint gIdx;

[shader("compute")]
[numthreads(1024, 1, 1)]
void sample_chunks(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint c = group.x, row = group.y, t = local.x;
    uint j = c * CHUNK + t;
    bool valid = j < p.vocab;
    float v = valid ? sanitize(logits[row * p.ldl + j]) : dg_negative_infinity();
    uint idx = valid ? j : NONE;

    // the chunk's max, ties to the lowest index
    sv[t] = v; si[t] = idx;
    [ForceUnroll]
    for (uint w = THREADS / 2; w > 0; w /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        if (t < w && before(sv[t + w], si[t + w], sv[t], si[t])) { sv[t] = sv[t + w]; si[t] = si[t + w]; }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) { gMax = sv[0]; gIdx = si[0]; }
    GroupMemoryBarrierWithGroupSync();
    float mx = gMax;
    uint arg = gIdx;
    GroupMemoryBarrierWithGroupSync();

    // the sum of the chunk's weights relative to its own max
    sv[t] = valid && mx != dg_negative_infinity() && p.invT > 0.0f ? exp((v - mx) * p.invT) : 0.0f;
    DG_GROUP_REDUCE(sv, t, THREADS, dg_op_add);
    if (t == 0) info[row * p.chunks + c] = uint4(asuint(mx), arg, asuint(sv[0]), 0u);
    if (p.restricted == 0) return;   // uniform
    GroupMemoryBarrierWithGroupSync();

    // the chunk in sorted runs of RUN: the first stages of a bitonic sort (runs alternate direction), each run then
    // written descending: value descending, then index. Merge rounds combine the runs (sample_merge.slang); sorting
    // the whole chunk here took 68 us for a vocabulary of 128256.
    sv[t] = v; si[t] = idx;
    GroupMemoryBarrierWithGroupSync();
    for (uint size = 2; size <= RUN; size *= 2)
    {
        for (uint stride = size / 2; stride > 0; stride /= 2)
        {
            uint partner = t ^ stride;
            if (partner > t)
            {
                bool descending = (t & size) == 0;
                float va = sv[t], vb = sv[partner]; uint ia = si[t], ib = si[partner];
                bool swap = descending ? before(vb, ib, va, ia) : before(va, ia, vb, ib);
                if (swap) { sv[t] = vb; sv[partner] = va; si[t] = ib; si[partner] = ia; }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    uint run = t / RUN, q = t % RUN;
    uint from = (run & 1) == 0 ? t : run * RUN + (RUN - 1 - q);   // odd runs are ascending
    uint at = (row * p.chunks + c) * CHUNK + t;
    sortedV[at] = sv[from];
    sortedI[at] = si[from];
}
