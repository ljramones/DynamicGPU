// M4 Max, k=40 of 128256, whole encode: 75.13 -> 41.46 us for varied logits,
// 76.78 -> 56.00 for concentrated logits. A chunk-local cutoff took 89.11 us on
// skewed logits and the minimum of k bucket winners took 94.28 us on suppressed chunks.
// Ranking in one lane per bucket took 55.41 us on varied logits; see docs/primitives.md.
// Small top-k on long rows. The k-th best of disjoint bucket winners and the strongest
// complete 64-token window minimum both bound the global k-th score from below. Rank
// winners by (logit descending, index ascending), compact every surviving token, and
// sort each chunk's survivors. The sort accepts all 1024 logits in the dense case.
#include "sample_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> logits;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> sortedV;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> sortedI;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint4> info;
[[vk::push_constant]] ConstantBuffer<SampleParams> p;

groupshared float sv[THREADS];
groupshared uint si[THREADS];
groupshared uint ranks[THREADS];
groupshared uint survivors;
groupshared float cutoffV;
groupshared uint cutoffI;
groupshared uint windowBound;

[shader("compute")]
[numthreads(1024, 1, 1)]
void sample_filter(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = local.x, k = p.head;
    uint j = group.x * CHUNK + t;
    uint chunk = group.y * p.chunks + group.x;
    float v = j < p.vocab ? sanitize(logits[group.y * p.ldl + j]) : dg_negative_infinity();
    uint idx = j < p.vocab ? j : NONE;
    // The first 128 tokens and the maxima of all later chunks occupy disjoint buckets.
    // Rank these winners to get a lower bound even if entire buckets contain only outliers.
    const uint buckets = 128;
    if (t == 0) windowBound = key(asfloat(info[group.y * p.chunks].z));
    GroupMemoryBarrierWithGroupSync();
    if (t < buckets)
    {
        float minBound = dg_negative_infinity();
        float best = t < p.vocab ? sanitize(logits[group.y * p.ldl + t]) : dg_negative_infinity();
        uint bestI = t < p.vocab ? t : NONE;
        for (uint g = 1 + t; g < p.chunks; g += buckets)
        {
            uint4 m = info[group.y * p.chunks + g];
            minBound = max(minBound, asfloat(m.z));
            if (before(asfloat(m.x), m.y, best, bestI)) { best = asfloat(m.x); bestI = m.y; }
        }
        InterlockedMax(windowBound, key(minBound));
        sv[t] = best; si[t] = bestI;
    }
    GroupMemoryBarrierWithGroupSync();
    // Eight lanes rank one bucket against the other winners. All 1024 lanes
    // contribute comparisons before the lanes for each bucket sum their counts.
    uint bucket = t / 8, slice = t % 8;
    float myV = sv[bucket]; uint myI = si[bucket], rank = 0;
    [ForceUnroll]
    for (uint i = 0; i < buckets / 8; i++)
    {
        uint other = 8 * i + slice;
        rank += before(sv[other], si[other], myV, myI) ? 1u : 0u;
    }
    ranks[t] = rank;
    [ForceUnroll]
    for (uint stride = 4; stride > 0; stride /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        if (slice < stride) ranks[t] += ranks[t + stride];
    }
    if (slice == 0 && ranks[t] == k - 1 && myI != NONE) { cutoffV = myV; cutoffI = myI; }
    if (t == 0) survivors = 0;
    GroupMemoryBarrierWithGroupSync();
    if (idx != NONE && key(v) >= windowBound && !before(cutoffV, cutoffI, v, idx))
    {
        uint slot;
        InterlockedAdd(survivors, 1u, slot);
        sv[slot] = v;
        si[slot] = idx;
    }
    GroupMemoryBarrierWithGroupSync();
    uint n = survivors, slots = 1;
    while (slots < n) slots *= 2;
    if (t >= n && t < slots) { sv[t] = dg_negative_infinity(); si[t] = NONE; }
    GroupMemoryBarrierWithGroupSync();
    for (uint size = 2; size <= slots; size *= 2)
    {
        for (uint stride = size / 2; stride > 0; stride /= 2)
        {
            uint partner = t ^ stride;
            if (t < slots && partner > t)
            {
                float a = sv[t], b = sv[partner];
                uint ai = si[t], bi = si[partner];
                bool swap = (t & size) == 0 ? before(b, bi, a, ai) : before(a, ai, b, bi);
                if (swap) { sv[t] = b; si[t] = bi; sv[partner] = a; si[partner] = ai; }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    if (t < k)
    {
        sortedV[chunk * k + t] = t < n ? sv[t] : dg_negative_infinity();
        sortedI[chunk * k + t] = t < n ? si[t] : NONE;
    }
}
