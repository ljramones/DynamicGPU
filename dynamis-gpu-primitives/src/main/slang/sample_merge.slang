// Sampler, merge rounds (1.10.1): with top-k or top-p, each group merges perGroup sorted lists of a row (a power of
// two) and keeps the first `head` entries of the result (only those can be among the k largest). Each list gives its
// first c = min(head, stride) entries to group memory, then a pairwise merge tree: at each level, lists A and B
// (adjacent, equal lengths) become one, each entry moving to its index plus the count of entries of the other list
// that go before it (a binary search; for equal entries, A's go first, so places are unique). The order (value
// descending, then index) is total, so this is the reference's sort. Rounds repeat until a row has one list, which
// sample_final merges the last lists itself. (Sorting 2048 slots per group instead took 67 us at a vocabulary of
// 128256; ranking each candidate against every other list, 77.) The tree is in sample_tree.slang.
#include "sample_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> inV;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> inI;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> outV;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> outI;
[[vk::push_constant]] ConstantBuffer<SampleParams> p;

static const uint SLOTS = 2048;
groupshared float sv[SLOTS];
groupshared uint si[SLOTS];
#include "sample_tree.slang"

[shader("compute")]
[numthreads(1024, 1, 1)]
void sample_merge(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint g = group.x, row = group.y, t = local.x, h = p.head;
    uint c = min(h, p.stride);                      // entries each input list gives
    uint P = p.perGroup;                            // lists per group, a power of two with P * c <= 2048
    uint outLists = (p.lists + P - 1) / P;
    uint first = g * P, count = min(P, p.lists - first), slots = P * c;
    LOAD_LISTS(inV, inI, row, p.lists, first, count, p.stride, c, slots, t)
    mergeTree(t, c, slots);
    if (t < h)
    {
        uint at = (row * outLists + g) * h + t;
        outV[at] = t < slots ? sv[t] : dg_negative_infinity();
        outI[at] = t < slots ? si[t] : NONE;
    }
}
