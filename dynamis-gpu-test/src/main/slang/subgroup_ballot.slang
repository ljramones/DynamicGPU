// SUBGROUP_BALLOT: per thread t, o[3t..3t+2] = lane index, the number of lanes whose index is a multiple of 3, and
// the t of the subgroup's first active lane.
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(128, 1, 1)]
void subgroup_ballot(uint3 id : SV_DispatchThreadID) {
    uint t = id.x, lane = WaveGetLaneIndex();
    uint count = WaveActiveCountBits(lane % 3 == 0);
    uint first = WaveReadLaneFirst(t);
    if (t >= p.n) return;
    o[3 * t] = lane;
    o[3 * t + 1] = count;
    o[3 * t + 2] = first;
}
