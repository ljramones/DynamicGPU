// SUBGROUP_VOTE: per thread t, o[t] has bit 0 = all lanes true, bit 1 = any lane is lane 0, bit 2 = all lanes are
// lane 0. (Slang has no Metal all-equal, so that vote is not tested.)
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(128, 1, 1)]
void subgroup_vote(uint3 id : SV_DispatchThreadID) {
    uint t = id.x, lane = WaveGetLaneIndex();
    uint bits = (WaveActiveAllTrue(lane < WaveGetLaneCount()) ? 1u : 0u) | (WaveActiveAnyTrue(lane == 0) ? 2u : 0u)
            | (WaveActiveAllTrue(lane == 0) ? 4u : 0u);
    if (t >= p.n) return;
    o[t] = bits;
}
