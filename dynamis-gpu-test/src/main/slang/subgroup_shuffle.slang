// SUBGROUP_SHUFFLE: per thread t, o[3t..3t+2] = lane count, lane index, and the t of lane (index + 1) mod count.
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(128, 1, 1)]
void subgroup_shuffle(uint3 id : SV_DispatchThreadID) {
    uint t = id.x;
    uint count = WaveGetLaneCount(), lane = WaveGetLaneIndex();
    uint next = WaveShuffle(t, (lane + 1) % count);
    if (t >= p.n) return;
    o[3 * t] = count;
    o[3 * t + 1] = lane;
    o[3 * t + 2] = next;
}
