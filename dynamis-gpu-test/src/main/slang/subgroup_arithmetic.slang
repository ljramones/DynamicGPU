// SUBGROUP_ARITHMETIC: per thread t, o[5t..5t+4] = lane count, lane index, subgroup sum of (t + 1), subgroup min of
// t (which identifies the subgroup), and the exclusive prefix sum of 1 (the lane's rank among active lanes).
// Groups of 128 threads hold whole subgroups of every size Vulkan allows.
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(128, 1, 1)]
void subgroup_arithmetic(uint3 id : SV_DispatchThreadID) {
    uint t = id.x;
    uint sum = WaveActiveSum(t + 1);
    uint first = WaveActiveMin(t);
    uint rank = WavePrefixSum(1u);
    if (t >= p.n) return;
    o[5 * t] = WaveGetLaneCount();
    o[5 * t + 1] = WaveGetLaneIndex();
    o[5 * t + 2] = sum;
    o[5 * t + 3] = first;
    o[5 * t + 4] = rank;
}
