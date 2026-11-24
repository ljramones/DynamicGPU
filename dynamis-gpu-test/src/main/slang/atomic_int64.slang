// ATOMIC_INT64: every thread t adds t + 2^40 to o[0] and takes the max of t << 33 into o[1].
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<uint64_t> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(64, 1, 1)]
void atomic_int64(uint3 id : SV_DispatchThreadID) {
    uint t = id.x; if (t >= p.n) return;
    InterlockedAdd(o[0], uint64_t(t) + (uint64_t(1) << 40));
    InterlockedMax(o[1], uint64_t(t) << 33);
}
