// ATOMIC_FLOAT32_ADD: every thread t adds 1 to o[0] and 0.25 to o[1 + t % 7]. All sums are exact floats.
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<float> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(64, 1, 1)]
void atomic_float_add(uint3 id : SV_DispatchThreadID) {
    uint t = id.x; if (t >= p.n) return;
    float old;
    InterlockedAdd(o[0], 1.0f, old);
    InterlockedAdd(o[1 + t % 7], 0.25f, old);
}
