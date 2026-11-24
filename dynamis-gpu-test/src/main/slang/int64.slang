// INT64: o[i] = (a[i] * b[i] + (a[i] >> 7)) ^ b[i], unsigned and wrapping.
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<uint64_t> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint64_t> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint64_t> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(64, 1, 1)]
void int64_arith(uint3 id : SV_DispatchThreadID) {
    uint i = id.x; if (i >= p.n) return;
    uint64_t x = a[i], y = b[i];
    o[i] = (x * y + (x >> 7)) ^ y;
}
