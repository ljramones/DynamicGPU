// INT8: o[i] = a[i] * b[i] + (a[i] ^ b[i]), wrapping, in 8-bit storage.
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<int8_t> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<int8_t> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<int8_t> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(64, 1, 1)]
void int8_arith(uint3 id : SV_DispatchThreadID) {
    uint i = id.x; if (i >= p.n) return;
    int8_t x = a[i], y = b[i];
    o[i] = int8_t(x * y + (x ^ y));
}
