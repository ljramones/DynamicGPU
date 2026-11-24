// INT16: o[i] = a[i] * b[i] + (a[i] ^ b[i]), wrapping, in 16-bit storage.
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<int16_t> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<int16_t> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<int16_t> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(64, 1, 1)]
void int16_arith(uint3 id : SV_DispatchThreadID) {
    uint i = id.x; if (i >= p.n) return;
    int16_t x = a[i], y = b[i];
    o[i] = int16_t(x * y + (x ^ y));
}
