// FP16: o[2i] = a[i] + b[i] and o[2i+1] = a[i] * b[i], in half arithmetic and half storage. Inputs and results are
// normal halves, so both are correctly rounded halves of exact float results.
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<half> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<half> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<half> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(64, 1, 1)]
void fp16_arith(uint3 id : SV_DispatchThreadID) {
    uint i = id.x; if (i >= p.n) return;
    half x = a[i], y = b[i];
    o[2 * i] = x + y;
    o[2 * i + 1] = x * y;
}
