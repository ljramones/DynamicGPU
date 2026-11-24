// FP64: o[2i] = a[i] + b[i] and o[2i+1] = a[i] * b[i], correctly rounded doubles.
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<double> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(64, 1, 1)]
void fp64_arith(uint3 id : SV_DispatchThreadID) {
    uint i = id.x; if (i >= p.n) return;
    double x = a[i], y = b[i];
    o[2 * i] = x + y;
    o[2 * i + 1] = x * y;
}
