// Whether fma is fused: o[0] = fma(x, x, c), with c = -(x*x) rounded on the host. For x = 1 + 2^-12 the square,
// 1 + 2^-11 + 2^-24, rounds to 1 + 2^-11, so a fused fma returns the rounding error 2^-24 and two roundings return
// 0. Both values arrive as parameters, so no compiler can simplify the expression. ComputeDevice runs this once per
// device (FeatureProbe) to report GpuCapabilities.FUSED_FMA. Built by dynamis-gpu-shaders: scripts/build-slang-kernels.sh.
struct FmaProbeParams
{
    float x;
    float c;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> o;
[[vk::push_constant]] ConstantBuffer<FmaProbeParams> p;

[shader("compute")]
[numthreads(1, 1, 1)]
void fma_probe(uint3 id : SV_DispatchThreadID)
{
    if (id.x == 0)
    {
        o[0] = fma(p.x, p.x, p.c);
    }
}
