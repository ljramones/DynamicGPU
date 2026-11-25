// Optimizer (1.12.0): one AdamW or SGD step per element of FP32 parameters, with FP32 or FP16 gradients (G_HALF),
// an optional gradient scale read from the GPU (clipping or loss-scale unscaling without a host sync) and an optional
// FP16 copy of the updated parameters (the forward pass's weights). The arithmetic is the CPU reference's
// (Optimizer.adamWReference, sgdReference) operation for operation: every product and sum rounded once (the build
// keeps them unfused), and division and sqrt correctly rounded on the devices tested.
//   AdamW: g = grad scale;  m = b1 m + a1 g;  v = b2 v + (a2 g) g;  u = (m / bc1) / (sqrt(v / bc2) + eps);
//          p = p decay + negLr u
//   SGD:   g = grad scale (+ wd p);  b = mu b + g;  p = p + negLr (nesterov ? g + mu b : b)
#include "dynamis/kit.slang"
#include "quant_common.slang"

struct OptimParams
{
    uint n;
    uint gridX;
    uint kind;        // 0 AdamW, 1 SGD
    uint scaled;      // multiply the gradients by scale[0]
    uint shadow;      // write an FP16 copy of the parameters
    uint nesterov;
    float negLr;
    float beta1;      // SGD: momentum
    float beta2;
    float eps;
    float bc1;
    float bc2;
    float a1;
    float a2;
    float decay;      // AdamW: 1 - lr wd; SGD: wd
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> param;
#ifdef G_HALF
[[vk::binding(1, 0)]] RWStructuredBuffer<half> grad;
#else
[[vk::binding(1, 0)]] RWStructuredBuffer<float> grad;
#endif
[[vk::binding(2, 0)]] RWStructuredBuffer<float> m;         // AdamW: m; SGD: the momentum buffer
[[vk::binding(3, 0)]] RWStructuredBuffer<float> v;         // AdamW
[[vk::binding(4, 0)]] RWStructuredBuffer<float> scale;     // [1], when scaled
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> shadowOut;  // FP16 pairs, when shadow
[[vk::push_constant]] ConstantBuffer<OptimParams> p;

float step(uint i)
{
    float g = float(grad[i]);
    if (p.scaled != 0) g = g * scale[0];
    float x = param[i];
    if (p.kind == 0)
    {
        float mi = p.beta1 * m[i] + p.a1 * g;
        float vi = p.beta2 * v[i] + (p.a2 * g) * g;
        m[i] = mi;
        v[i] = vi;
        float u = (mi / p.bc1) / (sqrt(vi / p.bc2) + p.eps);
        x = x * p.decay + p.negLr * u;
    }
    else
    {
        if (p.decay != 0.0f) g = g + p.decay * x;
        float b = p.beta1 * m[i] + g;
        m[i] = b;
        x = x + p.negLr * (p.nesterov != 0 ? g + p.beta1 * b : b);
    }
    param[i] = x;
    return x;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    // a thread takes a pair of elements, so the FP16 copy is written a word at a time
    uint w = dg_linear_thread(group, local, p.gridX, 256), i = 2 * w;
    if (i >= p.n) return;
    float lo = step(i), hi = i + 1 < p.n ? step(i + 1) : 0.0f;
    if (p.shadow != 0)
    {
        uint word = qF32ToF16(lo) | (qF32ToF16(hi) << 16);
        if (i + 1 < p.n) shadowOut[w] = word;
        else shadowOut[w] = (shadowOut[w] & 0xFFFF0000u) | (word & 0xFFFFu);
    }
}
