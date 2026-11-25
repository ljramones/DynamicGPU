// Norms (1.10.0): RMSNorm and LayerNorm of each row, optionally after a fused residual add, shared by
// norm_<in>[_h].slang, which define IN_T (the activations' type), OUT_T and ENTRY.
//
// For each row r (n elements; leading dimensions in elements):
//     v = x                       or, with residual,   h = IN_T(x + res), written to h, and v = h
//     RMSNorm:   y = v * rsqrt(sum v^2 / n + eps) * w
//     LayerNorm: y = (v - mean) * rsqrt(sum (v - mean)^2 / n + eps) * w [+ b],   mean = sum v / n
// in FP32, rounded once to OUT_T. w and b are FP32 vectors of n. h may be x itself (the residual stream updated in
// place): x is then bound once, as h (inPlace), and each thread reads back only the elements it wrote.
//
// One group of 256 threads per row; the sums are combined in a fixed tree (the kit's DG_GROUP_REDUCE), so a device's
// result does not depend on scheduling. rsqrt and the division by n are the device's, so results are bounded, not
// bit-identical across devices (Norms.contract).
#include "dynamis/kit.slang"
#include "quant_common.slang"

// x + res rounded to IN_T, as a float. For FP16 activations the rounding is done in integer arithmetic: lavapipe
// folded float(half(s)) back to s, so the row's sums used the unrounded value while h held the rounded one.
float roundIn(float s)
{
#ifdef IN_HALF
    return qF16ToF32(qF32ToF16(s));
#else
    return s;
#endif
}

struct NormParams
{
    uint rows;
    uint n;
    uint ldx;
    uint ldr;
    uint ldh;
    uint ldy;
    float eps;
    uint layer;      // 0 RMSNorm, 1 LayerNorm
    uint residual;   // h = x + res first
    uint bias;       // LayerNorm: + b
    uint gridX;
    uint inPlace;    // h is x (one binding: x is read through h)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<IN_T> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<IN_T> res;
[[vk::binding(2, 0)]] RWStructuredBuffer<IN_T> h;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> w;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> b;
[[vk::binding(5, 0)]] RWStructuredBuffer<OUT_T> y;
[[vk::push_constant]] ConstantBuffer<NormParams> p;

static const uint THREADS = 256;
groupshared float red[THREADS];
groupshared float rowStat;   // its own variable: see the kit's groupshared rules

// The value the norm takes for element j of row r (after the residual add, when there is one)
float value(uint r, uint j)
{
    return p.residual != 0 ? float(h[r * p.ldh + j]) : float(x[r * p.ldx + j]);
}

// sum over the threads' partials, in the fixed tree, broadcast to every thread
float groupSum(uint t, float partial)
{
    red[t] = partial;
    DG_GROUP_REDUCE(red, t, THREADS, dg_op_add);
    DG_GROUP_BROADCAST(red, t, rowStat);
    float s = rowStat;
    GroupMemoryBarrierWithGroupSync();   // before rowStat or red are written again
    return s;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint r = dg_linear_group(group, p.gridX);
    if (r >= p.rows) return;   // uniform across the group
    uint t = local.x, n = p.n;

    // pass 1: the residual add, and the row's sum (LayerNorm) or sum of squares (RMSNorm)
    float partial = 0.0f;
    for (uint j = t; j < n; j += THREADS)
    {
        float v;
        if (p.residual != 0)
        {
            float xv = p.inPlace != 0 ? float(h[r * p.ldh + j]) : float(x[r * p.ldx + j]);
            v = roundIn(xv + float(res[r * p.ldr + j]));
            h[r * p.ldh + j] = IN_T(v);   // exact: v is already an IN_T value
        }
        else v = float(x[r * p.ldx + j]);
        partial = p.layer != 0 ? partial + v : partial + v * v;
    }
    float first = groupSum(t, partial);

    float mean = 0.0f, var;
    if (p.layer != 0)
    {
        mean = first / float(n);
        // pass 2: the variance about the mean
        partial = 0.0f;
        for (uint j = t; j < n; j += THREADS)
        {
            float d = value(r, j) - mean;
            partial = partial + d * d;
        }
        var = groupSum(t, partial) / float(n);
    }
    else var = first / float(n);
    float inv = rsqrt(var + p.eps);

    // pass 3: the output
    for (uint j = t; j < n; j += THREADS)
    {
        float v = (value(r, j) - mean) * inv * w[j];
        if (p.bias != 0) v = v + b[j];
        y[r * p.ldy + j] = OUT_T(v);
    }
}
