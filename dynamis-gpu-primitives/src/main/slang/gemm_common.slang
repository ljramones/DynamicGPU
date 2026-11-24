// What every GEMM kernel shares (gemm_body.slang, gemm_skinny_body.slang): the parameter block, C, and the fused
// epilogue (1.8.0). The including file defines OUT_T and declares A and B at bindings 0 and 1 before including it.
//
// The epilogue is applied to r = fl(fl(alpha acc) + fl(beta c)) in FP32, in this order, before the one rounding to
// OUT_T: r + bias (per column j or per row i, shared by the batch); the activation (ReLU with Java max semantics,
// GELU in its tanh form, SiLU); r * gate; r + residual. Gate and residual are laid out as C (ldc, strideC); bias,
// gate and residual have C's element type. Bias, ReLU, gate and residual round as written, so the fixed-order
// kernels stay bit for bit with GemmF16's references; GELU and SiLU use the device's exp and are bounded.
#include "dynamis/numeric.slang"

struct GemmParams
{
    uint m;
    uint n;
    uint k;
    uint lda;
    uint ldb;
    uint ldc;
    float alpha;
    float beta;
    uint transB;
    uint strideA;
    uint strideB;
    uint strideC;
    uint transA;    // A stored k x m (lda >= m): A^T B, the weight gradient
    uint bias;      // 0 none, 1 per column, 2 per row
    uint act;       // 0 none, 1 ReLU, 2 GELU (tanh form), 3 SiLU
    uint gated;     // r * gate
    uint residual;  // r + residual
};

[[vk::binding(2, 0)]] RWStructuredBuffer<OUT_T> c;
[[vk::binding(3, 0)]] RWStructuredBuffer<OUT_T> epBias;
[[vk::binding(4, 0)]] RWStructuredBuffer<OUT_T> epGate;
[[vk::binding(5, 0)]] RWStructuredBuffer<OUT_T> epResidual;
[[vk::push_constant]] ConstantBuffer<GemmParams> p;

static const float GELU_SCALE = 0.7978845608f;   // sqrt(2 / pi)
static const float GELU_CUBIC = 0.044715f;

// The epilogue for C[i][j] at idx, which also indexes gate and residual.
float gemmEpilogue(float r, uint i, uint j, uint idx)
{
    if (p.bias == 1) r = r + float(epBias[j]);
    else if (p.bias == 2) r = r + float(epBias[i]);
    if (p.act == 1) r = dg_max(r, 0.0f);
    else if (p.act == 2)
    {
        // tanh(u) = 1 - 2 / (exp(2u) + 1): exp's overflow gives 1, its underflow -1
        float u = GELU_SCALE * (r + GELU_CUBIC * r * r * r);
        float t = 1.0f - 2.0f / (exp(2.0f * u) + 1.0f);
        r = 0.5f * r * (1.0f + t);
    }
    else if (p.act == 3) r = r / (1.0f + exp(-r));
    if (p.gated != 0) r = r * float(epGate[idx]);
    if (p.residual != 0) r = r + float(epResidual[idx]);
    return r;
}

// C[i][j] at idx = epilogue(fl(fl(alpha acc) + fl(beta c))), rounded once to OUT_T. With beta == 0, C is not read.
void gemmStore(uint i, uint j, uint idx, float acc)
{
    float r = p.beta == 0.0f ? p.alpha * acc : p.alpha * acc + p.beta * float(c[idx]);
    c[idx] = OUT_T(gemmEpilogue(r, i, j, idx));
}
