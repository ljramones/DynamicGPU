// The GEMM parameters and the fused epilogue (1.8.0), included after every buffer is declared (gemm_common.slang).
//
// The epilogue is applied to r = fl(fl(alpha acc) + fl(beta c)) in FP32, in this order, before the one rounding to
// OUT_T: r + bias (per column j or per row i, shared by the batch); the activation (ReLU with Java max semantics,
// GELU in its tanh form, SiLU); r * gate; r + residual. Gate and residual are laid out as C (ldc, strideC); bias,
// gate and residual have C's element type. Bias, ReLU, gate and residual round as written, so the fixed-order
// kernels stay bit for bit with GemmF16's references; GELU and SiLU use the device's exp and are bounded.
[[vk::push_constant]] ConstantBuffer<GemmParams> p;

static const float GELU_SCALE = 0.7978845608f;   // sqrt(2 / pi)
static const float GELU_CUBIC = 0.044715f;

// The problem of batch matrix z, from the parameters.
GemmProblem gemmBatchProblem(uint z)
{
    GemmProblem q;
    q.m = p.m; q.n = p.n; q.k = p.k; q.lda = p.lda; q.ldb = p.ldb; q.ldc = p.ldc;
    q.baseA = z * p.strideA; q.baseB = z * p.strideB; q.baseC = z * p.strideC; q.baseBias = 0;
    return q;
}

// The epilogue for C[i][j] at idx, which also indexes gate and residual; the bias starts at biasBase.
float gemmEpilogue(float r, uint i, uint j, uint idx, uint biasBase)
{
    if (p.bias == 1) r = r + float(epBias[biasBase + j]);
    else if (p.bias == 2) r = r + float(epBias[biasBase + i]);
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

// C[i][j] of problem q = epilogue(fl(fl(alpha acc) + fl(beta c))), rounded once to OUT_T. With beta == 0, C is not
// read.
void gemmStore(GemmProblem q, uint i, uint j, float acc)
{
    uint idx = q.baseC + i * q.ldc + j;
    float r = p.beta == 0.0f ? p.alpha * acc : p.alpha * acc + p.beta * float(c[idx]);
    c[idx] = OUT_T(gemmEpilogue(r, i, j, idx, q.baseBias));
}
