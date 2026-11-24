// What every GEMM kernel shares (gemm_body.slang, gemm_skinny_body.slang, gemm_small_body.slang,
// gemm_grouped_body.slang): the parameter block, one problem's layout, C and the epilogue's operands. The including
// file defines OUT_T, declares A and B at bindings 0 and 1 first (Slang numbers Metal buffers in declaration order),
// includes this file, declares any further buffers, then includes gemm_epilogue.slang, which declares the parameters.
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
#ifdef GROUPED
    uint count;     // problems
    uint aLength;   // the buffers' lengths in elements: problems outside them are skipped and reported
    uint bLength;
    uint cLength;
    uint biasLength;
#endif
};

// One multiply: its sizes, leading dimensions, and where its A, B, C (also gate and residual) and bias start.
struct GemmProblem
{
    uint m;
    uint n;
    uint k;
    uint lda;
    uint ldb;
    uint ldc;
    uint baseA;
    uint baseB;
    uint baseC;
    uint baseBias;
};

[[vk::binding(2, 0)]] RWStructuredBuffer<OUT_T> c;
[[vk::binding(3, 0)]] RWStructuredBuffer<OUT_T> epBias;
[[vk::binding(4, 0)]] RWStructuredBuffer<OUT_T> epGate;
[[vk::binding(5, 0)]] RWStructuredBuffer<OUT_T> epResidual;
