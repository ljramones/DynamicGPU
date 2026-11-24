// The grouped GEMM (1.8.0), shared by gemm_grouped.slang (FP32 C) and gemm_grouped_h.slang (FP16 C): many multiplies
// of different shapes in one dispatch, such as the experts of a mixture-of-experts layer. Each problem (a record in
// `problems`, which the GPU may have written) has its own sizes, leading dimensions and offsets into the shared A, B,
// C (also gate and residual) and bias buffers; alpha, beta, the layouts and the epilogue are shared.
//
// Group (x, y, z) computes the 64 x 64 tile (x, y) of problem z with the tiled kernel (gemm_tile.slang), so every
// problem is bit for bit GemmF16.reference. Tiles beyond a problem's C return at once. A problem that would reach
// outside a buffer, or beyond the bounds the parameters m and n carry (the largest problem, which sizes the dispatch),
// is skipped and reported (DG_ERROR_INDEX, index = the problem), so its submission fails.
#define GROUPED 1
// A and B come first: Slang numbers Metal buffers in declaration order.
[[vk::binding(0, 0)]] RWStructuredBuffer<half> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<half> b;
#include "gemm_common.slang"

// GemmF16.Problem.BYTES: 12 uints
struct ProblemRecord
{
    uint m;
    uint n;
    uint k;
    uint lda;
    uint ldb;
    uint ldc;
    uint offsetA;
    uint offsetB;
    uint offsetC;
    uint offsetBias;
    uint reserved0;
    uint reserved1;
};

[[vk::binding(6, 0)]] RWStructuredBuffer<ProblemRecord> problems;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> errors;
#include "gemm_epilogue.slang"
#include "gemm_tile.slang"
#include "dynamis/errors.slang"

// Whether rows x rowLength elements with leading dimension ld from off stay below length (nothing is read when
// either count is 0). Division keeps it free of 32-bit overflow.
bool fits(uint off, uint rows, uint rowLength, uint ld, uint length)
{
    if (rows == 0 || rowLength == 0) return true;
    if (ld < rowLength || off > length || rowLength > length - off) return false;
    return rows - 1 <= (length - off - rowLength) / ld;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint z = group.z;
    if (z >= p.count) return;
    ProblemRecord r = problems[z];
    if (group.x * BN >= r.n || group.y * BM >= r.m) return;   // uniform across the group
    bool transA = p.transA != 0, transB = p.transB != 0;
    bool ok = fits(r.offsetA, transA ? r.k : r.m, transA ? r.m : r.k, r.lda, p.aLength)
            && fits(r.offsetB, transB ? r.n : r.k, transB ? r.k : r.n, r.ldb, p.bLength)
            && fits(r.offsetC, r.m, r.n, r.ldc, p.cLength)
            && r.m <= p.m && r.n <= p.n
            && (p.bias == 0 || fits(r.offsetBias, 1, p.bias == 1 ? r.n : r.m, p.bias == 1 ? r.n : r.m, p.biasLength));
    if (!ok)
    {
        if (group.x == 0 && group.y == 0 && local.x == 0 && local.y == 0) dg_report_error(errors, DG_ERROR_INDEX, z);
        return;
    }
    GemmProblem pr;
    pr.m = r.m; pr.n = r.n; pr.k = r.k; pr.lda = r.lda; pr.ldb = r.ldb; pr.ldc = r.ldc;
    pr.baseA = r.offsetA; pr.baseB = r.offsetB; pr.baseC = r.offsetC; pr.baseBias = r.offsetBias;
    gemmTile(pr, group.xy, local);
}
