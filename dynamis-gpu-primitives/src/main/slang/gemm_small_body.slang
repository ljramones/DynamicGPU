// The small GEMM (1.8.0), shared by gemm_small.slang (FP32 C) and gemm_small_h.slang (FP16 C): many small multiplies
// (m and n up to 32, such as the layers of a game-AI network), where the tiled kernel's 64 x 64 tiles would waste
// most of their threads. Same parameter block, layouts and epilogue as gemm_body.slang.
//
// A group of 16 x 16 threads computes a 16 x 16 tile of C, one element per thread; k is walked in steps of 16 through
// shared tiles of A and B, converted to FP32 on load, zero beyond the matrices. Each element sums its k products in
// order, fl(acc + a b) with exact FP16 products, so it is bit for bit the tiled kernel and GemmF16.reference.
#include "gemm_inputs.slang"
#include "gemm_common.slang"
#include "gemm_epilogue.slang"

static const uint TS = 16;

groupshared float smallA[TS][TS];   // [k][row]
groupshared float smallB[TS][TS];   // [k][column]

[shader("compute")]
[numthreads(16, 16, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    GemmProblem pr = gemmBatchProblem(group.z);
    uint tx = local.x, ty = local.y;
    uint row0 = group.y * TS, col0 = group.x * TS;
    float acc = 0.0f;
    for (uint k0 = 0; k0 < pr.k; k0 += TS)
    {
        uint gr = row0 + ty, gk = k0 + tx;
        uint ai = p.transA != 0 ? gk * pr.lda + gr : gr * pr.lda + gk;
        smallA[tx][ty] = gr < pr.m && gk < pr.k ? loadA(pr.baseA + ai) : 0.0f;
        uint hk = k0 + ty, gc = col0 + tx;
        uint bi = p.transB != 0 ? gc * pr.ldb + hk : hk * pr.ldb + gc;
        smallB[ty][tx] = hk < pr.k && gc < pr.n ? loadB(pr.baseB + bi) : 0.0f;
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint kk = 0; kk < TS; kk++) acc = fma(smallA[kk][ty], smallB[kk][tx], acc);
        GroupMemoryBarrierWithGroupSync();
    }
    uint row = row0 + ty, col = col0 + tx;
    if (row < pr.m && col < pr.n) gemmStore(pr, row, col, acc);
}
