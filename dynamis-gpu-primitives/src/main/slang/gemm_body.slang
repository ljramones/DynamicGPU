// The tiled GEMM, shared by gemm_f16.slang (FP32 C) and gemm_f16_h.slang (FP16 C), which define OUT_T and ENTRY and
// include this file. C = alpha * A * B + beta * C, A (m x k, or k x m with transA) and B (k x n, or n x k with
// transB) in FP16, then the fused epilogue (gemm_common.slang), row-major
// with leading dimensions lda, ldb and ldc (in elements). Group z is the batch index: matrix z of A, B and C starts
// at z * strideA, z * strideB and z * strideC (a stride of 0 shares one matrix across the batch).
//
// FP16 products are exact in FP32, so fma(a, b, acc) rounds once whether or not the device fuses it, and each
// output accumulates its k products in order: acc = fl(acc + a b) for k = 0, 1, ... Then
// r = fl(fl(alpha acc) + fl(beta c)), with the add marked NoContraction, and C = r (FP32) or half(r) (FP16, one
// rounding to nearest even). Every device that rounds FP32 add and multiply correctly gives the same bits, and
// |error| <= k 2^-24 sum |a b| before alpha, beta and the FP16 rounding. With beta == 0, C is not read.
//
// A group of 16 x 16 threads computes a 64 x 64 tile of C: thread (tx, ty) owns rows ty + 16 i and columns
// tx + 16 j (i, j < 4). k is walked in steps of 16 through shared tiles of A (transposed, so a row of the tile is
// contiguous) and B, converted to FP32 once on load.
//
// The fixed loops are [ForceUnroll]ed: the build compiles at -O0, and without it the SPIR-V keeps the accumulators
// as dynamically indexed arrays, which MoltenVK ran 14x slower (376 against 5261 GFLOP/s at 4096^3 on an M4 Max).
// Measured alternatives on the same GPU: 128 x 128 tiles with 8 x 8 per thread, 3.6 TFLOP/s; k steps of 32,
// 4.2 TFLOP/s; this kernel, 5.6 TFLOP/s on Metal.
// A and B come first: Slang numbers Metal buffers in declaration order.
[[vk::binding(0, 0)]] RWStructuredBuffer<half> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<half> b;
#include "gemm_common.slang"

static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;

groupshared float tileA[BK][BM];
groupshared float tileB[BK][BN];

[shader("compute")]
[numthreads(16, 16, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint tx = local.x, ty = local.y, lid = ty * 16 + tx;
    uint row0 = group.y * BM, col0 = group.x * BN;
    uint baseA = group.z * p.strideA, baseB = group.z * p.strideB, baseC = group.z * p.strideC;
    float acc[4][4];
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
        [ForceUnroll]
        for (uint j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < p.k; k0 += BK)
    {
        // each thread loads 4 elements of each tile; out-of-range elements are zero
        [ForceUnroll]
        for (uint q = 0; q < 4; q++)
        {
            uint e = lid * 4 + q;
            uint ar = e / BK, ak = e % BK;
            uint gr = row0 + ar, gk = k0 + ak;
            uint ai = p.transA != 0 ? gk * p.lda + gr : gr * p.lda + gk;
            tileA[ak][ar] = gr < p.m && gk < p.k ? float(a[baseA + ai]) : 0.0f;
            uint br = e / BN, bc = e % BN;
            uint hk = k0 + br, gc = col0 + bc;
            uint bi = p.transB != 0 ? gc * p.ldb + hk : hk * p.ldb + gc;
            tileB[br][bc] = hk < p.k && gc < p.n ? float(b[baseB + bi]) : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint kk = 0; kk < BK; kk++)
        {
            float av[4], bv[4];
            [ForceUnroll]
            for (uint i = 0; i < 4; i++) av[i] = tileA[kk][ty + 16 * i];
            [ForceUnroll]
            for (uint j = 0; j < 4; j++) bv[j] = tileB[kk][tx + 16 * j];
            [ForceUnroll]
            for (uint i = 0; i < 4; i++)
                [ForceUnroll]
                for (uint j = 0; j < 4; j++)
                    acc[i][j] = fma(av[i], bv[j], acc[i][j]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
    {
        uint r = row0 + ty + 16 * i;
        if (r >= p.m) continue;
        [ForceUnroll]
        for (uint j = 0; j < 4; j++)
        {
            uint col = col0 + tx + 16 * j;
            if (col >= p.n) continue;
            gemmStore(r, col, baseC + r * p.ldc + col, acc[i][j]);
        }
    }
}
