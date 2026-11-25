// The quantized tiled GEMM (1.9.0), for prefill with quantized weights (any m): C = epilogue(alpha A W^T + beta C),
// A (m x k) FP16 or BF16, W a packed QuantizedMatrix (n x k). Built by gemm_q_tiled[_bf16][_h].slang.
//
// A group of 16 x 16 threads computes a 64 x 64 tile of C, thread (tx, ty) rows ty + 16 i and columns tx + 16 j.
// k is walked in steps of 32 through shared tiles of A and of W's codes (as FP32). The order is fixed
// (QuantizedGemm.reference with TILED): each element takes the blocks in order, each block's s = sum a q in order of
// k, and adds fl(d s) [+ fl(m sum a)] to its total as each block ends.
#include "gemm_q_common.slang"

static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 32;

groupshared float qTileA[BK][BM];
groupshared float qTileB[BK][BN];

[shader("compute")]
[numthreads(16, 16, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    GemmProblem pr = gemmBatchProblem(group.z);
    uint tx = local.x, ty = local.y, lid = ty * 16 + tx;
    uint row0 = group.y * BM, col0 = group.x * BN;
    float acc[4][4], s[4][4], sumA[4];
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
    {
        sumA[i] = 0.0f;
        [ForceUnroll]
        for (uint j = 0; j < 4; j++) { acc[i][j] = 0.0f; s[i][j] = 0.0f; }
    }

    for (uint k0 = 0; k0 < p.k; k0 += BK)
    {
        // each thread loads 8 elements of A and 8 codes of W (one 8-element run of one row)
        {
            uint ar = lid / 4, ak = (lid % 4) * 8, gr = row0 + ar;
            float av[8];
            if (gr < p.m) load8A(pr.baseA + gr * p.lda + k0 + ak, av);
            else { [ForceUnroll] for (uint t = 0; t < 8; t++) av[t] = 0.0f; }
            [ForceUnroll] for (uint t = 0; t < 8; t++) qTileA[ak + t][ar] = av[t];
            uint wr = col0 + lid / 4;
            float q[8];
            load8Codes(min(wr, p.n - 1), k0 + ak, q);
            [ForceUnroll] for (uint t = 0; t < 8; t++) qTileB[ak + t][lid / 4] = q[t];
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint kk = 0; kk < BK; kk++)
        {
            float av[4], bv[4];
            [ForceUnroll] for (uint i = 0; i < 4; i++) av[i] = qTileA[kk][ty + 16 * i];
            [ForceUnroll] for (uint j = 0; j < 4; j++) bv[j] = qTileB[kk][tx + 16 * j];
            [ForceUnroll]
            for (uint i = 0; i < 4; i++)
            {
                if (QFORMAT == QF_Q4M) sumA[i] = sumA[i] + av[i];
                [ForceUnroll] for (uint j = 0; j < 4; j++) s[i][j] = fma(av[i], bv[j], s[i][j]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if ((k0 + BK) % QBLOCK == 0)
        {
            uint bl = k0 / QBLOCK;
            [ForceUnroll]
            for (uint j = 0; j < 4; j++)
            {
                uint wr = min(col0 + tx + 16 * j, p.n - 1);
                float d = blockParameter(wr, bl, 0), mn = QFORMAT == QF_Q4M ? blockParameter(wr, bl, 1) : 0.0f;
                [ForceUnroll]
                for (uint i = 0; i < 4; i++) { acc[i][j] = acc[i][j] + blockTerm(s[i][j], sumA[i], d, mn); s[i][j] = 0.0f; }
            }
            [ForceUnroll] for (uint i = 0; i < 4; i++) sumA[i] = 0.0f;
        }
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
            if (col < p.n) gemmStore(pr, r, col, acc[i][j]);
        }
    }
}
