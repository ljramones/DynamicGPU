// The quantized tiled GEMM (1.9.0), for prefill with quantized weights (any m): C = epilogue(alpha A W^T + beta C),
// A (m x k) FP16 or BF16, W a packed QuantizedMatrix (n x k). Built by gemm_q_tiled[_bf16][_h].slang.
//
// A group of 16 x 16 threads computes a 64 x 64 tile of C, thread (tx, ty) rows ty + 16 i and columns tx + 16 j.
// k is walked in steps of 32 through shared tiles of A and of W's codes (as FP32). The order is fixed
// (QuantizedGemm.reference with TILED): each element takes the units (blocks; 32 elements of a K format) in order,
// each unit's s = sum a q in order of k, and adds its term (gemm_q_common.slang) to its total as each unit ends.
#include "gemm_q_common.slang"

static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 32;

// A as raw 16-bit activations, two rows a word: row r in the low half, row r + 32 in the high (exact either way; it
// keeps the kernel within the 16 KiB every device has)
groupshared uint qTileA[BK][BM / 2];
groupshared float qTileB[BK][BN];
// The scales of the unit ending at this k step, for the tile's 64 columns (second: Q6K's second sub-block), staged
// with the tiles: reading them after the step's barrier left each thread waiting on its loads (Q4K at 512 x 4096 x
// 4096 took 64 ms; staged, 4.1). Two slots, by unit: a unit's scales are read after the step's last barrier, so the
// next unit's are written to the other slot, and the one after waits for two more barriers.
// (d, m) of a block, (D, M) of a Q4K or Q5K sub-block, or Q6K's two sub-blocks' D (it has no M). With the tiles,
// 13 KiB of group memory.
groupshared float2 qScale[2][BN];

[shader("compute")]
[numthreads(16, 16, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    GemmProblem pr = gemmBatchProblem(group.z);
    uint tx = local.x, ty = local.y, lid = ty * 16 + tx;
    uint row0 = group.y * BM, col0 = group.x * BN;
    float acc[4][4], s[4][4], s2[4][4], sumA[4];   // s2: the second Q6K sub-block of a unit
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
    {
        sumA[i] = 0.0f;
        [ForceUnroll]
        for (uint j = 0; j < 4; j++) { acc[i][j] = 0.0f; s[i][j] = 0.0f; s2[i][j] = 0.0f; }
    }

    for (uint k0 = 0; k0 < p.k; k0 += BK)
    {
        // each thread loads 4 elements of two rows of A (r and r + 32) and 8 codes of W (an 8-element run of one row)
        {
            uint ar = lid / 8, ak = (lid % 8) * 4, g0 = row0 + ar, g1 = g0 + 32;
            [ForceUnroll]
            for (uint t = 0; t < 4; t++)
            {
                uint lo = g0 < p.m ? rawA(pr.baseA + g0 * p.lda + k0 + ak + t) : 0u;
                uint hi = g1 < p.m ? rawA(pr.baseA + g1 * p.lda + k0 + ak + t) : 0u;
                qTileA[ak + t][ar] = lo | (hi << 16);
            }
            uint wr = col0 + lid / 4, bk = (lid % 4) * 8;
            float q[8];
            load8Codes(min(wr, p.n - 1), k0 + bk, q);
            [ForceUnroll] for (uint t = 0; t < 8; t++) qTileB[bk + t][lid / 4] = q[t];
            if ((k0 + BK) % QUNIT == 0 && lid < BN)
            {
                uint sr = min(col0 + lid, p.n - 1), u = k0 / QUNIT;
                qScale[u & 1][lid] = unitScale(sr, u);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint kk = 0; kk < BK; kk++)
        {
            float av[4], bv[4];
            uint w0 = qTileA[kk][ty], w1 = qTileA[kk][ty + 16];   // rows ty, ty + 32 and ty + 16, ty + 48
            av[0] = aLo(w0); av[1] = aLo(w1); av[2] = aHi(w0); av[3] = aHi(w1);
            [ForceUnroll] for (uint j = 0; j < 4; j++) bv[j] = qTileB[kk][tx + 16 * j];
            [ForceUnroll]
            for (uint i = 0; i < 4; i++)
            {
                if (QMIN) sumA[i] = sumA[i] + av[i];
                [ForceUnroll]
                for (uint j = 0; j < 4; j++)
                {
                    if (QFORMAT == QF_Q6K && kk >= 16) s2[i][j] = fma(av[i], bv[j], s2[i][j]);
                    else s[i][j] = fma(av[i], bv[j], s[i][j]);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if ((k0 + BK) % QUNIT == 0)
        {
            [ForceUnroll]
            for (uint j = 0; j < 4; j++)
            {
                uint slot = (k0 / QUNIT) & 1;
                float2 sc = qScale[slot][tx + 16 * j];
                [ForceUnroll]
                for (uint i = 0; i < 4; i++)
                {
                    acc[i][j] = addTerms(acc[i][j], s[i][j], s2[i][j], sumA[i], sc);
                    s[i][j] = 0.0f; s2[i][j] = 0.0f;
                }
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
