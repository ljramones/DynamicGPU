// The tiled GEMM's 64 x 64 tile, shared by the tiled (gemm_body.slang) and grouped (gemm_grouped_body.slang)
// kernels: see gemm_body.slang for its order and measurements. Included after gemm_epilogue.slang.
static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;

groupshared float tileA[BK][BM];
groupshared float tileB[BK][BN];

// The 64 x 64 tile (tile.x, tile.y) of problem pr's C. Every thread of the group must call it.
void gemmTile(GemmProblem pr, uint2 tile, uint3 local)
{
    uint tx = local.x, ty = local.y, lid = ty * 16 + tx;
    uint row0 = tile.y * BM, col0 = tile.x * BN;
    uint baseA = pr.baseA, baseB = pr.baseB;
    float acc[4][4];
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
        [ForceUnroll]
        for (uint j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < pr.k; k0 += BK)
    {
        // each thread loads 4 elements of each tile; out-of-range elements are zero
        [ForceUnroll]
        for (uint q = 0; q < 4; q++)
        {
            uint e = lid * 4 + q;
            uint ar = e / BK, ak = e % BK;
            uint gr = row0 + ar, gk = k0 + ak;
            uint ai = p.transA != 0 ? gk * pr.lda + gr : gr * pr.lda + gk;
            tileA[ak][ar] = gr < pr.m && gk < pr.k ? loadA(baseA + ai) : 0.0f;
            uint br = e / BN, bc = e % BN;
            uint hk = k0 + br, gc = col0 + bc;
            uint bi = p.transB != 0 ? gc * pr.ldb + hk : hk * pr.ldb + gc;
            tileB[br][bc] = hk < pr.k && gc < pr.n ? loadB(baseB + bi) : 0.0f;
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
        if (r >= pr.m) continue;
        [ForceUnroll]
        for (uint j = 0; j < 4; j++)
        {
            uint col = col0 + tx + 16 * j;
            if (col >= pr.n) continue;
            gemmStore(pr, r, col, acc[i][j]);
        }
    }
}
