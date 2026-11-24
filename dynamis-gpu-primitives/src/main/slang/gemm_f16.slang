// C = alpha * A * B + beta * C, with A (m x k) and B (k x n) in FP16 and C (m x n) in FP32, all row-major with
// leading dimensions lda, ldb and ldc (in elements). FP16 products are exact in FP32, so fma(a, b, acc) rounds once
// whether or not the device fuses it, and each output accumulates its k products in order: acc = fl(acc + a b) for
// k = 0, 1, ... Then C = fl(fl(alpha acc) + fl(beta c)), with the add marked NoContraction. Every device that rounds
// FP32 add and multiply correctly (Vulkan requires it) gives the same bits, and |error| <= k 2^-24 sum |a b| before
// alpha and beta. With beta == 0, C is not read (as in BLAS), so it may hold anything.
//
// A group of 16 x 16 threads computes a 64 x 64 tile of C: thread (tx, ty) owns rows ty + 16 i and columns
// tx + 16 j (i, j < 4). The k dimension is walked in steps of 16 through shared tiles of A (transposed, so a row of
// the tile is contiguous) and B, converted to FP32 once on load. Built by scripts/build-slang-kernels.sh.
//
// The fixed loops are [ForceUnroll]ed: the build compiles at -O0, and without it the SPIR-V keeps the accumulators
// as dynamically indexed arrays, which MoltenVK ran 14x slower (376 against 5261 GFLOP/s at 4096^3 on an M4 Max).
// Measured alternatives on the same GPU: 128 x 128 tiles with 8 x 8 per thread, 3.6 TFLOP/s; k steps of 32,
// 4.2 TFLOP/s; this kernel, 5.6 TFLOP/s on Metal.
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
};

[[vk::binding(0, 0)]] RWStructuredBuffer<half> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<half> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c;
[[vk::push_constant]] ConstantBuffer<GemmParams> p;

static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;

groupshared float tileA[BK][BM];
groupshared float tileB[BK][BN];

[shader("compute")]
[numthreads(16, 16, 1)]
void gemm_f16(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint tx = local.x, ty = local.y, lid = ty * 16 + tx;
    uint row0 = group.y * BM, col0 = group.x * BN;
    float acc[4][4];
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
        [ForceUnroll]
        for (uint j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < p.k; k0 += BK)
    {
        // each thread loads 4 consecutive halves of each tile; out-of-range elements are zero
        [ForceUnroll]
        for (uint q = 0; q < 4; q++)
        {
            uint e = lid * 4 + q;
            uint ar = e / BK, ak = e % BK;
            uint gr = row0 + ar, gk = k0 + ak;
            tileA[ak][ar] = gr < p.m && gk < p.k ? float(a[gr * p.lda + gk]) : 0.0f;
            uint br = e / BN, bc = e % BN;
            uint hk = k0 + br, gc = col0 + bc;
            tileB[br][bc] = hk < p.k && gc < p.n ? float(b[hk * p.ldb + gc]) : 0.0f;
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
            uint idx = r * p.ldc + col;
            c[idx] = p.beta == 0.0f ? p.alpha * acc[i][j] : p.alpha * acc[i][j] + p.beta * c[idx];
        }
    }
}
