// The quantized skinny GEMM (1.9.0), the decode step with quantized weights: C = epilogue(alpha A W^T + beta C), A
// (m x k, m <= 16) FP16 or BF16 activations, W a packed QuantizedMatrix (n x k). Built per input and output type by
// gemm_q_skinny[_bf16][_h].slang; specialized per format, block size and rows (QROWS: 1, 4 or 16, the most rows of A
// it computes, so a decode step holds one row of accumulators, not sixteen).
//
// The order is fixed (QuantizedGemm.reference with SKINNY): the blocks of a row of W are dealt to 32 lanes (block bl
// to lane bl % 32); each lane takes its blocks in order, each block's s = sum a q in order of k, and adds
// fl(d s) [+ fl(m sum a)] to its total; then the lanes combine in a fixed tree (lane t takes lane t + w, w = 16 .. 1).
// A group of 256 threads computes 16 columns: 8 sets of 32 lanes, each thread 2 columns, so one read of A serves both.
#include "gemm_q_common.slang"

[vk::constant_id(3)] const uint QROWS = 16;

static const uint THREADS = 256;
static const uint MAXM = 16;
static const uint CPT = 2;
static const uint LANES = 32;

groupshared float red[CPT][THREADS];

[shader("compute")]
[numthreads(256, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint lid = local.x;
    GemmProblem pr = gemmBatchProblem(group.z);
    uint m = min(p.m, QROWS), lane = lid % LANES;
    uint col0 = group.x * (THREADS / LANES * CPT) + lid / LANES * CPT;
    float acc[MAXM][CPT];
    [ForceUnroll]
    for (uint i = 0; i < MAXM; i++)
        [ForceUnroll]
        for (uint cc = 0; cc < CPT; cc++)
            acc[i][cc] = 0.0f;

    if (col0 < p.n)
    {
        uint blocks = p.k / QBLOCK;
        uint rows[CPT];
        [ForceUnroll] for (uint cc = 0; cc < CPT; cc++) rows[cc] = min(col0 + cc, p.n - 1);
        for (uint bl = lane; bl < blocks; bl += LANES)
        {
            float d[CPT], mn[CPT];
            [ForceUnroll]
            for (uint cc = 0; cc < CPT; cc++)
            {
                d[cc] = blockParameter(rows[cc], bl, 0);
                mn[cc] = QFORMAT == QF_Q4M ? blockParameter(rows[cc], bl, 1) : 0.0f;
            }
            float s[MAXM][CPT], sumA[MAXM];
            [ForceUnroll]
            for (uint i = 0; i < MAXM; i++) { sumA[i] = 0.0f; [ForceUnroll] for (uint cc = 0; cc < CPT; cc++) s[i][cc] = 0.0f; }
            for (uint c8 = 0; c8 < QBLOCK; c8 += 8)
            {
                uint q0 = bl * QBLOCK + c8;
                // this run's codes for both columns, decoded once for every row of A
                float q[CPT][8];
                [ForceUnroll]
                for (uint cc = 0; cc < CPT; cc++) load8Codes(rows[cc], q0, q[cc]);
                [ForceUnroll]
                for (uint i = 0; i < MAXM; i++)
                {
                    if (i >= m) break;
                    float av[8];
                    load8A(pr.baseA + i * p.lda + q0, av);
                    [ForceUnroll]
                    for (uint t = 0; t < 8; t++)
                    {
                        if (QFORMAT == QF_Q4M) sumA[i] = sumA[i] + av[t];
                        [ForceUnroll]
                        for (uint cc = 0; cc < CPT; cc++)
                            s[i][cc] = fma(av[t], q[cc][t], s[i][cc]);
                    }
                }
            }
            [ForceUnroll]
            for (uint i = 0; i < MAXM; i++)
            {
                if (i >= m) break;
                [ForceUnroll] for (uint cc = 0; cc < CPT; cc++) acc[i][cc] = acc[i][cc] + blockTerm(s[i][cc], sumA[i], d[cc], mn[cc]);
            }
        }
    }

    for (uint i = 0; i < m; i++)
    {
        [ForceUnroll]
        for (uint cc = 0; cc < CPT; cc++) red[cc][lid] = acc[i][cc];
        for (uint w = LANES / 2; w > 0; w /= 2)
        {
            GroupMemoryBarrierWithGroupSync();
            if (lane < w)
            {
                [ForceUnroll]
                for (uint cc = 0; cc < CPT; cc++) red[cc][lid] = red[cc][lid] + red[cc][lid + w];
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0)
        {
            [ForceUnroll]
            for (uint cc = 0; cc < CPT; cc++)
                if (col0 + cc < p.n) gemmStore(pr, i, col0 + cc, red[cc][lid]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
