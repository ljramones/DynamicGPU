// The skinny GEMM (m <= 16, the decode step of an LLM), shared by gemm_skinny.slang (FP32 C) and gemm_skinny_h.slang
// (FP16 C), which define OUT_T and ENTRY. Same parameter block and epilogue as gemm_body.slang (gemm_common.slang);
// A is not transposed. It streams B once, with every row
// of A per element of B, so at m = 1 it is limited by memory bandwidth.
//
// Two builds: VEC (gemm_skinny_v*) reads A and B as 16-byte vectors of 8 halves, for B transposed with every row of
// A and B starting on a multiple of 8 elements and k a multiple of 8; the other reads 32-bit words (two halves, the
// even element in the low half), or single halves where a row starts on an odd element. A half that ends a buffer
// is read with its word's other half, which is never used. Both sum in the same order.
//
// The order is fixed, so the result is the same bits on every device and either build (GemmF16.referenceSkinny):
// the k elements are dealt to LANES lanes in chunks of CHUNK (element q to lane (q / CHUNK) % LANES); each lane sums
// its elements in order of q in FP32 (FP16 products are exact, so fma rounds once); then the lanes combine in a
// fixed tree (lane t takes lane t + w, for w = LANES / 2 .. 1). With B transposed LANES is 32 and CHUNK 8; without,
// 16 and 2.
// - B transposed (n x k, the weights of a linear layer): a group of 256 threads computes 8 * CPT_T columns: 8 sets
//   of 32 lanes, each thread CPT_T columns, so a value of A serves CPT_T columns and the lanes of a column read
//   consecutive chunks.
// - B not transposed (k x n): a group computes 64 columns: 16 lanes of 16 threads, each thread 4 adjacent columns,
//   so the 16 threads of a lane read 128 consecutive bytes of a row of B. (8 lanes of 128 columns ran at 38% of the
//   streaming-read bandwidth for n = 4096, too few groups; 16 lanes at 67%.)
// Then r = fl(fl(alpha acc) + fl(beta c)), as in the tiled kernel.

#ifdef VEC
typedef uint4 Word;
#else
typedef uint Word;
#endif
// A and B come first: Slang numbers Metal buffers in declaration order.
[[vk::binding(0, 0)]] RWStructuredBuffer<Word> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<Word> b;
#include "gemm_common.slang"

static const uint THREADS = 256;
static const uint MAXM = 16;
static const uint CPT = 4;        // columns per thread, B not transposed
static const uint CPT_T = 2;      // columns per thread, B transposed
static const uint LANES_T = 32;   // B transposed: 8 column sets of 32 lanes, chunks of 8
static const uint CHUNK_T = 8;
static const uint LANES_N = 16;   // B not transposed: 16 lanes of 16 column quads, chunks of 2

groupshared float red[CPT][THREADS];

float lowHalf(uint w) { return f16tof32(w & 0xFFFF); }
float highHalf(uint w) { return f16tof32(w >> 16); }

#ifdef VEC
// 8 halves from element e, a multiple of 8
void load8A(uint e, out float v[8]) { uint4 w = a[e >> 3]; [ForceUnroll] for (uint t = 0; t < 4; t++) { v[2 * t] = lowHalf(w[t]); v[2 * t + 1] = highHalf(w[t]); } }
void load8B(uint e, out float v[8]) { uint4 w = b[e >> 3]; [ForceUnroll] for (uint t = 0; t < 4; t++) { v[2 * t] = lowHalf(w[t]); v[2 * t + 1] = highHalf(w[t]); } }
float halfA(uint e) { return 0.0f; }
float halfB(uint e) { return 0.0f; }
#else
float halfA(uint e) { uint w = a[e >> 1]; return (e & 1) != 0 ? highHalf(w) : lowHalf(w); }
float halfB(uint e) { uint w = b[e >> 1]; return (e & 1) != 0 ? highHalf(w) : lowHalf(w); }
// 8 halves from element e: as words when e is even, else one at a time
void load8A(uint e, out float v[8])
{
    if ((e & 1) == 0) { [ForceUnroll] for (uint t = 0; t < 4; t++) { uint w = a[(e >> 1) + t]; v[2 * t] = lowHalf(w); v[2 * t + 1] = highHalf(w); } }
    else { [ForceUnroll] for (uint t = 0; t < 8; t++) v[t] = halfA(e + t); }
}
void load8B(uint e, out float v[8])
{
    if ((e & 1) == 0) { [ForceUnroll] for (uint t = 0; t < 4; t++) { uint w = b[(e >> 1) + t]; v[2 * t] = lowHalf(w); v[2 * t + 1] = highHalf(w); } }
    else { [ForceUnroll] for (uint t = 0; t < 8; t++) v[t] = halfB(e + t); }
}
#endif

[shader("compute")]
[numthreads(256, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint lid = local.x;
    uint baseA = group.z * p.strideA, baseB = group.z * p.strideB, baseC = group.z * p.strideC;
    uint m = min(p.m, MAXM), k = p.k;
    float acc[MAXM][CPT];
    [ForceUnroll]
    for (uint i = 0; i < MAXM; i++)
        [ForceUnroll]
        for (uint cc = 0; cc < CPT; cc++)
            acc[i][cc] = 0.0f;

    bool transB = p.transB != 0;
    uint lanes = transB ? LANES_T : LANES_N;
    uint lane = transB ? lid % LANES_T : lid / (THREADS / LANES_N);
    uint cols = transB ? CPT_T : CPT;
    uint col0 = transB ? group.x * (THREADS / LANES_T * CPT_T) + lid / LANES_T * CPT_T
                       : group.x * (THREADS / LANES_N * CPT) + lid % (THREADS / LANES_N) * CPT;

    if (col0 < p.n && transB)
    {
        uint chunks = k / CHUNK_T;
        for (uint ch = lane; ch < chunks; ch += LANES_T)
        {
            uint q = ch * CHUNK_T;
            float bv[CPT_T][8];
            [ForceUnroll]
            for (uint cc = 0; cc < CPT_T; cc++) load8B(baseB + min(col0 + cc, p.n - 1) * p.ldb + q, bv[cc]);
            [ForceUnroll]
            for (uint i = 0; i < MAXM; i++)
            {
                if (i >= m) break;
                float av[8];
                load8A(baseA + i * p.lda + q, av);
                [ForceUnroll]
                for (uint t = 0; t < 8; t++)
                    [ForceUnroll]
                    for (uint cc = 0; cc < CPT_T; cc++)
                        acc[i][cc] = fma(av[t], bv[cc][t], acc[i][cc]);
            }
        }
#ifndef VEC
        // the last, partial chunk is its lane's last
        if (k % CHUNK_T != 0 && lane == chunks % LANES_T)
        {
            for (uint q = chunks * CHUNK_T; q < k; q++)
            {
                float bv[CPT_T];
                [ForceUnroll]
                for (uint cc = 0; cc < CPT_T; cc++) bv[cc] = halfB(baseB + min(col0 + cc, p.n - 1) * p.ldb + q);
                [ForceUnroll]
                for (uint i = 0; i < MAXM; i++)
                {
                    if (i >= m) break;
                    float av = halfA(baseA + i * p.lda + q);
                    [ForceUnroll]
                    for (uint cc = 0; cc < CPT_T; cc++) acc[i][cc] = fma(av, bv[cc], acc[i][cc]);
                }
            }
        }
#endif
    }
#ifndef VEC
    else if (col0 < p.n)
    {
        bool evenA = ((p.lda | p.strideA) & 1) == 0;
        bool evenB = ((p.ldb | p.strideB) & 1) == 0 && col0 + CPT <= p.n;
        for (uint q2 = lane; 2 * q2 < k; q2 += LANES_N)
        {
            uint q = 2 * q2;
            bool two = q + 1 < k;
            float b0[CPT], b1[CPT];
            [ForceUnroll]
            for (uint h = 0; h < 2; h++)
            {
                uint e = baseB + min(q + h, k - 1) * p.ldb + col0;
                float v[CPT];
                if (evenB)
                {
                    uint w0 = b[e >> 1], w1 = b[(e >> 1) + 1];
                    v[0] = lowHalf(w0); v[1] = highHalf(w0); v[2] = lowHalf(w1); v[3] = highHalf(w1);
                }
                else
                {
                    [ForceUnroll]
                    for (uint cc = 0; cc < CPT; cc++) v[cc] = halfB(e + min(cc, p.n - 1 - col0));
                }
                [ForceUnroll]
                for (uint cc = 0; cc < CPT; cc++) { if (h == 0) b0[cc] = v[cc]; else b1[cc] = v[cc]; }
            }
            [ForceUnroll]
            for (uint i = 0; i < MAXM; i++)
            {
                if (i >= m) break;
                uint e = baseA + i * p.lda + q;
                float a0, a1;
                if (evenA && two) { uint w = a[e >> 1]; a0 = lowHalf(w); a1 = highHalf(w); }
                else { a0 = halfA(e); a1 = two ? halfA(e + 1) : 0.0f; }
                [ForceUnroll]
                for (uint cc = 0; cc < CPT; cc++)
                {
                    acc[i][cc] = fma(a0, b0[cc], acc[i][cc]);
                    if (two) acc[i][cc] = fma(a1, b1[cc], acc[i][cc]);
                }
            }
        }
    }
#endif

    // combine the lanes of each column in the fixed tree; lane l of a column is at lid + l * step
    uint step = transB ? 1 : THREADS / LANES_N;
    for (uint i = 0; i < m; i++)
    {
        [ForceUnroll]
        for (uint cc = 0; cc < CPT; cc++) red[cc][lid] = acc[i][cc];
        for (uint w = lanes / 2; w > 0; w /= 2)
        {
            GroupMemoryBarrierWithGroupSync();
            if (lane < w)
            {
                [ForceUnroll]
                for (uint cc = 0; cc < CPT; cc++)
                    if (cc < cols) red[cc][lid] = red[cc][lid] + red[cc][lid + w * step];
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0)
        {
            for (uint cc = 0; cc < cols; cc++)
                if (col0 + cc < p.n) gemmStore(i, col0 + cc, baseC + i * p.ldc + col0 + cc, red[cc][lid]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
