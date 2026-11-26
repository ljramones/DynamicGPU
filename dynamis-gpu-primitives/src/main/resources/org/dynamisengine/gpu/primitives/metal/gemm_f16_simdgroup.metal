// GemmF16 on Apple GPU matrix hardware (simdgroup_matrix, Apple7 and later). Hand-written: Slang cannot emit Metal
// SIMD-group matrices. Entries for FP16 inputs (gemm_f16[_h]_simdgroup) and FP32 inputs (gemm_f32[_h]_simdgroup,
// 1.8.0); "_h" writes an FP16 C, rounded once. Same ABI as gemm_body.slang (gemm_common.slang): buffers a, b, c at
// 0, 1, 2, the epilogue's bias, gate and residual at 3, 4, 5, and the parameters at 6; group z is the batch index.
// A may be transposed (stored k x m), and the epilogue is the same as the Slang kernels', in FP32 before the one
// rounding.
//
// A group of 128 threads (4 SIMD-groups) computes a 64 x 64 tile of C; each SIMD-group owns a 32 x 32 quarter as
// 4 x 4 accumulators of 8 x 8 floats. Interior tiles with k a multiple of 8 load 8 x 8 blocks straight from device
// memory (the caches serve the reuse; B^T blocks load transposed). Edge tiles stage zero-filled copies in threadgroup
// memory, k in steps of BK, except row-only edges (1.15.0, below). The result goes through threadgroup memory so edges
// and alpha/beta are applied per element. Staging every tile, a half at a time, ran at 3.7 TFLOP/s; the direct loads
// at 12.9 (4096^3, M4 Max).
#include <metal_stdlib>
using namespace metal;

struct GemmParams {
    uint m; uint n; uint k; uint lda; uint ldb; uint ldc; float alpha; float beta;
    uint transB; uint strideA; uint strideB; uint strideC;
    uint transA; uint bias; uint act; uint gated; uint residual;
};

// gemm_common.slang's gemmEpilogue: bias, activation (ReLU with Java max semantics, GELU's tanh form, SiLU), gate,
// residual
template <typename OutT>
static float epilogue(float r, uint i, uint j, uint idx, constant GemmParams &p, device const OutT *bias,
                      device const OutT *gate, device const OutT *residual) {
    if (p.bias == 1) r = r + float(bias[j]);
    else if (p.bias == 2) r = r + float(bias[i]);
    if (p.act == 1) r = isnan(r) ? r : r > 0.0f ? r : 0.0f;
    else if (p.act == 2) {
        float u = 0.7978845608f * (r + 0.044715f * r * r * r);
        float t = 1.0f - 2.0f / (exp(2.0f * u) + 1.0f);
        r = 0.5f * r * (1.0f + t);
    }
    else if (p.act == 3) r = r / (1.0f + exp(-r));
    if (p.gated != 0) r = r * float(gate[idx]);
    if (p.residual != 0) r = r + float(residual[idx]);
    return r;
}

constant constexpr uint BM = 64, BN = 64, THREADS = 128;

// InT: half (FP16 inputs, edge tiles k in steps of BK = 32) or float (FP32 inputs, 1.8.0; BK = 16 so the staging
// tiles and C fit in 32 KiB of threadgroup memory)
// EDGE: compile the row-edge path (1.15.0). Only the last row of tiles takes that copy. The path inside the same
// function made every tile 1.19-1.29 times slower, interior ones included (an A/B of the two kernels in the same
// rounds, PerfProbeConformance -Dperf.only=kernelab); as its own copy the interior runs at 0.99-1.00 of before.
template <typename InT, typename OutT, uint BK, bool EDGE>
static void gemm(device const InT *a, device const InT *b, device OutT *c, device const OutT *bias,
                 device const OutT *gate, device const OutT *residual, constant GemmParams &p, uint3 group,
                 uint tid, uint sg, threadgroup InT *tileA, threadgroup InT *tileB, threadgroup float *tileC) {
    a += group.z * p.strideA; b += group.z * p.strideB; c += group.z * p.strideC;
    gate += group.z * p.strideC; residual += group.z * p.strideC;
    const uint row0 = group.y * BM, col0 = group.x * BN;
    const uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;
    const bool transB = p.transB != 0, transA = p.transA != 0;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const bool interior = row0 + BM <= p.m && col0 + BN <= p.n && p.k % 8 == 0;
    const bool rowEdge = EDGE && !interior && !transA && p.m >= 8 && col0 + BN <= p.n && p.k % 8 == 0;
    if (interior && transA) {
        // A is k x m: the 8 x k block at row r is the transpose of rows k0..k0+7, columns r..r+7
        device const InT *acol = a + row0 + sr;
        for (uint k0 = 0; k0 < p.k; k0 += 8) {
            simdgroup_matrix<InT, 8, 8> ma[4], mb[4];
            for (uint i = 0; i < 4; i++) simdgroup_load(ma[i], acol + k0 * p.lda + 8 * i, p.lda, ulong2(0, 0), true);
            for (uint j = 0; j < 4; j++) {
                if (transB) simdgroup_load(mb[j], b + (col0 + sc + 8 * j) * p.ldb + k0, p.ldb, ulong2(0, 0), true);
                else simdgroup_load(mb[j], b + k0 * p.ldb + col0 + sc + 8 * j, p.ldb);
            }
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
    } else if (interior && transB) {
        // B is n x k: the k x 8 block at column col is the transpose of rows col..col+7, columns k0..k0+7
        device const InT *arow = a + (row0 + sr) * p.lda, *brow = b + (col0 + sc) * p.ldb;
        for (uint k0 = 0; k0 < p.k; k0 += 8) {
            simdgroup_matrix<InT, 8, 8> ma[4], mb[4];
            for (uint i = 0; i < 4; i++) simdgroup_load(ma[i], arow + 8 * i * p.lda + k0, p.lda);
            for (uint j = 0; j < 4; j++) simdgroup_load(mb[j], brow + 8 * j * p.ldb + k0, p.ldb, ulong2(0, 0), true);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
    } else if (interior) {
        device const InT *arow = a + (row0 + sr) * p.lda, *bcol = b + col0 + sc;
        for (uint k0 = 0; k0 < p.k; k0 += 8) {
            simdgroup_matrix<InT, 8, 8> ma[4], mb[4];
            for (uint i = 0; i < 4; i++) simdgroup_load(ma[i], arow + 8 * i * p.lda + k0, p.lda);
            for (uint j = 0; j < 4; j++) simdgroup_load(mb[j], bcol + k0 * p.ldb + 8 * j, p.ldb);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
    } else if (rowEdge) {
        // A row-only edge (1.15.0): the tile's columns are complete and k is aligned, only rows pass m. Every 8-row block
        // of A loads directly, from row min(block, m - 8), so each load stays inside A and the loop is the interior's.
        // A block that m cuts then holds its valid rows as its last ones, and one wholly past m is not stored (below).
        // Rows of a product are independent and the sums keep the interior's order, so the stored rows are the same
        // bits as with A padded by zero rows. Staging whole tiles here made m = 127 take twice m = 128's time.
        device const InT *ab[4];
        for (uint i = 0; i < 4; i++) ab[i] = a + min(row0 + sr + 8 * i, p.m - 8) * p.lda;
        if (transB) {
            device const InT *brow = b + (col0 + sc) * p.ldb;
            for (uint k0 = 0; k0 < p.k; k0 += 8) {
                simdgroup_matrix<InT, 8, 8> ma[4], mb[4];
                for (uint i = 0; i < 4; i++) simdgroup_load(ma[i], ab[i] + k0, p.lda);
                for (uint j = 0; j < 4; j++) simdgroup_load(mb[j], brow + 8 * j * p.ldb + k0, p.ldb, ulong2(0, 0), true);
                for (uint i = 0; i < 4; i++)
                    for (uint j = 0; j < 4; j++)
                        simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
            }
        } else {
            device const InT *bcol = b + col0 + sc;
            for (uint k0 = 0; k0 < p.k; k0 += 8) {
                simdgroup_matrix<InT, 8, 8> ma[4], mb[4];
                for (uint i = 0; i < 4; i++) simdgroup_load(ma[i], ab[i] + k0, p.lda);
                for (uint j = 0; j < 4; j++) simdgroup_load(mb[j], bcol + k0 * p.ldb + 8 * j, p.ldb);
                for (uint i = 0; i < 4; i++)
                    for (uint j = 0; j < 4; j++)
                        simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
            }
        }
    } else for (uint k0 = 0; k0 < p.k; k0 += BK) {
        for (uint q = 0; q < BM * BK / THREADS; q++) {
            uint e = tid + q * THREADS, r = e / BK, kk = e % BK, gr = row0 + r, gk = k0 + kk;
            tileA[e] = gr < p.m && gk < p.k ? a[transA ? gk * p.lda + gr : gr * p.lda + gk] : InT(0);
        }
        for (uint q = 0; q < BK * BN / THREADS; q++) {
            uint e = tid + q * THREADS, r = e / BN, cc = e % BN, gk = k0 + r, gc = col0 + cc;
            tileB[e] = gk < p.k && gc < p.n ? b[transB ? gc * p.ldb + gk : gk * p.ldb + gc] : InT(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<InT, 8, 8> ma[4], mb[4];
            for (uint i = 0; i < 4; i++) simdgroup_load(ma[i], tileA + (sr + 8 * i) * BK + kk, BK);
            for (uint j = 0; j < 4; j++) simdgroup_load(mb[j], tileB + kk * BN + sc + 8 * j, BN);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (rowEdge) {
        // a block whose rows came from m - 8 goes through this SIMD-group's share of tileA (free here) to its place
        threadgroup float *shifted = (threadgroup float *) tileA + sg * 256;
        const uint lane = tid & 31u;
        for (uint i = 0; i < 4; i++) {
            uint rb = row0 + sr + 8 * i;
            if (rb + 8 <= p.m) {
                for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], tileC + (sr + 8 * i) * BN + sc + 8 * j, BN);
            } else if (rb < p.m) {
                uint valid = p.m - rb;   // its last `valid` rows are rows rb .. m - 1
                for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], shifted + 8 * j, 32);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < valid * 32; e += 32)
                    tileC[(sr + 8 * i + e / 32) * BN + sc + e % 32] = shifted[(8 - valid + e / 32) * 32 + e % 32];
            }
        }
    } else {
        for (uint i = 0; i < 4; i++)
            for (uint j = 0; j < 4; j++)
                simdgroup_store(acc[i][j], tileC + (sr + 8 * i) * BN + sc + 8 * j, BN);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = 0; q < BM * BN / THREADS; q++) {
        uint e = tid + q * THREADS, r = e / BN, cc = e % BN, gr = row0 + r, gc = col0 + cc;
        if (gr < p.m && gc < p.n) {
            uint idx = gr * p.ldc + gc;
            float v = p.beta == 0.0f ? p.alpha * tileC[e] : p.alpha * tileC[e] + p.beta * float(c[idx]);
            c[idx] = OutT(epilogue<OutT>(v, gr, gc, idx, p, bias, gate, residual));
        }
    }
}

#define GEMM_ENTRY(NAME, InT, OutT, BK)                                                                          \
kernel void NAME(device const InT *a [[buffer(0)]], device const InT *b [[buffer(1)]], device OutT *c [[buffer(2)]],     \
                 device const OutT *bias [[buffer(3)]], device const OutT *gate [[buffer(4)]],                         \
                 device const OutT *residual [[buffer(5)]], constant GemmParams &p [[buffer(6)]],                     \
                 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                \
                 uint sg [[simdgroup_index_in_threadgroup]]) {                                                          \
    threadgroup InT tileA[BM * BK];                                                                                     \
    threadgroup InT tileB[BK * BN];                                                                                     \
    threadgroup float tileC[BM * BN];                                                                                   \
    /* full-height tiles run the interior code as it was before 1.15.0; the row-edge path is in its own copy */      \
    if (group.y * BM + BM <= p.m) gemm<InT, OutT, BK, false>(a, b, c, bias, gate, residual, p, group, tid, sg, tileA, tileB, tileC); \
    else gemm<InT, OutT, BK, true>(a, b, c, bias, gate, residual, p, group, tid, sg, tileA, tileB, tileC);             \
}

GEMM_ENTRY(gemm_f16_simdgroup, half, float, 32)
GEMM_ENTRY(gemm_f16_h_simdgroup, half, half, 32)
GEMM_ENTRY(gemm_f32_simdgroup, float, float, 16)
GEMM_ENTRY(gemm_f32_h_simdgroup, float, half, 16)
