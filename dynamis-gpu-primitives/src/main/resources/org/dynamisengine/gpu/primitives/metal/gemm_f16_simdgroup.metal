// GemmF16 on Apple GPU matrix hardware (simdgroup_matrix, Apple7 and later). Hand-written: Slang cannot emit Metal
// SIMD-group matrices. Same ABI as gemm_body.slang: buffers a, b, c at 0, 1, 2 and the parameters at 3; group z is
// the batch index. Two entry points: gemm_f16_simdgroup (FP32 C) and gemm_f16_h_simdgroup (FP16 C, one rounding).
//
// A group of 128 threads (4 SIMD-groups) computes a 64 x 64 tile of C; each SIMD-group owns a 32 x 32 quarter as
// 4 x 4 accumulators of 8 x 8 floats. Interior tiles with k a multiple of 8 load 8 x 8 blocks straight from device
// memory (the caches serve the reuse; B^T blocks load transposed). Edge tiles stage zero-filled copies in threadgroup
// memory, k in steps of 32. The result goes through threadgroup memory so edges and alpha/beta are applied per
// element. Staging every tile, a half at a time, ran at 3.7 TFLOP/s; the direct loads at 12.9 (4096^3, M4 Max).
#include <metal_stdlib>
using namespace metal;

struct GemmParams {
    uint m; uint n; uint k; uint lda; uint ldb; uint ldc; float alpha; float beta;
    uint transB; uint strideA; uint strideB; uint strideC;
};

constant constexpr uint BM = 64, BN = 64, BK = 32, THREADS = 128;

template <typename OutT>
static void gemm(device const half *a, device const half *b, device OutT *c, constant GemmParams &p, uint3 group,
                 uint tid, uint sg, threadgroup half *tileA, threadgroup half *tileB, threadgroup float *tileC) {
    a += group.z * p.strideA; b += group.z * p.strideB; c += group.z * p.strideC;
    const uint row0 = group.y * BM, col0 = group.x * BN;
    const uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;
    const bool transB = p.transB != 0;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const bool interior = row0 + BM <= p.m && col0 + BN <= p.n && p.k % 8 == 0;
    if (interior && transB) {
        // B is n x k: the k x 8 block at column col is the transpose of rows col..col+7, columns k0..k0+7
        device const half *arow = a + (row0 + sr) * p.lda, *brow = b + (col0 + sc) * p.ldb;
        for (uint k0 = 0; k0 < p.k; k0 += 8) {
            simdgroup_half8x8 ma[4], mb[4];
            for (uint i = 0; i < 4; i++) simdgroup_load(ma[i], arow + 8 * i * p.lda + k0, p.lda);
            for (uint j = 0; j < 4; j++) simdgroup_load(mb[j], brow + 8 * j * p.ldb + k0, p.ldb, ulong2(0, 0), true);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
    } else if (interior) {
        device const half *arow = a + (row0 + sr) * p.lda, *bcol = b + col0 + sc;
        for (uint k0 = 0; k0 < p.k; k0 += 8) {
            simdgroup_half8x8 ma[4], mb[4];
            for (uint i = 0; i < 4; i++) simdgroup_load(ma[i], arow + 8 * i * p.lda + k0, p.lda);
            for (uint j = 0; j < 4; j++) simdgroup_load(mb[j], bcol + k0 * p.ldb + 8 * j, p.ldb);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
    } else for (uint k0 = 0; k0 < p.k; k0 += BK) {
        for (uint q = 0; q < BM * BK / THREADS; q++) {
            uint e = tid + q * THREADS, r = e / BK, kk = e % BK, gr = row0 + r, gk = k0 + kk;
            tileA[e] = gr < p.m && gk < p.k ? a[gr * p.lda + gk] : half(0);
        }
        for (uint q = 0; q < BK * BN / THREADS; q++) {
            uint e = tid + q * THREADS, r = e / BN, cc = e % BN, gk = k0 + r, gc = col0 + cc;
            tileB[e] = gk < p.k && gc < p.n ? b[transB ? gc * p.ldb + gk : gk * p.ldb + gc] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 ma[4], mb[4];
            for (uint i = 0; i < 4; i++) simdgroup_load(ma[i], tileA + (sr + 8 * i) * BK + kk, BK);
            for (uint j = 0; j < 4; j++) simdgroup_load(mb[j], tileB + kk * BN + sc + 8 * j, BN);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            simdgroup_store(acc[i][j], tileC + (sr + 8 * i) * BN + sc + 8 * j, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = 0; q < BM * BN / THREADS; q++) {
        uint e = tid + q * THREADS, r = e / BN, cc = e % BN, gr = row0 + r, gc = col0 + cc;
        if (gr < p.m && gc < p.n) {
            uint idx = gr * p.ldc + gc;
            float v = p.beta == 0.0f ? p.alpha * tileC[e] : p.alpha * tileC[e] + p.beta * float(c[idx]);
            c[idx] = OutT(v);
        }
    }
}

kernel void gemm_f16_simdgroup(device const half *a [[buffer(0)]], device const half *b [[buffer(1)]],
                               device float *c [[buffer(2)]], constant GemmParams &p [[buffer(3)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half tileA[BM * BK];
    threadgroup half tileB[BK * BN];
    threadgroup float tileC[BM * BN];
    gemm<float>(a, b, c, p, group, tid, sg, tileA, tileB, tileC);
}

kernel void gemm_f16_h_simdgroup(device const half *a [[buffer(0)]], device const half *b [[buffer(1)]],
                                 device half *c [[buffer(2)]], constant GemmParams &p [[buffer(3)]],
                                 uint3 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half tileA[BM * BK];
    threadgroup half tileB[BK * BN];
    threadgroup float tileC[BM * BN];
    gemm<half>(a, b, c, p, group, tid, sg, tileA, tileB, tileC);
}
