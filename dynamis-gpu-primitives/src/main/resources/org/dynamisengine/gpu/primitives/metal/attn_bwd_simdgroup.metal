// AttentionF16.backward on Apple GPU matrix hardware (simdgroup_matrix; 1.12.1). Hand-written: Slang cannot emit Metal
// SIMD-group matrices. Same ABI as attn_bwd_body.slang (attn_bwd_common.slang): q, k, v, dO, lse, D and the outputs at
// buffers 0 to 7 (dQ at 6; dK, dV at 6, 7), the parameters after them. Head dimensions 64 and 128.
//     P = exp(scale Q K^T - L) (masked: 0),  dS = P (dO V^T - D)
//     dQ = scale dS K  (attn_bwd_q_*_simdgroup),  dV = P^T dO,  dK = scale dS^T Q  (attn_bwd_kv_*_simdgroup)
// The products take FP16 operands, as the forward matrix path's P V does: dO, P and dS are rounded to FP16 for them
// (Q, K and V are FP16 already), and every product accumulates in FP32 in the hardware's fixed order. S and dP are
// stored to threadgroup memory, where 4 lanes per row compute P and dS in FP32. Blocks of K and V (dQ) or of Q and dO
// (dK, dV) are staged in threadgroup memory, zero past the keys or rows, and blocks no pair of which is visible under
// the causal mask are skipped.
// Measured (M4 Max, 32 heads x 128, causal, sequence 2048, each kernel alone): dQ 16.8 ms here against about 22 for the
// portable kernel, dK and dV 21.1 against 29.1. Key blocks of 16 for dQ (half the threadgroup memory, two groups a
// core) measured the same; holding each SIMD-group's K and V in registers for dK and dV, with 32-row tiles, 23.6.
#include <metal_stdlib>
using namespace metal;

struct AttnBwdParams {
    uint heads; uint kvHeads; uint queries; uint keys; uint headDim; uint cacheKeys; float scale; uint causal;
    uint gridX; uint pad0; uint pad1; uint pad2;
};

// ---- dQ: a group of 4 SIMD-groups takes 32 rows of a KV head (8 each) and walks its keys in blocks of 32 ----

constant constexpr uint QR = 32, QK = 32, THREADS = 128;

template <uint D>
static void bwdQ(device const half *q, device const half *k, device const half *v, device const float *dO,
                 device const float *lse, device const float *dD, device float *dQ, constant AttnBwdParams &p,
                 uint3 group, uint tid, uint sg, uint lane, threadgroup half *sK, threadgroup half *sV,
                 threadgroup float *sS, threadgroup float *sP, threadgroup half *sDs) {
    const uint kvHead = group.y, hpk = p.heads / p.kvHeads, rowsKv = hpk * p.queries, r0 = group.x * QR;
    const uint offset = p.keys - p.queries, base = kvHead * rowsKv;
    device const half *kHead = k + ulong(kvHead) * p.cacheKeys * D, *vHead = v + ulong(kvHead) * p.cacheKeys * D;

    // the SIMD-group's Q and dO (to FP16) as matrices, through zero-filled copies in the K and V blocks' memory
    for (uint e = tid; e < QR * D; e += THREADS) {
        uint rr = r0 + e / D, c = e % D;
        bool ok = rr < rowsKv;
        sK[e] = ok ? q[ulong(base + rr) * D + c] : half(0.0h);
        sV[e] = ok ? half(dO[ulong(base + rr) * D + c]) : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qm[D / 8], om[D / 8];
    for (uint d = 0; d < D / 8; d++) {
        simdgroup_load(qm[d], sK + sg * 8 * D + d * 8, D);
        simdgroup_load(om[d], sV + sg * 8 * D + d * 8, D);
    }
    simdgroup_float8x8 acc[D / 8];
    for (uint d = 0; d < D / 8; d++) acc[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // the elementwise lanes: row lr of the SIMD-group's 8, columns c4 * 8 .. + 7 of a block
    const uint lr = lane / 4, c4 = lane % 4, rr = r0 + sg * 8 + lr;
    const bool live = rr < rowsKv;
    const uint qi = live ? rr % p.queries : 0;
    const uint limit = !live ? 0 : p.causal != 0 ? qi + offset + 1 : p.keys;
    const float L = live ? lse[base + rr] : 0.0f, Dr = live ? dD[base + rr] : 0.0f;
    uint kEnd = p.keys;
    if (p.causal != 0) {
        uint rLast = min(r0 + QR, rowsKv) - 1;
        uint iMax = (r0 / p.queries != rLast / p.queries) ? p.queries - 1 : rLast % p.queries;
        kEnd = min(kEnd, iMax + offset + 1);
    }
    threadgroup float *tS = sS + sg * 8 * QK, *tP = sP + sg * 8 * QK;
    threadgroup half *tDs = sDs + sg * 8 * QK;

    for (uint b = 0; b < kEnd; b += QK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);   // Q's and dO's copies, or the previous block, are no longer read
        {
            threadgroup uint4 *dk = (threadgroup uint4 *) sK, *dv = (threadgroup uint4 *) sV;
            device const uint4 *ks = (device const uint4 *) (kHead + ulong(b) * D), *vs = (device const uint4 *) (vHead + ulong(b) * D);
            for (uint e = tid; e < QK * D / 8; e += THREADS) {
                bool ok = b + e / (D / 8) < p.keys;
                dk[e] = ok ? ks[e] : uint4(0);
                dv[e] = ok ? vs[e] : uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T and dP = dO V^T for the SIMD-group's rows
        for (uint kb = 0; kb < QK / 8; kb++) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), dp = s;
            for (uint d = 0; d < D / 8; d++) {
                simdgroup_half8x8 kt, vt;
                simdgroup_load(kt, sK + kb * 8 * D + d * 8, D, ulong2(0, 0), true);
                simdgroup_load(vt, sV + kb * 8 * D + d * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qm[d], kt, s);
                simdgroup_multiply_accumulate(dp, om[d], vt, dp);
            }
            simdgroup_store(s, tS + kb * 8, QK);
            simdgroup_store(dp, tP + kb * 8, QK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < 8; c++) {
            uint col = c4 * 8 + c, key = b + col;
            float pv = key < limit ? exp(tS[lr * QK + col] * p.scale - L) : 0.0f;
            tDs[lr * QK + col] = half(pv * (tP[lr * QK + col] - Dr));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // dQ += dS K
        simdgroup_half8x8 dsm[QK / 8];
        for (uint kb = 0; kb < QK / 8; kb++) simdgroup_load(dsm[kb], tDs + kb * 8, QK);
        for (uint d = 0; d < D / 8; d++) {
            for (uint kb = 0; kb < QK / 8; kb++) {
                simdgroup_half8x8 km;
                simdgroup_load(km, sK + kb * 8 * D + d * 8, D);
                simdgroup_multiply_accumulate(acc[d], dsm[kb], km, acc[d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // dQ (times scale) through the SIMD-group's 8 x 8 scratch
    threadgroup float *tile = tS;
    for (uint d = 0; d < D / 8; d++) {
        simdgroup_store(acc[d], tile, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
            uint row = r0 + sg * 8 + e / 8, c = d * 8 + e % 8;
            if (row < rowsKv) dQ[ulong(base + row) * D + c] = tile[e] * p.scale;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define BWD_Q_ENTRY(NAME, D)                                                                                       \
kernel void NAME(device const half *q [[buffer(0)]], device const half *k [[buffer(1)]],                             \
                 device const half *v [[buffer(2)]], device const float *dO [[buffer(3)]],                           \
                 device const float *lse [[buffer(4)]], device const float *dD [[buffer(5)]],                        \
                 device float *dQ [[buffer(6)]], constant AttnBwdParams &p [[buffer(7)]],                            \
                 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],            \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {             \
    threadgroup half sK[QK * D];                                                                                  \
    threadgroup half sV[QK * D];                                                                                  \
    threadgroup float sS[4 * 8 * QK];                                                                             \
    threadgroup float sP[4 * 8 * QK];                                                                             \
    threadgroup half sDs[4 * 8 * QK];                                                                             \
    bwdQ<D>(q, k, v, dO, lse, dD, dQ, p, group, tid, sg, lane, sK, sV, sS, sP, sDs);                             \
}

BWD_Q_ENTRY(attn_bwd_q_64_simdgroup, 64)
BWD_Q_ENTRY(attn_bwd_q_128_simdgroup, 128)

// ---- dK, dV: a group of 4 SIMD-groups takes 32 keys of a KV head (8 each) and walks its rows in tiles of 16 ----

constant constexpr uint KK = 32, KR = 16;

template <uint D>
static void bwdKV(device const half *q, device const half *k, device const half *v, device const float *dO,
                  device const float *lse, device const float *dD, device float *dK, device float *dV,
                  constant AttnBwdParams &p, uint3 group, uint tid, uint sg, uint lane, threadgroup half *sK,
                  threadgroup half *sV, threadgroup half *sQ, threadgroup half *sO, threadgroup float *sL,
                  threadgroup float *sD, threadgroup float *sS, threadgroup float *sP, threadgroup half *sPh,
                  threadgroup half *sDs) {
    const uint kvHead = group.y, hpk = p.heads / p.kvHeads, rowsKv = hpk * p.queries, k0 = group.x * KK;
    const uint offset = p.keys - p.queries, base = kvHead * rowsKv;
    device const half *kHead = k + ulong(kvHead) * p.cacheKeys * D, *vHead = v + ulong(kvHead) * p.cacheKeys * D;
    {
        threadgroup uint4 *dk = (threadgroup uint4 *) sK, *dv = (threadgroup uint4 *) sV;
        device const uint4 *ks = (device const uint4 *) (kHead + ulong(k0) * D), *vs = (device const uint4 *) (vHead + ulong(k0) * D);
        for (uint e = tid; e < KK * D / 8; e += THREADS) {
            bool ok = k0 + e / (D / 8) < p.keys;
            dk[e] = ok ? ks[e] : uint4(0);
            dv[e] = ok ? vs[e] : uint4(0);
        }
    }
    simdgroup_float8x8 accK[D / 8], accV[D / 8];
    for (uint d = 0; d < D / 8; d++) {
        accK[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        accV[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    // the elementwise lanes: key lk of the SIMD-group's 8, rows c4 * 4 .. + 3 of a tile
    const uint lk = lane / 4, c4 = lane % 4, key = k0 + sg * 8 + lk;
    threadgroup float *tS = sS + sg * 8 * KR, *tP = sP + sg * 8 * KR;
    threadgroup half *tPh = sPh + sg * 8 * KR, *tDs = sDs + sg * 8 * KR;
    threadgroup const half *kSg = sK + sg * 8 * D, *vSg = sV + sg * 8 * D;

    for (uint r0 = 0; r0 < rowsKv; r0 += KR) {
        if (p.causal != 0) {
            uint rLast = min(r0 + KR, rowsKv) - 1;
            uint iMax = (r0 / p.queries != rLast / p.queries) ? p.queries - 1 : rLast % p.queries;
            if (iMax + offset < k0) continue;   // uniform across the group: no row of the tile sees these keys
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous tile is no longer read (and K, V are staged)
        for (uint e = tid; e < KR * D; e += THREADS) {
            uint rr = r0 + e / D, c = e % D;
            bool ok = rr < rowsKv;
            sQ[e] = ok ? q[ulong(base + rr) * D + c] : half(0.0h);
            sO[e] = ok ? half(dO[ulong(base + rr) * D + c]) : half(0.0h);
        }
        if (tid < KR) { uint rr = r0 + tid; sL[tid] = rr < rowsKv ? lse[base + rr] : 0.0f; sD[tid] = rr < rowsKv ? dD[base + rr] : 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S^T = K Q^T and dP^T = V dO^T for the SIMD-group's keys
        for (uint rb = 0; rb < KR / 8; rb++) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), dp = s;
            for (uint d = 0; d < D / 8; d++) {
                simdgroup_half8x8 km, vm, qt, ot;
                simdgroup_load(km, kSg + d * 8, D);
                simdgroup_load(vm, vSg + d * 8, D);
                simdgroup_load(qt, sQ + rb * 8 * D + d * 8, D, ulong2(0, 0), true);
                simdgroup_load(ot, sO + rb * 8 * D + d * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, km, qt, s);
                simdgroup_multiply_accumulate(dp, vm, ot, dp);
            }
            simdgroup_store(s, tS + rb * 8, KR);
            simdgroup_store(dp, tP + rb * 8, KR);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < KR / 4; c++) {
            uint ri = c4 * (KR / 4) + c, rr = r0 + ri;
            bool visible = rr < rowsKv && key < p.keys && (p.causal == 0 || key <= rr % p.queries + offset);
            float pv = visible ? exp(tS[lk * KR + ri] * p.scale - sL[ri]) : 0.0f;
            tPh[lk * KR + ri] = half(pv);
            tDs[lk * KR + ri] = half(pv * (tP[lk * KR + ri] - sD[ri]));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // dV += P^T dO, dK += dS^T Q
        simdgroup_half8x8 pm[KR / 8], dsm[KR / 8];
        for (uint rb = 0; rb < KR / 8; rb++) { simdgroup_load(pm[rb], tPh + rb * 8, KR); simdgroup_load(dsm[rb], tDs + rb * 8, KR); }
        for (uint d = 0; d < D / 8; d++) {
            for (uint rb = 0; rb < KR / 8; rb++) {
                simdgroup_half8x8 om, qm;
                simdgroup_load(om, sO + rb * 8 * D + d * 8, D);
                simdgroup_load(qm, sQ + rb * 8 * D + d * 8, D);
                simdgroup_multiply_accumulate(accV[d], pm[rb], om, accV[d]);
                simdgroup_multiply_accumulate(accK[d], dsm[rb], qm, accK[d]);
            }
        }
    }
    // dK (times scale) and dV through the SIMD-group's scratch
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tile = tS;
    for (uint d = 0; d < D / 8; d++) {
        for (uint w = 0; w < 2; w++) {
            simdgroup_store(w == 0 ? accK[d] : accV[d], tile, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                uint kk = k0 + sg * 8 + e / 8, c = d * 8 + e % 8;
                if (kk < p.keys) {
                    ulong at = (ulong(kvHead) * p.cacheKeys + kk) * D + c;
                    if (w == 0) dK[at] = tile[e] * p.scale; else dV[at] = tile[e];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define BWD_KV_ENTRY(NAME, D)                                                                                      \
kernel void NAME(device const half *q [[buffer(0)]], device const half *k [[buffer(1)]],                             \
                 device const half *v [[buffer(2)]], device const float *dO [[buffer(3)]],                           \
                 device const float *lse [[buffer(4)]], device const float *dD [[buffer(5)]],                        \
                 device float *dK [[buffer(6)]], device float *dV [[buffer(7)]], constant AttnBwdParams &p [[buffer(8)]], \
                 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],            \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {             \
    threadgroup half sK[KK * D];                                                                                  \
    threadgroup half sV[KK * D];                                                                                  \
    threadgroup half sQ[KR * D];                                                                                  \
    threadgroup half sO[KR * D];                                                                                  \
    threadgroup float sL[KR];                                                                                     \
    threadgroup float sD[KR];                                                                                     \
    threadgroup float sS[4 * 8 * KR];                                                                             \
    threadgroup float sP[4 * 8 * KR];                                                                             \
    threadgroup half sPh[4 * 8 * KR];                                                                             \
    threadgroup half sDs[4 * 8 * KR];                                                                             \
    bwdKV<D>(q, k, v, dO, lse, dD, dK, dV, p, group, tid, sg, lane, sK, sV, sQ, sO, sL, sD, sS, sP, sPh, sDs);   \
}

BWD_KV_ENTRY(attn_bwd_kv_64_simdgroup, 64)
BWD_KV_ENTRY(attn_bwd_kv_128_simdgroup, 128)
