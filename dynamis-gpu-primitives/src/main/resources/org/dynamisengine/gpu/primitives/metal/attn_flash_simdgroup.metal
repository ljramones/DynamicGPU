// AttentionF16's fused attention on Apple GPU matrix hardware (simdgroup_matrix, Apple7 and later; 1.11.0).
// Hand-written: Slang cannot emit Metal SIMD-group matrices. Same ABI as attn_flash_body.slang (attn_common.slang):
// q, k, v, o, partials, page table, lse, lengths at buffers 0 to 7, parameters at 8; groups (row blocks,
// KV heads, key ranges). Paged K and V (pageSize a multiple of BC) give each block's base from the table. Entries for head dimensions 64 and 128, FP16 or FP32 O.
//
// A group of 128 threads (4 SIMD-groups) takes BR = 32 rows of one KV head (its query heads' queries, which are
// consecutive rows of Q and O); each SIMD-group owns 8 rows. Per block of BC = 32 keys, staged in threadgroup memory
// (16-byte copies, zero past the range):
//   - S = Q K^T: the SIMD-group's 8 x headDim of Q (held as matrices) times K^T blocks, FP32 accumulation of exact FP16
//     products, to threadgroup memory;
//   - the online softmax, 4 lanes per row: s = S scale (masked keys -infinity), the block max m', the rescale
//     alpha = exp(m - m'), P = exp(s - m') rounded to FP16 for the product, l = l alpha + sum P (from the FP32 P);
//   - O = diag(alpha) O + P V, as matrix products (a diagonal matrix rescales the accumulators).
// Keys past the range (a cache's unwritten tail) are never read. With the causal mask, blocks no row of the group sees
// are skipped. With splits > 1 the group writes its (m, l, unnormalized O) for attn_combine; otherwise O = acc / l,
// tile by tile through an 8 x 8 scratch. Measured (M4 Max, 32 heads x 128, causal, FP16, at 512 / 1024 / 2048): this
// form 0.42 / 1.42 / 5.28 ms. K and V loaded from device memory took 0.85 / 3.03 / 11.4 (each matrix serves one
// product, and all four SIMD-groups load it), staged with per-half copies and O through a whole-group buffer 0.61 /
// 2.21 / 8.64, and 16 rows per SIMD-group (ROW_TILES 2: each K and V matrix serves two products, but twice the
// matrices per lane) 0.98 / 3.15 / 11.4.
// Testing the mask once for a fully visible block took 0.411 / 1.376 / 5.118 ms vs 0.426 / 1.440 / 5.291
// with a per-key mask (finite FP16 data). The score, sum and product order are identical.
// Other experiments in the same shapes (ms, release baseline 0.424 / 1.429 / 5.317): register fragments
// 0.449 / 1.473 / 5.357, those fragments with double-buffered K/V 0.484 / 1.651 / 6.133, with register
// prefetch 0.454 / 1.469 / 5.363, with key blocks of 64 0.489 / 1.673 / 6.370, or of 16 0.552 / 1.860 / 6.910.
// Tile-major K/V staging took 0.424 / 1.433 / 5.321. Pre-scaling half Q for exp2 took 0.428 / 1.431 / 5.272
// and rounds Q again. At dimension 256 the matrix kernel took 2.605 / 9.828 / 38.548 against the portable
// kernel's 1.439 / 5.471 / 21.115, using 16-key blocks on both.
#include <metal_stdlib>
using namespace metal;

struct AttnParams {
    uint heads; uint kvHeads; uint queries; uint keys; uint headDim; uint cacheKeys; float scale; uint causal;
    uint splits; uint splitKeys; uint rowBlocks; uint pageSize; uint batch;   // batch: 1.11.1, 0 here
    uint writeLse;   // 1.12.0: write each row's log-sum-exp (unsplit; the combine writes split rows')
    uint lengthGroup;
};

#ifndef ROW_TILES
#define ROW_TILES 1
#endif
// RT row tiles of 8 per SIMD-group; LPR lanes per row in the softmax, each over CPL columns of the block
constant constexpr uint RT = ROW_TILES, BC = 32, THREADS = 128, BR = 32 * RT, LPR = 4 / RT, CPL = BC / LPR;

template <uint D, typename OutT>
static void flash(device const half *q, device const half *k, device const half *v, device OutT *o,
                  device float *partials, device const uint *table, device float *lse,
                  device const uint *lens, constant AttnParams &p, uint3 group,
                  uint tid, uint sg, uint lane, threadgroup float *sS, threadgroup half *sP, threadgroup float *sDiag,
                  threadgroup float *sM, threadgroup float *sL, threadgroup half *sK, threadgroup half *sV) {
    const uint kvHead = group.y, split = group.z;
    const uint hpk = p.heads / p.kvHeads, rowsKv = hpk * p.queries;
    const uint r0 = group.x * BR;
    const uint offset = p.keys - p.queries;
    device const half *qBase = q + (ulong(kvHead) * rowsKv + r0) * D;   // the rows of a KV head are consecutive
    // contiguous: the KV head's keys; paged: the pool, each block's page from the table
    const ulong headAt = p.pageSize == 0 ? ulong(kvHead) * p.cacheKeys * D : 0;
    device const half *kHead = k + headAt, *vHead = v + headAt;

    // the keys this group covers
    uint kStart = split * p.splitKeys, kEnd = min(p.keys, kStart + p.splitKeys);
    uint rLast = min(r0 + BR, rowsKv) - 1;
    if (p.lengthGroup != 0) {
        uint first = (kvHead * hpk + r0 / p.queries) / p.lengthGroup;
        uint last = (kvHead * hpk + rLast / p.queries) / p.lengthGroup;
        uint longest = lens[first];
        for (uint i = first + 1; i <= last; i++) longest = max(longest, lens[i]);
        kEnd = min(kEnd, longest);
    }
    if (p.causal != 0) {
        uint iMax = (r0 / p.queries != rLast / p.queries) ? p.queries - 1 : rLast % p.queries;
        kEnd = min(kEnd, iMax + offset + 1);
    }

    // Q of the SIMD-group's rows, through a zero-filled copy (BR x D halves fit the K and V copies)
    {
        threadgroup uint4 *dst = (threadgroup uint4 *) sK;
        device const uint4 *src = (device const uint4 *) qBase;
        for (uint e = tid; e < BR * D / 8; e += THREADS) dst[e] = r0 + e / (D / 8) < rowsKv ? src[e] : uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qm[RT][D / 8];
    for (uint t = 0; t < RT; t++)
        for (uint d = 0; d < D / 8; d++) simdgroup_load(qm[t][d], sK + (sg * RT + t) * 8 * D + d * 8, D);

    simdgroup_float8x8 acc[RT][D / 8];
    for (uint t = 0; t < RT; t++)
        for (uint d = 0; d < D / 8; d++) acc[t][d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    // the softmax lanes: row lr of the SIMD-group's rows, columns h16 * CPL .. + CPL - 1 of the block
    const uint lr = lane / LPR, h16 = lane % LPR, row = sg * 8 * RT + lr, r = r0 + row;
    const uint qi = r < rowsKv ? r % p.queries : 0;
    uint limit = r >= rowsKv ? 0 : p.causal != 0 ? qi + offset + 1 : p.keys;
    if (r < rowsKv && p.lengthGroup != 0) limit = min(limit, lens[(kvHead * hpk + r / p.queries) / p.lengthGroup]);
    float m = -INFINITY, l = 0.0f;

    for (uint b = kStart; b < kEnd; b += BC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous block's copies (or Q's) are no longer read
        {
            threadgroup uint4 *dk = (threadgroup uint4 *) sK, *dv = (threadgroup uint4 *) sV;
            // the block's keys are consecutive: contiguous, or within one page
            ulong at = p.pageSize == 0 ? ulong(b) * D
                     : ((ulong(table[b / p.pageSize]) * p.kvHeads + kvHead) * p.pageSize + b % p.pageSize) * D;
            device const uint4 *ks = (device const uint4 *) (kHead + at), *vs = (device const uint4 *) (vHead + at);
            for (uint e = tid; e < BC * D / 8; e += THREADS) {
                bool ok = b + e / (D / 8) < kEnd;
                dk[e] = ok ? ks[e] : uint4(0);
                dv[e] = ok ? vs[e] : uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S for the SIMD-group's rows: each K^T matrix serves both row tiles
        for (uint kb = 0; kb < BC / 8; kb++) {
            simdgroup_float8x8 s[RT];
            for (uint t = 0; t < RT; t++) s[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < D / 8; d++) {
                simdgroup_half8x8 kt;
                simdgroup_load(kt, sK + kb * 8 * D + d * 8, D, ulong2(0, 0), true);
                for (uint t = 0; t < RT; t++) simdgroup_multiply_accumulate(s[t], qm[t][d], kt, s[t]);
            }
            for (uint t = 0; t < RT; t++) simdgroup_store(s[t], sS + (sg * RT + t) * 8 * BC + kb * 8, BC);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // the online softmax of the SIMD-group's rows
        float sv[CPL];
        float mb = -INFINITY;
        // A full block inside this row's visible prefix has no masked keys.
        if (b + BC <= min(limit, kEnd)) {
            for (uint c = 0; c < CPL; c++) {
                sv[c] = sS[row * BC + h16 * CPL + c] * p.scale;
                mb = max(mb, sv[c]);
            }
        } else {
            for (uint c = 0; c < CPL; c++) {
                uint col = h16 * CPL + c, key = b + col;
                float x = sS[row * BC + col] * p.scale;
                sv[c] = key < kEnd && key < limit ? x : -INFINITY;
                mb = max(mb, sv[c]);
            }
        }
        for (uint x = 1; x < LPR; x *= 2) mb = max(mb, simd_shuffle_xor(mb, x));
        float mNew = max(m, mb);
        float alpha = m == -INFINITY ? 0.0f : exp(m - mNew);
        float sum = 0.0f;
        for (uint c = 0; c < CPL; c++) {
            float pj = sv[c] == -INFINITY ? 0.0f : exp(sv[c] - mNew);
            sP[row * BC + h16 * CPL + c] = half(pj);
            sum += pj;
        }
        for (uint x = 1; x < LPR; x *= 2) sum += simd_shuffle_xor(sum, x);
        m = mNew;
        l = l * alpha + sum;
        // the rescale as diagonal matrices: row lr's alpha on its tile's diagonal
        if (h16 == 0) {
            uint t = lr / 8, rr = lr % 8;
            for (uint c = 0; c < 8; c++) sDiag[(sg * RT + t) * 64 + rr * 8 + c] = c == rr ? alpha : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dg[RT];
        simdgroup_half8x8 pm[RT][BC / 8];
        for (uint t = 0; t < RT; t++) {
            simdgroup_load(dg[t], sDiag + (sg * RT + t) * 64, 8);
            for (uint kb = 0; kb < BC / 8; kb++) simdgroup_load(pm[t][kb], sP + (sg * RT + t) * 8 * BC + kb * 8, BC);
        }
        for (uint d = 0; d < D / 8; d++) {
            for (uint t = 0; t < RT; t++) simdgroup_multiply(acc[t][d], dg[t], acc[t][d]);
            for (uint kb = 0; kb < BC / 8; kb++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, sV + kb * 8 * D + d * 8, D);
                for (uint t = 0; t < RT; t++) simdgroup_multiply_accumulate(acc[t][d], pm[t][kb], vm, acc[t][d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);   // sS, sP and sDiag are rewritten by the next block
    }

    // O tile by tile through the SIMD-group's 8 x 8 scratch (its diagonal space)
    if (h16 == 0) { sM[row] = m; sL[row] = l; }
    if (h16 == 0 && p.writeLse != 0 && p.splits <= 1 && r < rowsKv)
        lse[kvHead * rowsKv + r] = l == 0.0f ? -INFINITY : m + log(l);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tile = sDiag + sg * RT * 64;
    for (uint t = 0; t < RT; t++) {
        for (uint d = 0; d < D / 8; d++) {
            simdgroup_store(acc[t][d], tile, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                uint er = (sg * RT + t) * 8 + e / 8, c = d * 8 + e % 8, rr = r0 + er;
                if (rr < rowsKv) {
                    uint out = kvHead * rowsKv + rr;   // = head * queries + i
                    if (p.splits > 1) {
                        ulong rows = ulong(p.heads) * p.queries, pr = split * rows + out;
                        if (c == 0) { partials[2 * pr] = sM[er]; partials[2 * pr + 1] = sL[er]; }
                        partials[2 * p.splits * rows + pr * D + c] = sL[er] == 0.0f ? 0.0f : tile[e];
                    } else {
                        o[ulong(out) * D + c] = OutT(sL[er] == 0.0f ? 0.0f : tile[e] / sL[er]);
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define FLASH_ENTRY(NAME, D, OutT)                                                                                \
kernel void NAME(device const half *q [[buffer(0)]], device const half *k [[buffer(1)]],                             \
                 device const half *v [[buffer(2)]], device OutT *o [[buffer(3)]],                                   \
                 device float *partials [[buffer(4)]], device const uint *table [[buffer(5)]],                       \
                 device float *lse [[buffer(6)]], device const uint *lens [[buffer(7)]],                             \
                 constant AttnParams &p [[buffer(8)]],                                                               \
                 uint3 group [[threadgroup_position_in_grid]],                                                       \
                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],              \
                 uint lane [[thread_index_in_simdgroup]]) {                                                         \
    threadgroup float sS[BR * BC];                                                                               \
    threadgroup half sP[BR * BC];                                                                                \
    threadgroup float sDiag[4 * RT * 64];                                                                        \
    threadgroup float sM[BR];                                                                                    \
    threadgroup float sL[BR];                                                                                    \
    threadgroup uint4 sKV[2 * BC * D / 8];   /* K and V copies; Q's copy first (BR x D halves) */               \
    threadgroup half *sK = (threadgroup half *) sKV, *sV = sK + BC * D;                                          \
    flash<D, OutT>(q, k, v, o, partials, table, lse, lens, p, group, tid, sg, lane, sS, sP, sDiag, sM, sL, sK, sV); \
}

FLASH_ENTRY(attn_flash_64_simdgroup, 64, half)
FLASH_ENTRY(attn_flash_64_f32_simdgroup, 64, float)
FLASH_ENTRY(attn_flash_128_simdgroup, 128, half)
FLASH_ENTRY(attn_flash_128_f32_simdgroup, 128, float)
