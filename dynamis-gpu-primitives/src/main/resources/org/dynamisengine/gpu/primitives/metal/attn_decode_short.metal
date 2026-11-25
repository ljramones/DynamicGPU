// D128, at most 512 keys. Relative to split decode, dot products, FP32 weight sums and P V use a
// different reduction order within the scalar attention bound. No intermediate rounds to half.
// M4 Max Metal at 32 / 128 / 257 / 512 keys, 32 query heads sharing 8 KV heads, FP16 O:
// 1.11.0 took 39.5 / 54.5 / 61.4 / 64.8 us, and this single dispatch took 5.2 / 6.6 / 9.7 / 14.0 us.
// A padded 8-row matrix group took 87.0 / 123.9 / 759.9 us at 512 / 4096 / 32768; its paired release
// baseline was 98.0 / 100.7 / 412.4 us. The matrix path also rounds P to half for P V.
#include <metal_stdlib>
using namespace metal;

struct AttnParams {
    uint heads; uint kvHeads; uint queries; uint keys; uint headDim; uint cacheKeys; float scale; uint causal;
    uint splits; uint splitKeys; uint rowBlocks; uint pageSize; uint batch;   // batch: 1.11.1, 0 here
    uint writeLse;   // 1.12.0: unused here
    uint lengthGroup;
};

// Short decode: keep all keys in the group and write the final output in one dispatch.
// 8 lanes cooperate on each 128-element Q K dot product. After the softmax, each SIMD-group
// accumulates 128 columns (four per lane) across one sixteenth of V.
template<typename OutT>
static void shortDecode(device const packed_half4 *q, device const packed_half4 *k, device const packed_half4 *v, device OutT *o,
                        device const uint *table, device const uint *lens, constant AttnParams &p, ulong row, uint t, uint sg, uint lane,
                        threadgroup float *ss, threadgroup float *red, threadgroup float4 *parts) {
    uint kvHead = row / (p.queries * (p.heads / p.kvHeads));
    uint limit = p.causal ? p.keys - p.queries + row % p.queries + 1 : p.keys;
    if (p.lengthGroup != 0) limit = min(limit, lens[(row / p.queries) / p.lengthGroup]);
    uint sublane = lane % 8;
    float4 query[4];
    for (uint u = 0; u < 4; u++) query[u] = float4(q[ulong(row) * 32 + u * 8 + sublane]);
    for (uint first = 0; first < limit; first += 64) {
        uint j = first + t / 8;
        float dot = 0;
        if (j < limit) {
            ulong at = p.pageSize == 0 ? ulong(kvHead) * p.cacheKeys + j :
                (ulong(table[j / p.pageSize]) * p.kvHeads + kvHead) * p.pageSize + j % p.pageSize;
            for (uint u = 0; u < 4; u++) {
                float4 x = float4(k[at * 32 + u * 8 + sublane]);
                for (uint c = 0; c < 4; c++) dot = fma(query[u][c], x[c], dot);
            }
        }
        for (uint step = 4; step != 0; step /= 2) dot += simd_shuffle_xor(dot, step);
        if (sublane == 0 && j < limit) ss[j] = dot * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float score = t < limit ? ss[t] : -INFINITY;
    float sm = simd_max(score);
    if (lane == 0) red[sg] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = simd_max(lane < 16 ? red[lane] : -INFINITY);
    // Sums use separate slots so every SIMD-group can finish reading its maximum.
    float w = t < limit && mx != -INFINITY ? exp(score - mx) : 0;
    ss[t] = w;
    float sl = simd_sum(w);
    if (lane == 0) red[16 + sg] = sl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = simd_sum(lane < 16 ? red[16 + lane] : 0.0f);
    // SIMD-group sg takes keys sg + 16 * i. Each of its lanes owns four adjacent columns.
    float4 acc(0.0f);
    for (uint j = sg; j < limit; j += 16) {
        ulong at = p.pageSize == 0 ? ulong(kvHead) * p.cacheKeys + j :
            (ulong(table[j / p.pageSize]) * p.kvHeads + kvHead) * p.pageSize + j % p.pageSize;
        acc = fma(float4(ss[j]), float4(v[at * 32 + lane]), acc);
    }
    parts[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float4 out(0.0f);
        for (uint part = 0; part < 16; part++) out += parts[part * 32 + lane];
        for (uint c = 0; c < 4; c++) o[ulong(row) * 128 + lane * 4 + c] = OutT(sum == 0.0f ? 0.0f : out[c] / sum);
    }
}

#define SHORT_ENTRY(NAME, OutT) \
kernel void NAME(device const packed_half4 *q [[buffer(0)]], device const packed_half4 *k [[buffer(1)]], \
                 device const packed_half4 *v [[buffer(2)]], device OutT *o [[buffer(3)]], \
                 device const uint *table [[buffer(4)]], device const uint *lens [[buffer(5)]], constant AttnParams &p [[buffer(6)]], \
                 uint3 group [[threadgroup_position_in_grid]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float ss[512], red[32]; threadgroup float4 parts[512]; \
    ulong row = ulong(group.y) * p.rowBlocks + group.x; \
    if (row < ulong(p.heads) * p.queries) shortDecode(q, k, v, o, table, lens, p, row, sg * 32 + lane, sg, lane, ss, red, parts); \
}

SHORT_ENTRY(attn_decode_128_direct, half)
SHORT_ENTRY(attn_decode_128_f32_direct, float)
