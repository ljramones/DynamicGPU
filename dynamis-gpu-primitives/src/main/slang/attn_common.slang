// Fused attention (1.11.0), shared by attn_flash_body.slang and attn_combine_body.slang.
#pragma once
#include "dynamis/kit.slang"
#include "quant_common.slang"

struct AttnParams
{
    uint heads;       // query heads
    uint kvHeads;     // key and value heads; each serves heads / kvHeads query heads (grouped-query attention)
    uint queries;
    uint keys;        // keys attended
    uint headDim;     // even, at most DMAX
    uint cacheKeys;   // K and V are [kvHeads][cacheKeys][headDim]: keys beyond `keys` are not read
    float scale;
    uint causal;      // query i sees keys 0 .. i + keys - queries
    uint splits;      // key ranges per row, combined by attn_combine (1 writes O directly)
    uint splitKeys;   // keys per range, a multiple of the key block
    uint rowBlocks;   // groups along the rows of a KV head
    uint pageSize;    // 0: K and V are contiguous; else paged (1.11.0): key j is slot j % pageSize of page
                      // table[j / pageSize], and a page holds [kvHeads][pageSize][headDim]
};

// The word (pair of halves) index of pair pr of key `key` of KV head kvHead, contiguous or through the page table.
#define ATTN_KV_WORD(p, table, kvHead, key, pr, d2)                                                              \
    ((p).pageSize == 0 ? ((kvHead) * (p).cacheKeys + (key)) * (d2) + (pr)                                         \
                       : ((table[(key) / (p).pageSize] * (p).kvHeads + (kvHead)) * (p).pageSize + (key) % (p).pageSize) * (d2) + (pr))

float attnLo(uint w) { return f16tof32(w & 0xFFFFu); }
float attnHi(uint w) { return f16tof32(w >> 16); }
uint attnPack(float lo, float hi) { return qF32ToF16(lo) | (qF32ToF16(hi) << 16); }
