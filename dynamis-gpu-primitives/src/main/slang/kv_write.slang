// PagedKvCache.write (1.11.0): new keys and values of one layer into the cache's pages. Thread per word (pair of
// halves) of the new tokens: token t of KV head h goes to slot (position + t) % pageSize of page
// table[(position + t) / pageSize], whose KV head h holds [pageSize][headDim]. The sources' strides (in halves, even)
// take head-major ([kvHeads][tokens][headDim]) and token-major ([tokens][kvHeads][headDim]) layouts alike.
#include "dynamis/kit.slang"

struct KvWriteParams
{
    uint kvHeads;
    uint headDim;
    uint tokens;
    uint position;      // the first new token's position in the sequence
    uint pageSize;
    uint headStride;    // halves between KV heads of the sources
    uint tokenStride;   // halves between tokens of the sources
    uint gridX;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> kSrc;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> vSrc;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> kPool;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> vPool;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> table;
[[vk::push_constant]] ConstantBuffer<KvWriteParams> p;

[shader("compute")]
[numthreads(256, 1, 1)]
void kv_write(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint d2 = p.headDim / 2;
    uint id = dg_linear_group(group, p.gridX) * 256 + local.x;
    if (id >= p.tokens * p.kvHeads * d2) return;
    uint pr = id % d2, h = (id / d2) % p.kvHeads, t = id / (d2 * p.kvHeads);
    uint src = (h * p.headStride + t * p.tokenStride) / 2 + pr;
    uint pos = p.position + t;
    uint dst = ((table[pos / p.pageSize] * p.kvHeads + h) * p.pageSize + pos % p.pageSize) * d2 + pr;
    kPool[dst] = kSrc[src];
    vPool[dst] = vSrc[src];
}
