// AttentionF16.backward's main kernels (1.12.0); the entry file defines DMAX, BQ (query rows per tile), ENTRY, and KV
// (attn_bwd_kv: dK and dV) or not (attn_bwd_q: dQ). A group of 128 threads.
//   KV: owns BK = 16 keys of one KV head and walks its rows in tiles of BQ; per tile, each thread computes two
//       (row, key) pairs of S and dP (dot products in FP32) to group memory, P and dS from them, then the thread that
//       owns key kk and columns part, part + 8, .. adds P dO into dV and dS Q into dK, row by row in order.
//   Q:  owns BQ rows and walks the keys in tiles of BK; per tile the same pairs, then the thread that owns row r and
//       columns part, part + 8, .. adds dS K into dQ, key by key in order.
// With the causal mask, tiles no pair of which is visible are skipped. Every sum is in a fixed order, so a device's
// result does not depend on scheduling; exp is the device's, so the result is bounded.
#include "attn_bwd_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> q;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> k;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> v;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dO;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> lse;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> dD;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> out0;   // KV: dK; Q: dQ
#ifdef KV
[[vk::binding(7, 0)]] RWStructuredBuffer<float> out1;   // dV
#endif
[[vk::push_constant]] ConstantBuffer<AttnBwdParams> p;

static const uint THREADS = 128;
static const uint BK = 16;
static const uint D2 = DMAX / 2;
static const uint STRIDE = D2 + 1;
static const uint CPT = DMAX / 8;   // columns per thread: part, part + 8, ..
static const uint PPT = BQ * BK / THREADS;   // (row, key) pairs per thread

groupshared uint qs[BQ * STRIDE];
groupshared uint ks[BK * STRIDE];
groupshared uint vs[BK * STRIDE];
groupshared float dos[BQ * (DMAX + 1)];
groupshared float ls[BQ];
groupshared float ds[BQ];
groupshared float ps[BQ * BK];
groupshared float dss[BQ * BK];

float half2At(uint w, uint c) { return (c & 1) == 0 ? attnLo(w) : attnHi(w); }

// S, P, dP and dS of the tile's pairs (rows r0 .., keys k0 ..) into ps and dss; rows past `rows`, keys past `keys`
// and masked pairs give 0
void pairs(uint t, uint r0, uint rows, uint k0, uint d2)
{
    uint offset = p.keys - p.queries;
    // the thread's pairs side by side: their dot products as independent chains (one pair after the other left each
    // thread waiting on its own fma chain)
    float sv[PPT], dp[PPT];
    uint ii[PPT], kv[PPT];
    [ForceUnroll]
    for (uint u = 0; u < PPT; u++) { uint e = t + u * THREADS; ii[u] = e / BK; kv[u] = e % BK; sv[u] = 0.0f; dp[u] = 0.0f; }
    for (uint p0 = 0; p0 < D2; p0 += 8)
    {
        [ForceUnroll]
        for (uint pi = 0; pi < 8; pi++)
        {
            uint pr = p0 + pi;
            if (pr < d2)
            {
                [ForceUnroll]
                for (uint u = 0; u < PPT; u++)
                {
                    uint qw = qs[ii[u] * STRIDE + pr], kw = ks[kv[u] * STRIDE + pr], vw = vs[kv[u] * STRIDE + pr];
                    sv[u] = fma(attnLo(qw), attnLo(kw), sv[u]);   // FP16 products are exact
                    sv[u] = fma(attnHi(qw), attnHi(kw), sv[u]);
                    dp[u] = dp[u] + dos[ii[u] * (DMAX + 1) + 2 * pr] * attnLo(vw);
                    dp[u] = dp[u] + dos[ii[u] * (DMAX + 1) + 2 * pr + 1] * attnHi(vw);
                }
            }
        }
    }
    [ForceUnroll]
    for (uint u = 0; u < PPT; u++)
    {
        uint i = ii[u], kk = kv[u], r = r0 + i, key = k0 + kk;
        bool live = r < rows && key < p.keys && (p.causal == 0 || key <= (r % p.queries) + offset);
        float pv = live ? exp(sv[u] * p.scale - ls[i]) : 0.0f;
        ps[i * BK + kk] = pv;
        dss[i * BK + kk] = live ? pv * (dp[u] - ds[i]) : 0.0f;
    }
}

void loadRows(uint t, uint base, uint r0, uint rows, uint d2)
{
    for (uint e = t; e < BQ * D2; e += THREADS)
    {
        uint i = e / D2, pr = e % D2, r = r0 + i;
        bool ok = r < rows && pr < d2;
        qs[i * STRIDE + pr] = ok ? q[(base + r) * d2 + pr] : 0u;
        dos[i * (DMAX + 1) + 2 * pr] = ok ? dO[(base + r) * p.headDim + 2 * pr] : 0.0f;
        dos[i * (DMAX + 1) + 2 * pr + 1] = ok ? dO[(base + r) * p.headDim + 2 * pr + 1] : 0.0f;
    }
    if (t < BQ) { uint r = r0 + t; ls[t] = r < rows ? lse[base + r] : 0.0f; ds[t] = r < rows ? dD[base + r] : 0.0f; }
}

void loadKeys(uint t, uint kvHead, uint k0, uint d2)
{
    for (uint e = t; e < BK * D2; e += THREADS)
    {
        uint kk = e / D2, pr = e % D2, key = k0 + kk;
        bool ok = key < p.keys && pr < d2;
        uint at = (kvHead * p.cacheKeys + key) * d2 + pr;
        ks[kk * STRIDE + pr] = ok ? k[at] : 0u;
        vs[kk * STRIDE + pr] = ok ? v[at] : 0u;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = local.x, kvHead = group.y, d2 = p.headDim / 2;
    uint hpk = p.heads / p.kvHeads, rows = hpk * p.queries, base = kvHead * rows;
    uint offset = p.keys - p.queries;
    uint owner = t / 8, part = t % 8;   // KV: key owner; Q: row owner (BQ = 16) .. see below
    float a0[CPT], a1[CPT];
    [ForceUnroll] for (uint c = 0; c < CPT; c++) { a0[c] = 0.0f; a1[c] = 0.0f; }
#ifdef KV
    uint k0 = group.x * BK;
    loadKeys(t, kvHead, k0, d2);
    for (uint r0 = 0; r0 < rows; r0 += BQ)
    {
        if (p.causal != 0)
        {
            // the tile's largest query index: does any of its rows see key k0?
            uint rLast = min(r0 + BQ, rows) - 1;
            uint iMax = (r0 / p.queries != rLast / p.queries) ? p.queries - 1 : rLast % p.queries;
            if (iMax + offset < k0) continue;   // uniform across the group
        }
        GroupMemoryBarrierWithGroupSync();   // the previous tile's rows are no longer read
        loadRows(t, base, r0, rows, d2);
        GroupMemoryBarrierWithGroupSync();
        pairs(t, r0, rows, k0, d2);
        GroupMemoryBarrierWithGroupSync();
        // key `owner` (16 keys, 8 threads each): dV += P dO, dK += dS Q, row by row
        for (uint i = 0; i < BQ; i++)
        {
            float pv = ps[i * BK + owner], dsv = dss[i * BK + owner];
            [ForceUnroll]
            for (uint c = 0; c < CPT; c++)
            {
                uint col = part + 8 * c;
                a1[c] = a1[c] + pv * dos[i * (DMAX + 1) + col];
                a0[c] = a0[c] + dsv * half2At(qs[i * STRIDE + col / 2], col);
            }
        }
    }
    uint key = k0 + owner;
    if (key < p.keys)
    {
        uint at = (kvHead * p.cacheKeys + key) * p.headDim;
        [ForceUnroll]
        for (uint c = 0; c < CPT; c++)
        {
            uint col = part + 8 * c;
            if (col < p.headDim) { out0[at + col] = a0[c] * p.scale; out1[at + col] = a1[c]; }
        }
    }
#else
    uint r0 = group.x * BQ;
    loadRows(t, base, r0, rows, d2);
    uint kEnd = p.keys;
    if (p.causal != 0)
    {
        uint rLast = min(r0 + BQ, rows) - 1;
        uint iMax = (r0 / p.queries != rLast / p.queries) ? p.queries - 1 : rLast % p.queries;
        kEnd = min(kEnd, iMax + offset + 1);
    }
    for (uint k0 = 0; k0 < kEnd; k0 += BK)
    {
        GroupMemoryBarrierWithGroupSync();   // the previous tile's keys are no longer read
        loadKeys(t, kvHead, k0, d2);
        GroupMemoryBarrierWithGroupSync();
        pairs(t, r0, rows, k0, d2);
        GroupMemoryBarrierWithGroupSync();
        // row `owner` (BQ rows, 128 / BQ threads each): dQ += dS K, key by key
        if (owner < BQ)
        {
            for (uint kk = 0; kk < BK; kk++)
            {
                float dsv = dss[owner * BK + kk];
                [ForceUnroll]
                for (uint c = 0; c < CPT; c++)
                {
                    uint col = part + 8 * c;
                    a0[c] = a0[c] + dsv * half2At(ks[kk * STRIDE + col / 2], col);
                }
            }
        }
    }
    uint r = r0 + owner;
    if (owner < BQ && r < rows)
    {
        uint at = (base + r) * p.headDim;
        [ForceUnroll]
        for (uint c = 0; c < CPT; c++)
        {
            uint col = part + 8 * c;
            if (col < p.headDim) out0[at + col] = a0[c] * p.scale;
        }
    }
#endif
}
