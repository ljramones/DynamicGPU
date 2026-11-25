// AttentionF16.backward (1.12.0), shared by attn_bwd_dot, attn_bwd_kv and attn_bwd_q. Q, K and V are FP16 (packed in
// pairs), O the forward output, dO, dQ, dK and dV FP32, L the forward's log-sum-exp of each row's scaled scores.
//     P = exp(scale Q K^T - L) (masked: 0),  D = rowsum(dO * O),  dS = P (dO V^T - D)
//     dV = P^T dO,  dK = scale dS^T Q,  dQ = scale dS K
// The rows of a KV head (its query heads' queries) are consecutive rows of Q, O, dO and dQ, and of L and D.
#pragma once
#include "attn_common.slang"

struct AttnBwdParams
{
    uint heads;
    uint kvHeads;
    uint queries;
    uint keys;
    uint headDim;
    uint cacheKeys;   // K, V, dK and dV are [kvHeads][cacheKeys][headDim]
    float scale;
    uint causal;
    uint gridX;       // attn_bwd_dot
    uint pad0;
    uint pad1;
    uint pad2;
};
