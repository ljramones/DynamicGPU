// AttentionF16.backward, the pre-pass (1.12.0): D[row] = sum_c dO[row][c] O[row][c], a thread per row, in column
// order. O_HALF: O is FP16 (packed pairs).
#include "attn_bwd_common.slang"

#ifdef O_HALF
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> o;
#else
[[vk::binding(0, 0)]] RWStructuredBuffer<float> o;
#endif
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dO;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dD;
[[vk::push_constant]] ConstantBuffer<AttnBwdParams> p;

[shader("compute")]
[numthreads(256, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint row = dg_linear_thread(group, local, p.gridX, 256);
    if (row >= p.heads * p.queries) return;
    uint d = p.headDim, base = row * d;
    float acc = 0.0f;
    for (uint c = 0; c < d; c += 2)
    {
#ifdef O_HALF
        uint w = o[base / 2 + c / 2];
        float lo = attnLo(w), hi = attnHi(w);
#else
        float lo = o[base + c], hi = o[base + c + 1];
#endif
        acc = acc + dO[base + c] * lo;
        acc = acc + dO[base + c + 1] * hi;
    }
    dD[row] = acc;
}
