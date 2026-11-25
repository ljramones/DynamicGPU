// Norms.backward's column sums (1.12.0): dW[j] += sum over blocks of partW[block][j], and for LayerNorm dB[j] the
// same of partB, each thread one column adding the blocks in order (norm_backward_body.slang writes them).
#include "dynamis/kit.slang"

struct NormColumnsParams
{
    uint n;
    uint blocks;
    uint bias;
    uint gridX;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> partW;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> partB;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dW;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dB;
[[vk::push_constant]] ConstantBuffer<NormColumnsParams> p;

[shader("compute")]
[numthreads(256, 1, 1)]
void norm_backward_columns(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint j = dg_linear_thread(group, local, p.gridX, 256);
    if (j >= p.n) return;
    float sw = 0.0f, sb = 0.0f;
    for (uint k = 0; k < p.blocks; k++)
    {
        sw = sw + partW[k * p.n + j];
        if (p.bias != 0) sb = sb + partB[k * p.n + j];
    }
    dW[j] = dW[j] + sw;
    if (p.bias != 0) dB[j] = dB[j] + sb;
}
