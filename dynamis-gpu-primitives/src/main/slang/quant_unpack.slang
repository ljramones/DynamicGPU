// Quantize.unpack (1.9.0): a QuantizedMatrix to FP32 (rows x cols, row-major), one element per thread: q d, or
// fl(q d + m), bit for bit as QuantizedMatrix.dequantize.
#include "dynamis/kit.slang"
#include "quant_common.slang"

struct Params
{
    uint rows;
    uint cols;
    uint format;
    uint block;
    uint count;     // rows * cols
    uint qparams;   // the parameter words' start
    uint gridX;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> packed;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void quant_unpack(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint e = dg_linear_thread(group, local, p.gridX, THREADS);
    if (e >= p.count) return;
    uint r = e / p.cols, col = e % p.cols;
    uint b = e / p.block;   // blocks are row-major and cols is a multiple of the block
    float q = float(qCode(p.format, packed[qCodeWord(p.format, r, col, p.cols)], col));
    uint dh = qParamHalf(p.format, b, 0);
    float d = qF16ToF32(packed[p.qparams + (dh >> 1)] >> (16u * (dh & 1u)));
    float v = q * d;
    if (p.format == QF_Q4M)
    {
        uint mh = qParamHalf(p.format, b, 1);
        v = v + qF16ToF32(packed[p.qparams + (mh >> 1)] >> (16u * (mh & 1u)));
    }
    y[e] = v;
}
