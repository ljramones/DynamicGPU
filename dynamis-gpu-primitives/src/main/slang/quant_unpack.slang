// Quantize.unpack (1.9.0): a QuantizedMatrix to FP32 (rows x cols, row-major), one element per thread: q d,
// fl(q d + m), or (K formats, 1.9.1) fl(D q - M) and fl(D q), bit for bit as QuantizedMatrix.dequantize.
#include "dynamis/kit.slang"

struct Params
{
    uint rows;
    uint cols;
    uint format;
    uint block;
    uint count;     // rows * cols
    uint qparams;   // the parameter words' start
    uint gridX;
    uint qhigh;     // the high bits' start (Q5K, Q6K)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> packed;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;
[[vk::push_constant]] ConstantBuffer<Params> p;

uint qWord(uint w) { return packed[w]; }
#include "quant_read.slang"

static const uint THREADS = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void quant_unpack(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint e = dg_linear_thread(group, local, p.gridX, THREADS);
    if (e >= p.count) return;
    uint r = e / p.cols, col = e % p.cols;
    if (p.format >= QF_Q4K)
    {
        uint u = (packed[qCodeWord(p.format, r, col, p.cols)] >> (4u * (col & 7u))) & 0xFu;
        uint h = p.format == QF_Q4K ? 0u : kHigh8(p.format, p.qhigh, e & ~7u) >> ((p.format == QF_Q5K ? 1u : 2u) * (e & 7u));
        float qk = float(kCode(p.format, u, h));
        float2 dm = kScale(p.format, p.qparams, r, col, p.cols);
        y[e] = p.format == QF_Q6K ? dm.x * qk : dm.x * qk - dm.y;
        return;
    }
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
