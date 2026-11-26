// Quantize.unpack (1.9.0): a QuantizedMatrix to FP32 (rows x cols, row-major), one element per thread: q d,
// fl(q d + m), or (K formats, 1.9.1) fl(D q - M) and fl(D q), bit for bit as QuantizedMatrix.dequantize.
// Quantize.unpackF16 (1.15.0, OUT_F16): the same values rounded to FP16 (nearest even, as Java's Float.floatToFloat16),
// for an FP16 GEMM on the unpacked weights. A thread decodes 8 consecutive elements, which share a block, a code word
// (two for Q8), a sub-block scale and a high-bits word, and writes them as 4 words; each value is computed by the
// same operations as unpackAt. The entry file defines ENTRY and, for FP16, OUT_F16.
#include "dynamis/kit.slang"

struct Params
{
    uint rows;
    uint cols;
    uint format;
    uint block;
    uint count;     // rows * cols (even for FP16)
    uint qparams;   // the parameter words' start
    uint gridX;
    uint qhigh;     // the high bits' start (Q5K, Q6K)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> packed;
#if OUT_F16
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> y;     // FP16 pairs
#else
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;
#endif
[[vk::push_constant]] ConstantBuffer<Params> p;

uint qWord(uint w) { return packed[w]; }
#include "quant_read.slang"

static const uint THREADS = 256;

// element e's dequantized value
float unpackAt(uint e)
{
    uint r = e / p.cols, col = e % p.cols;
    if (p.format >= QF_Q4K)
    {
        uint u = (packed[qCodeWord(p.format, r, col, p.cols)] >> (4u * (col & 7u))) & 0xFu;
        uint h = p.format == QF_Q4K ? 0u : kHigh8(p.format, p.qhigh, e & ~7u) >> ((p.format == QF_Q5K ? 1u : 2u) * (e & 7u));
        float qk = float(kCode(p.format, u, h));
        float2 dm = kScale(p.format, p.qparams, r, col, p.cols);
        return p.format == QF_Q6K ? dm.x * qk : dm.x * qk - dm.y;
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
    return v;
}

#if OUT_F16
// elements e .. e + 7 (e a multiple of 8; every block and sub-block is a multiple of 8 and cols a multiple of 32)
void unpack8(uint e, out float v[8])
{
    uint r = e / p.cols, col = e % p.cols;
    if (p.format >= QF_Q4K)
    {
        uint w = packed[qCodeWord(p.format, r, col, p.cols)];
        uint h8 = p.format == QF_Q4K ? 0u : kHigh8(p.format, p.qhigh, e);
        uint hb = p.format == QF_Q5K ? 1u : 2u;
        float2 dm = kScale(p.format, p.qparams, r, col, p.cols);
        [ForceUnroll]
        for (uint i = 0; i < 8u; i++)
        {
            float qk = float(kCode(p.format, (w >> (4u * i)) & 0xFu, p.format == QF_Q4K ? 0u : h8 >> (hb * i)));
            v[i] = p.format == QF_Q6K ? dm.x * qk : dm.x * qk - dm.y;
        }
        return;
    }
    uint b = e / p.block;
    uint dh = qParamHalf(p.format, b, 0);
    float d = qF16ToF32(packed[p.qparams + (dh >> 1)] >> (16u * (dh & 1u)));
    float m = 0.0f;
    if (p.format == QF_Q4M)
    {
        uint mh = qParamHalf(p.format, b, 1);
        m = qF16ToF32(packed[p.qparams + (mh >> 1)] >> (16u * (mh & 1u)));
    }
    uint word = qCodeWord(p.format, r, col, p.cols);
    [ForceUnroll]
    for (uint i = 0; i < 8u; i++)
    {
        uint wd = p.format == QF_Q8 ? packed[word + (i >> 2)] : packed[word];
        float x = float(qCode(p.format, wd, col + i)) * d;
        v[i] = p.format == QF_Q4M ? x + m : x;
    }
}
#endif

[shader("compute")]
[numthreads(256, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = dg_linear_thread(group, local, p.gridX, THREADS);
#if OUT_F16
    uint e = 8u * t;   // count is a multiple of 8
    if (e >= p.count) return;
    float v[8];
    unpack8(e, v);
    [ForceUnroll]
    for (uint j = 0; j < 4u; j++) y[4u * t + j] = qF32ToF16(v[2u * j]) | (qF32ToF16(v[2u * j + 1u]) << 16);
#else
    if (t >= p.count) return;
    y[t] = unpackAt(t);
#endif
}
