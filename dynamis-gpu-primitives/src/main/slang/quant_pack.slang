// Quantize.pack (1.9.0): FP32 values (rows x cols, row-major) to a QuantizedMatrix, bit for bit as
// QuantizedMatrix.quantize. Thread t encodes blocks 2t and 2t + 1 (global, row-major), so it alone writes their code
// words and their parameter word(s). A non-finite value, or a scale that overflows FP16, is reported
// (DG_ERROR_NONFINITE, index = the block) and its block written as zeros.
#include "dynamis/kit.slang"
#include "quant_common.slang"

struct Params
{
    uint rows;
    uint cols;
    uint format;
    uint block;
    uint blocks;    // rows * cols / block
    uint qparams;   // the parameter words' start: QuantizedMatrix.parameterOffset / 4
    uint gridX;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> x;        // FP32 bits
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> packed;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> errors;
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 64;

// Encodes block b, writing its code words; returns its parameters as FP16 bits (m is 0 for Q8 and Q4)
void encodeBlock(uint b, out uint dBits, out uint mBits)
{
    uint bpr = p.cols / p.block, r = b / bpr, c0 = (b % bpr) * p.block, base = r * p.cols + c0;
    float lo = dg_positive_infinity(), hi = dg_negative_infinity(), amax = 0.0f, extreme = 0.0f;
    bool finite = true;
    for (uint i = 0; i < p.block; i++)
    {
        float v = asfloat(x[base + i]);
        finite = finite && dg_is_finite(v);
        lo = dg_min(lo, v); hi = dg_max(hi, v);
        if (abs(v) > amax) { amax = abs(v); extreme = v; }
    }
    float m = 0.0f;
    mBits = 0;
    if (p.format == QF_Q8) dBits = qF32ToF16(amax * qInv127());
    else if (p.format == QF_Q4) dBits = qF32ToF16(extreme * -0.125f);
    else
    {
        mBits = qF32ToF16(lo);
        m = qF16ToF32(mBits);
        dBits = qF32ToF16((hi - lo) * qInv15());
    }
    float d = qF16ToF32(dBits);
    if (!finite || !dg_is_finite(d) || !dg_is_finite(m))
    {
        dg_report_error(errors, DG_ERROR_NONFINITE, b);
        dBits = 0; mBits = 0; d = 0.0f;
    }
    int lowest = p.format == QF_Q8 ? -127 : p.format == QF_Q4 ? -8 : 0;
    int highest = p.format == QF_Q8 ? 127 : p.format == QF_Q4 ? 7 : 15;
    uint perWord = p.format == QF_Q8 ? 4u : 8u, bits = qBits(p.format);
    uint word0 = qCodeWord(p.format, r, c0, p.cols);
    for (uint w = 0; w < p.block / perWord; w++)
    {
        uint word = 0;
        for (uint j = 0; j < perWord; j++)
        {
            float v = asfloat(x[base + w * perWord + j]);
            float y = p.format == QF_Q4M ? v - m : v;
            int code = d == 0.0f ? 0 : clamp(d > 0.0f ? qNearest(y, d) : qNearest(-y, -d), lowest, highest);
            uint u = p.format == QF_Q8 ? uint(code) & 0xFFu : p.format == QF_Q4 ? uint(code + 8) : uint(code);
            word |= u << (bits * j);
        }
        packed[word0 + w] = word;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void quant_pack(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = dg_linear_thread(group, local, p.gridX, THREADS);
    uint b0 = 2 * t;
    if (b0 >= p.blocks) return;
    uint d0, m0, d1 = 0, m1 = 0;
    encodeBlock(b0, d0, m0);
    bool second = b0 + 1 < p.blocks;
    if (second) encodeBlock(b0 + 1, d1, m1);
    if (p.format == QF_Q4M)
    {
        packed[p.qparams + b0] = d0 | (m0 << 16);
        if (second) packed[p.qparams + b0 + 1] = d1 | (m1 << 16);
    }
    else packed[p.qparams + t] = d0 | (d1 << 16);
}
