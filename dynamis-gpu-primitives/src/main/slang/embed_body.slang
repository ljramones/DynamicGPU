// Embedding (1.10.0): rows of a table gathered by token, shared by embed[_h].slang (OUT_T, ENTRY):
//     y[t][j] = table[tokens[t]][j],   j < dim
// with the tokens read from a GPU buffer (the sampler's output, for instance) and the table FP32, FP16 or a packed
// QuantizedMatrix (vocabulary x dim), dequantized exactly as QuantizedMatrix.value: the same bits on every device.
// One thread per element. A token beyond the vocabulary is reported (DG_ERROR_INDEX, index = t) and its row left
// untouched.
#include "dynamis/kit.slang"

struct EmbedParams
{
    uint count;      // tokens * dim threads
    uint dim;
    uint vocab;
    uint ldy;
    uint kind;       // 0 FP32, 1 FP16, 2 quantized (format below)
    uint qformat;
    uint qblock;
    uint qparams;
    uint qhigh;
    uint gridX;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> tokens;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> table;
[[vk::binding(2, 0)]] RWStructuredBuffer<OUT_T> y;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> errors;
[[vk::push_constant]] ConstantBuffer<EmbedParams> p;

uint qWord(uint w) { return table[w]; }
#include "quant_read.slang"

static const uint THREADS = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint id = dg_linear_thread(group, local, p.gridX, THREADS);
    if (id >= p.count) return;
    uint t = id / p.dim, j = id % p.dim, row = tokens[t];
    if (row >= p.vocab)
    {
        if (j == 0) dg_report_error(errors, DG_ERROR_INDEX, t);
        return;
    }
    uint e = row * p.dim + j;
    float v;
    if (p.kind == 0) v = asfloat(table[e]);
    else if (p.kind == 1) v = qF16ToF32(table[e >> 1] >> (16u * (e & 1u)));
    else
    {
        uint f = p.qformat;
        if (f >= QF_Q4K)
        {
            uint u = (table[qCodeWord(f, row, j, p.dim)] >> (4u * (j & 7u))) & 0xFu;
            uint hi = f == QF_Q4K ? 0u : kHigh8(f, p.qhigh, e & ~7u) >> ((f == QF_Q5K ? 1u : 2u) * (e & 7u));
            float q = float(kCode(f, u, hi));
            float2 dm = kScale(f, p.qparams, row, j, p.dim);
            v = f == QF_Q6K ? dm.x * q : dm.x * q - dm.y;
        }
        else
        {
            float q = float(qCode(f, table[qCodeWord(f, row, j, p.dim)], j));
            uint bl = row * (p.dim / p.qblock) + j / p.qblock;
            uint dh = qParamHalf(f, bl, 0);
            float d = qF16ToF32(table[p.qparams + (dh >> 1)] >> (16u * (dh & 1u)));
            v = q * d;
            if (f == QF_Q4M)
            {
                uint mh = qParamHalf(f, bl, 1);
                v = v + qF16ToF32(table[p.qparams + (mh >> 1)] >> (16u * (mh & 1u)));
            }
        }
    }
    y[t * p.ldy + j] = OUT_T(v);
}
