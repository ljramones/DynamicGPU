// What the quantized GEMM kernels (1.9.0) share: A as 32-bit words of FP16 (IN_KIND 0) or BF16 (IN_KIND 1)
// activations, B as a packed QuantizedMatrix (n x k, the weights of a linear layer) read as 32-bit words, and the
// block reads. Products of an activation and a code are exact in FP32 (11 or 8 bits times at most 8), so each
// block's sum s = sum a q depends only on its order; the block then adds fl(d s), and Q4M also fl(m sum a). The K
// formats (1.9.1) work in units of 32 elements: a Q4K or Q5K sub-block adds fl(fl(D s) - fl(M sum a)), a unit of
// Q6K its two sub-blocks' fl(D s) in turn.
#ifndef IN_KIND
#define IN_KIND 0
#endif
#define QUANT 1
// A and B come first: Slang numbers Metal buffers in declaration order.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> b;
#include "gemm_common.slang"
#include "gemm_epilogue.slang"
uint qWord(uint w) { return b[w]; }
#include "quant_read.slang"

// The format and block size are specialization constants, so each pipeline is compiled for one of each (QF_*, 32,
// 64 or 128, 256 for the K formats): the format's branches and the block loops then cost nothing.
[vk::constant_id(1)] const uint QFORMAT = 0;
[vk::constant_id(2)] const uint QBLOCK = 32;
// (macros, not static constants: Slang cannot emit a global computed from a specialization constant)
#define QK (QFORMAT >= QF_Q4K)
// The unit the kernels deal out and add terms by: a block, or 32 elements of a K format (a Q4K or Q5K sub-block, two
// Q6K sub-blocks of 16)
#define QUNIT (QFORMAT >= QF_Q4K ? 32u : QBLOCK)
// Whether the terms need sum a: the formats with a minimum
#define QMIN (QFORMAT == QF_Q4M || QFORMAT == QF_Q4K || QFORMAT == QF_Q5K)

#if IN_KIND == 1
float aLo(uint w) { return asfloat(w << 16); }
float aHi(uint w) { return asfloat(w & 0xFFFF0000u); }
#else
float aLo(uint w) { return f16tof32(w & 0xFFFFu); }
float aHi(uint w) { return f16tof32(w >> 16); }
#endif
float elemA(uint e) { uint w = a[e >> 1]; return (e & 1) != 0 ? aHi(w) : aLo(w); }
// Element e's 16 raw bits
uint rawA(uint e) { return (a[e >> 1] >> (16u * (e & 1u))) & 0xFFFFu; }
// 8 activations from element e: as words when e is even, else one at a time
void load8A(uint e, out float v[8])
{
    if ((e & 1) == 0) { [ForceUnroll] for (uint t = 0; t < 4; t++) { uint w = a[(e >> 1) + t]; v[2 * t] = aLo(w); v[2 * t + 1] = aHi(w); } }
    else { [ForceUnroll] for (uint t = 0; t < 8; t++) v[t] = elemA(e + t); }
}

// B as scalar words: indexing a vector by a runtime component spilled it to memory (52 GB/s at m = 1)
uint wordB(uint w) { return b[w]; }

// The 8 codes of row r, elements c0 .. c0 + 7 (c0 a multiple of 8), as floats
void load8Codes(uint r, uint c0, out float q[8])
{
    uint format = QFORMAT;
    if (QK)
    {
        uint w0 = wordB((r * p.k + c0) >> 3);
        uint h = format == QF_Q4K ? 0u : kHigh8(format, p.qhigh, r * p.k + c0);
        uint hb = format == QF_Q5K ? 1u : 2u;
        [ForceUnroll] for (uint t = 0; t < 8; t++) q[t] = float(kCode(format, (w0 >> (4u * t)) & 0xFu, h >> (hb * t)));
    }
    else if (format == QF_Q8)
    {
        uint w = (r * p.k + c0) >> 2;
        uint w0 = wordB(w), w1 = wordB(w + 1);
        [ForceUnroll] for (uint t = 0; t < 4; t++) { q[t] = float(qCode(format, w0, t)); q[t + 4] = float(qCode(format, w1, t)); }
    }
    else
    {
        uint w0 = wordB((r * p.k + c0) >> 3);
        [ForceUnroll] for (uint t = 0; t < 8; t++) q[t] = float(qCode(format, w0, t));
    }
}

// Parameter t (0 = d, 1 = m) of block bl of row r
float blockParameter(uint r, uint bl, uint t)
{
    uint h = qParamHalf(QFORMAT, r * (p.k / QBLOCK) + bl, t);
    return qF16ToF32(wordB(p.qparams + (h >> 1)) >> (16u * (h & 1u)));
}

// A block's contribution: fl(d s), or fl(fl(d s) + fl(m sumA)) for Q4M; a Q4K or Q5K sub-block's
// fl(fl(D s) - fl(M sumA)); a Q6K sub-block's fl(D s)
float blockTerm(float s, float sumA, float d, float m)
{
    float t = d * s;
    if (QFORMAT == QF_Q4M) t = t + m * sumA;
    else if (QFORMAT == QF_Q4K || QFORMAT == QF_Q5K) t = t - m * sumA;
    return t;
}

// The scales of unit u of row r: (d, m) of a block, (D, M) of a Q4K or Q5K sub-block, or Q6K's two sub-blocks' D
// (it has no M)
float2 unitScale(uint r, uint u)
{
    if (QFORMAT == QF_Q6K) return kScale6Pair(p.qparams, r, u * QUNIT, p.k);
    if (QK) return kScale(QFORMAT, p.qparams, r, u * QUNIT, p.k);
    return float2(blockParameter(r, u, 0), QFORMAT == QF_Q4M ? blockParameter(r, u, 1) : 0.0f);
}

// The unit's terms added to acc: blockTerm, or for Q6K its two sub-blocks' in turn (s2 the second's sum)
float addTerms(float acc, float s, float s2, float sumA, float2 sc)
{
    if (QFORMAT == QF_Q6K) return (acc + blockTerm(s, 0.0f, sc.x, 0.0f)) + blockTerm(s2, 0.0f, sc.y, 0.0f);
    return acc + blockTerm(s, sumA, sc.x, sc.y);
}
