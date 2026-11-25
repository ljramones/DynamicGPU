// The block formats of QuantizedMatrix (1.9.0), shared by the quantize, dequantize and quantized GEMM kernels. Every
// operation here gives the same bits on every device: FP16 rounding in integer arithmetic, scales from one
// correctly rounded multiply, and codes found by exact comparisons with the midpoints (2c + 1) d / 2.
//
// Formats: 0 Q8 (x = q d, q in [-127, 127]), 1 Q4 (x = q d, q in [-8, 7], stored q + 8), 2 Q4M (x = fl(q d + m),
// q in [0, 15]); and the K formats (1.9.1) 3 Q4K, 4 Q5K, 5 Q6K, whose reads are in quant_read.slang. Codes are row-major, Q8 one byte each, Q4 and Q4M a nibble each (element 2t low); the block
// parameters are FP16, one d per block (Q8, Q4) or a pair d, m (Q4M), from parameter word p.qparams.
#pragma once

static const uint QF_Q8 = 0;
static const uint QF_Q4 = 1;
static const uint QF_Q4M = 2;
static const uint QF_Q4K = 3;   // 1.9.1: K formats, super-blocks of 256 (quant_read.slang)
static const uint QF_Q5K = 4;
static const uint QF_Q6K = 5;

float qInv127() { return asfloat(0x3c010204u); }   // Java's 1f / 127f
float qInv15() { return asfloat(0x3d888889u); }    // Java's 1f / 15f

uint qBits(uint format) { return format == QF_Q8 ? 8u : 4u; }

// float -> FP16 bits, round to nearest even (as Java's Float.floatToFloat16), overflow to infinity
uint qF32ToF16(float f)
{
    uint x = asuint(f);
    uint sign = (x >> 16) & 0x8000u;
    uint e = (x >> 23) & 0xFFu;
    uint mant = x & 0x7FFFFFu;
    if (e == 0xFFu) return sign | 0x7C00u | (mant != 0 ? 0x200u | (mant >> 13) : 0u);
    int exp = int(e) - 112;   // rebias 127 -> 15
    if (exp >= 31) return sign | 0x7C00u;
    if (exp <= 0)
    {
        if (exp < -10) return sign;
        mant |= 0x800000u;
        uint shift = uint(14 - exp);
        uint h = mant >> shift;
        uint rem = mant & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (h & 1u) != 0)) h++;
        return sign | h;
    }
    uint h = (uint(exp) << 10) | (mant >> 13);
    uint rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0)) h++;   // a carry rounds into the exponent, or to infinity
    return sign | h;
}

float qF16ToF32(uint h) { return f16tof32(h & 0xFFFFu); }

// The integer nearest to y / d (ties to even), d > 0: an estimate corrected against the exact midpoints
// (QuantizedMatrix.nearest)
int qNearest(float y, float d)
{
    float est = round(y * (1.0f / d));
    int c = int(clamp(est, -300.0f, 300.0f));
    [ForceUnroll]
    for (uint step = 0; step < 2; step++)
    {
        float up = float(2 * c + 1) * d * 0.5f;
        float down = float(2 * c - 1) * d * 0.5f;
        if (y > up || (y == up && (c & 1) != 0)) c++;
        else if (y < down || (y == down && (c & 1) != 0)) c--;
    }
    return c;
}

// Code of element col of a row from its code word: signed for Q8 and Q4
int qCode(uint format, uint word, uint col)
{
    if (format == QF_Q8)
    {
        uint b = (word >> (8u * (col & 3u))) & 0xFFu;
        return int(b) - (b >= 128u ? 256 : 0);
    }
    int u = int((word >> (4u * (col & 7u))) & 0xFu);
    return format == QF_Q4 ? u - 8 : u;
}

// The word index of element col's code in row r, for a matrix of cols columns
uint qCodeWord(uint format, uint r, uint col, uint cols)
{
    return format == QF_Q8 ? (r * cols + col) >> 2 : (r * cols + col) >> 3;
}

// Parameter word and half of block b (global, row-major) for parameter t (0 = d, 1 = m)
uint qParamHalf(uint format, uint b, uint t) { return format == QF_Q4M ? 2u * b + t : b; }
