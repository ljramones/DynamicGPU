// Reads of a packed QuantizedMatrix through the including file's uint qWord(uint w), for any format (1.9.1): codes
// (with Q5K's and Q6K's high bits) and the K formats' sub-block scales D = d sc and M = dmin mn, both exact in FP32.
#pragma once
#include "quant_common.slang"

uint qByte(uint at) { return (qWord(at >> 2) >> (8u * (at & 3u))) & 0xFFu; }

// D and M of element col's sub-block in row r of a K-format matrix (cols columns), parameters from word qparams.
// Q4K and Q5K: llama.cpp's get_scale_min_k4 over the 12 scale bytes; Q6K: an int8 scale per 16 elements, M = 0.
float2 kScale(uint format, uint qparams, uint r, uint col, uint cols)
{
    uint sb = r * (cols / 256u) + col / 256u;
    uint base = qparams + sb * (format == QF_Q6K ? 5u : 4u);
    uint w0 = qWord(base);
    float d = qF16ToF32(w0);
    uint at = 4u * (base + 1u);   // the scale bytes
    if (format == QF_Q6K)
    {
        int sc = int(qByte(at + (col % 256u) / 16u));
        if (sc >= 128) sc -= 256;
        return float2(d * float(sc), 0.0f);
    }
    // without branches: neighbouring lanes of the skinny kernel read different sub-blocks, and j < 4 diverged
    uint j = (col % 256u) / 32u;
    uint bj = qByte(at + j), b4 = qByte(at + j + 4u), bx = qByte(at + (j ^ 4u));   // j ^ 4 is j - 4 for j >= 4
    bool low = j < 4u;
    uint sc = low ? bj & 63u : (b4 & 0xFu) | ((bx >> 6) << 4);
    uint mn = low ? b4 & 63u : (b4 >> 4) | ((bj >> 6) << 4);
    return float2(d * float(sc), qF16ToF32(w0 >> 16) * float(mn));
}

// Q6K: the D of both 16-element sub-blocks of the 32-element unit holding col (col % 32 == 0): their int8 scales are
// adjacent bytes of one word
float2 kScale6Pair(uint qparams, uint r, uint col, uint cols)
{
    uint sb = r * (cols / 256u) + col / 256u, base = qparams + sb * 5u;
    float d = qF16ToF32(qWord(base));
    uint j = (col % 256u) / 16u;   // even
    uint w = qWord(base + 1u + j / 4u) >> (8u * (j % 4u));
    int s0 = int(w & 0xFFu), s1 = int((w >> 8) & 0xFFu);
    if (s0 >= 128) s0 -= 256;
    if (s1 >= 128) s1 -= 256;
    return float2(d * float(s0), d * float(s1));
}

// The high bits of the 8 codes from element e (a multiple of 8) of a Q5K or Q6K matrix, as one word: code t's at
// bit hb t, hb the high bits per code
uint kHigh8(uint format, uint qhigh, uint e)
{
    if (format == QF_Q5K) return qWord(qhigh + (e >> 5)) >> (e & 31u);
    return qWord(qhigh + (e >> 4)) >> (2u * (e & 15u));
}

// A K format's code from its low nibble u and high bits h
int kCode(uint format, uint u, uint h)
{
    if (format == QF_Q4K) return int(u);
    if (format == QF_Q5K) return int(u | ((h & 1u) << 4));
    return int(u | ((h & 3u) << 4)) - 32;
}
