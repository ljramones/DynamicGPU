// Logic (1.14.0): comparisons, logical operations, classification, int32 arithmetic and select over 32-bit words, shared
// by logic_unary.slang, logic_binary.slang and logic_select.slang. Every operation works on the bits in integer
// arithmetic, so results are the same on every device, subnormals included (devices that flush FP32 subnormals do so
// in float arithmetic, not here). Masks are uint32 words: 1 true, 0 false; an input mask is true when nonzero. OP, a
// specialization constant, picks the operation. Views have rank 4; an input dimension of extent 1 broadcasts.
#pragma once
#include "dynamis/kit.slang"

[vk::constant_id(0)] const uint OP = 0;

struct LogicUnaryParams
{
    uint count;
    uint gridX;
    uint xLength;   // buffer lengths in words, for debug builds
    uint yLength;
    DG_VIEW_FIELDS4(x)
    DG_VIEW_FIELDS4(y)
};

struct LogicBinaryParams
{
    uint count;
    uint gridX;
    uint aLength;
    uint bLength;
    uint yLength;
    DG_VIEW_FIELDS4(a)
    DG_VIEW_FIELDS4(b)
    DG_VIEW_FIELDS4(y)
};

// An input broadcast to the output's shape: an offset and four strides, 0 along the dimensions it broadcasts (it shares
// the output's extents), so four views fit the 128-byte parameter block.
#define LG_BROADCAST_FIELDS(x) uint x##Offset; uint x##Stride0; uint x##Stride1; uint x##Stride2; uint x##Stride3;
#define LG_BROADCAST_AT(P, V, C) (P.V##Offset + C.x * P.V##Stride0 + C.y * P.V##Stride1 + C.z * P.V##Stride2 + C.w * P.V##Stride3)

struct LogicSelectParams
{
    uint count;
    uint gridX;
    uint mLength;
    uint aLength;
    uint bLength;
    uint yLength;
    LG_BROADCAST_FIELDS(m)
    LG_BROADCAST_FIELDS(a)
    LG_BROADCAST_FIELDS(b)
    DG_VIEW_FIELDS4(y)
};

static const uint THREADS = 256;

bool lgIsNaN(uint x) { return (x & 0x7FFFFFFFu) > 0x7F800000u; }

// The order of two non-NaN FP32 values given as bits: -1, 0 or 1; -0 equals +0 (IEEE, as Java's float comparisons).
int lgOrderF32(uint a, uint b)
{
    if (((a | b) & 0x7FFFFFFFu) == 0u) return 0;   // both zeros, of any sign
    uint ka = (a & 0x80000000u) != 0u ? ~a : (a | 0x80000000u);
    uint kb = (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
    return ka < kb ? -1 : (ka > kb ? 1 : 0);
}

uint lgCompare(uint cmp, int order) // cmp: 0 ==, 1 !=, 2 <, 3 <=, 4 >, 5 >=
{
    bool r;
    switch (cmp)
    {
    case 0: r = order == 0; break;
    case 1: r = order != 0; break;
    case 2: r = order < 0; break;
    case 3: r = order <= 0; break;
    case 4: r = order > 0; break;
    default: r = order >= 0; break;
    }
    return r ? 1u : 0u;
}

// Logic.Binary ops, numbered as Logic.binaryOp: 0-5 FP32 comparisons, 6-11 INT32 comparisons, 12 AND, 13 OR, 14 XOR,
// 15 int ADD, 16 SUB, 17 MUL, 18 MIN, 19 MAX (two's complement, wrapping, as Java's int).
uint lgBinary(uint a, uint b)
{
    if (OP < 6u)
    {
        if (lgIsNaN(a) || lgIsNaN(b)) return OP == 1u ? 1u : 0u;   // NaN: only != holds
        return lgCompare(OP, lgOrderF32(a, b));
    }
    if (OP < 12u)
    {
        int ia = int(a), ib = int(b);
        return lgCompare(OP - 6u, ia < ib ? -1 : (ia > ib ? 1 : 0));
    }
    switch (OP)
    {
    case 12u: return (a != 0u && b != 0u) ? 1u : 0u;
    case 13u: return (a != 0u || b != 0u) ? 1u : 0u;
    case 14u: return ((a != 0u) != (b != 0u)) ? 1u : 0u;
    case 15u: return a + b;
    case 16u: return a - b;
    case 17u: return a * b;
    case 18u: return int(a) < int(b) ? a : b;
    default: return int(a) > int(b) ? a : b;
    }
}

// Logic unary ops: 0 NOT, 1 IS_FINITE, 2 IS_NAN, 3 IS_INF (FP32 classification by bits).
uint lgUnary(uint x)
{
    switch (OP)
    {
    case 0u: return x == 0u ? 1u : 0u;
    case 1u: return (x & 0x7F800000u) != 0x7F800000u ? 1u : 0u;
    case 2u: return lgIsNaN(x) ? 1u : 0u;
    default: return (x & 0x7FFFFFFFu) == 0x7F800000u ? 1u : 0u;
    }
}
