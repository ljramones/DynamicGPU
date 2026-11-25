// Cast (1.14.0): element type conversions among FP32, FP16, BF16, INT32 and BOOL (a uint32 word, 0 or 1), in integer
// arithmetic on the bits, so the result is the same on every device (Cast.reference):
// - FP32 -> FP16 and BF16 round to nearest even (Java's Float.floatToFloat16; GemmF16.floatToBf16), overflow to infinity;
// - FP16 and BF16 -> FP32 are exact;
// - FP32 -> INT32 is Java's (int) cast: toward zero, saturating, NaN to 0; INT32 -> FP32 rounds to nearest even;
// - other pairs go through FP32 (FP16 -> BF16 rounds once; INT32 -> FP16 rounds to FP32, then to FP16);
// - to BOOL: nonzero (NaN too) is 1; from BOOL: a nonzero word is 1 (1.0 for the float types).
// FROM and TO (specialization constants) are Cast.Type ordinals: 0 FP32, 1 FP16, 2 BF16, 3 INT32, 4 BOOL. Views count
// elements of their own type; a 16-bit element e of a buffer is half e & 1 of word e >> 1 (little-endian). A 32-bit
// output takes one thread per element; a 16-bit output must be contiguous and takes one thread per 32-bit word it
// touches, which writes both halves (keeping a half outside the view), so no two threads write one word.
#include "dynamis/kit.slang"

[vk::constant_id(0)] const uint FROM = 0;
[vk::constant_id(1)] const uint TO = 0;

struct CastParams
{
    uint count;       // threads: output elements (32-bit output) or output words (16-bit output)
    uint gridX;
    uint xLength;     // buffer lengths in words, for debug builds
    uint yLength;
    uint elements;    // output elements
    uint firstWord;   // a 16-bit output's first word
    DG_VIEW_FIELDS4(x)
    DG_VIEW_FIELDS4(y)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> y;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::push_constant]] ConstantBuffer<CastParams> p;

static const uint THREADS = 256;

bool castIs16(uint t) { return t == 1u || t == 2u; }

uint f32ToF16(uint x)
{
    uint sign = (x >> 16) & 0x8000u;
    uint e = (x >> 23) & 0xFFu;
    uint mant = x & 0x7FFFFFu;
    if (e == 0xFFu) return sign | 0x7C00u | (mant != 0u ? 0x200u | (mant >> 13) : 0u);
    int exp = int(e) - 112;   // rebias 127 -> 15
    if (exp >= 31) return sign | 0x7C00u;
    if (exp <= 0)
    {
        if (exp < -10) return sign;
        mant |= 0x800000u;
        uint shift = uint(14 - exp);
        uint h = mant >> shift;
        uint rem = mant & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (h & 1u) != 0u)) h++;
        return sign | h;
    }
    uint h = (uint(exp) << 10) | (mant >> 13);
    uint rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h++;   // a carry rounds into the exponent, or to infinity
    return sign | h;
}

uint f16ToF32(uint h)
{
    uint sign = (h & 0x8000u) << 16;
    uint e = (h >> 10) & 0x1Fu;
    uint m = h & 0x3FFu;
    if (e == 0x1Fu) return sign | 0x7F800000u | (m << 13);
    if (e == 0u)
    {
        if (m == 0u) return sign;
        uint top = firstbithigh(m);          // 0 .. 9: value m 2^-24 = 1.f 2^(top - 24)
        uint frac = (m << (10u - top)) & 0x3FFu;
        return sign | ((top + 103u) << 23) | (frac << 13);
    }
    return sign | ((e + 112u) << 23) | (m << 13);
}

uint f32ToBf16(uint x)
{
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) return (x >> 16) | 0x40u;
    return (x + 0x7FFFu + ((x >> 16) & 1u)) >> 16;
}

uint f32ToI32(uint x)
{
    uint e = (x >> 23) & 0xFFu;
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) return 0u;          // NaN
    bool neg = (x >> 31) != 0u;
    if (e < 127u) return 0u;                                 // |x| < 1
    if (e >= 158u) return neg ? 0x80000000u : 0x7FFFFFFFu;    // |x| >= 2^31: saturate
    uint m = (x & 0x7FFFFFu) | 0x800000u;
    uint mag = e >= 150u ? m << (e - 150u) : m >> (150u - e);
    return neg ? (0u - mag) : mag;
}

uint i32ToF32(uint v)
{
    if (v == 0u) return 0u;
    uint sign = v & 0x80000000u;
    uint mag = sign != 0u ? 0u - v : v;                      // 2^31 for INT_MIN
    uint top = firstbithigh(mag);
    uint m;
    if (top <= 23u)
    {
        m = mag << (23u - top);
    }
    else
    {
        uint shift = top - 23u;
        m = mag >> shift;
        uint rem = mag & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (m & 1u) != 0u)) m++;
        if (m == 0x1000000u) { m >>= 1; top++; }
    }
    return sign | ((top + 127u) << 23) | (m & 0x7FFFFFu);
}

// The value of an element of type FROM, given its bits, as FP32 bits (exact for every type but INT32).
uint toF32(uint v)
{
    switch (FROM)
    {
    case 0u: return v;
    case 1u: return f16ToF32(v);
    case 2u: return v << 16;
    case 3u: return i32ToF32(v);
    default: return v != 0u ? 0x3F800000u : 0u;
    }
}

bool isNonzero(uint v)
{
    switch (FROM)
    {
    case 0u: return (v & 0x7FFFFFFFu) != 0u;
    case 1u: return (v & 0x7FFFu) != 0u;
    case 2u: return (v & 0x7FFFu) != 0u;
    default: return v != 0u;
    }
}

uint convert(uint v)
{
    if (TO == 4u) return isNonzero(v) ? 1u : 0u;
    if (TO == 3u)
    {
        if (FROM == 3u) return v;
        if (FROM == 4u) return v != 0u ? 1u : 0u;
        return f32ToI32(toF32(v));
    }
    if (TO == FROM) return v;
    uint f = toF32(v);
    switch (TO)
    {
    case 0u: return f;
    case 1u: return f32ToF16(f);
    default: return f32ToBf16(f);
    }
}

// The bits of input element (row-major coordinates c of the output's shape, broadcasting).
uint load(uint4 c, uint at)
{
    uint xi = dg_at_broadcast(DG_VIEW4(p, x), c);
    if (castIs16(FROM))
    {
        if (!dg_in_range(errors, xi >> 1, p.xLength, at)) return 0u;
        return (x[xi >> 1] >> (16u * (xi & 1u))) & 0xFFFFu;
    }
    if (!dg_in_range(errors, xi, p.xLength, at)) return 0u;
    return x[xi];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cast(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint i = dg_linear_thread(group, local, p.gridX, THREADS);
    if (i >= p.count) return;
    DgView yv = DG_VIEW4(p, y);
    if (!castIs16(TO))
    {
        uint4 c = dg_coords(yv, i);
        uint yi = dg_at(yv, c);
        if (!dg_in_range(errors, yi, p.yLength, i)) return;
        y[yi] = convert(load(c, i));
        return;
    }
    uint word = p.firstWord + i;
    if (!dg_in_range(errors, word, p.yLength, i)) return;
    uint merged = y[word];
    [ForceUnroll]
    for (uint h = 0; h < 2u; h++)
    {
        uint slot = 2u * word + h;
        uint e = slot - yv.offset;   // wraps below the view's first element
        if (slot >= yv.offset && e < p.elements)   // a guard, not continue: no early exits in unrolled loops
        {
            uint v = convert(load(dg_coords(yv, e), i)) & 0xFFFFu;
            merged = (merged & ~(0xFFFFu << (16u * h))) | (v << (16u * h));
        }
    }
    y[word] = merged;
}
