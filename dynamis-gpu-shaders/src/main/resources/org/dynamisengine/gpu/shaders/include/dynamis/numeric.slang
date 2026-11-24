// Numeric helpers that never depend on fast-math folding: classification by bits.
#pragma once

static const uint DG_EXPONENT_MASK = 0x7F800000u;
static const uint DG_MANTISSA_MASK = 0x007FFFFFu;

bool dg_is_finite(float x) { return (asuint(x) & DG_EXPONENT_MASK) != DG_EXPONENT_MASK; }
bool dg_is_nan(float x) { return (asuint(x) & DG_EXPONENT_MASK) == DG_EXPONENT_MASK && (asuint(x) & DG_MANTISSA_MASK) != 0; }
bool dg_is_inf(float x) { return (asuint(x) & 0x7FFFFFFFu) == DG_EXPONENT_MASK; }

// Negative infinity, for max reductions.
float dg_negative_infinity() { return asfloat(0xFF800000u); }
// Positive infinity, for min reductions.
float dg_positive_infinity() { return asfloat(0x7F800000u); }

// Java's Math.max and Math.min, the same on every device: NaN propagates, and max(-0, +0) = +0, min(-0, +0) = -0.
// Hardware max and min leave NaN and the sign of zero to the device.
float dg_max(float a, float b)
{
    if (dg_is_nan(a)) return a;
    if (dg_is_nan(b)) return b;
    if (a == 0.0f && b == 0.0f) return asfloat(asuint(a) & asuint(b));   // -0 only if both are -0
    return a > b ? a : b;
}
float dg_min(float a, float b)
{
    if (dg_is_nan(a)) return a;
    if (dg_is_nan(b)) return b;
    if (a == 0.0f && b == 0.0f) return asfloat(asuint(a) | asuint(b));   // -0 if either is -0
    return a < b ? a : b;
}
