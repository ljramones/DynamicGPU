// Elementwise operations shared by ew_unary.slang and ew_binary.slang (Elementwise.java), FP32. OP, a specialization constant, picks the operation, so
// each pipeline is folded for one op. Views have rank 4 (padded with leading unit dimensions); an input dimension of
// extent 1 broadcasts. One thread per output element, linear over a kit.Grid.
#pragma once
#include "dynamis/kit.slang"

[vk::constant_id(0)] const uint OP = 0;

struct UnaryParams
{
    uint count;
    uint gridX;
    uint xLength;   // buffer lengths in elements, for debug builds
    uint yLength;
    DG_VIEW_FIELDS4(x)
    DG_VIEW_FIELDS4(y)
};

struct BinaryParams
{
    uint count;
    uint gridX;
    uint aLength;   // buffer lengths in elements, for debug builds
    uint bLength;
    uint yLength;
    DG_VIEW_FIELDS4(a)
    DG_VIEW_FIELDS4(b)
    DG_VIEW_FIELDS4(y)
};

static const uint THREADS = 256;

// tanh (1.14.0): an odd polynomial below |x| = 0.625 (the Taylor series through x^25, no cancellation), and
// 1 - 2 / (exp(2|x|) + 1) with x's sign above, where the subtraction amplifies exp's error at most 0.8 times.
float ewTanh(float x)
{
    float ax = asfloat(asuint(x) & 0x7FFFFFFFu);
    if (ax < asfloat(0x39800000u)) return x;   // |x| < 2^-12: tanh x rounds to x (keeps -0, and subnormals unflushed)
    if (ax < 0.625f)
    {
        float z = x * x;
        float q = asfloat(0x37858997u);
        q = q * z + asfloat(0xb824bec7u);
        q = q * z + asfloat(0x38cb3f0cu);
        q = q * z + asfloat(0xb97abebcu);
        q = q * z + asfloat(0x3a1aac12u);
        q = q * z + asfloat(0xbabed1b2u);
        q = q * z + asfloat(0x3b6b69e8u);
        q = q * z + asfloat(0xbc11371bu);
        q = q * z + asfloat(0x3cb327a4u);
        q = q * z + asfloat(0xbd5d0dd1u);
        q = q * z + asfloat(0x3e088889u);
        q = q * z + asfloat(0xbeaaaaabu);
        return x + x * (z * q);
    }
    float t = 1.0f - 2.0f / (exp(2.0f * ax) + 1.0f);   // NaN stays NaN; overflow gives 1
    return asfloat(asuint(t) | (asuint(x) & 0x80000000u));
}

// Elementwise.Unary, in declaration order.
float unary(float x)
{
    switch (OP)
    {
    case 0: return x;                                       // COPY
    case 1: return asfloat(asuint(x) ^ 0x80000000u);        // NEG: the sign bit, NaN payloads kept
    case 2: return asfloat(asuint(x) & 0x7FFFFFFFu);        // ABS
    case 3: return x * x;                                   // SQUARE
    case 4: return x < 0.0f ? 0.0f : x;                     // RELU: NaN passes through, -0 stays -0
    case 5: return sqrt(x);                                 // SQRT
    case 6: return exp(x);                                  // EXP
    case 7: return log(x);                                  // LOG
    case 8: return 1.0f / x;                                // RECIPROCAL
    case 9: return x / (1.0f + exp(-x));                    // SILU (1.10.0)
    case 10:                                                // GELU, its tanh form (1.10.0): as GemmF16's epilogue
    {
        float u = 0.7978845608f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + (1.0f - 2.0f / (exp(2.0f * u) + 1.0f)));
    }
    case 11: return 1.0f / (1.0f + exp(-x));                // SIGMOID (1.14.0)
    default: return ewTanh(x);                              // TANH (1.14.0)
    }
}

// Elementwise.Binary, in declaration order.
float binary(float a, float b)
{
    switch (OP)
    {
    case 0: return a + b;                                   // ADD
    case 1: return a - b;                                   // SUB
    case 2: return a * b;                                   // MUL
    case 3: return a / b;                                   // DIV
    case 4: return dg_max(a, b);                            // MAX: Java Math.max
    case 5: return dg_min(a, b);                            // MIN: Java Math.min
    case 6: return a / (1.0f + exp(-a)) * b;                // SWIGLU (1.10.0): silu(a) * b
    // the backward passes of the activations (1.12.0): a = x, b = dy, the result dx
    case 7: return a > 0.0f ? b : 0.0f;                     // RELU_BACKWARD: dy where x > 0, else 0 (NaN x gives 0)
    case 8:                                                 // SILU_BACKWARD: dy s (1 + x (1 - s)), s = sigmoid(x)
    {
        float s = 1.0f / (1.0f + exp(-a));
        return b * (s * (1.0f + a * (1.0f - s)));
    }
    case 9:                                                 // GELU_BACKWARD, the tanh form's derivative
    {
        float u = 0.7978845608f * (a + 0.044715f * a * a * a);
        float t = 1.0f - 2.0f / (exp(2.0f * u) + 1.0f);
        float du = 0.7978845608f * (1.0f + 3.0f * 0.044715f * a * a);
        return b * (0.5f * (1.0f + t) + 0.5f * a * (1.0f - t * t) * du);
    }
    // 1.14.0, from the forward input and without cancellation: both derivatives are even, in e = exp(-|x|) or exp(-2|x|)
    case 10:                                                // SIGMOID_BACKWARD: dy s (1 - s) = dy e / (1 + e)^2
    {
        float e = exp(-asfloat(asuint(a) & 0x7FFFFFFFu));
        return b * (e / ((1.0f + e) * (1.0f + e)));
    }
    default:                                                // TANH_BACKWARD: dy (1 - t^2) = dy 4e / (1 + e)^2
    {
        float e = exp(-2.0f * asfloat(asuint(a) & 0x7FFFFFFFu));
        return b * (4.0f * e / ((1.0f + e) * (1.0f + e)));
    }
    }
}
