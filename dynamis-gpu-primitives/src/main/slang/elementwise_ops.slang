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
    default:                                                // GELU, its tanh form (1.10.0): as GemmF16's epilogue
    {
        float u = 0.7978845608f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + (1.0f - 2.0f / (exp(2.0f * u) + 1.0f)));
    }
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
    default: return a / (1.0f + exp(-a)) * b;               // SWIGLU (1.10.0): silu(a) * b
    }
}
