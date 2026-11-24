// A and B of the tiled, small and grouped GEMM kernels by input type (1.8.0), declared first (Slang numbers Metal
// buffers in declaration order), with loadA and loadB reading element e as FP32. IN_KIND: 0 FP16 (the default),
// 1 BF16 (two per 32-bit word, the even element in the low half, widened exactly by a shift), 2 FP32.
// FP16 and BF16 products are exact in FP32; FP32 products are not, so fma's single rounding matters there.
#ifndef IN_KIND
#define IN_KIND 0
#endif
#if IN_KIND == 2
[[vk::binding(0, 0)]] RWStructuredBuffer<float> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> b;
float loadA(uint e) { return a[e]; }
float loadB(uint e) { return b[e]; }
#elif IN_KIND == 1
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> b;
float bf16At(uint w, uint e) { return asfloat((e & 1) != 0 ? (w & 0xFFFF0000u) : (w << 16)); }
float loadA(uint e) { return bf16At(a[e >> 1], e); }
float loadB(uint e) { return bf16At(b[e >> 1], e); }
#else
[[vk::binding(0, 0)]] RWStructuredBuffer<half> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<half> b;
float loadA(uint e) { return float(a[e]); }
float loadB(uint e) { return float(b[e]); }
#endif
