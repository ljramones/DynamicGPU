// GEMM, small kernel: FP32 A and B, FP16 C (gemm_small_body.slang). Generated with the kernel list.
#define IN_KIND 2
#define OUT_T half
#define ENTRY gemm_small_f32_h
#include "gemm_small_body.slang"
