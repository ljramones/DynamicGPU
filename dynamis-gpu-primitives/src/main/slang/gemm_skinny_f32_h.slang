// GEMM, skinny kernel: FP32 A and B, FP16 C (gemm_skinny_body.slang). Generated with the kernel list.
#define IN_KIND 2
#define OUT_T half
#define ENTRY gemm_skinny_f32_h
#include "gemm_skinny_body.slang"
