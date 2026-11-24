// GEMM, tiled kernel: FP16 A and B, FP16 C (gemm_body.slang). Generated with the kernel list.
#define IN_KIND 0
#define OUT_T half
#define ENTRY gemm_f16_h
#include "gemm_body.slang"
