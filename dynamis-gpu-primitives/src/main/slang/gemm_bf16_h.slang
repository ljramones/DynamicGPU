// GEMM, tiled kernel: BF16 A and B, FP16 C (gemm_body.slang). Generated with the kernel list.
#define IN_KIND 1
#define OUT_T half
#define ENTRY gemm_bf16_h
#include "gemm_body.slang"
