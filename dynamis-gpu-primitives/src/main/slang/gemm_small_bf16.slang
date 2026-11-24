// GEMM, small kernel: BF16 A and B, FP32 C (gemm_small_body.slang). Generated with the kernel list.
#define IN_KIND 1
#define OUT_T float
#define ENTRY gemm_small_bf16
#include "gemm_small_body.slang"
