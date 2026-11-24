// GEMM, grouped kernel: BF16 A and B, FP16 C (gemm_grouped_body.slang). Generated with the kernel list.
#define IN_KIND 1
#define OUT_T half
#define ENTRY gemm_grouped_bf16_h
#include "gemm_grouped_body.slang"
