// GEMM, grouped kernel: BF16 A and B, FP32 C (gemm_grouped_body.slang). Generated with the kernel list.
#define IN_KIND 1
#define OUT_T float
#define ENTRY gemm_grouped_bf16
#include "gemm_grouped_body.slang"
