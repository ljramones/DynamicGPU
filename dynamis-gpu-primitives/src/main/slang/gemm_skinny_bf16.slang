// GEMM, skinny kernel: BF16 A and B, FP32 C (gemm_skinny_body.slang). Generated with the kernel list.
#define IN_KIND 1
#define OUT_T float
#define ENTRY gemm_skinny_bf16
#include "gemm_skinny_body.slang"
