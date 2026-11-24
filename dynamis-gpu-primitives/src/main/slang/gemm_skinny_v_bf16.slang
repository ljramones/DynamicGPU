// GEMM, skinny (vector build) kernel: BF16 A and B, FP32 C (gemm_skinny_body.slang). Generated with the kernel list.
#define VEC 1
#define IN_KIND 1
#define OUT_T float
#define ENTRY gemm_skinny_v_bf16
#include "gemm_skinny_body.slang"
