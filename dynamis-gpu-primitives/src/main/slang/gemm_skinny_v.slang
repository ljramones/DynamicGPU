// GEMM, skinny (vector build) kernel: FP16 A and B, FP32 C (gemm_skinny_body.slang). Generated with the kernel list.
#define VEC 1
#define IN_KIND 0
#define OUT_T float
#define ENTRY gemm_skinny_v
#include "gemm_skinny_body.slang"
