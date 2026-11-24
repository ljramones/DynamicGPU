// GEMM, skinny (vector build) kernel: FP16 A and B, FP16 C (gemm_skinny_body.slang). Generated with the kernel list.
#define VEC 1
#define IN_KIND 0
#define OUT_T half
#define ENTRY gemm_skinny_v_h
#include "gemm_skinny_body.slang"
