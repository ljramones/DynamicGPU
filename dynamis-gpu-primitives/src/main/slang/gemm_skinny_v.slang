// The skinny GEMM, vector build (FP32 C): gemm_skinny_body.slang.
#define VEC 1
#define OUT_T float
#define ENTRY gemm_skinny_v
#include "gemm_skinny_body.slang"
