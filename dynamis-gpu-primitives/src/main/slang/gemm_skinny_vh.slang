// The skinny GEMM, vector build (FP16 C): gemm_skinny_body.slang.
#define VEC 1
#define OUT_T half
#define ENTRY gemm_skinny_vh
#include "gemm_skinny_body.slang"
