// The skinny GEMM with an FP16 C (FP32 accumulation, one rounding to half): gemm_skinny_body.slang.
#define OUT_T half
#define ENTRY gemm_skinny_h
#include "gemm_skinny_body.slang"
