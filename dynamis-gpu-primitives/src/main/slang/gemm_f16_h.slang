// GEMM with FP16 A, B and C (FP32 accumulation, one rounding to half): the tiled kernel in gemm_body.slang.
#define OUT_T half
#define ENTRY gemm_f16_h
#include "gemm_body.slang"
