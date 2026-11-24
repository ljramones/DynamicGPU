// The small GEMM with an FP16 C (FP32 accumulation, one rounding to half): gemm_small_body.slang.
#define OUT_T half
#define ENTRY gemm_small_h
#include "gemm_small_body.slang"
