// The grouped GEMM with an FP16 C (FP32 accumulation, one rounding to half): gemm_grouped_body.slang.
#define OUT_T half
#define ENTRY gemm_grouped_h
#include "gemm_grouped_body.slang"
