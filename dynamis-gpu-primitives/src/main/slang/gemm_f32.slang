// GEMM, tiled kernel: FP32 A and B, FP32 C (gemm_body.slang). Generated with the kernel list.
#define IN_KIND 2
#define OUT_T float
#define ENTRY gemm_f32
#include "gemm_body.slang"
