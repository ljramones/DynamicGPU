// GEMM with FP16 A and B and an FP32 C: the tiled kernel in gemm_body.slang. Built by scripts/build-slang-kernels.sh.
#define OUT_T float
#define ENTRY gemm_f16
#include "gemm_body.slang"
