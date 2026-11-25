// Quantized GEMM, tiled kernel: FP16 activations, FP32 C (gemm_q_tiled_body.slang).
#define IN_KIND 0
#define OUT_T float
#define ENTRY gemm_q_tiled
#include "gemm_q_tiled_body.slang"
