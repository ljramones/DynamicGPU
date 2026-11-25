// Quantized GEMM, tiled kernel: BF16 activations, FP32 C (gemm_q_tiled_body.slang).
#define IN_KIND 1
#define OUT_T float
#define ENTRY gemm_q_tiled_bf16
#include "gemm_q_tiled_body.slang"
