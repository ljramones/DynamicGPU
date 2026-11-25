// Quantized GEMM, tiled kernel: BF16 activations, FP16 C (gemm_q_tiled_body.slang).
#define IN_KIND 1
#define OUT_T half
#define ENTRY gemm_q_tiled_bf16_h
#include "gemm_q_tiled_body.slang"
