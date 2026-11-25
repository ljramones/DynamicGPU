// Quantized GEMM, tiled kernel: FP16 activations, FP16 C (gemm_q_tiled_body.slang).
#define IN_KIND 0
#define OUT_T half
#define ENTRY gemm_q_tiled_h
#include "gemm_q_tiled_body.slang"
