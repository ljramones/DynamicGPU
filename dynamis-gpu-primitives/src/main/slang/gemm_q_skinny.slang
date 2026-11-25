// Quantized GEMM, skinny kernel: FP16 activations, FP32 C (gemm_q_skinny_body.slang).
#define IN_KIND 0
#define OUT_T float
#define ENTRY gemm_q_skinny
#include "gemm_q_skinny_body.slang"
