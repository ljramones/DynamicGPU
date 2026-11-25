// Quantized GEMM, skinny kernel: FP16 activations, FP16 C (gemm_q_skinny_body.slang).
#define IN_KIND 0
#define OUT_T half
#define ENTRY gemm_q_skinny_h
#include "gemm_q_skinny_body.slang"
