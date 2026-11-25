// Quantized GEMM, skinny kernel: BF16 activations, FP32 C (gemm_q_skinny_body.slang).
#define IN_KIND 1
#define OUT_T float
#define ENTRY gemm_q_skinny_bf16
#include "gemm_q_skinny_body.slang"
