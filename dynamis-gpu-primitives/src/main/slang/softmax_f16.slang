// Row-wise softmax of FP32 scores into FP16 (one rounding): softmax_body.slang.
#define OUT_T half
#define ENTRY softmax_f16
#include "softmax_body.slang"
