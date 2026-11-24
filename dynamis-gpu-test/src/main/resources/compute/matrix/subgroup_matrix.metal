// SUBGROUP_MATRIX_F16 conformance (hand-written: Slang cannot emit Metal SIMD-group matrices). One SIMD-group
// multiplies 8 x 8 half matrices into a float accumulator: c = a * b + c, all row-major. With small integers the
// result is exact in any summation order.
#include <metal_stdlib>
using namespace metal;
kernel void subgroup_matrix(device const half *a [[buffer(0)]], device const half *b [[buffer(1)]],
                            device float *c [[buffer(2)]], constant uint &n [[buffer(3)]]) {
    simdgroup_half8x8 ma, mb; simdgroup_float8x8 mc;
    simdgroup_load(ma, a, 8); simdgroup_load(mb, b, 8); simdgroup_load(mc, c, 8);
    simdgroup_multiply_accumulate(mc, ma, mb, mc);
    simdgroup_store(mc, c, 8);
}
