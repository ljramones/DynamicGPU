#include <metal_stdlib>
using namespace metal;
// IEEE floating-point checks for the conformance suite; see fp_strict.comp for the four outputs. No pragmas or
// precise marks: the runtime compiler's defaults must round x*y + z twice, keep (x + c) - c, and see NaN.
kernel void fp_strict(device const float *v [[buffer(0)]], device float *o [[buffer(1)]],
                      constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float x = v[3 * i];
    float y = v[3 * i + 1];
    float z = v[3 * i + 2];
    o[4 * i] = (x + 12582912.0f) - 12582912.0f;
    o[4 * i + 1] = (x + 12582912.0f) - 12582912.0f;
    o[4 * i + 2] = x * y + z;
    o[4 * i + 3] = isnan(x) ? 1.0f : 0.0f;
}
