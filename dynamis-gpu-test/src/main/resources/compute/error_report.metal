// Conformance kernel for kernel error buffers; see error_report.comp. The test prepends kernel_errors.metal.
kernel void error_report(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                         device metal::atomic_uint *errors [[buffer(2)]], constant uint &count [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float v = x[i];
    if (metal::isnan(v)) reportError(errors, 8u, i);
    else if (v < 0.0f) reportError(errors, 1u << (i % 3u), i);
    y[i] = v;
}
