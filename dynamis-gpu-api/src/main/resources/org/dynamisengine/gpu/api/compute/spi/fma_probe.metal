#include <metal_stdlib>
using namespace metal;
// Whether fma is fused; see fma_probe.comp.
struct FmaProbeParameters { float x; float c; };
kernel void fma_probe(device float *o [[buffer(0)]], constant FmaProbeParameters &p [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    if (i == 0) {
        o[0] = fma(p.x, p.x, p.c);
    }
}
