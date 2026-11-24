// DynamisGPU kernel error buffer (org.dynamisengine.gpu.api.compute.KernelErrors), Metal. Bind the buffer as
// `device atomic_uint *errors`. reportError(errors, code, index): code is a nonzero, application-defined bit mask;
// the report keeps the OR of the codes, the smallest index and the count, whatever order the threads run in.
#include <metal_stdlib>
inline void reportError(device metal::atomic_uint *errors, uint code, uint index) {
    metal::atomic_fetch_or_explicit(&errors[0], code, metal::memory_order_relaxed);
    metal::atomic_fetch_max_explicit(&errors[1], ~index, metal::memory_order_relaxed);
    metal::atomic_fetch_add_explicit(&errors[2], 1u, metal::memory_order_relaxed);
}
