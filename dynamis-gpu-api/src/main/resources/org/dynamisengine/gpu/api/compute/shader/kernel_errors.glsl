// DynamisGPU kernel error buffer (org.dynamisengine.gpu.api.compute.KernelErrors), GLSL. Declare the buffer first,
// bound read-write, then include this file:
//   layout(set = 0, binding = N) buffer KernelErrorsBlock { uint kernelErrors[4]; };
//   #include "kernel_errors.glsl"   (needs GL_GOOGLE_include_directive)
// reportError(code, index): code is a nonzero, application-defined bit mask; the report keeps the OR of the codes,
// the smallest index and the count, whatever order the threads run in.
void reportError(uint code, uint index) {
    atomicOr(kernelErrors[0], code);
    atomicMax(kernelErrors[1], ~index);
    atomicAdd(kernelErrors[2], 1u);
}
