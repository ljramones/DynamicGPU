// DynamisGPU kernel error buffer (org.dynamisengine.gpu.api.compute.KernelErrors), Slang. Bind the buffer as
// RWStructuredBuffer<uint> and pass it in. reportError(errors, code, index): code is a nonzero, application-defined
// bit mask; the report keeps the OR of the codes, the smallest index and the count, whatever order the threads run in.
void reportError(RWStructuredBuffer<uint> errors, uint code, uint index)
{
    InterlockedOr(errors[0], code);
    InterlockedMax(errors[1], ~index);
    InterlockedAdd(errors[2], 1u);
}
