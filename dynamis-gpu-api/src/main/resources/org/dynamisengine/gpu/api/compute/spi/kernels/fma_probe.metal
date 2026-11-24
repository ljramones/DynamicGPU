#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct FmaProbeParams_0
{
    float x_0;
    float c_0;
};

struct KernelContext_0
{
    float device* o_0;
    FmaProbeParams_0 constant* p_0;
};

[[kernel]] void fma_probe(uint3 id_0 [[thread_position_in_grid]], float device* o_1 [[buffer(0)]], FmaProbeParams_0 constant* p_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->o_0 = o_1;
    (&kernelContext_0)->p_0 = p_1;
    if((id_0.x) == 0U)
    {
        *((&kernelContext_0)->o_0+int(0)) = fma((&kernelContext_0)->p_0->x_0, (&kernelContext_0)->p_0->x_0, (&kernelContext_0)->p_0->c_0);
    }
    return;
}

