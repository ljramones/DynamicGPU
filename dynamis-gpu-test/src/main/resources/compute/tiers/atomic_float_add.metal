#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct Params_0
{
    uint n_0;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    float device* o_0;
};

[[kernel]] void atomic_float_add(uint3 id_0 [[thread_position_in_grid]], Params_0 constant* p_1 [[buffer(1)]], float device* o_1 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->o_0 = o_1;
    uint t_0 = id_0.x;
    if(t_0 >= (p_1->n_0))
    {
        return;
    }
    float _S1 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->o_0+int(0))), 1.0, memory_order_relaxed);
    float _S2 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->o_0+(1U + t_0 % 7U))), 0.25, memory_order_relaxed);
    return;
}

