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
    ulong device* o_0;
};

[[kernel]] void atomic_int64(uint3 id_0 [[thread_position_in_grid]], Params_0 constant* p_1 [[buffer(1)]], ulong device* o_1 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->o_0 = o_1;
    uint t_0 = id_0.x;
    if(t_0 >= (p_1->n_0))
    {
        return;
    }
    ulong _S1 = ulong(t_0);
    ulong _S2 = atomic_fetch_add_explicit(((atomic_ulong device*)((&kernelContext_0)->o_0+int(0))), _S1 + 1099511627776ULL, memory_order_relaxed);
    ulong _S3 = atomic_fetch_max_explicit(((atomic_ulong device*)((&kernelContext_0)->o_0+int(1))), _S1 << 33U, memory_order_relaxed);
    return;
}

