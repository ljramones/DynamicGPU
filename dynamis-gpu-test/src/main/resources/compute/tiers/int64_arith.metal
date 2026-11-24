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
    ulong device* a_0;
    ulong device* b_0;
    ulong device* o_0;
};

[[kernel]] void int64_arith(uint3 id_0 [[thread_position_in_grid]], Params_0 constant* p_1 [[buffer(3)]], ulong device* a_1 [[buffer(0)]], ulong device* b_1 [[buffer(1)]], ulong device* o_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->a_0 = a_1;
    (&kernelContext_0)->b_0 = b_1;
    (&kernelContext_0)->o_0 = o_1;
    uint i_0 = id_0.x;
    if(i_0 >= (p_1->n_0))
    {
        return;
    }
    ulong device* _S1 = (&kernelContext_0)->a_0+i_0;
    ulong device* _S2 = (&kernelContext_0)->b_0+i_0;
    *((&kernelContext_0)->o_0+i_0) = (*_S1 * *_S2 + ((*_S1) >> 7U)) ^ (*_S2);
    return;
}

