#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant bool fc_DG_DEBUG_0 [[function_constant(100)]];
constant bool DG_DEBUG_0 = is_function_constant_defined(fc_DG_DEBUG_0) ? fc_DG_DEBUG_0 : false;
void dg_report_error_0(uint device* errors_0, uint code_0, uint index_0)
{
    uint _S1 = atomic_fetch_or_explicit(((atomic_uint device*)(errors_0+int(0))), code_0, memory_order_relaxed);
    uint _S2 = atomic_fetch_max_explicit(((atomic_uint device*)(errors_0+int(1))), ~index_0, memory_order_relaxed);
    uint _S3 = atomic_fetch_add_explicit(((atomic_uint device*)(errors_0+int(2))), 1U, memory_order_relaxed);
    return;
}

bool dg_in_range_0(uint device* errors_1, uint index_1, uint length_0, uint at_0)
{
    bool _S4;
    if(!DG_DEBUG_0)
    {
        _S4 = true;
    }
    else
    {
        _S4 = index_1 < length_0;
    }
    if(_S4)
    {
        return true;
    }
    dg_report_error_0(errors_1, 1U, at_0);
    return false;
}

struct Params_0
{
    uint n_0;
    uint shift_0;
    uint oLength_0;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    uint device* errors_2;
    float device* o_0;
    float device* a_0;
};

[[kernel]] void debug_overrun(uint3 id_0 [[thread_position_in_grid]], Params_0 constant* p_1 [[buffer(3)]], uint device* errors_3 [[buffer(2)]], float device* o_1 [[buffer(1)]], float device* a_1 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->errors_2 = errors_3;
    (&kernelContext_0)->o_0 = o_1;
    (&kernelContext_0)->a_0 = a_1;
    uint i_0 = id_0.x;
    if(i_0 >= (p_1->n_0))
    {
        return;
    }
    uint at_1 = i_0 + (&kernelContext_0)->p_0->shift_0;
    bool _S5 = dg_in_range_0((&kernelContext_0)->errors_2, at_1, (&kernelContext_0)->p_0->oLength_0, i_0);
    if(!_S5)
    {
        return;
    }
    *((&kernelContext_0)->o_0+at_1) = *((&kernelContext_0)->a_0+i_0);
    return;
}

