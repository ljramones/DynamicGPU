#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant bool fc_DG_DEBUG_0 [[function_constant(100)]];
constant bool DG_DEBUG_0 = is_function_constant_defined(fc_DG_DEBUG_0) ? fc_DG_DEBUG_0 : false;
uint dg_linear_group_0(uint3 group_0, uint gridX_0)
{
    return group_0.y * gridX_0 + group_0.x;
}

uint dg_linear_thread_0(uint3 group_1, uint3 local_0, uint gridX_1, uint groupThreads_0)
{
    return dg_linear_group_0(group_1, gridX_1) * groupThreads_0 + local_0.x;
}

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
    uint gridX_2;
    uint extent_0;
    uint iLength_0;
    uint iOffset_0;
    uint iStride_0;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    uint device* errors_2;
    uint device* indices_0;
    uint device* counts_0;
};

[[kernel]] void scatter_count(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(3)]], uint device* errors_3 [[buffer(2)]], uint device* indices_1 [[buffer(0)]], uint device* counts_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->errors_2 = errors_3;
    (&kernelContext_0)->indices_0 = indices_1;
    (&kernelContext_0)->counts_0 = counts_1;
    uint j_0 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(j_0 >= (p_1->n_0))
    {
        return;
    }
    uint ii_0 = (&kernelContext_0)->p_0->iOffset_0 + j_0 * (&kernelContext_0)->p_0->iStride_0;
    bool _S5 = dg_in_range_0((&kernelContext_0)->errors_2, ii_0, (&kernelContext_0)->p_0->iLength_0, j_0);
    if(!_S5)
    {
        return;
    }
    uint device* _S6 = (&kernelContext_0)->indices_0+ii_0;
    uint k_0 = *_S6;
    if((*_S6) >= ((&kernelContext_0)->p_0->extent_0))
    {
        dg_report_error_0((&kernelContext_0)->errors_2, 1U, j_0);
        return;
    }
    uint _S7 = atomic_fetch_add_explicit(((atomic_uint device*)((&kernelContext_0)->counts_0+k_0)), 1U, memory_order_relaxed);
    return;
}

