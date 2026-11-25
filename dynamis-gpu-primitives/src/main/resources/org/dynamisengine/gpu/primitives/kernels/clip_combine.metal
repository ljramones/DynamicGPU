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

float dg_op_add_0(float a_0, float b_0)
{
    return a_0 + b_0;
}

struct Params_0
{
    uint n_0;
    uint pLength_0;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    uint device* errors_2;
    float device* partial_0;
    float device* out_0;
    array<float, int(256)> threadgroup* red_0;
};

[[kernel]] void clip_combine(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(3)]], uint device* errors_3 [[buffer(2)]], float device* partial_1 [[buffer(0)]], float device* out_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->errors_2 = errors_3;
    (&kernelContext_0)->partial_0 = partial_1;
    (&kernelContext_0)->out_0 = out_1;
    threadgroup array<float, int(256)> red_1;
    (&kernelContext_0)->red_0 = &red_1;
    uint t_0 = local_0.x;
    uint i_0 = t_0;
    float acc_0 = 0.0;
    for(;;)
    {
        if(i_0 < ((&kernelContext_0)->p_0->n_0))
        {
        }
        else
        {
            break;
        }
        bool _S5 = dg_in_range_0((&kernelContext_0)->errors_2, i_0, (&kernelContext_0)->p_0->pLength_0, 0U);
        if(_S5)
        {
            acc_0 = acc_0 + *((&kernelContext_0)->partial_0+i_0);
        }
        i_0 = i_0 + 256U;
    }
    (*(&kernelContext_0)->red_0)[t_0] = acc_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 128U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 128U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 64U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 32U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 16U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 8U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 4U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 2U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 1U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 == 0U)
    {
        *((&kernelContext_0)->out_0+int(0)) = (*(&kernelContext_0)->red_0)[int(0)];
    }
    return;
}

