#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant float fc_SCALE_0 [[function_constant(1)]];
constant float SCALE_0 = is_function_constant_defined(fc_SCALE_0) ? fc_SCALE_0 : 1.0;
constant uint fc_OFFSET_0 [[function_constant(0)]];
constant uint OFFSET_0 = is_function_constant_defined(fc_OFFSET_0) ? fc_OFFSET_0 : 0U;
constant int fc_BIAS_0 [[function_constant(3)]];
constant int BIAS_0 = is_function_constant_defined(fc_BIAS_0) ? fc_BIAS_0 : int(0);
constant bool fc_NEGATE_0 [[function_constant(2)]];
constant bool NEGATE_0 = is_function_constant_defined(fc_NEGATE_0) ? fc_NEGATE_0 : false;
struct Params_0
{
    uint n_0;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    float device* a_0;
    float device* o_0;
    array<float, int(64)> threadgroup* stage_0;
};

[[kernel]] void specialize(uint3 id_0 [[thread_position_in_grid]], uint3 lt_0 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(2)]], float device* a_1 [[buffer(0)]], float device* o_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->a_0 = a_1;
    (&kernelContext_0)->o_0 = o_1;
    threadgroup array<float, int(64)> stage_1;
    (&kernelContext_0)->stage_0 = &stage_1;
    uint i_0 = id_0.x;
    float _S1;
    if(i_0 < (p_1->n_0))
    {
        _S1 = *((&kernelContext_0)->a_0+i_0);
    }
    else
    {
        _S1 = 0.0;
    }
    stage_1[lt_0.x] = _S1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(i_0 >= (p_1->n_0))
    {
        return;
    }
    float v_0 = stage_1[lt_0.x] * SCALE_0 + float(OFFSET_0) + float(BIAS_0);
    float device* _S2 = (&kernelContext_0)->o_0+i_0;
    if(NEGATE_0)
    {
        _S1 = - v_0;
    }
    else
    {
        _S1 = v_0;
    }
    *_S2 = _S1;
    return;
}

