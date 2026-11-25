#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct SoftmaxParams_0
{
    uint m_0;
    uint n_0;
    uint lds_0;
    uint ldy_0;
    uint causal_0;
    uint causalOffset_0;
    uint strideS_0;
    uint strideY_0;
};

struct KernelContext_0
{
    SoftmaxParams_0 constant* p_0;
    float device* dy_0;
    float device* y_0;
    float device* dx_0;
    array<float, int(256)> threadgroup* red_0;
    float threadgroup* rowSum_0;
};

[[kernel]] void softmax_backward_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], SoftmaxParams_0 constant* p_1 [[buffer(3)]], float device* dy_1 [[buffer(1)]], float device* y_1 [[buffer(0)]], float device* dx_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->dy_0 = dy_1;
    (&kernelContext_0)->y_0 = y_1;
    (&kernelContext_0)->dx_0 = dx_1;
    threadgroup array<float, int(256)> red_1;
    (&kernelContext_0)->red_0 = &red_1;
    threadgroup float rowSum_1;
    (&kernelContext_0)->rowSum_0 = &rowSum_1;
    uint t_0 = local_0.x;
    uint r_0 = group_0.x;
    uint _S1 = group_0.z;
    uint _S2 = _S1 * p_1->strideS_0 + r_0 * p_1->lds_0;
    uint _S3 = _S1 * p_1->strideY_0 + r_0 * p_1->ldy_0;
    uint _S4;
    if((p_1->causal_0) != 0U)
    {
        _S4 = min((&kernelContext_0)->p_0->n_0, r_0 + (&kernelContext_0)->p_0->causalOffset_0 + 1U);
    }
    else
    {
        _S4 = (&kernelContext_0)->p_0->n_0;
    }
    uint j_0 = t_0;
    float acc_0 = 0.0;
    for(;;)
    {
        if(j_0 < _S4)
        {
        }
        else
        {
            break;
        }
        uint _S5 = _S3 + j_0;
        float _S6 = acc_0 + *((&kernelContext_0)->dy_0+_S5) * *((&kernelContext_0)->y_0+_S5);
        j_0 = j_0 + 256U;
        acc_0 = _S6;
    }
    (*(&kernelContext_0)->red_0)[t_0] = acc_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 128U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = (*(&kernelContext_0)->red_0)[t_0] + (*(&kernelContext_0)->red_0)[t_0 + 128U];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 64U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = (*(&kernelContext_0)->red_0)[t_0] + (*(&kernelContext_0)->red_0)[t_0 + 64U];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 32U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = (*(&kernelContext_0)->red_0)[t_0] + (*(&kernelContext_0)->red_0)[t_0 + 32U];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 16U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = (*(&kernelContext_0)->red_0)[t_0] + (*(&kernelContext_0)->red_0)[t_0 + 16U];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 8U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = (*(&kernelContext_0)->red_0)[t_0] + (*(&kernelContext_0)->red_0)[t_0 + 8U];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 4U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = (*(&kernelContext_0)->red_0)[t_0] + (*(&kernelContext_0)->red_0)[t_0 + 4U];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 2U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = (*(&kernelContext_0)->red_0)[t_0] + (*(&kernelContext_0)->red_0)[t_0 + 2U];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 1U)
    {
        (*(&kernelContext_0)->red_0)[t_0] = (*(&kernelContext_0)->red_0)[t_0] + (*(&kernelContext_0)->red_0)[t_0 + 1U];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 == 0U)
    {
        *(&kernelContext_0)->rowSum_0 = (*(&kernelContext_0)->red_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S7 = *(&kernelContext_0)->rowSum_0;
    j_0 = t_0;
    for(;;)
    {
        if(j_0 < ((&kernelContext_0)->p_0->n_0))
        {
        }
        else
        {
            break;
        }
        float device* _S8 = (&kernelContext_0)->dx_0+(_S2 + j_0);
        if(j_0 < _S4)
        {
            uint _S9 = _S3 + j_0;
            acc_0 = *((&kernelContext_0)->y_0+_S9) * (*((&kernelContext_0)->dy_0+_S9) - _S7);
        }
        else
        {
            acc_0 = 0.0;
        }
        *_S8 = acc_0;
        j_0 = j_0 + 256U;
    }
    return;
}

