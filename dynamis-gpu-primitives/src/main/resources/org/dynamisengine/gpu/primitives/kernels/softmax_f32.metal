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
    float device* s_0;
    float device* y_0;
    array<float, int(256)> threadgroup* red_0;
    float threadgroup* rowMax_0;
    float threadgroup* rowSum_0;
};

void reduce_0(uint t_0, bool isMax_0, KernelContext_0 thread* kernelContext_0)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S1;
    if(t_0 < 128U)
    {
        float a_0 = (*kernelContext_0->red_0)[t_0];
        float b_0 = (*kernelContext_0->red_0)[t_0 + 128U];
        if(isMax_0)
        {
            _S1 = max(a_0, b_0);
        }
        else
        {
            _S1 = a_0 + b_0;
        }
        (*kernelContext_0->red_0)[t_0] = _S1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 64U)
    {
        float a_1 = (*kernelContext_0->red_0)[t_0];
        float b_1 = (*kernelContext_0->red_0)[t_0 + 64U];
        if(isMax_0)
        {
            _S1 = max(a_1, b_1);
        }
        else
        {
            _S1 = a_1 + b_1;
        }
        (*kernelContext_0->red_0)[t_0] = _S1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 32U)
    {
        float a_2 = (*kernelContext_0->red_0)[t_0];
        float b_2 = (*kernelContext_0->red_0)[t_0 + 32U];
        if(isMax_0)
        {
            _S1 = max(a_2, b_2);
        }
        else
        {
            _S1 = a_2 + b_2;
        }
        (*kernelContext_0->red_0)[t_0] = _S1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 16U)
    {
        float a_3 = (*kernelContext_0->red_0)[t_0];
        float b_3 = (*kernelContext_0->red_0)[t_0 + 16U];
        if(isMax_0)
        {
            _S1 = max(a_3, b_3);
        }
        else
        {
            _S1 = a_3 + b_3;
        }
        (*kernelContext_0->red_0)[t_0] = _S1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 8U)
    {
        float a_4 = (*kernelContext_0->red_0)[t_0];
        float b_4 = (*kernelContext_0->red_0)[t_0 + 8U];
        if(isMax_0)
        {
            _S1 = max(a_4, b_4);
        }
        else
        {
            _S1 = a_4 + b_4;
        }
        (*kernelContext_0->red_0)[t_0] = _S1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 4U)
    {
        float a_5 = (*kernelContext_0->red_0)[t_0];
        float b_5 = (*kernelContext_0->red_0)[t_0 + 4U];
        if(isMax_0)
        {
            _S1 = max(a_5, b_5);
        }
        else
        {
            _S1 = a_5 + b_5;
        }
        (*kernelContext_0->red_0)[t_0] = _S1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 2U)
    {
        float a_6 = (*kernelContext_0->red_0)[t_0];
        float b_6 = (*kernelContext_0->red_0)[t_0 + 2U];
        if(isMax_0)
        {
            _S1 = max(a_6, b_6);
        }
        else
        {
            _S1 = a_6 + b_6;
        }
        (*kernelContext_0->red_0)[t_0] = _S1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 1U)
    {
        float a_7 = (*kernelContext_0->red_0)[t_0];
        float b_7 = (*kernelContext_0->red_0)[t_0 + 1U];
        if(isMax_0)
        {
            _S1 = max(a_7, b_7);
        }
        else
        {
            _S1 = a_7 + b_7;
        }
        (*kernelContext_0->red_0)[t_0] = _S1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
}

[[kernel]] void softmax_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], SoftmaxParams_0 constant* p_1 [[buffer(2)]], float device* s_1 [[buffer(0)]], float device* y_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->p_0 = p_1;
    (&kernelContext_1)->s_0 = s_1;
    (&kernelContext_1)->y_0 = y_1;
    threadgroup array<float, int(256)> red_1;
    (&kernelContext_1)->red_0 = &red_1;
    threadgroup float rowMax_1;
    (&kernelContext_1)->rowMax_0 = &rowMax_1;
    threadgroup float rowSum_1;
    (&kernelContext_1)->rowSum_0 = &rowSum_1;
    uint t_1 = local_0.x;
    uint r_0 = group_0.x;
    uint _S2 = group_0.z;
    uint _S3 = _S2 * p_1->strideS_0 + r_0 * p_1->lds_0;
    uint _S4 = _S2 * p_1->strideY_0 + r_0 * p_1->ldy_0;
    uint _S5;
    if((p_1->causal_0) != 0U)
    {
        _S5 = min((&kernelContext_1)->p_0->n_0, r_0 + (&kernelContext_1)->p_0->causalOffset_0 + 1U);
    }
    else
    {
        _S5 = (&kernelContext_1)->p_0->n_0;
    }
    float _S6 = (as_type<float>((4286578688U)));
    uint j_0 = t_1;
    float local_1 = _S6;
    for(;;)
    {
        if(j_0 < _S5)
        {
        }
        else
        {
            break;
        }
        float _S7 = max(local_1, *((&kernelContext_1)->s_0+(_S3 + j_0)));
        j_0 = j_0 + 256U;
        local_1 = _S7;
    }
    (*(&kernelContext_1)->red_0)[t_1] = local_1;
    reduce_0(t_1, true, &kernelContext_1);
    bool _S8 = t_1 == 0U;
    if(_S8)
    {
        *(&kernelContext_1)->rowMax_0 = (*(&kernelContext_1)->red_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S9 = *(&kernelContext_1)->rowMax_0;
    j_0 = t_1;
    local_1 = 0.0;
    for(;;)
    {
        if(j_0 < _S5)
        {
        }
        else
        {
            break;
        }
        float local_2 = local_1 + exp(*((&kernelContext_1)->s_0+(_S3 + j_0)) - _S9);
        j_0 = j_0 + 256U;
        local_1 = local_2;
    }
    (*(&kernelContext_1)->red_0)[t_1] = local_1;
    reduce_0(t_1, false, &kernelContext_1);
    if(_S8)
    {
        *(&kernelContext_1)->rowSum_0 = (*(&kernelContext_1)->red_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threadgroup* _S10 = (&kernelContext_1)->rowSum_0;
    float sum_0 = *_S10;
    if((*_S10) > 0.0)
    {
        local_1 = 1.0 / sum_0;
    }
    else
    {
        local_1 = 0.0;
    }
    j_0 = t_1;
    for(;;)
    {
        if(j_0 < ((&kernelContext_1)->p_0->n_0))
        {
        }
        else
        {
            break;
        }
        float device* _S11 = (&kernelContext_1)->y_0+(_S4 + j_0);
        float _S12;
        if(j_0 < _S5)
        {
            _S12 = exp(*((&kernelContext_1)->s_0+(_S3 + j_0)) - _S9) * local_1;
        }
        else
        {
            _S12 = 0.0;
        }
        *_S11 = _S12;
        j_0 = j_0 + 256U;
    }
    return;
}

