#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
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

struct CeParams_0
{
    uint rows_0;
    uint vocab_0;
    uint ld_0;
    uint lddl_0;
    float scale_0;
    uint gridX_2;
    uint inPlace_0;
    uint grad_0;
};

struct KernelContext_0
{
    CeParams_0 constant* p_0;
    uint device* targets_0;
    float device* loss_0;
    uint device* errors_1;
    float device* lse_0;
    float device* logits_0;
    float device* dl_0;
};

[[kernel]] void cross_entropy(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], CeParams_0 constant* p_1 [[buffer(6)]], uint device* targets_1 [[buffer(1)]], float device* loss_1 [[buffer(3)]], uint device* errors_2 [[buffer(5)]], float device* lse_1 [[buffer(2)]], float device* logits_1 [[buffer(0)]], float device* dl_1 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->targets_0 = targets_1;
    (&kernelContext_0)->loss_0 = loss_1;
    (&kernelContext_0)->errors_1 = errors_2;
    (&kernelContext_0)->lse_0 = lse_1;
    (&kernelContext_0)->logits_0 = logits_1;
    (&kernelContext_0)->dl_0 = dl_1;
    uint id_0 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if((p_1->grad_0) == 0U)
    {
        if(id_0 >= ((&kernelContext_0)->p_0->rows_0))
        {
            return;
        }
        uint device* _S4 = (&kernelContext_0)->targets_0+id_0;
        uint t_0 = *_S4;
        if((*_S4) == 4294967295U)
        {
            *((&kernelContext_0)->loss_0+id_0) = 0.0;
            return;
        }
        if(t_0 >= ((&kernelContext_0)->p_0->vocab_0))
        {
            dg_report_error_0((&kernelContext_0)->errors_1, 1U, id_0);
            return;
        }
        *((&kernelContext_0)->loss_0+id_0) = *((&kernelContext_0)->lse_0+id_0) - *((&kernelContext_0)->logits_0+(id_0 * (&kernelContext_0)->p_0->ld_0 + t_0));
        return;
    }
    if(id_0 >= ((&kernelContext_0)->p_0->rows_0 * (&kernelContext_0)->p_0->vocab_0))
    {
        return;
    }
    uint r_0 = id_0 / (&kernelContext_0)->p_0->vocab_0;
    uint j_0 = id_0 % (&kernelContext_0)->p_0->vocab_0;
    uint device* _S5 = (&kernelContext_0)->targets_0+r_0;
    uint t_1 = *_S5;
    bool ignored_0 = (*_S5) == 4294967295U;
    bool _S6;
    if(!ignored_0)
    {
        _S6 = t_1 >= ((&kernelContext_0)->p_0->vocab_0);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        if(j_0 == 0U)
        {
            dg_report_error_0((&kernelContext_0)->errors_1, 1U, r_0);
        }
        return;
    }
    uint at_0 = r_0 * (&kernelContext_0)->p_0->lddl_0 + j_0;
    float l_0;
    if(((&kernelContext_0)->p_0->inPlace_0) != 0U)
    {
        l_0 = *((&kernelContext_0)->dl_0+at_0);
    }
    else
    {
        l_0 = *((&kernelContext_0)->logits_0+(r_0 * (&kernelContext_0)->p_0->ld_0 + j_0));
    }
    if(ignored_0)
    {
        if(j_0 == 0U)
        {
            *((&kernelContext_0)->loss_0+r_0) = 0.0;
        }
        *((&kernelContext_0)->dl_0+at_0) = 0.0;
        return;
    }
    float m_0 = *((&kernelContext_0)->lse_0+r_0);
    bool _S7 = j_0 == t_1;
    if(_S7)
    {
        *((&kernelContext_0)->loss_0+r_0) = m_0 - l_0;
    }
    float device* _S8 = (&kernelContext_0)->dl_0+at_0;
    float _S9 = (&kernelContext_0)->p_0->scale_0;
    float _S10 = exp(l_0 - m_0);
    if(_S7)
    {
        l_0 = 1.0;
    }
    else
    {
        l_0 = 0.0;
    }
    *_S8 = _S9 * (_S10 - l_0);
    return;
}

