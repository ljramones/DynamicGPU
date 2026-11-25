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

struct OptimParams_0
{
    uint n_0;
    uint gridX_2;
    uint kind_0;
    uint scaled_0;
    uint shadow_0;
    uint nesterov_0;
    float negLr_0;
    float beta1_0;
    float beta2_0;
    float eps_0;
    float bc1_0;
    float bc2_0;
    float a1_0;
    float a2_0;
    float decay_0;
};

struct KernelContext_0
{
    OptimParams_0 constant* p_0;
    half device* grad_0;
    float device* scale_0;
    float device* param_0;
    float device* m_0;
    float device* v_0;
    uint device* shadowOut_0;
};

float step_0(uint i_0, KernelContext_0 thread* kernelContext_0)
{
    float g_0 = float(*(kernelContext_0->grad_0+i_0));
    float g_1;
    if((kernelContext_0->p_0->scaled_0) != 0U)
    {
        g_1 = g_0 * *(kernelContext_0->scale_0+int(0));
    }
    else
    {
        g_1 = g_0;
    }
    float x_0 = *(kernelContext_0->param_0+i_0);
    float x_1;
    if((kernelContext_0->p_0->kind_0) == 0U)
    {
        float mi_0 = kernelContext_0->p_0->beta1_0 * *(kernelContext_0->m_0+i_0) + kernelContext_0->p_0->a1_0 * g_1;
        float vi_0 = kernelContext_0->p_0->beta2_0 * *(kernelContext_0->v_0+i_0) + kernelContext_0->p_0->a2_0 * g_1 * g_1;
        *(kernelContext_0->m_0+i_0) = mi_0;
        *(kernelContext_0->v_0+i_0) = vi_0;
        x_1 = x_0 * kernelContext_0->p_0->decay_0 + kernelContext_0->p_0->negLr_0 * (mi_0 / kernelContext_0->p_0->bc1_0 / (sqrt(vi_0 / kernelContext_0->p_0->bc2_0) + kernelContext_0->p_0->eps_0));
    }
    else
    {
        if((kernelContext_0->p_0->decay_0) != 0.0)
        {
            g_1 = g_1 + kernelContext_0->p_0->decay_0 * x_0;
        }
        float b_0 = kernelContext_0->p_0->beta1_0 * *(kernelContext_0->m_0+i_0) + g_1;
        *(kernelContext_0->m_0+i_0) = b_0;
        float _S1 = kernelContext_0->p_0->negLr_0;
        if((kernelContext_0->p_0->nesterov_0) != 0U)
        {
            g_1 = g_1 + kernelContext_0->p_0->beta1_0 * b_0;
        }
        else
        {
            g_1 = b_0;
        }
        x_1 = x_0 + _S1 * g_1;
    }
    *(kernelContext_0->param_0+i_0) = x_1;
    return x_1;
}

uint qF32ToF16_0(float f_0)
{
    uint x_2 = (as_type<uint>((f_0)));
    uint sign_0 = (x_2 >> 16U) & 32768U;
    uint e_0 = (x_2 >> 23U) & 255U;
    uint mant_0 = x_2 & 8388607U;
    uint h_0;
    if(e_0 == 255U)
    {
        uint _S2 = sign_0 | 31744U;
        if(mant_0 != 0U)
        {
            h_0 = 512U | (mant_0 >> 13U);
        }
        else
        {
            h_0 = 0U;
        }
        return _S2 | h_0;
    }
    int exp_0 = int(e_0) - int(112);
    if(exp_0 >= int(31))
    {
        return sign_0 | 31744U;
    }
    bool _S3;
    if(exp_0 <= int(0))
    {
        if(exp_0 < int(-10))
        {
            return sign_0;
        }
        uint mant_1 = mant_0 | 8388608U;
        uint shift_0 = uint(int(14) - exp_0);
        uint h_1 = mant_1 >> shift_0;
        uint rem_0 = mant_1 & ((1U << shift_0) - 1U);
        uint halfway_0 = 1U << (shift_0 - 1U);
        if(rem_0 > halfway_0)
        {
            _S3 = true;
        }
        else
        {
            if(rem_0 == halfway_0)
            {
                _S3 = (h_1 & 1U) != 0U;
            }
            else
            {
                _S3 = false;
            }
        }
        if(_S3)
        {
            h_0 = h_1 + 1U;
        }
        else
        {
            h_0 = h_1;
        }
        return sign_0 | h_0;
    }
    uint h_2 = (uint(exp_0) << 10U) | (mant_0 >> 13U);
    uint rem_1 = mant_0 & 8191U;
    if(rem_1 > 4096U)
    {
        _S3 = true;
    }
    else
    {
        if(rem_1 == 4096U)
        {
            _S3 = (h_2 & 1U) != 0U;
        }
        else
        {
            _S3 = false;
        }
    }
    if(_S3)
    {
        h_0 = h_2 + 1U;
    }
    else
    {
        h_0 = h_2;
    }
    return sign_0 | h_0;
}

[[kernel]] void optimizer_f16(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], OptimParams_0 constant* p_1 [[buffer(6)]], half device* grad_1 [[buffer(1)]], float device* scale_1 [[buffer(4)]], float device* param_1 [[buffer(0)]], float device* m_1 [[buffer(2)]], float device* v_1 [[buffer(3)]], uint device* shadowOut_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->p_0 = p_1;
    (&kernelContext_1)->grad_0 = grad_1;
    (&kernelContext_1)->scale_0 = scale_1;
    (&kernelContext_1)->param_0 = param_1;
    (&kernelContext_1)->m_0 = m_1;
    (&kernelContext_1)->v_0 = v_1;
    (&kernelContext_1)->shadowOut_0 = shadowOut_1;
    uint w_0 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    uint i_1 = 2U * w_0;
    if(i_1 >= (p_1->n_0))
    {
        return;
    }
    float _S4 = step_0(i_1, &kernelContext_1);
    uint _S5 = i_1 + 1U;
    float hi_0;
    if(_S5 < (p_1->n_0))
    {
        float _S6 = step_0(_S5, &kernelContext_1);
        hi_0 = _S6;
    }
    else
    {
        hi_0 = 0.0;
    }
    if(((&kernelContext_1)->p_0->shadow_0) != 0U)
    {
        uint word_0 = (qF32ToF16_0(_S4)) | ((qF32ToF16_0(hi_0)) << 16U);
        if(_S5 < (p_1->n_0))
        {
            *((&kernelContext_1)->shadowOut_0+w_0) = word_0;
        }
        else
        {
            *((&kernelContext_1)->shadowOut_0+w_0) = ((*((&kernelContext_1)->shadowOut_0+w_0)) & 4294901760U) | (word_0 & 65535U);
        }
    }
    return;
}

