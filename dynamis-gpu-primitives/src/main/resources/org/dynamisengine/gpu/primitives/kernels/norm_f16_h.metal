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

uint qF32ToF16_0(float f_0)
{
    uint x_0 = (as_type<uint>((f_0)));
    uint sign_0 = (x_0 >> 16U) & 32768U;
    uint e_0 = (x_0 >> 23U) & 255U;
    uint mant_0 = x_0 & 8388607U;
    uint h_0;
    if(e_0 == 255U)
    {
        uint _S1 = sign_0 | 31744U;
        if(mant_0 != 0U)
        {
            h_0 = 512U | (mant_0 >> 13U);
        }
        else
        {
            h_0 = 0U;
        }
        return _S1 | h_0;
    }
    int exp_0 = int(e_0) - int(112);
    if(exp_0 >= int(31))
    {
        return sign_0 | 31744U;
    }
    bool _S2;
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
            _S2 = true;
        }
        else
        {
            if(rem_0 == halfway_0)
            {
                _S2 = (h_1 & 1U) != 0U;
            }
            else
            {
                _S2 = false;
            }
        }
        if(_S2)
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
        _S2 = true;
    }
    else
    {
        if(rem_1 == 4096U)
        {
            _S2 = (h_2 & 1U) != 0U;
        }
        else
        {
            _S2 = false;
        }
    }
    if(_S2)
    {
        h_0 = h_2 + 1U;
    }
    else
    {
        h_0 = h_2;
    }
    return sign_0 | h_0;
}

float qF16ToF32_0(uint h_3)
{
    return (as_type<half>((ushort)((h_3 & 65535U))));
}

float roundIn_0(float s_0)
{
    return qF16ToF32_0(qF32ToF16_0(s_0));
}

float dg_op_add_0(float a_0, float b_0)
{
    return a_0 + b_0;
}

struct NormParams_0
{
    uint rows_0;
    uint n_0;
    uint ldx_0;
    uint ldr_0;
    uint ldh_0;
    uint ldy_0;
    float eps_0;
    uint layer_0;
    uint residual_0;
    uint bias_0;
    uint gridX_1;
    uint inPlace_0;
};

struct KernelContext_0
{
    NormParams_0 constant* p_0;
    half device* h_4;
    half device* x_1;
    half device* res_0;
    float device* w_0;
    float device* b_1;
    half device* y_0;
    array<float, int(256)> threadgroup* red_0;
    float threadgroup* rowStat_0;
};

float groupSum_0(uint t_0, float partial_0, KernelContext_0 thread* kernelContext_0)
{
    (*kernelContext_0->red_0)[t_0] = partial_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 128U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 128U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 64U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 64U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 32U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 32U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 16U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 16U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 8U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 8U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 4U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 4U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 2U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 2U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 1U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 1U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 == 0U)
    {
        *kernelContext_0->rowStat_0 = (*kernelContext_0->red_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threadgroup* _S3 = kernelContext_0->rowStat_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *_S3;
}

float value_0(uint r_0, uint j_0, KernelContext_0 thread* kernelContext_1)
{
    float _S4;
    if((kernelContext_1->p_0->residual_0) != 0U)
    {
        _S4 = float(*(kernelContext_1->h_4+(r_0 * kernelContext_1->p_0->ldh_0 + j_0)));
    }
    else
    {
        _S4 = float(*(kernelContext_1->x_1+(r_0 * kernelContext_1->p_0->ldx_0 + j_0)));
    }
    return _S4;
}

[[kernel]] void norm_f16_h(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], NormParams_0 constant* p_1 [[buffer(6)]], half device* h_5 [[buffer(2)]], half device* x_2 [[buffer(0)]], half device* res_1 [[buffer(1)]], float device* w_1 [[buffer(3)]], float device* b_2 [[buffer(4)]], half device* y_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->p_0 = p_1;
    (&kernelContext_2)->h_4 = h_5;
    (&kernelContext_2)->x_1 = x_2;
    (&kernelContext_2)->res_0 = res_1;
    (&kernelContext_2)->w_0 = w_1;
    (&kernelContext_2)->b_1 = b_2;
    (&kernelContext_2)->y_0 = y_1;
    threadgroup array<float, int(256)> red_1;
    (&kernelContext_2)->red_0 = &red_1;
    threadgroup float rowStat_1;
    (&kernelContext_2)->rowStat_0 = &rowStat_1;
    uint r_1 = dg_linear_group_0(group_1, p_1->gridX_1);
    if(r_1 >= (p_1->rows_0))
    {
        return;
    }
    float v_0;
    float xv_0;
    uint t_1 = local_0.x;
    uint n_1 = (&kernelContext_2)->p_0->n_0;
    float partial_1 = 0.0;
    uint j_1 = t_1;
    for(;;)
    {
        if(j_1 < n_1)
        {
        }
        else
        {
            break;
        }
        if(((&kernelContext_2)->p_0->residual_0) != 0U)
        {
            if(((&kernelContext_2)->p_0->inPlace_0) != 0U)
            {
                xv_0 = float(*((&kernelContext_2)->h_4+(r_1 * (&kernelContext_2)->p_0->ldh_0 + j_1)));
            }
            else
            {
                xv_0 = float(*((&kernelContext_2)->x_1+(r_1 * (&kernelContext_2)->p_0->ldx_0 + j_1)));
            }
            float v_1 = roundIn_0(xv_0 + float(*((&kernelContext_2)->res_0+(r_1 * (&kernelContext_2)->p_0->ldr_0 + j_1))));
            *((&kernelContext_2)->h_4+(r_1 * (&kernelContext_2)->p_0->ldh_0 + j_1)) = half(v_1);
            v_0 = v_1;
        }
        else
        {
            v_0 = float(*((&kernelContext_2)->x_1+(r_1 * (&kernelContext_2)->p_0->ldx_0 + j_1)));
        }
        if(((&kernelContext_2)->p_0->layer_0) != 0U)
        {
            partial_1 = partial_1 + v_0;
        }
        else
        {
            partial_1 = partial_1 + v_0 * v_0;
        }
        j_1 = j_1 + 256U;
    }
    float _S5 = groupSum_0(t_1, partial_1, &kernelContext_2);
    float mean_0;
    if(((&kernelContext_2)->p_0->layer_0) != 0U)
    {
        float _S6 = float(n_1);
        float _S7 = _S5 / _S6;
        partial_1 = 0.0;
        j_1 = t_1;
        for(;;)
        {
            if(j_1 < n_1)
            {
            }
            else
            {
                break;
            }
            float _S8 = value_0(r_1, j_1, &kernelContext_2);
            float d_0 = _S8 - _S7;
            float _S9 = partial_1 + d_0 * d_0;
            uint j_2 = j_1 + 256U;
            partial_1 = _S9;
            j_1 = j_2;
        }
        float _S10 = groupSum_0(t_1, partial_1, &kernelContext_2);
        xv_0 = _S10 / _S6;
        mean_0 = _S7;
    }
    else
    {
        xv_0 = _S5 / float(n_1);
        mean_0 = 0.0;
    }
    float _S11 = rsqrt(xv_0 + (&kernelContext_2)->p_0->eps_0);
    j_1 = t_1;
    for(;;)
    {
        if(j_1 < n_1)
        {
        }
        else
        {
            break;
        }
        float _S12 = value_0(r_1, j_1, &kernelContext_2);
        float v_2 = (_S12 - mean_0) * _S11 * *((&kernelContext_2)->w_0+j_1);
        if(((&kernelContext_2)->p_0->bias_0) != 0U)
        {
            v_0 = v_2 + *((&kernelContext_2)->b_1+j_1);
        }
        else
        {
            v_0 = v_2;
        }
        *((&kernelContext_2)->y_0+(r_1 * (&kernelContext_2)->p_0->ldy_0 + j_1)) = half(v_0);
        j_1 = j_1 + 256U;
    }
    return;
}

