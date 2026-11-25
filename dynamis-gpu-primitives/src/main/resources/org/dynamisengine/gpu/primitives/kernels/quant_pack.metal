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

float dg_positive_infinity_0()
{
    return (as_type<float>((2139095040U)));
}

float dg_negative_infinity_0()
{
    return (as_type<float>((4286578688U)));
}

bool dg_is_finite_0(float x_0)
{
    return ((as_type<uint>((x_0))) & 2139095040U) != 2139095040U;
}

bool dg_is_nan_0(float x_1)
{
    uint _S1 = (as_type<uint>((x_1)));
    bool _S2;
    if((_S1 & 2139095040U) == 2139095040U)
    {
        _S2 = (_S1 & 8388607U) != 0U;
    }
    else
    {
        _S2 = false;
    }
    return _S2;
}

float dg_min_0(float a_0, float b_0)
{
    if(dg_is_nan_0(a_0))
    {
        return a_0;
    }
    if(dg_is_nan_0(b_0))
    {
        return b_0;
    }
    bool _S3;
    if(a_0 == 0.0)
    {
        _S3 = b_0 == 0.0;
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        return (as_type<float>(((as_type<uint>((a_0))) | (as_type<uint>((b_0))))));
    }
    float _S4;
    if(a_0 < b_0)
    {
        _S4 = a_0;
    }
    else
    {
        _S4 = b_0;
    }
    return _S4;
}

float dg_max_0(float a_1, float b_1)
{
    if(dg_is_nan_0(a_1))
    {
        return a_1;
    }
    if(dg_is_nan_0(b_1))
    {
        return b_1;
    }
    bool _S5;
    if(a_1 == 0.0)
    {
        _S5 = b_1 == 0.0;
    }
    else
    {
        _S5 = false;
    }
    if(_S5)
    {
        return (as_type<float>(((as_type<uint>((a_1))) & (as_type<uint>((b_1))))));
    }
    float _S6;
    if(a_1 > b_1)
    {
        _S6 = a_1;
    }
    else
    {
        _S6 = b_1;
    }
    return _S6;
}

float qInv127_0()
{
    return (as_type<float>((1006699012U)));
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
        uint _S7 = sign_0 | 31744U;
        if(mant_0 != 0U)
        {
            h_0 = 512U | (mant_0 >> 13U);
        }
        else
        {
            h_0 = 0U;
        }
        return _S7 | h_0;
    }
    int exp_0 = int(e_0) - int(112);
    if(exp_0 >= int(31))
    {
        return sign_0 | 31744U;
    }
    bool _S8;
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
            _S8 = true;
        }
        else
        {
            if(rem_0 == halfway_0)
            {
                _S8 = (h_1 & 1U) != 0U;
            }
            else
            {
                _S8 = false;
            }
        }
        if(_S8)
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
        _S8 = true;
    }
    else
    {
        if(rem_1 == 4096U)
        {
            _S8 = (h_2 & 1U) != 0U;
        }
        else
        {
            _S8 = false;
        }
    }
    if(_S8)
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

float qInv15_0()
{
    return (as_type<float>((1032358025U)));
}

void dg_report_error_0(uint device* errors_0, uint code_0, uint index_0)
{
    uint _S9 = atomic_fetch_or_explicit(((atomic_uint device*)(errors_0+int(0))), code_0, memory_order_relaxed);
    uint _S10 = atomic_fetch_max_explicit(((atomic_uint device*)(errors_0+int(1))), ~index_0, memory_order_relaxed);
    uint _S11 = atomic_fetch_add_explicit(((atomic_uint device*)(errors_0+int(2))), 1U, memory_order_relaxed);
    return;
}

uint qBits_0(uint format_0)
{
    uint _S12;
    if(format_0 == 0U)
    {
        _S12 = 8U;
    }
    else
    {
        _S12 = 4U;
    }
    return _S12;
}

uint qCodeWord_0(uint format_1, uint r_0, uint col_0, uint cols_0)
{
    uint _S13;
    if(format_1 == 0U)
    {
        _S13 = (r_0 * cols_0 + col_0) >> 2U;
    }
    else
    {
        _S13 = (r_0 * cols_0 + col_0) >> 3U;
    }
    return _S13;
}

int qNearest_0(float y_0, float d_0)
{
    int _S14 = int(clamp(round(y_0 * (1.0 / d_0)), -300.0, 300.0));
    int _S15 = int(2) * _S14;
    float up_0 = float(_S15 + int(1)) * d_0 * 0.5;
    float down_0 = float(_S15 - int(1)) * d_0 * 0.5;
    bool _S16;
    if(y_0 > up_0)
    {
        _S16 = true;
    }
    else
    {
        if(y_0 == up_0)
        {
            _S16 = (_S14 & int(1)) != int(0);
        }
        else
        {
            _S16 = false;
        }
    }
    int c_0;
    if(_S16)
    {
        c_0 = _S14 + int(1);
    }
    else
    {
        if(y_0 < down_0)
        {
            _S16 = true;
        }
        else
        {
            if(y_0 == down_0)
            {
                _S16 = (_S14 & int(1)) != int(0);
            }
            else
            {
                _S16 = false;
            }
        }
        if(_S16)
        {
            c_0 = _S14 - int(1);
        }
        else
        {
            c_0 = _S14;
        }
    }
    int _S17 = int(2) * c_0;
    float up_1 = float(_S17 + int(1)) * d_0 * 0.5;
    float down_1 = float(_S17 - int(1)) * d_0 * 0.5;
    if(y_0 > up_1)
    {
        _S16 = true;
    }
    else
    {
        if(y_0 == up_1)
        {
            _S16 = (c_0 & int(1)) != int(0);
        }
        else
        {
            _S16 = false;
        }
    }
    if(_S16)
    {
        c_0 = c_0 + int(1);
    }
    else
    {
        if(y_0 < down_1)
        {
            _S16 = true;
        }
        else
        {
            if(y_0 == down_1)
            {
                _S16 = (c_0 & int(1)) != int(0);
            }
            else
            {
                _S16 = false;
            }
        }
        if(_S16)
        {
            c_0 = c_0 - int(1);
        }
    }
    return c_0;
}

struct Params_0
{
    uint rows_0;
    uint cols_1;
    uint format_2;
    uint block_0;
    uint blocks_0;
    uint qparams_0;
    uint gridX_2;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    uint device* x_3;
    uint device* errors_1;
    uint device* packed_0;
};

void encodeBlock_0(uint b_2, uint thread* dBits_0, uint thread* mBits_0, KernelContext_0 thread* kernelContext_0)
{
    uint bpr_0 = kernelContext_0->p_0->cols_1 / kernelContext_0->p_0->block_0;
    uint r_1 = b_2 / bpr_0;
    uint _S18 = b_2 % bpr_0;
    uint c0_0 = _S18 * kernelContext_0->p_0->block_0;
    uint _S19 = r_1 * kernelContext_0->p_0->cols_1 + c0_0;
    float _S20 = dg_positive_infinity_0();
    float _S21 = dg_negative_infinity_0();
    bool finite_0 = true;
    float lo_0 = _S20;
    float hi_0 = _S21;
    float amax_0 = 0.0;
    float extreme_0 = 0.0;
    uint i_0 = 0U;
    for(;;)
    {
        if(i_0 < (kernelContext_0->p_0->block_0))
        {
        }
        else
        {
            break;
        }
        float v_0 = (as_type<float>((*(kernelContext_0->x_3+(_S19 + i_0)))));
        if(finite_0)
        {
            finite_0 = dg_is_finite_0(v_0);
        }
        else
        {
            finite_0 = false;
        }
        float _S22 = dg_min_0(lo_0, v_0);
        float _S23 = dg_max_0(hi_0, v_0);
        float _S24 = abs(v_0);
        if(_S24 > amax_0)
        {
            amax_0 = _S24;
            extreme_0 = v_0;
        }
        uint _S25 = i_0 + 1U;
        lo_0 = _S22;
        hi_0 = _S23;
        i_0 = _S25;
    }
    *mBits_0 = 0U;
    float m_0;
    if((kernelContext_0->p_0->format_2) == 0U)
    {
        *dBits_0 = qF32ToF16_0(amax_0 * qInv127_0());
        m_0 = 0.0;
    }
    else
    {
        if((kernelContext_0->p_0->format_2) == 1U)
        {
            *dBits_0 = qF32ToF16_0(extreme_0 * -0.125);
            m_0 = 0.0;
        }
        else
        {
            uint _S26 = qF32ToF16_0(lo_0);
            *mBits_0 = _S26;
            float _S27 = qF16ToF32_0(_S26);
            *dBits_0 = qF32ToF16_0((hi_0 - lo_0) * qInv15_0());
            m_0 = _S27;
        }
    }
    float d_1 = qF16ToF32_0(*dBits_0);
    if(!finite_0)
    {
        finite_0 = true;
    }
    else
    {
        finite_0 = !dg_is_finite_0(d_1);
    }
    if(finite_0)
    {
        finite_0 = true;
    }
    else
    {
        finite_0 = !dg_is_finite_0(m_0);
    }
    float d_2;
    if(finite_0)
    {
        dg_report_error_0(kernelContext_0->errors_1, 2U, b_2);
        *dBits_0 = 0U;
        *mBits_0 = 0U;
        d_2 = 0.0;
    }
    else
    {
        d_2 = d_1;
    }
    int _S28;
    if((kernelContext_0->p_0->format_2) == 0U)
    {
        _S28 = int(-127);
    }
    else
    {
        if((kernelContext_0->p_0->format_2) == 1U)
        {
            _S28 = int(-8);
        }
        else
        {
            _S28 = int(0);
        }
    }
    int _S29;
    if((kernelContext_0->p_0->format_2) == 0U)
    {
        _S29 = int(127);
    }
    else
    {
        if((kernelContext_0->p_0->format_2) == 1U)
        {
            _S29 = int(7);
        }
        else
        {
            _S29 = int(15);
        }
    }
    if((kernelContext_0->p_0->format_2) == 0U)
    {
        i_0 = 4U;
    }
    else
    {
        i_0 = 8U;
    }
    uint _S30 = qBits_0(kernelContext_0->p_0->format_2);
    uint _S31 = qCodeWord_0(kernelContext_0->p_0->format_2, r_1, c0_0, kernelContext_0->p_0->cols_1);
    uint w_0 = 0U;
    for(;;)
    {
        uint _S32 = kernelContext_0->p_0->block_0 / i_0;
        if(w_0 < _S32)
        {
        }
        else
        {
            break;
        }
        uint j_0 = 0U;
        uint word_0 = 0U;
        for(;;)
        {
            if(j_0 < i_0)
            {
            }
            else
            {
                break;
            }
            float v_1 = (as_type<float>((*(kernelContext_0->x_3+(_S19 + w_0 * i_0 + j_0)))));
            float y_1;
            if((kernelContext_0->p_0->format_2) == 2U)
            {
                y_1 = v_1 - m_0;
            }
            else
            {
                y_1 = v_1;
            }
            int code_1;
            if(d_2 == 0.0)
            {
                code_1 = int(0);
            }
            else
            {
                if(d_2 > 0.0)
                {
                    code_1 = qNearest_0(y_1, d_2);
                }
                else
                {
                    code_1 = qNearest_0(- y_1, - d_2);
                }
                code_1 = clamp(code_1, _S28, _S29);
            }
            uint u_0;
            if((kernelContext_0->p_0->format_2) == 0U)
            {
                u_0 = uint(code_1) & 255U;
            }
            else
            {
                if((kernelContext_0->p_0->format_2) == 1U)
                {
                    u_0 = uint(code_1 + int(8));
                }
                else
                {
                    u_0 = uint(code_1);
                }
            }
            uint word_1 = word_0 | (u_0 << (_S30 * j_0));
            j_0 = j_0 + 1U;
            word_0 = word_1;
        }
        *(kernelContext_0->packed_0+(_S31 + w_0)) = word_0;
        w_0 = w_0 + 1U;
    }
    return;
}

[[kernel]] void quant_pack(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(3)]], uint device* x_4 [[buffer(0)]], uint device* errors_2 [[buffer(2)]], uint device* packed_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->p_0 = p_1;
    (&kernelContext_1)->x_3 = x_4;
    (&kernelContext_1)->errors_1 = errors_2;
    (&kernelContext_1)->packed_0 = packed_1;
    uint t_0 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 64U);
    uint b0_0 = 2U * t_0;
    if(b0_0 >= (p_1->blocks_0))
    {
        return;
    }
    thread uint d1_0 = 0U;
    thread uint m1_0 = 0U;
    thread uint d0_0;
    thread uint m0_0;
    encodeBlock_0(b0_0, &d0_0, &m0_0, &kernelContext_1);
    uint _S33 = b0_0 + 1U;
    bool second_0 = _S33 < (p_1->blocks_0);
    if(second_0)
    {
        encodeBlock_0(_S33, &d1_0, &m1_0, &kernelContext_1);
    }
    if(((&kernelContext_1)->p_0->format_2) == 2U)
    {
        *((&kernelContext_1)->packed_0+((&kernelContext_1)->p_0->qparams_0 + b0_0)) = d0_0 | (m0_0 << 16U);
        if(second_0)
        {
            *((&kernelContext_1)->packed_0+((&kernelContext_1)->p_0->qparams_0 + b0_0 + 1U)) = d1_0 | (m1_0 << 16U);
        }
    }
    else
    {
        *((&kernelContext_1)->packed_0+((&kernelContext_1)->p_0->qparams_0 + t_0)) = d0_0 | (d1_0 << 16U);
    }
    return;
}

