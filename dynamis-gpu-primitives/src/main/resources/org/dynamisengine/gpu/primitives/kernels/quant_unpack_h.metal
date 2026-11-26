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

uint qCodeWord_0(uint format_0, uint r_0, uint col_0, uint cols_0)
{
    uint _S1;
    if(format_0 == 0U)
    {
        _S1 = (r_0 * cols_0 + col_0) >> 2U;
    }
    else
    {
        _S1 = (r_0 * cols_0 + col_0) >> 3U;
    }
    return _S1;
}

struct Params_0
{
    uint rows_0;
    uint cols_1;
    uint format_1;
    uint block_0;
    uint count_0;
    uint qparams_0;
    uint gridX_2;
    uint qhigh_0;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    uint device* packed_0;
    uint device* y_0;
};

uint qWord_0(uint w_0, KernelContext_0 thread* kernelContext_0)
{
    return *(kernelContext_0->packed_0+w_0);
}

uint kHigh8_0(uint format_2, uint qhigh_1, uint e_0, KernelContext_0 thread* kernelContext_1)
{
    if(format_2 == 4U)
    {
        uint _S2 = qWord_0(qhigh_1 + (e_0 >> 5U), kernelContext_1);
        return _S2 >> (e_0 & 31U);
    }
    uint _S3 = qWord_0(qhigh_1 + (e_0 >> 4U), kernelContext_1);
    return _S3 >> (2U * (e_0 & 15U));
}

float qF16ToF32_0(uint h_0)
{
    return (as_type<half>((ushort)((h_0 & 65535U))));
}

uint qByte_0(uint at_0, KernelContext_0 thread* kernelContext_2)
{
    uint _S4 = qWord_0(at_0 >> 2U, kernelContext_2);
    return (_S4 >> (8U * (at_0 & 3U))) & 255U;
}

float2 kScale_0(uint format_3, uint qparams_1, uint r_1, uint col_1, uint cols_2, KernelContext_0 thread* kernelContext_3)
{
    uint sb_0 = r_1 * (cols_2 / 256U) + col_1 / 256U;
    bool _S5 = format_3 == 5U;
    uint sc_0;
    if(_S5)
    {
        sc_0 = 5U;
    }
    else
    {
        sc_0 = 4U;
    }
    uint base_0 = qparams_1 + sb_0 * sc_0;
    uint _S6 = qWord_0(base_0, kernelContext_3);
    float d_0 = qF16ToF32_0(_S6);
    uint at_1 = 4U * (base_0 + 1U);
    if(_S5)
    {
        uint _S7 = qByte_0(at_1 + col_1 % 256U / 16U, kernelContext_3);
        int sc_1 = int(_S7);
        int sc_2;
        if(sc_1 >= int(128))
        {
            sc_2 = sc_1 - int(256);
        }
        else
        {
            sc_2 = sc_1;
        }
        return float2(d_0 * float(sc_2), 0.0);
    }
    uint j_0 = col_1 % 256U / 32U;
    uint _S8 = at_1 + j_0;
    uint _S9 = qByte_0(_S8, kernelContext_3);
    uint _S10 = qByte_0(_S8 + 4U, kernelContext_3);
    uint _S11 = qByte_0(at_1 + (j_0 ^ 4U), kernelContext_3);
    bool low_0 = j_0 < 4U;
    if(low_0)
    {
        sc_0 = _S9 & 63U;
    }
    else
    {
        sc_0 = (_S10 & 15U) | ((_S11 >> 6U) << 4U);
    }
    uint mn_0;
    if(low_0)
    {
        mn_0 = _S10 & 63U;
    }
    else
    {
        mn_0 = (_S10 >> 4U) | ((_S9 >> 6U) << 4U);
    }
    return float2(d_0 * float(sc_0), qF16ToF32_0(_S6 >> 16U) * float(mn_0));
}

int kCode_0(uint format_4, uint u_0, uint h_1)
{
    if(format_4 == 3U)
    {
        return int(u_0);
    }
    if(format_4 == 4U)
    {
        return int(u_0 | ((h_1 & 1U) << 4U));
    }
    return int(u_0 | ((h_1 & 3U) << 4U)) - int(32);
}

uint qParamHalf_0(uint format_5, uint b_0, uint t_0)
{
    uint _S12;
    if(format_5 == 2U)
    {
        _S12 = 2U * b_0 + t_0;
    }
    else
    {
        _S12 = b_0;
    }
    return _S12;
}

int qCode_0(uint format_6, uint word_0, uint col_2)
{
    int _S13;
    if(format_6 == 0U)
    {
        uint b_1 = (word_0 >> (8U * (col_2 & 3U))) & 255U;
        int _S14 = int(b_1);
        if(b_1 >= 128U)
        {
            _S13 = int(256);
        }
        else
        {
            _S13 = int(0);
        }
        return _S14 - _S13;
    }
    int u_1 = int((word_0 >> (4U * (col_2 & 7U))) & 15U);
    if(format_6 == 1U)
    {
        _S13 = u_1 - int(8);
    }
    else
    {
        _S13 = u_1;
    }
    return _S13;
}

void unpack8_0(uint e_1, array<float, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_4)
{
    uint r_2 = e_1 / kernelContext_4->p_0->cols_1;
    uint col_3 = e_1 % kernelContext_4->p_0->cols_1;
    float m_0;
    uint wd_0;
    if((kernelContext_4->p_0->format_1) >= 3U)
    {
        uint _S15 = *(kernelContext_4->packed_0+qCodeWord_0(kernelContext_4->p_0->format_1, r_2, col_3, kernelContext_4->p_0->cols_1));
        if((kernelContext_4->p_0->format_1) == 3U)
        {
            wd_0 = 0U;
        }
        else
        {
            uint _S16 = kHigh8_0(kernelContext_4->p_0->format_1, kernelContext_4->p_0->qhigh_0, e_1, kernelContext_4);
            wd_0 = _S16;
        }
        uint _S17;
        if((kernelContext_4->p_0->format_1) == 4U)
        {
            _S17 = 1U;
        }
        else
        {
            _S17 = 2U;
        }
        float2 _S18 = kScale_0(kernelContext_4->p_0->format_1, kernelContext_4->p_0->qparams_0, r_2, col_3, kernelContext_4->p_0->cols_1, kernelContext_4);
        uint _S19 = kernelContext_4->p_0->format_1;
        uint _S20 = (_S15 >> 0U) & 15U;
        uint _S21;
        if((kernelContext_4->p_0->format_1) == 3U)
        {
            _S21 = 0U;
        }
        else
        {
            _S21 = wd_0 >> 0U;
        }
        float qk_0 = float(kCode_0(_S19, _S20, _S21));
        if((kernelContext_4->p_0->format_1) == 5U)
        {
            m_0 = _S18.x * qk_0;
        }
        else
        {
            m_0 = _S18.x * qk_0 - _S18.y;
        }
        (*v_0)[0U] = m_0;
        uint _S22 = kernelContext_4->p_0->format_1;
        uint _S23 = (_S15 >> 4U) & 15U;
        if((kernelContext_4->p_0->format_1) == 3U)
        {
            _S21 = 0U;
        }
        else
        {
            _S21 = wd_0 >> _S17;
        }
        float qk_1 = float(kCode_0(_S22, _S23, _S21));
        if((kernelContext_4->p_0->format_1) == 5U)
        {
            m_0 = _S18.x * qk_1;
        }
        else
        {
            m_0 = _S18.x * qk_1 - _S18.y;
        }
        (*v_0)[1U] = m_0;
        uint _S24 = kernelContext_4->p_0->format_1;
        uint _S25 = (_S15 >> 8U) & 15U;
        if((kernelContext_4->p_0->format_1) == 3U)
        {
            _S21 = 0U;
        }
        else
        {
            _S21 = wd_0 >> (_S17 * 2U);
        }
        float qk_2 = float(kCode_0(_S24, _S25, _S21));
        if((kernelContext_4->p_0->format_1) == 5U)
        {
            m_0 = _S18.x * qk_2;
        }
        else
        {
            m_0 = _S18.x * qk_2 - _S18.y;
        }
        (*v_0)[2U] = m_0;
        uint _S26 = kernelContext_4->p_0->format_1;
        uint _S27 = (_S15 >> 12U) & 15U;
        if((kernelContext_4->p_0->format_1) == 3U)
        {
            _S21 = 0U;
        }
        else
        {
            _S21 = wd_0 >> (_S17 * 3U);
        }
        float qk_3 = float(kCode_0(_S26, _S27, _S21));
        if((kernelContext_4->p_0->format_1) == 5U)
        {
            m_0 = _S18.x * qk_3;
        }
        else
        {
            m_0 = _S18.x * qk_3 - _S18.y;
        }
        (*v_0)[3U] = m_0;
        uint _S28 = kernelContext_4->p_0->format_1;
        uint _S29 = (_S15 >> 16U) & 15U;
        if((kernelContext_4->p_0->format_1) == 3U)
        {
            _S21 = 0U;
        }
        else
        {
            _S21 = wd_0 >> (_S17 * 4U);
        }
        float qk_4 = float(kCode_0(_S28, _S29, _S21));
        if((kernelContext_4->p_0->format_1) == 5U)
        {
            m_0 = _S18.x * qk_4;
        }
        else
        {
            m_0 = _S18.x * qk_4 - _S18.y;
        }
        (*v_0)[4U] = m_0;
        uint _S30 = kernelContext_4->p_0->format_1;
        uint _S31 = (_S15 >> 20U) & 15U;
        if((kernelContext_4->p_0->format_1) == 3U)
        {
            _S21 = 0U;
        }
        else
        {
            _S21 = wd_0 >> (_S17 * 5U);
        }
        float qk_5 = float(kCode_0(_S30, _S31, _S21));
        if((kernelContext_4->p_0->format_1) == 5U)
        {
            m_0 = _S18.x * qk_5;
        }
        else
        {
            m_0 = _S18.x * qk_5 - _S18.y;
        }
        (*v_0)[5U] = m_0;
        uint _S32 = kernelContext_4->p_0->format_1;
        uint _S33 = (_S15 >> 24U) & 15U;
        if((kernelContext_4->p_0->format_1) == 3U)
        {
            _S21 = 0U;
        }
        else
        {
            _S21 = wd_0 >> (_S17 * 6U);
        }
        float qk_6 = float(kCode_0(_S32, _S33, _S21));
        if((kernelContext_4->p_0->format_1) == 5U)
        {
            m_0 = _S18.x * qk_6;
        }
        else
        {
            m_0 = _S18.x * qk_6 - _S18.y;
        }
        (*v_0)[6U] = m_0;
        uint _S34 = kernelContext_4->p_0->format_1;
        uint _S35 = (_S15 >> 28U) & 15U;
        if((kernelContext_4->p_0->format_1) == 3U)
        {
            wd_0 = 0U;
        }
        else
        {
            wd_0 = wd_0 >> (_S17 * 7U);
        }
        float qk_7 = float(kCode_0(_S34, _S35, wd_0));
        if((kernelContext_4->p_0->format_1) == 5U)
        {
            m_0 = _S18.x * qk_7;
        }
        else
        {
            m_0 = _S18.x * qk_7 - _S18.y;
        }
        (*v_0)[7U] = m_0;
        return;
    }
    uint b_2 = e_1 / kernelContext_4->p_0->block_0;
    uint dh_0 = qParamHalf_0(kernelContext_4->p_0->format_1, b_2, 0U);
    float _S36 = qF16ToF32_0((*(kernelContext_4->packed_0+(kernelContext_4->p_0->qparams_0 + (dh_0 >> 1U)))) >> (16U * (dh_0 & 1U)));
    if((kernelContext_4->p_0->format_1) == 2U)
    {
        uint mh_0 = qParamHalf_0(kernelContext_4->p_0->format_1, b_2, 1U);
        m_0 = qF16ToF32_0((*(kernelContext_4->packed_0+(kernelContext_4->p_0->qparams_0 + (mh_0 >> 1U)))) >> (16U * (mh_0 & 1U)));
    }
    else
    {
        m_0 = 0.0;
    }
    uint _S37 = qCodeWord_0(kernelContext_4->p_0->format_1, r_2, col_3, kernelContext_4->p_0->cols_1);
    if((kernelContext_4->p_0->format_1) == 0U)
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    else
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    float x_0 = float(qCode_0(kernelContext_4->p_0->format_1, wd_0, col_3)) * _S36;
    float _S38;
    if((kernelContext_4->p_0->format_1) == 2U)
    {
        _S38 = x_0 + m_0;
    }
    else
    {
        _S38 = x_0;
    }
    (*v_0)[0U] = _S38;
    if((kernelContext_4->p_0->format_1) == 0U)
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    else
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    float x_1 = float(qCode_0(kernelContext_4->p_0->format_1, wd_0, col_3 + 1U)) * _S36;
    if((kernelContext_4->p_0->format_1) == 2U)
    {
        _S38 = x_1 + m_0;
    }
    else
    {
        _S38 = x_1;
    }
    (*v_0)[1U] = _S38;
    if((kernelContext_4->p_0->format_1) == 0U)
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    else
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    float x_2 = float(qCode_0(kernelContext_4->p_0->format_1, wd_0, col_3 + 2U)) * _S36;
    if((kernelContext_4->p_0->format_1) == 2U)
    {
        _S38 = x_2 + m_0;
    }
    else
    {
        _S38 = x_2;
    }
    (*v_0)[2U] = _S38;
    if((kernelContext_4->p_0->format_1) == 0U)
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    else
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    float x_3 = float(qCode_0(kernelContext_4->p_0->format_1, wd_0, col_3 + 3U)) * _S36;
    if((kernelContext_4->p_0->format_1) == 2U)
    {
        _S38 = x_3 + m_0;
    }
    else
    {
        _S38 = x_3;
    }
    (*v_0)[3U] = _S38;
    if((kernelContext_4->p_0->format_1) == 0U)
    {
        wd_0 = *(kernelContext_4->packed_0+(_S37 + 1U));
    }
    else
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    float x_4 = float(qCode_0(kernelContext_4->p_0->format_1, wd_0, col_3 + 4U)) * _S36;
    if((kernelContext_4->p_0->format_1) == 2U)
    {
        _S38 = x_4 + m_0;
    }
    else
    {
        _S38 = x_4;
    }
    (*v_0)[4U] = _S38;
    if((kernelContext_4->p_0->format_1) == 0U)
    {
        wd_0 = *(kernelContext_4->packed_0+(_S37 + 1U));
    }
    else
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    float x_5 = float(qCode_0(kernelContext_4->p_0->format_1, wd_0, col_3 + 5U)) * _S36;
    if((kernelContext_4->p_0->format_1) == 2U)
    {
        _S38 = x_5 + m_0;
    }
    else
    {
        _S38 = x_5;
    }
    (*v_0)[5U] = _S38;
    if((kernelContext_4->p_0->format_1) == 0U)
    {
        wd_0 = *(kernelContext_4->packed_0+(_S37 + 1U));
    }
    else
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    float x_6 = float(qCode_0(kernelContext_4->p_0->format_1, wd_0, col_3 + 6U)) * _S36;
    if((kernelContext_4->p_0->format_1) == 2U)
    {
        _S38 = x_6 + m_0;
    }
    else
    {
        _S38 = x_6;
    }
    (*v_0)[6U] = _S38;
    if((kernelContext_4->p_0->format_1) == 0U)
    {
        wd_0 = *(kernelContext_4->packed_0+(_S37 + 1U));
    }
    else
    {
        wd_0 = *(kernelContext_4->packed_0+_S37);
    }
    float x_7 = float(qCode_0(kernelContext_4->p_0->format_1, wd_0, col_3 + 7U)) * _S36;
    if((kernelContext_4->p_0->format_1) == 2U)
    {
        m_0 = x_7 + m_0;
    }
    else
    {
        m_0 = x_7;
    }
    (*v_0)[7U] = m_0;
    return;
}

uint qF32ToF16_0(float f_0)
{
    uint x_8 = (as_type<uint>((f_0)));
    uint sign_0 = (x_8 >> 16U) & 32768U;
    uint e_2 = (x_8 >> 23U) & 255U;
    uint mant_0 = x_8 & 8388607U;
    uint h_2;
    if(e_2 == 255U)
    {
        uint _S39 = sign_0 | 31744U;
        if(mant_0 != 0U)
        {
            h_2 = 512U | (mant_0 >> 13U);
        }
        else
        {
            h_2 = 0U;
        }
        return _S39 | h_2;
    }
    int exp_0 = int(e_2) - int(112);
    if(exp_0 >= int(31))
    {
        return sign_0 | 31744U;
    }
    bool _S40;
    if(exp_0 <= int(0))
    {
        if(exp_0 < int(-10))
        {
            return sign_0;
        }
        uint mant_1 = mant_0 | 8388608U;
        uint shift_0 = uint(int(14) - exp_0);
        uint h_3 = mant_1 >> shift_0;
        uint rem_0 = mant_1 & ((1U << shift_0) - 1U);
        uint halfway_0 = 1U << (shift_0 - 1U);
        if(rem_0 > halfway_0)
        {
            _S40 = true;
        }
        else
        {
            if(rem_0 == halfway_0)
            {
                _S40 = (h_3 & 1U) != 0U;
            }
            else
            {
                _S40 = false;
            }
        }
        if(_S40)
        {
            h_2 = h_3 + 1U;
        }
        else
        {
            h_2 = h_3;
        }
        return sign_0 | h_2;
    }
    uint h_4 = (uint(exp_0) << 10U) | (mant_0 >> 13U);
    uint rem_1 = mant_0 & 8191U;
    if(rem_1 > 4096U)
    {
        _S40 = true;
    }
    else
    {
        if(rem_1 == 4096U)
        {
            _S40 = (h_4 & 1U) != 0U;
        }
        else
        {
            _S40 = false;
        }
    }
    if(_S40)
    {
        h_2 = h_4 + 1U;
    }
    else
    {
        h_2 = h_4;
    }
    return sign_0 | h_2;
}

[[kernel]] void quant_unpack_h(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(2)]], uint device* packed_1 [[buffer(0)]], uint device* y_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->p_0 = p_1;
    (&kernelContext_5)->packed_0 = packed_1;
    (&kernelContext_5)->y_0 = y_1;
    uint t_1 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    uint e_3 = 8U * t_1;
    if(e_3 >= (p_1->count_0))
    {
        return;
    }
    thread array<float, int(8)> v_1;
    unpack8_0(e_3, &v_1, &kernelContext_5);
    uint _S41 = 4U * t_1;
    *((&kernelContext_5)->y_0+_S41) = (qF32ToF16_0(v_1[0U])) | ((qF32ToF16_0(v_1[1U])) << 16U);
    *((&kernelContext_5)->y_0+(_S41 + 1U)) = (qF32ToF16_0(v_1[2U])) | ((qF32ToF16_0(v_1[3U])) << 16U);
    *((&kernelContext_5)->y_0+(_S41 + 2U)) = (qF32ToF16_0(v_1[4U])) | ((qF32ToF16_0(v_1[5U])) << 16U);
    *((&kernelContext_5)->y_0+(_S41 + 3U)) = (qF32ToF16_0(v_1[6U])) | ((qF32ToF16_0(v_1[7U])) << 16U);
    return;
}

