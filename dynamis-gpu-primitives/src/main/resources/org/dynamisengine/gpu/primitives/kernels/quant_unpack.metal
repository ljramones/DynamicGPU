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
    float device* y_0;
    uint device* packed_0;
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

int kCode_0(uint format_3, uint u_0, uint h_0)
{
    if(format_3 == 3U)
    {
        return int(u_0);
    }
    if(format_3 == 4U)
    {
        return int(u_0 | ((h_0 & 1U) << 4U));
    }
    return int(u_0 | ((h_0 & 3U) << 4U)) - int(32);
}

float qF16ToF32_0(uint h_1)
{
    return (as_type<half>((ushort)((h_1 & 65535U))));
}

uint qByte_0(uint at_0, KernelContext_0 thread* kernelContext_2)
{
    uint _S4 = qWord_0(at_0 >> 2U, kernelContext_2);
    return (_S4 >> (8U * (at_0 & 3U))) & 255U;
}

float2 kScale_0(uint format_4, uint qparams_1, uint r_1, uint col_1, uint cols_2, KernelContext_0 thread* kernelContext_3)
{
    uint sb_0 = r_1 * (cols_2 / 256U) + col_1 / 256U;
    bool _S5 = format_4 == 5U;
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

int qCode_0(uint format_5, uint word_0, uint col_2)
{
    int _S12;
    if(format_5 == 0U)
    {
        uint b_0 = (word_0 >> (8U * (col_2 & 3U))) & 255U;
        int _S13 = int(b_0);
        if(b_0 >= 128U)
        {
            _S12 = int(256);
        }
        else
        {
            _S12 = int(0);
        }
        return _S13 - _S12;
    }
    int u_1 = int((word_0 >> (4U * (col_2 & 7U))) & 15U);
    if(format_5 == 1U)
    {
        _S12 = u_1 - int(8);
    }
    else
    {
        _S12 = u_1;
    }
    return _S12;
}

uint qParamHalf_0(uint format_6, uint b_1, uint t_0)
{
    uint _S14;
    if(format_6 == 2U)
    {
        _S14 = 2U * b_1 + t_0;
    }
    else
    {
        _S14 = b_1;
    }
    return _S14;
}

float unpackAt_0(uint e_1, KernelContext_0 thread* kernelContext_4)
{
    uint r_2 = e_1 / kernelContext_4->p_0->cols_1;
    uint col_3 = e_1 % kernelContext_4->p_0->cols_1;
    float v_0;
    if((kernelContext_4->p_0->format_1) >= 3U)
    {
        uint u_2 = ((*(kernelContext_4->packed_0+qCodeWord_0(kernelContext_4->p_0->format_1, r_2, col_3, kernelContext_4->p_0->cols_1))) >> (4U * (col_3 & 7U))) & 15U;
        uint h_2;
        if((kernelContext_4->p_0->format_1) == 3U)
        {
            h_2 = 0U;
        }
        else
        {
            uint _S15 = kHigh8_0(kernelContext_4->p_0->format_1, kernelContext_4->p_0->qhigh_0, e_1 & 4294967288U, kernelContext_4);
            if((kernelContext_4->p_0->format_1) == 4U)
            {
                h_2 = 1U;
            }
            else
            {
                h_2 = 2U;
            }
            h_2 = _S15 >> (h_2 * (e_1 & 7U));
        }
        float qk_0 = float(kCode_0(kernelContext_4->p_0->format_1, u_2, h_2));
        float2 _S16 = kScale_0(kernelContext_4->p_0->format_1, kernelContext_4->p_0->qparams_0, r_2, col_3, kernelContext_4->p_0->cols_1, kernelContext_4);
        if((kernelContext_4->p_0->format_1) == 5U)
        {
            v_0 = _S16.x * qk_0;
        }
        else
        {
            v_0 = _S16.x * qk_0 - _S16.y;
        }
        return v_0;
    }
    uint b_2 = e_1 / kernelContext_4->p_0->block_0;
    uint dh_0 = qParamHalf_0(kernelContext_4->p_0->format_1, b_2, 0U);
    float v_1 = float(qCode_0(kernelContext_4->p_0->format_1, *(kernelContext_4->packed_0+qCodeWord_0(kernelContext_4->p_0->format_1, r_2, col_3, kernelContext_4->p_0->cols_1)), col_3)) * qF16ToF32_0((*(kernelContext_4->packed_0+(kernelContext_4->p_0->qparams_0 + (dh_0 >> 1U)))) >> (16U * (dh_0 & 1U)));
    if((kernelContext_4->p_0->format_1) == 2U)
    {
        uint mh_0 = qParamHalf_0(kernelContext_4->p_0->format_1, b_2, 1U);
        v_0 = v_1 + qF16ToF32_0((*(kernelContext_4->packed_0+(kernelContext_4->p_0->qparams_0 + (mh_0 >> 1U)))) >> (16U * (mh_0 & 1U)));
    }
    else
    {
        v_0 = v_1;
    }
    return v_0;
}

[[kernel]] void quant_unpack(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(2)]], float device* y_1 [[buffer(1)]], uint device* packed_1 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->p_0 = p_1;
    (&kernelContext_5)->y_0 = y_1;
    (&kernelContext_5)->packed_0 = packed_1;
    uint t_1 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(t_1 >= (p_1->count_0))
    {
        return;
    }
    float device* _S17 = (&kernelContext_5)->y_0+t_1;
    float _S18 = unpackAt_0(t_1, &kernelContext_5);
    *_S17 = _S18;
    return;
}

