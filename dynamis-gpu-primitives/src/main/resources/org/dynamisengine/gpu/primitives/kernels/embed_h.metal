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

float qF16ToF32_0(uint h_0)
{
    return (as_type<half>((ushort)((h_0 & 65535U))));
}

uint qCodeWord_0(uint format_0, uint r_0, uint col_0, uint cols_0)
{
    uint _S4;
    if(format_0 == 0U)
    {
        _S4 = (r_0 * cols_0 + col_0) >> 2U;
    }
    else
    {
        _S4 = (r_0 * cols_0 + col_0) >> 3U;
    }
    return _S4;
}

struct EmbedParams_0
{
    uint count_0;
    uint dim_0;
    uint vocab_0;
    uint ldy_0;
    uint kind_0;
    uint qformat_0;
    uint qblock_0;
    uint qparams_0;
    uint qhigh_0;
    uint gridX_2;
};

struct KernelContext_0
{
    EmbedParams_0 constant* p_0;
    uint device* tokens_0;
    uint device* errors_1;
    uint device* table_0;
    half device* y_0;
};

uint qWord_0(uint w_0, KernelContext_0 thread* kernelContext_0)
{
    return *(kernelContext_0->table_0+w_0);
}

uint kHigh8_0(uint format_1, uint qhigh_1, uint e_0, KernelContext_0 thread* kernelContext_1)
{
    if(format_1 == 4U)
    {
        uint _S5 = qWord_0(qhigh_1 + (e_0 >> 5U), kernelContext_1);
        return _S5 >> (e_0 & 31U);
    }
    uint _S6 = qWord_0(qhigh_1 + (e_0 >> 4U), kernelContext_1);
    return _S6 >> (2U * (e_0 & 15U));
}

int kCode_0(uint format_2, uint u_0, uint h_1)
{
    if(format_2 == 3U)
    {
        return int(u_0);
    }
    if(format_2 == 4U)
    {
        return int(u_0 | ((h_1 & 1U) << 4U));
    }
    return int(u_0 | ((h_1 & 3U) << 4U)) - int(32);
}

uint qByte_0(uint at_0, KernelContext_0 thread* kernelContext_2)
{
    uint _S7 = qWord_0(at_0 >> 2U, kernelContext_2);
    return (_S7 >> (8U * (at_0 & 3U))) & 255U;
}

float2 kScale_0(uint format_3, uint qparams_1, uint r_1, uint col_1, uint cols_1, KernelContext_0 thread* kernelContext_3)
{
    uint sb_0 = r_1 * (cols_1 / 256U) + col_1 / 256U;
    bool _S8 = format_3 == 5U;
    uint sc_0;
    if(_S8)
    {
        sc_0 = 5U;
    }
    else
    {
        sc_0 = 4U;
    }
    uint base_0 = qparams_1 + sb_0 * sc_0;
    uint _S9 = qWord_0(base_0, kernelContext_3);
    float d_0 = qF16ToF32_0(_S9);
    uint at_1 = 4U * (base_0 + 1U);
    if(_S8)
    {
        uint _S10 = qByte_0(at_1 + col_1 % 256U / 16U, kernelContext_3);
        int sc_1 = int(_S10);
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
    uint _S11 = at_1 + j_0;
    uint _S12 = qByte_0(_S11, kernelContext_3);
    uint _S13 = qByte_0(_S11 + 4U, kernelContext_3);
    uint _S14 = qByte_0(at_1 + (j_0 ^ 4U), kernelContext_3);
    bool low_0 = j_0 < 4U;
    if(low_0)
    {
        sc_0 = _S12 & 63U;
    }
    else
    {
        sc_0 = (_S13 & 15U) | ((_S14 >> 6U) << 4U);
    }
    uint mn_0;
    if(low_0)
    {
        mn_0 = _S13 & 63U;
    }
    else
    {
        mn_0 = (_S13 >> 4U) | ((_S12 >> 6U) << 4U);
    }
    return float2(d_0 * float(sc_0), qF16ToF32_0(_S9 >> 16U) * float(mn_0));
}

int qCode_0(uint format_4, uint word_0, uint col_2)
{
    int _S15;
    if(format_4 == 0U)
    {
        uint b_0 = (word_0 >> (8U * (col_2 & 3U))) & 255U;
        int _S16 = int(b_0);
        if(b_0 >= 128U)
        {
            _S15 = int(256);
        }
        else
        {
            _S15 = int(0);
        }
        return _S16 - _S15;
    }
    int u_1 = int((word_0 >> (4U * (col_2 & 7U))) & 15U);
    if(format_4 == 1U)
    {
        _S15 = u_1 - int(8);
    }
    else
    {
        _S15 = u_1;
    }
    return _S15;
}

uint qParamHalf_0(uint format_5, uint b_1, uint t_0)
{
    uint _S17;
    if(format_5 == 2U)
    {
        _S17 = 2U * b_1 + t_0;
    }
    else
    {
        _S17 = b_1;
    }
    return _S17;
}

[[kernel]] void embed_h(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], EmbedParams_0 constant* p_1 [[buffer(4)]], uint device* tokens_1 [[buffer(0)]], uint device* errors_2 [[buffer(3)]], uint device* table_1 [[buffer(1)]], half device* y_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->p_0 = p_1;
    (&kernelContext_4)->tokens_0 = tokens_1;
    (&kernelContext_4)->errors_1 = errors_2;
    (&kernelContext_4)->table_0 = table_1;
    (&kernelContext_4)->y_0 = y_1;
    uint id_0 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(id_0 >= (p_1->count_0))
    {
        return;
    }
    uint t_1 = id_0 / (&kernelContext_4)->p_0->dim_0;
    uint j_1 = id_0 % (&kernelContext_4)->p_0->dim_0;
    uint device* _S18 = (&kernelContext_4)->tokens_0+t_1;
    uint row_0 = *_S18;
    if((*_S18) >= ((&kernelContext_4)->p_0->vocab_0))
    {
        if(j_1 == 0U)
        {
            dg_report_error_0((&kernelContext_4)->errors_1, 1U, t_1);
        }
        return;
    }
    uint e_1 = row_0 * (&kernelContext_4)->p_0->dim_0 + j_1;
    float v_0;
    if(((&kernelContext_4)->p_0->kind_0) == 0U)
    {
        v_0 = (as_type<float>((*((&kernelContext_4)->table_0+e_1))));
    }
    else
    {
        if(((&kernelContext_4)->p_0->kind_0) == 1U)
        {
            v_0 = qF16ToF32_0((*((&kernelContext_4)->table_0+(e_1 >> 1U))) >> (16U * (e_1 & 1U)));
        }
        else
        {
            uint f_0 = (&kernelContext_4)->p_0->qformat_0;
            if(((&kernelContext_4)->p_0->qformat_0) >= 3U)
            {
                uint u_2 = ((*((&kernelContext_4)->table_0+qCodeWord_0(f_0, row_0, j_1, (&kernelContext_4)->p_0->dim_0))) >> (4U * (j_1 & 7U))) & 15U;
                uint hi_0;
                if(f_0 == 3U)
                {
                    hi_0 = 0U;
                }
                else
                {
                    uint _S19 = kHigh8_0(f_0, (&kernelContext_4)->p_0->qhigh_0, e_1 & 4294967288U, &kernelContext_4);
                    if(f_0 == 4U)
                    {
                        hi_0 = 1U;
                    }
                    else
                    {
                        hi_0 = 2U;
                    }
                    hi_0 = _S19 >> (hi_0 * (e_1 & 7U));
                }
                float q_0 = float(kCode_0(f_0, u_2, hi_0));
                float2 _S20 = kScale_0(f_0, (&kernelContext_4)->p_0->qparams_0, row_0, j_1, (&kernelContext_4)->p_0->dim_0, &kernelContext_4);
                if(f_0 == 5U)
                {
                    v_0 = _S20.x * q_0;
                }
                else
                {
                    v_0 = _S20.x * q_0 - _S20.y;
                }
            }
            else
            {
                float q_1 = float(qCode_0(f_0, *((&kernelContext_4)->table_0+qCodeWord_0(f_0, row_0, j_1, (&kernelContext_4)->p_0->dim_0)), j_1));
                uint _S21 = (&kernelContext_4)->p_0->dim_0 / (&kernelContext_4)->p_0->qblock_0;
                uint _S22 = row_0 * _S21;
                uint _S23 = j_1 / (&kernelContext_4)->p_0->qblock_0;
                uint bl_0 = _S22 + _S23;
                uint dh_0 = qParamHalf_0(f_0, bl_0, 0U);
                float v_1 = q_1 * qF16ToF32_0((*((&kernelContext_4)->table_0+((&kernelContext_4)->p_0->qparams_0 + (dh_0 >> 1U)))) >> (16U * (dh_0 & 1U)));
                if(f_0 == 2U)
                {
                    uint mh_0 = qParamHalf_0(f_0, bl_0, 1U);
                    v_0 = v_1 + qF16ToF32_0((*((&kernelContext_4)->table_0+((&kernelContext_4)->p_0->qparams_0 + (mh_0 >> 1U)))) >> (16U * (mh_0 & 1U)));
                }
                else
                {
                    v_0 = v_1;
                }
            }
        }
    }
    *((&kernelContext_4)->y_0+(t_1 * (&kernelContext_4)->p_0->ldy_0 + j_1)) = half(v_0);
    return;
}

