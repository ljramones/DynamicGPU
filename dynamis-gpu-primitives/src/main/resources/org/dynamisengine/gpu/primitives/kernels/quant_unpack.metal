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

int qCode_0(uint format_1, uint word_0, uint col_1)
{
    int _S2;
    if(format_1 == 0U)
    {
        uint b_0 = (word_0 >> (8U * (col_1 & 3U))) & 255U;
        int _S3 = int(b_0);
        if(b_0 >= 128U)
        {
            _S2 = int(256);
        }
        else
        {
            _S2 = int(0);
        }
        return _S3 - _S2;
    }
    int u_0 = int((word_0 >> (4U * (col_1 & 7U))) & 15U);
    if(format_1 == 1U)
    {
        _S2 = u_0 - int(8);
    }
    else
    {
        _S2 = u_0;
    }
    return _S2;
}

uint qParamHalf_0(uint format_2, uint b_1, uint t_0)
{
    uint _S4;
    if(format_2 == 2U)
    {
        _S4 = 2U * b_1 + t_0;
    }
    else
    {
        _S4 = b_1;
    }
    return _S4;
}

float qF16ToF32_0(uint h_0)
{
    return (as_type<half>((ushort)((h_0 & 65535U))));
}

struct Params_0
{
    uint rows_0;
    uint cols_1;
    uint format_3;
    uint block_0;
    uint count_0;
    uint qparams_0;
    uint gridX_2;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    uint device* packed_0;
    float device* y_0;
};

[[kernel]] void quant_unpack(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(2)]], uint device* packed_1 [[buffer(0)]], float device* y_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->packed_0 = packed_1;
    (&kernelContext_0)->y_0 = y_1;
    uint e_0 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(e_0 >= (p_1->count_0))
    {
        return;
    }
    uint r_1 = e_0 / (&kernelContext_0)->p_0->cols_1;
    uint col_2 = e_0 % (&kernelContext_0)->p_0->cols_1;
    uint b_2 = e_0 / (&kernelContext_0)->p_0->block_0;
    uint dh_0 = qParamHalf_0((&kernelContext_0)->p_0->format_3, b_2, 0U);
    float v_0 = float(qCode_0((&kernelContext_0)->p_0->format_3, *((&kernelContext_0)->packed_0+qCodeWord_0((&kernelContext_0)->p_0->format_3, r_1, col_2, (&kernelContext_0)->p_0->cols_1)), col_2)) * qF16ToF32_0((*((&kernelContext_0)->packed_0+((&kernelContext_0)->p_0->qparams_0 + (dh_0 >> 1U)))) >> (16U * (dh_0 & 1U)));
    float v_1;
    if(((&kernelContext_0)->p_0->format_3) == 2U)
    {
        uint mh_0 = qParamHalf_0((&kernelContext_0)->p_0->format_3, b_2, 1U);
        v_1 = v_0 + qF16ToF32_0((*((&kernelContext_0)->packed_0+((&kernelContext_0)->p_0->qparams_0 + (mh_0 >> 1U)))) >> (16U * (mh_0 & 1U)));
    }
    else
    {
        v_1 = v_0;
    }
    *((&kernelContext_0)->y_0+e_0) = v_1;
    return;
}

