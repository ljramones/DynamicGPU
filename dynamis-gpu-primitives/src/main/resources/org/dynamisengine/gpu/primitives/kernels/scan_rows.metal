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

struct DgView_0
{
    uint offset_0;
    uint4 extent_0;
    uint4 stride_0;
};

DgView_0 dg_view_0(uint offset_1, uint4 extent_1, uint4 stride_1)
{
    thread DgView_0 v_0;
    (&v_0)->offset_0 = offset_1;
    (&v_0)->extent_0 = extent_1;
    (&v_0)->stride_0 = stride_1;
    return v_0;
}

uint3 dg_row_0(const DgView_0 thread* v_1, uint row_0)
{
    thread uint3 i_0;
    uint4 _S1 = v_1->extent_0;
    uint _S2 = v_1->extent_0.z;
    uint _S3 = row_0 % _S2;
    i_0.z = _S3;
    uint _S4 = row_0 / _S2;
    uint _S5 = _S1.y;
    uint _S6 = _S4 % _S5;
    i_0.y = _S6;
    uint _S7 = _S4 / _S5;
    i_0.x = _S7;
    return i_0;
}

constant uint fc_OP_0 [[function_constant(0)]];
constant uint OP_0 = is_function_constant_defined(fc_OP_0) ? fc_OP_0 : 0U;
float dg_negative_infinity_0()
{
    return (as_type<float>((4286578688U)));
}

float dg_positive_infinity_0()
{
    return (as_type<float>((2139095040U)));
}

float identity_0()
{
    float _S8;
    if(OP_0 == 0U)
    {
        _S8 = 0.0;
    }
    else
    {
        if(OP_0 == 1U)
        {
            _S8 = dg_negative_infinity_0();
        }
        else
        {
            _S8 = dg_positive_infinity_0();
        }
    }
    return _S8;
}

uint dg_at_0(const DgView_0 thread* v_2, uint4 i_1)
{
    return v_2->offset_0 + i_1.x * v_2->stride_0.x + i_1.y * v_2->stride_0.y + i_1.z * v_2->stride_0.z + i_1.w * v_2->stride_0.w;
}

bool dg_is_nan_0(float x_0)
{
    uint _S9 = (as_type<uint>((x_0)));
    bool _S10;
    if((_S9 & 2139095040U) == 2139095040U)
    {
        _S10 = (_S9 & 8388607U) != 0U;
    }
    else
    {
        _S10 = false;
    }
    return _S10;
}

float dg_max_0(float a_0, float b_0)
{
    if(dg_is_nan_0(a_0))
    {
        return a_0;
    }
    if(dg_is_nan_0(b_0))
    {
        return b_0;
    }
    bool _S11;
    if(a_0 == 0.0)
    {
        _S11 = b_0 == 0.0;
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        return (as_type<float>(((as_type<uint>((a_0))) & (as_type<uint>((b_0))))));
    }
    float _S12;
    if(a_0 > b_0)
    {
        _S12 = a_0;
    }
    else
    {
        _S12 = b_0;
    }
    return _S12;
}

float dg_min_0(float a_1, float b_1)
{
    if(dg_is_nan_0(a_1))
    {
        return a_1;
    }
    if(dg_is_nan_0(b_1))
    {
        return b_1;
    }
    bool _S13;
    if(a_1 == 0.0)
    {
        _S13 = b_1 == 0.0;
    }
    else
    {
        _S13 = false;
    }
    if(_S13)
    {
        return (as_type<float>(((as_type<uint>((a_1))) | (as_type<uint>((b_1))))));
    }
    float _S14;
    if(a_1 < b_1)
    {
        _S14 = a_1;
    }
    else
    {
        _S14 = b_1;
    }
    return _S14;
}

float combine_0(float a_2, float b_2)
{
    float _S15;
    if(OP_0 == 0U)
    {
        _S15 = a_2 + b_2;
    }
    else
    {
        if(OP_0 == 1U)
        {
            _S15 = dg_max_0(a_2, b_2);
        }
        else
        {
            _S15 = dg_min_0(a_2, b_2);
        }
    }
    return _S15;
}

struct Params_0
{
    uint rows_0;
    uint gridX_1;
    uint xOffset_0;
    uint xExtent0_0;
    uint xStride0_0;
    uint xExtent1_0;
    uint xStride1_0;
    uint xExtent2_0;
    uint xStride2_0;
    uint xExtent3_0;
    uint xStride3_0;
    uint yOffset_0;
    uint yExtent0_0;
    uint yStride0_0;
    uint yExtent1_0;
    uint yStride1_0;
    uint yExtent2_0;
    uint yStride2_0;
    uint yExtent3_0;
    uint yStride3_0;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    float device* x_1;
    float device* y_0;
    array<float, int(256)> threadgroup* buf_0;
    array<float, int(2)> threadgroup* carries_0;
};

[[kernel]] void scan_rows(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(2)]], float device* x_2 [[buffer(0)]], float device* y_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->x_1 = x_2;
    (&kernelContext_0)->y_0 = y_1;
    threadgroup array<float, int(256)> buf_1;
    (&kernelContext_0)->buf_0 = &buf_1;
    threadgroup array<float, int(2)> carries_1;
    (&kernelContext_0)->carries_0 = &carries_1;
    uint row_1 = dg_linear_group_0(group_1, p_1->gridX_1);
    if(row_1 >= (p_1->rows_0))
    {
        return;
    }
    uint _S16 = local_0.x;
    DgView_0 xv_0 = dg_view_0((&kernelContext_0)->p_0->xOffset_0, uint4((&kernelContext_0)->p_0->xExtent0_0, (&kernelContext_0)->p_0->xExtent1_0, (&kernelContext_0)->p_0->xExtent2_0, (&kernelContext_0)->p_0->xExtent3_0), uint4((&kernelContext_0)->p_0->xStride0_0, (&kernelContext_0)->p_0->xStride1_0, (&kernelContext_0)->p_0->xStride2_0, (&kernelContext_0)->p_0->xStride3_0));
    DgView_0 _S17 = dg_view_0((&kernelContext_0)->p_0->yOffset_0, uint4((&kernelContext_0)->p_0->yExtent0_0, (&kernelContext_0)->p_0->yExtent1_0, (&kernelContext_0)->p_0->yExtent2_0, (&kernelContext_0)->p_0->yExtent3_0), uint4((&kernelContext_0)->p_0->yStride0_0, (&kernelContext_0)->p_0->yStride1_0, (&kernelContext_0)->p_0->yStride2_0, (&kernelContext_0)->p_0->yStride3_0));
    uint _S18 = xv_0.extent_0.w;
    thread DgView_0 _S19 = xv_0;
    uint3 _S20 = dg_row_0(&_S19, row_1);
    float _S21 = identity_0();
    uint base_0 = 0U;
    float carry_0 = _S21;
    uint slot_0 = 0U;
    for(;;)
    {
        if(base_0 < _S18)
        {
        }
        else
        {
            break;
        }
        uint j_0 = base_0 + _S16;
        bool _S22 = j_0 < _S18;
        float v_3;
        if(_S22)
        {
            uint4 _S23 = uint4(_S20, j_0);
            thread DgView_0 _S24 = xv_0;
            uint _S25 = dg_at_0(&_S24, _S23);
            v_3 = *((&kernelContext_0)->x_1+_S25);
        }
        else
        {
            v_3 = _S21;
        }
        (*(&kernelContext_0)->buf_0)[_S16] = v_3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S26 = _S16 + 1U;
        uint _S27 = _S26 * 2U;
        uint dgI_0 = _S27 - 1U;
        bool _S28 = dgI_0 < 256U;
        if(_S28)
        {
            (*(&kernelContext_0)->buf_0)[dgI_0] = combine_0((*(&kernelContext_0)->buf_0)[dgI_0 - 1U], (*(&kernelContext_0)->buf_0)[dgI_0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_1 = _S27 * 2U - 1U;
        bool _S29 = dgI_1 < 256U;
        if(_S29)
        {
            (*(&kernelContext_0)->buf_0)[dgI_1] = combine_0((*(&kernelContext_0)->buf_0)[dgI_1 - 2U], (*(&kernelContext_0)->buf_0)[dgI_1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_2 = _S26 * 4U * 2U - 1U;
        bool _S30 = dgI_2 < 256U;
        if(_S30)
        {
            (*(&kernelContext_0)->buf_0)[dgI_2] = combine_0((*(&kernelContext_0)->buf_0)[dgI_2 - 4U], (*(&kernelContext_0)->buf_0)[dgI_2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_3 = _S26 * 8U * 2U - 1U;
        bool _S31 = dgI_3 < 256U;
        if(_S31)
        {
            (*(&kernelContext_0)->buf_0)[dgI_3] = combine_0((*(&kernelContext_0)->buf_0)[dgI_3 - 8U], (*(&kernelContext_0)->buf_0)[dgI_3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_4 = _S26 * 16U * 2U - 1U;
        bool _S32 = dgI_4 < 256U;
        if(_S32)
        {
            (*(&kernelContext_0)->buf_0)[dgI_4] = combine_0((*(&kernelContext_0)->buf_0)[dgI_4 - 16U], (*(&kernelContext_0)->buf_0)[dgI_4]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_5 = _S26 * 32U * 2U - 1U;
        bool _S33 = dgI_5 < 256U;
        if(_S33)
        {
            (*(&kernelContext_0)->buf_0)[dgI_5] = combine_0((*(&kernelContext_0)->buf_0)[dgI_5 - 32U], (*(&kernelContext_0)->buf_0)[dgI_5]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_6 = _S26 * 64U * 2U - 1U;
        bool _S34 = dgI_6 < 256U;
        if(_S34)
        {
            (*(&kernelContext_0)->buf_0)[dgI_6] = combine_0((*(&kernelContext_0)->buf_0)[dgI_6 - 64U], (*(&kernelContext_0)->buf_0)[dgI_6]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_7 = _S26 * 128U * 2U - 1U;
        bool _S35 = dgI_7 < 256U;
        if(_S35)
        {
            (*(&kernelContext_0)->buf_0)[dgI_7] = combine_0((*(&kernelContext_0)->buf_0)[dgI_7 - 128U], (*(&kernelContext_0)->buf_0)[dgI_7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S16 == 0U)
        {
            (*(&kernelContext_0)->buf_0)[255U] = _S21;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S35)
        {
            float _S36 = combine_0((*(&kernelContext_0)->buf_0)[dgI_7], (*(&kernelContext_0)->buf_0)[dgI_7 - 128U]);
            (*(&kernelContext_0)->buf_0)[dgI_7 - 128U] = (*(&kernelContext_0)->buf_0)[dgI_7];
            (*(&kernelContext_0)->buf_0)[dgI_7] = _S36;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S34)
        {
            float _S37 = combine_0((*(&kernelContext_0)->buf_0)[dgI_6], (*(&kernelContext_0)->buf_0)[dgI_6 - 64U]);
            (*(&kernelContext_0)->buf_0)[dgI_6 - 64U] = (*(&kernelContext_0)->buf_0)[dgI_6];
            (*(&kernelContext_0)->buf_0)[dgI_6] = _S37;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S33)
        {
            float _S38 = combine_0((*(&kernelContext_0)->buf_0)[dgI_5], (*(&kernelContext_0)->buf_0)[dgI_5 - 32U]);
            (*(&kernelContext_0)->buf_0)[dgI_5 - 32U] = (*(&kernelContext_0)->buf_0)[dgI_5];
            (*(&kernelContext_0)->buf_0)[dgI_5] = _S38;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S32)
        {
            float _S39 = combine_0((*(&kernelContext_0)->buf_0)[dgI_4], (*(&kernelContext_0)->buf_0)[dgI_4 - 16U]);
            (*(&kernelContext_0)->buf_0)[dgI_4 - 16U] = (*(&kernelContext_0)->buf_0)[dgI_4];
            (*(&kernelContext_0)->buf_0)[dgI_4] = _S39;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S31)
        {
            float _S40 = combine_0((*(&kernelContext_0)->buf_0)[dgI_3], (*(&kernelContext_0)->buf_0)[dgI_3 - 8U]);
            (*(&kernelContext_0)->buf_0)[dgI_3 - 8U] = (*(&kernelContext_0)->buf_0)[dgI_3];
            (*(&kernelContext_0)->buf_0)[dgI_3] = _S40;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S30)
        {
            float _S41 = combine_0((*(&kernelContext_0)->buf_0)[dgI_2], (*(&kernelContext_0)->buf_0)[dgI_2 - 4U]);
            (*(&kernelContext_0)->buf_0)[dgI_2 - 4U] = (*(&kernelContext_0)->buf_0)[dgI_2];
            (*(&kernelContext_0)->buf_0)[dgI_2] = _S41;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S29)
        {
            float _S42 = combine_0((*(&kernelContext_0)->buf_0)[dgI_1], (*(&kernelContext_0)->buf_0)[dgI_1 - 2U]);
            (*(&kernelContext_0)->buf_0)[dgI_1 - 2U] = (*(&kernelContext_0)->buf_0)[dgI_1];
            (*(&kernelContext_0)->buf_0)[dgI_1] = _S42;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S28)
        {
            float _S43 = combine_0((*(&kernelContext_0)->buf_0)[dgI_0], (*(&kernelContext_0)->buf_0)[dgI_0 - 1U]);
            (*(&kernelContext_0)->buf_0)[dgI_0 - 1U] = (*(&kernelContext_0)->buf_0)[dgI_0];
            (*(&kernelContext_0)->buf_0)[dgI_0] = _S43;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float out_0 = combine_0(carry_0, combine_0((*(&kernelContext_0)->buf_0)[_S16], v_3));
        if(_S22)
        {
            uint4 _S44 = uint4(_S20, j_0);
            thread DgView_0 _S45 = _S17;
            uint _S46 = dg_at_0(&_S45, _S44);
            *((&kernelContext_0)->y_0+_S46) = out_0;
        }
        if(_S16 == 255U)
        {
            (*(&kernelContext_0)->carries_0)[slot_0] = out_0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S47 = (*(&kernelContext_0)->carries_0)[slot_0];
        uint slot_1 = slot_0 ^ 1U;
        base_0 = base_0 + 256U;
        carry_0 = _S47;
        slot_0 = slot_1;
    }
    return;
}

