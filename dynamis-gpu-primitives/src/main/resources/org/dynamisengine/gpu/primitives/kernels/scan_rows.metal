#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant bool fc_DG_DEBUG_0 [[function_constant(100)]];
constant bool DG_DEBUG_0 = is_function_constant_defined(fc_DG_DEBUG_0) ? fc_DG_DEBUG_0 : false;
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

void dg_report_error_0(uint device* errors_0, uint code_0, uint index_0)
{
    uint _S9 = atomic_fetch_or_explicit(((atomic_uint device*)(errors_0+int(0))), code_0, memory_order_relaxed);
    uint _S10 = atomic_fetch_max_explicit(((atomic_uint device*)(errors_0+int(1))), ~index_0, memory_order_relaxed);
    uint _S11 = atomic_fetch_add_explicit(((atomic_uint device*)(errors_0+int(2))), 1U, memory_order_relaxed);
    return;
}

bool dg_in_range_0(uint device* errors_1, uint index_1, uint length_0, uint at_0)
{
    bool _S12;
    if(!DG_DEBUG_0)
    {
        _S12 = true;
    }
    else
    {
        _S12 = index_1 < length_0;
    }
    if(_S12)
    {
        return true;
    }
    dg_report_error_0(errors_1, 1U, at_0);
    return false;
}

bool dg_is_nan_0(float x_0)
{
    uint _S13 = (as_type<uint>((x_0)));
    bool _S14;
    if((_S13 & 2139095040U) == 2139095040U)
    {
        _S14 = (_S13 & 8388607U) != 0U;
    }
    else
    {
        _S14 = false;
    }
    return _S14;
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
    bool _S15;
    if(a_0 == 0.0)
    {
        _S15 = b_0 == 0.0;
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        return (as_type<float>(((as_type<uint>((a_0))) & (as_type<uint>((b_0))))));
    }
    float _S16;
    if(a_0 > b_0)
    {
        _S16 = a_0;
    }
    else
    {
        _S16 = b_0;
    }
    return _S16;
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
    bool _S17;
    if(a_1 == 0.0)
    {
        _S17 = b_1 == 0.0;
    }
    else
    {
        _S17 = false;
    }
    if(_S17)
    {
        return (as_type<float>(((as_type<uint>((a_1))) | (as_type<uint>((b_1))))));
    }
    float _S18;
    if(a_1 < b_1)
    {
        _S18 = a_1;
    }
    else
    {
        _S18 = b_1;
    }
    return _S18;
}

float combine_0(float a_2, float b_2)
{
    float _S19;
    if(OP_0 == 0U)
    {
        _S19 = a_2 + b_2;
    }
    else
    {
        if(OP_0 == 1U)
        {
            _S19 = dg_max_0(a_2, b_2);
        }
        else
        {
            _S19 = dg_min_0(a_2, b_2);
        }
    }
    return _S19;
}

struct Params_0
{
    uint rows_0;
    uint gridX_1;
    uint xLength_0;
    uint yLength_0;
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
    uint device* errors_2;
    float device* x_1;
    float device* y_0;
    array<float, int(256)> threadgroup* buf_0;
    array<float, int(2)> threadgroup* carries_0;
};

[[kernel]] void scan_rows(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(3)]], uint device* errors_3 [[buffer(2)]], float device* x_2 [[buffer(0)]], float device* y_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->errors_2 = errors_3;
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
    uint _S20 = local_0.x;
    DgView_0 xv_0 = dg_view_0((&kernelContext_0)->p_0->xOffset_0, uint4((&kernelContext_0)->p_0->xExtent0_0, (&kernelContext_0)->p_0->xExtent1_0, (&kernelContext_0)->p_0->xExtent2_0, (&kernelContext_0)->p_0->xExtent3_0), uint4((&kernelContext_0)->p_0->xStride0_0, (&kernelContext_0)->p_0->xStride1_0, (&kernelContext_0)->p_0->xStride2_0, (&kernelContext_0)->p_0->xStride3_0));
    DgView_0 _S21 = dg_view_0((&kernelContext_0)->p_0->yOffset_0, uint4((&kernelContext_0)->p_0->yExtent0_0, (&kernelContext_0)->p_0->yExtent1_0, (&kernelContext_0)->p_0->yExtent2_0, (&kernelContext_0)->p_0->yExtent3_0), uint4((&kernelContext_0)->p_0->yStride0_0, (&kernelContext_0)->p_0->yStride1_0, (&kernelContext_0)->p_0->yStride2_0, (&kernelContext_0)->p_0->yStride3_0));
    uint _S22 = xv_0.extent_0.w;
    thread DgView_0 _S23 = xv_0;
    uint3 _S24 = dg_row_0(&_S23, row_1);
    float _S25 = identity_0();
    uint base_0 = 0U;
    float carry_0 = _S25;
    uint slot_0 = 0U;
    for(;;)
    {
        if(base_0 < _S22)
        {
        }
        else
        {
            break;
        }
        uint j_0 = base_0 + _S20;
        uint4 _S26 = uint4(_S24, j_0);
        thread DgView_0 _S27 = xv_0;
        uint _S28 = dg_at_0(&_S27, _S26);
        bool _S29 = j_0 < _S22;
        bool _S30;
        if(_S29)
        {
            bool _S31 = dg_in_range_0((&kernelContext_0)->errors_2, _S28, (&kernelContext_0)->p_0->xLength_0, row_1);
            _S30 = _S31;
        }
        else
        {
            _S30 = false;
        }
        float v_3;
        if(_S30)
        {
            v_3 = *((&kernelContext_0)->x_1+_S28);
        }
        else
        {
            v_3 = _S25;
        }
        (*(&kernelContext_0)->buf_0)[_S20] = v_3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S32 = _S20 + 1U;
        uint _S33 = _S32 * 2U;
        uint dgI_0 = _S33 - 1U;
        bool _S34 = dgI_0 < 256U;
        if(_S34)
        {
            (*(&kernelContext_0)->buf_0)[dgI_0] = combine_0((*(&kernelContext_0)->buf_0)[dgI_0 - 1U], (*(&kernelContext_0)->buf_0)[dgI_0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_1 = _S33 * 2U - 1U;
        bool _S35 = dgI_1 < 256U;
        if(_S35)
        {
            (*(&kernelContext_0)->buf_0)[dgI_1] = combine_0((*(&kernelContext_0)->buf_0)[dgI_1 - 2U], (*(&kernelContext_0)->buf_0)[dgI_1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_2 = _S32 * 4U * 2U - 1U;
        bool _S36 = dgI_2 < 256U;
        if(_S36)
        {
            (*(&kernelContext_0)->buf_0)[dgI_2] = combine_0((*(&kernelContext_0)->buf_0)[dgI_2 - 4U], (*(&kernelContext_0)->buf_0)[dgI_2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_3 = _S32 * 8U * 2U - 1U;
        bool _S37 = dgI_3 < 256U;
        if(_S37)
        {
            (*(&kernelContext_0)->buf_0)[dgI_3] = combine_0((*(&kernelContext_0)->buf_0)[dgI_3 - 8U], (*(&kernelContext_0)->buf_0)[dgI_3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_4 = _S32 * 16U * 2U - 1U;
        bool _S38 = dgI_4 < 256U;
        if(_S38)
        {
            (*(&kernelContext_0)->buf_0)[dgI_4] = combine_0((*(&kernelContext_0)->buf_0)[dgI_4 - 16U], (*(&kernelContext_0)->buf_0)[dgI_4]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_5 = _S32 * 32U * 2U - 1U;
        bool _S39 = dgI_5 < 256U;
        if(_S39)
        {
            (*(&kernelContext_0)->buf_0)[dgI_5] = combine_0((*(&kernelContext_0)->buf_0)[dgI_5 - 32U], (*(&kernelContext_0)->buf_0)[dgI_5]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_6 = _S32 * 64U * 2U - 1U;
        bool _S40 = dgI_6 < 256U;
        if(_S40)
        {
            (*(&kernelContext_0)->buf_0)[dgI_6] = combine_0((*(&kernelContext_0)->buf_0)[dgI_6 - 64U], (*(&kernelContext_0)->buf_0)[dgI_6]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_7 = _S32 * 128U * 2U - 1U;
        bool _S41 = dgI_7 < 256U;
        if(_S41)
        {
            (*(&kernelContext_0)->buf_0)[dgI_7] = combine_0((*(&kernelContext_0)->buf_0)[dgI_7 - 128U], (*(&kernelContext_0)->buf_0)[dgI_7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S20 == 0U)
        {
            (*(&kernelContext_0)->buf_0)[255U] = _S25;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S41)
        {
            float _S42 = combine_0((*(&kernelContext_0)->buf_0)[dgI_7], (*(&kernelContext_0)->buf_0)[dgI_7 - 128U]);
            (*(&kernelContext_0)->buf_0)[dgI_7 - 128U] = (*(&kernelContext_0)->buf_0)[dgI_7];
            (*(&kernelContext_0)->buf_0)[dgI_7] = _S42;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S40)
        {
            float _S43 = combine_0((*(&kernelContext_0)->buf_0)[dgI_6], (*(&kernelContext_0)->buf_0)[dgI_6 - 64U]);
            (*(&kernelContext_0)->buf_0)[dgI_6 - 64U] = (*(&kernelContext_0)->buf_0)[dgI_6];
            (*(&kernelContext_0)->buf_0)[dgI_6] = _S43;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S39)
        {
            float _S44 = combine_0((*(&kernelContext_0)->buf_0)[dgI_5], (*(&kernelContext_0)->buf_0)[dgI_5 - 32U]);
            (*(&kernelContext_0)->buf_0)[dgI_5 - 32U] = (*(&kernelContext_0)->buf_0)[dgI_5];
            (*(&kernelContext_0)->buf_0)[dgI_5] = _S44;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S38)
        {
            float _S45 = combine_0((*(&kernelContext_0)->buf_0)[dgI_4], (*(&kernelContext_0)->buf_0)[dgI_4 - 16U]);
            (*(&kernelContext_0)->buf_0)[dgI_4 - 16U] = (*(&kernelContext_0)->buf_0)[dgI_4];
            (*(&kernelContext_0)->buf_0)[dgI_4] = _S45;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S37)
        {
            float _S46 = combine_0((*(&kernelContext_0)->buf_0)[dgI_3], (*(&kernelContext_0)->buf_0)[dgI_3 - 8U]);
            (*(&kernelContext_0)->buf_0)[dgI_3 - 8U] = (*(&kernelContext_0)->buf_0)[dgI_3];
            (*(&kernelContext_0)->buf_0)[dgI_3] = _S46;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S36)
        {
            float _S47 = combine_0((*(&kernelContext_0)->buf_0)[dgI_2], (*(&kernelContext_0)->buf_0)[dgI_2 - 4U]);
            (*(&kernelContext_0)->buf_0)[dgI_2 - 4U] = (*(&kernelContext_0)->buf_0)[dgI_2];
            (*(&kernelContext_0)->buf_0)[dgI_2] = _S47;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S35)
        {
            float _S48 = combine_0((*(&kernelContext_0)->buf_0)[dgI_1], (*(&kernelContext_0)->buf_0)[dgI_1 - 2U]);
            (*(&kernelContext_0)->buf_0)[dgI_1 - 2U] = (*(&kernelContext_0)->buf_0)[dgI_1];
            (*(&kernelContext_0)->buf_0)[dgI_1] = _S48;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S34)
        {
            float _S49 = combine_0((*(&kernelContext_0)->buf_0)[dgI_0], (*(&kernelContext_0)->buf_0)[dgI_0 - 1U]);
            (*(&kernelContext_0)->buf_0)[dgI_0 - 1U] = (*(&kernelContext_0)->buf_0)[dgI_0];
            (*(&kernelContext_0)->buf_0)[dgI_0] = _S49;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float out_0 = combine_0(carry_0, combine_0((*(&kernelContext_0)->buf_0)[_S20], v_3));
        thread DgView_0 _S50 = _S21;
        uint _S51 = dg_at_0(&_S50, _S26);
        bool _S52;
        if(_S29)
        {
            bool _S53 = dg_in_range_0((&kernelContext_0)->errors_2, _S51, (&kernelContext_0)->p_0->yLength_0, row_1);
            _S52 = _S53;
        }
        else
        {
            _S52 = false;
        }
        if(_S52)
        {
            *((&kernelContext_0)->y_0+_S51) = out_0;
        }
        if(_S20 == 255U)
        {
            (*(&kernelContext_0)->carries_0)[slot_0] = out_0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S54 = (*(&kernelContext_0)->carries_0)[slot_0];
        uint slot_1 = slot_0 ^ 1U;
        base_0 = base_0 + 256U;
        carry_0 = _S54;
        slot_0 = slot_1;
    }
    return;
}

