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

float dg_negative_infinity_0()
{
    return (as_type<float>((4286578688U)));
}

bool dg_is_nan_0(float x_0)
{
    uint _S1 = (as_type<uint>((x_0)));
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
        return (as_type<float>(((as_type<uint>((a_0))) & (as_type<uint>((b_0))))));
    }
    float _S4;
    if(a_0 > b_0)
    {
        _S4 = a_0;
    }
    else
    {
        _S4 = b_0;
    }
    return _S4;
}

float dg_op_max_0(float a_1, float b_1)
{
    return dg_max_0(a_1, b_1);
}

float dg_op_add_0(float a_2, float b_2)
{
    return a_2 + b_2;
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
    uint4 _S5 = v_1->extent_0;
    uint _S6 = v_1->extent_0.z;
    uint _S7 = row_0 % _S6;
    i_0.z = _S7;
    uint _S8 = row_0 / _S6;
    uint _S9 = _S5.y;
    uint _S10 = _S8 % _S9;
    i_0.y = _S10;
    uint _S11 = _S8 / _S9;
    i_0.x = _S11;
    return i_0;
}

uint dg_at_0(const DgView_0 thread* v_2, uint4 i_1)
{
    return v_2->offset_0 + i_1.x * v_2->stride_0.x + i_1.y * v_2->stride_0.y + i_1.z * v_2->stride_0.z + i_1.w * v_2->stride_0.w;
}

void dg_report_error_0(uint device* errors_0, uint code_0, uint index_0)
{
    uint _S12 = atomic_fetch_or_explicit(((atomic_uint device*)(errors_0+int(0))), code_0, memory_order_relaxed);
    uint _S13 = atomic_fetch_max_explicit(((atomic_uint device*)(errors_0+int(1))), ~index_0, memory_order_relaxed);
    uint _S14 = atomic_fetch_add_explicit(((atomic_uint device*)(errors_0+int(2))), 1U, memory_order_relaxed);
    return;
}

bool dg_in_range_0(uint device* errors_1, uint index_1, uint length_0, uint at_0)
{
    bool _S15;
    if(!DG_DEBUG_0)
    {
        _S15 = true;
    }
    else
    {
        _S15 = index_1 < length_0;
    }
    if(_S15)
    {
        return true;
    }
    dg_report_error_0(errors_1, 1U, at_0);
    return false;
}

struct Params_0
{
    uint rows_0;
    uint gridX_1;
    uint yLength_0;
    uint chunks_0;
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
    float device* partial_0;
    uint device* errors_2;
    float device* y_0;
    array<float, int(256)> threadgroup* red_0;
    float threadgroup* rowMax_0;
};

[[kernel]] void reduce_lse_combine(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(3)]], float device* partial_1 [[buffer(0)]], uint device* errors_3 [[buffer(2)]], float device* y_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->partial_0 = partial_1;
    (&kernelContext_0)->errors_2 = errors_3;
    (&kernelContext_0)->y_0 = y_1;
    threadgroup array<float, int(256)> red_1;
    (&kernelContext_0)->red_0 = &red_1;
    threadgroup float rowMax_1;
    (&kernelContext_0)->rowMax_0 = &rowMax_1;
    uint row_1 = dg_linear_group_0(group_1, p_1->gridX_1);
    if(row_1 >= (p_1->rows_0))
    {
        return;
    }
    uint t_0 = local_0.x;
    uint _S16 = row_1 * (&kernelContext_0)->p_0->chunks_0;
    float _S17 = dg_negative_infinity_0();
    uint g_0 = t_0;
    float acc_0 = _S17;
    for(;;)
    {
        if(g_0 < ((&kernelContext_0)->p_0->chunks_0))
        {
        }
        else
        {
            break;
        }
        float _S18 = dg_max_0(acc_0, *((&kernelContext_0)->partial_0+2U * (_S16 + g_0)));
        g_0 = g_0 + 256U;
        acc_0 = _S18;
    }
    (*(&kernelContext_0)->red_0)[t_0] = acc_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S19 = t_0 < 128U;
    if(_S19)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 128U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S20 = t_0 < 64U;
    if(_S20)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S21 = t_0 < 32U;
    if(_S21)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S22 = t_0 < 16U;
    if(_S22)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S23 = t_0 < 8U;
    if(_S23)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S24 = t_0 < 4U;
    if(_S24)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S25 = t_0 < 2U;
    if(_S25)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S26 = t_0 < 1U;
    if(_S26)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S27 = t_0 == 0U;
    if(_S27)
    {
        *(&kernelContext_0)->rowMax_0 = (*(&kernelContext_0)->red_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threadgroup* _S28 = (&kernelContext_0)->rowMax_0;
    float mx_0 = *_S28;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S29;
    if((*_S28) != _S17)
    {
        _S29 = !dg_is_nan_0(mx_0);
    }
    else
    {
        _S29 = false;
    }
    if(_S29)
    {
        g_0 = t_0;
        acc_0 = 0.0;
        for(;;)
        {
            if(g_0 < ((&kernelContext_0)->p_0->chunks_0))
            {
            }
            else
            {
                break;
            }
            uint _S30 = 2U * (_S16 + g_0);
            float device* _S31 = (&kernelContext_0)->partial_0+_S30;
            float m_0 = *_S31;
            if((*_S31) != _S17)
            {
                acc_0 = acc_0 + *((&kernelContext_0)->partial_0+(_S30 + 1U)) * exp(m_0 - mx_0);
            }
            g_0 = g_0 + 256U;
        }
    }
    else
    {
        acc_0 = 0.0;
    }
    (*(&kernelContext_0)->red_0)[t_0] = acc_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S19)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 128U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S20)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S21)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S22)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S23)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S24)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S25)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S26)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    DgView_0 yv_0 = dg_view_0((&kernelContext_0)->p_0->yOffset_0, uint4((&kernelContext_0)->p_0->yExtent0_0, (&kernelContext_0)->p_0->yExtent1_0, (&kernelContext_0)->p_0->yExtent2_0, (&kernelContext_0)->p_0->yExtent3_0), uint4((&kernelContext_0)->p_0->yStride0_0, (&kernelContext_0)->p_0->yStride1_0, (&kernelContext_0)->p_0->yStride2_0, (&kernelContext_0)->p_0->yStride3_0));
    thread DgView_0 _S32 = yv_0;
    uint3 _S33 = dg_row_0(&_S32, row_1);
    uint4 _S34 = uint4(_S33, 0U);
    thread DgView_0 _S35 = yv_0;
    uint _S36 = dg_at_0(&_S35, _S34);
    if(_S27)
    {
        bool _S37 = dg_in_range_0((&kernelContext_0)->errors_2, _S36, (&kernelContext_0)->p_0->yLength_0, row_1);
        _S29 = _S37;
    }
    else
    {
        _S29 = false;
    }
    if(_S29)
    {
        float device* _S38 = (&kernelContext_0)->y_0+_S36;
        if(mx_0 == _S17)
        {
            _S29 = true;
        }
        else
        {
            _S29 = dg_is_nan_0(mx_0);
        }
        if(_S29)
        {
            acc_0 = mx_0;
        }
        else
        {
            acc_0 = mx_0 + log((*(&kernelContext_0)->red_0)[int(0)]);
        }
        *_S38 = acc_0;
    }
    return;
}

