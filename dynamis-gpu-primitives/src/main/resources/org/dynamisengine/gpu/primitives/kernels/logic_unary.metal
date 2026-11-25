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

uint dg_linear_thread_0(uint3 group_1, uint3 local_0, uint gridX_1, uint groupThreads_0)
{
    return dg_linear_group_0(group_1, gridX_1) * groupThreads_0 + local_0.x;
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

uint4 dg_coords_0(const DgView_0 thread* v_1, uint linear_0)
{
    thread uint4 i_0;
    uint4 _S1 = v_1->extent_0;
    uint _S2 = v_1->extent_0.w;
    uint _S3 = linear_0 % _S2;
    i_0.w = _S3;
    uint _S4 = linear_0 / _S2;
    uint _S5 = _S1.z;
    uint _S6 = _S4 % _S5;
    i_0.z = _S6;
    uint _S7 = _S4 / _S5;
    uint _S8 = _S1.y;
    uint _S9 = _S7 % _S8;
    i_0.y = _S9;
    uint _S10 = _S7 / _S8;
    i_0.x = _S10;
    return i_0;
}

uint dg_at_0(const DgView_0 thread* v_2, uint4 i_1)
{
    return v_2->offset_0 + i_1.x * v_2->stride_0.x + i_1.y * v_2->stride_0.y + i_1.z * v_2->stride_0.z + i_1.w * v_2->stride_0.w;
}

uint dg_at_broadcast_0(const DgView_0 thread* v_3, uint4 i_2)
{
    uint _S11 = dg_at_0(v_3, select(i_2, uint4(0U, 0U, 0U, 0U), (v_3->extent_0) == uint4(1U, 1U, 1U, 1U)));
    return _S11;
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

constant uint fc_OP_0 [[function_constant(0)]];
constant uint OP_0 = is_function_constant_defined(fc_OP_0) ? fc_OP_0 : 0U;
bool lgIsNaN_0(uint x_0)
{
    return (x_0 & 2147483647U) > 2139095040U;
}

uint lgUnary_0(uint x_1)
{
    uint _S16;
    switch(OP_0)
    {
    case 0U:
        {
            if(x_1 == 0U)
            {
                _S16 = 1U;
            }
            else
            {
                _S16 = 0U;
            }
            return _S16;
        }
    case 1U:
        {
            if((x_1 & 2139095040U) != 2139095040U)
            {
                _S16 = 1U;
            }
            else
            {
                _S16 = 0U;
            }
            return _S16;
        }
    case 2U:
        {
            if(lgIsNaN_0(x_1))
            {
                _S16 = 1U;
            }
            else
            {
                _S16 = 0U;
            }
            return _S16;
        }
    default:
        {
            if((x_1 & 2147483647U) == 2139095040U)
            {
                _S16 = 1U;
            }
            else
            {
                _S16 = 0U;
            }
            return _S16;
        }
    }
}

struct LogicUnaryParams_0
{
    uint count_0;
    uint gridX_2;
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
    LogicUnaryParams_0 constant* p_0;
    uint device* errors_2;
    uint device* y_0;
    uint device* x_2;
};

[[kernel]] void logic_unary(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], LogicUnaryParams_0 constant* p_1 [[buffer(3)]], uint device* errors_3 [[buffer(2)]], uint device* y_1 [[buffer(1)]], uint device* x_3 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->errors_2 = errors_3;
    (&kernelContext_0)->y_0 = y_1;
    (&kernelContext_0)->x_2 = x_3;
    uint i_3 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(i_3 >= (p_1->count_0))
    {
        return;
    }
    DgView_0 yv_0 = dg_view_0((&kernelContext_0)->p_0->yOffset_0, uint4((&kernelContext_0)->p_0->yExtent0_0, (&kernelContext_0)->p_0->yExtent1_0, (&kernelContext_0)->p_0->yExtent2_0, (&kernelContext_0)->p_0->yExtent3_0), uint4((&kernelContext_0)->p_0->yStride0_0, (&kernelContext_0)->p_0->yStride1_0, (&kernelContext_0)->p_0->yStride2_0, (&kernelContext_0)->p_0->yStride3_0));
    thread DgView_0 _S17 = yv_0;
    uint4 _S18 = dg_coords_0(&_S17, i_3);
    thread DgView_0 _S19 = dg_view_0((&kernelContext_0)->p_0->xOffset_0, uint4((&kernelContext_0)->p_0->xExtent0_0, (&kernelContext_0)->p_0->xExtent1_0, (&kernelContext_0)->p_0->xExtent2_0, (&kernelContext_0)->p_0->xExtent3_0), uint4((&kernelContext_0)->p_0->xStride0_0, (&kernelContext_0)->p_0->xStride1_0, (&kernelContext_0)->p_0->xStride2_0, (&kernelContext_0)->p_0->xStride3_0));
    uint _S20 = dg_at_broadcast_0(&_S19, _S18);
    thread DgView_0 _S21 = yv_0;
    uint _S22 = dg_at_0(&_S21, _S18);
    bool _S23 = dg_in_range_0((&kernelContext_0)->errors_2, _S20, (&kernelContext_0)->p_0->xLength_0, i_3);
    bool _S24;
    if(!_S23)
    {
        _S24 = true;
    }
    else
    {
        bool _S25 = dg_in_range_0((&kernelContext_0)->errors_2, _S22, (&kernelContext_0)->p_0->yLength_0, i_3);
        _S24 = !_S25;
    }
    if(_S24)
    {
        return;
    }
    *((&kernelContext_0)->y_0+_S22) = lgUnary_0(*((&kernelContext_0)->x_2+_S20));
    return;
}

