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

void dg_report_error_0(uint device* errors_0, uint code_0, uint index_0)
{
    uint _S11 = atomic_fetch_or_explicit(((atomic_uint device*)(errors_0+int(0))), code_0, memory_order_relaxed);
    uint _S12 = atomic_fetch_max_explicit(((atomic_uint device*)(errors_0+int(1))), ~index_0, memory_order_relaxed);
    uint _S13 = atomic_fetch_add_explicit(((atomic_uint device*)(errors_0+int(2))), 1U, memory_order_relaxed);
    return;
}

bool dg_in_range_0(uint device* errors_1, uint index_1, uint length_0, uint at_0)
{
    bool _S14;
    if(!DG_DEBUG_0)
    {
        _S14 = true;
    }
    else
    {
        _S14 = index_1 < length_0;
    }
    if(_S14)
    {
        return true;
    }
    dg_report_error_0(errors_1, 1U, at_0);
    return false;
}

struct LogicSelectParams_0
{
    uint count_0;
    uint gridX_2;
    uint mLength_0;
    uint aLength_0;
    uint bLength_0;
    uint yLength_0;
    uint mOffset_0;
    uint mStride0_0;
    uint mStride1_0;
    uint mStride2_0;
    uint mStride3_0;
    uint aOffset_0;
    uint aStride0_0;
    uint aStride1_0;
    uint aStride2_0;
    uint aStride3_0;
    uint bOffset_0;
    uint bStride0_0;
    uint bStride1_0;
    uint bStride2_0;
    uint bStride3_0;
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
    LogicSelectParams_0 constant* p_0;
    uint device* errors_2;
    uint device* y_0;
    uint device* m_0;
    uint device* a_0;
    uint device* b_0;
};

[[kernel]] void logic_select(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], LogicSelectParams_0 constant* p_1 [[buffer(5)]], uint device* errors_3 [[buffer(4)]], uint device* y_1 [[buffer(3)]], uint device* m_1 [[buffer(0)]], uint device* a_1 [[buffer(1)]], uint device* b_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->errors_2 = errors_3;
    (&kernelContext_0)->y_0 = y_1;
    (&kernelContext_0)->m_0 = m_1;
    (&kernelContext_0)->a_0 = a_1;
    (&kernelContext_0)->b_0 = b_1;
    uint i_2 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(i_2 >= (p_1->count_0))
    {
        return;
    }
    DgView_0 yv_0 = dg_view_0((&kernelContext_0)->p_0->yOffset_0, uint4((&kernelContext_0)->p_0->yExtent0_0, (&kernelContext_0)->p_0->yExtent1_0, (&kernelContext_0)->p_0->yExtent2_0, (&kernelContext_0)->p_0->yExtent3_0), uint4((&kernelContext_0)->p_0->yStride0_0, (&kernelContext_0)->p_0->yStride1_0, (&kernelContext_0)->p_0->yStride2_0, (&kernelContext_0)->p_0->yStride3_0));
    thread DgView_0 _S15 = yv_0;
    uint4 _S16 = dg_coords_0(&_S15, i_2);
    uint _S17 = _S16.x;
    uint _S18 = _S16.y;
    uint _S19 = _S16.z;
    uint _S20 = _S16.w;
    uint mi_0 = (&kernelContext_0)->p_0->mOffset_0 + _S17 * (&kernelContext_0)->p_0->mStride0_0 + _S18 * (&kernelContext_0)->p_0->mStride1_0 + _S19 * (&kernelContext_0)->p_0->mStride2_0 + _S20 * (&kernelContext_0)->p_0->mStride3_0;
    uint ai_0 = (&kernelContext_0)->p_0->aOffset_0 + _S17 * (&kernelContext_0)->p_0->aStride0_0 + _S18 * (&kernelContext_0)->p_0->aStride1_0 + _S19 * (&kernelContext_0)->p_0->aStride2_0 + _S20 * (&kernelContext_0)->p_0->aStride3_0;
    uint bi_0 = (&kernelContext_0)->p_0->bOffset_0 + _S17 * (&kernelContext_0)->p_0->bStride0_0 + _S18 * (&kernelContext_0)->p_0->bStride1_0 + _S19 * (&kernelContext_0)->p_0->bStride2_0 + _S20 * (&kernelContext_0)->p_0->bStride3_0;
    thread DgView_0 _S21 = yv_0;
    uint _S22 = dg_at_0(&_S21, _S16);
    bool _S23 = dg_in_range_0((&kernelContext_0)->errors_2, mi_0, (&kernelContext_0)->p_0->mLength_0, i_2);
    bool _S24;
    if(!_S23)
    {
        _S24 = true;
    }
    else
    {
        bool _S25 = dg_in_range_0((&kernelContext_0)->errors_2, ai_0, (&kernelContext_0)->p_0->aLength_0, i_2);
        _S24 = !_S25;
    }
    if(_S24)
    {
        _S24 = true;
    }
    else
    {
        bool _S26 = dg_in_range_0((&kernelContext_0)->errors_2, bi_0, (&kernelContext_0)->p_0->bLength_0, i_2);
        _S24 = !_S26;
    }
    if(_S24)
    {
        _S24 = true;
    }
    else
    {
        bool _S27 = dg_in_range_0((&kernelContext_0)->errors_2, _S22, (&kernelContext_0)->p_0->yLength_0, i_2);
        _S24 = !_S27;
    }
    if(_S24)
    {
        return;
    }
    uint device* _S28 = (&kernelContext_0)->y_0+_S22;
    uint _S29;
    if((*((&kernelContext_0)->m_0+mi_0)) != 0U)
    {
        _S29 = *((&kernelContext_0)->a_0+ai_0);
    }
    else
    {
        _S29 = *((&kernelContext_0)->b_0+bi_0);
    }
    *_S28 = _S29;
    return;
}

