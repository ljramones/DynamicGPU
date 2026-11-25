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

uint dg_at_linear_0(const DgView_0 thread* v_3, uint linear_1)
{
    uint4 _S11 = dg_coords_0(v_3, linear_1);
    uint _S12 = dg_at_0(v_3, _S11);
    return _S12;
}

void dg_report_error_0(uint device* errors_0, uint code_0, uint index_0)
{
    uint _S13 = atomic_fetch_or_explicit(((atomic_uint device*)(errors_0+int(0))), code_0, memory_order_relaxed);
    uint _S14 = atomic_fetch_max_explicit(((atomic_uint device*)(errors_0+int(1))), ~index_0, memory_order_relaxed);
    uint _S15 = atomic_fetch_add_explicit(((atomic_uint device*)(errors_0+int(2))), 1U, memory_order_relaxed);
    return;
}

bool dg_in_range_0(uint device* errors_1, uint index_1, uint length_0, uint at_0)
{
    bool _S16;
    if(!DG_DEBUG_0)
    {
        _S16 = true;
    }
    else
    {
        _S16 = index_1 < length_0;
    }
    if(_S16)
    {
        return true;
    }
    dg_report_error_0(errors_1, 1U, at_0);
    return false;
}

struct Params_0
{
    uint count_0;
    uint gridX_2;
    float maxNorm_0;
    uint xLength_0;
    uint xOffset_0;
    uint xExtent0_0;
    uint xStride0_0;
    uint xExtent1_0;
    uint xStride1_0;
    uint xExtent2_0;
    uint xStride2_0;
    uint xExtent3_0;
    uint xStride3_0;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    float device* sumSq_0;
    uint device* errors_2;
    float device* x_0;
};

[[kernel]] void clip_scale(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(3)]], float device* sumSq_1 [[buffer(0)]], uint device* errors_3 [[buffer(2)]], float device* x_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->sumSq_0 = sumSq_1;
    (&kernelContext_0)->errors_2 = errors_3;
    (&kernelContext_0)->x_0 = x_1;
    uint i_2 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(i_2 >= (p_1->count_0))
    {
        return;
    }
    float norm_0 = sqrt(*((&kernelContext_0)->sumSq_0+int(0)));
    if(!(norm_0 > ((&kernelContext_0)->p_0->maxNorm_0)))
    {
        return;
    }
    float f_0 = (&kernelContext_0)->p_0->maxNorm_0 / norm_0;
    thread DgView_0 _S17 = dg_view_0((&kernelContext_0)->p_0->xOffset_0, uint4((&kernelContext_0)->p_0->xExtent0_0, (&kernelContext_0)->p_0->xExtent1_0, (&kernelContext_0)->p_0->xExtent2_0, (&kernelContext_0)->p_0->xExtent3_0), uint4((&kernelContext_0)->p_0->xStride0_0, (&kernelContext_0)->p_0->xStride1_0, (&kernelContext_0)->p_0->xStride2_0, (&kernelContext_0)->p_0->xStride3_0));
    uint _S18 = dg_at_linear_0(&_S17, i_2);
    bool _S19 = dg_in_range_0((&kernelContext_0)->errors_2, _S18, (&kernelContext_0)->p_0->xLength_0, i_2);
    if(_S19)
    {
        *((&kernelContext_0)->x_0+_S18) = *((&kernelContext_0)->x_0+_S18) * f_0;
    }
    return;
}

