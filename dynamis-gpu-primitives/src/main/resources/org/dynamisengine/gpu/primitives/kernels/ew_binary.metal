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
bool dg_is_nan_0(float x_0)
{
    uint _S16 = (as_type<uint>((x_0)));
    bool _S17;
    if((_S16 & 2139095040U) == 2139095040U)
    {
        _S17 = (_S16 & 8388607U) != 0U;
    }
    else
    {
        _S17 = false;
    }
    return _S17;
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
    bool _S18;
    if(a_0 == 0.0)
    {
        _S18 = b_0 == 0.0;
    }
    else
    {
        _S18 = false;
    }
    if(_S18)
    {
        return (as_type<float>(((as_type<uint>((a_0))) & (as_type<uint>((b_0))))));
    }
    float _S19;
    if(a_0 > b_0)
    {
        _S19 = a_0;
    }
    else
    {
        _S19 = b_0;
    }
    return _S19;
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
    bool _S20;
    if(a_1 == 0.0)
    {
        _S20 = b_1 == 0.0;
    }
    else
    {
        _S20 = false;
    }
    if(_S20)
    {
        return (as_type<float>(((as_type<uint>((a_1))) | (as_type<uint>((b_1))))));
    }
    float _S21;
    if(a_1 < b_1)
    {
        _S21 = a_1;
    }
    else
    {
        _S21 = b_1;
    }
    return _S21;
}

float binary_0(float a_2, float b_2)
{
    switch(OP_0)
    {
    case 0U:
        {
            return a_2 + b_2;
        }
    case 1U:
        {
            return a_2 - b_2;
        }
    case 2U:
        {
            return a_2 * b_2;
        }
    case 3U:
        {
            return a_2 / b_2;
        }
    case 4U:
        {
            return dg_max_0(a_2, b_2);
        }
    case 5U:
        {
            return dg_min_0(a_2, b_2);
        }
    case 6U:
        {
            return a_2 / (1.0 + exp(- a_2)) * b_2;
        }
    case 7U:
        {
            float _S22;
            if(a_2 > 0.0)
            {
                _S22 = b_2;
            }
            else
            {
                _S22 = 0.0;
            }
            return _S22;
        }
    case 8U:
        {
            float s_0 = 1.0 / (1.0 + exp(- a_2));
            return b_2 * (s_0 * (1.0 + a_2 * (1.0 - s_0)));
        }
    default:
        {
            float t_0 = 1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (a_2 + 0.04471499845385551 * a_2 * a_2 * a_2))) + 1.0);
            return b_2 * (0.5 * (1.0 + t_0) + 0.5 * a_2 * (1.0 - t_0 * t_0) * (0.79788458347320557 * (1.0 + 0.13414499163627625 * a_2 * a_2)));
        }
    }
}

struct BinaryParams_0
{
    uint count_0;
    uint gridX_2;
    uint aLength_0;
    uint bLength_0;
    uint yLength_0;
    uint aOffset_0;
    uint aExtent0_0;
    uint aStride0_0;
    uint aExtent1_0;
    uint aStride1_0;
    uint aExtent2_0;
    uint aStride2_0;
    uint aExtent3_0;
    uint aStride3_0;
    uint bOffset_0;
    uint bExtent0_0;
    uint bStride0_0;
    uint bExtent1_0;
    uint bStride1_0;
    uint bExtent2_0;
    uint bStride2_0;
    uint bExtent3_0;
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
    BinaryParams_0 constant* p_0;
    uint device* errors_2;
    float device* y_0;
    float device* a_3;
    float device* b_3;
};

[[kernel]] void ew_binary(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], BinaryParams_0 constant* p_1 [[buffer(4)]], uint device* errors_3 [[buffer(3)]], float device* y_1 [[buffer(2)]], float device* a_4 [[buffer(0)]], float device* b_4 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->errors_2 = errors_3;
    (&kernelContext_0)->y_0 = y_1;
    (&kernelContext_0)->a_3 = a_4;
    (&kernelContext_0)->b_3 = b_4;
    uint i_3 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(i_3 >= (p_1->count_0))
    {
        return;
    }
    DgView_0 yv_0 = dg_view_0((&kernelContext_0)->p_0->yOffset_0, uint4((&kernelContext_0)->p_0->yExtent0_0, (&kernelContext_0)->p_0->yExtent1_0, (&kernelContext_0)->p_0->yExtent2_0, (&kernelContext_0)->p_0->yExtent3_0), uint4((&kernelContext_0)->p_0->yStride0_0, (&kernelContext_0)->p_0->yStride1_0, (&kernelContext_0)->p_0->yStride2_0, (&kernelContext_0)->p_0->yStride3_0));
    thread DgView_0 _S23 = yv_0;
    uint4 _S24 = dg_coords_0(&_S23, i_3);
    thread DgView_0 _S25 = dg_view_0((&kernelContext_0)->p_0->aOffset_0, uint4((&kernelContext_0)->p_0->aExtent0_0, (&kernelContext_0)->p_0->aExtent1_0, (&kernelContext_0)->p_0->aExtent2_0, (&kernelContext_0)->p_0->aExtent3_0), uint4((&kernelContext_0)->p_0->aStride0_0, (&kernelContext_0)->p_0->aStride1_0, (&kernelContext_0)->p_0->aStride2_0, (&kernelContext_0)->p_0->aStride3_0));
    uint _S26 = dg_at_broadcast_0(&_S25, _S24);
    thread DgView_0 _S27 = dg_view_0((&kernelContext_0)->p_0->bOffset_0, uint4((&kernelContext_0)->p_0->bExtent0_0, (&kernelContext_0)->p_0->bExtent1_0, (&kernelContext_0)->p_0->bExtent2_0, (&kernelContext_0)->p_0->bExtent3_0), uint4((&kernelContext_0)->p_0->bStride0_0, (&kernelContext_0)->p_0->bStride1_0, (&kernelContext_0)->p_0->bStride2_0, (&kernelContext_0)->p_0->bStride3_0));
    uint _S28 = dg_at_broadcast_0(&_S27, _S24);
    thread DgView_0 _S29 = yv_0;
    uint _S30 = dg_at_0(&_S29, _S24);
    bool _S31 = dg_in_range_0((&kernelContext_0)->errors_2, _S26, (&kernelContext_0)->p_0->aLength_0, i_3);
    bool _S32;
    if(!_S31)
    {
        _S32 = true;
    }
    else
    {
        bool _S33 = dg_in_range_0((&kernelContext_0)->errors_2, _S28, (&kernelContext_0)->p_0->bLength_0, i_3);
        _S32 = !_S33;
    }
    if(_S32)
    {
        _S32 = true;
    }
    else
    {
        bool _S34 = dg_in_range_0((&kernelContext_0)->errors_2, _S30, (&kernelContext_0)->p_0->yLength_0, i_3);
        _S32 = !_S34;
    }
    if(_S32)
    {
        return;
    }
    *((&kernelContext_0)->y_0+_S30) = binary_0(*((&kernelContext_0)->a_3+_S26), *((&kernelContext_0)->b_3+_S28));
    return;
}

