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

bool castIs16_0(uint t_0)
{
    bool _S1;
    if(t_0 == 1U)
    {
        _S1 = true;
    }
    else
    {
        _S1 = t_0 == 2U;
    }
    return _S1;
}

constant uint fc_TO_0 [[function_constant(1)]];
constant uint TO_0 = is_function_constant_defined(fc_TO_0) ? fc_TO_0 : 0U;
uint4 dg_coords_0(const DgView_0 thread* v_1, uint linear_0)
{
    thread uint4 i_0;
    uint4 _S2 = v_1->extent_0;
    uint _S3 = v_1->extent_0.w;
    uint _S4 = linear_0 % _S3;
    i_0.w = _S4;
    uint _S5 = linear_0 / _S3;
    uint _S6 = _S2.z;
    uint _S7 = _S5 % _S6;
    i_0.z = _S7;
    uint _S8 = _S5 / _S6;
    uint _S9 = _S2.y;
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

uint dg_at_broadcast_0(const DgView_0 thread* v_3, uint4 i_2)
{
    uint _S16 = dg_at_0(v_3, select(i_2, uint4(0U, 0U, 0U, 0U), (v_3->extent_0) == uint4(1U, 1U, 1U, 1U)));
    return _S16;
}

constant uint fc_FROM_0 [[function_constant(0)]];
constant uint FROM_0 = is_function_constant_defined(fc_FROM_0) ? fc_FROM_0 : 0U;
struct CastParams_0
{
    uint count_0;
    uint gridX_2;
    uint xLength_0;
    uint yLength_0;
    uint elements_0;
    uint firstWord_0;
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
    CastParams_0 constant* p_0;
    uint device* errors_2;
    uint device* y_0;
    uint device* x_0;
};

uint load_0(uint4 c_0, uint at_1, KernelContext_0 thread* kernelContext_0)
{
    thread DgView_0 _S17 = dg_view_0(kernelContext_0->p_0->xOffset_0, uint4(kernelContext_0->p_0->xExtent0_0, kernelContext_0->p_0->xExtent1_0, kernelContext_0->p_0->xExtent2_0, kernelContext_0->p_0->xExtent3_0), uint4(kernelContext_0->p_0->xStride0_0, kernelContext_0->p_0->xStride1_0, kernelContext_0->p_0->xStride2_0, kernelContext_0->p_0->xStride3_0));
    uint _S18 = dg_at_broadcast_0(&_S17, c_0);
    if(castIs16_0(FROM_0))
    {
        uint _S19 = _S18 >> 1U;
        bool _S20 = dg_in_range_0(kernelContext_0->errors_2, _S19, kernelContext_0->p_0->xLength_0, at_1);
        if(!_S20)
        {
            return 0U;
        }
        return ((*(kernelContext_0->x_0+_S19)) >> (16U * (_S18 & 1U))) & 65535U;
    }
    bool _S21 = dg_in_range_0(kernelContext_0->errors_2, _S18, kernelContext_0->p_0->xLength_0, at_1);
    if(!_S21)
    {
        return 0U;
    }
    return *(kernelContext_0->x_0+_S18);
}

bool isNonzero_0(uint v_4)
{
    switch(FROM_0)
    {
    case 0U:
        {
            return (v_4 & 2147483647U) != 0U;
        }
    case 1U:
        {
            return (v_4 & 32767U) != 0U;
        }
    case 2U:
        {
            return (v_4 & 32767U) != 0U;
        }
    default:
        {
            return v_4 != 0U;
        }
    }
}

uint firstbithigh_0(uint value_0)
{
    uint _S22;
    if(int(value_0) < int(0))
    {
        _S22 = ~value_0;
    }
    else
    {
        _S22 = value_0;
    }
    if(_S22 == 0U)
    {
        return 4294967295U;
    }
    uint _S23 = clz(_S22);
    return 31U - _S23;
}

uint f16ToF32_0(uint h_0)
{
    uint sign_0 = (h_0 & 32768U) << 16U;
    uint e_0 = (h_0 >> 10U) & 31U;
    uint m_0 = h_0 & 1023U;
    if(e_0 == 31U)
    {
        return (sign_0 | 2139095040U) | (m_0 << 13U);
    }
    if(e_0 == 0U)
    {
        if(m_0 == 0U)
        {
            return sign_0;
        }
        uint top_0 = firstbithigh_0(m_0);
        return (sign_0 | ((top_0 + 103U) << 23U)) | (((m_0 << (10U - top_0)) & 1023U) << 13U);
    }
    return (sign_0 | ((e_0 + 112U) << 23U)) | (m_0 << 13U);
}

uint i32ToF32_0(uint v_5)
{
    if(v_5 == 0U)
    {
        return 0U;
    }
    uint sign_1 = v_5 & 2147483648U;
    uint mag_0;
    if(sign_1 != 0U)
    {
        mag_0 = 0U - v_5;
    }
    else
    {
        mag_0 = v_5;
    }
    uint top_1 = firstbithigh_0(mag_0);
    uint m_1;
    uint top_2;
    if(top_1 <= 23U)
    {
        uint _S24 = mag_0 << (23U - top_1);
        top_2 = top_1;
        m_1 = _S24;
    }
    else
    {
        uint shift_0 = top_1 - 23U;
        uint m_2 = mag_0 >> shift_0;
        uint rem_0 = mag_0 & ((1U << shift_0) - 1U);
        uint halfway_0 = 1U << (shift_0 - 1U);
        bool _S25;
        if(rem_0 > halfway_0)
        {
            _S25 = true;
        }
        else
        {
            if(rem_0 == halfway_0)
            {
                _S25 = (m_2 & 1U) != 0U;
            }
            else
            {
                _S25 = false;
            }
        }
        if(_S25)
        {
            m_1 = m_2 + 1U;
        }
        else
        {
            m_1 = m_2;
        }
        if(m_1 == 16777216U)
        {
            uint m_3 = m_1 >> 1U;
            top_2 = top_1 + 1U;
            m_1 = m_3;
        }
        else
        {
            top_2 = top_1;
        }
    }
    return (sign_1 | ((top_2 + 127U) << 23U)) | (m_1 & 8388607U);
}

uint toF32_0(uint v_6)
{
    switch(FROM_0)
    {
    case 0U:
        {
            return v_6;
        }
    case 1U:
        {
            return f16ToF32_0(v_6);
        }
    case 2U:
        {
            return v_6 << 16U;
        }
    case 3U:
        {
            return i32ToF32_0(v_6);
        }
    default:
        {
            uint _S26;
            if(v_6 != 0U)
            {
                _S26 = 1065353216U;
            }
            else
            {
                _S26 = 0U;
            }
            return _S26;
        }
    }
}

uint f32ToI32_0(uint x_1)
{
    uint e_1 = (x_1 >> 23U) & 255U;
    if((x_1 & 2147483647U) > 2139095040U)
    {
        return 0U;
    }
    bool neg_0 = (x_1 >> 31U) != 0U;
    if(e_1 < 127U)
    {
        return 0U;
    }
    uint mag_1;
    if(e_1 >= 158U)
    {
        if(neg_0)
        {
            mag_1 = 2147483648U;
        }
        else
        {
            mag_1 = 2147483647U;
        }
        return mag_1;
    }
    uint m_4 = (x_1 & 8388607U) | 8388608U;
    if(e_1 >= 150U)
    {
        mag_1 = m_4 << (e_1 - 150U);
    }
    else
    {
        mag_1 = m_4 >> (150U - e_1);
    }
    if(neg_0)
    {
        mag_1 = 0U - mag_1;
    }
    return mag_1;
}

uint f32ToF16_0(uint x_2)
{
    uint sign_2 = (x_2 >> 16U) & 32768U;
    uint e_2 = (x_2 >> 23U) & 255U;
    uint mant_0 = x_2 & 8388607U;
    uint h_1;
    if(e_2 == 255U)
    {
        uint _S27 = sign_2 | 31744U;
        if(mant_0 != 0U)
        {
            h_1 = 512U | (mant_0 >> 13U);
        }
        else
        {
            h_1 = 0U;
        }
        return _S27 | h_1;
    }
    int exp_0 = int(e_2) - int(112);
    if(exp_0 >= int(31))
    {
        return sign_2 | 31744U;
    }
    bool _S28;
    if(exp_0 <= int(0))
    {
        if(exp_0 < int(-10))
        {
            return sign_2;
        }
        uint mant_1 = mant_0 | 8388608U;
        uint shift_1 = uint(int(14) - exp_0);
        uint h_2 = mant_1 >> shift_1;
        uint rem_1 = mant_1 & ((1U << shift_1) - 1U);
        uint halfway_1 = 1U << (shift_1 - 1U);
        if(rem_1 > halfway_1)
        {
            _S28 = true;
        }
        else
        {
            if(rem_1 == halfway_1)
            {
                _S28 = (h_2 & 1U) != 0U;
            }
            else
            {
                _S28 = false;
            }
        }
        if(_S28)
        {
            h_1 = h_2 + 1U;
        }
        else
        {
            h_1 = h_2;
        }
        return sign_2 | h_1;
    }
    uint h_3 = (uint(exp_0) << 10U) | (mant_0 >> 13U);
    uint rem_2 = mant_0 & 8191U;
    if(rem_2 > 4096U)
    {
        _S28 = true;
    }
    else
    {
        if(rem_2 == 4096U)
        {
            _S28 = (h_3 & 1U) != 0U;
        }
        else
        {
            _S28 = false;
        }
    }
    if(_S28)
    {
        h_1 = h_3 + 1U;
    }
    else
    {
        h_1 = h_3;
    }
    return sign_2 | h_1;
}

uint f32ToBf16_0(uint x_3)
{
    if((x_3 & 2147483647U) > 2139095040U)
    {
        return (x_3 >> 16U) | 64U;
    }
    return (x_3 + 32767U + ((x_3 >> 16U) & 1U)) >> 16U;
}

uint convert_0(uint v_7)
{
    uint _S29;
    if(TO_0 == 4U)
    {
        if(isNonzero_0(v_7))
        {
            _S29 = 1U;
        }
        else
        {
            _S29 = 0U;
        }
        return _S29;
    }
    if(TO_0 == 3U)
    {
        if(FROM_0 == 3U)
        {
            return v_7;
        }
        if(FROM_0 == 4U)
        {
            if(v_7 != 0U)
            {
                _S29 = 1U;
            }
            else
            {
                _S29 = 0U;
            }
            return _S29;
        }
        return f32ToI32_0(toF32_0(v_7));
    }
    if(TO_0 == FROM_0)
    {
        return v_7;
    }
    uint f_0 = toF32_0(v_7);
    switch(TO_0)
    {
    case 0U:
        {
            return f_0;
        }
    case 1U:
        {
            return f32ToF16_0(f_0);
        }
    default:
        {
            return f32ToBf16_0(f_0);
        }
    }
}

[[kernel]] void cast(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], CastParams_0 constant* p_1 [[buffer(3)]], uint device* errors_3 [[buffer(2)]], uint device* y_1 [[buffer(1)]], uint device* x_4 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->p_0 = p_1;
    (&kernelContext_1)->errors_2 = errors_3;
    (&kernelContext_1)->y_0 = y_1;
    (&kernelContext_1)->x_0 = x_4;
    uint i_3 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(i_3 >= (p_1->count_0))
    {
        return;
    }
    DgView_0 yv_0 = dg_view_0((&kernelContext_1)->p_0->yOffset_0, uint4((&kernelContext_1)->p_0->yExtent0_0, (&kernelContext_1)->p_0->yExtent1_0, (&kernelContext_1)->p_0->yExtent2_0, (&kernelContext_1)->p_0->yExtent3_0), uint4((&kernelContext_1)->p_0->yStride0_0, (&kernelContext_1)->p_0->yStride1_0, (&kernelContext_1)->p_0->yStride2_0, (&kernelContext_1)->p_0->yStride3_0));
    if(!castIs16_0(TO_0))
    {
        thread DgView_0 _S30 = yv_0;
        uint4 _S31 = dg_coords_0(&_S30, i_3);
        thread DgView_0 _S32 = yv_0;
        uint _S33 = dg_at_0(&_S32, _S31);
        bool _S34 = dg_in_range_0((&kernelContext_1)->errors_2, _S33, (&kernelContext_1)->p_0->yLength_0, i_3);
        if(!_S34)
        {
            return;
        }
        uint device* _S35 = (&kernelContext_1)->y_0+_S33;
        uint _S36 = load_0(_S31, i_3, &kernelContext_1);
        *_S35 = convert_0(_S36);
        return;
    }
    uint word_0 = (&kernelContext_1)->p_0->firstWord_0 + i_3;
    bool _S37 = dg_in_range_0((&kernelContext_1)->errors_2, word_0, (&kernelContext_1)->p_0->yLength_0, i_3);
    if(!_S37)
    {
        return;
    }
    uint _S38 = *((&kernelContext_1)->y_0+word_0);
    uint _S39 = 2U * word_0;
    uint e_3 = _S39 - yv_0.offset_0;
    bool _S40;
    if(_S39 >= (yv_0.offset_0))
    {
        _S40 = e_3 < ((&kernelContext_1)->p_0->elements_0);
    }
    else
    {
        _S40 = false;
    }
    uint merged_0;
    if(_S40)
    {
        thread DgView_0 _S41 = yv_0;
        uint4 _S42 = dg_coords_0(&_S41, e_3);
        uint _S43 = load_0(_S42, i_3, &kernelContext_1);
        merged_0 = (_S38 & 4294901760U) | (((convert_0(_S43)) & 65535U) << 0U);
    }
    else
    {
        merged_0 = _S38;
    }
    uint slot_0 = _S39 + 1U;
    uint e_4 = slot_0 - yv_0.offset_0;
    if(slot_0 >= (yv_0.offset_0))
    {
        _S40 = e_4 < ((&kernelContext_1)->p_0->elements_0);
    }
    else
    {
        _S40 = false;
    }
    if(_S40)
    {
        thread DgView_0 _S44 = yv_0;
        uint4 _S45 = dg_coords_0(&_S44, e_4);
        uint _S46 = load_0(_S45, i_3, &kernelContext_1);
        merged_0 = (merged_0 & 65535U) | (((convert_0(_S46)) & 65535U) << 16U);
    }
    *((&kernelContext_1)->y_0+word_0) = merged_0;
    return;
}

