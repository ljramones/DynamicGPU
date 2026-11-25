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

int lgOrderF32_0(uint a_0, uint b_0)
{
    if(((a_0 | b_0) & 2147483647U) == 0U)
    {
        return int(0);
    }
    uint ka_0;
    if((a_0 & 2147483648U) != 0U)
    {
        ka_0 = ~a_0;
    }
    else
    {
        ka_0 = a_0 | 2147483648U;
    }
    uint kb_0;
    if((b_0 & 2147483648U) != 0U)
    {
        kb_0 = ~b_0;
    }
    else
    {
        kb_0 = b_0 | 2147483648U;
    }
    int _S16;
    if(ka_0 < kb_0)
    {
        _S16 = int(-1);
    }
    else
    {
        if(ka_0 > kb_0)
        {
            _S16 = int(1);
        }
        else
        {
            _S16 = int(0);
        }
    }
    return _S16;
}

uint lgCompare_0(uint cmp_0, int order_0)
{
    bool r_0;
    switch(cmp_0)
    {
    case 0U:
        {
            r_0 = order_0 == int(0);
            break;
        }
    case 1U:
        {
            r_0 = order_0 != int(0);
            break;
        }
    case 2U:
        {
            r_0 = order_0 < int(0);
            break;
        }
    case 3U:
        {
            r_0 = order_0 <= int(0);
            break;
        }
    case 4U:
        {
            r_0 = order_0 > int(0);
            break;
        }
    default:
        {
            r_0 = order_0 >= int(0);
            break;
        }
    }
    uint _S17;
    if(r_0)
    {
        _S17 = 1U;
    }
    else
    {
        _S17 = 0U;
    }
    return _S17;
}

uint lgBinary_0(uint a_1, uint b_1)
{
    uint _S18;
    bool _S19;
    if(OP_0 < 6U)
    {
        if(lgIsNaN_0(a_1))
        {
            _S19 = true;
        }
        else
        {
            _S19 = lgIsNaN_0(b_1);
        }
        if(_S19)
        {
            if(OP_0 == 1U)
            {
                _S18 = 1U;
            }
            else
            {
                _S18 = 0U;
            }
            return _S18;
        }
        return lgCompare_0(OP_0, lgOrderF32_0(a_1, b_1));
    }
    if(OP_0 < 12U)
    {
        int ia_0 = int(a_1);
        int ib_0 = int(b_1);
        uint _S20 = OP_0 - 6U;
        int _S21;
        if(ia_0 < ib_0)
        {
            _S21 = int(-1);
        }
        else
        {
            if(ia_0 > ib_0)
            {
                _S21 = int(1);
            }
            else
            {
                _S21 = int(0);
            }
        }
        return lgCompare_0(_S20, _S21);
    }
    switch(OP_0)
    {
    case 12U:
        {
            if(a_1 != 0U)
            {
                _S19 = b_1 != 0U;
            }
            else
            {
                _S19 = false;
            }
            if(_S19)
            {
                _S18 = 1U;
            }
            else
            {
                _S18 = 0U;
            }
            return _S18;
        }
    case 13U:
        {
            if(a_1 != 0U)
            {
                _S19 = true;
            }
            else
            {
                _S19 = b_1 != 0U;
            }
            if(_S19)
            {
                _S18 = 1U;
            }
            else
            {
                _S18 = 0U;
            }
            return _S18;
        }
    case 14U:
        {
            if((a_1 != 0U) != (b_1 != 0U))
            {
                _S18 = 1U;
            }
            else
            {
                _S18 = 0U;
            }
            return _S18;
        }
    case 15U:
        {
            return a_1 + b_1;
        }
    case 16U:
        {
            return a_1 - b_1;
        }
    case 17U:
        {
            return a_1 * b_1;
        }
    case 18U:
        {
            if(int(a_1) < int(b_1))
            {
                _S18 = a_1;
            }
            else
            {
                _S18 = b_1;
            }
            return _S18;
        }
    default:
        {
            if(int(a_1) > int(b_1))
            {
                _S18 = a_1;
            }
            else
            {
                _S18 = b_1;
            }
            return _S18;
        }
    }
}

struct LogicBinaryParams_0
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
    LogicBinaryParams_0 constant* p_0;
    uint device* errors_2;
    uint device* y_0;
    uint device* a_2;
    uint device* b_2;
};

[[kernel]] void logic_binary(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], LogicBinaryParams_0 constant* p_1 [[buffer(4)]], uint device* errors_3 [[buffer(3)]], uint device* y_1 [[buffer(2)]], uint device* a_3 [[buffer(0)]], uint device* b_3 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->errors_2 = errors_3;
    (&kernelContext_0)->y_0 = y_1;
    (&kernelContext_0)->a_2 = a_3;
    (&kernelContext_0)->b_2 = b_3;
    uint i_3 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(i_3 >= (p_1->count_0))
    {
        return;
    }
    DgView_0 yv_0 = dg_view_0((&kernelContext_0)->p_0->yOffset_0, uint4((&kernelContext_0)->p_0->yExtent0_0, (&kernelContext_0)->p_0->yExtent1_0, (&kernelContext_0)->p_0->yExtent2_0, (&kernelContext_0)->p_0->yExtent3_0), uint4((&kernelContext_0)->p_0->yStride0_0, (&kernelContext_0)->p_0->yStride1_0, (&kernelContext_0)->p_0->yStride2_0, (&kernelContext_0)->p_0->yStride3_0));
    thread DgView_0 _S22 = yv_0;
    uint4 _S23 = dg_coords_0(&_S22, i_3);
    thread DgView_0 _S24 = dg_view_0((&kernelContext_0)->p_0->aOffset_0, uint4((&kernelContext_0)->p_0->aExtent0_0, (&kernelContext_0)->p_0->aExtent1_0, (&kernelContext_0)->p_0->aExtent2_0, (&kernelContext_0)->p_0->aExtent3_0), uint4((&kernelContext_0)->p_0->aStride0_0, (&kernelContext_0)->p_0->aStride1_0, (&kernelContext_0)->p_0->aStride2_0, (&kernelContext_0)->p_0->aStride3_0));
    uint _S25 = dg_at_broadcast_0(&_S24, _S23);
    thread DgView_0 _S26 = dg_view_0((&kernelContext_0)->p_0->bOffset_0, uint4((&kernelContext_0)->p_0->bExtent0_0, (&kernelContext_0)->p_0->bExtent1_0, (&kernelContext_0)->p_0->bExtent2_0, (&kernelContext_0)->p_0->bExtent3_0), uint4((&kernelContext_0)->p_0->bStride0_0, (&kernelContext_0)->p_0->bStride1_0, (&kernelContext_0)->p_0->bStride2_0, (&kernelContext_0)->p_0->bStride3_0));
    uint _S27 = dg_at_broadcast_0(&_S26, _S23);
    thread DgView_0 _S28 = yv_0;
    uint _S29 = dg_at_0(&_S28, _S23);
    bool _S30 = dg_in_range_0((&kernelContext_0)->errors_2, _S25, (&kernelContext_0)->p_0->aLength_0, i_3);
    bool _S31;
    if(!_S30)
    {
        _S31 = true;
    }
    else
    {
        bool _S32 = dg_in_range_0((&kernelContext_0)->errors_2, _S27, (&kernelContext_0)->p_0->bLength_0, i_3);
        _S31 = !_S32;
    }
    if(_S31)
    {
        _S31 = true;
    }
    else
    {
        bool _S33 = dg_in_range_0((&kernelContext_0)->errors_2, _S29, (&kernelContext_0)->p_0->yLength_0, i_3);
        _S31 = !_S33;
    }
    if(_S31)
    {
        return;
    }
    *((&kernelContext_0)->y_0+_S29) = lgBinary_0(*((&kernelContext_0)->a_2+_S25), *((&kernelContext_0)->b_2+_S27));
    return;
}

