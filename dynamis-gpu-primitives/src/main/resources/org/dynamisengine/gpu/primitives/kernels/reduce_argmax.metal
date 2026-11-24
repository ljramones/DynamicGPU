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

float dg_negative_infinity_0()
{
    return (as_type<float>((4286578688U)));
}

uint dg_at_0(const DgView_0 thread* v_2, uint4 i_1)
{
    return v_2->offset_0 + i_1.x * v_2->stride_0.x + i_1.y * v_2->stride_0.y + i_1.z * v_2->stride_0.z + i_1.w * v_2->stride_0.w;
}

void dg_report_error_0(uint device* errors_0, uint code_0, uint index_0)
{
    uint _S8 = atomic_fetch_or_explicit(((atomic_uint device*)(errors_0+int(0))), code_0, memory_order_relaxed);
    uint _S9 = atomic_fetch_max_explicit(((atomic_uint device*)(errors_0+int(1))), ~index_0, memory_order_relaxed);
    uint _S10 = atomic_fetch_add_explicit(((atomic_uint device*)(errors_0+int(2))), 1U, memory_order_relaxed);
    return;
}

bool dg_in_range_0(uint device* errors_1, uint index_1, uint length_0, uint at_0)
{
    bool _S11;
    if(!DG_DEBUG_0)
    {
        _S11 = true;
    }
    else
    {
        _S11 = index_1 < length_0;
    }
    if(_S11)
    {
        return true;
    }
    dg_report_error_0(errors_1, 1U, at_0);
    return false;
}

bool dg_is_nan_0(float x_0)
{
    uint _S12 = (as_type<uint>((x_0)));
    bool _S13;
    if((_S12 & 2139095040U) == 2139095040U)
    {
        _S13 = (_S12 & 8388607U) != 0U;
    }
    else
    {
        _S13 = false;
    }
    return _S13;
}

bool beats_0(float vb_0, uint ib_0, float va_0, uint ia_0)
{
    bool nb_0 = dg_is_nan_0(vb_0);
    if(nb_0 != (dg_is_nan_0(va_0)))
    {
        return nb_0;
    }
    bool _S14;
    if(!nb_0)
    {
        _S14 = vb_0 != va_0;
    }
    else
    {
        _S14 = false;
    }
    if(_S14)
    {
        return vb_0 > va_0;
    }
    return ib_0 < ia_0;
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
    uint device* y_0;
    array<float, int(256)> threadgroup* values_0;
    array<uint, int(256)> threadgroup* indices_0;
};

[[kernel]] void reduce_argmax(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(3)]], uint device* errors_3 [[buffer(2)]], float device* x_2 [[buffer(0)]], uint device* y_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->errors_2 = errors_3;
    (&kernelContext_0)->x_1 = x_2;
    (&kernelContext_0)->y_0 = y_1;
    threadgroup array<float, int(256)> values_1;
    (&kernelContext_0)->values_0 = &values_1;
    threadgroup array<uint, int(256)> indices_1;
    (&kernelContext_0)->indices_0 = &indices_1;
    uint row_1 = dg_linear_group_0(group_1, p_1->gridX_1);
    if(row_1 >= (p_1->rows_0))
    {
        return;
    }
    bool _S15;
    uint t_0 = local_0.x;
    DgView_0 xv_0 = dg_view_0((&kernelContext_0)->p_0->xOffset_0, uint4((&kernelContext_0)->p_0->xExtent0_0, (&kernelContext_0)->p_0->xExtent1_0, (&kernelContext_0)->p_0->xExtent2_0, (&kernelContext_0)->p_0->xExtent3_0), uint4((&kernelContext_0)->p_0->xStride0_0, (&kernelContext_0)->p_0->xStride1_0, (&kernelContext_0)->p_0->xStride2_0, (&kernelContext_0)->p_0->xStride3_0));
    uint _S16 = xv_0.extent_0.w;
    thread DgView_0 _S17 = xv_0;
    uint3 _S18 = dg_row_0(&_S17, row_1);
    float _S19 = dg_negative_infinity_0();
    uint j_0 = t_0;
    uint at_1 = 4294967295U;
    float best_0 = _S19;
    for(;;)
    {
        if(j_0 < _S16)
        {
        }
        else
        {
            break;
        }
        uint4 _S20 = uint4(_S18, j_0);
        thread DgView_0 _S21 = xv_0;
        uint _S22 = dg_at_0(&_S21, _S20);
        bool _S23 = dg_in_range_0((&kernelContext_0)->errors_2, _S22, (&kernelContext_0)->p_0->xLength_0, row_1);
        if(!_S23)
        {
            j_0 = j_0 + 256U;
            continue;
        }
        float v_3 = *((&kernelContext_0)->x_1+_S22);
        if(at_1 == 4294967295U)
        {
            _S15 = true;
        }
        else
        {
            _S15 = beats_0(v_3, j_0, best_0, at_1);
        }
        uint at_2;
        float best_1;
        if(_S15)
        {
            at_2 = j_0;
            best_1 = v_3;
        }
        else
        {
            at_2 = at_1;
            best_1 = best_0;
        }
        at_1 = at_2;
        best_0 = best_1;
        j_0 = j_0 + 256U;
    }
    (*(&kernelContext_0)->values_0)[t_0] = best_0;
    (*(&kernelContext_0)->indices_0)[t_0] = at_1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 128U)
    {
        _S15 = ((*(&kernelContext_0)->indices_0)[t_0 + 128U]) != 4294967295U;
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S15 = true;
        }
        else
        {
            uint _S24 = t_0 + 128U;
            _S15 = beats_0((*(&kernelContext_0)->values_0)[_S24], (*(&kernelContext_0)->indices_0)[_S24], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        uint _S25 = t_0 + 128U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S25];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S25];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 64U)
    {
        _S15 = ((*(&kernelContext_0)->indices_0)[t_0 + 64U]) != 4294967295U;
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S15 = true;
        }
        else
        {
            uint _S26 = t_0 + 64U;
            _S15 = beats_0((*(&kernelContext_0)->values_0)[_S26], (*(&kernelContext_0)->indices_0)[_S26], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        uint _S27 = t_0 + 64U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S27];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S27];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 32U)
    {
        _S15 = ((*(&kernelContext_0)->indices_0)[t_0 + 32U]) != 4294967295U;
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S15 = true;
        }
        else
        {
            uint _S28 = t_0 + 32U;
            _S15 = beats_0((*(&kernelContext_0)->values_0)[_S28], (*(&kernelContext_0)->indices_0)[_S28], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        uint _S29 = t_0 + 32U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S29];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S29];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 16U)
    {
        _S15 = ((*(&kernelContext_0)->indices_0)[t_0 + 16U]) != 4294967295U;
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S15 = true;
        }
        else
        {
            uint _S30 = t_0 + 16U;
            _S15 = beats_0((*(&kernelContext_0)->values_0)[_S30], (*(&kernelContext_0)->indices_0)[_S30], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        uint _S31 = t_0 + 16U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S31];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S31];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 8U)
    {
        _S15 = ((*(&kernelContext_0)->indices_0)[t_0 + 8U]) != 4294967295U;
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S15 = true;
        }
        else
        {
            uint _S32 = t_0 + 8U;
            _S15 = beats_0((*(&kernelContext_0)->values_0)[_S32], (*(&kernelContext_0)->indices_0)[_S32], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        uint _S33 = t_0 + 8U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S33];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S33];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 4U)
    {
        _S15 = ((*(&kernelContext_0)->indices_0)[t_0 + 4U]) != 4294967295U;
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S15 = true;
        }
        else
        {
            uint _S34 = t_0 + 4U;
            _S15 = beats_0((*(&kernelContext_0)->values_0)[_S34], (*(&kernelContext_0)->indices_0)[_S34], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        uint _S35 = t_0 + 4U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S35];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S35];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 2U)
    {
        _S15 = ((*(&kernelContext_0)->indices_0)[t_0 + 2U]) != 4294967295U;
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S15 = true;
        }
        else
        {
            uint _S36 = t_0 + 2U;
            _S15 = beats_0((*(&kernelContext_0)->values_0)[_S36], (*(&kernelContext_0)->indices_0)[_S36], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        uint _S37 = t_0 + 2U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S37];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S37];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 1U)
    {
        _S15 = ((*(&kernelContext_0)->indices_0)[t_0 + 1U]) != 4294967295U;
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S15 = true;
        }
        else
        {
            uint _S38 = t_0 + 1U;
            _S15 = beats_0((*(&kernelContext_0)->values_0)[_S38], (*(&kernelContext_0)->indices_0)[_S38], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        uint _S39 = t_0 + 1U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S39];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S39];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint4 _S40 = uint4(_S18, 0U);
    thread DgView_0 _S41 = dg_view_0((&kernelContext_0)->p_0->yOffset_0, uint4((&kernelContext_0)->p_0->yExtent0_0, (&kernelContext_0)->p_0->yExtent1_0, (&kernelContext_0)->p_0->yExtent2_0, (&kernelContext_0)->p_0->yExtent3_0), uint4((&kernelContext_0)->p_0->yStride0_0, (&kernelContext_0)->p_0->yStride1_0, (&kernelContext_0)->p_0->yStride2_0, (&kernelContext_0)->p_0->yStride3_0));
    uint _S42 = dg_at_0(&_S41, _S40);
    if(t_0 == 0U)
    {
        bool _S43 = dg_in_range_0((&kernelContext_0)->errors_2, _S42, (&kernelContext_0)->p_0->yLength_0, row_1);
        _S15 = _S43;
    }
    else
    {
        _S15 = false;
    }
    if(_S15)
    {
        *((&kernelContext_0)->y_0+_S42) = (*(&kernelContext_0)->indices_0)[int(0)];
    }
    return;
}

