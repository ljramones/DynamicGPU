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

float dg_negative_infinity_0()
{
    return (as_type<float>((4286578688U)));
}

uint dg_at_0(const DgView_0 thread* v_2, uint4 i_1)
{
    return v_2->offset_0 + i_1.x * v_2->stride_0.x + i_1.y * v_2->stride_0.y + i_1.z * v_2->stride_0.z + i_1.w * v_2->stride_0.w;
}

bool dg_is_nan_0(float x_0)
{
    uint _S8 = (as_type<uint>((x_0)));
    bool _S9;
    if((_S8 & 2139095040U) == 2139095040U)
    {
        _S9 = (_S8 & 8388607U) != 0U;
    }
    else
    {
        _S9 = false;
    }
    return _S9;
}

bool beats_0(float vb_0, uint ib_0, float va_0, uint ia_0)
{
    bool nb_0 = dg_is_nan_0(vb_0);
    if(nb_0 != (dg_is_nan_0(va_0)))
    {
        return nb_0;
    }
    bool _S10;
    if(!nb_0)
    {
        _S10 = vb_0 != va_0;
    }
    else
    {
        _S10 = false;
    }
    if(_S10)
    {
        return vb_0 > va_0;
    }
    return ib_0 < ia_0;
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
    uint device* y_0;
    array<float, int(256)> threadgroup* values_0;
    array<uint, int(256)> threadgroup* indices_0;
};

[[kernel]] void reduce_argmax(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(2)]], float device* x_2 [[buffer(0)]], uint device* y_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
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
    bool _S11;
    uint t_0 = local_0.x;
    DgView_0 xv_0 = dg_view_0((&kernelContext_0)->p_0->xOffset_0, uint4((&kernelContext_0)->p_0->xExtent0_0, (&kernelContext_0)->p_0->xExtent1_0, (&kernelContext_0)->p_0->xExtent2_0, (&kernelContext_0)->p_0->xExtent3_0), uint4((&kernelContext_0)->p_0->xStride0_0, (&kernelContext_0)->p_0->xStride1_0, (&kernelContext_0)->p_0->xStride2_0, (&kernelContext_0)->p_0->xStride3_0));
    uint _S12 = xv_0.extent_0.w;
    thread DgView_0 _S13 = xv_0;
    uint3 _S14 = dg_row_0(&_S13, row_1);
    float _S15 = dg_negative_infinity_0();
    uint j_0 = t_0;
    uint at_0 = 4294967295U;
    float best_0 = _S15;
    for(;;)
    {
        if(j_0 < _S12)
        {
        }
        else
        {
            break;
        }
        uint4 _S16 = uint4(_S14, j_0);
        thread DgView_0 _S17 = xv_0;
        uint _S18 = dg_at_0(&_S17, _S16);
        float v_3 = *((&kernelContext_0)->x_1+_S18);
        if(at_0 == 4294967295U)
        {
            _S11 = true;
        }
        else
        {
            _S11 = beats_0(v_3, j_0, best_0, at_0);
        }
        if(_S11)
        {
            at_0 = j_0;
            best_0 = v_3;
        }
        j_0 = j_0 + 256U;
    }
    (*(&kernelContext_0)->values_0)[t_0] = best_0;
    (*(&kernelContext_0)->indices_0)[t_0] = at_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 128U)
    {
        _S11 = ((*(&kernelContext_0)->indices_0)[t_0 + 128U]) != 4294967295U;
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S11 = true;
        }
        else
        {
            uint _S19 = t_0 + 128U;
            _S11 = beats_0((*(&kernelContext_0)->values_0)[_S19], (*(&kernelContext_0)->indices_0)[_S19], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        uint _S20 = t_0 + 128U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S20];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S20];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 64U)
    {
        _S11 = ((*(&kernelContext_0)->indices_0)[t_0 + 64U]) != 4294967295U;
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S11 = true;
        }
        else
        {
            uint _S21 = t_0 + 64U;
            _S11 = beats_0((*(&kernelContext_0)->values_0)[_S21], (*(&kernelContext_0)->indices_0)[_S21], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        uint _S22 = t_0 + 64U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S22];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S22];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 32U)
    {
        _S11 = ((*(&kernelContext_0)->indices_0)[t_0 + 32U]) != 4294967295U;
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S11 = true;
        }
        else
        {
            uint _S23 = t_0 + 32U;
            _S11 = beats_0((*(&kernelContext_0)->values_0)[_S23], (*(&kernelContext_0)->indices_0)[_S23], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        uint _S24 = t_0 + 32U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S24];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S24];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 16U)
    {
        _S11 = ((*(&kernelContext_0)->indices_0)[t_0 + 16U]) != 4294967295U;
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S11 = true;
        }
        else
        {
            uint _S25 = t_0 + 16U;
            _S11 = beats_0((*(&kernelContext_0)->values_0)[_S25], (*(&kernelContext_0)->indices_0)[_S25], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        uint _S26 = t_0 + 16U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S26];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S26];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 8U)
    {
        _S11 = ((*(&kernelContext_0)->indices_0)[t_0 + 8U]) != 4294967295U;
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S11 = true;
        }
        else
        {
            uint _S27 = t_0 + 8U;
            _S11 = beats_0((*(&kernelContext_0)->values_0)[_S27], (*(&kernelContext_0)->indices_0)[_S27], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        uint _S28 = t_0 + 8U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S28];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S28];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 4U)
    {
        _S11 = ((*(&kernelContext_0)->indices_0)[t_0 + 4U]) != 4294967295U;
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S11 = true;
        }
        else
        {
            uint _S29 = t_0 + 4U;
            _S11 = beats_0((*(&kernelContext_0)->values_0)[_S29], (*(&kernelContext_0)->indices_0)[_S29], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        uint _S30 = t_0 + 4U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S30];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S30];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 2U)
    {
        _S11 = ((*(&kernelContext_0)->indices_0)[t_0 + 2U]) != 4294967295U;
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S11 = true;
        }
        else
        {
            uint _S31 = t_0 + 2U;
            _S11 = beats_0((*(&kernelContext_0)->values_0)[_S31], (*(&kernelContext_0)->indices_0)[_S31], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        uint _S32 = t_0 + 2U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S32];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S32];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 1U)
    {
        _S11 = ((*(&kernelContext_0)->indices_0)[t_0 + 1U]) != 4294967295U;
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        if(((*(&kernelContext_0)->indices_0)[t_0]) == 4294967295U)
        {
            _S11 = true;
        }
        else
        {
            uint _S33 = t_0 + 1U;
            _S11 = beats_0((*(&kernelContext_0)->values_0)[_S33], (*(&kernelContext_0)->indices_0)[_S33], (*(&kernelContext_0)->values_0)[t_0], (*(&kernelContext_0)->indices_0)[t_0]);
        }
    }
    else
    {
        _S11 = false;
    }
    if(_S11)
    {
        uint _S34 = t_0 + 1U;
        (*(&kernelContext_0)->values_0)[t_0] = (*(&kernelContext_0)->values_0)[_S34];
        (*(&kernelContext_0)->indices_0)[t_0] = (*(&kernelContext_0)->indices_0)[_S34];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 == 0U)
    {
        uint4 _S35 = uint4(_S14, 0U);
        thread DgView_0 _S36 = dg_view_0((&kernelContext_0)->p_0->yOffset_0, uint4((&kernelContext_0)->p_0->yExtent0_0, (&kernelContext_0)->p_0->yExtent1_0, (&kernelContext_0)->p_0->yExtent2_0, (&kernelContext_0)->p_0->yExtent3_0), uint4((&kernelContext_0)->p_0->yStride0_0, (&kernelContext_0)->p_0->yStride1_0, (&kernelContext_0)->p_0->yStride2_0, (&kernelContext_0)->p_0->yStride3_0));
        uint _S37 = dg_at_0(&_S36, _S35);
        *((&kernelContext_0)->y_0+_S37) = (*(&kernelContext_0)->indices_0)[int(0)];
    }
    return;
}

