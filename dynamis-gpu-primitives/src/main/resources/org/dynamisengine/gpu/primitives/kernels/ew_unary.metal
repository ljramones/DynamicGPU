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

constant uint fc_OP_0 [[function_constant(0)]];
constant uint OP_0 = is_function_constant_defined(fc_OP_0) ? fc_OP_0 : 0U;
float unary_0(float x_0)
{
    switch(OP_0)
    {
    case 0U:
        {
            return x_0;
        }
    case 1U:
        {
            return (as_type<float>(((as_type<uint>((x_0))) ^ 2147483648U)));
        }
    case 2U:
        {
            return (as_type<float>(((as_type<uint>((x_0))) & 2147483647U)));
        }
    case 3U:
        {
            return x_0 * x_0;
        }
    case 4U:
        {
            float _S12;
            if(x_0 < 0.0)
            {
                _S12 = 0.0;
            }
            else
            {
                _S12 = x_0;
            }
            return _S12;
        }
    case 5U:
        {
            return sqrt(x_0);
        }
    case 6U:
        {
            return exp(x_0);
        }
    case 7U:
        {
            return log(x_0);
        }
    default:
        {
            return 1.0 / x_0;
        }
    }
}

struct UnaryParams_0
{
    uint count_0;
    uint gridX_2;
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
    UnaryParams_0 constant* p_0;
    float device* y_0;
    float device* x_1;
};

[[kernel]] void ew_unary(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], UnaryParams_0 constant* p_1 [[buffer(2)]], float device* y_1 [[buffer(1)]], float device* x_2 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->y_0 = y_1;
    (&kernelContext_0)->x_1 = x_2;
    uint i_3 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(i_3 >= (p_1->count_0))
    {
        return;
    }
    DgView_0 yv_0 = dg_view_0((&kernelContext_0)->p_0->yOffset_0, uint4((&kernelContext_0)->p_0->yExtent0_0, (&kernelContext_0)->p_0->yExtent1_0, (&kernelContext_0)->p_0->yExtent2_0, (&kernelContext_0)->p_0->yExtent3_0), uint4((&kernelContext_0)->p_0->yStride0_0, (&kernelContext_0)->p_0->yStride1_0, (&kernelContext_0)->p_0->yStride2_0, (&kernelContext_0)->p_0->yStride3_0));
    thread DgView_0 _S13 = yv_0;
    uint4 _S14 = dg_coords_0(&_S13, i_3);
    thread DgView_0 _S15 = yv_0;
    uint _S16 = dg_at_0(&_S15, _S14);
    float device* _S17 = (&kernelContext_0)->y_0+_S16;
    thread DgView_0 _S18 = dg_view_0((&kernelContext_0)->p_0->xOffset_0, uint4((&kernelContext_0)->p_0->xExtent0_0, (&kernelContext_0)->p_0->xExtent1_0, (&kernelContext_0)->p_0->xExtent2_0, (&kernelContext_0)->p_0->xExtent3_0), uint4((&kernelContext_0)->p_0->xStride0_0, (&kernelContext_0)->p_0->xStride1_0, (&kernelContext_0)->p_0->xStride2_0, (&kernelContext_0)->p_0->xStride3_0));
    uint _S19 = dg_at_broadcast_0(&_S18, _S14);
    *_S17 = unary_0(*((&kernelContext_0)->x_1+_S19));
    return;
}

