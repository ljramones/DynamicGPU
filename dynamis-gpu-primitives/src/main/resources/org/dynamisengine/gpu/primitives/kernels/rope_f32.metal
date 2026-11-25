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

void dg_report_error_0(uint device* errors_0, uint code_0, uint index_0)
{
    uint _S1 = atomic_fetch_or_explicit(((atomic_uint device*)(errors_0+int(0))), code_0, memory_order_relaxed);
    uint _S2 = atomic_fetch_max_explicit(((atomic_uint device*)(errors_0+int(1))), ~index_0, memory_order_relaxed);
    uint _S3 = atomic_fetch_add_explicit(((atomic_uint device*)(errors_0+int(2))), 1U, memory_order_relaxed);
    return;
}

struct RopeParams_0
{
    uint tokens_0;
    uint heads_0;
    uint headDim_0;
    uint rotaryDim_0;
    uint ldx_0;
    uint ldy_0;
    uint strideHead_0;
    uint layout_0;
    uint positions_0;
    uint count_0;
    uint gridX_2;
    uint inPlace_0;
};

struct KernelContext_0
{
    RopeParams_0 constant* p_0;
    uint device* pos_0;
    uint device* errors_1;
    float device* y_0;
    float device* x_0;
    float2 device* table_0;
};

[[kernel]] void rope_f32(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], RopeParams_0 constant* p_1 [[buffer(5)]], uint device* pos_1 [[buffer(1)]], uint device* errors_2 [[buffer(4)]], float device* y_1 [[buffer(3)]], float device* x_1 [[buffer(0)]], float2 device* table_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->pos_0 = pos_1;
    (&kernelContext_0)->errors_1 = errors_2;
    (&kernelContext_0)->y_0 = y_1;
    (&kernelContext_0)->x_0 = x_1;
    (&kernelContext_0)->table_0 = table_1;
    uint id_0 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(id_0 >= (p_1->count_0))
    {
        return;
    }
    uint halfHead_0 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint i_0 = id_0 % halfHead_0;
    uint _S4 = id_0 / halfHead_0;
    uint head_0 = _S4 % (&kernelContext_0)->p_0->heads_0;
    uint _S5 = id_0 / halfHead_0;
    uint tok_0 = _S5 / (&kernelContext_0)->p_0->heads_0;
    uint device* _S6 = (&kernelContext_0)->pos_0+tok_0;
    uint position_0 = *_S6;
    if((*_S6) >= ((&kernelContext_0)->p_0->positions_0))
    {
        bool _S7;
        if(i_0 == 0U)
        {
            _S7 = head_0 == 0U;
        }
        else
        {
            _S7 = false;
        }
        if(_S7)
        {
            dg_report_error_0((&kernelContext_0)->errors_1, 1U, tok_0);
        }
        return;
    }
    uint bx_0 = tok_0 * (&kernelContext_0)->p_0->ldx_0 + head_0 * (&kernelContext_0)->p_0->strideHead_0;
    uint by_0 = tok_0 * (&kernelContext_0)->p_0->ldy_0 + head_0 * (&kernelContext_0)->p_0->strideHead_0;
    uint halfRot_0 = (&kernelContext_0)->p_0->rotaryDim_0 / 2U;
    if(i_0 < halfRot_0)
    {
        uint ia_0;
        if(((&kernelContext_0)->p_0->layout_0) == 0U)
        {
            ia_0 = 2U * i_0;
        }
        else
        {
            ia_0 = i_0;
        }
        uint ib_0;
        if(((&kernelContext_0)->p_0->layout_0) == 0U)
        {
            ib_0 = 2U * i_0 + 1U;
        }
        else
        {
            ib_0 = i_0 + halfRot_0;
        }
        float a_0;
        if(((&kernelContext_0)->p_0->inPlace_0) != 0U)
        {
            a_0 = *((&kernelContext_0)->y_0+(by_0 + ia_0));
        }
        else
        {
            a_0 = *((&kernelContext_0)->x_0+(bx_0 + ia_0));
        }
        float b_0;
        if(((&kernelContext_0)->p_0->inPlace_0) != 0U)
        {
            b_0 = *((&kernelContext_0)->y_0+(by_0 + ib_0));
        }
        else
        {
            b_0 = *((&kernelContext_0)->x_0+(bx_0 + ib_0));
        }
        float2 device* _S8 = (&kernelContext_0)->table_0+(position_0 * halfRot_0 + i_0);
        float _S9 = (*_S8).x;
        float _S10 = (*_S8).y;
        *((&kernelContext_0)->y_0+(by_0 + ia_0)) = a_0 * _S9 - b_0 * _S10;
        *((&kernelContext_0)->y_0+(by_0 + ib_0)) = a_0 * _S10 + b_0 * _S9;
    }
    else
    {
        uint e_0 = (&kernelContext_0)->p_0->rotaryDim_0 + 2U * (i_0 - halfRot_0);
        if(((&kernelContext_0)->p_0->inPlace_0) == 0U)
        {
            uint _S11 = by_0 + e_0;
            uint _S12 = bx_0 + e_0;
            *((&kernelContext_0)->y_0+_S11) = *((&kernelContext_0)->x_0+_S12);
            *((&kernelContext_0)->y_0+(_S11 + 1U)) = *((&kernelContext_0)->x_0+(_S12 + 1U));
        }
    }
    return;
}

