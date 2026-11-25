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

float dg_negative_infinity_0()
{
    return (as_type<float>((4286578688U)));
}

bool dg_is_nan_0(float x_0)
{
    uint _S1 = (as_type<uint>((x_0)));
    bool _S2;
    if((_S1 & 2139095040U) == 2139095040U)
    {
        _S2 = (_S1 & 8388607U) != 0U;
    }
    else
    {
        _S2 = false;
    }
    return _S2;
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
    bool _S3;
    if(a_0 == 0.0)
    {
        _S3 = b_0 == 0.0;
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        return (as_type<float>(((as_type<uint>((a_0))) & (as_type<uint>((b_0))))));
    }
    float _S4;
    if(a_0 > b_0)
    {
        _S4 = a_0;
    }
    else
    {
        _S4 = b_0;
    }
    return _S4;
}

float dg_op_max_0(float a_1, float b_1)
{
    return dg_max_0(a_1, b_1);
}

float dg_op_add_0(float a_2, float b_2)
{
    return a_2 + b_2;
}

struct AttnParams_0
{
    uint heads_0;
    uint kvHeads_0;
    uint queries_0;
    uint keys_0;
    uint headDim_0;
    uint cacheKeys_0;
    float scale_0;
    uint causal_0;
    uint splits_0;
    uint splitKeys_0;
    uint rowBlocks_0;
    uint pageSize_0;
    uint batch_0;
};

struct KernelContext_0
{
    AttnParams_0 constant* p_0;
    float device* partialML_0;
    float device* partialAcc_0;
    float device* o_0;
    array<float, int(128)> threadgroup* red_0;
    float threadgroup* rowM_0;
    array<float, int(1024)> threadgroup* ws_0;
    float threadgroup* rowL_0;
};

[[kernel]] void attn_combine_f32(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(3)]], float device* partialML_1 [[buffer(0)]], float device* partialAcc_1 [[buffer(1)]], float device* o_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->partialML_0 = partialML_1;
    (&kernelContext_0)->partialAcc_0 = partialAcc_1;
    (&kernelContext_0)->o_0 = o_1;
    threadgroup array<float, int(128)> red_1;
    (&kernelContext_0)->red_0 = &red_1;
    threadgroup float rowM_1;
    (&kernelContext_0)->rowM_0 = &rowM_1;
    threadgroup array<float, int(1024)> ws_1;
    (&kernelContext_0)->ws_0 = &ws_1;
    threadgroup float rowL_1;
    (&kernelContext_0)->rowL_0 = &rowL_1;
    uint t_0 = local_0.x;
    uint rows_0 = max(p_1->batch_0, 1U) * p_1->heads_0 * p_1->queries_0;
    uint _S5 = p_1->headDim_0 / 2U;
    uint row_0 = dg_linear_group_0(group_1, p_1->rowBlocks_0);
    if(row_0 >= rows_0)
    {
        return;
    }
    float _S6 = dg_negative_infinity_0();
    uint s_0 = t_0;
    float acc_0 = _S6;
    for(;;)
    {
        if(s_0 < ((&kernelContext_0)->p_0->splits_0))
        {
        }
        else
        {
            break;
        }
        float _S7 = max(acc_0, *((&kernelContext_0)->partialML_0+2U * (s_0 * rows_0 + row_0)));
        s_0 = s_0 + 128U;
        acc_0 = _S7;
    }
    (*(&kernelContext_0)->red_0)[t_0] = acc_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S8 = t_0 < 64U;
    if(_S8)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S9 = t_0 < 32U;
    if(_S9)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S10 = t_0 < 16U;
    if(_S10)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S11 = t_0 < 8U;
    if(_S11)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S12 = t_0 < 4U;
    if(_S12)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S13 = t_0 < 2U;
    if(_S13)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S14 = t_0 < 1U;
    if(_S14)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S15 = t_0 == 0U;
    if(_S15)
    {
        *(&kernelContext_0)->rowM_0 = (*(&kernelContext_0)->red_0)[int(0)];
    }
    float w_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S16 = *(&kernelContext_0)->rowM_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s_0 = t_0;
    acc_0 = 0.0;
    for(;;)
    {
        if(s_0 < ((&kernelContext_0)->p_0->splits_0))
        {
        }
        else
        {
            break;
        }
        uint _S17 = 2U * (s_0 * rows_0 + row_0);
        float device* _S18 = (&kernelContext_0)->partialML_0+_S17;
        float ms_0 = *_S18;
        if((*_S18) == _S6)
        {
            w_0 = 0.0;
        }
        else
        {
            w_0 = exp(ms_0 - _S16);
        }
        (*(&kernelContext_0)->ws_0)[s_0] = w_0;
        float _S19 = fma(*((&kernelContext_0)->partialML_0+(_S17 + 1U)), w_0, acc_0);
        s_0 = s_0 + 128U;
        acc_0 = _S19;
    }
    (*(&kernelContext_0)->red_0)[t_0] = acc_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S8)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S9)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S10)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S11)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S12)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S13)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S14)
    {
        (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S15)
    {
        *(&kernelContext_0)->rowL_0 = (*(&kernelContext_0)->red_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S20 = *(&kernelContext_0)->rowL_0;
    uint pair_0 = t_0;
    for(;;)
    {
        if(pair_0 < _S5)
        {
        }
        else
        {
            break;
        }
        uint s0_0 = 0U;
        w_0 = 0.0;
        float hi_0 = 0.0;
        for(;;)
        {
            if(s0_0 < ((&kernelContext_0)->p_0->splits_0))
            {
            }
            else
            {
                break;
            }
            thread array<float2, int(8)> a_3;
            uint _S21 = 2U * pair_0;
            uint at_0 = (s0_0 * rows_0 + row_0) * p_1->headDim_0 + _S21;
            float2 _S22;
            if(s0_0 < ((&kernelContext_0)->p_0->splits_0))
            {
                _S22 = float2(*((&kernelContext_0)->partialAcc_0+at_0), *((&kernelContext_0)->partialAcc_0+(at_0 + 1U)));
            }
            else
            {
                _S22 = float2(0.0, 0.0);
            }
            a_3[0U] = _S22;
            uint s_1 = s0_0 + 1U;
            uint at_1 = (s_1 * rows_0 + row_0) * p_1->headDim_0 + _S21;
            float2 _S23;
            if(s_1 < ((&kernelContext_0)->p_0->splits_0))
            {
                _S23 = float2(*((&kernelContext_0)->partialAcc_0+at_1), *((&kernelContext_0)->partialAcc_0+(at_1 + 1U)));
            }
            else
            {
                _S23 = float2(0.0, 0.0);
            }
            a_3[1U] = _S23;
            uint s_2 = s0_0 + 2U;
            uint at_2 = (s_2 * rows_0 + row_0) * p_1->headDim_0 + _S21;
            float2 _S24;
            if(s_2 < ((&kernelContext_0)->p_0->splits_0))
            {
                _S24 = float2(*((&kernelContext_0)->partialAcc_0+at_2), *((&kernelContext_0)->partialAcc_0+(at_2 + 1U)));
            }
            else
            {
                _S24 = float2(0.0, 0.0);
            }
            a_3[2U] = _S24;
            uint s_3 = s0_0 + 3U;
            uint at_3 = (s_3 * rows_0 + row_0) * p_1->headDim_0 + _S21;
            float2 _S25;
            if(s_3 < ((&kernelContext_0)->p_0->splits_0))
            {
                _S25 = float2(*((&kernelContext_0)->partialAcc_0+at_3), *((&kernelContext_0)->partialAcc_0+(at_3 + 1U)));
            }
            else
            {
                _S25 = float2(0.0, 0.0);
            }
            a_3[3U] = _S25;
            uint s_4 = s0_0 + 4U;
            uint at_4 = (s_4 * rows_0 + row_0) * p_1->headDim_0 + _S21;
            float2 _S26;
            if(s_4 < ((&kernelContext_0)->p_0->splits_0))
            {
                _S26 = float2(*((&kernelContext_0)->partialAcc_0+at_4), *((&kernelContext_0)->partialAcc_0+(at_4 + 1U)));
            }
            else
            {
                _S26 = float2(0.0, 0.0);
            }
            a_3[4U] = _S26;
            uint s_5 = s0_0 + 5U;
            uint at_5 = (s_5 * rows_0 + row_0) * p_1->headDim_0 + _S21;
            float2 _S27;
            if(s_5 < ((&kernelContext_0)->p_0->splits_0))
            {
                _S27 = float2(*((&kernelContext_0)->partialAcc_0+at_5), *((&kernelContext_0)->partialAcc_0+(at_5 + 1U)));
            }
            else
            {
                _S27 = float2(0.0, 0.0);
            }
            a_3[5U] = _S27;
            uint s_6 = s0_0 + 6U;
            uint at_6 = (s_6 * rows_0 + row_0) * p_1->headDim_0 + _S21;
            float2 _S28;
            if(s_6 < ((&kernelContext_0)->p_0->splits_0))
            {
                _S28 = float2(*((&kernelContext_0)->partialAcc_0+at_6), *((&kernelContext_0)->partialAcc_0+(at_6 + 1U)));
            }
            else
            {
                _S28 = float2(0.0, 0.0);
            }
            a_3[6U] = _S28;
            uint s_7 = s0_0 + 7U;
            uint at_7 = (s_7 * rows_0 + row_0) * p_1->headDim_0 + _S21;
            float2 _S29;
            if(s_7 < ((&kernelContext_0)->p_0->splits_0))
            {
                _S29 = float2(*((&kernelContext_0)->partialAcc_0+at_7), *((&kernelContext_0)->partialAcc_0+(at_7 + 1U)));
            }
            else
            {
                _S29 = float2(0.0, 0.0);
            }
            a_3[7U] = _S29;
            float lo_0;
            float hi_1;
            if(s0_0 < ((&kernelContext_0)->p_0->splits_0))
            {
                float _S30 = fma(a_3[0U].y, (*(&kernelContext_0)->ws_0)[s0_0], hi_0);
                lo_0 = fma(a_3[0U].x, (*(&kernelContext_0)->ws_0)[s0_0], w_0);
                hi_1 = _S30;
            }
            else
            {
                lo_0 = w_0;
                hi_1 = hi_0;
            }
            float lo_1;
            float hi_2;
            if(s_1 < ((&kernelContext_0)->p_0->splits_0))
            {
                float _S31 = fma(a_3[1U].y, (*(&kernelContext_0)->ws_0)[s_1], hi_1);
                lo_1 = fma(a_3[1U].x, (*(&kernelContext_0)->ws_0)[s_1], lo_0);
                hi_2 = _S31;
            }
            else
            {
                lo_1 = lo_0;
                hi_2 = hi_1;
            }
            float lo_2;
            float hi_3;
            if(s_2 < ((&kernelContext_0)->p_0->splits_0))
            {
                float _S32 = fma(a_3[2U].y, (*(&kernelContext_0)->ws_0)[s_2], hi_2);
                lo_2 = fma(a_3[2U].x, (*(&kernelContext_0)->ws_0)[s_2], lo_1);
                hi_3 = _S32;
            }
            else
            {
                lo_2 = lo_1;
                hi_3 = hi_2;
            }
            float lo_3;
            float hi_4;
            if(s_3 < ((&kernelContext_0)->p_0->splits_0))
            {
                float _S33 = fma(a_3[3U].y, (*(&kernelContext_0)->ws_0)[s_3], hi_3);
                lo_3 = fma(a_3[3U].x, (*(&kernelContext_0)->ws_0)[s_3], lo_2);
                hi_4 = _S33;
            }
            else
            {
                lo_3 = lo_2;
                hi_4 = hi_3;
            }
            float lo_4;
            float hi_5;
            if(s_4 < ((&kernelContext_0)->p_0->splits_0))
            {
                float _S34 = fma(a_3[4U].y, (*(&kernelContext_0)->ws_0)[s_4], hi_4);
                lo_4 = fma(a_3[4U].x, (*(&kernelContext_0)->ws_0)[s_4], lo_3);
                hi_5 = _S34;
            }
            else
            {
                lo_4 = lo_3;
                hi_5 = hi_4;
            }
            float lo_5;
            float hi_6;
            if(s_5 < ((&kernelContext_0)->p_0->splits_0))
            {
                float _S35 = fma(a_3[5U].y, (*(&kernelContext_0)->ws_0)[s_5], hi_5);
                lo_5 = fma(a_3[5U].x, (*(&kernelContext_0)->ws_0)[s_5], lo_4);
                hi_6 = _S35;
            }
            else
            {
                lo_5 = lo_4;
                hi_6 = hi_5;
            }
            float lo_6;
            float hi_7;
            if(s_6 < ((&kernelContext_0)->p_0->splits_0))
            {
                float _S36 = fma(a_3[6U].y, (*(&kernelContext_0)->ws_0)[s_6], hi_6);
                lo_6 = fma(a_3[6U].x, (*(&kernelContext_0)->ws_0)[s_6], lo_5);
                hi_7 = _S36;
            }
            else
            {
                lo_6 = lo_5;
                hi_7 = hi_6;
            }
            if(s_7 < ((&kernelContext_0)->p_0->splits_0))
            {
                float _S37 = fma(a_3[7U].y, (*(&kernelContext_0)->ws_0)[s_7], hi_7);
                w_0 = fma(a_3[7U].x, (*(&kernelContext_0)->ws_0)[s_7], lo_6);
                hi_0 = _S37;
            }
            else
            {
                w_0 = lo_6;
                hi_0 = hi_7;
            }
            s0_0 = s0_0 + 8U;
        }
        float hi_8 = hi_0 / _S20;
        uint _S38 = 2U * pair_0;
        *((&kernelContext_0)->o_0+(row_0 * p_1->headDim_0 + _S38)) = w_0 / _S20;
        *((&kernelContext_0)->o_0+(row_0 * p_1->headDim_0 + _S38 + 1U)) = hi_8;
        pair_0 = pair_0 + 128U;
    }
    return;
}

