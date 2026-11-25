#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
float dg_negative_infinity_0()
{
    return (as_type<float>((4286578688U)));
}

bool before_0(float va_0, uint ia_0, float vb_0, uint ib_0)
{
    bool _S1;
    if(va_0 > vb_0)
    {
        _S1 = true;
    }
    else
    {
        if(va_0 == vb_0)
        {
            _S1 = ia_0 < ib_0;
        }
        else
        {
            _S1 = false;
        }
    }
    return _S1;
}

struct SampleParams_0
{
    uint rows_0;
    uint vocab_0;
    uint ldl_0;
    float invT_0;
    uint topK_0;
    float topP_0;
    uint seedLo_0;
    uint seedHi_0;
    uint step_0;
    uint chunks_0;
    uint restricted_0;
    uint lists_0;
    uint stride_0;
    uint head_0;
    uint perGroup_0;
};

struct KernelContext_0
{
    SampleParams_0 constant* p_0;
    float device* inV_0;
    uint device* inI_0;
    float device* outV_0;
    uint device* outI_0;
    array<float, int(2048)> threadgroup* sv_0;
    array<uint, int(2048)> threadgroup* si_0;
};

uint countBefore_0(uint b_0, uint len_0, float v_0, uint i_0, bool orEqual_0, KernelContext_0 thread* kernelContext_0)
{
    uint lo_0 = 0U;
    uint hi_0 = len_0;
    for(;;)
    {
        if(lo_0 < hi_0)
        {
        }
        else
        {
            break;
        }
        uint mid_0 = (lo_0 + hi_0) / 2U;
        uint _S2 = b_0 + mid_0;
        float mv_0 = (*kernelContext_0->sv_0)[_S2];
        uint mi_0 = (*kernelContext_0->si_0)[_S2];
        bool goesBefore_0;
        if(before_0((*kernelContext_0->sv_0)[_S2], (*kernelContext_0->si_0)[_S2], v_0, i_0))
        {
            goesBefore_0 = true;
        }
        else
        {
            if(orEqual_0)
            {
                goesBefore_0 = !before_0(v_0, i_0, mv_0, mi_0);
            }
            else
            {
                goesBefore_0 = false;
            }
        }
        if(goesBefore_0)
        {
            lo_0 = mid_0 + 1U;
        }
        else
        {
            hi_0 = mid_0;
        }
    }
    return lo_0;
}

void mergeTree_0(uint t_0, uint c_0, uint slots_0, KernelContext_0 thread* kernelContext_1)
{
    uint len_1 = c_0;
    for(;;)
    {
        if(len_1 < slots_0)
        {
        }
        else
        {
            break;
        }
        thread array<float, int(2)> v_1;
        thread array<uint, int(2)> ix_0;
        thread array<uint, int(2)> to_0;
        for(;;)
        {
            uint other_0;
            uint q_0;
            for(;;)
            {
                for(;;)
                {
                    to_0[0U] = 4294967295U;
                    v_1[0U] = 0.0;
                    ix_0[0U] = 4294967295U;
                    if(t_0 >= slots_0)
                    {
                        break;
                    }
                    uint _S3 = 2U * len_1;
                    uint _S4 = t_0 / _S3;
                    uint pb_0 = _S4 * 2U * len_1;
                    uint inPair_0 = t_0 % _S3;
                    bool inA_0 = inPair_0 < len_1;
                    if(inA_0)
                    {
                        q_0 = inPair_0;
                    }
                    else
                    {
                        q_0 = inPair_0 - len_1;
                    }
                    v_1[0U] = (*kernelContext_1->sv_0)[t_0];
                    ix_0[0U] = (*kernelContext_1->si_0)[t_0];
                    if(inA_0)
                    {
                        uint _S5 = countBefore_0(pb_0 + len_1, len_1, v_1[0U], ix_0[0U], false, kernelContext_1);
                        other_0 = _S5;
                    }
                    else
                    {
                        uint _S6 = countBefore_0(pb_0, len_1, v_1[0U], ix_0[0U], true, kernelContext_1);
                        other_0 = _S6;
                    }
                    to_0[0U] = pb_0 + q_0 + other_0;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint s_0 = t_0 + 1024U;
                    to_0[1U] = 4294967295U;
                    v_1[1U] = 0.0;
                    ix_0[1U] = 4294967295U;
                    if(s_0 >= slots_0)
                    {
                        break;
                    }
                    uint _S7 = 2U * len_1;
                    uint _S8 = s_0 / _S7;
                    uint pb_1 = _S8 * 2U * len_1;
                    uint inPair_1 = s_0 % _S7;
                    bool inA_1 = inPair_1 < len_1;
                    if(inA_1)
                    {
                        q_0 = inPair_1;
                    }
                    else
                    {
                        q_0 = inPair_1 - len_1;
                    }
                    v_1[1U] = (*kernelContext_1->sv_0)[s_0];
                    ix_0[1U] = (*kernelContext_1->si_0)[s_0];
                    if(inA_1)
                    {
                        uint _S9 = countBefore_0(pb_1 + len_1, len_1, v_1[1U], ix_0[1U], false, kernelContext_1);
                        other_0 = _S9;
                    }
                    else
                    {
                        uint _S10 = countBefore_0(pb_1, len_1, v_1[1U], ix_0[1U], true, kernelContext_1);
                        other_0 = _S10;
                    }
                    to_0[1U] = pb_1 + q_0 + other_0;
                    break;
                }
                break;
            }
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if((to_0[0U]) != 4294967295U)
        {
            (*kernelContext_1->sv_0)[to_0[0U]] = v_1[0U];
            (*kernelContext_1->si_0)[to_0[0U]] = ix_0[0U];
        }
        if((to_0[1U]) != 4294967295U)
        {
            (*kernelContext_1->sv_0)[to_0[1U]] = v_1[1U];
            (*kernelContext_1->si_0)[to_0[1U]] = ix_0[1U];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        len_1 = len_1 * 2U;
    }
    return;
}

[[kernel]] void sample_merge(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], SampleParams_0 constant* p_1 [[buffer(4)]], float device* inV_1 [[buffer(0)]], uint device* inI_1 [[buffer(1)]], float device* outV_1 [[buffer(2)]], uint device* outI_1 [[buffer(3)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->p_0 = p_1;
    (&kernelContext_2)->inV_0 = inV_1;
    (&kernelContext_2)->inI_0 = inI_1;
    (&kernelContext_2)->outV_0 = outV_1;
    (&kernelContext_2)->outI_0 = outI_1;
    threadgroup array<float, int(2048)> sv_1;
    (&kernelContext_2)->sv_0 = &sv_1;
    threadgroup array<uint, int(2048)> si_1;
    (&kernelContext_2)->si_0 = &si_1;
    uint g_0 = group_0.x;
    uint row_0 = group_0.y;
    uint t_1 = local_0.x;
    uint h_0 = p_1->head_0;
    uint _S11 = min(p_1->head_0, p_1->stride_0);
    uint P_0 = p_1->perGroup_0;
    uint outLists_0 = (p_1->lists_0 + p_1->perGroup_0 - 1U) / p_1->perGroup_0;
    uint first_0 = g_0 * P_0;
    uint _S12 = min(P_0, p_1->lists_0 - first_0);
    uint slots_1 = P_0 * _S11;
    uint dgL_0 = t_1 / _S11;
    uint dgQ_0 = t_1 % _S11;
    bool _S13 = t_1 < slots_1;
    bool dgValid_0;
    if(_S13)
    {
        dgValid_0 = dgL_0 < _S12;
    }
    else
    {
        dgValid_0 = false;
    }
    uint dgAt_0 = (row_0 * p_1->lists_0 + first_0 + dgL_0) * p_1->stride_0 + dgQ_0;
    float _S14;
    if(dgValid_0)
    {
        _S14 = *((&kernelContext_2)->inV_0+dgAt_0);
    }
    else
    {
        _S14 = dg_negative_infinity_0();
    }
    (*(&kernelContext_2)->sv_0)[t_1] = _S14;
    uint _S15;
    if(dgValid_0)
    {
        _S15 = *((&kernelContext_2)->inI_0+dgAt_0);
    }
    else
    {
        _S15 = 4294967295U;
    }
    (*(&kernelContext_2)->si_0)[t_1] = _S15;
    uint dgS_0 = t_1 + 1024U;
    uint dgL_1 = dgS_0 / _S11;
    uint dgQ_1 = dgS_0 % _S11;
    if(dgS_0 < slots_1)
    {
        dgValid_0 = dgL_1 < _S12;
    }
    else
    {
        dgValid_0 = false;
    }
    uint dgAt_1 = (row_0 * p_1->lists_0 + first_0 + dgL_1) * p_1->stride_0 + dgQ_1;
    if(dgValid_0)
    {
        _S14 = *((&kernelContext_2)->inV_0+dgAt_1);
    }
    else
    {
        _S14 = dg_negative_infinity_0();
    }
    (*(&kernelContext_2)->sv_0)[dgS_0] = _S14;
    if(dgValid_0)
    {
        _S15 = *((&kernelContext_2)->inI_0+dgAt_1);
    }
    else
    {
        _S15 = 4294967295U;
    }
    (*(&kernelContext_2)->si_0)[dgS_0] = _S15;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mergeTree_0(t_1, _S11, slots_1, &kernelContext_2);
    if(t_1 < h_0)
    {
        uint at_0 = (row_0 * outLists_0 + g_0) * h_0 + t_1;
        float device* _S16 = (&kernelContext_2)->outV_0+at_0;
        if(_S13)
        {
            _S14 = (*(&kernelContext_2)->sv_0)[t_1];
        }
        else
        {
            _S14 = dg_negative_infinity_0();
        }
        *_S16 = _S14;
        uint device* _S17 = (&kernelContext_2)->outI_0+at_0;
        if(_S13)
        {
            _S15 = (*(&kernelContext_2)->si_0)[t_1];
        }
        else
        {
            _S15 = 4294967295U;
        }
        *_S17 = _S15;
    }
    return;
}

