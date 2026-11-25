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

uint pow2Above_0(uint x_0)
{
    uint s_0 = 1U;
    for(;;)
    {
        if(s_0 < x_0)
        {
        }
        else
        {
            break;
        }
        s_0 = s_0 * 2U;
    }
    return s_0;
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

uint fmix_0(uint h_0)
{
    uint _S2 = (h_0 ^ (h_0 >> 16U)) * 2246822507U;
    uint _S3 = (_S2 ^ (_S2 >> 13U)) * 3266489909U;
    return _S3 ^ (_S3 >> 16U);
}

float rand01_0(uint seedLo_0, uint seedHi_0, uint row_0, uint step_0)
{
    return float((fmix_0(seedLo_0 ^ (fmix_0(seedHi_0 ^ (fmix_0((row_0 * 2654435769U) ^ (fmix_0(step_0 + 1663821227U)))))))) >> 8U) * (as_type<float>((864026624U)));
}

struct SampleParams_0
{
    uint rows_0;
    uint vocab_0;
    uint ldl_0;
    float invT_0;
    uint topK_0;
    float topP_0;
    uint seedLo_1;
    uint seedHi_1;
    uint step_1;
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
    uint4 device* info_0;
    uint device* tokens_0;
    float device* logits_0;
    float device* sortedV_0;
    uint device* sortedI_0;
    array<float, int(1024)> threadgroup* sv_0;
    array<uint, int(1024)> threadgroup* si_0;
    float threadgroup* gMax_0;
    uint threadgroup* gIdx_0;
    float threadgroup* gTotal_0;
    uint threadgroup* gFound_0;
    uint threadgroup* gLast_0;
    float threadgroup* gU_0;
};

void scanF_0(uint t_0, uint size_0, KernelContext_0 thread* kernelContext_0)
{
    uint d_0 = 1U;
    for(;;)
    {
        if(d_0 < size_0)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint i_0 = (t_0 + 1U) * d_0 * 2U - 1U;
        if(i_0 < size_0)
        {
            (*kernelContext_0->sv_0)[i_0] = (*kernelContext_0->sv_0)[i_0 - d_0] + (*kernelContext_0->sv_0)[i_0];
        }
        d_0 = d_0 * 2U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 == 0U)
    {
        (*kernelContext_0->sv_0)[size_0 - 1U] = 0.0;
    }
    uint d2_0 = size_0 / 2U;
    for(;;)
    {
        if(d2_0 > 0U)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint i_1 = (t_0 + 1U) * d2_0 * 2U - 1U;
        if(i_1 < size_0)
        {
            float _S4 = (*kernelContext_0->sv_0)[i_1] + (*kernelContext_0->sv_0)[i_1 - d2_0];
            (*kernelContext_0->sv_0)[i_1 - d2_0] = (*kernelContext_0->sv_0)[i_1];
            (*kernelContext_0->sv_0)[i_1] = _S4;
        }
        d2_0 = d2_0 / 2U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
}

bool dg_is_nan_0(float x_1)
{
    uint _S5 = (as_type<uint>((x_1)));
    bool _S6;
    if((_S5 & 2139095040U) == 2139095040U)
    {
        _S6 = (_S5 & 8388607U) != 0U;
    }
    else
    {
        _S6 = false;
    }
    return _S6;
}

float sanitize_0(float l_0)
{
    float _S7;
    if(dg_is_nan_0(l_0))
    {
        _S7 = dg_negative_infinity_0();
    }
    else
    {
        if(l_0 == 0.0)
        {
            _S7 = 0.0;
        }
        else
        {
            _S7 = l_0;
        }
    }
    return _S7;
}

float dg_op_add_0(float a_0, float b_0)
{
    return a_0 + b_0;
}

uint countBefore_0(uint b_1, uint len_0, float v_0, uint i_2, bool orEqual_0, KernelContext_0 thread* kernelContext_1)
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
        uint _S8 = b_1 + mid_0;
        float mv_0 = (*kernelContext_1->sv_0)[_S8];
        uint mi_0 = (*kernelContext_1->si_0)[_S8];
        bool goesBefore_0;
        if(before_0((*kernelContext_1->sv_0)[_S8], (*kernelContext_1->si_0)[_S8], v_0, i_2))
        {
            goesBefore_0 = true;
        }
        else
        {
            if(orEqual_0)
            {
                goesBefore_0 = !before_0(v_0, i_2, mv_0, mi_0);
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

void mergeTree_0(uint t_1, uint c_0, uint slots_0, KernelContext_0 thread* kernelContext_2)
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
        thread array<float, int(1)> v_1;
        thread array<uint, int(1)> ix_0;
        thread array<uint, int(1)> to_0;
        for(;;)
        {
            for(;;)
            {
                for(;;)
                {
                    to_0[0U] = 4294967295U;
                    v_1[0U] = 0.0;
                    ix_0[0U] = 4294967295U;
                    if(t_1 >= slots_0)
                    {
                        break;
                    }
                    uint _S9 = 2U * len_1;
                    uint _S10 = t_1 / _S9;
                    uint pb_0 = _S10 * 2U * len_1;
                    uint inPair_0 = t_1 % _S9;
                    bool inA_0 = inPair_0 < len_1;
                    uint q_0;
                    if(inA_0)
                    {
                        q_0 = inPair_0;
                    }
                    else
                    {
                        q_0 = inPair_0 - len_1;
                    }
                    v_1[0U] = (*kernelContext_2->sv_0)[t_1];
                    ix_0[0U] = (*kernelContext_2->si_0)[t_1];
                    uint other_0;
                    if(inA_0)
                    {
                        uint _S11 = countBefore_0(pb_0 + len_1, len_1, v_1[0U], ix_0[0U], false, kernelContext_2);
                        other_0 = _S11;
                    }
                    else
                    {
                        uint _S12 = countBefore_0(pb_0, len_1, v_1[0U], ix_0[0U], true, kernelContext_2);
                        other_0 = _S12;
                    }
                    to_0[0U] = pb_0 + q_0 + other_0;
                    break;
                }
                break;
            }
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if((to_0[0U]) != 4294967295U)
        {
            (*kernelContext_2->sv_0)[to_0[0U]] = v_1[0U];
            (*kernelContext_2->si_0)[to_0[0U]] = ix_0[0U];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        len_1 = len_1 * 2U;
    }
    return;
}

[[kernel]] void sample_final(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], SampleParams_0 constant* p_1 [[buffer(5)]], uint4 device* info_1 [[buffer(3)]], uint device* tokens_1 [[buffer(4)]], float device* logits_1 [[buffer(0)]], float device* sortedV_1 [[buffer(1)]], uint device* sortedI_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->p_0 = p_1;
    (&kernelContext_3)->info_0 = info_1;
    (&kernelContext_3)->tokens_0 = tokens_1;
    (&kernelContext_3)->logits_0 = logits_1;
    (&kernelContext_3)->sortedV_0 = sortedV_1;
    (&kernelContext_3)->sortedI_0 = sortedI_1;
    threadgroup array<float, int(1024)> sv_1;
    (&kernelContext_3)->sv_0 = &sv_1;
    threadgroup array<uint, int(1024)> si_1;
    (&kernelContext_3)->si_0 = &si_1;
    threadgroup float gMax_1;
    (&kernelContext_3)->gMax_0 = &gMax_1;
    threadgroup uint gIdx_1;
    (&kernelContext_3)->gIdx_0 = &gIdx_1;
    threadgroup float gTotal_1;
    (&kernelContext_3)->gTotal_0 = &gTotal_1;
    threadgroup uint gFound_1;
    (&kernelContext_3)->gFound_0 = &gFound_1;
    threadgroup uint gLast_1;
    (&kernelContext_3)->gLast_0 = &gLast_1;
    threadgroup float gU_1;
    (&kernelContext_3)->gU_0 = &gU_1;
    uint row_1 = group_0.x;
    uint t_2 = local_0.x;
    uint G_0 = p_1->chunks_0;
    uint n_0 = p_1->vocab_0;
    bool _S13 = t_2 < (p_1->chunks_0);
    uint4 ci_0;
    if(_S13)
    {
        ci_0 = *((&kernelContext_3)->info_0+(row_1 * G_0 + t_2));
    }
    else
    {
        ci_0 = uint4((as_type<uint>((dg_negative_infinity_0()))), 4294967295U, 0U, 0U);
    }
    bool dgValid_0;
    float mg_0 = (as_type<float>((ci_0.x)));
    float sg_0 = (as_type<float>((ci_0.z)));
    uint G2_0 = pow2Above_0(G_0);
    (*(&kernelContext_3)->sv_0)[t_2] = mg_0;
    (*(&kernelContext_3)->si_0)[t_2] = ci_0.y;
    uint w_0 = G2_0 / 2U;
    for(;;)
    {
        if(w_0 > 0U)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(t_2 < w_0)
        {
            uint _S14 = t_2 + w_0;
            dgValid_0 = before_0((*(&kernelContext_3)->sv_0)[_S14], (*(&kernelContext_3)->si_0)[_S14], (*(&kernelContext_3)->sv_0)[t_2], (*(&kernelContext_3)->si_0)[t_2]);
        }
        else
        {
            dgValid_0 = false;
        }
        if(dgValid_0)
        {
            uint _S15 = t_2 + w_0;
            (*(&kernelContext_3)->sv_0)[t_2] = (*(&kernelContext_3)->sv_0)[_S15];
            (*(&kernelContext_3)->si_0)[t_2] = (*(&kernelContext_3)->si_0)[_S15];
        }
        w_0 = w_0 / 2U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S16 = t_2 == 0U;
    if(_S16)
    {
        *(&kernelContext_3)->gMax_0 = (*(&kernelContext_3)->sv_0)[int(0)];
        if(((*(&kernelContext_3)->si_0)[int(0)]) == 4294967295U)
        {
            w_0 = 0U;
        }
        else
        {
            w_0 = (*(&kernelContext_3)->si_0)[int(0)];
        }
        *(&kernelContext_3)->gIdx_0 = w_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx_0 = *(&kernelContext_3)->gMax_0;
    uint argmax_0 = *(&kernelContext_3)->gIdx_0;
    if(((&kernelContext_3)->p_0->invT_0) <= 0.0)
    {
        dgValid_0 = true;
    }
    else
    {
        dgValid_0 = ((&kernelContext_3)->p_0->topK_0) == 1U;
    }
    if(dgValid_0)
    {
        dgValid_0 = true;
    }
    else
    {
        dgValid_0 = mx_0 == (dg_negative_infinity_0());
    }
    if(dgValid_0)
    {
        if(_S16)
        {
            *((&kernelContext_3)->tokens_0+row_1) = argmax_0;
        }
        return;
    }
    float rnd_0 = rand01_0((&kernelContext_3)->p_0->seedLo_1, (&kernelContext_3)->p_0->seedHi_1, row_1, (&kernelContext_3)->p_0->step_1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint g_0;
    float wg_0;
    float e_0;
    if(((&kernelContext_3)->p_0->restricted_0) == 0U)
    {
        if(_S13)
        {
            dgValid_0 = sg_0 > 0.0;
        }
        else
        {
            dgValid_0 = false;
        }
        if(dgValid_0)
        {
            wg_0 = sg_0 * exp((mg_0 - mx_0) * (&kernelContext_3)->p_0->invT_0);
        }
        else
        {
            wg_0 = 0.0;
        }
        (*(&kernelContext_3)->sv_0)[t_2] = wg_0;
        scanF_0(t_2, G2_0, &kernelContext_3);
        float before_g_0 = (*(&kernelContext_3)->sv_0)[t_2];
        if(t_2 == (G2_0 - 1U))
        {
            *(&kernelContext_3)->gTotal_0 = before_g_0 + wg_0;
        }
        if(_S16)
        {
            *(&kernelContext_3)->gFound_0 = 4294967295U;
            *(&kernelContext_3)->gLast_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float u_0 = rnd_0 * *(&kernelContext_3)->gTotal_0;
        bool _S17 = wg_0 > 0.0;
        if(_S17)
        {
            uint _S18 = atomic_fetch_max_explicit(((atomic_uint threadgroup*)((&kernelContext_3)->gLast_0)), t_2, memory_order_relaxed);
        }
        if(_S17)
        {
            dgValid_0 = u_0 >= before_g_0;
        }
        else
        {
            dgValid_0 = false;
        }
        if(dgValid_0)
        {
            dgValid_0 = u_0 < (before_g_0 + wg_0);
        }
        else
        {
            dgValid_0 = false;
        }
        if(dgValid_0)
        {
            uint _S19 = atomic_fetch_min_explicit(((atomic_uint threadgroup*)((&kernelContext_3)->gFound_0)), t_2, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if((*(&kernelContext_3)->gFound_0) == 4294967295U)
        {
            g_0 = *(&kernelContext_3)->gLast_0;
        }
        else
        {
            g_0 = *(&kernelContext_3)->gFound_0;
        }
        if(t_2 == g_0)
        {
            *(&kernelContext_3)->gU_0 = (u_0 - before_g_0) / wg_0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float frac_0 = *(&kernelContext_3)->gU_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S20 = g_0 * 1024U;
        uint j_0 = _S20 + t_2;
        if(j_0 < n_0)
        {
            e_0 = exp((sanitize_0(*((&kernelContext_3)->logits_0+(row_1 * (&kernelContext_3)->p_0->ldl_0 + j_0))) - mx_0) * (&kernelContext_3)->p_0->invT_0);
        }
        else
        {
            e_0 = 0.0;
        }
        (*(&kernelContext_3)->sv_0)[t_2] = e_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S21 = t_2 + 1U;
        uint _S22 = _S21 * 2U;
        uint dgI_0 = _S22 - 1U;
        bool _S23 = dgI_0 < 1024U;
        if(_S23)
        {
            (*(&kernelContext_3)->sv_0)[dgI_0] = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_0 - 1U], (*(&kernelContext_3)->sv_0)[dgI_0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_1 = _S22 * 2U - 1U;
        bool _S24 = dgI_1 < 1024U;
        if(_S24)
        {
            (*(&kernelContext_3)->sv_0)[dgI_1] = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_1 - 2U], (*(&kernelContext_3)->sv_0)[dgI_1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_2 = _S21 * 4U * 2U - 1U;
        bool _S25 = dgI_2 < 1024U;
        if(_S25)
        {
            (*(&kernelContext_3)->sv_0)[dgI_2] = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_2 - 4U], (*(&kernelContext_3)->sv_0)[dgI_2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_3 = _S21 * 8U * 2U - 1U;
        bool _S26 = dgI_3 < 1024U;
        if(_S26)
        {
            (*(&kernelContext_3)->sv_0)[dgI_3] = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_3 - 8U], (*(&kernelContext_3)->sv_0)[dgI_3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_4 = _S21 * 16U * 2U - 1U;
        bool _S27 = dgI_4 < 1024U;
        if(_S27)
        {
            (*(&kernelContext_3)->sv_0)[dgI_4] = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_4 - 16U], (*(&kernelContext_3)->sv_0)[dgI_4]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_5 = _S21 * 32U * 2U - 1U;
        bool _S28 = dgI_5 < 1024U;
        if(_S28)
        {
            (*(&kernelContext_3)->sv_0)[dgI_5] = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_5 - 32U], (*(&kernelContext_3)->sv_0)[dgI_5]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_6 = _S21 * 64U * 2U - 1U;
        bool _S29 = dgI_6 < 1024U;
        if(_S29)
        {
            (*(&kernelContext_3)->sv_0)[dgI_6] = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_6 - 64U], (*(&kernelContext_3)->sv_0)[dgI_6]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_7 = _S21 * 128U * 2U - 1U;
        bool _S30 = dgI_7 < 1024U;
        if(_S30)
        {
            (*(&kernelContext_3)->sv_0)[dgI_7] = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_7 - 128U], (*(&kernelContext_3)->sv_0)[dgI_7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_8 = _S21 * 256U * 2U - 1U;
        bool _S31 = dgI_8 < 1024U;
        if(_S31)
        {
            (*(&kernelContext_3)->sv_0)[dgI_8] = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_8 - 256U], (*(&kernelContext_3)->sv_0)[dgI_8]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_9 = _S21 * 512U * 2U - 1U;
        bool _S32 = dgI_9 < 1024U;
        if(_S32)
        {
            (*(&kernelContext_3)->sv_0)[dgI_9] = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_9 - 512U], (*(&kernelContext_3)->sv_0)[dgI_9]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S16)
        {
            (*(&kernelContext_3)->sv_0)[1023U] = 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S32)
        {
            float _S33 = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_9], (*(&kernelContext_3)->sv_0)[dgI_9 - 512U]);
            (*(&kernelContext_3)->sv_0)[dgI_9 - 512U] = (*(&kernelContext_3)->sv_0)[dgI_9];
            (*(&kernelContext_3)->sv_0)[dgI_9] = _S33;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S31)
        {
            float _S34 = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_8], (*(&kernelContext_3)->sv_0)[dgI_8 - 256U]);
            (*(&kernelContext_3)->sv_0)[dgI_8 - 256U] = (*(&kernelContext_3)->sv_0)[dgI_8];
            (*(&kernelContext_3)->sv_0)[dgI_8] = _S34;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S30)
        {
            float _S35 = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_7], (*(&kernelContext_3)->sv_0)[dgI_7 - 128U]);
            (*(&kernelContext_3)->sv_0)[dgI_7 - 128U] = (*(&kernelContext_3)->sv_0)[dgI_7];
            (*(&kernelContext_3)->sv_0)[dgI_7] = _S35;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S29)
        {
            float _S36 = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_6], (*(&kernelContext_3)->sv_0)[dgI_6 - 64U]);
            (*(&kernelContext_3)->sv_0)[dgI_6 - 64U] = (*(&kernelContext_3)->sv_0)[dgI_6];
            (*(&kernelContext_3)->sv_0)[dgI_6] = _S36;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S28)
        {
            float _S37 = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_5], (*(&kernelContext_3)->sv_0)[dgI_5 - 32U]);
            (*(&kernelContext_3)->sv_0)[dgI_5 - 32U] = (*(&kernelContext_3)->sv_0)[dgI_5];
            (*(&kernelContext_3)->sv_0)[dgI_5] = _S37;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S27)
        {
            float _S38 = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_4], (*(&kernelContext_3)->sv_0)[dgI_4 - 16U]);
            (*(&kernelContext_3)->sv_0)[dgI_4 - 16U] = (*(&kernelContext_3)->sv_0)[dgI_4];
            (*(&kernelContext_3)->sv_0)[dgI_4] = _S38;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S26)
        {
            float _S39 = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_3], (*(&kernelContext_3)->sv_0)[dgI_3 - 8U]);
            (*(&kernelContext_3)->sv_0)[dgI_3 - 8U] = (*(&kernelContext_3)->sv_0)[dgI_3];
            (*(&kernelContext_3)->sv_0)[dgI_3] = _S39;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S25)
        {
            float _S40 = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_2], (*(&kernelContext_3)->sv_0)[dgI_2 - 4U]);
            (*(&kernelContext_3)->sv_0)[dgI_2 - 4U] = (*(&kernelContext_3)->sv_0)[dgI_2];
            (*(&kernelContext_3)->sv_0)[dgI_2] = _S40;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S24)
        {
            float _S41 = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_1], (*(&kernelContext_3)->sv_0)[dgI_1 - 2U]);
            (*(&kernelContext_3)->sv_0)[dgI_1 - 2U] = (*(&kernelContext_3)->sv_0)[dgI_1];
            (*(&kernelContext_3)->sv_0)[dgI_1] = _S41;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S23)
        {
            float _S42 = dg_op_add_0((*(&kernelContext_3)->sv_0)[dgI_0], (*(&kernelContext_3)->sv_0)[dgI_0 - 1U]);
            (*(&kernelContext_3)->sv_0)[dgI_0 - 1U] = (*(&kernelContext_3)->sv_0)[dgI_0];
            (*(&kernelContext_3)->sv_0)[dgI_0] = _S42;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ex_0 = (*(&kernelContext_3)->sv_0)[t_2];
        if(t_2 == 1023U)
        {
            *(&kernelContext_3)->gTotal_0 = ex_0 + e_0;
        }
        if(_S16)
        {
            *(&kernelContext_3)->gFound_0 = 4294967295U;
            *(&kernelContext_3)->gLast_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float target_0 = frac_0 * *(&kernelContext_3)->gTotal_0;
        bool _S43 = e_0 > 0.0;
        if(_S43)
        {
            uint _S44 = atomic_fetch_max_explicit(((atomic_uint threadgroup*)((&kernelContext_3)->gLast_0)), t_2, memory_order_relaxed);
        }
        if(_S43)
        {
            dgValid_0 = (ex_0 + e_0) > target_0;
        }
        else
        {
            dgValid_0 = false;
        }
        if(dgValid_0)
        {
            dgValid_0 = ex_0 <= target_0;
        }
        else
        {
            dgValid_0 = false;
        }
        if(dgValid_0)
        {
            uint _S45 = atomic_fetch_min_explicit(((atomic_uint threadgroup*)((&kernelContext_3)->gFound_0)), t_2, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S16)
        {
            uint device* _S46 = (&kernelContext_3)->tokens_0+row_1;
            if((*(&kernelContext_3)->gFound_0) == 4294967295U)
            {
                w_0 = *(&kernelContext_3)->gLast_0;
            }
            else
            {
                w_0 = *(&kernelContext_3)->gFound_0;
            }
            *_S46 = _S20 + w_0;
        }
        return;
    }
    if(((&kernelContext_3)->p_0->topK_0) == 0U)
    {
        g_0 = min(n_0, 1024U);
    }
    else
    {
        g_0 = min((&kernelContext_3)->p_0->topK_0, n_0);
    }
    uint K2_0 = pow2Above_0(g_0);
    uint _S47 = min(g_0, (&kernelContext_3)->p_0->stride_0);
    uint slots_1 = pow2Above_0((&kernelContext_3)->p_0->lists_0) * _S47;
    uint dgL_0 = t_2 / _S47;
    uint dgQ_0 = t_2 % _S47;
    if(t_2 < slots_1)
    {
        dgValid_0 = dgL_0 < ((&kernelContext_3)->p_0->lists_0);
    }
    else
    {
        dgValid_0 = false;
    }
    uint dgAt_0 = (row_1 * (&kernelContext_3)->p_0->lists_0 + dgL_0) * (&kernelContext_3)->p_0->stride_0 + dgQ_0;
    if(dgValid_0)
    {
        wg_0 = *((&kernelContext_3)->sortedV_0+dgAt_0);
    }
    else
    {
        wg_0 = dg_negative_infinity_0();
    }
    (*(&kernelContext_3)->sv_0)[t_2] = wg_0;
    if(dgValid_0)
    {
        w_0 = *((&kernelContext_3)->sortedI_0+dgAt_0);
    }
    else
    {
        w_0 = 4294967295U;
    }
    (*(&kernelContext_3)->si_0)[t_2] = w_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mergeTree_0(t_2, _S47, slots_1, &kernelContext_3);
    bool _S48 = t_2 < g_0;
    if(_S48)
    {
        wg_0 = (*(&kernelContext_3)->sv_0)[t_2];
    }
    else
    {
        wg_0 = dg_negative_infinity_0();
    }
    uint cix_0;
    if(_S48)
    {
        cix_0 = (*(&kernelContext_3)->si_0)[t_2];
    }
    else
    {
        cix_0 = 4294967295U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*(&kernelContext_3)->sv_0)[t_2] = wg_0;
    uint _S49 = cix_0;
    (*(&kernelContext_3)->si_0)[t_2] = cix_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S48)
    {
        e_0 = exp(((*(&kernelContext_3)->sv_0)[t_2] - mx_0) * (&kernelContext_3)->p_0->invT_0);
    }
    else
    {
        e_0 = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*(&kernelContext_3)->sv_0)[t_2] = e_0;
    scanF_0(t_2, K2_0, &kernelContext_3);
    float incl_0 = (*(&kernelContext_3)->sv_0)[t_2] + e_0;
    if(t_2 == (K2_0 - 1U))
    {
        *(&kernelContext_3)->gTotal_0 = incl_0;
    }
    if(_S16)
    {
        *(&kernelContext_3)->gFound_0 = 4294967295U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float cut_0 = (&kernelContext_3)->p_0->topP_0 * *(&kernelContext_3)->gTotal_0;
    if(((&kernelContext_3)->p_0->topP_0) < 1.0)
    {
        dgValid_0 = _S48;
    }
    else
    {
        dgValid_0 = false;
    }
    if(dgValid_0)
    {
        dgValid_0 = incl_0 >= cut_0;
    }
    else
    {
        dgValid_0 = false;
    }
    if(dgValid_0)
    {
        if(_S16)
        {
            dgValid_0 = true;
        }
        else
        {
            dgValid_0 = ((*(&kernelContext_3)->sv_0)[t_2]) < cut_0;
        }
    }
    else
    {
        dgValid_0 = false;
    }
    if(dgValid_0)
    {
        *(&kernelContext_3)->gFound_0 = t_2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint last_0;
    if((*(&kernelContext_3)->gFound_0) == 4294967295U)
    {
        last_0 = g_0 - 1U;
    }
    else
    {
        last_0 = *(&kernelContext_3)->gFound_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_2 == last_0)
    {
        *(&kernelContext_3)->gU_0 = rnd_0 * incl_0;
    }
    if(_S16)
    {
        *(&kernelContext_3)->gFound_0 = 4294967295U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float u_1 = *(&kernelContext_3)->gU_0;
    if(t_2 <= last_0)
    {
        dgValid_0 = incl_0 > u_1;
    }
    else
    {
        dgValid_0 = false;
    }
    if(dgValid_0)
    {
        if(_S16)
        {
            dgValid_0 = true;
        }
        else
        {
            dgValid_0 = ((*(&kernelContext_3)->sv_0)[t_2]) <= u_1;
        }
    }
    else
    {
        dgValid_0 = false;
    }
    if(dgValid_0)
    {
        uint _S50 = atomic_fetch_min_explicit(((atomic_uint threadgroup*)((&kernelContext_3)->gFound_0)), t_2, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pick_0;
    if((*(&kernelContext_3)->gFound_0) == 4294967295U)
    {
        pick_0 = last_0;
    }
    else
    {
        pick_0 = *(&kernelContext_3)->gFound_0;
    }
    if(t_2 == pick_0)
    {
        *((&kernelContext_3)->tokens_0+row_1) = _S49;
    }
    return;
}

