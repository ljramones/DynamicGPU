#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant uint fc_ROW_TIER_0 [[function_constant(0)]];
constant uint ROW_TIER_0 = is_function_constant_defined(fc_ROW_TIER_0) ? fc_ROW_TIER_0 : 8U;
float dg_negative_infinity_0()
{
    return (as_type<float>((4286578688U)));
}

float attnLo_0(uint w_0)
{
    return (as_type<half>((ushort)((w_0 & 65535U))));
}

float attnHi_0(uint w_1)
{
    return (as_type<half>((ushort)((w_1 >> 16U))));
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
    uint device* table_0;
    uint device* q_0;
    uint device* k_0;
    uint device* v_0;
    float device* partialML_0;
    float device* partialAcc_0;
    float device* o_0;
    array<float, int(8)> threadgroup* rowM_0;
    array<float, int(8)> threadgroup* rowL_0;
    array<uint, int(512)> threadgroup* qs_0;
    array<float, int(1024)> threadgroup* ss_0;
    array<float, int(128)> threadgroup* red_0;
    array<float, int(8)> threadgroup* rowA_0;
};

[[kernel]] void attn_decode_128_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(7)]], uint device* table_1 [[buffer(6)]], uint device* q_1 [[buffer(0)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partialML_1 [[buffer(4)]], float device* partialAcc_1 [[buffer(5)]], float device* o_1 [[buffer(3)]])
{
    bool _S5;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->table_0 = table_1;
    (&kernelContext_0)->q_0 = q_1;
    (&kernelContext_0)->k_0 = k_1;
    (&kernelContext_0)->v_0 = v_1;
    (&kernelContext_0)->partialML_0 = partialML_1;
    (&kernelContext_0)->partialAcc_0 = partialAcc_1;
    (&kernelContext_0)->o_0 = o_1;
    threadgroup array<float, int(8)> rowM_1;
    (&kernelContext_0)->rowM_0 = &rowM_1;
    threadgroup array<float, int(8)> rowL_1;
    (&kernelContext_0)->rowL_0 = &rowL_1;
    threadgroup array<uint, int(512)> qs_1;
    (&kernelContext_0)->qs_0 = &qs_1;
    threadgroup array<float, int(1024)> ss_1;
    (&kernelContext_0)->ss_0 = &ss_1;
    threadgroup array<float, int(128)> red_1;
    (&kernelContext_0)->red_0 = &red_1;
    threadgroup array<float, int(8)> rowA_1;
    (&kernelContext_0)->rowA_0 = &rowA_1;
    uint t_0 = local_0.x;
    uint _S6 = group_0.y;
    uint split_0 = group_0.z;
    uint seq_0 = group_0.x;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint _S7 = min(hpk_0 * (&kernelContext_0)->p_0->queries_0, ROW_TIER_0);
    uint keys_1;
    if(((&kernelContext_0)->p_0->batch_0) == 0U)
    {
        keys_1 = (&kernelContext_0)->p_0->keys_0;
    }
    else
    {
        keys_1 = *((&kernelContext_0)->table_0+2U * seq_0);
    }
    uint r_0;
    if(((&kernelContext_0)->p_0->batch_0) == 0U)
    {
        r_0 = 0U;
    }
    else
    {
        r_0 = 2U * (&kernelContext_0)->p_0->batch_0 + *((&kernelContext_0)->table_0+(2U * seq_0 + 1U));
    }
    uint _S8 = seq_0 * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0;
    uint _S9 = max((&kernelContext_0)->p_0->batch_0, 1U) * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0;
    uint _S10 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint _S11 = keys_1 - (&kernelContext_0)->p_0->queries_0;
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S12 = min(keys_1, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    if(_S12 > kStart_0)
    {
        keys_1 = _S12 - kStart_0;
    }
    else
    {
        keys_1 = 0U;
    }
    uint _S13 = 1024U / max(_S7, 1U);
    uint _S14 = max(128U, _S13 / 128U * 128U);
    bool _S15 = t_0 < 8U;
    if(_S15)
    {
        (*(&kernelContext_0)->rowM_0)[t_0] = dg_negative_infinity_0();
        (*(&kernelContext_0)->rowL_0)[t_0] = 0.0;
    }
    bool visible_0;
    uint w_2;
    uint e_0 = t_0;
    for(;;)
    {
        if(e_0 < (ROW_TIER_0 * 64U))
        {
        }
        else
        {
            break;
        }
        uint r_1 = e_0 / 64U;
        uint pr_0 = e_0 % 64U;
        if(r_1 < _S7)
        {
            visible_0 = pr_0 < _S10;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S16 = _S6 * hpk_0;
            uint _S17 = r_1 / (&kernelContext_0)->p_0->queries_0;
            uint h_0 = _S16 + _S17;
            uint i_0 = r_1 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((_S8 + h_0 * (&kernelContext_0)->p_0->queries_0 + i_0) * _S10 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_0] = w_2;
        e_0 = e_0 + 128U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S18 = t_0 % 64U;
    uint _S19 = t_0 / 64U;
    thread array<float, int(8)> lo_0;
    thread array<float, int(8)> hi_0;
    float pj_0;
    float acc_0;
    float acc_1;
    lo_0[0U] = 0.0;
    hi_0[0U] = 0.0;
    lo_0[1U] = 0.0;
    hi_0[1U] = 0.0;
    lo_0[2U] = 0.0;
    hi_0[2U] = 0.0;
    lo_0[3U] = 0.0;
    hi_0[3U] = 0.0;
    lo_0[4U] = 0.0;
    hi_0[4U] = 0.0;
    lo_0[5U] = 0.0;
    hi_0[5U] = 0.0;
    lo_0[6U] = 0.0;
    hi_0[6U] = 0.0;
    lo_0[7U] = 0.0;
    hi_0[7U] = 0.0;
    w_2 = kStart_0;
    for(;;)
    {
        if(w_2 < _S12)
        {
        }
        else
        {
            break;
        }
        int _S20;
        int _S21;
        int _S22;
        int _S23;
        int _S24;
        int _S25;
        int _S26;
        int _S27;
        int _S28;
        int _S29;
        int _S30;
        int _S31;
        int _S32;
        int _S33;
        int _S34;
        int _S35;
        uint kw_0;
        uint kw_1;
        uint kw_2;
        uint kw_3;
        uint kw_4;
        uint kw_5;
        uint kw_6;
        uint kw_7;
        uint p0_0;
        uint _S36 = min(_S14, _S12 - w_2);
        uint j_0 = t_0;
        for(;;)
        {
            if(j_0 < _S36)
            {
            }
            else
            {
                break;
            }
            uint key_0 = w_2 + j_0;
            if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
            {
                e_0 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * _S10;
            }
            else
            {
                uint _S37 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
                uint _S38 = (*((&kernelContext_0)->table_0+(r_0 + _S37)) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                uint _S39 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
                e_0 = (_S38 + _S39) * _S10;
            }
            thread array<float, int(8)> s_0;
            s_0[0U] = 0.0;
            s_0[1U] = 0.0;
            s_0[2U] = 0.0;
            s_0[3U] = 0.0;
            s_0[4U] = 0.0;
            s_0[5U] = 0.0;
            s_0[6U] = 0.0;
            s_0[7U] = 0.0;
            p0_0 = 0U;
            for(;;)
            {
                if(p0_0 < 64U)
                {
                }
                else
                {
                    break;
                }
                if(p0_0 < _S10)
                {
                    kw_7 = *((&kernelContext_0)->k_0+(e_0 + p0_0));
                }
                else
                {
                    kw_7 = 0U;
                }
                float _S40 = attnLo_0(kw_7);
                float _S41 = attnHi_0(kw_7);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[p0_0]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[p0_0]), _S40, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S40, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S40, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S40, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + p0_0]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + p0_0]), _S40, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[320U + p0_0]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[320U + p0_0]), _S40, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + p0_0]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + p0_0]), _S40, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[448U + p0_0]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[448U + p0_0]), _S40, s_0[7U]));
                }
                uint pr_1 = p0_0 + 1U;
                if(pr_1 < _S10)
                {
                    kw_6 = *((&kernelContext_0)->k_0+(e_0 + pr_1));
                }
                else
                {
                    kw_6 = 0U;
                }
                float _S42 = attnLo_0(kw_6);
                float _S43 = attnHi_0(kw_6);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_1]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_1]), _S42, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S42, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S42, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S42, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_1]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_1]), _S42, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[320U + pr_1]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[320U + pr_1]), _S42, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_1]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_1]), _S42, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[448U + pr_1]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[448U + pr_1]), _S42, s_0[7U]));
                }
                uint pr_2 = p0_0 + 2U;
                if(pr_2 < _S10)
                {
                    kw_5 = *((&kernelContext_0)->k_0+(e_0 + pr_2));
                }
                else
                {
                    kw_5 = 0U;
                }
                float _S44 = attnLo_0(kw_5);
                float _S45 = attnHi_0(kw_5);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_2]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_2]), _S44, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S44, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S44, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S44, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_2]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_2]), _S44, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[320U + pr_2]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[320U + pr_2]), _S44, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_2]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_2]), _S44, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[448U + pr_2]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[448U + pr_2]), _S44, s_0[7U]));
                }
                uint pr_3 = p0_0 + 3U;
                if(pr_3 < _S10)
                {
                    kw_4 = *((&kernelContext_0)->k_0+(e_0 + pr_3));
                }
                else
                {
                    kw_4 = 0U;
                }
                float _S46 = attnLo_0(kw_4);
                float _S47 = attnHi_0(kw_4);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_3]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_3]), _S46, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S46, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S46, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S46, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_3]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_3]), _S46, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[320U + pr_3]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[320U + pr_3]), _S46, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_3]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_3]), _S46, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[448U + pr_3]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[448U + pr_3]), _S46, s_0[7U]));
                }
                uint pr_4 = p0_0 + 4U;
                if(pr_4 < _S10)
                {
                    kw_3 = *((&kernelContext_0)->k_0+(e_0 + pr_4));
                }
                else
                {
                    kw_3 = 0U;
                }
                float _S48 = attnLo_0(kw_3);
                float _S49 = attnHi_0(kw_3);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_4]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_4]), _S48, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S48, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S48, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S48, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_4]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_4]), _S48, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[320U + pr_4]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[320U + pr_4]), _S48, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_4]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_4]), _S48, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[448U + pr_4]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[448U + pr_4]), _S48, s_0[7U]));
                }
                uint pr_5 = p0_0 + 5U;
                if(pr_5 < _S10)
                {
                    kw_2 = *((&kernelContext_0)->k_0+(e_0 + pr_5));
                }
                else
                {
                    kw_2 = 0U;
                }
                float _S50 = attnLo_0(kw_2);
                float _S51 = attnHi_0(kw_2);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_5]), _S51, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_5]), _S50, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S51, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S50, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S51, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S50, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S51, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S50, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_5]), _S51, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_5]), _S50, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[320U + pr_5]), _S51, fma(attnLo_0((*(&kernelContext_0)->qs_0)[320U + pr_5]), _S50, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_5]), _S51, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_5]), _S50, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[448U + pr_5]), _S51, fma(attnLo_0((*(&kernelContext_0)->qs_0)[448U + pr_5]), _S50, s_0[7U]));
                }
                uint pr_6 = p0_0 + 6U;
                if(pr_6 < _S10)
                {
                    kw_1 = *((&kernelContext_0)->k_0+(e_0 + pr_6));
                }
                else
                {
                    kw_1 = 0U;
                }
                float _S52 = attnLo_0(kw_1);
                float _S53 = attnHi_0(kw_1);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_6]), _S53, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_6]), _S52, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S53, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S52, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S53, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S52, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S53, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S52, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_6]), _S53, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_6]), _S52, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[320U + pr_6]), _S53, fma(attnLo_0((*(&kernelContext_0)->qs_0)[320U + pr_6]), _S52, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_6]), _S53, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_6]), _S52, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[448U + pr_6]), _S53, fma(attnLo_0((*(&kernelContext_0)->qs_0)[448U + pr_6]), _S52, s_0[7U]));
                }
                uint pr_7 = p0_0 + 7U;
                if(pr_7 < _S10)
                {
                    kw_0 = *((&kernelContext_0)->k_0+(e_0 + pr_7));
                }
                else
                {
                    kw_0 = 0U;
                }
                float _S54 = attnLo_0(kw_0);
                float _S55 = attnHi_0(kw_0);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_7]), _S55, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_7]), _S54, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S55, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S54, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S55, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S54, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S55, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S54, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_7]), _S55, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_7]), _S54, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[320U + pr_7]), _S55, fma(attnLo_0((*(&kernelContext_0)->qs_0)[320U + pr_7]), _S54, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_7]), _S55, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_7]), _S54, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[448U + pr_7]), _S55, fma(attnLo_0((*(&kernelContext_0)->qs_0)[448U + pr_7]), _S54, s_0[7U]));
                }
                p0_0 = p0_0 + 8U;
            }
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        if(0U >= _S7)
                        {
                            _S35 = int(2);
                            break;
                        }
                        uint i_1 = 0U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_1 + _S11);
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[0U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[j_0] = acc_1;
                        _S35 = int(4);
                        break;
                    }
                    if(_S35 != int(4))
                    {
                        _S34 = _S35;
                        break;
                    }
                    _S34 = int(3);
                    break;
                }
                if(_S34 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(1U >= _S7)
                        {
                            _S33 = int(2);
                            break;
                        }
                        uint i_2 = 1U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_2 + _S11);
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[1U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[_S14 + j_0] = acc_1;
                        _S33 = int(4);
                        break;
                    }
                    if(_S33 != int(4))
                    {
                        _S32 = _S33;
                        break;
                    }
                    _S32 = int(3);
                    break;
                }
                if(_S32 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(2U >= _S7)
                        {
                            _S31 = int(2);
                            break;
                        }
                        uint i_3 = 2U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_3 + _S11);
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[2U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[2U * _S14 + j_0] = acc_1;
                        _S31 = int(4);
                        break;
                    }
                    if(_S31 != int(4))
                    {
                        _S30 = _S31;
                        break;
                    }
                    _S30 = int(3);
                    break;
                }
                if(_S30 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(3U >= _S7)
                        {
                            _S29 = int(2);
                            break;
                        }
                        uint i_4 = 3U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_4 + _S11);
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[3U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[3U * _S14 + j_0] = acc_1;
                        _S29 = int(4);
                        break;
                    }
                    if(_S29 != int(4))
                    {
                        _S28 = _S29;
                        break;
                    }
                    _S28 = int(3);
                    break;
                }
                if(_S28 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(4U >= _S7)
                        {
                            _S27 = int(2);
                            break;
                        }
                        uint i_5 = 4U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_5 + _S11);
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[4U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[4U * _S14 + j_0] = acc_1;
                        _S27 = int(4);
                        break;
                    }
                    if(_S27 != int(4))
                    {
                        _S26 = _S27;
                        break;
                    }
                    _S26 = int(3);
                    break;
                }
                if(_S26 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(5U >= _S7)
                        {
                            _S25 = int(2);
                            break;
                        }
                        uint i_6 = 5U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_6 + _S11);
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[5U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[5U * _S14 + j_0] = acc_1;
                        _S25 = int(4);
                        break;
                    }
                    if(_S25 != int(4))
                    {
                        _S24 = _S25;
                        break;
                    }
                    _S24 = int(3);
                    break;
                }
                if(_S24 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(6U >= _S7)
                        {
                            _S23 = int(2);
                            break;
                        }
                        uint i_7 = 6U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_7 + _S11);
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[6U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[6U * _S14 + j_0] = acc_1;
                        _S23 = int(4);
                        break;
                    }
                    if(_S23 != int(4))
                    {
                        _S22 = _S23;
                        break;
                    }
                    _S22 = int(3);
                    break;
                }
                if(_S22 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(7U >= _S7)
                        {
                            _S21 = int(2);
                            break;
                        }
                        uint i_8 = 7U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_8 + _S11);
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[7U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[7U * _S14 + j_0] = acc_1;
                        _S21 = int(4);
                        break;
                    }
                    if(_S21 != int(4))
                    {
                        _S20 = _S21;
                        break;
                    }
                    _S20 = int(3);
                    break;
                }
                if(_S20 != int(3))
                {
                    break;
                }
                break;
            }
            j_0 = j_0 + 128U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        p0_0 = 0U;
        for(;;)
        {
            if(p0_0 < _S7)
            {
            }
            else
            {
                break;
            }
            float _S56 = dg_negative_infinity_0();
            kw_7 = t_0;
            acc_1 = _S56;
            for(;;)
            {
                if(kw_7 < _S36)
                {
                }
                else
                {
                    break;
                }
                float _S57 = max(acc_1, (*(&kernelContext_0)->ss_0)[p0_0 * _S14 + kw_7]);
                kw_7 = kw_7 + 128U;
                acc_1 = _S57;
            }
            (*(&kernelContext_0)->red_0)[t_0] = acc_1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S58 = t_0 < 64U;
            if(_S58)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S59 = t_0 < 32U;
            if(_S59)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S60 = t_0 < 16U;
            if(_S60)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S15)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S61 = t_0 < 4U;
            if(_S61)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S62 = t_0 < 2U;
            if(_S62)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S63 = t_0 < 1U;
            if(_S63)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S64 = t_0 == 0U;
            if(_S64)
            {
                float mOld_0 = (*(&kernelContext_0)->rowM_0)[p0_0];
                float _S65 = max((*(&kernelContext_0)->rowM_0)[p0_0], (*(&kernelContext_0)->red_0)[int(0)]);
                if(((*(&kernelContext_0)->rowM_0)[p0_0]) == _S56)
                {
                    acc_0 = 0.0;
                }
                else
                {
                    acc_0 = exp(mOld_0 - _S65);
                }
                (*(&kernelContext_0)->rowA_0)[p0_0] = acc_0;
                (*(&kernelContext_0)->rowM_0)[p0_0] = _S65;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float _S66 = (*(&kernelContext_0)->rowM_0)[p0_0];
            kw_6 = t_0;
            acc_0 = 0.0;
            for(;;)
            {
                if(kw_6 < _S36)
                {
                }
                else
                {
                    break;
                }
                float sv_0 = (*(&kernelContext_0)->ss_0)[p0_0 * _S14 + kw_6];
                if(((*(&kernelContext_0)->ss_0)[p0_0 * _S14 + kw_6]) == _S56)
                {
                    pj_0 = 0.0;
                }
                else
                {
                    pj_0 = exp(sv_0 - _S66);
                }
                (*(&kernelContext_0)->ss_0)[p0_0 * _S14 + kw_6] = pj_0;
                float _S67 = acc_0 + pj_0;
                kw_6 = kw_6 + 128U;
                acc_0 = _S67;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (*(&kernelContext_0)->red_0)[t_0] = acc_0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S58)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S59)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S60)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S15)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S61)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S62)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S63)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S64)
            {
                (*(&kernelContext_0)->rowL_0)[p0_0] = (*(&kernelContext_0)->rowL_0)[p0_0] * (*(&kernelContext_0)->rowA_0)[p0_0] + (*(&kernelContext_0)->red_0)[int(0)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            p0_0 = p0_0 + 1U;
        }
        if(0U < _S7)
        {
            float a_3 = (*(&kernelContext_0)->rowA_0)[0U];
            lo_0[0U] = lo_0[0U] * (*(&kernelContext_0)->rowA_0)[0U];
            hi_0[0U] = hi_0[0U] * a_3;
        }
        if(1U < _S7)
        {
            float a_4 = (*(&kernelContext_0)->rowA_0)[1U];
            lo_0[1U] = lo_0[1U] * (*(&kernelContext_0)->rowA_0)[1U];
            hi_0[1U] = hi_0[1U] * a_4;
        }
        if(2U < _S7)
        {
            float a_5 = (*(&kernelContext_0)->rowA_0)[2U];
            lo_0[2U] = lo_0[2U] * (*(&kernelContext_0)->rowA_0)[2U];
            hi_0[2U] = hi_0[2U] * a_5;
        }
        if(3U < _S7)
        {
            float a_6 = (*(&kernelContext_0)->rowA_0)[3U];
            lo_0[3U] = lo_0[3U] * (*(&kernelContext_0)->rowA_0)[3U];
            hi_0[3U] = hi_0[3U] * a_6;
        }
        if(4U < _S7)
        {
            float a_7 = (*(&kernelContext_0)->rowA_0)[4U];
            lo_0[4U] = lo_0[4U] * (*(&kernelContext_0)->rowA_0)[4U];
            hi_0[4U] = hi_0[4U] * a_7;
        }
        if(5U < _S7)
        {
            float a_8 = (*(&kernelContext_0)->rowA_0)[5U];
            lo_0[5U] = lo_0[5U] * (*(&kernelContext_0)->rowA_0)[5U];
            hi_0[5U] = hi_0[5U] * a_8;
        }
        if(6U < _S7)
        {
            float a_9 = (*(&kernelContext_0)->rowA_0)[6U];
            lo_0[6U] = lo_0[6U] * (*(&kernelContext_0)->rowA_0)[6U];
            hi_0[6U] = hi_0[6U] * a_9;
        }
        if(7U < _S7)
        {
            float a_10 = (*(&kernelContext_0)->rowA_0)[7U];
            lo_0[7U] = lo_0[7U] * (*(&kernelContext_0)->rowA_0)[7U];
            hi_0[7U] = hi_0[7U] * a_10;
        }
        if(_S18 < _S10)
        {
            kw_7 = _S19;
            for(;;)
            {
                bool _S68 = kw_7 < _S36;
                if(_S68)
                {
                }
                else
                {
                    break;
                }
                thread array<uint, int(8)> vw_0;
                if(_S68)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        e_0 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + kw_7)) * _S10 + _S18;
                    }
                    else
                    {
                        uint _S69 = w_2 + kw_7;
                        uint _S70 = _S69 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S71 = (*((&kernelContext_0)->table_0+(r_0 + _S70)) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S72 = _S69 % (&kernelContext_0)->p_0->pageSize_0;
                        e_0 = (_S71 + _S72) * _S10 + _S18;
                    }
                    e_0 = *((&kernelContext_0)->v_0+e_0);
                }
                else
                {
                    e_0 = 0U;
                }
                vw_0[0U] = e_0;
                uint j_1 = kw_7 + 2U;
                if(j_1 < _S36)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_6 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_1)) * _S10 + _S18;
                    }
                    else
                    {
                        uint _S73 = w_2 + j_1;
                        uint _S74 = _S73 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S75 = (*((&kernelContext_0)->table_0+(r_0 + _S74)) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S76 = _S73 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_6 = (_S75 + _S76) * _S10 + _S18;
                    }
                    kw_6 = *((&kernelContext_0)->v_0+kw_6);
                }
                else
                {
                    kw_6 = 0U;
                }
                vw_0[1U] = kw_6;
                uint j_2 = kw_7 + 4U;
                if(j_2 < _S36)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_5 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_2)) * _S10 + _S18;
                    }
                    else
                    {
                        uint _S77 = w_2 + j_2;
                        uint _S78 = _S77 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S79 = (*((&kernelContext_0)->table_0+(r_0 + _S78)) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S80 = _S77 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_5 = (_S79 + _S80) * _S10 + _S18;
                    }
                    kw_5 = *((&kernelContext_0)->v_0+kw_5);
                }
                else
                {
                    kw_5 = 0U;
                }
                vw_0[2U] = kw_5;
                uint j_3 = kw_7 + 6U;
                if(j_3 < _S36)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_4 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_3)) * _S10 + _S18;
                    }
                    else
                    {
                        uint _S81 = w_2 + j_3;
                        uint _S82 = _S81 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S83 = (*((&kernelContext_0)->table_0+(r_0 + _S82)) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S84 = _S81 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_4 = (_S83 + _S84) * _S10 + _S18;
                    }
                    kw_4 = *((&kernelContext_0)->v_0+kw_4);
                }
                else
                {
                    kw_4 = 0U;
                }
                vw_0[3U] = kw_4;
                uint j_4 = kw_7 + 8U;
                if(j_4 < _S36)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_3 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_4)) * _S10 + _S18;
                    }
                    else
                    {
                        uint _S85 = w_2 + j_4;
                        uint _S86 = _S85 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S87 = (*((&kernelContext_0)->table_0+(r_0 + _S86)) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S88 = _S85 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_3 = (_S87 + _S88) * _S10 + _S18;
                    }
                    kw_3 = *((&kernelContext_0)->v_0+kw_3);
                }
                else
                {
                    kw_3 = 0U;
                }
                vw_0[4U] = kw_3;
                uint j_5 = kw_7 + 10U;
                if(j_5 < _S36)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_2 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_5)) * _S10 + _S18;
                    }
                    else
                    {
                        uint _S89 = w_2 + j_5;
                        uint _S90 = _S89 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S91 = (*((&kernelContext_0)->table_0+(r_0 + _S90)) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S92 = _S89 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_2 = (_S91 + _S92) * _S10 + _S18;
                    }
                    kw_2 = *((&kernelContext_0)->v_0+kw_2);
                }
                else
                {
                    kw_2 = 0U;
                }
                vw_0[5U] = kw_2;
                uint j_6 = kw_7 + 12U;
                if(j_6 < _S36)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_1 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_6)) * _S10 + _S18;
                    }
                    else
                    {
                        uint _S93 = w_2 + j_6;
                        uint _S94 = _S93 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S95 = (*((&kernelContext_0)->table_0+(r_0 + _S94)) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S96 = _S93 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_1 = (_S95 + _S96) * _S10 + _S18;
                    }
                    kw_1 = *((&kernelContext_0)->v_0+kw_1);
                }
                else
                {
                    kw_1 = 0U;
                }
                vw_0[6U] = kw_1;
                uint j_7 = kw_7 + 14U;
                if(j_7 < _S36)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_0 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_7)) * _S10 + _S18;
                    }
                    else
                    {
                        uint _S97 = w_2 + j_7;
                        uint _S98 = _S97 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S99 = (*((&kernelContext_0)->table_0+(r_0 + _S98)) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S100 = _S97 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_0 = (_S99 + _S100) * _S10 + _S18;
                    }
                    kw_0 = *((&kernelContext_0)->v_0+kw_0);
                }
                else
                {
                    kw_0 = 0U;
                }
                vw_0[7U] = kw_0;
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            if(kw_7 >= _S36)
                            {
                                _S35 = int(2);
                                visible_0 = false;
                                break;
                            }
                            float _S101 = attnLo_0(vw_0[0U]);
                            float _S102 = attnHi_0(vw_0[0U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        bool _S103 = 0U >= _S7;
                                        _S5 = _S103;
                                        if(_S103)
                                        {
                                            _S35 = int(5);
                                            break;
                                        }
                                        float pj_1 = (*(&kernelContext_0)->ss_0)[kw_7];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[kw_7], _S101, lo_0[0U]);
                                        hi_0[0U] = fma(pj_1, _S102, hi_0[0U]);
                                        _S35 = int(7);
                                        break;
                                    }
                                    if(_S35 != int(7))
                                    {
                                        _S34 = _S35;
                                        break;
                                    }
                                    _S34 = int(6);
                                    break;
                                }
                                if(_S34 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S7)
                                        {
                                            _S33 = int(5);
                                            break;
                                        }
                                        float pj_2 = (*(&kernelContext_0)->ss_0)[_S14 + kw_7];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S14 + kw_7], _S101, lo_0[1U]);
                                        hi_0[1U] = fma(pj_2, _S102, hi_0[1U]);
                                        _S33 = int(7);
                                        break;
                                    }
                                    if(_S33 != int(7))
                                    {
                                        _S32 = _S33;
                                        break;
                                    }
                                    _S32 = int(6);
                                    break;
                                }
                                if(_S32 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S7)
                                        {
                                            _S31 = int(5);
                                            break;
                                        }
                                        float pj_3 = (*(&kernelContext_0)->ss_0)[2U * _S14 + kw_7];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S14 + kw_7], _S101, lo_0[2U]);
                                        hi_0[2U] = fma(pj_3, _S102, hi_0[2U]);
                                        _S31 = int(7);
                                        break;
                                    }
                                    if(_S31 != int(7))
                                    {
                                        _S30 = _S31;
                                        break;
                                    }
                                    _S30 = int(6);
                                    break;
                                }
                                if(_S30 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S7)
                                        {
                                            _S29 = int(5);
                                            break;
                                        }
                                        float pj_4 = (*(&kernelContext_0)->ss_0)[3U * _S14 + kw_7];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S14 + kw_7], _S101, lo_0[3U]);
                                        hi_0[3U] = fma(pj_4, _S102, hi_0[3U]);
                                        _S29 = int(7);
                                        break;
                                    }
                                    if(_S29 != int(7))
                                    {
                                        _S28 = _S29;
                                        break;
                                    }
                                    _S28 = int(6);
                                    break;
                                }
                                if(_S28 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S7)
                                        {
                                            _S27 = int(5);
                                            break;
                                        }
                                        float pj_5 = (*(&kernelContext_0)->ss_0)[4U * _S14 + kw_7];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S14 + kw_7], _S101, lo_0[4U]);
                                        hi_0[4U] = fma(pj_5, _S102, hi_0[4U]);
                                        _S27 = int(7);
                                        break;
                                    }
                                    if(_S27 != int(7))
                                    {
                                        _S26 = _S27;
                                        break;
                                    }
                                    _S26 = int(6);
                                    break;
                                }
                                if(_S26 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S7)
                                        {
                                            _S25 = int(5);
                                            break;
                                        }
                                        float pj_6 = (*(&kernelContext_0)->ss_0)[5U * _S14 + kw_7];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S14 + kw_7], _S101, lo_0[5U]);
                                        hi_0[5U] = fma(pj_6, _S102, hi_0[5U]);
                                        _S25 = int(7);
                                        break;
                                    }
                                    if(_S25 != int(7))
                                    {
                                        _S24 = _S25;
                                        break;
                                    }
                                    _S24 = int(6);
                                    break;
                                }
                                if(_S24 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S7)
                                        {
                                            _S23 = int(5);
                                            break;
                                        }
                                        float pj_7 = (*(&kernelContext_0)->ss_0)[6U * _S14 + kw_7];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S14 + kw_7], _S101, lo_0[6U]);
                                        hi_0[6U] = fma(pj_7, _S102, hi_0[6U]);
                                        _S23 = int(7);
                                        break;
                                    }
                                    if(_S23 != int(7))
                                    {
                                        _S22 = _S23;
                                        break;
                                    }
                                    _S22 = int(6);
                                    break;
                                }
                                if(_S22 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S7)
                                        {
                                            _S21 = int(5);
                                            break;
                                        }
                                        float pj_8 = (*(&kernelContext_0)->ss_0)[7U * _S14 + kw_7];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S14 + kw_7], _S101, lo_0[7U]);
                                        hi_0[7U] = fma(pj_8, _S102, hi_0[7U]);
                                        _S21 = int(7);
                                        break;
                                    }
                                    if(_S21 != int(7))
                                    {
                                        _S20 = _S21;
                                        break;
                                    }
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S35 = int(4);
                            visible_0 = _S5;
                            break;
                        }
                        if(_S35 != int(4))
                        {
                            _S34 = _S35;
                            break;
                        }
                        _S34 = int(3);
                        break;
                    }
                    if(_S34 != int(3))
                    {
                        break;
                    }
                    int _S104;
                    int _S105;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_1 >= _S36)
                            {
                                _S33 = int(2);
                                break;
                            }
                            float _S106 = attnLo_0(vw_0[1U]);
                            float _S107 = attnHi_0(vw_0[1U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S33 = int(5);
                                            break;
                                        }
                                        float pj_9 = (*(&kernelContext_0)->ss_0)[j_1];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_1], _S106, lo_0[0U]);
                                        hi_0[0U] = fma(pj_9, _S107, hi_0[0U]);
                                        _S33 = int(7);
                                        break;
                                    }
                                    if(_S33 != int(7))
                                    {
                                        _S32 = _S33;
                                        break;
                                    }
                                    _S32 = int(6);
                                    break;
                                }
                                if(_S32 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S7)
                                        {
                                            _S31 = int(5);
                                            break;
                                        }
                                        float pj_10 = (*(&kernelContext_0)->ss_0)[_S14 + j_1];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S14 + j_1], _S106, lo_0[1U]);
                                        hi_0[1U] = fma(pj_10, _S107, hi_0[1U]);
                                        _S31 = int(7);
                                        break;
                                    }
                                    if(_S31 != int(7))
                                    {
                                        _S30 = _S31;
                                        break;
                                    }
                                    _S30 = int(6);
                                    break;
                                }
                                if(_S30 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S7)
                                        {
                                            _S29 = int(5);
                                            break;
                                        }
                                        float pj_11 = (*(&kernelContext_0)->ss_0)[2U * _S14 + j_1];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S14 + j_1], _S106, lo_0[2U]);
                                        hi_0[2U] = fma(pj_11, _S107, hi_0[2U]);
                                        _S29 = int(7);
                                        break;
                                    }
                                    if(_S29 != int(7))
                                    {
                                        _S28 = _S29;
                                        break;
                                    }
                                    _S28 = int(6);
                                    break;
                                }
                                if(_S28 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S7)
                                        {
                                            _S27 = int(5);
                                            break;
                                        }
                                        float pj_12 = (*(&kernelContext_0)->ss_0)[3U * _S14 + j_1];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S14 + j_1], _S106, lo_0[3U]);
                                        hi_0[3U] = fma(pj_12, _S107, hi_0[3U]);
                                        _S27 = int(7);
                                        break;
                                    }
                                    if(_S27 != int(7))
                                    {
                                        _S26 = _S27;
                                        break;
                                    }
                                    _S26 = int(6);
                                    break;
                                }
                                if(_S26 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S7)
                                        {
                                            _S25 = int(5);
                                            break;
                                        }
                                        float pj_13 = (*(&kernelContext_0)->ss_0)[4U * _S14 + j_1];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S14 + j_1], _S106, lo_0[4U]);
                                        hi_0[4U] = fma(pj_13, _S107, hi_0[4U]);
                                        _S25 = int(7);
                                        break;
                                    }
                                    if(_S25 != int(7))
                                    {
                                        _S24 = _S25;
                                        break;
                                    }
                                    _S24 = int(6);
                                    break;
                                }
                                if(_S24 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S7)
                                        {
                                            _S23 = int(5);
                                            break;
                                        }
                                        float pj_14 = (*(&kernelContext_0)->ss_0)[5U * _S14 + j_1];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S14 + j_1], _S106, lo_0[5U]);
                                        hi_0[5U] = fma(pj_14, _S107, hi_0[5U]);
                                        _S23 = int(7);
                                        break;
                                    }
                                    if(_S23 != int(7))
                                    {
                                        _S22 = _S23;
                                        break;
                                    }
                                    _S22 = int(6);
                                    break;
                                }
                                if(_S22 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S7)
                                        {
                                            _S21 = int(5);
                                            break;
                                        }
                                        float pj_15 = (*(&kernelContext_0)->ss_0)[6U * _S14 + j_1];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S14 + j_1], _S106, lo_0[6U]);
                                        hi_0[6U] = fma(pj_15, _S107, hi_0[6U]);
                                        _S21 = int(7);
                                        break;
                                    }
                                    if(_S21 != int(7))
                                    {
                                        _S20 = _S21;
                                        break;
                                    }
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S7)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_16 = (*(&kernelContext_0)->ss_0)[7U * _S14 + j_1];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S14 + j_1], _S106, lo_0[7U]);
                                        hi_0[7U] = fma(pj_16, _S107, hi_0[7U]);
                                        _S105 = int(7);
                                        break;
                                    }
                                    if(_S105 != int(7))
                                    {
                                        _S104 = _S105;
                                        break;
                                    }
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S33 = int(4);
                            break;
                        }
                        if(_S33 != int(4))
                        {
                            _S32 = _S33;
                            break;
                        }
                        _S32 = int(3);
                        break;
                    }
                    if(_S32 != int(3))
                    {
                        break;
                    }
                    int _S108;
                    int _S109;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_2 >= _S36)
                            {
                                _S31 = int(2);
                                break;
                            }
                            float _S110 = attnLo_0(vw_0[2U]);
                            float _S111 = attnHi_0(vw_0[2U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S31 = int(5);
                                            break;
                                        }
                                        float pj_17 = (*(&kernelContext_0)->ss_0)[j_2];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_2], _S110, lo_0[0U]);
                                        hi_0[0U] = fma(pj_17, _S111, hi_0[0U]);
                                        _S31 = int(7);
                                        break;
                                    }
                                    if(_S31 != int(7))
                                    {
                                        _S30 = _S31;
                                        break;
                                    }
                                    _S30 = int(6);
                                    break;
                                }
                                if(_S30 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S7)
                                        {
                                            _S29 = int(5);
                                            break;
                                        }
                                        float pj_18 = (*(&kernelContext_0)->ss_0)[_S14 + j_2];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S14 + j_2], _S110, lo_0[1U]);
                                        hi_0[1U] = fma(pj_18, _S111, hi_0[1U]);
                                        _S29 = int(7);
                                        break;
                                    }
                                    if(_S29 != int(7))
                                    {
                                        _S28 = _S29;
                                        break;
                                    }
                                    _S28 = int(6);
                                    break;
                                }
                                if(_S28 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S7)
                                        {
                                            _S27 = int(5);
                                            break;
                                        }
                                        float pj_19 = (*(&kernelContext_0)->ss_0)[2U * _S14 + j_2];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S14 + j_2], _S110, lo_0[2U]);
                                        hi_0[2U] = fma(pj_19, _S111, hi_0[2U]);
                                        _S27 = int(7);
                                        break;
                                    }
                                    if(_S27 != int(7))
                                    {
                                        _S26 = _S27;
                                        break;
                                    }
                                    _S26 = int(6);
                                    break;
                                }
                                if(_S26 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S7)
                                        {
                                            _S25 = int(5);
                                            break;
                                        }
                                        float pj_20 = (*(&kernelContext_0)->ss_0)[3U * _S14 + j_2];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S14 + j_2], _S110, lo_0[3U]);
                                        hi_0[3U] = fma(pj_20, _S111, hi_0[3U]);
                                        _S25 = int(7);
                                        break;
                                    }
                                    if(_S25 != int(7))
                                    {
                                        _S24 = _S25;
                                        break;
                                    }
                                    _S24 = int(6);
                                    break;
                                }
                                if(_S24 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S7)
                                        {
                                            _S23 = int(5);
                                            break;
                                        }
                                        float pj_21 = (*(&kernelContext_0)->ss_0)[4U * _S14 + j_2];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S14 + j_2], _S110, lo_0[4U]);
                                        hi_0[4U] = fma(pj_21, _S111, hi_0[4U]);
                                        _S23 = int(7);
                                        break;
                                    }
                                    if(_S23 != int(7))
                                    {
                                        _S22 = _S23;
                                        break;
                                    }
                                    _S22 = int(6);
                                    break;
                                }
                                if(_S22 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S7)
                                        {
                                            _S21 = int(5);
                                            break;
                                        }
                                        float pj_22 = (*(&kernelContext_0)->ss_0)[5U * _S14 + j_2];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S14 + j_2], _S110, lo_0[5U]);
                                        hi_0[5U] = fma(pj_22, _S111, hi_0[5U]);
                                        _S21 = int(7);
                                        break;
                                    }
                                    if(_S21 != int(7))
                                    {
                                        _S20 = _S21;
                                        break;
                                    }
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S7)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_23 = (*(&kernelContext_0)->ss_0)[6U * _S14 + j_2];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S14 + j_2], _S110, lo_0[6U]);
                                        hi_0[6U] = fma(pj_23, _S111, hi_0[6U]);
                                        _S105 = int(7);
                                        break;
                                    }
                                    if(_S105 != int(7))
                                    {
                                        _S104 = _S105;
                                        break;
                                    }
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S7)
                                        {
                                            _S109 = int(5);
                                            break;
                                        }
                                        float pj_24 = (*(&kernelContext_0)->ss_0)[7U * _S14 + j_2];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S14 + j_2], _S110, lo_0[7U]);
                                        hi_0[7U] = fma(pj_24, _S111, hi_0[7U]);
                                        _S109 = int(7);
                                        break;
                                    }
                                    if(_S109 != int(7))
                                    {
                                        _S108 = _S109;
                                        break;
                                    }
                                    _S108 = int(6);
                                    break;
                                }
                                if(_S108 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S31 = int(4);
                            break;
                        }
                        if(_S31 != int(4))
                        {
                            _S30 = _S31;
                            break;
                        }
                        _S30 = int(3);
                        break;
                    }
                    if(_S30 != int(3))
                    {
                        break;
                    }
                    int _S112;
                    int _S113;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_3 >= _S36)
                            {
                                _S29 = int(2);
                                break;
                            }
                            float _S114 = attnLo_0(vw_0[3U]);
                            float _S115 = attnHi_0(vw_0[3U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S29 = int(5);
                                            break;
                                        }
                                        float pj_25 = (*(&kernelContext_0)->ss_0)[j_3];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_3], _S114, lo_0[0U]);
                                        hi_0[0U] = fma(pj_25, _S115, hi_0[0U]);
                                        _S29 = int(7);
                                        break;
                                    }
                                    if(_S29 != int(7))
                                    {
                                        _S28 = _S29;
                                        break;
                                    }
                                    _S28 = int(6);
                                    break;
                                }
                                if(_S28 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S7)
                                        {
                                            _S27 = int(5);
                                            break;
                                        }
                                        float pj_26 = (*(&kernelContext_0)->ss_0)[_S14 + j_3];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S14 + j_3], _S114, lo_0[1U]);
                                        hi_0[1U] = fma(pj_26, _S115, hi_0[1U]);
                                        _S27 = int(7);
                                        break;
                                    }
                                    if(_S27 != int(7))
                                    {
                                        _S26 = _S27;
                                        break;
                                    }
                                    _S26 = int(6);
                                    break;
                                }
                                if(_S26 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S7)
                                        {
                                            _S25 = int(5);
                                            break;
                                        }
                                        float pj_27 = (*(&kernelContext_0)->ss_0)[2U * _S14 + j_3];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S14 + j_3], _S114, lo_0[2U]);
                                        hi_0[2U] = fma(pj_27, _S115, hi_0[2U]);
                                        _S25 = int(7);
                                        break;
                                    }
                                    if(_S25 != int(7))
                                    {
                                        _S24 = _S25;
                                        break;
                                    }
                                    _S24 = int(6);
                                    break;
                                }
                                if(_S24 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S7)
                                        {
                                            _S23 = int(5);
                                            break;
                                        }
                                        float pj_28 = (*(&kernelContext_0)->ss_0)[3U * _S14 + j_3];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S14 + j_3], _S114, lo_0[3U]);
                                        hi_0[3U] = fma(pj_28, _S115, hi_0[3U]);
                                        _S23 = int(7);
                                        break;
                                    }
                                    if(_S23 != int(7))
                                    {
                                        _S22 = _S23;
                                        break;
                                    }
                                    _S22 = int(6);
                                    break;
                                }
                                if(_S22 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S7)
                                        {
                                            _S21 = int(5);
                                            break;
                                        }
                                        float pj_29 = (*(&kernelContext_0)->ss_0)[4U * _S14 + j_3];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S14 + j_3], _S114, lo_0[4U]);
                                        hi_0[4U] = fma(pj_29, _S115, hi_0[4U]);
                                        _S21 = int(7);
                                        break;
                                    }
                                    if(_S21 != int(7))
                                    {
                                        _S20 = _S21;
                                        break;
                                    }
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S7)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_30 = (*(&kernelContext_0)->ss_0)[5U * _S14 + j_3];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S14 + j_3], _S114, lo_0[5U]);
                                        hi_0[5U] = fma(pj_30, _S115, hi_0[5U]);
                                        _S105 = int(7);
                                        break;
                                    }
                                    if(_S105 != int(7))
                                    {
                                        _S104 = _S105;
                                        break;
                                    }
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S7)
                                        {
                                            _S109 = int(5);
                                            break;
                                        }
                                        float pj_31 = (*(&kernelContext_0)->ss_0)[6U * _S14 + j_3];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S14 + j_3], _S114, lo_0[6U]);
                                        hi_0[6U] = fma(pj_31, _S115, hi_0[6U]);
                                        _S109 = int(7);
                                        break;
                                    }
                                    if(_S109 != int(7))
                                    {
                                        _S108 = _S109;
                                        break;
                                    }
                                    _S108 = int(6);
                                    break;
                                }
                                if(_S108 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S7)
                                        {
                                            _S113 = int(5);
                                            break;
                                        }
                                        float pj_32 = (*(&kernelContext_0)->ss_0)[7U * _S14 + j_3];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S14 + j_3], _S114, lo_0[7U]);
                                        hi_0[7U] = fma(pj_32, _S115, hi_0[7U]);
                                        _S113 = int(7);
                                        break;
                                    }
                                    if(_S113 != int(7))
                                    {
                                        _S112 = _S113;
                                        break;
                                    }
                                    _S112 = int(6);
                                    break;
                                }
                                if(_S112 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S29 = int(4);
                            break;
                        }
                        if(_S29 != int(4))
                        {
                            _S28 = _S29;
                            break;
                        }
                        _S28 = int(3);
                        break;
                    }
                    if(_S28 != int(3))
                    {
                        break;
                    }
                    int _S116;
                    int _S117;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_4 >= _S36)
                            {
                                _S27 = int(2);
                                break;
                            }
                            float _S118 = attnLo_0(vw_0[4U]);
                            float _S119 = attnHi_0(vw_0[4U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S27 = int(5);
                                            break;
                                        }
                                        float pj_33 = (*(&kernelContext_0)->ss_0)[j_4];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_4], _S118, lo_0[0U]);
                                        hi_0[0U] = fma(pj_33, _S119, hi_0[0U]);
                                        _S27 = int(7);
                                        break;
                                    }
                                    if(_S27 != int(7))
                                    {
                                        _S26 = _S27;
                                        break;
                                    }
                                    _S26 = int(6);
                                    break;
                                }
                                if(_S26 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S7)
                                        {
                                            _S25 = int(5);
                                            break;
                                        }
                                        float pj_34 = (*(&kernelContext_0)->ss_0)[_S14 + j_4];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S14 + j_4], _S118, lo_0[1U]);
                                        hi_0[1U] = fma(pj_34, _S119, hi_0[1U]);
                                        _S25 = int(7);
                                        break;
                                    }
                                    if(_S25 != int(7))
                                    {
                                        _S24 = _S25;
                                        break;
                                    }
                                    _S24 = int(6);
                                    break;
                                }
                                if(_S24 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S7)
                                        {
                                            _S23 = int(5);
                                            break;
                                        }
                                        float pj_35 = (*(&kernelContext_0)->ss_0)[2U * _S14 + j_4];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S14 + j_4], _S118, lo_0[2U]);
                                        hi_0[2U] = fma(pj_35, _S119, hi_0[2U]);
                                        _S23 = int(7);
                                        break;
                                    }
                                    if(_S23 != int(7))
                                    {
                                        _S22 = _S23;
                                        break;
                                    }
                                    _S22 = int(6);
                                    break;
                                }
                                if(_S22 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S7)
                                        {
                                            _S21 = int(5);
                                            break;
                                        }
                                        float pj_36 = (*(&kernelContext_0)->ss_0)[3U * _S14 + j_4];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S14 + j_4], _S118, lo_0[3U]);
                                        hi_0[3U] = fma(pj_36, _S119, hi_0[3U]);
                                        _S21 = int(7);
                                        break;
                                    }
                                    if(_S21 != int(7))
                                    {
                                        _S20 = _S21;
                                        break;
                                    }
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S7)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_37 = (*(&kernelContext_0)->ss_0)[4U * _S14 + j_4];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S14 + j_4], _S118, lo_0[4U]);
                                        hi_0[4U] = fma(pj_37, _S119, hi_0[4U]);
                                        _S105 = int(7);
                                        break;
                                    }
                                    if(_S105 != int(7))
                                    {
                                        _S104 = _S105;
                                        break;
                                    }
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S7)
                                        {
                                            _S109 = int(5);
                                            break;
                                        }
                                        float pj_38 = (*(&kernelContext_0)->ss_0)[5U * _S14 + j_4];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S14 + j_4], _S118, lo_0[5U]);
                                        hi_0[5U] = fma(pj_38, _S119, hi_0[5U]);
                                        _S109 = int(7);
                                        break;
                                    }
                                    if(_S109 != int(7))
                                    {
                                        _S108 = _S109;
                                        break;
                                    }
                                    _S108 = int(6);
                                    break;
                                }
                                if(_S108 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S7)
                                        {
                                            _S113 = int(5);
                                            break;
                                        }
                                        float pj_39 = (*(&kernelContext_0)->ss_0)[6U * _S14 + j_4];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S14 + j_4], _S118, lo_0[6U]);
                                        hi_0[6U] = fma(pj_39, _S119, hi_0[6U]);
                                        _S113 = int(7);
                                        break;
                                    }
                                    if(_S113 != int(7))
                                    {
                                        _S112 = _S113;
                                        break;
                                    }
                                    _S112 = int(6);
                                    break;
                                }
                                if(_S112 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S7)
                                        {
                                            _S117 = int(5);
                                            break;
                                        }
                                        float pj_40 = (*(&kernelContext_0)->ss_0)[7U * _S14 + j_4];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S14 + j_4], _S118, lo_0[7U]);
                                        hi_0[7U] = fma(pj_40, _S119, hi_0[7U]);
                                        _S117 = int(7);
                                        break;
                                    }
                                    if(_S117 != int(7))
                                    {
                                        _S116 = _S117;
                                        break;
                                    }
                                    _S116 = int(6);
                                    break;
                                }
                                if(_S116 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S27 = int(4);
                            break;
                        }
                        if(_S27 != int(4))
                        {
                            _S26 = _S27;
                            break;
                        }
                        _S26 = int(3);
                        break;
                    }
                    if(_S26 != int(3))
                    {
                        break;
                    }
                    int _S120;
                    int _S121;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_5 >= _S36)
                            {
                                _S25 = int(2);
                                break;
                            }
                            float _S122 = attnLo_0(vw_0[5U]);
                            float _S123 = attnHi_0(vw_0[5U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S25 = int(5);
                                            break;
                                        }
                                        float pj_41 = (*(&kernelContext_0)->ss_0)[j_5];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_5], _S122, lo_0[0U]);
                                        hi_0[0U] = fma(pj_41, _S123, hi_0[0U]);
                                        _S25 = int(7);
                                        break;
                                    }
                                    if(_S25 != int(7))
                                    {
                                        _S24 = _S25;
                                        break;
                                    }
                                    _S24 = int(6);
                                    break;
                                }
                                if(_S24 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S7)
                                        {
                                            _S23 = int(5);
                                            break;
                                        }
                                        float pj_42 = (*(&kernelContext_0)->ss_0)[_S14 + j_5];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S14 + j_5], _S122, lo_0[1U]);
                                        hi_0[1U] = fma(pj_42, _S123, hi_0[1U]);
                                        _S23 = int(7);
                                        break;
                                    }
                                    if(_S23 != int(7))
                                    {
                                        _S22 = _S23;
                                        break;
                                    }
                                    _S22 = int(6);
                                    break;
                                }
                                if(_S22 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S7)
                                        {
                                            _S21 = int(5);
                                            break;
                                        }
                                        float pj_43 = (*(&kernelContext_0)->ss_0)[2U * _S14 + j_5];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S14 + j_5], _S122, lo_0[2U]);
                                        hi_0[2U] = fma(pj_43, _S123, hi_0[2U]);
                                        _S21 = int(7);
                                        break;
                                    }
                                    if(_S21 != int(7))
                                    {
                                        _S20 = _S21;
                                        break;
                                    }
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S7)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_44 = (*(&kernelContext_0)->ss_0)[3U * _S14 + j_5];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S14 + j_5], _S122, lo_0[3U]);
                                        hi_0[3U] = fma(pj_44, _S123, hi_0[3U]);
                                        _S105 = int(7);
                                        break;
                                    }
                                    if(_S105 != int(7))
                                    {
                                        _S104 = _S105;
                                        break;
                                    }
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S7)
                                        {
                                            _S109 = int(5);
                                            break;
                                        }
                                        float pj_45 = (*(&kernelContext_0)->ss_0)[4U * _S14 + j_5];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S14 + j_5], _S122, lo_0[4U]);
                                        hi_0[4U] = fma(pj_45, _S123, hi_0[4U]);
                                        _S109 = int(7);
                                        break;
                                    }
                                    if(_S109 != int(7))
                                    {
                                        _S108 = _S109;
                                        break;
                                    }
                                    _S108 = int(6);
                                    break;
                                }
                                if(_S108 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S7)
                                        {
                                            _S113 = int(5);
                                            break;
                                        }
                                        float pj_46 = (*(&kernelContext_0)->ss_0)[5U * _S14 + j_5];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S14 + j_5], _S122, lo_0[5U]);
                                        hi_0[5U] = fma(pj_46, _S123, hi_0[5U]);
                                        _S113 = int(7);
                                        break;
                                    }
                                    if(_S113 != int(7))
                                    {
                                        _S112 = _S113;
                                        break;
                                    }
                                    _S112 = int(6);
                                    break;
                                }
                                if(_S112 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S7)
                                        {
                                            _S117 = int(5);
                                            break;
                                        }
                                        float pj_47 = (*(&kernelContext_0)->ss_0)[6U * _S14 + j_5];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S14 + j_5], _S122, lo_0[6U]);
                                        hi_0[6U] = fma(pj_47, _S123, hi_0[6U]);
                                        _S117 = int(7);
                                        break;
                                    }
                                    if(_S117 != int(7))
                                    {
                                        _S116 = _S117;
                                        break;
                                    }
                                    _S116 = int(6);
                                    break;
                                }
                                if(_S116 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S7)
                                        {
                                            _S121 = int(5);
                                            break;
                                        }
                                        float pj_48 = (*(&kernelContext_0)->ss_0)[7U * _S14 + j_5];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S14 + j_5], _S122, lo_0[7U]);
                                        hi_0[7U] = fma(pj_48, _S123, hi_0[7U]);
                                        _S121 = int(7);
                                        break;
                                    }
                                    if(_S121 != int(7))
                                    {
                                        _S120 = _S121;
                                        break;
                                    }
                                    _S120 = int(6);
                                    break;
                                }
                                if(_S120 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S25 = int(4);
                            break;
                        }
                        if(_S25 != int(4))
                        {
                            _S24 = _S25;
                            break;
                        }
                        _S24 = int(3);
                        break;
                    }
                    if(_S24 != int(3))
                    {
                        break;
                    }
                    int _S124;
                    int _S125;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_6 >= _S36)
                            {
                                _S23 = int(2);
                                break;
                            }
                            float _S126 = attnLo_0(vw_0[6U]);
                            float _S127 = attnHi_0(vw_0[6U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S23 = int(5);
                                            break;
                                        }
                                        float pj_49 = (*(&kernelContext_0)->ss_0)[j_6];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_6], _S126, lo_0[0U]);
                                        hi_0[0U] = fma(pj_49, _S127, hi_0[0U]);
                                        _S23 = int(7);
                                        break;
                                    }
                                    if(_S23 != int(7))
                                    {
                                        _S22 = _S23;
                                        break;
                                    }
                                    _S22 = int(6);
                                    break;
                                }
                                if(_S22 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S7)
                                        {
                                            _S21 = int(5);
                                            break;
                                        }
                                        float pj_50 = (*(&kernelContext_0)->ss_0)[_S14 + j_6];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S14 + j_6], _S126, lo_0[1U]);
                                        hi_0[1U] = fma(pj_50, _S127, hi_0[1U]);
                                        _S21 = int(7);
                                        break;
                                    }
                                    if(_S21 != int(7))
                                    {
                                        _S20 = _S21;
                                        break;
                                    }
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S7)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_51 = (*(&kernelContext_0)->ss_0)[2U * _S14 + j_6];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S14 + j_6], _S126, lo_0[2U]);
                                        hi_0[2U] = fma(pj_51, _S127, hi_0[2U]);
                                        _S105 = int(7);
                                        break;
                                    }
                                    if(_S105 != int(7))
                                    {
                                        _S104 = _S105;
                                        break;
                                    }
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S7)
                                        {
                                            _S109 = int(5);
                                            break;
                                        }
                                        float pj_52 = (*(&kernelContext_0)->ss_0)[3U * _S14 + j_6];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S14 + j_6], _S126, lo_0[3U]);
                                        hi_0[3U] = fma(pj_52, _S127, hi_0[3U]);
                                        _S109 = int(7);
                                        break;
                                    }
                                    if(_S109 != int(7))
                                    {
                                        _S108 = _S109;
                                        break;
                                    }
                                    _S108 = int(6);
                                    break;
                                }
                                if(_S108 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S7)
                                        {
                                            _S113 = int(5);
                                            break;
                                        }
                                        float pj_53 = (*(&kernelContext_0)->ss_0)[4U * _S14 + j_6];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S14 + j_6], _S126, lo_0[4U]);
                                        hi_0[4U] = fma(pj_53, _S127, hi_0[4U]);
                                        _S113 = int(7);
                                        break;
                                    }
                                    if(_S113 != int(7))
                                    {
                                        _S112 = _S113;
                                        break;
                                    }
                                    _S112 = int(6);
                                    break;
                                }
                                if(_S112 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S7)
                                        {
                                            _S117 = int(5);
                                            break;
                                        }
                                        float pj_54 = (*(&kernelContext_0)->ss_0)[5U * _S14 + j_6];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S14 + j_6], _S126, lo_0[5U]);
                                        hi_0[5U] = fma(pj_54, _S127, hi_0[5U]);
                                        _S117 = int(7);
                                        break;
                                    }
                                    if(_S117 != int(7))
                                    {
                                        _S116 = _S117;
                                        break;
                                    }
                                    _S116 = int(6);
                                    break;
                                }
                                if(_S116 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S7)
                                        {
                                            _S121 = int(5);
                                            break;
                                        }
                                        float pj_55 = (*(&kernelContext_0)->ss_0)[6U * _S14 + j_6];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S14 + j_6], _S126, lo_0[6U]);
                                        hi_0[6U] = fma(pj_55, _S127, hi_0[6U]);
                                        _S121 = int(7);
                                        break;
                                    }
                                    if(_S121 != int(7))
                                    {
                                        _S120 = _S121;
                                        break;
                                    }
                                    _S120 = int(6);
                                    break;
                                }
                                if(_S120 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S7)
                                        {
                                            _S125 = int(5);
                                            break;
                                        }
                                        float pj_56 = (*(&kernelContext_0)->ss_0)[7U * _S14 + j_6];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S14 + j_6], _S126, lo_0[7U]);
                                        hi_0[7U] = fma(pj_56, _S127, hi_0[7U]);
                                        _S125 = int(7);
                                        break;
                                    }
                                    if(_S125 != int(7))
                                    {
                                        _S124 = _S125;
                                        break;
                                    }
                                    _S124 = int(6);
                                    break;
                                }
                                if(_S124 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S23 = int(4);
                            break;
                        }
                        if(_S23 != int(4))
                        {
                            _S22 = _S23;
                            break;
                        }
                        _S22 = int(3);
                        break;
                    }
                    if(_S22 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_7 >= _S36)
                            {
                                _S21 = int(2);
                                break;
                            }
                            float _S128 = attnLo_0(vw_0[7U]);
                            float _S129 = attnHi_0(vw_0[7U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S21 = int(5);
                                            break;
                                        }
                                        float pj_57 = (*(&kernelContext_0)->ss_0)[j_7];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_7], _S128, lo_0[0U]);
                                        hi_0[0U] = fma(pj_57, _S129, hi_0[0U]);
                                        _S21 = int(7);
                                        break;
                                    }
                                    if(_S21 != int(7))
                                    {
                                        _S20 = _S21;
                                        break;
                                    }
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S7)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_58 = (*(&kernelContext_0)->ss_0)[_S14 + j_7];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S14 + j_7], _S128, lo_0[1U]);
                                        hi_0[1U] = fma(pj_58, _S129, hi_0[1U]);
                                        _S105 = int(7);
                                        break;
                                    }
                                    if(_S105 != int(7))
                                    {
                                        _S104 = _S105;
                                        break;
                                    }
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S7)
                                        {
                                            _S109 = int(5);
                                            break;
                                        }
                                        float pj_59 = (*(&kernelContext_0)->ss_0)[2U * _S14 + j_7];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S14 + j_7], _S128, lo_0[2U]);
                                        hi_0[2U] = fma(pj_59, _S129, hi_0[2U]);
                                        _S109 = int(7);
                                        break;
                                    }
                                    if(_S109 != int(7))
                                    {
                                        _S108 = _S109;
                                        break;
                                    }
                                    _S108 = int(6);
                                    break;
                                }
                                if(_S108 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S7)
                                        {
                                            _S113 = int(5);
                                            break;
                                        }
                                        float pj_60 = (*(&kernelContext_0)->ss_0)[3U * _S14 + j_7];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S14 + j_7], _S128, lo_0[3U]);
                                        hi_0[3U] = fma(pj_60, _S129, hi_0[3U]);
                                        _S113 = int(7);
                                        break;
                                    }
                                    if(_S113 != int(7))
                                    {
                                        _S112 = _S113;
                                        break;
                                    }
                                    _S112 = int(6);
                                    break;
                                }
                                if(_S112 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S7)
                                        {
                                            _S117 = int(5);
                                            break;
                                        }
                                        float pj_61 = (*(&kernelContext_0)->ss_0)[4U * _S14 + j_7];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S14 + j_7], _S128, lo_0[4U]);
                                        hi_0[4U] = fma(pj_61, _S129, hi_0[4U]);
                                        _S117 = int(7);
                                        break;
                                    }
                                    if(_S117 != int(7))
                                    {
                                        _S116 = _S117;
                                        break;
                                    }
                                    _S116 = int(6);
                                    break;
                                }
                                if(_S116 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S7)
                                        {
                                            _S121 = int(5);
                                            break;
                                        }
                                        float pj_62 = (*(&kernelContext_0)->ss_0)[5U * _S14 + j_7];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S14 + j_7], _S128, lo_0[5U]);
                                        hi_0[5U] = fma(pj_62, _S129, hi_0[5U]);
                                        _S121 = int(7);
                                        break;
                                    }
                                    if(_S121 != int(7))
                                    {
                                        _S120 = _S121;
                                        break;
                                    }
                                    _S120 = int(6);
                                    break;
                                }
                                if(_S120 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S7)
                                        {
                                            _S125 = int(5);
                                            break;
                                        }
                                        float pj_63 = (*(&kernelContext_0)->ss_0)[6U * _S14 + j_7];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S14 + j_7], _S128, lo_0[6U]);
                                        hi_0[6U] = fma(pj_63, _S129, hi_0[6U]);
                                        _S125 = int(7);
                                        break;
                                    }
                                    if(_S125 != int(7))
                                    {
                                        _S124 = _S125;
                                        break;
                                    }
                                    _S124 = int(6);
                                    break;
                                }
                                if(_S124 != int(6))
                                {
                                    break;
                                }
                                int _S130;
                                for(;;)
                                {
                                    int _S131;
                                    for(;;)
                                    {
                                        if(7U >= _S7)
                                        {
                                            _S131 = int(5);
                                            break;
                                        }
                                        float pj_64 = (*(&kernelContext_0)->ss_0)[7U * _S14 + j_7];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S14 + j_7], _S128, lo_0[7U]);
                                        hi_0[7U] = fma(pj_64, _S129, hi_0[7U]);
                                        _S131 = int(7);
                                        break;
                                    }
                                    if(_S131 != int(7))
                                    {
                                        _S130 = _S131;
                                        break;
                                    }
                                    _S130 = int(6);
                                    break;
                                }
                                if(_S130 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S21 = int(4);
                            break;
                        }
                        if(_S21 != int(4))
                        {
                            _S20 = _S21;
                            break;
                        }
                        _S20 = int(3);
                        break;
                    }
                    if(_S20 != int(3))
                    {
                        break;
                    }
                    break;
                }
                kw_7 = kw_7 + 16U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        w_2 = w_2 + _S14;
    }
    r_0 = 0U;
    for(;;)
    {
        if(r_0 < _S7)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(0U == r_0)
        {
            acc_1 = lo_0[0U];
            acc_0 = hi_0[0U];
        }
        else
        {
            acc_1 = 0.0;
            acc_0 = 0.0;
        }
        float b_3;
        if(1U == r_0)
        {
            pj_0 = lo_0[1U];
            b_3 = hi_0[1U];
        }
        else
        {
            pj_0 = acc_1;
            b_3 = acc_0;
        }
        float a_11;
        float b_4;
        if(2U == r_0)
        {
            a_11 = lo_0[2U];
            b_4 = hi_0[2U];
        }
        else
        {
            a_11 = pj_0;
            b_4 = b_3;
        }
        float a_12;
        float b_5;
        if(3U == r_0)
        {
            a_12 = lo_0[3U];
            b_5 = hi_0[3U];
        }
        else
        {
            a_12 = a_11;
            b_5 = b_4;
        }
        float a_13;
        float b_6;
        if(4U == r_0)
        {
            a_13 = lo_0[4U];
            b_6 = hi_0[4U];
        }
        else
        {
            a_13 = a_12;
            b_6 = b_5;
        }
        float a_14;
        float b_7;
        if(5U == r_0)
        {
            a_14 = lo_0[5U];
            b_7 = hi_0[5U];
        }
        else
        {
            a_14 = a_13;
            b_7 = b_6;
        }
        float a_15;
        float b_8;
        if(6U == r_0)
        {
            a_15 = lo_0[6U];
            b_8 = hi_0[6U];
        }
        else
        {
            a_15 = a_14;
            b_8 = b_7;
        }
        float a_16;
        float b_9;
        if(7U == r_0)
        {
            a_16 = lo_0[7U];
            b_9 = hi_0[7U];
        }
        else
        {
            a_16 = a_15;
            b_9 = b_8;
        }
        uint _S132 = 2U * _S18;
        uint _S133 = _S19 * 128U + _S132;
        (*(&kernelContext_0)->ss_0)[_S133] = a_16;
        (*(&kernelContext_0)->ss_0)[_S133 + 1U] = b_9;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S19 != 0U)
        {
            visible_0 = true;
        }
        else
        {
            visible_0 = _S18 >= _S10;
        }
        if(visible_0)
        {
            r_0 = r_0 + 1U;
            continue;
        }
        uint _S134 = 128U + _S132;
        float a_17 = (*(&kernelContext_0)->ss_0)[_S132] + (*(&kernelContext_0)->ss_0)[_S134];
        float b_10 = (*(&kernelContext_0)->ss_0)[_S132 + 1U] + (*(&kernelContext_0)->ss_0)[_S134 + 1U];
        uint _S135 = _S6 * hpk_0;
        uint _S136 = r_0 / (&kernelContext_0)->p_0->queries_0;
        uint h_1 = _S135 + _S136;
        uint i_9 = r_0 % (&kernelContext_0)->p_0->queries_0;
        uint row_0 = _S8 + h_1 * (&kernelContext_0)->p_0->queries_0 + i_9;
        if(((&kernelContext_0)->p_0->splits_0) > 1U)
        {
            uint pr_8 = split_0 * _S9 + row_0;
            if(_S18 == 0U)
            {
                uint _S137 = 2U * pr_8;
                float device* _S138 = (&kernelContext_0)->partialML_0+_S137;
                bool _S139 = keys_1 == 0U;
                float _S140;
                if(_S139)
                {
                    _S140 = dg_negative_infinity_0();
                }
                else
                {
                    _S140 = (*(&kernelContext_0)->rowM_0)[r_0];
                }
                *_S138 = _S140;
                float device* _S141 = (&kernelContext_0)->partialML_0+(_S137 + 1U);
                float _S142;
                if(_S139)
                {
                    _S142 = 0.0;
                }
                else
                {
                    _S142 = (*(&kernelContext_0)->rowL_0)[r_0];
                }
                *_S141 = _S142;
            }
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S132)) = a_17;
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S132 + 1U)) = b_10;
            r_0 = r_0 + 1U;
            continue;
        }
        float l_0 = (*(&kernelContext_0)->rowL_0)[r_0];
        *((&kernelContext_0)->o_0+(row_0 * (&kernelContext_0)->p_0->headDim_0 + _S132)) = a_17 / (*(&kernelContext_0)->rowL_0)[r_0];
        *((&kernelContext_0)->o_0+(row_0 * (&kernelContext_0)->p_0->headDim_0 + _S132 + 1U)) = b_10 / l_0;
        r_0 = r_0 + 1U;
    }
    return;
}

