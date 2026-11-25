#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant uint fc_ROW_TIER_0 [[function_constant(0)]];
constant uint ROW_TIER_0 = is_function_constant_defined(fc_ROW_TIER_0) ? fc_ROW_TIER_0 : 8U;
float attnLo_0(uint w_0)
{
    return (as_type<half>((ushort)((w_0 & 65535U))));
}

float attnHi_0(uint w_1)
{
    return (as_type<half>((ushort)((w_1 >> 16U))));
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
};

struct KernelContext_0
{
    AttnParams_0 constant* p_0;
    uint device* q_0;
    uint device* table_0;
    uint device* k_0;
    uint device* v_0;
    float device* partialML_0;
    float device* partialAcc_0;
    float device* o_0;
    array<uint, int(1024)> threadgroup* qs_0;
    array<float, int(1024)> threadgroup* ss_0;
    array<float, int(128)> threadgroup* red_0;
    array<float, int(8)> threadgroup* rowM_0;
    array<float, int(8)> threadgroup* rowL_0;
};

[[kernel]] void attn_decode_256_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(7)]], uint device* q_1 [[buffer(0)]], uint device* table_1 [[buffer(6)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partialML_1 [[buffer(4)]], float device* partialAcc_1 [[buffer(5)]], float device* o_1 [[buffer(3)]])
{
    bool _S5;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->q_0 = q_1;
    (&kernelContext_0)->table_0 = table_1;
    (&kernelContext_0)->k_0 = k_1;
    (&kernelContext_0)->v_0 = v_1;
    (&kernelContext_0)->partialML_0 = partialML_1;
    (&kernelContext_0)->partialAcc_0 = partialAcc_1;
    (&kernelContext_0)->o_0 = o_1;
    threadgroup array<uint, int(1024)> qs_1;
    (&kernelContext_0)->qs_0 = &qs_1;
    threadgroup array<float, int(1024)> ss_1;
    (&kernelContext_0)->ss_0 = &ss_1;
    threadgroup array<float, int(128)> red_1;
    (&kernelContext_0)->red_0 = &red_1;
    threadgroup array<float, int(8)> rowM_1;
    (&kernelContext_0)->rowM_0 = &rowM_1;
    threadgroup array<float, int(8)> rowL_1;
    (&kernelContext_0)->rowL_0 = &rowL_1;
    uint t_0 = local_0.x;
    uint _S6 = group_0.y;
    uint split_0 = group_0.z;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint _S7 = min(hpk_0 * (&kernelContext_0)->p_0->queries_0, ROW_TIER_0);
    uint d2_0 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint _S8 = (&kernelContext_0)->p_0->keys_0 - (&kernelContext_0)->p_0->queries_0;
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S9 = min((&kernelContext_0)->p_0->keys_0, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    uint _S10;
    if(_S9 > kStart_0)
    {
        _S10 = _S9 - kStart_0;
    }
    else
    {
        _S10 = 0U;
    }
    bool visible_0;
    uint w_2;
    uint _S11 = (&kernelContext_0)->p_0->splitKeys_0;
    uint e_0 = t_0;
    for(;;)
    {
        if(e_0 < (ROW_TIER_0 * 128U))
        {
        }
        else
        {
            break;
        }
        uint r_0 = e_0 / 128U;
        uint pr_0 = e_0 % 128U;
        if(r_0 < _S7)
        {
            visible_0 = pr_0 < d2_0;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S12 = _S6 * hpk_0;
            uint _S13 = r_0 / (&kernelContext_0)->p_0->queries_0;
            uint h_0 = _S12 + _S13;
            uint i_0 = r_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((h_0 * (&kernelContext_0)->p_0->queries_0 + i_0) * d2_0 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_0] = w_2;
        e_0 = e_0 + 128U;
    }
    float acc_0;
    int _S14;
    int _S15;
    int _S16;
    int _S17;
    int _S18;
    int _S19;
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
    uint kw_0;
    uint kw_1;
    uint kw_2;
    uint kw_3;
    uint kw_4;
    uint kw_5;
    uint p0_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    w_2 = t_0;
    for(;;)
    {
        if(w_2 < _S10)
        {
        }
        else
        {
            break;
        }
        uint key_0 = kStart_0 + w_2;
        if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
        {
            e_0 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * d2_0;
        }
        else
        {
            uint _S30 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
            uint _S31 = (*((&kernelContext_0)->table_0+_S30) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
            uint _S32 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
            e_0 = (_S31 + _S32) * d2_0;
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
            if(p0_0 < 128U)
            {
            }
            else
            {
                break;
            }
            if(p0_0 < d2_0)
            {
                kw_5 = *((&kernelContext_0)->k_0+(e_0 + p0_0));
            }
            else
            {
                kw_5 = 0U;
            }
            float _S33 = attnLo_0(kw_5);
            float _S34 = attnHi_0(kw_5);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[p0_0]), _S34, fma(attnLo_0((*(&kernelContext_0)->qs_0)[p0_0]), _S33, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S34, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S33, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + p0_0]), _S34, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + p0_0]), _S33, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + p0_0]), _S34, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + p0_0]), _S33, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + p0_0]), _S34, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + p0_0]), _S33, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + p0_0]), _S34, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + p0_0]), _S33, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + p0_0]), _S34, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + p0_0]), _S33, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + p0_0]), _S34, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + p0_0]), _S33, s_0[7U]));
            }
            uint pr_1 = p0_0 + 1U;
            if(pr_1 < d2_0)
            {
                kw_4 = *((&kernelContext_0)->k_0+(e_0 + pr_1));
            }
            else
            {
                kw_4 = 0U;
            }
            float _S35 = attnLo_0(kw_4);
            float _S36 = attnHi_0(kw_4);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_1]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_1]), _S35, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S35, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_1]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_1]), _S35, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_1]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_1]), _S35, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_1]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_1]), _S35, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_1]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_1]), _S35, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_1]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_1]), _S35, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_1]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_1]), _S35, s_0[7U]));
            }
            uint pr_2 = p0_0 + 2U;
            if(pr_2 < d2_0)
            {
                kw_3 = *((&kernelContext_0)->k_0+(e_0 + pr_2));
            }
            else
            {
                kw_3 = 0U;
            }
            float _S37 = attnLo_0(kw_3);
            float _S38 = attnHi_0(kw_3);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_2]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_2]), _S37, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S37, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_2]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_2]), _S37, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_2]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_2]), _S37, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_2]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_2]), _S37, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_2]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_2]), _S37, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_2]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_2]), _S37, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_2]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_2]), _S37, s_0[7U]));
            }
            uint pr_3 = p0_0 + 3U;
            if(pr_3 < d2_0)
            {
                kw_2 = *((&kernelContext_0)->k_0+(e_0 + pr_3));
            }
            else
            {
                kw_2 = 0U;
            }
            float _S39 = attnLo_0(kw_2);
            float _S40 = attnHi_0(kw_2);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_3]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_3]), _S39, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S39, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_3]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_3]), _S39, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_3]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_3]), _S39, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_3]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_3]), _S39, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_3]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_3]), _S39, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_3]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_3]), _S39, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_3]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_3]), _S39, s_0[7U]));
            }
            uint pr_4 = p0_0 + 4U;
            if(pr_4 < d2_0)
            {
                kw_1 = *((&kernelContext_0)->k_0+(e_0 + pr_4));
            }
            else
            {
                kw_1 = 0U;
            }
            float _S41 = attnLo_0(kw_1);
            float _S42 = attnHi_0(kw_1);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_4]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_4]), _S41, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S41, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_4]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_4]), _S41, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_4]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_4]), _S41, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_4]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_4]), _S41, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_4]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_4]), _S41, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_4]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_4]), _S41, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_4]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_4]), _S41, s_0[7U]));
            }
            uint pr_5 = p0_0 + 5U;
            if(pr_5 < d2_0)
            {
                kw_0 = *((&kernelContext_0)->k_0+(e_0 + pr_5));
            }
            else
            {
                kw_0 = 0U;
            }
            float _S43 = attnLo_0(kw_0);
            float _S44 = attnHi_0(kw_0);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_5]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_5]), _S43, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S43, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_5]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_5]), _S43, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_5]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_5]), _S43, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_5]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_5]), _S43, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_5]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_5]), _S43, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_5]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_5]), _S43, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_5]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_5]), _S43, s_0[7U]));
            }
            uint pr_6 = p0_0 + 6U;
            uint kw_6;
            if(pr_6 < d2_0)
            {
                kw_6 = *((&kernelContext_0)->k_0+(e_0 + pr_6));
            }
            else
            {
                kw_6 = 0U;
            }
            float _S45 = attnLo_0(kw_6);
            float _S46 = attnHi_0(kw_6);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_6]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_6]), _S45, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S45, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_6]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_6]), _S45, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_6]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_6]), _S45, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_6]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_6]), _S45, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_6]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_6]), _S45, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_6]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_6]), _S45, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_6]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_6]), _S45, s_0[7U]));
            }
            uint pr_7 = p0_0 + 7U;
            uint kw_7;
            if(pr_7 < d2_0)
            {
                kw_7 = *((&kernelContext_0)->k_0+(e_0 + pr_7));
            }
            else
            {
                kw_7 = 0U;
            }
            float _S47 = attnLo_0(kw_7);
            float _S48 = attnHi_0(kw_7);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_7]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_7]), _S47, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S47, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_7]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_7]), _S47, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_7]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_7]), _S47, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_7]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_7]), _S47, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_7]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_7]), _S47, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_7]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_7]), _S47, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_7]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_7]), _S47, s_0[7U]));
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
                        _S29 = int(1);
                        break;
                    }
                    uint i_1 = 0U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_1 + _S8);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[0U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[w_2] = acc_0;
                    _S29 = int(3);
                    break;
                }
                if(_S29 != int(3))
                {
                    _S28 = _S29;
                    break;
                }
                _S28 = int(2);
                break;
            }
            if(_S28 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(1U >= _S7)
                    {
                        _S27 = int(1);
                        break;
                    }
                    uint i_2 = 1U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_2 + _S8);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[1U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[_S11 + w_2] = acc_0;
                    _S27 = int(3);
                    break;
                }
                if(_S27 != int(3))
                {
                    _S26 = _S27;
                    break;
                }
                _S26 = int(2);
                break;
            }
            if(_S26 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(2U >= _S7)
                    {
                        _S25 = int(1);
                        break;
                    }
                    uint i_3 = 2U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_3 + _S8);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[2U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[2U * _S11 + w_2] = acc_0;
                    _S25 = int(3);
                    break;
                }
                if(_S25 != int(3))
                {
                    _S24 = _S25;
                    break;
                }
                _S24 = int(2);
                break;
            }
            if(_S24 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(3U >= _S7)
                    {
                        _S23 = int(1);
                        break;
                    }
                    uint i_4 = 3U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_4 + _S8);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[3U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[3U * _S11 + w_2] = acc_0;
                    _S23 = int(3);
                    break;
                }
                if(_S23 != int(3))
                {
                    _S22 = _S23;
                    break;
                }
                _S22 = int(2);
                break;
            }
            if(_S22 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(4U >= _S7)
                    {
                        _S21 = int(1);
                        break;
                    }
                    uint i_5 = 4U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_5 + _S8);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[4U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[4U * _S11 + w_2] = acc_0;
                    _S21 = int(3);
                    break;
                }
                if(_S21 != int(3))
                {
                    _S20 = _S21;
                    break;
                }
                _S20 = int(2);
                break;
            }
            if(_S20 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(5U >= _S7)
                    {
                        _S19 = int(1);
                        break;
                    }
                    uint i_6 = 5U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_6 + _S8);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[5U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[5U * _S11 + w_2] = acc_0;
                    _S19 = int(3);
                    break;
                }
                if(_S19 != int(3))
                {
                    _S18 = _S19;
                    break;
                }
                _S18 = int(2);
                break;
            }
            if(_S18 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(6U >= _S7)
                    {
                        _S17 = int(1);
                        break;
                    }
                    uint i_7 = 6U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_7 + _S8);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[6U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[6U * _S11 + w_2] = acc_0;
                    _S17 = int(3);
                    break;
                }
                if(_S17 != int(3))
                {
                    _S16 = _S17;
                    break;
                }
                _S16 = int(2);
                break;
            }
            if(_S16 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(7U >= _S7)
                    {
                        _S15 = int(1);
                        break;
                    }
                    uint i_8 = 7U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_8 + _S8);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[7U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[7U * _S11 + w_2] = acc_0;
                    _S15 = int(3);
                    break;
                }
                if(_S15 != int(3))
                {
                    _S14 = _S15;
                    break;
                }
                _S14 = int(2);
                break;
            }
            if(_S14 != int(2))
            {
                break;
            }
            break;
        }
        w_2 = w_2 + 128U;
    }
    float pj_0;
    float acc_1;
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
        float _S49 = dg_negative_infinity_0();
        w_2 = t_0;
        acc_0 = _S49;
        for(;;)
        {
            if(w_2 < _S10)
            {
            }
            else
            {
                break;
            }
            float _S50 = max(acc_0, (*(&kernelContext_0)->ss_0)[p0_0 * _S11 + w_2]);
            w_2 = w_2 + 128U;
            acc_0 = _S50;
        }
        (*(&kernelContext_0)->red_0)[t_0] = acc_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S51 = t_0 < 64U;
        if(_S51)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S52 = t_0 < 32U;
        if(_S52)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S53 = t_0 < 16U;
        if(_S53)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S54 = t_0 < 8U;
        if(_S54)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S55 = t_0 < 4U;
        if(_S55)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S56 = t_0 < 2U;
        if(_S56)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S57 = t_0 < 1U;
        if(_S57)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S58 = t_0 == 0U;
        if(_S58)
        {
            (*(&kernelContext_0)->rowM_0)[p0_0] = (*(&kernelContext_0)->red_0)[int(0)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S59 = (*(&kernelContext_0)->rowM_0)[p0_0];
        kw_5 = t_0;
        acc_1 = 0.0;
        for(;;)
        {
            if(kw_5 < _S10)
            {
            }
            else
            {
                break;
            }
            float sv_0 = (*(&kernelContext_0)->ss_0)[p0_0 * _S11 + kw_5];
            if(((*(&kernelContext_0)->ss_0)[p0_0 * _S11 + kw_5]) == _S49)
            {
                pj_0 = 0.0;
            }
            else
            {
                pj_0 = exp(sv_0 - _S59);
            }
            (*(&kernelContext_0)->ss_0)[p0_0 * _S11 + kw_5] = pj_0;
            float _S60 = acc_1 + pj_0;
            kw_5 = kw_5 + 128U;
            acc_1 = _S60;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_0)->red_0)[t_0] = acc_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S51)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S52)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S53)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S54)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S55)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S56)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S57)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S58)
        {
            (*(&kernelContext_0)->rowL_0)[p0_0] = (*(&kernelContext_0)->red_0)[int(0)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        p0_0 = p0_0 + 1U;
    }
    uint pair_0 = t_0 % 128U;
    uint part_0 = t_0 / 128U;
    thread array<float, int(8)> lo_0;
    thread array<float, int(8)> hi_0;
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
    if(pair_0 < d2_0)
    {
        kw_5 = part_0;
        for(;;)
        {
            bool _S61 = kw_5 < _S10;
            if(_S61)
            {
            }
            else
            {
                break;
            }
            thread array<uint, int(8)> vw_0;
            if(_S61)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    e_0 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + kw_5)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S62 = kStart_0 + kw_5;
                    uint _S63 = _S62 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S64 = (*((&kernelContext_0)->table_0+_S63) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S65 = _S62 % (&kernelContext_0)->p_0->pageSize_0;
                    e_0 = (_S64 + _S65) * d2_0 + pair_0;
                }
                e_0 = *((&kernelContext_0)->v_0+e_0);
            }
            else
            {
                e_0 = 0U;
            }
            vw_0[0U] = e_0;
            uint j_0 = kw_5 + 1U;
            if(j_0 < _S10)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    w_2 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_0)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S66 = kStart_0 + j_0;
                    uint _S67 = _S66 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S68 = (*((&kernelContext_0)->table_0+_S67) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S69 = _S66 % (&kernelContext_0)->p_0->pageSize_0;
                    w_2 = (_S68 + _S69) * d2_0 + pair_0;
                }
                w_2 = *((&kernelContext_0)->v_0+w_2);
            }
            else
            {
                w_2 = 0U;
            }
            vw_0[1U] = w_2;
            uint j_1 = kw_5 + 2U;
            if(j_1 < _S10)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    p0_0 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_1)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S70 = kStart_0 + j_1;
                    uint _S71 = _S70 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S72 = (*((&kernelContext_0)->table_0+_S71) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S73 = _S70 % (&kernelContext_0)->p_0->pageSize_0;
                    p0_0 = (_S72 + _S73) * d2_0 + pair_0;
                }
                p0_0 = *((&kernelContext_0)->v_0+p0_0);
            }
            else
            {
                p0_0 = 0U;
            }
            vw_0[2U] = p0_0;
            uint j_2 = kw_5 + 3U;
            if(j_2 < _S10)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_4 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_2)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S74 = kStart_0 + j_2;
                    uint _S75 = _S74 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S76 = (*((&kernelContext_0)->table_0+_S75) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S77 = _S74 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_4 = (_S76 + _S77) * d2_0 + pair_0;
                }
                kw_4 = *((&kernelContext_0)->v_0+kw_4);
            }
            else
            {
                kw_4 = 0U;
            }
            vw_0[3U] = kw_4;
            uint j_3 = kw_5 + 4U;
            if(j_3 < _S10)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_3 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_3)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S78 = kStart_0 + j_3;
                    uint _S79 = _S78 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S80 = (*((&kernelContext_0)->table_0+_S79) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S81 = _S78 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_3 = (_S80 + _S81) * d2_0 + pair_0;
                }
                kw_3 = *((&kernelContext_0)->v_0+kw_3);
            }
            else
            {
                kw_3 = 0U;
            }
            vw_0[4U] = kw_3;
            uint j_4 = kw_5 + 5U;
            if(j_4 < _S10)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_2 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_4)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S82 = kStart_0 + j_4;
                    uint _S83 = _S82 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S84 = (*((&kernelContext_0)->table_0+_S83) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S85 = _S82 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_2 = (_S84 + _S85) * d2_0 + pair_0;
                }
                kw_2 = *((&kernelContext_0)->v_0+kw_2);
            }
            else
            {
                kw_2 = 0U;
            }
            vw_0[5U] = kw_2;
            uint j_5 = kw_5 + 6U;
            if(j_5 < _S10)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_1 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_5)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S86 = kStart_0 + j_5;
                    uint _S87 = _S86 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S88 = (*((&kernelContext_0)->table_0+_S87) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S89 = _S86 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_1 = (_S88 + _S89) * d2_0 + pair_0;
                }
                kw_1 = *((&kernelContext_0)->v_0+kw_1);
            }
            else
            {
                kw_1 = 0U;
            }
            vw_0[6U] = kw_1;
            uint j_6 = kw_5 + 7U;
            if(j_6 < _S10)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_0 = (_S6 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_6)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S90 = kStart_0 + j_6;
                    uint _S91 = _S90 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S92 = (*((&kernelContext_0)->table_0+_S91) * p_1->kvHeads_0 + _S6) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S93 = _S90 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_0 = (_S92 + _S93) * d2_0 + pair_0;
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
                        if(kw_5 >= _S10)
                        {
                            _S29 = int(1);
                            visible_0 = false;
                            break;
                        }
                        float _S94 = attnLo_0(vw_0[0U]);
                        float _S95 = attnHi_0(vw_0[0U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    bool _S96 = 0U >= _S7;
                                    _S5 = _S96;
                                    if(_S96)
                                    {
                                        _S29 = int(4);
                                        break;
                                    }
                                    float pj_1 = (*(&kernelContext_0)->ss_0)[kw_5];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[kw_5], _S94, lo_0[0U]);
                                    hi_0[0U] = fma(pj_1, _S95, hi_0[0U]);
                                    _S29 = int(6);
                                    break;
                                }
                                if(_S29 != int(6))
                                {
                                    _S28 = _S29;
                                    break;
                                }
                                _S28 = int(5);
                                break;
                            }
                            if(_S28 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S7)
                                    {
                                        _S27 = int(4);
                                        break;
                                    }
                                    float pj_2 = (*(&kernelContext_0)->ss_0)[_S11 + kw_5];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S11 + kw_5], _S94, lo_0[1U]);
                                    hi_0[1U] = fma(pj_2, _S95, hi_0[1U]);
                                    _S27 = int(6);
                                    break;
                                }
                                if(_S27 != int(6))
                                {
                                    _S26 = _S27;
                                    break;
                                }
                                _S26 = int(5);
                                break;
                            }
                            if(_S26 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S7)
                                    {
                                        _S25 = int(4);
                                        break;
                                    }
                                    float pj_3 = (*(&kernelContext_0)->ss_0)[2U * _S11 + kw_5];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S11 + kw_5], _S94, lo_0[2U]);
                                    hi_0[2U] = fma(pj_3, _S95, hi_0[2U]);
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    _S24 = _S25;
                                    break;
                                }
                                _S24 = int(5);
                                break;
                            }
                            if(_S24 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S7)
                                    {
                                        _S23 = int(4);
                                        break;
                                    }
                                    float pj_4 = (*(&kernelContext_0)->ss_0)[3U * _S11 + kw_5];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S11 + kw_5], _S94, lo_0[3U]);
                                    hi_0[3U] = fma(pj_4, _S95, hi_0[3U]);
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    _S22 = _S23;
                                    break;
                                }
                                _S22 = int(5);
                                break;
                            }
                            if(_S22 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S7)
                                    {
                                        _S21 = int(4);
                                        break;
                                    }
                                    float pj_5 = (*(&kernelContext_0)->ss_0)[4U * _S11 + kw_5];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S11 + kw_5], _S94, lo_0[4U]);
                                    hi_0[4U] = fma(pj_5, _S95, hi_0[4U]);
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    _S20 = _S21;
                                    break;
                                }
                                _S20 = int(5);
                                break;
                            }
                            if(_S20 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S7)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_6 = (*(&kernelContext_0)->ss_0)[5U * _S11 + kw_5];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S11 + kw_5], _S94, lo_0[5U]);
                                    hi_0[5U] = fma(pj_6, _S95, hi_0[5U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S7)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_7 = (*(&kernelContext_0)->ss_0)[6U * _S11 + kw_5];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S11 + kw_5], _S94, lo_0[6U]);
                                    hi_0[6U] = fma(pj_7, _S95, hi_0[6U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S7)
                                    {
                                        _S15 = int(4);
                                        break;
                                    }
                                    float pj_8 = (*(&kernelContext_0)->ss_0)[7U * _S11 + kw_5];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S11 + kw_5], _S94, lo_0[7U]);
                                    hi_0[7U] = fma(pj_8, _S95, hi_0[7U]);
                                    _S15 = int(6);
                                    break;
                                }
                                if(_S15 != int(6))
                                {
                                    _S14 = _S15;
                                    break;
                                }
                                _S14 = int(5);
                                break;
                            }
                            if(_S14 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S29 = int(3);
                        visible_0 = _S5;
                        break;
                    }
                    if(_S29 != int(3))
                    {
                        _S28 = _S29;
                        break;
                    }
                    _S28 = int(2);
                    break;
                }
                if(_S28 != int(2))
                {
                    break;
                }
                int _S97;
                int _S98;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_0 >= _S10)
                        {
                            _S27 = int(1);
                            break;
                        }
                        float _S99 = attnLo_0(vw_0[1U]);
                        float _S100 = attnHi_0(vw_0[1U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S27 = int(4);
                                        break;
                                    }
                                    float pj_9 = (*(&kernelContext_0)->ss_0)[j_0];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_0], _S99, lo_0[0U]);
                                    hi_0[0U] = fma(pj_9, _S100, hi_0[0U]);
                                    _S27 = int(6);
                                    break;
                                }
                                if(_S27 != int(6))
                                {
                                    _S26 = _S27;
                                    break;
                                }
                                _S26 = int(5);
                                break;
                            }
                            if(_S26 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S7)
                                    {
                                        _S25 = int(4);
                                        break;
                                    }
                                    float pj_10 = (*(&kernelContext_0)->ss_0)[_S11 + j_0];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S11 + j_0], _S99, lo_0[1U]);
                                    hi_0[1U] = fma(pj_10, _S100, hi_0[1U]);
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    _S24 = _S25;
                                    break;
                                }
                                _S24 = int(5);
                                break;
                            }
                            if(_S24 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S7)
                                    {
                                        _S23 = int(4);
                                        break;
                                    }
                                    float pj_11 = (*(&kernelContext_0)->ss_0)[2U * _S11 + j_0];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S11 + j_0], _S99, lo_0[2U]);
                                    hi_0[2U] = fma(pj_11, _S100, hi_0[2U]);
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    _S22 = _S23;
                                    break;
                                }
                                _S22 = int(5);
                                break;
                            }
                            if(_S22 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S7)
                                    {
                                        _S21 = int(4);
                                        break;
                                    }
                                    float pj_12 = (*(&kernelContext_0)->ss_0)[3U * _S11 + j_0];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S11 + j_0], _S99, lo_0[3U]);
                                    hi_0[3U] = fma(pj_12, _S100, hi_0[3U]);
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    _S20 = _S21;
                                    break;
                                }
                                _S20 = int(5);
                                break;
                            }
                            if(_S20 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S7)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_13 = (*(&kernelContext_0)->ss_0)[4U * _S11 + j_0];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S11 + j_0], _S99, lo_0[4U]);
                                    hi_0[4U] = fma(pj_13, _S100, hi_0[4U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S7)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_14 = (*(&kernelContext_0)->ss_0)[5U * _S11 + j_0];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S11 + j_0], _S99, lo_0[5U]);
                                    hi_0[5U] = fma(pj_14, _S100, hi_0[5U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S7)
                                    {
                                        _S15 = int(4);
                                        break;
                                    }
                                    float pj_15 = (*(&kernelContext_0)->ss_0)[6U * _S11 + j_0];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S11 + j_0], _S99, lo_0[6U]);
                                    hi_0[6U] = fma(pj_15, _S100, hi_0[6U]);
                                    _S15 = int(6);
                                    break;
                                }
                                if(_S15 != int(6))
                                {
                                    _S14 = _S15;
                                    break;
                                }
                                _S14 = int(5);
                                break;
                            }
                            if(_S14 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S7)
                                    {
                                        _S98 = int(4);
                                        break;
                                    }
                                    float pj_16 = (*(&kernelContext_0)->ss_0)[7U * _S11 + j_0];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S11 + j_0], _S99, lo_0[7U]);
                                    hi_0[7U] = fma(pj_16, _S100, hi_0[7U]);
                                    _S98 = int(6);
                                    break;
                                }
                                if(_S98 != int(6))
                                {
                                    _S97 = _S98;
                                    break;
                                }
                                _S97 = int(5);
                                break;
                            }
                            if(_S97 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S27 = int(3);
                        break;
                    }
                    if(_S27 != int(3))
                    {
                        _S26 = _S27;
                        break;
                    }
                    _S26 = int(2);
                    break;
                }
                if(_S26 != int(2))
                {
                    break;
                }
                int _S101;
                int _S102;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_1 >= _S10)
                        {
                            _S25 = int(1);
                            break;
                        }
                        float _S103 = attnLo_0(vw_0[2U]);
                        float _S104 = attnHi_0(vw_0[2U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S25 = int(4);
                                        break;
                                    }
                                    float pj_17 = (*(&kernelContext_0)->ss_0)[j_1];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_1], _S103, lo_0[0U]);
                                    hi_0[0U] = fma(pj_17, _S104, hi_0[0U]);
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    _S24 = _S25;
                                    break;
                                }
                                _S24 = int(5);
                                break;
                            }
                            if(_S24 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S7)
                                    {
                                        _S23 = int(4);
                                        break;
                                    }
                                    float pj_18 = (*(&kernelContext_0)->ss_0)[_S11 + j_1];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S11 + j_1], _S103, lo_0[1U]);
                                    hi_0[1U] = fma(pj_18, _S104, hi_0[1U]);
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    _S22 = _S23;
                                    break;
                                }
                                _S22 = int(5);
                                break;
                            }
                            if(_S22 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S7)
                                    {
                                        _S21 = int(4);
                                        break;
                                    }
                                    float pj_19 = (*(&kernelContext_0)->ss_0)[2U * _S11 + j_1];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S11 + j_1], _S103, lo_0[2U]);
                                    hi_0[2U] = fma(pj_19, _S104, hi_0[2U]);
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    _S20 = _S21;
                                    break;
                                }
                                _S20 = int(5);
                                break;
                            }
                            if(_S20 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S7)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_20 = (*(&kernelContext_0)->ss_0)[3U * _S11 + j_1];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S11 + j_1], _S103, lo_0[3U]);
                                    hi_0[3U] = fma(pj_20, _S104, hi_0[3U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S7)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_21 = (*(&kernelContext_0)->ss_0)[4U * _S11 + j_1];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S11 + j_1], _S103, lo_0[4U]);
                                    hi_0[4U] = fma(pj_21, _S104, hi_0[4U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S7)
                                    {
                                        _S15 = int(4);
                                        break;
                                    }
                                    float pj_22 = (*(&kernelContext_0)->ss_0)[5U * _S11 + j_1];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S11 + j_1], _S103, lo_0[5U]);
                                    hi_0[5U] = fma(pj_22, _S104, hi_0[5U]);
                                    _S15 = int(6);
                                    break;
                                }
                                if(_S15 != int(6))
                                {
                                    _S14 = _S15;
                                    break;
                                }
                                _S14 = int(5);
                                break;
                            }
                            if(_S14 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S7)
                                    {
                                        _S98 = int(4);
                                        break;
                                    }
                                    float pj_23 = (*(&kernelContext_0)->ss_0)[6U * _S11 + j_1];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S11 + j_1], _S103, lo_0[6U]);
                                    hi_0[6U] = fma(pj_23, _S104, hi_0[6U]);
                                    _S98 = int(6);
                                    break;
                                }
                                if(_S98 != int(6))
                                {
                                    _S97 = _S98;
                                    break;
                                }
                                _S97 = int(5);
                                break;
                            }
                            if(_S97 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S7)
                                    {
                                        _S102 = int(4);
                                        break;
                                    }
                                    float pj_24 = (*(&kernelContext_0)->ss_0)[7U * _S11 + j_1];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S11 + j_1], _S103, lo_0[7U]);
                                    hi_0[7U] = fma(pj_24, _S104, hi_0[7U]);
                                    _S102 = int(6);
                                    break;
                                }
                                if(_S102 != int(6))
                                {
                                    _S101 = _S102;
                                    break;
                                }
                                _S101 = int(5);
                                break;
                            }
                            if(_S101 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S25 = int(3);
                        break;
                    }
                    if(_S25 != int(3))
                    {
                        _S24 = _S25;
                        break;
                    }
                    _S24 = int(2);
                    break;
                }
                if(_S24 != int(2))
                {
                    break;
                }
                int _S105;
                int _S106;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_2 >= _S10)
                        {
                            _S23 = int(1);
                            break;
                        }
                        float _S107 = attnLo_0(vw_0[3U]);
                        float _S108 = attnHi_0(vw_0[3U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S23 = int(4);
                                        break;
                                    }
                                    float pj_25 = (*(&kernelContext_0)->ss_0)[j_2];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_2], _S107, lo_0[0U]);
                                    hi_0[0U] = fma(pj_25, _S108, hi_0[0U]);
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    _S22 = _S23;
                                    break;
                                }
                                _S22 = int(5);
                                break;
                            }
                            if(_S22 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S7)
                                    {
                                        _S21 = int(4);
                                        break;
                                    }
                                    float pj_26 = (*(&kernelContext_0)->ss_0)[_S11 + j_2];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S11 + j_2], _S107, lo_0[1U]);
                                    hi_0[1U] = fma(pj_26, _S108, hi_0[1U]);
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    _S20 = _S21;
                                    break;
                                }
                                _S20 = int(5);
                                break;
                            }
                            if(_S20 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S7)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_27 = (*(&kernelContext_0)->ss_0)[2U * _S11 + j_2];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S11 + j_2], _S107, lo_0[2U]);
                                    hi_0[2U] = fma(pj_27, _S108, hi_0[2U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S7)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_28 = (*(&kernelContext_0)->ss_0)[3U * _S11 + j_2];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S11 + j_2], _S107, lo_0[3U]);
                                    hi_0[3U] = fma(pj_28, _S108, hi_0[3U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S7)
                                    {
                                        _S15 = int(4);
                                        break;
                                    }
                                    float pj_29 = (*(&kernelContext_0)->ss_0)[4U * _S11 + j_2];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S11 + j_2], _S107, lo_0[4U]);
                                    hi_0[4U] = fma(pj_29, _S108, hi_0[4U]);
                                    _S15 = int(6);
                                    break;
                                }
                                if(_S15 != int(6))
                                {
                                    _S14 = _S15;
                                    break;
                                }
                                _S14 = int(5);
                                break;
                            }
                            if(_S14 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S7)
                                    {
                                        _S98 = int(4);
                                        break;
                                    }
                                    float pj_30 = (*(&kernelContext_0)->ss_0)[5U * _S11 + j_2];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S11 + j_2], _S107, lo_0[5U]);
                                    hi_0[5U] = fma(pj_30, _S108, hi_0[5U]);
                                    _S98 = int(6);
                                    break;
                                }
                                if(_S98 != int(6))
                                {
                                    _S97 = _S98;
                                    break;
                                }
                                _S97 = int(5);
                                break;
                            }
                            if(_S97 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S7)
                                    {
                                        _S102 = int(4);
                                        break;
                                    }
                                    float pj_31 = (*(&kernelContext_0)->ss_0)[6U * _S11 + j_2];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S11 + j_2], _S107, lo_0[6U]);
                                    hi_0[6U] = fma(pj_31, _S108, hi_0[6U]);
                                    _S102 = int(6);
                                    break;
                                }
                                if(_S102 != int(6))
                                {
                                    _S101 = _S102;
                                    break;
                                }
                                _S101 = int(5);
                                break;
                            }
                            if(_S101 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S7)
                                    {
                                        _S106 = int(4);
                                        break;
                                    }
                                    float pj_32 = (*(&kernelContext_0)->ss_0)[7U * _S11 + j_2];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S11 + j_2], _S107, lo_0[7U]);
                                    hi_0[7U] = fma(pj_32, _S108, hi_0[7U]);
                                    _S106 = int(6);
                                    break;
                                }
                                if(_S106 != int(6))
                                {
                                    _S105 = _S106;
                                    break;
                                }
                                _S105 = int(5);
                                break;
                            }
                            if(_S105 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S23 = int(3);
                        break;
                    }
                    if(_S23 != int(3))
                    {
                        _S22 = _S23;
                        break;
                    }
                    _S22 = int(2);
                    break;
                }
                if(_S22 != int(2))
                {
                    break;
                }
                int _S109;
                int _S110;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_3 >= _S10)
                        {
                            _S21 = int(1);
                            break;
                        }
                        float _S111 = attnLo_0(vw_0[4U]);
                        float _S112 = attnHi_0(vw_0[4U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S21 = int(4);
                                        break;
                                    }
                                    float pj_33 = (*(&kernelContext_0)->ss_0)[j_3];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_3], _S111, lo_0[0U]);
                                    hi_0[0U] = fma(pj_33, _S112, hi_0[0U]);
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    _S20 = _S21;
                                    break;
                                }
                                _S20 = int(5);
                                break;
                            }
                            if(_S20 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S7)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_34 = (*(&kernelContext_0)->ss_0)[_S11 + j_3];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S11 + j_3], _S111, lo_0[1U]);
                                    hi_0[1U] = fma(pj_34, _S112, hi_0[1U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S7)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_35 = (*(&kernelContext_0)->ss_0)[2U * _S11 + j_3];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S11 + j_3], _S111, lo_0[2U]);
                                    hi_0[2U] = fma(pj_35, _S112, hi_0[2U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S7)
                                    {
                                        _S15 = int(4);
                                        break;
                                    }
                                    float pj_36 = (*(&kernelContext_0)->ss_0)[3U * _S11 + j_3];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S11 + j_3], _S111, lo_0[3U]);
                                    hi_0[3U] = fma(pj_36, _S112, hi_0[3U]);
                                    _S15 = int(6);
                                    break;
                                }
                                if(_S15 != int(6))
                                {
                                    _S14 = _S15;
                                    break;
                                }
                                _S14 = int(5);
                                break;
                            }
                            if(_S14 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S7)
                                    {
                                        _S98 = int(4);
                                        break;
                                    }
                                    float pj_37 = (*(&kernelContext_0)->ss_0)[4U * _S11 + j_3];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S11 + j_3], _S111, lo_0[4U]);
                                    hi_0[4U] = fma(pj_37, _S112, hi_0[4U]);
                                    _S98 = int(6);
                                    break;
                                }
                                if(_S98 != int(6))
                                {
                                    _S97 = _S98;
                                    break;
                                }
                                _S97 = int(5);
                                break;
                            }
                            if(_S97 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S7)
                                    {
                                        _S102 = int(4);
                                        break;
                                    }
                                    float pj_38 = (*(&kernelContext_0)->ss_0)[5U * _S11 + j_3];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S11 + j_3], _S111, lo_0[5U]);
                                    hi_0[5U] = fma(pj_38, _S112, hi_0[5U]);
                                    _S102 = int(6);
                                    break;
                                }
                                if(_S102 != int(6))
                                {
                                    _S101 = _S102;
                                    break;
                                }
                                _S101 = int(5);
                                break;
                            }
                            if(_S101 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S7)
                                    {
                                        _S106 = int(4);
                                        break;
                                    }
                                    float pj_39 = (*(&kernelContext_0)->ss_0)[6U * _S11 + j_3];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S11 + j_3], _S111, lo_0[6U]);
                                    hi_0[6U] = fma(pj_39, _S112, hi_0[6U]);
                                    _S106 = int(6);
                                    break;
                                }
                                if(_S106 != int(6))
                                {
                                    _S105 = _S106;
                                    break;
                                }
                                _S105 = int(5);
                                break;
                            }
                            if(_S105 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S7)
                                    {
                                        _S110 = int(4);
                                        break;
                                    }
                                    float pj_40 = (*(&kernelContext_0)->ss_0)[7U * _S11 + j_3];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S11 + j_3], _S111, lo_0[7U]);
                                    hi_0[7U] = fma(pj_40, _S112, hi_0[7U]);
                                    _S110 = int(6);
                                    break;
                                }
                                if(_S110 != int(6))
                                {
                                    _S109 = _S110;
                                    break;
                                }
                                _S109 = int(5);
                                break;
                            }
                            if(_S109 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S21 = int(3);
                        break;
                    }
                    if(_S21 != int(3))
                    {
                        _S20 = _S21;
                        break;
                    }
                    _S20 = int(2);
                    break;
                }
                if(_S20 != int(2))
                {
                    break;
                }
                int _S113;
                int _S114;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_4 >= _S10)
                        {
                            _S19 = int(1);
                            break;
                        }
                        float _S115 = attnLo_0(vw_0[5U]);
                        float _S116 = attnHi_0(vw_0[5U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_41 = (*(&kernelContext_0)->ss_0)[j_4];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_4], _S115, lo_0[0U]);
                                    hi_0[0U] = fma(pj_41, _S116, hi_0[0U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S7)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_42 = (*(&kernelContext_0)->ss_0)[_S11 + j_4];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S11 + j_4], _S115, lo_0[1U]);
                                    hi_0[1U] = fma(pj_42, _S116, hi_0[1U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S7)
                                    {
                                        _S15 = int(4);
                                        break;
                                    }
                                    float pj_43 = (*(&kernelContext_0)->ss_0)[2U * _S11 + j_4];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S11 + j_4], _S115, lo_0[2U]);
                                    hi_0[2U] = fma(pj_43, _S116, hi_0[2U]);
                                    _S15 = int(6);
                                    break;
                                }
                                if(_S15 != int(6))
                                {
                                    _S14 = _S15;
                                    break;
                                }
                                _S14 = int(5);
                                break;
                            }
                            if(_S14 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S7)
                                    {
                                        _S98 = int(4);
                                        break;
                                    }
                                    float pj_44 = (*(&kernelContext_0)->ss_0)[3U * _S11 + j_4];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S11 + j_4], _S115, lo_0[3U]);
                                    hi_0[3U] = fma(pj_44, _S116, hi_0[3U]);
                                    _S98 = int(6);
                                    break;
                                }
                                if(_S98 != int(6))
                                {
                                    _S97 = _S98;
                                    break;
                                }
                                _S97 = int(5);
                                break;
                            }
                            if(_S97 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S7)
                                    {
                                        _S102 = int(4);
                                        break;
                                    }
                                    float pj_45 = (*(&kernelContext_0)->ss_0)[4U * _S11 + j_4];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S11 + j_4], _S115, lo_0[4U]);
                                    hi_0[4U] = fma(pj_45, _S116, hi_0[4U]);
                                    _S102 = int(6);
                                    break;
                                }
                                if(_S102 != int(6))
                                {
                                    _S101 = _S102;
                                    break;
                                }
                                _S101 = int(5);
                                break;
                            }
                            if(_S101 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S7)
                                    {
                                        _S106 = int(4);
                                        break;
                                    }
                                    float pj_46 = (*(&kernelContext_0)->ss_0)[5U * _S11 + j_4];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S11 + j_4], _S115, lo_0[5U]);
                                    hi_0[5U] = fma(pj_46, _S116, hi_0[5U]);
                                    _S106 = int(6);
                                    break;
                                }
                                if(_S106 != int(6))
                                {
                                    _S105 = _S106;
                                    break;
                                }
                                _S105 = int(5);
                                break;
                            }
                            if(_S105 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S7)
                                    {
                                        _S110 = int(4);
                                        break;
                                    }
                                    float pj_47 = (*(&kernelContext_0)->ss_0)[6U * _S11 + j_4];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S11 + j_4], _S115, lo_0[6U]);
                                    hi_0[6U] = fma(pj_47, _S116, hi_0[6U]);
                                    _S110 = int(6);
                                    break;
                                }
                                if(_S110 != int(6))
                                {
                                    _S109 = _S110;
                                    break;
                                }
                                _S109 = int(5);
                                break;
                            }
                            if(_S109 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S7)
                                    {
                                        _S114 = int(4);
                                        break;
                                    }
                                    float pj_48 = (*(&kernelContext_0)->ss_0)[7U * _S11 + j_4];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S11 + j_4], _S115, lo_0[7U]);
                                    hi_0[7U] = fma(pj_48, _S116, hi_0[7U]);
                                    _S114 = int(6);
                                    break;
                                }
                                if(_S114 != int(6))
                                {
                                    _S113 = _S114;
                                    break;
                                }
                                _S113 = int(5);
                                break;
                            }
                            if(_S113 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S19 = int(3);
                        break;
                    }
                    if(_S19 != int(3))
                    {
                        _S18 = _S19;
                        break;
                    }
                    _S18 = int(2);
                    break;
                }
                if(_S18 != int(2))
                {
                    break;
                }
                int _S117;
                int _S118;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_5 >= _S10)
                        {
                            _S17 = int(1);
                            break;
                        }
                        float _S119 = attnLo_0(vw_0[6U]);
                        float _S120 = attnHi_0(vw_0[6U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_49 = (*(&kernelContext_0)->ss_0)[j_5];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_5], _S119, lo_0[0U]);
                                    hi_0[0U] = fma(pj_49, _S120, hi_0[0U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S7)
                                    {
                                        _S15 = int(4);
                                        break;
                                    }
                                    float pj_50 = (*(&kernelContext_0)->ss_0)[_S11 + j_5];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S11 + j_5], _S119, lo_0[1U]);
                                    hi_0[1U] = fma(pj_50, _S120, hi_0[1U]);
                                    _S15 = int(6);
                                    break;
                                }
                                if(_S15 != int(6))
                                {
                                    _S14 = _S15;
                                    break;
                                }
                                _S14 = int(5);
                                break;
                            }
                            if(_S14 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S7)
                                    {
                                        _S98 = int(4);
                                        break;
                                    }
                                    float pj_51 = (*(&kernelContext_0)->ss_0)[2U * _S11 + j_5];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S11 + j_5], _S119, lo_0[2U]);
                                    hi_0[2U] = fma(pj_51, _S120, hi_0[2U]);
                                    _S98 = int(6);
                                    break;
                                }
                                if(_S98 != int(6))
                                {
                                    _S97 = _S98;
                                    break;
                                }
                                _S97 = int(5);
                                break;
                            }
                            if(_S97 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S7)
                                    {
                                        _S102 = int(4);
                                        break;
                                    }
                                    float pj_52 = (*(&kernelContext_0)->ss_0)[3U * _S11 + j_5];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S11 + j_5], _S119, lo_0[3U]);
                                    hi_0[3U] = fma(pj_52, _S120, hi_0[3U]);
                                    _S102 = int(6);
                                    break;
                                }
                                if(_S102 != int(6))
                                {
                                    _S101 = _S102;
                                    break;
                                }
                                _S101 = int(5);
                                break;
                            }
                            if(_S101 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S7)
                                    {
                                        _S106 = int(4);
                                        break;
                                    }
                                    float pj_53 = (*(&kernelContext_0)->ss_0)[4U * _S11 + j_5];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S11 + j_5], _S119, lo_0[4U]);
                                    hi_0[4U] = fma(pj_53, _S120, hi_0[4U]);
                                    _S106 = int(6);
                                    break;
                                }
                                if(_S106 != int(6))
                                {
                                    _S105 = _S106;
                                    break;
                                }
                                _S105 = int(5);
                                break;
                            }
                            if(_S105 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S7)
                                    {
                                        _S110 = int(4);
                                        break;
                                    }
                                    float pj_54 = (*(&kernelContext_0)->ss_0)[5U * _S11 + j_5];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S11 + j_5], _S119, lo_0[5U]);
                                    hi_0[5U] = fma(pj_54, _S120, hi_0[5U]);
                                    _S110 = int(6);
                                    break;
                                }
                                if(_S110 != int(6))
                                {
                                    _S109 = _S110;
                                    break;
                                }
                                _S109 = int(5);
                                break;
                            }
                            if(_S109 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S7)
                                    {
                                        _S114 = int(4);
                                        break;
                                    }
                                    float pj_55 = (*(&kernelContext_0)->ss_0)[6U * _S11 + j_5];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S11 + j_5], _S119, lo_0[6U]);
                                    hi_0[6U] = fma(pj_55, _S120, hi_0[6U]);
                                    _S114 = int(6);
                                    break;
                                }
                                if(_S114 != int(6))
                                {
                                    _S113 = _S114;
                                    break;
                                }
                                _S113 = int(5);
                                break;
                            }
                            if(_S113 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S7)
                                    {
                                        _S118 = int(4);
                                        break;
                                    }
                                    float pj_56 = (*(&kernelContext_0)->ss_0)[7U * _S11 + j_5];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S11 + j_5], _S119, lo_0[7U]);
                                    hi_0[7U] = fma(pj_56, _S120, hi_0[7U]);
                                    _S118 = int(6);
                                    break;
                                }
                                if(_S118 != int(6))
                                {
                                    _S117 = _S118;
                                    break;
                                }
                                _S117 = int(5);
                                break;
                            }
                            if(_S117 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S17 = int(3);
                        break;
                    }
                    if(_S17 != int(3))
                    {
                        _S16 = _S17;
                        break;
                    }
                    _S16 = int(2);
                    break;
                }
                if(_S16 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(j_6 >= _S10)
                        {
                            _S15 = int(1);
                            break;
                        }
                        float _S121 = attnLo_0(vw_0[7U]);
                        float _S122 = attnHi_0(vw_0[7U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S15 = int(4);
                                        break;
                                    }
                                    float pj_57 = (*(&kernelContext_0)->ss_0)[j_6];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_6], _S121, lo_0[0U]);
                                    hi_0[0U] = fma(pj_57, _S122, hi_0[0U]);
                                    _S15 = int(6);
                                    break;
                                }
                                if(_S15 != int(6))
                                {
                                    _S14 = _S15;
                                    break;
                                }
                                _S14 = int(5);
                                break;
                            }
                            if(_S14 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S7)
                                    {
                                        _S98 = int(4);
                                        break;
                                    }
                                    float pj_58 = (*(&kernelContext_0)->ss_0)[_S11 + j_6];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S11 + j_6], _S121, lo_0[1U]);
                                    hi_0[1U] = fma(pj_58, _S122, hi_0[1U]);
                                    _S98 = int(6);
                                    break;
                                }
                                if(_S98 != int(6))
                                {
                                    _S97 = _S98;
                                    break;
                                }
                                _S97 = int(5);
                                break;
                            }
                            if(_S97 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S7)
                                    {
                                        _S102 = int(4);
                                        break;
                                    }
                                    float pj_59 = (*(&kernelContext_0)->ss_0)[2U * _S11 + j_6];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S11 + j_6], _S121, lo_0[2U]);
                                    hi_0[2U] = fma(pj_59, _S122, hi_0[2U]);
                                    _S102 = int(6);
                                    break;
                                }
                                if(_S102 != int(6))
                                {
                                    _S101 = _S102;
                                    break;
                                }
                                _S101 = int(5);
                                break;
                            }
                            if(_S101 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S7)
                                    {
                                        _S106 = int(4);
                                        break;
                                    }
                                    float pj_60 = (*(&kernelContext_0)->ss_0)[3U * _S11 + j_6];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S11 + j_6], _S121, lo_0[3U]);
                                    hi_0[3U] = fma(pj_60, _S122, hi_0[3U]);
                                    _S106 = int(6);
                                    break;
                                }
                                if(_S106 != int(6))
                                {
                                    _S105 = _S106;
                                    break;
                                }
                                _S105 = int(5);
                                break;
                            }
                            if(_S105 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S7)
                                    {
                                        _S110 = int(4);
                                        break;
                                    }
                                    float pj_61 = (*(&kernelContext_0)->ss_0)[4U * _S11 + j_6];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S11 + j_6], _S121, lo_0[4U]);
                                    hi_0[4U] = fma(pj_61, _S122, hi_0[4U]);
                                    _S110 = int(6);
                                    break;
                                }
                                if(_S110 != int(6))
                                {
                                    _S109 = _S110;
                                    break;
                                }
                                _S109 = int(5);
                                break;
                            }
                            if(_S109 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S7)
                                    {
                                        _S114 = int(4);
                                        break;
                                    }
                                    float pj_62 = (*(&kernelContext_0)->ss_0)[5U * _S11 + j_6];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S11 + j_6], _S121, lo_0[5U]);
                                    hi_0[5U] = fma(pj_62, _S122, hi_0[5U]);
                                    _S114 = int(6);
                                    break;
                                }
                                if(_S114 != int(6))
                                {
                                    _S113 = _S114;
                                    break;
                                }
                                _S113 = int(5);
                                break;
                            }
                            if(_S113 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S7)
                                    {
                                        _S118 = int(4);
                                        break;
                                    }
                                    float pj_63 = (*(&kernelContext_0)->ss_0)[6U * _S11 + j_6];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S11 + j_6], _S121, lo_0[6U]);
                                    hi_0[6U] = fma(pj_63, _S122, hi_0[6U]);
                                    _S118 = int(6);
                                    break;
                                }
                                if(_S118 != int(6))
                                {
                                    _S117 = _S118;
                                    break;
                                }
                                _S117 = int(5);
                                break;
                            }
                            if(_S117 != int(5))
                            {
                                break;
                            }
                            int _S123;
                            for(;;)
                            {
                                int _S124;
                                for(;;)
                                {
                                    if(7U >= _S7)
                                    {
                                        _S124 = int(4);
                                        break;
                                    }
                                    float pj_64 = (*(&kernelContext_0)->ss_0)[7U * _S11 + j_6];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S11 + j_6], _S121, lo_0[7U]);
                                    hi_0[7U] = fma(pj_64, _S122, hi_0[7U]);
                                    _S124 = int(6);
                                    break;
                                }
                                if(_S124 != int(6))
                                {
                                    _S123 = _S124;
                                    break;
                                }
                                _S123 = int(5);
                                break;
                            }
                            if(_S123 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S15 = int(3);
                        break;
                    }
                    if(_S15 != int(3))
                    {
                        _S14 = _S15;
                        break;
                    }
                    _S14 = int(2);
                    break;
                }
                if(_S14 != int(2))
                {
                    break;
                }
                break;
            }
            kw_5 = kw_5 + 8U;
        }
    }
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(0U == p0_0)
        {
            acc_0 = lo_0[0U];
            acc_1 = hi_0[0U];
        }
        else
        {
            acc_0 = 0.0;
            acc_1 = 0.0;
        }
        float b_3;
        if(1U == p0_0)
        {
            pj_0 = lo_0[1U];
            b_3 = hi_0[1U];
        }
        else
        {
            pj_0 = acc_0;
            b_3 = acc_1;
        }
        float a_3;
        float b_4;
        if(2U == p0_0)
        {
            a_3 = lo_0[2U];
            b_4 = hi_0[2U];
        }
        else
        {
            a_3 = pj_0;
            b_4 = b_3;
        }
        float a_4;
        float b_5;
        if(3U == p0_0)
        {
            a_4 = lo_0[3U];
            b_5 = hi_0[3U];
        }
        else
        {
            a_4 = a_3;
            b_5 = b_4;
        }
        float a_5;
        float b_6;
        if(4U == p0_0)
        {
            a_5 = lo_0[4U];
            b_6 = hi_0[4U];
        }
        else
        {
            a_5 = a_4;
            b_6 = b_5;
        }
        float a_6;
        float b_7;
        if(5U == p0_0)
        {
            a_6 = lo_0[5U];
            b_7 = hi_0[5U];
        }
        else
        {
            a_6 = a_5;
            b_7 = b_6;
        }
        float a_7;
        float b_8;
        if(6U == p0_0)
        {
            a_7 = lo_0[6U];
            b_8 = hi_0[6U];
        }
        else
        {
            a_7 = a_6;
            b_8 = b_7;
        }
        float a_8;
        float b_9;
        if(7U == p0_0)
        {
            a_8 = lo_0[7U];
            b_9 = hi_0[7U];
        }
        else
        {
            a_8 = a_7;
            b_9 = b_8;
        }
        uint _S125 = 2U * pair_0;
        uint _S126 = part_0 * 256U + _S125;
        (*(&kernelContext_0)->ss_0)[_S126] = a_8;
        (*(&kernelContext_0)->ss_0)[_S126 + 1U] = b_9;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(part_0 != 0U)
        {
            visible_0 = true;
        }
        else
        {
            visible_0 = pair_0 >= d2_0;
        }
        if(visible_0)
        {
            p0_0 = p0_0 + 1U;
            continue;
        }
        float a_9 = (*(&kernelContext_0)->ss_0)[_S125];
        float b_10 = (*(&kernelContext_0)->ss_0)[_S125 + 1U];
        uint _S127 = _S6 * hpk_0;
        uint _S128 = p0_0 / (&kernelContext_0)->p_0->queries_0;
        uint h_1 = _S127 + _S128;
        uint i_9 = p0_0 % (&kernelContext_0)->p_0->queries_0;
        uint row_0 = h_1 * (&kernelContext_0)->p_0->queries_0 + i_9;
        if(((&kernelContext_0)->p_0->splits_0) > 1U)
        {
            uint pr_8 = split_0 * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0 + row_0;
            if(pair_0 == 0U)
            {
                uint _S129 = 2U * pr_8;
                float device* _S130 = (&kernelContext_0)->partialML_0+_S129;
                bool _S131 = _S10 == 0U;
                float _S132;
                if(_S131)
                {
                    _S132 = dg_negative_infinity_0();
                }
                else
                {
                    _S132 = (*(&kernelContext_0)->rowM_0)[p0_0];
                }
                *_S130 = _S132;
                float device* _S133 = (&kernelContext_0)->partialML_0+(_S129 + 1U);
                float _S134;
                if(_S131)
                {
                    _S134 = 0.0;
                }
                else
                {
                    _S134 = (*(&kernelContext_0)->rowL_0)[p0_0];
                }
                *_S133 = _S134;
            }
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S125)) = a_9;
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S125 + 1U)) = b_10;
            p0_0 = p0_0 + 1U;
            continue;
        }
        float l_0 = (*(&kernelContext_0)->rowL_0)[p0_0];
        *((&kernelContext_0)->o_0+(row_0 * (&kernelContext_0)->p_0->headDim_0 + _S125)) = a_9 / (*(&kernelContext_0)->rowL_0)[p0_0];
        *((&kernelContext_0)->o_0+(row_0 * (&kernelContext_0)->p_0->headDim_0 + _S125 + 1U)) = b_10 / l_0;
        p0_0 = p0_0 + 1U;
    }
    return;
}

