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
    uint threadgroup* waves_0;
    array<uint, int(256)> threadgroup* qs_0;
    array<float, int(1024)> threadgroup* ss_0;
    array<float, int(8)> threadgroup* rowM_0;
    array<float, int(8)> threadgroup* rowL_0;
};

[[kernel]] void attn_decode_wave_64_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(7)]], uint device* q_1 [[buffer(0)]], uint device* table_1 [[buffer(6)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partialML_1 [[buffer(4)]], float device* partialAcc_1 [[buffer(5)]], float device* o_1 [[buffer(3)]])
{
    bool _S1;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->q_0 = q_1;
    (&kernelContext_0)->table_0 = table_1;
    (&kernelContext_0)->k_0 = k_1;
    (&kernelContext_0)->v_0 = v_1;
    (&kernelContext_0)->partialML_0 = partialML_1;
    (&kernelContext_0)->partialAcc_0 = partialAcc_1;
    (&kernelContext_0)->o_0 = o_1;
    threadgroup uint waves_1;
    (&kernelContext_0)->waves_0 = &waves_1;
    threadgroup array<uint, int(256)> qs_1;
    (&kernelContext_0)->qs_0 = &qs_1;
    threadgroup array<float, int(1024)> ss_1;
    (&kernelContext_0)->ss_0 = &ss_1;
    threadgroup array<float, int(8)> rowM_1;
    (&kernelContext_0)->rowM_0 = &rowM_1;
    threadgroup array<float, int(8)> rowL_1;
    (&kernelContext_0)->rowL_0 = &rowL_1;
    uint t_0 = local_0.x;
    uint _S2 = group_0.y;
    uint split_0 = group_0.z;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint _S3 = min(hpk_0 * (&kernelContext_0)->p_0->queries_0, ROW_TIER_0);
    uint d2_0 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint _S4 = (&kernelContext_0)->p_0->keys_0 - (&kernelContext_0)->p_0->queries_0;
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S5 = min((&kernelContext_0)->p_0->keys_0, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    uint _S6;
    if(_S5 > kStart_0)
    {
        _S6 = _S5 - kStart_0;
    }
    else
    {
        _S6 = 0U;
    }
    uint _S7 = (&kernelContext_0)->p_0->splitKeys_0;
    if(t_0 == 0U)
    {
        *(&kernelContext_0)->waves_0 = 0U;
    }
    bool visible_0;
    uint w_2;
    uint e_0 = t_0;
    for(;;)
    {
        if(e_0 < (ROW_TIER_0 * 32U))
        {
        }
        else
        {
            break;
        }
        uint r_0 = e_0 / 32U;
        uint pr_0 = e_0 % 32U;
        if(r_0 < _S3)
        {
            visible_0 = pr_0 < d2_0;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S8 = _S2 * hpk_0;
            uint _S9 = r_0 / (&kernelContext_0)->p_0->queries_0;
            uint h_0 = _S8 + _S9;
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S10 = simd_sum(1U);
    uint lane_0 = simd_prefix_exclusive_sum(1U);
    bool _S11 = lane_0 == 0U;
    if(_S11)
    {
        uint _S12 = atomic_fetch_add_explicit(((atomic_uint threadgroup*)((&kernelContext_0)->waves_0)), 1U, memory_order_relaxed);
        w_2 = _S12;
    }
    else
    {
        w_2 = 0U;
    }
    float m_0;
    int _S13;
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
    uint kw_0;
    uint kw_1;
    uint kw_2;
    uint kw_3;
    uint kw_4;
    uint p0_0;
    uint wid_0 = simd_sum(w_2);
    uint j_0 = t_0;
    for(;;)
    {
        if(j_0 < _S6)
        {
        }
        else
        {
            break;
        }
        uint key_0 = kStart_0 + j_0;
        if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
        {
            e_0 = (_S2 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * d2_0;
        }
        else
        {
            uint _S29 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
            uint _S30 = (*((&kernelContext_0)->table_0+_S29) * p_1->kvHeads_0 + _S2) * (&kernelContext_0)->p_0->pageSize_0;
            uint _S31 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
            e_0 = (_S30 + _S31) * d2_0;
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
            if(p0_0 < 32U)
            {
            }
            else
            {
                break;
            }
            if(p0_0 < d2_0)
            {
                kw_4 = *((&kernelContext_0)->k_0+(e_0 + p0_0));
            }
            else
            {
                kw_4 = 0U;
            }
            float _S32 = attnLo_0(kw_4);
            float _S33 = attnHi_0(kw_4);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[p0_0]), _S33, fma(attnLo_0((*(&kernelContext_0)->qs_0)[p0_0]), _S32, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + p0_0]), _S33, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + p0_0]), _S32, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S33, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S32, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + p0_0]), _S33, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + p0_0]), _S32, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S33, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S32, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + p0_0]), _S33, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + p0_0]), _S32, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S33, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S32, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + p0_0]), _S33, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + p0_0]), _S32, s_0[7U]));
            }
            uint pr_1 = p0_0 + 1U;
            if(pr_1 < d2_0)
            {
                kw_3 = *((&kernelContext_0)->k_0+(e_0 + pr_1));
            }
            else
            {
                kw_3 = 0U;
            }
            float _S34 = attnLo_0(kw_3);
            float _S35 = attnHi_0(kw_3);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_1]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_1]), _S34, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_1]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_1]), _S34, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S34, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_1]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_1]), _S34, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S34, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_1]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_1]), _S34, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S34, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_1]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_1]), _S34, s_0[7U]));
            }
            uint pr_2 = p0_0 + 2U;
            if(pr_2 < d2_0)
            {
                kw_2 = *((&kernelContext_0)->k_0+(e_0 + pr_2));
            }
            else
            {
                kw_2 = 0U;
            }
            float _S36 = attnLo_0(kw_2);
            float _S37 = attnHi_0(kw_2);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_2]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_2]), _S36, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_2]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_2]), _S36, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S36, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_2]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_2]), _S36, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S36, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_2]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_2]), _S36, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S36, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_2]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_2]), _S36, s_0[7U]));
            }
            uint pr_3 = p0_0 + 3U;
            if(pr_3 < d2_0)
            {
                kw_1 = *((&kernelContext_0)->k_0+(e_0 + pr_3));
            }
            else
            {
                kw_1 = 0U;
            }
            float _S38 = attnLo_0(kw_1);
            float _S39 = attnHi_0(kw_1);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_3]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_3]), _S38, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_3]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_3]), _S38, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S38, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_3]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_3]), _S38, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S38, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_3]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_3]), _S38, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S38, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_3]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_3]), _S38, s_0[7U]));
            }
            uint pr_4 = p0_0 + 4U;
            if(pr_4 < d2_0)
            {
                kw_0 = *((&kernelContext_0)->k_0+(e_0 + pr_4));
            }
            else
            {
                kw_0 = 0U;
            }
            float _S40 = attnLo_0(kw_0);
            float _S41 = attnHi_0(kw_0);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_4]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_4]), _S40, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_4]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_4]), _S40, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S40, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_4]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_4]), _S40, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S40, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_4]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_4]), _S40, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S40, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_4]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_4]), _S40, s_0[7U]));
            }
            uint pr_5 = p0_0 + 5U;
            uint kw_5;
            if(pr_5 < d2_0)
            {
                kw_5 = *((&kernelContext_0)->k_0+(e_0 + pr_5));
            }
            else
            {
                kw_5 = 0U;
            }
            float _S42 = attnLo_0(kw_5);
            float _S43 = attnHi_0(kw_5);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_5]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_5]), _S42, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_5]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_5]), _S42, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S42, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_5]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_5]), _S42, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S42, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_5]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_5]), _S42, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S42, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_5]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_5]), _S42, s_0[7U]));
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
            float _S44 = attnLo_0(kw_6);
            float _S45 = attnHi_0(kw_6);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_6]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_6]), _S44, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_6]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_6]), _S44, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S44, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_6]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_6]), _S44, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S44, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_6]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_6]), _S44, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S44, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_6]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_6]), _S44, s_0[7U]));
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
            float _S46 = attnLo_0(kw_7);
            float _S47 = attnHi_0(kw_7);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_7]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_7]), _S46, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_7]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_7]), _S46, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S46, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_7]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_7]), _S46, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S46, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_7]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_7]), _S46, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S46, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_7]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_7]), _S46, s_0[7U]));
            }
            p0_0 = p0_0 + 8U;
        }
        for(;;)
        {
            for(;;)
            {
                for(;;)
                {
                    if(0U >= _S3)
                    {
                        _S28 = int(1);
                        break;
                    }
                    uint i_1 = 0U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_1 + _S4);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[0U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[j_0] = m_0;
                    _S28 = int(3);
                    break;
                }
                if(_S28 != int(3))
                {
                    _S27 = _S28;
                    break;
                }
                _S27 = int(2);
                break;
            }
            if(_S27 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(1U >= _S3)
                    {
                        _S26 = int(1);
                        break;
                    }
                    uint i_2 = 1U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_2 + _S4);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[1U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[_S7 + j_0] = m_0;
                    _S26 = int(3);
                    break;
                }
                if(_S26 != int(3))
                {
                    _S25 = _S26;
                    break;
                }
                _S25 = int(2);
                break;
            }
            if(_S25 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(2U >= _S3)
                    {
                        _S24 = int(1);
                        break;
                    }
                    uint i_3 = 2U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_3 + _S4);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[2U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[2U * _S7 + j_0] = m_0;
                    _S24 = int(3);
                    break;
                }
                if(_S24 != int(3))
                {
                    _S23 = _S24;
                    break;
                }
                _S23 = int(2);
                break;
            }
            if(_S23 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(3U >= _S3)
                    {
                        _S22 = int(1);
                        break;
                    }
                    uint i_4 = 3U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_4 + _S4);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[3U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[3U * _S7 + j_0] = m_0;
                    _S22 = int(3);
                    break;
                }
                if(_S22 != int(3))
                {
                    _S21 = _S22;
                    break;
                }
                _S21 = int(2);
                break;
            }
            if(_S21 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(4U >= _S3)
                    {
                        _S20 = int(1);
                        break;
                    }
                    uint i_5 = 4U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_5 + _S4);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[4U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[4U * _S7 + j_0] = m_0;
                    _S20 = int(3);
                    break;
                }
                if(_S20 != int(3))
                {
                    _S19 = _S20;
                    break;
                }
                _S19 = int(2);
                break;
            }
            if(_S19 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(5U >= _S3)
                    {
                        _S18 = int(1);
                        break;
                    }
                    uint i_6 = 5U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_6 + _S4);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[5U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[5U * _S7 + j_0] = m_0;
                    _S18 = int(3);
                    break;
                }
                if(_S18 != int(3))
                {
                    _S17 = _S18;
                    break;
                }
                _S17 = int(2);
                break;
            }
            if(_S17 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(6U >= _S3)
                    {
                        _S16 = int(1);
                        break;
                    }
                    uint i_7 = 6U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_7 + _S4);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[6U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[6U * _S7 + j_0] = m_0;
                    _S16 = int(3);
                    break;
                }
                if(_S16 != int(3))
                {
                    _S15 = _S16;
                    break;
                }
                _S15 = int(2);
                break;
            }
            if(_S15 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(7U >= _S3)
                    {
                        _S14 = int(1);
                        break;
                    }
                    uint i_8 = 7U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_8 + _S4);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[7U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[7U * _S7 + j_0] = m_0;
                    _S14 = int(3);
                    break;
                }
                if(_S14 != int(3))
                {
                    _S13 = _S14;
                    break;
                }
                _S13 = int(2);
                break;
            }
            if(_S13 != int(2))
            {
                break;
            }
            break;
        }
        j_0 = j_0 + 128U;
    }
    float w_3;
    float sum_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p0_0 = wid_0;
    for(;;)
    {
        if(p0_0 < _S3)
        {
        }
        else
        {
            break;
        }
        float _S48 = dg_negative_infinity_0();
        j_0 = lane_0;
        m_0 = _S48;
        for(;;)
        {
            if(j_0 < _S6)
            {
            }
            else
            {
                break;
            }
            float _S49 = max(m_0, (*(&kernelContext_0)->ss_0)[p0_0 * _S7 + j_0]);
            j_0 = j_0 + _S10;
            m_0 = _S49;
        }
        float m_1 = simd_max(m_0);
        kw_4 = lane_0;
        sum_0 = 0.0;
        for(;;)
        {
            if(kw_4 < _S6)
            {
            }
            else
            {
                break;
            }
            float sv_0 = (*(&kernelContext_0)->ss_0)[p0_0 * _S7 + kw_4];
            if(((*(&kernelContext_0)->ss_0)[p0_0 * _S7 + kw_4]) == _S48)
            {
                w_3 = 0.0;
            }
            else
            {
                w_3 = exp(sv_0 - m_1);
            }
            (*(&kernelContext_0)->ss_0)[p0_0 * _S7 + kw_4] = w_3;
            float sum_1 = sum_0 + w_3;
            kw_4 = kw_4 + _S10;
            sum_0 = sum_1;
        }
        float sum_2 = simd_sum(sum_0);
        if(_S11)
        {
            (*(&kernelContext_0)->rowM_0)[p0_0] = m_1;
            (*(&kernelContext_0)->rowL_0)[p0_0] = sum_2;
        }
        p0_0 = p0_0 + *(&kernelContext_0)->waves_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pair_0 = t_0 % 32U;
    uint part_0 = t_0 / 32U;
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
        kw_4 = part_0;
        for(;;)
        {
            bool _S50 = kw_4 < _S6;
            if(_S50)
            {
            }
            else
            {
                break;
            }
            thread array<uint, int(8)> vw_0;
            if(_S50)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    e_0 = (_S2 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + kw_4)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S51 = kStart_0 + kw_4;
                    uint _S52 = _S51 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S53 = (*((&kernelContext_0)->table_0+_S52) * p_1->kvHeads_0 + _S2) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S54 = _S51 % (&kernelContext_0)->p_0->pageSize_0;
                    e_0 = (_S53 + _S54) * d2_0 + pair_0;
                }
                e_0 = *((&kernelContext_0)->v_0+e_0);
            }
            else
            {
                e_0 = 0U;
            }
            vw_0[0U] = e_0;
            uint j_1 = kw_4 + 4U;
            if(j_1 < _S6)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    w_2 = (_S2 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_1)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S55 = kStart_0 + j_1;
                    uint _S56 = _S55 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S57 = (*((&kernelContext_0)->table_0+_S56) * p_1->kvHeads_0 + _S2) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S58 = _S55 % (&kernelContext_0)->p_0->pageSize_0;
                    w_2 = (_S57 + _S58) * d2_0 + pair_0;
                }
                w_2 = *((&kernelContext_0)->v_0+w_2);
            }
            else
            {
                w_2 = 0U;
            }
            vw_0[1U] = w_2;
            uint j_2 = kw_4 + 8U;
            if(j_2 < _S6)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    j_0 = (_S2 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_2)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S59 = kStart_0 + j_2;
                    uint _S60 = _S59 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S61 = (*((&kernelContext_0)->table_0+_S60) * p_1->kvHeads_0 + _S2) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S62 = _S59 % (&kernelContext_0)->p_0->pageSize_0;
                    j_0 = (_S61 + _S62) * d2_0 + pair_0;
                }
                j_0 = *((&kernelContext_0)->v_0+j_0);
            }
            else
            {
                j_0 = 0U;
            }
            vw_0[2U] = j_0;
            uint j_3 = kw_4 + 12U;
            if(j_3 < _S6)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    p0_0 = (_S2 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_3)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S63 = kStart_0 + j_3;
                    uint _S64 = _S63 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S65 = (*((&kernelContext_0)->table_0+_S64) * p_1->kvHeads_0 + _S2) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S66 = _S63 % (&kernelContext_0)->p_0->pageSize_0;
                    p0_0 = (_S65 + _S66) * d2_0 + pair_0;
                }
                p0_0 = *((&kernelContext_0)->v_0+p0_0);
            }
            else
            {
                p0_0 = 0U;
            }
            vw_0[3U] = p0_0;
            uint j_4 = kw_4 + 16U;
            if(j_4 < _S6)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_3 = (_S2 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_4)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S67 = kStart_0 + j_4;
                    uint _S68 = _S67 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S69 = (*((&kernelContext_0)->table_0+_S68) * p_1->kvHeads_0 + _S2) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S70 = _S67 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_3 = (_S69 + _S70) * d2_0 + pair_0;
                }
                kw_3 = *((&kernelContext_0)->v_0+kw_3);
            }
            else
            {
                kw_3 = 0U;
            }
            vw_0[4U] = kw_3;
            uint j_5 = kw_4 + 20U;
            if(j_5 < _S6)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_2 = (_S2 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_5)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S71 = kStart_0 + j_5;
                    uint _S72 = _S71 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S73 = (*((&kernelContext_0)->table_0+_S72) * p_1->kvHeads_0 + _S2) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S74 = _S71 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_2 = (_S73 + _S74) * d2_0 + pair_0;
                }
                kw_2 = *((&kernelContext_0)->v_0+kw_2);
            }
            else
            {
                kw_2 = 0U;
            }
            vw_0[5U] = kw_2;
            uint j_6 = kw_4 + 24U;
            if(j_6 < _S6)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_1 = (_S2 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_6)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S75 = kStart_0 + j_6;
                    uint _S76 = _S75 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S77 = (*((&kernelContext_0)->table_0+_S76) * p_1->kvHeads_0 + _S2) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S78 = _S75 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_1 = (_S77 + _S78) * d2_0 + pair_0;
                }
                kw_1 = *((&kernelContext_0)->v_0+kw_1);
            }
            else
            {
                kw_1 = 0U;
            }
            vw_0[6U] = kw_1;
            uint j_7 = kw_4 + 28U;
            if(j_7 < _S6)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_0 = (_S2 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_7)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S79 = kStart_0 + j_7;
                    uint _S80 = _S79 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S81 = (*((&kernelContext_0)->table_0+_S80) * p_1->kvHeads_0 + _S2) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S82 = _S79 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_0 = (_S81 + _S82) * d2_0 + pair_0;
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
                        if(kw_4 >= _S6)
                        {
                            _S28 = int(1);
                            visible_0 = false;
                            break;
                        }
                        float _S83 = attnLo_0(vw_0[0U]);
                        float _S84 = attnHi_0(vw_0[0U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    bool _S85 = 0U >= _S3;
                                    _S1 = _S85;
                                    if(_S85)
                                    {
                                        _S28 = int(4);
                                        break;
                                    }
                                    float pj_0 = (*(&kernelContext_0)->ss_0)[kw_4];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[kw_4], _S83, lo_0[0U]);
                                    hi_0[0U] = fma(pj_0, _S84, hi_0[0U]);
                                    _S28 = int(6);
                                    break;
                                }
                                if(_S28 != int(6))
                                {
                                    _S27 = _S28;
                                    break;
                                }
                                _S27 = int(5);
                                break;
                            }
                            if(_S27 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S3)
                                    {
                                        _S26 = int(4);
                                        break;
                                    }
                                    float pj_1 = (*(&kernelContext_0)->ss_0)[_S7 + kw_4];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S7 + kw_4], _S83, lo_0[1U]);
                                    hi_0[1U] = fma(pj_1, _S84, hi_0[1U]);
                                    _S26 = int(6);
                                    break;
                                }
                                if(_S26 != int(6))
                                {
                                    _S25 = _S26;
                                    break;
                                }
                                _S25 = int(5);
                                break;
                            }
                            if(_S25 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S3)
                                    {
                                        _S24 = int(4);
                                        break;
                                    }
                                    float pj_2 = (*(&kernelContext_0)->ss_0)[2U * _S7 + kw_4];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S7 + kw_4], _S83, lo_0[2U]);
                                    hi_0[2U] = fma(pj_2, _S84, hi_0[2U]);
                                    _S24 = int(6);
                                    break;
                                }
                                if(_S24 != int(6))
                                {
                                    _S23 = _S24;
                                    break;
                                }
                                _S23 = int(5);
                                break;
                            }
                            if(_S23 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S3)
                                    {
                                        _S22 = int(4);
                                        break;
                                    }
                                    float pj_3 = (*(&kernelContext_0)->ss_0)[3U * _S7 + kw_4];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S7 + kw_4], _S83, lo_0[3U]);
                                    hi_0[3U] = fma(pj_3, _S84, hi_0[3U]);
                                    _S22 = int(6);
                                    break;
                                }
                                if(_S22 != int(6))
                                {
                                    _S21 = _S22;
                                    break;
                                }
                                _S21 = int(5);
                                break;
                            }
                            if(_S21 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S3)
                                    {
                                        _S20 = int(4);
                                        break;
                                    }
                                    float pj_4 = (*(&kernelContext_0)->ss_0)[4U * _S7 + kw_4];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S7 + kw_4], _S83, lo_0[4U]);
                                    hi_0[4U] = fma(pj_4, _S84, hi_0[4U]);
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    _S19 = _S20;
                                    break;
                                }
                                _S19 = int(5);
                                break;
                            }
                            if(_S19 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S3)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_5 = (*(&kernelContext_0)->ss_0)[5U * _S7 + kw_4];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S7 + kw_4], _S83, lo_0[5U]);
                                    hi_0[5U] = fma(pj_5, _S84, hi_0[5U]);
                                    _S18 = int(6);
                                    break;
                                }
                                if(_S18 != int(6))
                                {
                                    _S17 = _S18;
                                    break;
                                }
                                _S17 = int(5);
                                break;
                            }
                            if(_S17 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S3)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_6 = (*(&kernelContext_0)->ss_0)[6U * _S7 + kw_4];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S7 + kw_4], _S83, lo_0[6U]);
                                    hi_0[6U] = fma(pj_6, _S84, hi_0[6U]);
                                    _S16 = int(6);
                                    break;
                                }
                                if(_S16 != int(6))
                                {
                                    _S15 = _S16;
                                    break;
                                }
                                _S15 = int(5);
                                break;
                            }
                            if(_S15 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S3)
                                    {
                                        _S14 = int(4);
                                        break;
                                    }
                                    float pj_7 = (*(&kernelContext_0)->ss_0)[7U * _S7 + kw_4];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S7 + kw_4], _S83, lo_0[7U]);
                                    hi_0[7U] = fma(pj_7, _S84, hi_0[7U]);
                                    _S14 = int(6);
                                    break;
                                }
                                if(_S14 != int(6))
                                {
                                    _S13 = _S14;
                                    break;
                                }
                                _S13 = int(5);
                                break;
                            }
                            if(_S13 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S28 = int(3);
                        visible_0 = _S1;
                        break;
                    }
                    if(_S28 != int(3))
                    {
                        _S27 = _S28;
                        break;
                    }
                    _S27 = int(2);
                    break;
                }
                if(_S27 != int(2))
                {
                    break;
                }
                int _S86;
                int _S87;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_1 >= _S6)
                        {
                            _S26 = int(1);
                            break;
                        }
                        float _S88 = attnLo_0(vw_0[1U]);
                        float _S89 = attnHi_0(vw_0[1U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S26 = int(4);
                                        break;
                                    }
                                    float pj_8 = (*(&kernelContext_0)->ss_0)[j_1];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_1], _S88, lo_0[0U]);
                                    hi_0[0U] = fma(pj_8, _S89, hi_0[0U]);
                                    _S26 = int(6);
                                    break;
                                }
                                if(_S26 != int(6))
                                {
                                    _S25 = _S26;
                                    break;
                                }
                                _S25 = int(5);
                                break;
                            }
                            if(_S25 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S3)
                                    {
                                        _S24 = int(4);
                                        break;
                                    }
                                    float pj_9 = (*(&kernelContext_0)->ss_0)[_S7 + j_1];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S7 + j_1], _S88, lo_0[1U]);
                                    hi_0[1U] = fma(pj_9, _S89, hi_0[1U]);
                                    _S24 = int(6);
                                    break;
                                }
                                if(_S24 != int(6))
                                {
                                    _S23 = _S24;
                                    break;
                                }
                                _S23 = int(5);
                                break;
                            }
                            if(_S23 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S3)
                                    {
                                        _S22 = int(4);
                                        break;
                                    }
                                    float pj_10 = (*(&kernelContext_0)->ss_0)[2U * _S7 + j_1];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S7 + j_1], _S88, lo_0[2U]);
                                    hi_0[2U] = fma(pj_10, _S89, hi_0[2U]);
                                    _S22 = int(6);
                                    break;
                                }
                                if(_S22 != int(6))
                                {
                                    _S21 = _S22;
                                    break;
                                }
                                _S21 = int(5);
                                break;
                            }
                            if(_S21 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S3)
                                    {
                                        _S20 = int(4);
                                        break;
                                    }
                                    float pj_11 = (*(&kernelContext_0)->ss_0)[3U * _S7 + j_1];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S7 + j_1], _S88, lo_0[3U]);
                                    hi_0[3U] = fma(pj_11, _S89, hi_0[3U]);
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    _S19 = _S20;
                                    break;
                                }
                                _S19 = int(5);
                                break;
                            }
                            if(_S19 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S3)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_12 = (*(&kernelContext_0)->ss_0)[4U * _S7 + j_1];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S7 + j_1], _S88, lo_0[4U]);
                                    hi_0[4U] = fma(pj_12, _S89, hi_0[4U]);
                                    _S18 = int(6);
                                    break;
                                }
                                if(_S18 != int(6))
                                {
                                    _S17 = _S18;
                                    break;
                                }
                                _S17 = int(5);
                                break;
                            }
                            if(_S17 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S3)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_13 = (*(&kernelContext_0)->ss_0)[5U * _S7 + j_1];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S7 + j_1], _S88, lo_0[5U]);
                                    hi_0[5U] = fma(pj_13, _S89, hi_0[5U]);
                                    _S16 = int(6);
                                    break;
                                }
                                if(_S16 != int(6))
                                {
                                    _S15 = _S16;
                                    break;
                                }
                                _S15 = int(5);
                                break;
                            }
                            if(_S15 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S3)
                                    {
                                        _S14 = int(4);
                                        break;
                                    }
                                    float pj_14 = (*(&kernelContext_0)->ss_0)[6U * _S7 + j_1];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S7 + j_1], _S88, lo_0[6U]);
                                    hi_0[6U] = fma(pj_14, _S89, hi_0[6U]);
                                    _S14 = int(6);
                                    break;
                                }
                                if(_S14 != int(6))
                                {
                                    _S13 = _S14;
                                    break;
                                }
                                _S13 = int(5);
                                break;
                            }
                            if(_S13 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S3)
                                    {
                                        _S87 = int(4);
                                        break;
                                    }
                                    float pj_15 = (*(&kernelContext_0)->ss_0)[7U * _S7 + j_1];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S7 + j_1], _S88, lo_0[7U]);
                                    hi_0[7U] = fma(pj_15, _S89, hi_0[7U]);
                                    _S87 = int(6);
                                    break;
                                }
                                if(_S87 != int(6))
                                {
                                    _S86 = _S87;
                                    break;
                                }
                                _S86 = int(5);
                                break;
                            }
                            if(_S86 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S26 = int(3);
                        break;
                    }
                    if(_S26 != int(3))
                    {
                        _S25 = _S26;
                        break;
                    }
                    _S25 = int(2);
                    break;
                }
                if(_S25 != int(2))
                {
                    break;
                }
                int _S90;
                int _S91;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_2 >= _S6)
                        {
                            _S24 = int(1);
                            break;
                        }
                        float _S92 = attnLo_0(vw_0[2U]);
                        float _S93 = attnHi_0(vw_0[2U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S24 = int(4);
                                        break;
                                    }
                                    float pj_16 = (*(&kernelContext_0)->ss_0)[j_2];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_2], _S92, lo_0[0U]);
                                    hi_0[0U] = fma(pj_16, _S93, hi_0[0U]);
                                    _S24 = int(6);
                                    break;
                                }
                                if(_S24 != int(6))
                                {
                                    _S23 = _S24;
                                    break;
                                }
                                _S23 = int(5);
                                break;
                            }
                            if(_S23 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S3)
                                    {
                                        _S22 = int(4);
                                        break;
                                    }
                                    float pj_17 = (*(&kernelContext_0)->ss_0)[_S7 + j_2];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S7 + j_2], _S92, lo_0[1U]);
                                    hi_0[1U] = fma(pj_17, _S93, hi_0[1U]);
                                    _S22 = int(6);
                                    break;
                                }
                                if(_S22 != int(6))
                                {
                                    _S21 = _S22;
                                    break;
                                }
                                _S21 = int(5);
                                break;
                            }
                            if(_S21 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S3)
                                    {
                                        _S20 = int(4);
                                        break;
                                    }
                                    float pj_18 = (*(&kernelContext_0)->ss_0)[2U * _S7 + j_2];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S7 + j_2], _S92, lo_0[2U]);
                                    hi_0[2U] = fma(pj_18, _S93, hi_0[2U]);
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    _S19 = _S20;
                                    break;
                                }
                                _S19 = int(5);
                                break;
                            }
                            if(_S19 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S3)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_19 = (*(&kernelContext_0)->ss_0)[3U * _S7 + j_2];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S7 + j_2], _S92, lo_0[3U]);
                                    hi_0[3U] = fma(pj_19, _S93, hi_0[3U]);
                                    _S18 = int(6);
                                    break;
                                }
                                if(_S18 != int(6))
                                {
                                    _S17 = _S18;
                                    break;
                                }
                                _S17 = int(5);
                                break;
                            }
                            if(_S17 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S3)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_20 = (*(&kernelContext_0)->ss_0)[4U * _S7 + j_2];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S7 + j_2], _S92, lo_0[4U]);
                                    hi_0[4U] = fma(pj_20, _S93, hi_0[4U]);
                                    _S16 = int(6);
                                    break;
                                }
                                if(_S16 != int(6))
                                {
                                    _S15 = _S16;
                                    break;
                                }
                                _S15 = int(5);
                                break;
                            }
                            if(_S15 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S3)
                                    {
                                        _S14 = int(4);
                                        break;
                                    }
                                    float pj_21 = (*(&kernelContext_0)->ss_0)[5U * _S7 + j_2];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S7 + j_2], _S92, lo_0[5U]);
                                    hi_0[5U] = fma(pj_21, _S93, hi_0[5U]);
                                    _S14 = int(6);
                                    break;
                                }
                                if(_S14 != int(6))
                                {
                                    _S13 = _S14;
                                    break;
                                }
                                _S13 = int(5);
                                break;
                            }
                            if(_S13 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S3)
                                    {
                                        _S87 = int(4);
                                        break;
                                    }
                                    float pj_22 = (*(&kernelContext_0)->ss_0)[6U * _S7 + j_2];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S7 + j_2], _S92, lo_0[6U]);
                                    hi_0[6U] = fma(pj_22, _S93, hi_0[6U]);
                                    _S87 = int(6);
                                    break;
                                }
                                if(_S87 != int(6))
                                {
                                    _S86 = _S87;
                                    break;
                                }
                                _S86 = int(5);
                                break;
                            }
                            if(_S86 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S3)
                                    {
                                        _S91 = int(4);
                                        break;
                                    }
                                    float pj_23 = (*(&kernelContext_0)->ss_0)[7U * _S7 + j_2];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S7 + j_2], _S92, lo_0[7U]);
                                    hi_0[7U] = fma(pj_23, _S93, hi_0[7U]);
                                    _S91 = int(6);
                                    break;
                                }
                                if(_S91 != int(6))
                                {
                                    _S90 = _S91;
                                    break;
                                }
                                _S90 = int(5);
                                break;
                            }
                            if(_S90 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S24 = int(3);
                        break;
                    }
                    if(_S24 != int(3))
                    {
                        _S23 = _S24;
                        break;
                    }
                    _S23 = int(2);
                    break;
                }
                if(_S23 != int(2))
                {
                    break;
                }
                int _S94;
                int _S95;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_3 >= _S6)
                        {
                            _S22 = int(1);
                            break;
                        }
                        float _S96 = attnLo_0(vw_0[3U]);
                        float _S97 = attnHi_0(vw_0[3U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S22 = int(4);
                                        break;
                                    }
                                    float pj_24 = (*(&kernelContext_0)->ss_0)[j_3];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_3], _S96, lo_0[0U]);
                                    hi_0[0U] = fma(pj_24, _S97, hi_0[0U]);
                                    _S22 = int(6);
                                    break;
                                }
                                if(_S22 != int(6))
                                {
                                    _S21 = _S22;
                                    break;
                                }
                                _S21 = int(5);
                                break;
                            }
                            if(_S21 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S3)
                                    {
                                        _S20 = int(4);
                                        break;
                                    }
                                    float pj_25 = (*(&kernelContext_0)->ss_0)[_S7 + j_3];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S7 + j_3], _S96, lo_0[1U]);
                                    hi_0[1U] = fma(pj_25, _S97, hi_0[1U]);
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    _S19 = _S20;
                                    break;
                                }
                                _S19 = int(5);
                                break;
                            }
                            if(_S19 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S3)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_26 = (*(&kernelContext_0)->ss_0)[2U * _S7 + j_3];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S7 + j_3], _S96, lo_0[2U]);
                                    hi_0[2U] = fma(pj_26, _S97, hi_0[2U]);
                                    _S18 = int(6);
                                    break;
                                }
                                if(_S18 != int(6))
                                {
                                    _S17 = _S18;
                                    break;
                                }
                                _S17 = int(5);
                                break;
                            }
                            if(_S17 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S3)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_27 = (*(&kernelContext_0)->ss_0)[3U * _S7 + j_3];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S7 + j_3], _S96, lo_0[3U]);
                                    hi_0[3U] = fma(pj_27, _S97, hi_0[3U]);
                                    _S16 = int(6);
                                    break;
                                }
                                if(_S16 != int(6))
                                {
                                    _S15 = _S16;
                                    break;
                                }
                                _S15 = int(5);
                                break;
                            }
                            if(_S15 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S3)
                                    {
                                        _S14 = int(4);
                                        break;
                                    }
                                    float pj_28 = (*(&kernelContext_0)->ss_0)[4U * _S7 + j_3];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S7 + j_3], _S96, lo_0[4U]);
                                    hi_0[4U] = fma(pj_28, _S97, hi_0[4U]);
                                    _S14 = int(6);
                                    break;
                                }
                                if(_S14 != int(6))
                                {
                                    _S13 = _S14;
                                    break;
                                }
                                _S13 = int(5);
                                break;
                            }
                            if(_S13 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S3)
                                    {
                                        _S87 = int(4);
                                        break;
                                    }
                                    float pj_29 = (*(&kernelContext_0)->ss_0)[5U * _S7 + j_3];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S7 + j_3], _S96, lo_0[5U]);
                                    hi_0[5U] = fma(pj_29, _S97, hi_0[5U]);
                                    _S87 = int(6);
                                    break;
                                }
                                if(_S87 != int(6))
                                {
                                    _S86 = _S87;
                                    break;
                                }
                                _S86 = int(5);
                                break;
                            }
                            if(_S86 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S3)
                                    {
                                        _S91 = int(4);
                                        break;
                                    }
                                    float pj_30 = (*(&kernelContext_0)->ss_0)[6U * _S7 + j_3];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S7 + j_3], _S96, lo_0[6U]);
                                    hi_0[6U] = fma(pj_30, _S97, hi_0[6U]);
                                    _S91 = int(6);
                                    break;
                                }
                                if(_S91 != int(6))
                                {
                                    _S90 = _S91;
                                    break;
                                }
                                _S90 = int(5);
                                break;
                            }
                            if(_S90 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S3)
                                    {
                                        _S95 = int(4);
                                        break;
                                    }
                                    float pj_31 = (*(&kernelContext_0)->ss_0)[7U * _S7 + j_3];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S7 + j_3], _S96, lo_0[7U]);
                                    hi_0[7U] = fma(pj_31, _S97, hi_0[7U]);
                                    _S95 = int(6);
                                    break;
                                }
                                if(_S95 != int(6))
                                {
                                    _S94 = _S95;
                                    break;
                                }
                                _S94 = int(5);
                                break;
                            }
                            if(_S94 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S22 = int(3);
                        break;
                    }
                    if(_S22 != int(3))
                    {
                        _S21 = _S22;
                        break;
                    }
                    _S21 = int(2);
                    break;
                }
                if(_S21 != int(2))
                {
                    break;
                }
                int _S98;
                int _S99;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_4 >= _S6)
                        {
                            _S20 = int(1);
                            break;
                        }
                        float _S100 = attnLo_0(vw_0[4U]);
                        float _S101 = attnHi_0(vw_0[4U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S20 = int(4);
                                        break;
                                    }
                                    float pj_32 = (*(&kernelContext_0)->ss_0)[j_4];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_4], _S100, lo_0[0U]);
                                    hi_0[0U] = fma(pj_32, _S101, hi_0[0U]);
                                    _S20 = int(6);
                                    break;
                                }
                                if(_S20 != int(6))
                                {
                                    _S19 = _S20;
                                    break;
                                }
                                _S19 = int(5);
                                break;
                            }
                            if(_S19 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S3)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_33 = (*(&kernelContext_0)->ss_0)[_S7 + j_4];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S7 + j_4], _S100, lo_0[1U]);
                                    hi_0[1U] = fma(pj_33, _S101, hi_0[1U]);
                                    _S18 = int(6);
                                    break;
                                }
                                if(_S18 != int(6))
                                {
                                    _S17 = _S18;
                                    break;
                                }
                                _S17 = int(5);
                                break;
                            }
                            if(_S17 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S3)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_34 = (*(&kernelContext_0)->ss_0)[2U * _S7 + j_4];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S7 + j_4], _S100, lo_0[2U]);
                                    hi_0[2U] = fma(pj_34, _S101, hi_0[2U]);
                                    _S16 = int(6);
                                    break;
                                }
                                if(_S16 != int(6))
                                {
                                    _S15 = _S16;
                                    break;
                                }
                                _S15 = int(5);
                                break;
                            }
                            if(_S15 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S3)
                                    {
                                        _S14 = int(4);
                                        break;
                                    }
                                    float pj_35 = (*(&kernelContext_0)->ss_0)[3U * _S7 + j_4];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S7 + j_4], _S100, lo_0[3U]);
                                    hi_0[3U] = fma(pj_35, _S101, hi_0[3U]);
                                    _S14 = int(6);
                                    break;
                                }
                                if(_S14 != int(6))
                                {
                                    _S13 = _S14;
                                    break;
                                }
                                _S13 = int(5);
                                break;
                            }
                            if(_S13 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S3)
                                    {
                                        _S87 = int(4);
                                        break;
                                    }
                                    float pj_36 = (*(&kernelContext_0)->ss_0)[4U * _S7 + j_4];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S7 + j_4], _S100, lo_0[4U]);
                                    hi_0[4U] = fma(pj_36, _S101, hi_0[4U]);
                                    _S87 = int(6);
                                    break;
                                }
                                if(_S87 != int(6))
                                {
                                    _S86 = _S87;
                                    break;
                                }
                                _S86 = int(5);
                                break;
                            }
                            if(_S86 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S3)
                                    {
                                        _S91 = int(4);
                                        break;
                                    }
                                    float pj_37 = (*(&kernelContext_0)->ss_0)[5U * _S7 + j_4];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S7 + j_4], _S100, lo_0[5U]);
                                    hi_0[5U] = fma(pj_37, _S101, hi_0[5U]);
                                    _S91 = int(6);
                                    break;
                                }
                                if(_S91 != int(6))
                                {
                                    _S90 = _S91;
                                    break;
                                }
                                _S90 = int(5);
                                break;
                            }
                            if(_S90 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S3)
                                    {
                                        _S95 = int(4);
                                        break;
                                    }
                                    float pj_38 = (*(&kernelContext_0)->ss_0)[6U * _S7 + j_4];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S7 + j_4], _S100, lo_0[6U]);
                                    hi_0[6U] = fma(pj_38, _S101, hi_0[6U]);
                                    _S95 = int(6);
                                    break;
                                }
                                if(_S95 != int(6))
                                {
                                    _S94 = _S95;
                                    break;
                                }
                                _S94 = int(5);
                                break;
                            }
                            if(_S94 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S3)
                                    {
                                        _S99 = int(4);
                                        break;
                                    }
                                    float pj_39 = (*(&kernelContext_0)->ss_0)[7U * _S7 + j_4];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S7 + j_4], _S100, lo_0[7U]);
                                    hi_0[7U] = fma(pj_39, _S101, hi_0[7U]);
                                    _S99 = int(6);
                                    break;
                                }
                                if(_S99 != int(6))
                                {
                                    _S98 = _S99;
                                    break;
                                }
                                _S98 = int(5);
                                break;
                            }
                            if(_S98 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S20 = int(3);
                        break;
                    }
                    if(_S20 != int(3))
                    {
                        _S19 = _S20;
                        break;
                    }
                    _S19 = int(2);
                    break;
                }
                if(_S19 != int(2))
                {
                    break;
                }
                int _S102;
                int _S103;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_5 >= _S6)
                        {
                            _S18 = int(1);
                            break;
                        }
                        float _S104 = attnLo_0(vw_0[5U]);
                        float _S105 = attnHi_0(vw_0[5U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_40 = (*(&kernelContext_0)->ss_0)[j_5];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_5], _S104, lo_0[0U]);
                                    hi_0[0U] = fma(pj_40, _S105, hi_0[0U]);
                                    _S18 = int(6);
                                    break;
                                }
                                if(_S18 != int(6))
                                {
                                    _S17 = _S18;
                                    break;
                                }
                                _S17 = int(5);
                                break;
                            }
                            if(_S17 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S3)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_41 = (*(&kernelContext_0)->ss_0)[_S7 + j_5];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S7 + j_5], _S104, lo_0[1U]);
                                    hi_0[1U] = fma(pj_41, _S105, hi_0[1U]);
                                    _S16 = int(6);
                                    break;
                                }
                                if(_S16 != int(6))
                                {
                                    _S15 = _S16;
                                    break;
                                }
                                _S15 = int(5);
                                break;
                            }
                            if(_S15 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S3)
                                    {
                                        _S14 = int(4);
                                        break;
                                    }
                                    float pj_42 = (*(&kernelContext_0)->ss_0)[2U * _S7 + j_5];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S7 + j_5], _S104, lo_0[2U]);
                                    hi_0[2U] = fma(pj_42, _S105, hi_0[2U]);
                                    _S14 = int(6);
                                    break;
                                }
                                if(_S14 != int(6))
                                {
                                    _S13 = _S14;
                                    break;
                                }
                                _S13 = int(5);
                                break;
                            }
                            if(_S13 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S3)
                                    {
                                        _S87 = int(4);
                                        break;
                                    }
                                    float pj_43 = (*(&kernelContext_0)->ss_0)[3U * _S7 + j_5];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S7 + j_5], _S104, lo_0[3U]);
                                    hi_0[3U] = fma(pj_43, _S105, hi_0[3U]);
                                    _S87 = int(6);
                                    break;
                                }
                                if(_S87 != int(6))
                                {
                                    _S86 = _S87;
                                    break;
                                }
                                _S86 = int(5);
                                break;
                            }
                            if(_S86 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S3)
                                    {
                                        _S91 = int(4);
                                        break;
                                    }
                                    float pj_44 = (*(&kernelContext_0)->ss_0)[4U * _S7 + j_5];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S7 + j_5], _S104, lo_0[4U]);
                                    hi_0[4U] = fma(pj_44, _S105, hi_0[4U]);
                                    _S91 = int(6);
                                    break;
                                }
                                if(_S91 != int(6))
                                {
                                    _S90 = _S91;
                                    break;
                                }
                                _S90 = int(5);
                                break;
                            }
                            if(_S90 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S3)
                                    {
                                        _S95 = int(4);
                                        break;
                                    }
                                    float pj_45 = (*(&kernelContext_0)->ss_0)[5U * _S7 + j_5];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S7 + j_5], _S104, lo_0[5U]);
                                    hi_0[5U] = fma(pj_45, _S105, hi_0[5U]);
                                    _S95 = int(6);
                                    break;
                                }
                                if(_S95 != int(6))
                                {
                                    _S94 = _S95;
                                    break;
                                }
                                _S94 = int(5);
                                break;
                            }
                            if(_S94 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S3)
                                    {
                                        _S99 = int(4);
                                        break;
                                    }
                                    float pj_46 = (*(&kernelContext_0)->ss_0)[6U * _S7 + j_5];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S7 + j_5], _S104, lo_0[6U]);
                                    hi_0[6U] = fma(pj_46, _S105, hi_0[6U]);
                                    _S99 = int(6);
                                    break;
                                }
                                if(_S99 != int(6))
                                {
                                    _S98 = _S99;
                                    break;
                                }
                                _S98 = int(5);
                                break;
                            }
                            if(_S98 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S3)
                                    {
                                        _S103 = int(4);
                                        break;
                                    }
                                    float pj_47 = (*(&kernelContext_0)->ss_0)[7U * _S7 + j_5];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S7 + j_5], _S104, lo_0[7U]);
                                    hi_0[7U] = fma(pj_47, _S105, hi_0[7U]);
                                    _S103 = int(6);
                                    break;
                                }
                                if(_S103 != int(6))
                                {
                                    _S102 = _S103;
                                    break;
                                }
                                _S102 = int(5);
                                break;
                            }
                            if(_S102 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S18 = int(3);
                        break;
                    }
                    if(_S18 != int(3))
                    {
                        _S17 = _S18;
                        break;
                    }
                    _S17 = int(2);
                    break;
                }
                if(_S17 != int(2))
                {
                    break;
                }
                int _S106;
                int _S107;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_6 >= _S6)
                        {
                            _S16 = int(1);
                            break;
                        }
                        float _S108 = attnLo_0(vw_0[6U]);
                        float _S109 = attnHi_0(vw_0[6U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_48 = (*(&kernelContext_0)->ss_0)[j_6];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_6], _S108, lo_0[0U]);
                                    hi_0[0U] = fma(pj_48, _S109, hi_0[0U]);
                                    _S16 = int(6);
                                    break;
                                }
                                if(_S16 != int(6))
                                {
                                    _S15 = _S16;
                                    break;
                                }
                                _S15 = int(5);
                                break;
                            }
                            if(_S15 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S3)
                                    {
                                        _S14 = int(4);
                                        break;
                                    }
                                    float pj_49 = (*(&kernelContext_0)->ss_0)[_S7 + j_6];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S7 + j_6], _S108, lo_0[1U]);
                                    hi_0[1U] = fma(pj_49, _S109, hi_0[1U]);
                                    _S14 = int(6);
                                    break;
                                }
                                if(_S14 != int(6))
                                {
                                    _S13 = _S14;
                                    break;
                                }
                                _S13 = int(5);
                                break;
                            }
                            if(_S13 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S3)
                                    {
                                        _S87 = int(4);
                                        break;
                                    }
                                    float pj_50 = (*(&kernelContext_0)->ss_0)[2U * _S7 + j_6];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S7 + j_6], _S108, lo_0[2U]);
                                    hi_0[2U] = fma(pj_50, _S109, hi_0[2U]);
                                    _S87 = int(6);
                                    break;
                                }
                                if(_S87 != int(6))
                                {
                                    _S86 = _S87;
                                    break;
                                }
                                _S86 = int(5);
                                break;
                            }
                            if(_S86 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S3)
                                    {
                                        _S91 = int(4);
                                        break;
                                    }
                                    float pj_51 = (*(&kernelContext_0)->ss_0)[3U * _S7 + j_6];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S7 + j_6], _S108, lo_0[3U]);
                                    hi_0[3U] = fma(pj_51, _S109, hi_0[3U]);
                                    _S91 = int(6);
                                    break;
                                }
                                if(_S91 != int(6))
                                {
                                    _S90 = _S91;
                                    break;
                                }
                                _S90 = int(5);
                                break;
                            }
                            if(_S90 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S3)
                                    {
                                        _S95 = int(4);
                                        break;
                                    }
                                    float pj_52 = (*(&kernelContext_0)->ss_0)[4U * _S7 + j_6];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S7 + j_6], _S108, lo_0[4U]);
                                    hi_0[4U] = fma(pj_52, _S109, hi_0[4U]);
                                    _S95 = int(6);
                                    break;
                                }
                                if(_S95 != int(6))
                                {
                                    _S94 = _S95;
                                    break;
                                }
                                _S94 = int(5);
                                break;
                            }
                            if(_S94 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S3)
                                    {
                                        _S99 = int(4);
                                        break;
                                    }
                                    float pj_53 = (*(&kernelContext_0)->ss_0)[5U * _S7 + j_6];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S7 + j_6], _S108, lo_0[5U]);
                                    hi_0[5U] = fma(pj_53, _S109, hi_0[5U]);
                                    _S99 = int(6);
                                    break;
                                }
                                if(_S99 != int(6))
                                {
                                    _S98 = _S99;
                                    break;
                                }
                                _S98 = int(5);
                                break;
                            }
                            if(_S98 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S3)
                                    {
                                        _S103 = int(4);
                                        break;
                                    }
                                    float pj_54 = (*(&kernelContext_0)->ss_0)[6U * _S7 + j_6];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S7 + j_6], _S108, lo_0[6U]);
                                    hi_0[6U] = fma(pj_54, _S109, hi_0[6U]);
                                    _S103 = int(6);
                                    break;
                                }
                                if(_S103 != int(6))
                                {
                                    _S102 = _S103;
                                    break;
                                }
                                _S102 = int(5);
                                break;
                            }
                            if(_S102 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S3)
                                    {
                                        _S107 = int(4);
                                        break;
                                    }
                                    float pj_55 = (*(&kernelContext_0)->ss_0)[7U * _S7 + j_6];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S7 + j_6], _S108, lo_0[7U]);
                                    hi_0[7U] = fma(pj_55, _S109, hi_0[7U]);
                                    _S107 = int(6);
                                    break;
                                }
                                if(_S107 != int(6))
                                {
                                    _S106 = _S107;
                                    break;
                                }
                                _S106 = int(5);
                                break;
                            }
                            if(_S106 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S16 = int(3);
                        break;
                    }
                    if(_S16 != int(3))
                    {
                        _S15 = _S16;
                        break;
                    }
                    _S15 = int(2);
                    break;
                }
                if(_S15 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(j_7 >= _S6)
                        {
                            _S14 = int(1);
                            break;
                        }
                        float _S110 = attnLo_0(vw_0[7U]);
                        float _S111 = attnHi_0(vw_0[7U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S14 = int(4);
                                        break;
                                    }
                                    float pj_56 = (*(&kernelContext_0)->ss_0)[j_7];
                                    lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_7], _S110, lo_0[0U]);
                                    hi_0[0U] = fma(pj_56, _S111, hi_0[0U]);
                                    _S14 = int(6);
                                    break;
                                }
                                if(_S14 != int(6))
                                {
                                    _S13 = _S14;
                                    break;
                                }
                                _S13 = int(5);
                                break;
                            }
                            if(_S13 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S3)
                                    {
                                        _S87 = int(4);
                                        break;
                                    }
                                    float pj_57 = (*(&kernelContext_0)->ss_0)[_S7 + j_7];
                                    lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S7 + j_7], _S110, lo_0[1U]);
                                    hi_0[1U] = fma(pj_57, _S111, hi_0[1U]);
                                    _S87 = int(6);
                                    break;
                                }
                                if(_S87 != int(6))
                                {
                                    _S86 = _S87;
                                    break;
                                }
                                _S86 = int(5);
                                break;
                            }
                            if(_S86 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S3)
                                    {
                                        _S91 = int(4);
                                        break;
                                    }
                                    float pj_58 = (*(&kernelContext_0)->ss_0)[2U * _S7 + j_7];
                                    lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S7 + j_7], _S110, lo_0[2U]);
                                    hi_0[2U] = fma(pj_58, _S111, hi_0[2U]);
                                    _S91 = int(6);
                                    break;
                                }
                                if(_S91 != int(6))
                                {
                                    _S90 = _S91;
                                    break;
                                }
                                _S90 = int(5);
                                break;
                            }
                            if(_S90 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S3)
                                    {
                                        _S95 = int(4);
                                        break;
                                    }
                                    float pj_59 = (*(&kernelContext_0)->ss_0)[3U * _S7 + j_7];
                                    lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S7 + j_7], _S110, lo_0[3U]);
                                    hi_0[3U] = fma(pj_59, _S111, hi_0[3U]);
                                    _S95 = int(6);
                                    break;
                                }
                                if(_S95 != int(6))
                                {
                                    _S94 = _S95;
                                    break;
                                }
                                _S94 = int(5);
                                break;
                            }
                            if(_S94 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S3)
                                    {
                                        _S99 = int(4);
                                        break;
                                    }
                                    float pj_60 = (*(&kernelContext_0)->ss_0)[4U * _S7 + j_7];
                                    lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S7 + j_7], _S110, lo_0[4U]);
                                    hi_0[4U] = fma(pj_60, _S111, hi_0[4U]);
                                    _S99 = int(6);
                                    break;
                                }
                                if(_S99 != int(6))
                                {
                                    _S98 = _S99;
                                    break;
                                }
                                _S98 = int(5);
                                break;
                            }
                            if(_S98 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S3)
                                    {
                                        _S103 = int(4);
                                        break;
                                    }
                                    float pj_61 = (*(&kernelContext_0)->ss_0)[5U * _S7 + j_7];
                                    lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S7 + j_7], _S110, lo_0[5U]);
                                    hi_0[5U] = fma(pj_61, _S111, hi_0[5U]);
                                    _S103 = int(6);
                                    break;
                                }
                                if(_S103 != int(6))
                                {
                                    _S102 = _S103;
                                    break;
                                }
                                _S102 = int(5);
                                break;
                            }
                            if(_S102 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S3)
                                    {
                                        _S107 = int(4);
                                        break;
                                    }
                                    float pj_62 = (*(&kernelContext_0)->ss_0)[6U * _S7 + j_7];
                                    lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S7 + j_7], _S110, lo_0[6U]);
                                    hi_0[6U] = fma(pj_62, _S111, hi_0[6U]);
                                    _S107 = int(6);
                                    break;
                                }
                                if(_S107 != int(6))
                                {
                                    _S106 = _S107;
                                    break;
                                }
                                _S106 = int(5);
                                break;
                            }
                            if(_S106 != int(5))
                            {
                                break;
                            }
                            int _S112;
                            for(;;)
                            {
                                int _S113;
                                for(;;)
                                {
                                    if(7U >= _S3)
                                    {
                                        _S113 = int(4);
                                        break;
                                    }
                                    float pj_63 = (*(&kernelContext_0)->ss_0)[7U * _S7 + j_7];
                                    lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S7 + j_7], _S110, lo_0[7U]);
                                    hi_0[7U] = fma(pj_63, _S111, hi_0[7U]);
                                    _S113 = int(6);
                                    break;
                                }
                                if(_S113 != int(6))
                                {
                                    _S112 = _S113;
                                    break;
                                }
                                _S112 = int(5);
                                break;
                            }
                            if(_S112 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S14 = int(3);
                        break;
                    }
                    if(_S14 != int(3))
                    {
                        _S13 = _S14;
                        break;
                    }
                    _S13 = int(2);
                    break;
                }
                if(_S13 != int(2))
                {
                    break;
                }
                break;
            }
            kw_4 = kw_4 + 32U;
        }
    }
    p0_0 = 0U;
    for(;;)
    {
        if(p0_0 < _S3)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(0U == p0_0)
        {
            m_0 = lo_0[0U];
            sum_0 = hi_0[0U];
        }
        else
        {
            m_0 = 0.0;
            sum_0 = 0.0;
        }
        float b_0;
        if(1U == p0_0)
        {
            w_3 = lo_0[1U];
            b_0 = hi_0[1U];
        }
        else
        {
            w_3 = m_0;
            b_0 = sum_0;
        }
        float a_0;
        float b_1;
        if(2U == p0_0)
        {
            a_0 = lo_0[2U];
            b_1 = hi_0[2U];
        }
        else
        {
            a_0 = w_3;
            b_1 = b_0;
        }
        float a_1;
        float b_2;
        if(3U == p0_0)
        {
            a_1 = lo_0[3U];
            b_2 = hi_0[3U];
        }
        else
        {
            a_1 = a_0;
            b_2 = b_1;
        }
        float a_2;
        float b_3;
        if(4U == p0_0)
        {
            a_2 = lo_0[4U];
            b_3 = hi_0[4U];
        }
        else
        {
            a_2 = a_1;
            b_3 = b_2;
        }
        float a_3;
        float b_4;
        if(5U == p0_0)
        {
            a_3 = lo_0[5U];
            b_4 = hi_0[5U];
        }
        else
        {
            a_3 = a_2;
            b_4 = b_3;
        }
        float a_4;
        float b_5;
        if(6U == p0_0)
        {
            a_4 = lo_0[6U];
            b_5 = hi_0[6U];
        }
        else
        {
            a_4 = a_3;
            b_5 = b_4;
        }
        float a_5;
        float b_6;
        if(7U == p0_0)
        {
            a_5 = lo_0[7U];
            b_6 = hi_0[7U];
        }
        else
        {
            a_5 = a_4;
            b_6 = b_5;
        }
        uint _S114 = 2U * pair_0;
        uint _S115 = part_0 * 64U + _S114;
        (*(&kernelContext_0)->ss_0)[_S115] = a_5;
        (*(&kernelContext_0)->ss_0)[_S115 + 1U] = b_6;
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
        uint _S116 = 64U + _S114;
        uint _S117 = 128U + _S114;
        uint _S118 = 192U + _S114;
        float a_6 = (*(&kernelContext_0)->ss_0)[_S114] + (*(&kernelContext_0)->ss_0)[_S116] + (*(&kernelContext_0)->ss_0)[_S117] + (*(&kernelContext_0)->ss_0)[_S118];
        float b_7 = (*(&kernelContext_0)->ss_0)[_S114 + 1U] + (*(&kernelContext_0)->ss_0)[_S116 + 1U] + (*(&kernelContext_0)->ss_0)[_S117 + 1U] + (*(&kernelContext_0)->ss_0)[_S118 + 1U];
        uint _S119 = _S2 * hpk_0;
        uint _S120 = p0_0 / (&kernelContext_0)->p_0->queries_0;
        uint h_1 = _S119 + _S120;
        uint i_9 = p0_0 % (&kernelContext_0)->p_0->queries_0;
        uint row_0 = h_1 * (&kernelContext_0)->p_0->queries_0 + i_9;
        if(((&kernelContext_0)->p_0->splits_0) > 1U)
        {
            uint pr_8 = split_0 * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0 + row_0;
            if(pair_0 == 0U)
            {
                uint _S121 = 2U * pr_8;
                float device* _S122 = (&kernelContext_0)->partialML_0+_S121;
                bool _S123 = _S6 == 0U;
                float _S124;
                if(_S123)
                {
                    _S124 = dg_negative_infinity_0();
                }
                else
                {
                    _S124 = (*(&kernelContext_0)->rowM_0)[p0_0];
                }
                *_S122 = _S124;
                float device* _S125 = (&kernelContext_0)->partialML_0+(_S121 + 1U);
                float _S126;
                if(_S123)
                {
                    _S126 = 0.0;
                }
                else
                {
                    _S126 = (*(&kernelContext_0)->rowL_0)[p0_0];
                }
                *_S125 = _S126;
            }
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S114)) = a_6;
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S114 + 1U)) = b_7;
            p0_0 = p0_0 + 1U;
            continue;
        }
        float l_0 = (*(&kernelContext_0)->rowL_0)[p0_0];
        *((&kernelContext_0)->o_0+(row_0 * (&kernelContext_0)->p_0->headDim_0 + _S114)) = a_6 / (*(&kernelContext_0)->rowL_0)[p0_0];
        *((&kernelContext_0)->o_0+(row_0 * (&kernelContext_0)->p_0->headDim_0 + _S114 + 1U)) = b_7 / l_0;
        p0_0 = p0_0 + 1U;
    }
    return;
}

