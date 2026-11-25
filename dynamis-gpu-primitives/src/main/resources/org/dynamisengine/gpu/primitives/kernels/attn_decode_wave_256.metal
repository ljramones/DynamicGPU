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

uint qF32ToF16_0(float f_0)
{
    uint x_0 = (as_type<uint>((f_0)));
    uint sign_0 = (x_0 >> 16U) & 32768U;
    uint e_0 = (x_0 >> 23U) & 255U;
    uint mant_0 = x_0 & 8388607U;
    uint h_0;
    if(e_0 == 255U)
    {
        uint _S1 = sign_0 | 31744U;
        if(mant_0 != 0U)
        {
            h_0 = 512U | (mant_0 >> 13U);
        }
        else
        {
            h_0 = 0U;
        }
        return _S1 | h_0;
    }
    int exp_0 = int(e_0) - int(112);
    if(exp_0 >= int(31))
    {
        return sign_0 | 31744U;
    }
    bool _S2;
    if(exp_0 <= int(0))
    {
        if(exp_0 < int(-10))
        {
            return sign_0;
        }
        uint mant_1 = mant_0 | 8388608U;
        uint shift_0 = uint(int(14) - exp_0);
        uint h_1 = mant_1 >> shift_0;
        uint rem_0 = mant_1 & ((1U << shift_0) - 1U);
        uint halfway_0 = 1U << (shift_0 - 1U);
        if(rem_0 > halfway_0)
        {
            _S2 = true;
        }
        else
        {
            if(rem_0 == halfway_0)
            {
                _S2 = (h_1 & 1U) != 0U;
            }
            else
            {
                _S2 = false;
            }
        }
        if(_S2)
        {
            h_0 = h_1 + 1U;
        }
        else
        {
            h_0 = h_1;
        }
        return sign_0 | h_0;
    }
    uint h_2 = (uint(exp_0) << 10U) | (mant_0 >> 13U);
    uint rem_1 = mant_0 & 8191U;
    if(rem_1 > 4096U)
    {
        _S2 = true;
    }
    else
    {
        if(rem_1 == 4096U)
        {
            _S2 = (h_2 & 1U) != 0U;
        }
        else
        {
            _S2 = false;
        }
    }
    if(_S2)
    {
        h_0 = h_2 + 1U;
    }
    else
    {
        h_0 = h_2;
    }
    return sign_0 | h_0;
}

uint attnPack_0(float lo_0, float hi_0)
{
    return (qF32ToF16_0(lo_0)) | ((qF32ToF16_0(hi_0)) << 16U);
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
    uint device* o_0;
    uint threadgroup* waves_0;
    array<uint, int(1024)> threadgroup* qs_0;
    array<float, int(1024)> threadgroup* ss_0;
    array<float, int(8)> threadgroup* rowM_0;
    array<float, int(8)> threadgroup* rowL_0;
};

[[kernel]] void attn_decode_wave_256(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(7)]], uint device* q_1 [[buffer(0)]], uint device* table_1 [[buffer(6)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partialML_1 [[buffer(4)]], float device* partialAcc_1 [[buffer(5)]], uint device* o_1 [[buffer(3)]])
{
    bool _S3;
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
    threadgroup array<uint, int(1024)> qs_1;
    (&kernelContext_0)->qs_0 = &qs_1;
    threadgroup array<float, int(1024)> ss_1;
    (&kernelContext_0)->ss_0 = &ss_1;
    threadgroup array<float, int(8)> rowM_1;
    (&kernelContext_0)->rowM_0 = &rowM_1;
    threadgroup array<float, int(8)> rowL_1;
    (&kernelContext_0)->rowL_0 = &rowL_1;
    uint t_0 = local_0.x;
    uint _S4 = group_0.y;
    uint split_0 = group_0.z;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint _S5 = min(hpk_0 * (&kernelContext_0)->p_0->queries_0, ROW_TIER_0);
    uint d2_0 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint _S6 = (&kernelContext_0)->p_0->keys_0 - (&kernelContext_0)->p_0->queries_0;
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S7 = min((&kernelContext_0)->p_0->keys_0, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    uint _S8;
    if(_S7 > kStart_0)
    {
        _S8 = _S7 - kStart_0;
    }
    else
    {
        _S8 = 0U;
    }
    uint _S9 = (&kernelContext_0)->p_0->splitKeys_0;
    if(t_0 == 0U)
    {
        *(&kernelContext_0)->waves_0 = 0U;
    }
    bool visible_0;
    uint w_2;
    uint e_1 = t_0;
    for(;;)
    {
        if(e_1 < (ROW_TIER_0 * 128U))
        {
        }
        else
        {
            break;
        }
        uint r_0 = e_1 / 128U;
        uint pr_0 = e_1 % 128U;
        if(r_0 < _S5)
        {
            visible_0 = pr_0 < d2_0;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S10 = _S4 * hpk_0;
            uint _S11 = r_0 / (&kernelContext_0)->p_0->queries_0;
            uint h_3 = _S10 + _S11;
            uint i_0 = r_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((h_3 * (&kernelContext_0)->p_0->queries_0 + i_0) * d2_0 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_1] = w_2;
        e_1 = e_1 + 128U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S12 = simd_sum(1U);
    uint lane_0 = simd_prefix_exclusive_sum(1U);
    bool _S13 = lane_0 == 0U;
    if(_S13)
    {
        uint _S14 = atomic_fetch_add_explicit(((atomic_uint threadgroup*)((&kernelContext_0)->waves_0)), 1U, memory_order_relaxed);
        w_2 = _S14;
    }
    else
    {
        w_2 = 0U;
    }
    float m_0;
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
    int _S30;
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
        if(j_0 < _S8)
        {
        }
        else
        {
            break;
        }
        uint key_0 = kStart_0 + j_0;
        if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
        {
            e_1 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * d2_0;
        }
        else
        {
            uint _S31 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
            uint _S32 = (*((&kernelContext_0)->table_0+_S31) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
            uint _S33 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
            e_1 = (_S32 + _S33) * d2_0;
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
                kw_4 = *((&kernelContext_0)->k_0+(e_1 + p0_0));
            }
            else
            {
                kw_4 = 0U;
            }
            float _S34 = attnLo_0(kw_4);
            float _S35 = attnHi_0(kw_4);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[p0_0]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[p0_0]), _S34, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S34, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + p0_0]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + p0_0]), _S34, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + p0_0]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + p0_0]), _S34, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + p0_0]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + p0_0]), _S34, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + p0_0]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + p0_0]), _S34, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + p0_0]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + p0_0]), _S34, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + p0_0]), _S35, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + p0_0]), _S34, s_0[7U]));
            }
            uint pr_1 = p0_0 + 1U;
            if(pr_1 < d2_0)
            {
                kw_3 = *((&kernelContext_0)->k_0+(e_1 + pr_1));
            }
            else
            {
                kw_3 = 0U;
            }
            float _S36 = attnLo_0(kw_3);
            float _S37 = attnHi_0(kw_3);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_1]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_1]), _S36, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S36, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_1]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_1]), _S36, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_1]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_1]), _S36, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_1]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_1]), _S36, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_1]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_1]), _S36, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_1]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_1]), _S36, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_1]), _S37, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_1]), _S36, s_0[7U]));
            }
            uint pr_2 = p0_0 + 2U;
            if(pr_2 < d2_0)
            {
                kw_2 = *((&kernelContext_0)->k_0+(e_1 + pr_2));
            }
            else
            {
                kw_2 = 0U;
            }
            float _S38 = attnLo_0(kw_2);
            float _S39 = attnHi_0(kw_2);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_2]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_2]), _S38, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S38, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_2]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_2]), _S38, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_2]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_2]), _S38, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_2]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_2]), _S38, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_2]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_2]), _S38, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_2]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_2]), _S38, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_2]), _S39, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_2]), _S38, s_0[7U]));
            }
            uint pr_3 = p0_0 + 3U;
            if(pr_3 < d2_0)
            {
                kw_1 = *((&kernelContext_0)->k_0+(e_1 + pr_3));
            }
            else
            {
                kw_1 = 0U;
            }
            float _S40 = attnLo_0(kw_1);
            float _S41 = attnHi_0(kw_1);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_3]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_3]), _S40, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S40, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_3]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_3]), _S40, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_3]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_3]), _S40, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_3]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_3]), _S40, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_3]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_3]), _S40, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_3]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_3]), _S40, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_3]), _S41, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_3]), _S40, s_0[7U]));
            }
            uint pr_4 = p0_0 + 4U;
            if(pr_4 < d2_0)
            {
                kw_0 = *((&kernelContext_0)->k_0+(e_1 + pr_4));
            }
            else
            {
                kw_0 = 0U;
            }
            float _S42 = attnLo_0(kw_0);
            float _S43 = attnHi_0(kw_0);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_4]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_4]), _S42, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S42, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_4]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_4]), _S42, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_4]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_4]), _S42, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_4]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_4]), _S42, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_4]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_4]), _S42, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_4]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_4]), _S42, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_4]), _S43, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_4]), _S42, s_0[7U]));
            }
            uint pr_5 = p0_0 + 5U;
            uint kw_5;
            if(pr_5 < d2_0)
            {
                kw_5 = *((&kernelContext_0)->k_0+(e_1 + pr_5));
            }
            else
            {
                kw_5 = 0U;
            }
            float _S44 = attnLo_0(kw_5);
            float _S45 = attnHi_0(kw_5);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_5]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_5]), _S44, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S44, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_5]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_5]), _S44, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_5]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_5]), _S44, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_5]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_5]), _S44, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_5]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_5]), _S44, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_5]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_5]), _S44, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_5]), _S45, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_5]), _S44, s_0[7U]));
            }
            uint pr_6 = p0_0 + 6U;
            uint kw_6;
            if(pr_6 < d2_0)
            {
                kw_6 = *((&kernelContext_0)->k_0+(e_1 + pr_6));
            }
            else
            {
                kw_6 = 0U;
            }
            float _S46 = attnLo_0(kw_6);
            float _S47 = attnHi_0(kw_6);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_6]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_6]), _S46, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S46, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_6]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_6]), _S46, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_6]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_6]), _S46, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_6]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_6]), _S46, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_6]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_6]), _S46, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_6]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_6]), _S46, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_6]), _S47, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_6]), _S46, s_0[7U]));
            }
            uint pr_7 = p0_0 + 7U;
            uint kw_7;
            if(pr_7 < d2_0)
            {
                kw_7 = *((&kernelContext_0)->k_0+(e_1 + pr_7));
            }
            else
            {
                kw_7 = 0U;
            }
            float _S48 = attnLo_0(kw_7);
            float _S49 = attnHi_0(kw_7);
            if(0U < ROW_TIER_0)
            {
                s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_7]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_7]), _S48, s_0[0U]));
            }
            if(1U < ROW_TIER_0)
            {
                s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S48, s_0[1U]));
            }
            if(2U < ROW_TIER_0)
            {
                s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_7]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_7]), _S48, s_0[2U]));
            }
            if(3U < ROW_TIER_0)
            {
                s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_7]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_7]), _S48, s_0[3U]));
            }
            if(4U < ROW_TIER_0)
            {
                s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_7]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_7]), _S48, s_0[4U]));
            }
            if(5U < ROW_TIER_0)
            {
                s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_7]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_7]), _S48, s_0[5U]));
            }
            if(6U < ROW_TIER_0)
            {
                s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_7]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_7]), _S48, s_0[6U]));
            }
            if(7U < ROW_TIER_0)
            {
                s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_7]), _S49, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_7]), _S48, s_0[7U]));
            }
            p0_0 = p0_0 + 8U;
        }
        for(;;)
        {
            for(;;)
            {
                for(;;)
                {
                    if(0U >= _S5)
                    {
                        _S30 = int(1);
                        break;
                    }
                    uint i_1 = 0U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_1 + _S6);
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
                    _S30 = int(3);
                    break;
                }
                if(_S30 != int(3))
                {
                    _S29 = _S30;
                    break;
                }
                _S29 = int(2);
                break;
            }
            if(_S29 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(1U >= _S5)
                    {
                        _S28 = int(1);
                        break;
                    }
                    uint i_2 = 1U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_2 + _S6);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[1U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[_S9 + j_0] = m_0;
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
                    if(2U >= _S5)
                    {
                        _S26 = int(1);
                        break;
                    }
                    uint i_3 = 2U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_3 + _S6);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[2U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[2U * _S9 + j_0] = m_0;
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
                    if(3U >= _S5)
                    {
                        _S24 = int(1);
                        break;
                    }
                    uint i_4 = 3U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_4 + _S6);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[3U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[3U * _S9 + j_0] = m_0;
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
                    if(4U >= _S5)
                    {
                        _S22 = int(1);
                        break;
                    }
                    uint i_5 = 4U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_5 + _S6);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[4U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[4U * _S9 + j_0] = m_0;
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
                    if(5U >= _S5)
                    {
                        _S20 = int(1);
                        break;
                    }
                    uint i_6 = 5U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_6 + _S6);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[5U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[5U * _S9 + j_0] = m_0;
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
                    if(6U >= _S5)
                    {
                        _S18 = int(1);
                        break;
                    }
                    uint i_7 = 6U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_7 + _S6);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[6U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[6U * _S9 + j_0] = m_0;
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
                    if(7U >= _S5)
                    {
                        _S16 = int(1);
                        break;
                    }
                    uint i_8 = 7U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_8 + _S6);
                    }
                    if(visible_0)
                    {
                        m_0 = s_0[7U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        m_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[7U * _S9 + j_0] = m_0;
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
        if(p0_0 < _S5)
        {
        }
        else
        {
            break;
        }
        float _S50 = dg_negative_infinity_0();
        j_0 = lane_0;
        m_0 = _S50;
        for(;;)
        {
            if(j_0 < _S8)
            {
            }
            else
            {
                break;
            }
            float _S51 = max(m_0, (*(&kernelContext_0)->ss_0)[p0_0 * _S9 + j_0]);
            j_0 = j_0 + _S12;
            m_0 = _S51;
        }
        float m_1 = simd_max(m_0);
        kw_4 = lane_0;
        sum_0 = 0.0;
        for(;;)
        {
            if(kw_4 < _S8)
            {
            }
            else
            {
                break;
            }
            float sv_0 = (*(&kernelContext_0)->ss_0)[p0_0 * _S9 + kw_4];
            if(((*(&kernelContext_0)->ss_0)[p0_0 * _S9 + kw_4]) == _S50)
            {
                w_3 = 0.0;
            }
            else
            {
                w_3 = exp(sv_0 - m_1);
            }
            (*(&kernelContext_0)->ss_0)[p0_0 * _S9 + kw_4] = w_3;
            float sum_1 = sum_0 + w_3;
            kw_4 = kw_4 + _S12;
            sum_0 = sum_1;
        }
        float sum_2 = simd_sum(sum_0);
        if(_S13)
        {
            (*(&kernelContext_0)->rowM_0)[p0_0] = m_1;
            (*(&kernelContext_0)->rowL_0)[p0_0] = sum_2;
        }
        p0_0 = p0_0 + *(&kernelContext_0)->waves_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pair_0 = t_0 % 128U;
    uint part_0 = t_0 / 128U;
    thread array<float, int(8)> lo_1;
    thread array<float, int(8)> hi_1;
    lo_1[0U] = 0.0;
    hi_1[0U] = 0.0;
    lo_1[1U] = 0.0;
    hi_1[1U] = 0.0;
    lo_1[2U] = 0.0;
    hi_1[2U] = 0.0;
    lo_1[3U] = 0.0;
    hi_1[3U] = 0.0;
    lo_1[4U] = 0.0;
    hi_1[4U] = 0.0;
    lo_1[5U] = 0.0;
    hi_1[5U] = 0.0;
    lo_1[6U] = 0.0;
    hi_1[6U] = 0.0;
    lo_1[7U] = 0.0;
    hi_1[7U] = 0.0;
    if(pair_0 < d2_0)
    {
        kw_4 = part_0;
        for(;;)
        {
            bool _S52 = kw_4 < _S8;
            if(_S52)
            {
            }
            else
            {
                break;
            }
            thread array<uint, int(8)> vw_0;
            if(_S52)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    e_1 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + kw_4)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S53 = kStart_0 + kw_4;
                    uint _S54 = _S53 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S55 = (*((&kernelContext_0)->table_0+_S54) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S56 = _S53 % (&kernelContext_0)->p_0->pageSize_0;
                    e_1 = (_S55 + _S56) * d2_0 + pair_0;
                }
                e_1 = *((&kernelContext_0)->v_0+e_1);
            }
            else
            {
                e_1 = 0U;
            }
            vw_0[0U] = e_1;
            uint j_1 = kw_4 + 1U;
            if(j_1 < _S8)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    w_2 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_1)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S57 = kStart_0 + j_1;
                    uint _S58 = _S57 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S59 = (*((&kernelContext_0)->table_0+_S58) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S60 = _S57 % (&kernelContext_0)->p_0->pageSize_0;
                    w_2 = (_S59 + _S60) * d2_0 + pair_0;
                }
                w_2 = *((&kernelContext_0)->v_0+w_2);
            }
            else
            {
                w_2 = 0U;
            }
            vw_0[1U] = w_2;
            uint j_2 = kw_4 + 2U;
            if(j_2 < _S8)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    j_0 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_2)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S61 = kStart_0 + j_2;
                    uint _S62 = _S61 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S63 = (*((&kernelContext_0)->table_0+_S62) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S64 = _S61 % (&kernelContext_0)->p_0->pageSize_0;
                    j_0 = (_S63 + _S64) * d2_0 + pair_0;
                }
                j_0 = *((&kernelContext_0)->v_0+j_0);
            }
            else
            {
                j_0 = 0U;
            }
            vw_0[2U] = j_0;
            uint j_3 = kw_4 + 3U;
            if(j_3 < _S8)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    p0_0 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_3)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S65 = kStart_0 + j_3;
                    uint _S66 = _S65 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S67 = (*((&kernelContext_0)->table_0+_S66) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S68 = _S65 % (&kernelContext_0)->p_0->pageSize_0;
                    p0_0 = (_S67 + _S68) * d2_0 + pair_0;
                }
                p0_0 = *((&kernelContext_0)->v_0+p0_0);
            }
            else
            {
                p0_0 = 0U;
            }
            vw_0[3U] = p0_0;
            uint j_4 = kw_4 + 4U;
            if(j_4 < _S8)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_3 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_4)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S69 = kStart_0 + j_4;
                    uint _S70 = _S69 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S71 = (*((&kernelContext_0)->table_0+_S70) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S72 = _S69 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_3 = (_S71 + _S72) * d2_0 + pair_0;
                }
                kw_3 = *((&kernelContext_0)->v_0+kw_3);
            }
            else
            {
                kw_3 = 0U;
            }
            vw_0[4U] = kw_3;
            uint j_5 = kw_4 + 5U;
            if(j_5 < _S8)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_2 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_5)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S73 = kStart_0 + j_5;
                    uint _S74 = _S73 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S75 = (*((&kernelContext_0)->table_0+_S74) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S76 = _S73 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_2 = (_S75 + _S76) * d2_0 + pair_0;
                }
                kw_2 = *((&kernelContext_0)->v_0+kw_2);
            }
            else
            {
                kw_2 = 0U;
            }
            vw_0[5U] = kw_2;
            uint j_6 = kw_4 + 6U;
            if(j_6 < _S8)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_1 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_6)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S77 = kStart_0 + j_6;
                    uint _S78 = _S77 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S79 = (*((&kernelContext_0)->table_0+_S78) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S80 = _S77 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_1 = (_S79 + _S80) * d2_0 + pair_0;
                }
                kw_1 = *((&kernelContext_0)->v_0+kw_1);
            }
            else
            {
                kw_1 = 0U;
            }
            vw_0[6U] = kw_1;
            uint j_7 = kw_4 + 7U;
            if(j_7 < _S8)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_0 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_7)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S81 = kStart_0 + j_7;
                    uint _S82 = _S81 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S83 = (*((&kernelContext_0)->table_0+_S82) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S84 = _S81 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_0 = (_S83 + _S84) * d2_0 + pair_0;
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
                        if(kw_4 >= _S8)
                        {
                            _S30 = int(1);
                            visible_0 = false;
                            break;
                        }
                        float _S85 = attnLo_0(vw_0[0U]);
                        float _S86 = attnHi_0(vw_0[0U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    bool _S87 = 0U >= _S5;
                                    _S3 = _S87;
                                    if(_S87)
                                    {
                                        _S30 = int(4);
                                        break;
                                    }
                                    float pj_0 = (*(&kernelContext_0)->ss_0)[kw_4];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[kw_4], _S85, lo_1[0U]);
                                    hi_1[0U] = fma(pj_0, _S86, hi_1[0U]);
                                    _S30 = int(6);
                                    break;
                                }
                                if(_S30 != int(6))
                                {
                                    _S29 = _S30;
                                    break;
                                }
                                _S29 = int(5);
                                break;
                            }
                            if(_S29 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S5)
                                    {
                                        _S28 = int(4);
                                        break;
                                    }
                                    float pj_1 = (*(&kernelContext_0)->ss_0)[_S9 + kw_4];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S9 + kw_4], _S85, lo_1[1U]);
                                    hi_1[1U] = fma(pj_1, _S86, hi_1[1U]);
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
                                    if(2U >= _S5)
                                    {
                                        _S26 = int(4);
                                        break;
                                    }
                                    float pj_2 = (*(&kernelContext_0)->ss_0)[2U * _S9 + kw_4];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S9 + kw_4], _S85, lo_1[2U]);
                                    hi_1[2U] = fma(pj_2, _S86, hi_1[2U]);
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
                                    if(3U >= _S5)
                                    {
                                        _S24 = int(4);
                                        break;
                                    }
                                    float pj_3 = (*(&kernelContext_0)->ss_0)[3U * _S9 + kw_4];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S9 + kw_4], _S85, lo_1[3U]);
                                    hi_1[3U] = fma(pj_3, _S86, hi_1[3U]);
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
                                    if(4U >= _S5)
                                    {
                                        _S22 = int(4);
                                        break;
                                    }
                                    float pj_4 = (*(&kernelContext_0)->ss_0)[4U * _S9 + kw_4];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S9 + kw_4], _S85, lo_1[4U]);
                                    hi_1[4U] = fma(pj_4, _S86, hi_1[4U]);
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
                                    if(5U >= _S5)
                                    {
                                        _S20 = int(4);
                                        break;
                                    }
                                    float pj_5 = (*(&kernelContext_0)->ss_0)[5U * _S9 + kw_4];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S9 + kw_4], _S85, lo_1[5U]);
                                    hi_1[5U] = fma(pj_5, _S86, hi_1[5U]);
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
                                    if(6U >= _S5)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_6 = (*(&kernelContext_0)->ss_0)[6U * _S9 + kw_4];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S9 + kw_4], _S85, lo_1[6U]);
                                    hi_1[6U] = fma(pj_6, _S86, hi_1[6U]);
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
                                    if(7U >= _S5)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_7 = (*(&kernelContext_0)->ss_0)[7U * _S9 + kw_4];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S9 + kw_4], _S85, lo_1[7U]);
                                    hi_1[7U] = fma(pj_7, _S86, hi_1[7U]);
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
                            break;
                        }
                        _S30 = int(3);
                        visible_0 = _S3;
                        break;
                    }
                    if(_S30 != int(3))
                    {
                        _S29 = _S30;
                        break;
                    }
                    _S29 = int(2);
                    break;
                }
                if(_S29 != int(2))
                {
                    break;
                }
                int _S88;
                int _S89;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_1 >= _S8)
                        {
                            _S28 = int(1);
                            break;
                        }
                        float _S90 = attnLo_0(vw_0[1U]);
                        float _S91 = attnHi_0(vw_0[1U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S28 = int(4);
                                        break;
                                    }
                                    float pj_8 = (*(&kernelContext_0)->ss_0)[j_1];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_1], _S90, lo_1[0U]);
                                    hi_1[0U] = fma(pj_8, _S91, hi_1[0U]);
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
                                    if(1U >= _S5)
                                    {
                                        _S26 = int(4);
                                        break;
                                    }
                                    float pj_9 = (*(&kernelContext_0)->ss_0)[_S9 + j_1];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S9 + j_1], _S90, lo_1[1U]);
                                    hi_1[1U] = fma(pj_9, _S91, hi_1[1U]);
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
                                    if(2U >= _S5)
                                    {
                                        _S24 = int(4);
                                        break;
                                    }
                                    float pj_10 = (*(&kernelContext_0)->ss_0)[2U * _S9 + j_1];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S9 + j_1], _S90, lo_1[2U]);
                                    hi_1[2U] = fma(pj_10, _S91, hi_1[2U]);
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
                                    if(3U >= _S5)
                                    {
                                        _S22 = int(4);
                                        break;
                                    }
                                    float pj_11 = (*(&kernelContext_0)->ss_0)[3U * _S9 + j_1];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S9 + j_1], _S90, lo_1[3U]);
                                    hi_1[3U] = fma(pj_11, _S91, hi_1[3U]);
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
                                    if(4U >= _S5)
                                    {
                                        _S20 = int(4);
                                        break;
                                    }
                                    float pj_12 = (*(&kernelContext_0)->ss_0)[4U * _S9 + j_1];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S9 + j_1], _S90, lo_1[4U]);
                                    hi_1[4U] = fma(pj_12, _S91, hi_1[4U]);
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
                                    if(5U >= _S5)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_13 = (*(&kernelContext_0)->ss_0)[5U * _S9 + j_1];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S9 + j_1], _S90, lo_1[5U]);
                                    hi_1[5U] = fma(pj_13, _S91, hi_1[5U]);
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
                                    if(6U >= _S5)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_14 = (*(&kernelContext_0)->ss_0)[6U * _S9 + j_1];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S9 + j_1], _S90, lo_1[6U]);
                                    hi_1[6U] = fma(pj_14, _S91, hi_1[6U]);
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
                                    if(7U >= _S5)
                                    {
                                        _S89 = int(4);
                                        break;
                                    }
                                    float pj_15 = (*(&kernelContext_0)->ss_0)[7U * _S9 + j_1];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S9 + j_1], _S90, lo_1[7U]);
                                    hi_1[7U] = fma(pj_15, _S91, hi_1[7U]);
                                    _S89 = int(6);
                                    break;
                                }
                                if(_S89 != int(6))
                                {
                                    _S88 = _S89;
                                    break;
                                }
                                _S88 = int(5);
                                break;
                            }
                            if(_S88 != int(5))
                            {
                                break;
                            }
                            break;
                        }
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
                int _S92;
                int _S93;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_2 >= _S8)
                        {
                            _S26 = int(1);
                            break;
                        }
                        float _S94 = attnLo_0(vw_0[2U]);
                        float _S95 = attnHi_0(vw_0[2U]);
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
                                    float pj_16 = (*(&kernelContext_0)->ss_0)[j_2];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_2], _S94, lo_1[0U]);
                                    hi_1[0U] = fma(pj_16, _S95, hi_1[0U]);
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
                                    if(1U >= _S5)
                                    {
                                        _S24 = int(4);
                                        break;
                                    }
                                    float pj_17 = (*(&kernelContext_0)->ss_0)[_S9 + j_2];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S9 + j_2], _S94, lo_1[1U]);
                                    hi_1[1U] = fma(pj_17, _S95, hi_1[1U]);
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
                                    if(2U >= _S5)
                                    {
                                        _S22 = int(4);
                                        break;
                                    }
                                    float pj_18 = (*(&kernelContext_0)->ss_0)[2U * _S9 + j_2];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S9 + j_2], _S94, lo_1[2U]);
                                    hi_1[2U] = fma(pj_18, _S95, hi_1[2U]);
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
                                    if(3U >= _S5)
                                    {
                                        _S20 = int(4);
                                        break;
                                    }
                                    float pj_19 = (*(&kernelContext_0)->ss_0)[3U * _S9 + j_2];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S9 + j_2], _S94, lo_1[3U]);
                                    hi_1[3U] = fma(pj_19, _S95, hi_1[3U]);
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
                                    if(4U >= _S5)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_20 = (*(&kernelContext_0)->ss_0)[4U * _S9 + j_2];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S9 + j_2], _S94, lo_1[4U]);
                                    hi_1[4U] = fma(pj_20, _S95, hi_1[4U]);
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
                                    if(5U >= _S5)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_21 = (*(&kernelContext_0)->ss_0)[5U * _S9 + j_2];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S9 + j_2], _S94, lo_1[5U]);
                                    hi_1[5U] = fma(pj_21, _S95, hi_1[5U]);
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
                                    if(6U >= _S5)
                                    {
                                        _S89 = int(4);
                                        break;
                                    }
                                    float pj_22 = (*(&kernelContext_0)->ss_0)[6U * _S9 + j_2];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S9 + j_2], _S94, lo_1[6U]);
                                    hi_1[6U] = fma(pj_22, _S95, hi_1[6U]);
                                    _S89 = int(6);
                                    break;
                                }
                                if(_S89 != int(6))
                                {
                                    _S88 = _S89;
                                    break;
                                }
                                _S88 = int(5);
                                break;
                            }
                            if(_S88 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S5)
                                    {
                                        _S93 = int(4);
                                        break;
                                    }
                                    float pj_23 = (*(&kernelContext_0)->ss_0)[7U * _S9 + j_2];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S9 + j_2], _S94, lo_1[7U]);
                                    hi_1[7U] = fma(pj_23, _S95, hi_1[7U]);
                                    _S93 = int(6);
                                    break;
                                }
                                if(_S93 != int(6))
                                {
                                    _S92 = _S93;
                                    break;
                                }
                                _S92 = int(5);
                                break;
                            }
                            if(_S92 != int(5))
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
                int _S96;
                int _S97;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_3 >= _S8)
                        {
                            _S24 = int(1);
                            break;
                        }
                        float _S98 = attnLo_0(vw_0[3U]);
                        float _S99 = attnHi_0(vw_0[3U]);
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
                                    float pj_24 = (*(&kernelContext_0)->ss_0)[j_3];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_3], _S98, lo_1[0U]);
                                    hi_1[0U] = fma(pj_24, _S99, hi_1[0U]);
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
                                    if(1U >= _S5)
                                    {
                                        _S22 = int(4);
                                        break;
                                    }
                                    float pj_25 = (*(&kernelContext_0)->ss_0)[_S9 + j_3];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S9 + j_3], _S98, lo_1[1U]);
                                    hi_1[1U] = fma(pj_25, _S99, hi_1[1U]);
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
                                    if(2U >= _S5)
                                    {
                                        _S20 = int(4);
                                        break;
                                    }
                                    float pj_26 = (*(&kernelContext_0)->ss_0)[2U * _S9 + j_3];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S9 + j_3], _S98, lo_1[2U]);
                                    hi_1[2U] = fma(pj_26, _S99, hi_1[2U]);
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
                                    if(3U >= _S5)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_27 = (*(&kernelContext_0)->ss_0)[3U * _S9 + j_3];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S9 + j_3], _S98, lo_1[3U]);
                                    hi_1[3U] = fma(pj_27, _S99, hi_1[3U]);
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
                                    if(4U >= _S5)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_28 = (*(&kernelContext_0)->ss_0)[4U * _S9 + j_3];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S9 + j_3], _S98, lo_1[4U]);
                                    hi_1[4U] = fma(pj_28, _S99, hi_1[4U]);
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
                                    if(5U >= _S5)
                                    {
                                        _S89 = int(4);
                                        break;
                                    }
                                    float pj_29 = (*(&kernelContext_0)->ss_0)[5U * _S9 + j_3];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S9 + j_3], _S98, lo_1[5U]);
                                    hi_1[5U] = fma(pj_29, _S99, hi_1[5U]);
                                    _S89 = int(6);
                                    break;
                                }
                                if(_S89 != int(6))
                                {
                                    _S88 = _S89;
                                    break;
                                }
                                _S88 = int(5);
                                break;
                            }
                            if(_S88 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S5)
                                    {
                                        _S93 = int(4);
                                        break;
                                    }
                                    float pj_30 = (*(&kernelContext_0)->ss_0)[6U * _S9 + j_3];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S9 + j_3], _S98, lo_1[6U]);
                                    hi_1[6U] = fma(pj_30, _S99, hi_1[6U]);
                                    _S93 = int(6);
                                    break;
                                }
                                if(_S93 != int(6))
                                {
                                    _S92 = _S93;
                                    break;
                                }
                                _S92 = int(5);
                                break;
                            }
                            if(_S92 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S5)
                                    {
                                        _S97 = int(4);
                                        break;
                                    }
                                    float pj_31 = (*(&kernelContext_0)->ss_0)[7U * _S9 + j_3];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S9 + j_3], _S98, lo_1[7U]);
                                    hi_1[7U] = fma(pj_31, _S99, hi_1[7U]);
                                    _S97 = int(6);
                                    break;
                                }
                                if(_S97 != int(6))
                                {
                                    _S96 = _S97;
                                    break;
                                }
                                _S96 = int(5);
                                break;
                            }
                            if(_S96 != int(5))
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
                int _S100;
                int _S101;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_4 >= _S8)
                        {
                            _S22 = int(1);
                            break;
                        }
                        float _S102 = attnLo_0(vw_0[4U]);
                        float _S103 = attnHi_0(vw_0[4U]);
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
                                    float pj_32 = (*(&kernelContext_0)->ss_0)[j_4];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_4], _S102, lo_1[0U]);
                                    hi_1[0U] = fma(pj_32, _S103, hi_1[0U]);
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
                                    if(1U >= _S5)
                                    {
                                        _S20 = int(4);
                                        break;
                                    }
                                    float pj_33 = (*(&kernelContext_0)->ss_0)[_S9 + j_4];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S9 + j_4], _S102, lo_1[1U]);
                                    hi_1[1U] = fma(pj_33, _S103, hi_1[1U]);
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
                                    if(2U >= _S5)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_34 = (*(&kernelContext_0)->ss_0)[2U * _S9 + j_4];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S9 + j_4], _S102, lo_1[2U]);
                                    hi_1[2U] = fma(pj_34, _S103, hi_1[2U]);
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
                                    if(3U >= _S5)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_35 = (*(&kernelContext_0)->ss_0)[3U * _S9 + j_4];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S9 + j_4], _S102, lo_1[3U]);
                                    hi_1[3U] = fma(pj_35, _S103, hi_1[3U]);
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
                                    if(4U >= _S5)
                                    {
                                        _S89 = int(4);
                                        break;
                                    }
                                    float pj_36 = (*(&kernelContext_0)->ss_0)[4U * _S9 + j_4];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S9 + j_4], _S102, lo_1[4U]);
                                    hi_1[4U] = fma(pj_36, _S103, hi_1[4U]);
                                    _S89 = int(6);
                                    break;
                                }
                                if(_S89 != int(6))
                                {
                                    _S88 = _S89;
                                    break;
                                }
                                _S88 = int(5);
                                break;
                            }
                            if(_S88 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S5)
                                    {
                                        _S93 = int(4);
                                        break;
                                    }
                                    float pj_37 = (*(&kernelContext_0)->ss_0)[5U * _S9 + j_4];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S9 + j_4], _S102, lo_1[5U]);
                                    hi_1[5U] = fma(pj_37, _S103, hi_1[5U]);
                                    _S93 = int(6);
                                    break;
                                }
                                if(_S93 != int(6))
                                {
                                    _S92 = _S93;
                                    break;
                                }
                                _S92 = int(5);
                                break;
                            }
                            if(_S92 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S5)
                                    {
                                        _S97 = int(4);
                                        break;
                                    }
                                    float pj_38 = (*(&kernelContext_0)->ss_0)[6U * _S9 + j_4];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S9 + j_4], _S102, lo_1[6U]);
                                    hi_1[6U] = fma(pj_38, _S103, hi_1[6U]);
                                    _S97 = int(6);
                                    break;
                                }
                                if(_S97 != int(6))
                                {
                                    _S96 = _S97;
                                    break;
                                }
                                _S96 = int(5);
                                break;
                            }
                            if(_S96 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S5)
                                    {
                                        _S101 = int(4);
                                        break;
                                    }
                                    float pj_39 = (*(&kernelContext_0)->ss_0)[7U * _S9 + j_4];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S9 + j_4], _S102, lo_1[7U]);
                                    hi_1[7U] = fma(pj_39, _S103, hi_1[7U]);
                                    _S101 = int(6);
                                    break;
                                }
                                if(_S101 != int(6))
                                {
                                    _S100 = _S101;
                                    break;
                                }
                                _S100 = int(5);
                                break;
                            }
                            if(_S100 != int(5))
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
                int _S104;
                int _S105;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_5 >= _S8)
                        {
                            _S20 = int(1);
                            break;
                        }
                        float _S106 = attnLo_0(vw_0[5U]);
                        float _S107 = attnHi_0(vw_0[5U]);
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
                                    float pj_40 = (*(&kernelContext_0)->ss_0)[j_5];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_5], _S106, lo_1[0U]);
                                    hi_1[0U] = fma(pj_40, _S107, hi_1[0U]);
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
                                    if(1U >= _S5)
                                    {
                                        _S18 = int(4);
                                        break;
                                    }
                                    float pj_41 = (*(&kernelContext_0)->ss_0)[_S9 + j_5];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S9 + j_5], _S106, lo_1[1U]);
                                    hi_1[1U] = fma(pj_41, _S107, hi_1[1U]);
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
                                    if(2U >= _S5)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_42 = (*(&kernelContext_0)->ss_0)[2U * _S9 + j_5];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S9 + j_5], _S106, lo_1[2U]);
                                    hi_1[2U] = fma(pj_42, _S107, hi_1[2U]);
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
                                    if(3U >= _S5)
                                    {
                                        _S89 = int(4);
                                        break;
                                    }
                                    float pj_43 = (*(&kernelContext_0)->ss_0)[3U * _S9 + j_5];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S9 + j_5], _S106, lo_1[3U]);
                                    hi_1[3U] = fma(pj_43, _S107, hi_1[3U]);
                                    _S89 = int(6);
                                    break;
                                }
                                if(_S89 != int(6))
                                {
                                    _S88 = _S89;
                                    break;
                                }
                                _S88 = int(5);
                                break;
                            }
                            if(_S88 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S5)
                                    {
                                        _S93 = int(4);
                                        break;
                                    }
                                    float pj_44 = (*(&kernelContext_0)->ss_0)[4U * _S9 + j_5];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S9 + j_5], _S106, lo_1[4U]);
                                    hi_1[4U] = fma(pj_44, _S107, hi_1[4U]);
                                    _S93 = int(6);
                                    break;
                                }
                                if(_S93 != int(6))
                                {
                                    _S92 = _S93;
                                    break;
                                }
                                _S92 = int(5);
                                break;
                            }
                            if(_S92 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S5)
                                    {
                                        _S97 = int(4);
                                        break;
                                    }
                                    float pj_45 = (*(&kernelContext_0)->ss_0)[5U * _S9 + j_5];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S9 + j_5], _S106, lo_1[5U]);
                                    hi_1[5U] = fma(pj_45, _S107, hi_1[5U]);
                                    _S97 = int(6);
                                    break;
                                }
                                if(_S97 != int(6))
                                {
                                    _S96 = _S97;
                                    break;
                                }
                                _S96 = int(5);
                                break;
                            }
                            if(_S96 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S5)
                                    {
                                        _S101 = int(4);
                                        break;
                                    }
                                    float pj_46 = (*(&kernelContext_0)->ss_0)[6U * _S9 + j_5];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S9 + j_5], _S106, lo_1[6U]);
                                    hi_1[6U] = fma(pj_46, _S107, hi_1[6U]);
                                    _S101 = int(6);
                                    break;
                                }
                                if(_S101 != int(6))
                                {
                                    _S100 = _S101;
                                    break;
                                }
                                _S100 = int(5);
                                break;
                            }
                            if(_S100 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S5)
                                    {
                                        _S105 = int(4);
                                        break;
                                    }
                                    float pj_47 = (*(&kernelContext_0)->ss_0)[7U * _S9 + j_5];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S9 + j_5], _S106, lo_1[7U]);
                                    hi_1[7U] = fma(pj_47, _S107, hi_1[7U]);
                                    _S105 = int(6);
                                    break;
                                }
                                if(_S105 != int(6))
                                {
                                    _S104 = _S105;
                                    break;
                                }
                                _S104 = int(5);
                                break;
                            }
                            if(_S104 != int(5))
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
                int _S108;
                int _S109;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_6 >= _S8)
                        {
                            _S18 = int(1);
                            break;
                        }
                        float _S110 = attnLo_0(vw_0[6U]);
                        float _S111 = attnHi_0(vw_0[6U]);
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
                                    float pj_48 = (*(&kernelContext_0)->ss_0)[j_6];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_6], _S110, lo_1[0U]);
                                    hi_1[0U] = fma(pj_48, _S111, hi_1[0U]);
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
                                    if(1U >= _S5)
                                    {
                                        _S16 = int(4);
                                        break;
                                    }
                                    float pj_49 = (*(&kernelContext_0)->ss_0)[_S9 + j_6];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S9 + j_6], _S110, lo_1[1U]);
                                    hi_1[1U] = fma(pj_49, _S111, hi_1[1U]);
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
                                    if(2U >= _S5)
                                    {
                                        _S89 = int(4);
                                        break;
                                    }
                                    float pj_50 = (*(&kernelContext_0)->ss_0)[2U * _S9 + j_6];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S9 + j_6], _S110, lo_1[2U]);
                                    hi_1[2U] = fma(pj_50, _S111, hi_1[2U]);
                                    _S89 = int(6);
                                    break;
                                }
                                if(_S89 != int(6))
                                {
                                    _S88 = _S89;
                                    break;
                                }
                                _S88 = int(5);
                                break;
                            }
                            if(_S88 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S5)
                                    {
                                        _S93 = int(4);
                                        break;
                                    }
                                    float pj_51 = (*(&kernelContext_0)->ss_0)[3U * _S9 + j_6];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S9 + j_6], _S110, lo_1[3U]);
                                    hi_1[3U] = fma(pj_51, _S111, hi_1[3U]);
                                    _S93 = int(6);
                                    break;
                                }
                                if(_S93 != int(6))
                                {
                                    _S92 = _S93;
                                    break;
                                }
                                _S92 = int(5);
                                break;
                            }
                            if(_S92 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S5)
                                    {
                                        _S97 = int(4);
                                        break;
                                    }
                                    float pj_52 = (*(&kernelContext_0)->ss_0)[4U * _S9 + j_6];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S9 + j_6], _S110, lo_1[4U]);
                                    hi_1[4U] = fma(pj_52, _S111, hi_1[4U]);
                                    _S97 = int(6);
                                    break;
                                }
                                if(_S97 != int(6))
                                {
                                    _S96 = _S97;
                                    break;
                                }
                                _S96 = int(5);
                                break;
                            }
                            if(_S96 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S5)
                                    {
                                        _S101 = int(4);
                                        break;
                                    }
                                    float pj_53 = (*(&kernelContext_0)->ss_0)[5U * _S9 + j_6];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S9 + j_6], _S110, lo_1[5U]);
                                    hi_1[5U] = fma(pj_53, _S111, hi_1[5U]);
                                    _S101 = int(6);
                                    break;
                                }
                                if(_S101 != int(6))
                                {
                                    _S100 = _S101;
                                    break;
                                }
                                _S100 = int(5);
                                break;
                            }
                            if(_S100 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S5)
                                    {
                                        _S105 = int(4);
                                        break;
                                    }
                                    float pj_54 = (*(&kernelContext_0)->ss_0)[6U * _S9 + j_6];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S9 + j_6], _S110, lo_1[6U]);
                                    hi_1[6U] = fma(pj_54, _S111, hi_1[6U]);
                                    _S105 = int(6);
                                    break;
                                }
                                if(_S105 != int(6))
                                {
                                    _S104 = _S105;
                                    break;
                                }
                                _S104 = int(5);
                                break;
                            }
                            if(_S104 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S5)
                                    {
                                        _S109 = int(4);
                                        break;
                                    }
                                    float pj_55 = (*(&kernelContext_0)->ss_0)[7U * _S9 + j_6];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S9 + j_6], _S110, lo_1[7U]);
                                    hi_1[7U] = fma(pj_55, _S111, hi_1[7U]);
                                    _S109 = int(6);
                                    break;
                                }
                                if(_S109 != int(6))
                                {
                                    _S108 = _S109;
                                    break;
                                }
                                _S108 = int(5);
                                break;
                            }
                            if(_S108 != int(5))
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
                for(;;)
                {
                    for(;;)
                    {
                        if(j_7 >= _S8)
                        {
                            _S16 = int(1);
                            break;
                        }
                        float _S112 = attnLo_0(vw_0[7U]);
                        float _S113 = attnHi_0(vw_0[7U]);
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
                                    float pj_56 = (*(&kernelContext_0)->ss_0)[j_7];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_7], _S112, lo_1[0U]);
                                    hi_1[0U] = fma(pj_56, _S113, hi_1[0U]);
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
                                    if(1U >= _S5)
                                    {
                                        _S89 = int(4);
                                        break;
                                    }
                                    float pj_57 = (*(&kernelContext_0)->ss_0)[_S9 + j_7];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S9 + j_7], _S112, lo_1[1U]);
                                    hi_1[1U] = fma(pj_57, _S113, hi_1[1U]);
                                    _S89 = int(6);
                                    break;
                                }
                                if(_S89 != int(6))
                                {
                                    _S88 = _S89;
                                    break;
                                }
                                _S88 = int(5);
                                break;
                            }
                            if(_S88 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S5)
                                    {
                                        _S93 = int(4);
                                        break;
                                    }
                                    float pj_58 = (*(&kernelContext_0)->ss_0)[2U * _S9 + j_7];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S9 + j_7], _S112, lo_1[2U]);
                                    hi_1[2U] = fma(pj_58, _S113, hi_1[2U]);
                                    _S93 = int(6);
                                    break;
                                }
                                if(_S93 != int(6))
                                {
                                    _S92 = _S93;
                                    break;
                                }
                                _S92 = int(5);
                                break;
                            }
                            if(_S92 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S5)
                                    {
                                        _S97 = int(4);
                                        break;
                                    }
                                    float pj_59 = (*(&kernelContext_0)->ss_0)[3U * _S9 + j_7];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S9 + j_7], _S112, lo_1[3U]);
                                    hi_1[3U] = fma(pj_59, _S113, hi_1[3U]);
                                    _S97 = int(6);
                                    break;
                                }
                                if(_S97 != int(6))
                                {
                                    _S96 = _S97;
                                    break;
                                }
                                _S96 = int(5);
                                break;
                            }
                            if(_S96 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S5)
                                    {
                                        _S101 = int(4);
                                        break;
                                    }
                                    float pj_60 = (*(&kernelContext_0)->ss_0)[4U * _S9 + j_7];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S9 + j_7], _S112, lo_1[4U]);
                                    hi_1[4U] = fma(pj_60, _S113, hi_1[4U]);
                                    _S101 = int(6);
                                    break;
                                }
                                if(_S101 != int(6))
                                {
                                    _S100 = _S101;
                                    break;
                                }
                                _S100 = int(5);
                                break;
                            }
                            if(_S100 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S5)
                                    {
                                        _S105 = int(4);
                                        break;
                                    }
                                    float pj_61 = (*(&kernelContext_0)->ss_0)[5U * _S9 + j_7];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S9 + j_7], _S112, lo_1[5U]);
                                    hi_1[5U] = fma(pj_61, _S113, hi_1[5U]);
                                    _S105 = int(6);
                                    break;
                                }
                                if(_S105 != int(6))
                                {
                                    _S104 = _S105;
                                    break;
                                }
                                _S104 = int(5);
                                break;
                            }
                            if(_S104 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S5)
                                    {
                                        _S109 = int(4);
                                        break;
                                    }
                                    float pj_62 = (*(&kernelContext_0)->ss_0)[6U * _S9 + j_7];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S9 + j_7], _S112, lo_1[6U]);
                                    hi_1[6U] = fma(pj_62, _S113, hi_1[6U]);
                                    _S109 = int(6);
                                    break;
                                }
                                if(_S109 != int(6))
                                {
                                    _S108 = _S109;
                                    break;
                                }
                                _S108 = int(5);
                                break;
                            }
                            if(_S108 != int(5))
                            {
                                break;
                            }
                            int _S114;
                            for(;;)
                            {
                                int _S115;
                                for(;;)
                                {
                                    if(7U >= _S5)
                                    {
                                        _S115 = int(4);
                                        break;
                                    }
                                    float pj_63 = (*(&kernelContext_0)->ss_0)[7U * _S9 + j_7];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S9 + j_7], _S112, lo_1[7U]);
                                    hi_1[7U] = fma(pj_63, _S113, hi_1[7U]);
                                    _S115 = int(6);
                                    break;
                                }
                                if(_S115 != int(6))
                                {
                                    _S114 = _S115;
                                    break;
                                }
                                _S114 = int(5);
                                break;
                            }
                            if(_S114 != int(5))
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
                break;
            }
            kw_4 = kw_4 + 8U;
        }
    }
    p0_0 = 0U;
    for(;;)
    {
        if(p0_0 < _S5)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(0U == p0_0)
        {
            m_0 = lo_1[0U];
            sum_0 = hi_1[0U];
        }
        else
        {
            m_0 = 0.0;
            sum_0 = 0.0;
        }
        float b_0;
        if(1U == p0_0)
        {
            w_3 = lo_1[1U];
            b_0 = hi_1[1U];
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
            a_0 = lo_1[2U];
            b_1 = hi_1[2U];
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
            a_1 = lo_1[3U];
            b_2 = hi_1[3U];
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
            a_2 = lo_1[4U];
            b_3 = hi_1[4U];
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
            a_3 = lo_1[5U];
            b_4 = hi_1[5U];
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
            a_4 = lo_1[6U];
            b_5 = hi_1[6U];
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
            a_5 = lo_1[7U];
            b_6 = hi_1[7U];
        }
        else
        {
            a_5 = a_4;
            b_6 = b_5;
        }
        uint _S116 = 2U * pair_0;
        uint _S117 = part_0 * 256U + _S116;
        (*(&kernelContext_0)->ss_0)[_S117] = a_5;
        (*(&kernelContext_0)->ss_0)[_S117 + 1U] = b_6;
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
        float a_6 = (*(&kernelContext_0)->ss_0)[_S116];
        float b_7 = (*(&kernelContext_0)->ss_0)[_S116 + 1U];
        uint _S118 = _S4 * hpk_0;
        uint _S119 = p0_0 / (&kernelContext_0)->p_0->queries_0;
        uint h_4 = _S118 + _S119;
        uint i_9 = p0_0 % (&kernelContext_0)->p_0->queries_0;
        uint row_0 = h_4 * (&kernelContext_0)->p_0->queries_0 + i_9;
        if(((&kernelContext_0)->p_0->splits_0) > 1U)
        {
            uint pr_8 = split_0 * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0 + row_0;
            if(pair_0 == 0U)
            {
                uint _S120 = 2U * pr_8;
                float device* _S121 = (&kernelContext_0)->partialML_0+_S120;
                bool _S122 = _S8 == 0U;
                float _S123;
                if(_S122)
                {
                    _S123 = dg_negative_infinity_0();
                }
                else
                {
                    _S123 = (*(&kernelContext_0)->rowM_0)[p0_0];
                }
                *_S121 = _S123;
                float device* _S124 = (&kernelContext_0)->partialML_0+(_S120 + 1U);
                float _S125;
                if(_S122)
                {
                    _S125 = 0.0;
                }
                else
                {
                    _S125 = (*(&kernelContext_0)->rowL_0)[p0_0];
                }
                *_S124 = _S125;
            }
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S116)) = a_6;
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S116 + 1U)) = b_7;
            p0_0 = p0_0 + 1U;
            continue;
        }
        *((&kernelContext_0)->o_0+(row_0 * d2_0 + pair_0)) = attnPack_0(a_6 / (*(&kernelContext_0)->rowL_0)[p0_0], b_7 / (*(&kernelContext_0)->rowL_0)[p0_0]);
        p0_0 = p0_0 + 1U;
    }
    return;
}

