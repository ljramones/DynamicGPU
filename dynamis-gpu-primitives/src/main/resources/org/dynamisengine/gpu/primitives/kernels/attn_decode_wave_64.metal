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
    uint batch_0;
    uint writeLse_0;
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
    uint device* o_0;
    uint threadgroup* waves_0;
    array<float, int(8)> threadgroup* rowM_0;
    array<float, int(8)> threadgroup* rowL_0;
    array<uint, int(256)> threadgroup* qs_0;
    array<float, int(1024)> threadgroup* ss_0;
    array<float, int(8)> threadgroup* rowA_0;
};

[[kernel]] void attn_decode_wave_64(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(7)]], uint device* table_1 [[buffer(6)]], uint device* q_1 [[buffer(0)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partialML_1 [[buffer(4)]], float device* partialAcc_1 [[buffer(5)]], uint device* o_1 [[buffer(3)]])
{
    bool _S3;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->table_0 = table_1;
    (&kernelContext_0)->q_0 = q_1;
    (&kernelContext_0)->k_0 = k_1;
    (&kernelContext_0)->v_0 = v_1;
    (&kernelContext_0)->partialML_0 = partialML_1;
    (&kernelContext_0)->partialAcc_0 = partialAcc_1;
    (&kernelContext_0)->o_0 = o_1;
    threadgroup uint waves_1;
    (&kernelContext_0)->waves_0 = &waves_1;
    threadgroup array<float, int(8)> rowM_1;
    (&kernelContext_0)->rowM_0 = &rowM_1;
    threadgroup array<float, int(8)> rowL_1;
    (&kernelContext_0)->rowL_0 = &rowL_1;
    threadgroup array<uint, int(256)> qs_1;
    (&kernelContext_0)->qs_0 = &qs_1;
    threadgroup array<float, int(1024)> ss_1;
    (&kernelContext_0)->ss_0 = &ss_1;
    threadgroup array<float, int(8)> rowA_1;
    (&kernelContext_0)->rowA_0 = &rowA_1;
    uint t_0 = local_0.x;
    uint _S4 = group_0.y;
    uint split_0 = group_0.z;
    uint seq_0 = group_0.x;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint _S5 = min(hpk_0 * (&kernelContext_0)->p_0->queries_0, ROW_TIER_0);
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
    uint _S6 = seq_0 * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0;
    uint _S7 = max((&kernelContext_0)->p_0->batch_0, 1U) * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0;
    uint _S8 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint _S9 = keys_1 - (&kernelContext_0)->p_0->queries_0;
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S10 = min(keys_1, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    if(_S10 > kStart_0)
    {
        keys_1 = _S10 - kStart_0;
    }
    else
    {
        keys_1 = 0U;
    }
    uint _S11 = 1024U / max(_S5, 1U);
    uint _S12 = max(128U, _S11 / 128U * 128U);
    if(t_0 == 0U)
    {
        *(&kernelContext_0)->waves_0 = 0U;
    }
    if(t_0 < 8U)
    {
        (*(&kernelContext_0)->rowM_0)[t_0] = dg_negative_infinity_0();
        (*(&kernelContext_0)->rowL_0)[t_0] = 0.0;
    }
    bool visible_0;
    uint w_2;
    uint e_1 = t_0;
    for(;;)
    {
        if(e_1 < (ROW_TIER_0 * 32U))
        {
        }
        else
        {
            break;
        }
        uint r_1 = e_1 / 32U;
        uint pr_0 = e_1 % 32U;
        if(r_1 < _S5)
        {
            visible_0 = pr_0 < _S8;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S13 = _S4 * hpk_0;
            uint _S14 = r_1 / (&kernelContext_0)->p_0->queries_0;
            uint h_3 = _S13 + _S14;
            uint i_0 = r_1 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((_S6 + h_3 * (&kernelContext_0)->p_0->queries_0 + i_0) * _S8 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_1] = w_2;
        e_1 = e_1 + 128U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S15 = simd_sum(1U);
    uint lane_0 = simd_prefix_exclusive_sum(1U);
    bool _S16 = lane_0 == 0U;
    if(_S16)
    {
        uint _S17 = atomic_fetch_add_explicit(((atomic_uint threadgroup*)((&kernelContext_0)->waves_0)), 1U, memory_order_relaxed);
        w_2 = _S17;
    }
    else
    {
        w_2 = 0U;
    }
    uint _S18 = simd_sum(w_2);
    uint _S19 = t_0 % 32U;
    uint _S20 = t_0 / 32U;
    thread array<float, int(8)> lo_1;
    thread array<float, int(8)> hi_1;
    float w_3;
    float sum_0;
    float m_0;
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
    uint c0_0 = kStart_0;
    for(;;)
    {
        if(c0_0 < _S10)
        {
        }
        else
        {
            break;
        }
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
        int _S36;
        uint kw_0;
        uint kw_1;
        uint kw_2;
        uint kw_3;
        uint kw_4;
        uint kw_5;
        uint kw_6;
        uint p0_0;
        uint _S37 = min(_S12, _S10 - c0_0);
        uint j_0 = t_0;
        for(;;)
        {
            if(j_0 < _S37)
            {
            }
            else
            {
                break;
            }
            uint key_0 = c0_0 + j_0;
            if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
            {
                e_1 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * _S8;
            }
            else
            {
                uint _S38 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
                uint _S39 = (*((&kernelContext_0)->table_0+(r_0 + _S38)) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                uint _S40 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
                e_1 = (_S39 + _S40) * _S8;
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
                if(p0_0 < _S8)
                {
                    kw_6 = *((&kernelContext_0)->k_0+(e_1 + p0_0));
                }
                else
                {
                    kw_6 = 0U;
                }
                float _S41 = attnLo_0(kw_6);
                float _S42 = attnHi_0(kw_6);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[p0_0]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[p0_0]), _S41, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + p0_0]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + p0_0]), _S41, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S41, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + p0_0]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + p0_0]), _S41, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S41, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + p0_0]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + p0_0]), _S41, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S41, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + p0_0]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + p0_0]), _S41, s_0[7U]));
                }
                uint pr_1 = p0_0 + 1U;
                if(pr_1 < _S8)
                {
                    kw_5 = *((&kernelContext_0)->k_0+(e_1 + pr_1));
                }
                else
                {
                    kw_5 = 0U;
                }
                float _S43 = attnLo_0(kw_5);
                float _S44 = attnHi_0(kw_5);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_1]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_1]), _S43, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_1]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_1]), _S43, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S43, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_1]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_1]), _S43, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S43, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_1]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_1]), _S43, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S43, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_1]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_1]), _S43, s_0[7U]));
                }
                uint pr_2 = p0_0 + 2U;
                if(pr_2 < _S8)
                {
                    kw_4 = *((&kernelContext_0)->k_0+(e_1 + pr_2));
                }
                else
                {
                    kw_4 = 0U;
                }
                float _S45 = attnLo_0(kw_4);
                float _S46 = attnHi_0(kw_4);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_2]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_2]), _S45, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_2]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_2]), _S45, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S45, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_2]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_2]), _S45, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S45, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_2]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_2]), _S45, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S45, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_2]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_2]), _S45, s_0[7U]));
                }
                uint pr_3 = p0_0 + 3U;
                if(pr_3 < _S8)
                {
                    kw_3 = *((&kernelContext_0)->k_0+(e_1 + pr_3));
                }
                else
                {
                    kw_3 = 0U;
                }
                float _S47 = attnLo_0(kw_3);
                float _S48 = attnHi_0(kw_3);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_3]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_3]), _S47, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_3]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_3]), _S47, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S47, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_3]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_3]), _S47, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S47, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_3]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_3]), _S47, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S47, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_3]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_3]), _S47, s_0[7U]));
                }
                uint pr_4 = p0_0 + 4U;
                if(pr_4 < _S8)
                {
                    kw_2 = *((&kernelContext_0)->k_0+(e_1 + pr_4));
                }
                else
                {
                    kw_2 = 0U;
                }
                float _S49 = attnLo_0(kw_2);
                float _S50 = attnHi_0(kw_2);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_4]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_4]), _S49, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_4]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_4]), _S49, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S49, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_4]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_4]), _S49, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S49, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_4]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_4]), _S49, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S49, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_4]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_4]), _S49, s_0[7U]));
                }
                uint pr_5 = p0_0 + 5U;
                if(pr_5 < _S8)
                {
                    kw_1 = *((&kernelContext_0)->k_0+(e_1 + pr_5));
                }
                else
                {
                    kw_1 = 0U;
                }
                float _S51 = attnLo_0(kw_1);
                float _S52 = attnHi_0(kw_1);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_5]), _S52, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_5]), _S51, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_5]), _S52, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_5]), _S51, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S52, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S51, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_5]), _S52, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_5]), _S51, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S52, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S51, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_5]), _S52, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_5]), _S51, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S52, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S51, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_5]), _S52, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_5]), _S51, s_0[7U]));
                }
                uint pr_6 = p0_0 + 6U;
                if(pr_6 < _S8)
                {
                    kw_0 = *((&kernelContext_0)->k_0+(e_1 + pr_6));
                }
                else
                {
                    kw_0 = 0U;
                }
                float _S53 = attnLo_0(kw_0);
                float _S54 = attnHi_0(kw_0);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_6]), _S54, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_6]), _S53, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_6]), _S54, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_6]), _S53, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S54, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S53, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_6]), _S54, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_6]), _S53, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S54, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S53, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_6]), _S54, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_6]), _S53, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S54, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S53, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_6]), _S54, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_6]), _S53, s_0[7U]));
                }
                uint pr_7 = p0_0 + 7U;
                uint kw_7;
                if(pr_7 < _S8)
                {
                    kw_7 = *((&kernelContext_0)->k_0+(e_1 + pr_7));
                }
                else
                {
                    kw_7 = 0U;
                }
                float _S55 = attnLo_0(kw_7);
                float _S56 = attnHi_0(kw_7);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_7]), _S56, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_7]), _S55, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_7]), _S56, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_7]), _S55, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S56, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S55, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_7]), _S56, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_7]), _S55, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S56, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S55, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_7]), _S56, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_7]), _S55, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S56, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S55, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_7]), _S56, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_7]), _S55, s_0[7U]));
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
                            _S36 = int(2);
                            break;
                        }
                        uint i_1 = 0U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_1 + _S9);
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
                        _S36 = int(4);
                        break;
                    }
                    if(_S36 != int(4))
                    {
                        _S35 = _S36;
                        break;
                    }
                    _S35 = int(3);
                    break;
                }
                if(_S35 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(1U >= _S5)
                        {
                            _S34 = int(2);
                            break;
                        }
                        uint i_2 = 1U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_2 + _S9);
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[1U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[_S12 + j_0] = m_0;
                        _S34 = int(4);
                        break;
                    }
                    if(_S34 != int(4))
                    {
                        _S33 = _S34;
                        break;
                    }
                    _S33 = int(3);
                    break;
                }
                if(_S33 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(2U >= _S5)
                        {
                            _S32 = int(2);
                            break;
                        }
                        uint i_3 = 2U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_3 + _S9);
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[2U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[2U * _S12 + j_0] = m_0;
                        _S32 = int(4);
                        break;
                    }
                    if(_S32 != int(4))
                    {
                        _S31 = _S32;
                        break;
                    }
                    _S31 = int(3);
                    break;
                }
                if(_S31 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(3U >= _S5)
                        {
                            _S30 = int(2);
                            break;
                        }
                        uint i_4 = 3U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_4 + _S9);
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[3U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[3U * _S12 + j_0] = m_0;
                        _S30 = int(4);
                        break;
                    }
                    if(_S30 != int(4))
                    {
                        _S29 = _S30;
                        break;
                    }
                    _S29 = int(3);
                    break;
                }
                if(_S29 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(4U >= _S5)
                        {
                            _S28 = int(2);
                            break;
                        }
                        uint i_5 = 4U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_5 + _S9);
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[4U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[4U * _S12 + j_0] = m_0;
                        _S28 = int(4);
                        break;
                    }
                    if(_S28 != int(4))
                    {
                        _S27 = _S28;
                        break;
                    }
                    _S27 = int(3);
                    break;
                }
                if(_S27 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(5U >= _S5)
                        {
                            _S26 = int(2);
                            break;
                        }
                        uint i_6 = 5U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_6 + _S9);
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[5U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[5U * _S12 + j_0] = m_0;
                        _S26 = int(4);
                        break;
                    }
                    if(_S26 != int(4))
                    {
                        _S25 = _S26;
                        break;
                    }
                    _S25 = int(3);
                    break;
                }
                if(_S25 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(6U >= _S5)
                        {
                            _S24 = int(2);
                            break;
                        }
                        uint i_7 = 6U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_7 + _S9);
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[6U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[6U * _S12 + j_0] = m_0;
                        _S24 = int(4);
                        break;
                    }
                    if(_S24 != int(4))
                    {
                        _S23 = _S24;
                        break;
                    }
                    _S23 = int(3);
                    break;
                }
                if(_S23 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(7U >= _S5)
                        {
                            _S22 = int(2);
                            break;
                        }
                        uint i_8 = 7U % (&kernelContext_0)->p_0->queries_0;
                        if(((&kernelContext_0)->p_0->causal_0) == 0U)
                        {
                            visible_0 = true;
                        }
                        else
                        {
                            visible_0 = key_0 <= (i_8 + _S9);
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[7U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[7U * _S12 + j_0] = m_0;
                        _S22 = int(4);
                        break;
                    }
                    if(_S22 != int(4))
                    {
                        _S21 = _S22;
                        break;
                    }
                    _S21 = int(3);
                    break;
                }
                if(_S21 != int(3))
                {
                    break;
                }
                break;
            }
            j_0 = j_0 + 128U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        p0_0 = _S18;
        for(;;)
        {
            if(p0_0 < _S5)
            {
            }
            else
            {
                break;
            }
            float _S57 = dg_negative_infinity_0();
            kw_6 = lane_0;
            m_0 = _S57;
            for(;;)
            {
                if(kw_6 < _S37)
                {
                }
                else
                {
                    break;
                }
                float _S58 = max(m_0, (*(&kernelContext_0)->ss_0)[p0_0 * _S12 + kw_6]);
                kw_6 = kw_6 + _S15;
                m_0 = _S58;
            }
            float m_1 = simd_max(m_0);
            float mOld_0 = (*(&kernelContext_0)->rowM_0)[p0_0];
            float _S59 = max((*(&kernelContext_0)->rowM_0)[p0_0], m_1);
            kw_5 = lane_0;
            sum_0 = 0.0;
            for(;;)
            {
                if(kw_5 < _S37)
                {
                }
                else
                {
                    break;
                }
                float sv_0 = (*(&kernelContext_0)->ss_0)[p0_0 * _S12 + kw_5];
                if(((*(&kernelContext_0)->ss_0)[p0_0 * _S12 + kw_5]) == _S57)
                {
                    w_3 = 0.0;
                }
                else
                {
                    w_3 = exp(sv_0 - _S59);
                }
                (*(&kernelContext_0)->ss_0)[p0_0 * _S12 + kw_5] = w_3;
                float sum_1 = sum_0 + w_3;
                kw_5 = kw_5 + _S15;
                sum_0 = sum_1;
            }
            float sum_2 = simd_sum(sum_0);
            if(mOld_0 == _S57)
            {
                w_3 = 0.0;
            }
            else
            {
                w_3 = exp(mOld_0 - _S59);
            }
            if(_S16)
            {
                (*(&kernelContext_0)->rowA_0)[p0_0] = w_3;
                (*(&kernelContext_0)->rowL_0)[p0_0] = (*(&kernelContext_0)->rowL_0)[p0_0] * w_3 + sum_2;
                (*(&kernelContext_0)->rowM_0)[p0_0] = _S59;
            }
            p0_0 = p0_0 + *(&kernelContext_0)->waves_0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(0U < _S5)
        {
            float a_0 = (*(&kernelContext_0)->rowA_0)[0U];
            lo_1[0U] = lo_1[0U] * (*(&kernelContext_0)->rowA_0)[0U];
            hi_1[0U] = hi_1[0U] * a_0;
        }
        if(1U < _S5)
        {
            float a_1 = (*(&kernelContext_0)->rowA_0)[1U];
            lo_1[1U] = lo_1[1U] * (*(&kernelContext_0)->rowA_0)[1U];
            hi_1[1U] = hi_1[1U] * a_1;
        }
        if(2U < _S5)
        {
            float a_2 = (*(&kernelContext_0)->rowA_0)[2U];
            lo_1[2U] = lo_1[2U] * (*(&kernelContext_0)->rowA_0)[2U];
            hi_1[2U] = hi_1[2U] * a_2;
        }
        if(3U < _S5)
        {
            float a_3 = (*(&kernelContext_0)->rowA_0)[3U];
            lo_1[3U] = lo_1[3U] * (*(&kernelContext_0)->rowA_0)[3U];
            hi_1[3U] = hi_1[3U] * a_3;
        }
        if(4U < _S5)
        {
            float a_4 = (*(&kernelContext_0)->rowA_0)[4U];
            lo_1[4U] = lo_1[4U] * (*(&kernelContext_0)->rowA_0)[4U];
            hi_1[4U] = hi_1[4U] * a_4;
        }
        if(5U < _S5)
        {
            float a_5 = (*(&kernelContext_0)->rowA_0)[5U];
            lo_1[5U] = lo_1[5U] * (*(&kernelContext_0)->rowA_0)[5U];
            hi_1[5U] = hi_1[5U] * a_5;
        }
        if(6U < _S5)
        {
            float a_6 = (*(&kernelContext_0)->rowA_0)[6U];
            lo_1[6U] = lo_1[6U] * (*(&kernelContext_0)->rowA_0)[6U];
            hi_1[6U] = hi_1[6U] * a_6;
        }
        if(7U < _S5)
        {
            float a_7 = (*(&kernelContext_0)->rowA_0)[7U];
            lo_1[7U] = lo_1[7U] * (*(&kernelContext_0)->rowA_0)[7U];
            hi_1[7U] = hi_1[7U] * a_7;
        }
        if(_S19 < _S8)
        {
            kw_6 = _S20;
            for(;;)
            {
                bool _S60 = kw_6 < _S37;
                if(_S60)
                {
                }
                else
                {
                    break;
                }
                thread array<uint, int(8)> vw_0;
                if(_S60)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        e_1 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + kw_6)) * _S8 + _S19;
                    }
                    else
                    {
                        uint _S61 = c0_0 + kw_6;
                        uint _S62 = _S61 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S63 = (*((&kernelContext_0)->table_0+(r_0 + _S62)) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S64 = _S61 % (&kernelContext_0)->p_0->pageSize_0;
                        e_1 = (_S63 + _S64) * _S8 + _S19;
                    }
                    e_1 = *((&kernelContext_0)->v_0+e_1);
                }
                else
                {
                    e_1 = 0U;
                }
                vw_0[0U] = e_1;
                uint j_1 = kw_6 + 4U;
                if(j_1 < _S37)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        w_2 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_1)) * _S8 + _S19;
                    }
                    else
                    {
                        uint _S65 = c0_0 + j_1;
                        uint _S66 = _S65 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S67 = (*((&kernelContext_0)->table_0+(r_0 + _S66)) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S68 = _S65 % (&kernelContext_0)->p_0->pageSize_0;
                        w_2 = (_S67 + _S68) * _S8 + _S19;
                    }
                    w_2 = *((&kernelContext_0)->v_0+w_2);
                }
                else
                {
                    w_2 = 0U;
                }
                vw_0[1U] = w_2;
                uint j_2 = kw_6 + 8U;
                if(j_2 < _S37)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_5 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_2)) * _S8 + _S19;
                    }
                    else
                    {
                        uint _S69 = c0_0 + j_2;
                        uint _S70 = _S69 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S71 = (*((&kernelContext_0)->table_0+(r_0 + _S70)) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S72 = _S69 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_5 = (_S71 + _S72) * _S8 + _S19;
                    }
                    kw_5 = *((&kernelContext_0)->v_0+kw_5);
                }
                else
                {
                    kw_5 = 0U;
                }
                vw_0[2U] = kw_5;
                uint j_3 = kw_6 + 12U;
                if(j_3 < _S37)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_4 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_3)) * _S8 + _S19;
                    }
                    else
                    {
                        uint _S73 = c0_0 + j_3;
                        uint _S74 = _S73 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S75 = (*((&kernelContext_0)->table_0+(r_0 + _S74)) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S76 = _S73 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_4 = (_S75 + _S76) * _S8 + _S19;
                    }
                    kw_4 = *((&kernelContext_0)->v_0+kw_4);
                }
                else
                {
                    kw_4 = 0U;
                }
                vw_0[3U] = kw_4;
                uint j_4 = kw_6 + 16U;
                if(j_4 < _S37)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_3 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_4)) * _S8 + _S19;
                    }
                    else
                    {
                        uint _S77 = c0_0 + j_4;
                        uint _S78 = _S77 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S79 = (*((&kernelContext_0)->table_0+(r_0 + _S78)) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S80 = _S77 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_3 = (_S79 + _S80) * _S8 + _S19;
                    }
                    kw_3 = *((&kernelContext_0)->v_0+kw_3);
                }
                else
                {
                    kw_3 = 0U;
                }
                vw_0[4U] = kw_3;
                uint j_5 = kw_6 + 20U;
                if(j_5 < _S37)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_2 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_5)) * _S8 + _S19;
                    }
                    else
                    {
                        uint _S81 = c0_0 + j_5;
                        uint _S82 = _S81 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S83 = (*((&kernelContext_0)->table_0+(r_0 + _S82)) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S84 = _S81 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_2 = (_S83 + _S84) * _S8 + _S19;
                    }
                    kw_2 = *((&kernelContext_0)->v_0+kw_2);
                }
                else
                {
                    kw_2 = 0U;
                }
                vw_0[5U] = kw_2;
                uint j_6 = kw_6 + 24U;
                if(j_6 < _S37)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_1 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_6)) * _S8 + _S19;
                    }
                    else
                    {
                        uint _S85 = c0_0 + j_6;
                        uint _S86 = _S85 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S87 = (*((&kernelContext_0)->table_0+(r_0 + _S86)) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S88 = _S85 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_1 = (_S87 + _S88) * _S8 + _S19;
                    }
                    kw_1 = *((&kernelContext_0)->v_0+kw_1);
                }
                else
                {
                    kw_1 = 0U;
                }
                vw_0[6U] = kw_1;
                uint j_7 = kw_6 + 28U;
                if(j_7 < _S37)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_0 = (_S4 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_7)) * _S8 + _S19;
                    }
                    else
                    {
                        uint _S89 = c0_0 + j_7;
                        uint _S90 = _S89 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S91 = (*((&kernelContext_0)->table_0+(r_0 + _S90)) * p_1->kvHeads_0 + _S4) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S92 = _S89 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_0 = (_S91 + _S92) * _S8 + _S19;
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
                            if(kw_6 >= _S37)
                            {
                                _S36 = int(2);
                                visible_0 = false;
                                break;
                            }
                            float _S93 = attnLo_0(vw_0[0U]);
                            float _S94 = attnHi_0(vw_0[0U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        bool _S95 = 0U >= _S5;
                                        _S3 = _S95;
                                        if(_S95)
                                        {
                                            _S36 = int(5);
                                            break;
                                        }
                                        float pj_0 = (*(&kernelContext_0)->ss_0)[kw_6];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[kw_6], _S93, lo_1[0U]);
                                        hi_1[0U] = fma(pj_0, _S94, hi_1[0U]);
                                        _S36 = int(7);
                                        break;
                                    }
                                    if(_S36 != int(7))
                                    {
                                        _S35 = _S36;
                                        break;
                                    }
                                    _S35 = int(6);
                                    break;
                                }
                                if(_S35 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S5)
                                        {
                                            _S34 = int(5);
                                            break;
                                        }
                                        float pj_1 = (*(&kernelContext_0)->ss_0)[_S12 + kw_6];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S12 + kw_6], _S93, lo_1[1U]);
                                        hi_1[1U] = fma(pj_1, _S94, hi_1[1U]);
                                        _S34 = int(7);
                                        break;
                                    }
                                    if(_S34 != int(7))
                                    {
                                        _S33 = _S34;
                                        break;
                                    }
                                    _S33 = int(6);
                                    break;
                                }
                                if(_S33 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S5)
                                        {
                                            _S32 = int(5);
                                            break;
                                        }
                                        float pj_2 = (*(&kernelContext_0)->ss_0)[2U * _S12 + kw_6];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S12 + kw_6], _S93, lo_1[2U]);
                                        hi_1[2U] = fma(pj_2, _S94, hi_1[2U]);
                                        _S32 = int(7);
                                        break;
                                    }
                                    if(_S32 != int(7))
                                    {
                                        _S31 = _S32;
                                        break;
                                    }
                                    _S31 = int(6);
                                    break;
                                }
                                if(_S31 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S5)
                                        {
                                            _S30 = int(5);
                                            break;
                                        }
                                        float pj_3 = (*(&kernelContext_0)->ss_0)[3U * _S12 + kw_6];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S12 + kw_6], _S93, lo_1[3U]);
                                        hi_1[3U] = fma(pj_3, _S94, hi_1[3U]);
                                        _S30 = int(7);
                                        break;
                                    }
                                    if(_S30 != int(7))
                                    {
                                        _S29 = _S30;
                                        break;
                                    }
                                    _S29 = int(6);
                                    break;
                                }
                                if(_S29 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S5)
                                        {
                                            _S28 = int(5);
                                            break;
                                        }
                                        float pj_4 = (*(&kernelContext_0)->ss_0)[4U * _S12 + kw_6];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S12 + kw_6], _S93, lo_1[4U]);
                                        hi_1[4U] = fma(pj_4, _S94, hi_1[4U]);
                                        _S28 = int(7);
                                        break;
                                    }
                                    if(_S28 != int(7))
                                    {
                                        _S27 = _S28;
                                        break;
                                    }
                                    _S27 = int(6);
                                    break;
                                }
                                if(_S27 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S5)
                                        {
                                            _S26 = int(5);
                                            break;
                                        }
                                        float pj_5 = (*(&kernelContext_0)->ss_0)[5U * _S12 + kw_6];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S12 + kw_6], _S93, lo_1[5U]);
                                        hi_1[5U] = fma(pj_5, _S94, hi_1[5U]);
                                        _S26 = int(7);
                                        break;
                                    }
                                    if(_S26 != int(7))
                                    {
                                        _S25 = _S26;
                                        break;
                                    }
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S5)
                                        {
                                            _S24 = int(5);
                                            break;
                                        }
                                        float pj_6 = (*(&kernelContext_0)->ss_0)[6U * _S12 + kw_6];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S12 + kw_6], _S93, lo_1[6U]);
                                        hi_1[6U] = fma(pj_6, _S94, hi_1[6U]);
                                        _S24 = int(7);
                                        break;
                                    }
                                    if(_S24 != int(7))
                                    {
                                        _S23 = _S24;
                                        break;
                                    }
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S5)
                                        {
                                            _S22 = int(5);
                                            break;
                                        }
                                        float pj_7 = (*(&kernelContext_0)->ss_0)[7U * _S12 + kw_6];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S12 + kw_6], _S93, lo_1[7U]);
                                        hi_1[7U] = fma(pj_7, _S94, hi_1[7U]);
                                        _S22 = int(7);
                                        break;
                                    }
                                    if(_S22 != int(7))
                                    {
                                        _S21 = _S22;
                                        break;
                                    }
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S36 = int(4);
                            visible_0 = _S3;
                            break;
                        }
                        if(_S36 != int(4))
                        {
                            _S35 = _S36;
                            break;
                        }
                        _S35 = int(3);
                        break;
                    }
                    if(_S35 != int(3))
                    {
                        break;
                    }
                    int _S96;
                    int _S97;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_1 >= _S37)
                            {
                                _S34 = int(2);
                                break;
                            }
                            float _S98 = attnLo_0(vw_0[1U]);
                            float _S99 = attnHi_0(vw_0[1U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S34 = int(5);
                                            break;
                                        }
                                        float pj_8 = (*(&kernelContext_0)->ss_0)[j_1];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_1], _S98, lo_1[0U]);
                                        hi_1[0U] = fma(pj_8, _S99, hi_1[0U]);
                                        _S34 = int(7);
                                        break;
                                    }
                                    if(_S34 != int(7))
                                    {
                                        _S33 = _S34;
                                        break;
                                    }
                                    _S33 = int(6);
                                    break;
                                }
                                if(_S33 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S5)
                                        {
                                            _S32 = int(5);
                                            break;
                                        }
                                        float pj_9 = (*(&kernelContext_0)->ss_0)[_S12 + j_1];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S12 + j_1], _S98, lo_1[1U]);
                                        hi_1[1U] = fma(pj_9, _S99, hi_1[1U]);
                                        _S32 = int(7);
                                        break;
                                    }
                                    if(_S32 != int(7))
                                    {
                                        _S31 = _S32;
                                        break;
                                    }
                                    _S31 = int(6);
                                    break;
                                }
                                if(_S31 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S5)
                                        {
                                            _S30 = int(5);
                                            break;
                                        }
                                        float pj_10 = (*(&kernelContext_0)->ss_0)[2U * _S12 + j_1];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S12 + j_1], _S98, lo_1[2U]);
                                        hi_1[2U] = fma(pj_10, _S99, hi_1[2U]);
                                        _S30 = int(7);
                                        break;
                                    }
                                    if(_S30 != int(7))
                                    {
                                        _S29 = _S30;
                                        break;
                                    }
                                    _S29 = int(6);
                                    break;
                                }
                                if(_S29 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S5)
                                        {
                                            _S28 = int(5);
                                            break;
                                        }
                                        float pj_11 = (*(&kernelContext_0)->ss_0)[3U * _S12 + j_1];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S12 + j_1], _S98, lo_1[3U]);
                                        hi_1[3U] = fma(pj_11, _S99, hi_1[3U]);
                                        _S28 = int(7);
                                        break;
                                    }
                                    if(_S28 != int(7))
                                    {
                                        _S27 = _S28;
                                        break;
                                    }
                                    _S27 = int(6);
                                    break;
                                }
                                if(_S27 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S5)
                                        {
                                            _S26 = int(5);
                                            break;
                                        }
                                        float pj_12 = (*(&kernelContext_0)->ss_0)[4U * _S12 + j_1];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S12 + j_1], _S98, lo_1[4U]);
                                        hi_1[4U] = fma(pj_12, _S99, hi_1[4U]);
                                        _S26 = int(7);
                                        break;
                                    }
                                    if(_S26 != int(7))
                                    {
                                        _S25 = _S26;
                                        break;
                                    }
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S5)
                                        {
                                            _S24 = int(5);
                                            break;
                                        }
                                        float pj_13 = (*(&kernelContext_0)->ss_0)[5U * _S12 + j_1];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S12 + j_1], _S98, lo_1[5U]);
                                        hi_1[5U] = fma(pj_13, _S99, hi_1[5U]);
                                        _S24 = int(7);
                                        break;
                                    }
                                    if(_S24 != int(7))
                                    {
                                        _S23 = _S24;
                                        break;
                                    }
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S5)
                                        {
                                            _S22 = int(5);
                                            break;
                                        }
                                        float pj_14 = (*(&kernelContext_0)->ss_0)[6U * _S12 + j_1];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S12 + j_1], _S98, lo_1[6U]);
                                        hi_1[6U] = fma(pj_14, _S99, hi_1[6U]);
                                        _S22 = int(7);
                                        break;
                                    }
                                    if(_S22 != int(7))
                                    {
                                        _S21 = _S22;
                                        break;
                                    }
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S5)
                                        {
                                            _S97 = int(5);
                                            break;
                                        }
                                        float pj_15 = (*(&kernelContext_0)->ss_0)[7U * _S12 + j_1];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S12 + j_1], _S98, lo_1[7U]);
                                        hi_1[7U] = fma(pj_15, _S99, hi_1[7U]);
                                        _S97 = int(7);
                                        break;
                                    }
                                    if(_S97 != int(7))
                                    {
                                        _S96 = _S97;
                                        break;
                                    }
                                    _S96 = int(6);
                                    break;
                                }
                                if(_S96 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S34 = int(4);
                            break;
                        }
                        if(_S34 != int(4))
                        {
                            _S33 = _S34;
                            break;
                        }
                        _S33 = int(3);
                        break;
                    }
                    if(_S33 != int(3))
                    {
                        break;
                    }
                    int _S100;
                    int _S101;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_2 >= _S37)
                            {
                                _S32 = int(2);
                                break;
                            }
                            float _S102 = attnLo_0(vw_0[2U]);
                            float _S103 = attnHi_0(vw_0[2U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S32 = int(5);
                                            break;
                                        }
                                        float pj_16 = (*(&kernelContext_0)->ss_0)[j_2];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_2], _S102, lo_1[0U]);
                                        hi_1[0U] = fma(pj_16, _S103, hi_1[0U]);
                                        _S32 = int(7);
                                        break;
                                    }
                                    if(_S32 != int(7))
                                    {
                                        _S31 = _S32;
                                        break;
                                    }
                                    _S31 = int(6);
                                    break;
                                }
                                if(_S31 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S5)
                                        {
                                            _S30 = int(5);
                                            break;
                                        }
                                        float pj_17 = (*(&kernelContext_0)->ss_0)[_S12 + j_2];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S12 + j_2], _S102, lo_1[1U]);
                                        hi_1[1U] = fma(pj_17, _S103, hi_1[1U]);
                                        _S30 = int(7);
                                        break;
                                    }
                                    if(_S30 != int(7))
                                    {
                                        _S29 = _S30;
                                        break;
                                    }
                                    _S29 = int(6);
                                    break;
                                }
                                if(_S29 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S5)
                                        {
                                            _S28 = int(5);
                                            break;
                                        }
                                        float pj_18 = (*(&kernelContext_0)->ss_0)[2U * _S12 + j_2];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S12 + j_2], _S102, lo_1[2U]);
                                        hi_1[2U] = fma(pj_18, _S103, hi_1[2U]);
                                        _S28 = int(7);
                                        break;
                                    }
                                    if(_S28 != int(7))
                                    {
                                        _S27 = _S28;
                                        break;
                                    }
                                    _S27 = int(6);
                                    break;
                                }
                                if(_S27 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S5)
                                        {
                                            _S26 = int(5);
                                            break;
                                        }
                                        float pj_19 = (*(&kernelContext_0)->ss_0)[3U * _S12 + j_2];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S12 + j_2], _S102, lo_1[3U]);
                                        hi_1[3U] = fma(pj_19, _S103, hi_1[3U]);
                                        _S26 = int(7);
                                        break;
                                    }
                                    if(_S26 != int(7))
                                    {
                                        _S25 = _S26;
                                        break;
                                    }
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S5)
                                        {
                                            _S24 = int(5);
                                            break;
                                        }
                                        float pj_20 = (*(&kernelContext_0)->ss_0)[4U * _S12 + j_2];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S12 + j_2], _S102, lo_1[4U]);
                                        hi_1[4U] = fma(pj_20, _S103, hi_1[4U]);
                                        _S24 = int(7);
                                        break;
                                    }
                                    if(_S24 != int(7))
                                    {
                                        _S23 = _S24;
                                        break;
                                    }
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S5)
                                        {
                                            _S22 = int(5);
                                            break;
                                        }
                                        float pj_21 = (*(&kernelContext_0)->ss_0)[5U * _S12 + j_2];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S12 + j_2], _S102, lo_1[5U]);
                                        hi_1[5U] = fma(pj_21, _S103, hi_1[5U]);
                                        _S22 = int(7);
                                        break;
                                    }
                                    if(_S22 != int(7))
                                    {
                                        _S21 = _S22;
                                        break;
                                    }
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S5)
                                        {
                                            _S97 = int(5);
                                            break;
                                        }
                                        float pj_22 = (*(&kernelContext_0)->ss_0)[6U * _S12 + j_2];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S12 + j_2], _S102, lo_1[6U]);
                                        hi_1[6U] = fma(pj_22, _S103, hi_1[6U]);
                                        _S97 = int(7);
                                        break;
                                    }
                                    if(_S97 != int(7))
                                    {
                                        _S96 = _S97;
                                        break;
                                    }
                                    _S96 = int(6);
                                    break;
                                }
                                if(_S96 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S5)
                                        {
                                            _S101 = int(5);
                                            break;
                                        }
                                        float pj_23 = (*(&kernelContext_0)->ss_0)[7U * _S12 + j_2];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S12 + j_2], _S102, lo_1[7U]);
                                        hi_1[7U] = fma(pj_23, _S103, hi_1[7U]);
                                        _S101 = int(7);
                                        break;
                                    }
                                    if(_S101 != int(7))
                                    {
                                        _S100 = _S101;
                                        break;
                                    }
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S32 = int(4);
                            break;
                        }
                        if(_S32 != int(4))
                        {
                            _S31 = _S32;
                            break;
                        }
                        _S31 = int(3);
                        break;
                    }
                    if(_S31 != int(3))
                    {
                        break;
                    }
                    int _S104;
                    int _S105;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_3 >= _S37)
                            {
                                _S30 = int(2);
                                break;
                            }
                            float _S106 = attnLo_0(vw_0[3U]);
                            float _S107 = attnHi_0(vw_0[3U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S30 = int(5);
                                            break;
                                        }
                                        float pj_24 = (*(&kernelContext_0)->ss_0)[j_3];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_3], _S106, lo_1[0U]);
                                        hi_1[0U] = fma(pj_24, _S107, hi_1[0U]);
                                        _S30 = int(7);
                                        break;
                                    }
                                    if(_S30 != int(7))
                                    {
                                        _S29 = _S30;
                                        break;
                                    }
                                    _S29 = int(6);
                                    break;
                                }
                                if(_S29 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S5)
                                        {
                                            _S28 = int(5);
                                            break;
                                        }
                                        float pj_25 = (*(&kernelContext_0)->ss_0)[_S12 + j_3];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S12 + j_3], _S106, lo_1[1U]);
                                        hi_1[1U] = fma(pj_25, _S107, hi_1[1U]);
                                        _S28 = int(7);
                                        break;
                                    }
                                    if(_S28 != int(7))
                                    {
                                        _S27 = _S28;
                                        break;
                                    }
                                    _S27 = int(6);
                                    break;
                                }
                                if(_S27 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S5)
                                        {
                                            _S26 = int(5);
                                            break;
                                        }
                                        float pj_26 = (*(&kernelContext_0)->ss_0)[2U * _S12 + j_3];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S12 + j_3], _S106, lo_1[2U]);
                                        hi_1[2U] = fma(pj_26, _S107, hi_1[2U]);
                                        _S26 = int(7);
                                        break;
                                    }
                                    if(_S26 != int(7))
                                    {
                                        _S25 = _S26;
                                        break;
                                    }
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S5)
                                        {
                                            _S24 = int(5);
                                            break;
                                        }
                                        float pj_27 = (*(&kernelContext_0)->ss_0)[3U * _S12 + j_3];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S12 + j_3], _S106, lo_1[3U]);
                                        hi_1[3U] = fma(pj_27, _S107, hi_1[3U]);
                                        _S24 = int(7);
                                        break;
                                    }
                                    if(_S24 != int(7))
                                    {
                                        _S23 = _S24;
                                        break;
                                    }
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S5)
                                        {
                                            _S22 = int(5);
                                            break;
                                        }
                                        float pj_28 = (*(&kernelContext_0)->ss_0)[4U * _S12 + j_3];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S12 + j_3], _S106, lo_1[4U]);
                                        hi_1[4U] = fma(pj_28, _S107, hi_1[4U]);
                                        _S22 = int(7);
                                        break;
                                    }
                                    if(_S22 != int(7))
                                    {
                                        _S21 = _S22;
                                        break;
                                    }
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S5)
                                        {
                                            _S97 = int(5);
                                            break;
                                        }
                                        float pj_29 = (*(&kernelContext_0)->ss_0)[5U * _S12 + j_3];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S12 + j_3], _S106, lo_1[5U]);
                                        hi_1[5U] = fma(pj_29, _S107, hi_1[5U]);
                                        _S97 = int(7);
                                        break;
                                    }
                                    if(_S97 != int(7))
                                    {
                                        _S96 = _S97;
                                        break;
                                    }
                                    _S96 = int(6);
                                    break;
                                }
                                if(_S96 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S5)
                                        {
                                            _S101 = int(5);
                                            break;
                                        }
                                        float pj_30 = (*(&kernelContext_0)->ss_0)[6U * _S12 + j_3];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S12 + j_3], _S106, lo_1[6U]);
                                        hi_1[6U] = fma(pj_30, _S107, hi_1[6U]);
                                        _S101 = int(7);
                                        break;
                                    }
                                    if(_S101 != int(7))
                                    {
                                        _S100 = _S101;
                                        break;
                                    }
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S5)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_31 = (*(&kernelContext_0)->ss_0)[7U * _S12 + j_3];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S12 + j_3], _S106, lo_1[7U]);
                                        hi_1[7U] = fma(pj_31, _S107, hi_1[7U]);
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
                            _S30 = int(4);
                            break;
                        }
                        if(_S30 != int(4))
                        {
                            _S29 = _S30;
                            break;
                        }
                        _S29 = int(3);
                        break;
                    }
                    if(_S29 != int(3))
                    {
                        break;
                    }
                    int _S108;
                    int _S109;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_4 >= _S37)
                            {
                                _S28 = int(2);
                                break;
                            }
                            float _S110 = attnLo_0(vw_0[4U]);
                            float _S111 = attnHi_0(vw_0[4U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S28 = int(5);
                                            break;
                                        }
                                        float pj_32 = (*(&kernelContext_0)->ss_0)[j_4];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_4], _S110, lo_1[0U]);
                                        hi_1[0U] = fma(pj_32, _S111, hi_1[0U]);
                                        _S28 = int(7);
                                        break;
                                    }
                                    if(_S28 != int(7))
                                    {
                                        _S27 = _S28;
                                        break;
                                    }
                                    _S27 = int(6);
                                    break;
                                }
                                if(_S27 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S5)
                                        {
                                            _S26 = int(5);
                                            break;
                                        }
                                        float pj_33 = (*(&kernelContext_0)->ss_0)[_S12 + j_4];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S12 + j_4], _S110, lo_1[1U]);
                                        hi_1[1U] = fma(pj_33, _S111, hi_1[1U]);
                                        _S26 = int(7);
                                        break;
                                    }
                                    if(_S26 != int(7))
                                    {
                                        _S25 = _S26;
                                        break;
                                    }
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S5)
                                        {
                                            _S24 = int(5);
                                            break;
                                        }
                                        float pj_34 = (*(&kernelContext_0)->ss_0)[2U * _S12 + j_4];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S12 + j_4], _S110, lo_1[2U]);
                                        hi_1[2U] = fma(pj_34, _S111, hi_1[2U]);
                                        _S24 = int(7);
                                        break;
                                    }
                                    if(_S24 != int(7))
                                    {
                                        _S23 = _S24;
                                        break;
                                    }
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S5)
                                        {
                                            _S22 = int(5);
                                            break;
                                        }
                                        float pj_35 = (*(&kernelContext_0)->ss_0)[3U * _S12 + j_4];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S12 + j_4], _S110, lo_1[3U]);
                                        hi_1[3U] = fma(pj_35, _S111, hi_1[3U]);
                                        _S22 = int(7);
                                        break;
                                    }
                                    if(_S22 != int(7))
                                    {
                                        _S21 = _S22;
                                        break;
                                    }
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S5)
                                        {
                                            _S97 = int(5);
                                            break;
                                        }
                                        float pj_36 = (*(&kernelContext_0)->ss_0)[4U * _S12 + j_4];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S12 + j_4], _S110, lo_1[4U]);
                                        hi_1[4U] = fma(pj_36, _S111, hi_1[4U]);
                                        _S97 = int(7);
                                        break;
                                    }
                                    if(_S97 != int(7))
                                    {
                                        _S96 = _S97;
                                        break;
                                    }
                                    _S96 = int(6);
                                    break;
                                }
                                if(_S96 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S5)
                                        {
                                            _S101 = int(5);
                                            break;
                                        }
                                        float pj_37 = (*(&kernelContext_0)->ss_0)[5U * _S12 + j_4];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S12 + j_4], _S110, lo_1[5U]);
                                        hi_1[5U] = fma(pj_37, _S111, hi_1[5U]);
                                        _S101 = int(7);
                                        break;
                                    }
                                    if(_S101 != int(7))
                                    {
                                        _S100 = _S101;
                                        break;
                                    }
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S5)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_38 = (*(&kernelContext_0)->ss_0)[6U * _S12 + j_4];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S12 + j_4], _S110, lo_1[6U]);
                                        hi_1[6U] = fma(pj_38, _S111, hi_1[6U]);
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
                                        if(7U >= _S5)
                                        {
                                            _S109 = int(5);
                                            break;
                                        }
                                        float pj_39 = (*(&kernelContext_0)->ss_0)[7U * _S12 + j_4];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S12 + j_4], _S110, lo_1[7U]);
                                        hi_1[7U] = fma(pj_39, _S111, hi_1[7U]);
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
                            _S28 = int(4);
                            break;
                        }
                        if(_S28 != int(4))
                        {
                            _S27 = _S28;
                            break;
                        }
                        _S27 = int(3);
                        break;
                    }
                    if(_S27 != int(3))
                    {
                        break;
                    }
                    int _S112;
                    int _S113;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_5 >= _S37)
                            {
                                _S26 = int(2);
                                break;
                            }
                            float _S114 = attnLo_0(vw_0[5U]);
                            float _S115 = attnHi_0(vw_0[5U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S26 = int(5);
                                            break;
                                        }
                                        float pj_40 = (*(&kernelContext_0)->ss_0)[j_5];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_5], _S114, lo_1[0U]);
                                        hi_1[0U] = fma(pj_40, _S115, hi_1[0U]);
                                        _S26 = int(7);
                                        break;
                                    }
                                    if(_S26 != int(7))
                                    {
                                        _S25 = _S26;
                                        break;
                                    }
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S5)
                                        {
                                            _S24 = int(5);
                                            break;
                                        }
                                        float pj_41 = (*(&kernelContext_0)->ss_0)[_S12 + j_5];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S12 + j_5], _S114, lo_1[1U]);
                                        hi_1[1U] = fma(pj_41, _S115, hi_1[1U]);
                                        _S24 = int(7);
                                        break;
                                    }
                                    if(_S24 != int(7))
                                    {
                                        _S23 = _S24;
                                        break;
                                    }
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S5)
                                        {
                                            _S22 = int(5);
                                            break;
                                        }
                                        float pj_42 = (*(&kernelContext_0)->ss_0)[2U * _S12 + j_5];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S12 + j_5], _S114, lo_1[2U]);
                                        hi_1[2U] = fma(pj_42, _S115, hi_1[2U]);
                                        _S22 = int(7);
                                        break;
                                    }
                                    if(_S22 != int(7))
                                    {
                                        _S21 = _S22;
                                        break;
                                    }
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S5)
                                        {
                                            _S97 = int(5);
                                            break;
                                        }
                                        float pj_43 = (*(&kernelContext_0)->ss_0)[3U * _S12 + j_5];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S12 + j_5], _S114, lo_1[3U]);
                                        hi_1[3U] = fma(pj_43, _S115, hi_1[3U]);
                                        _S97 = int(7);
                                        break;
                                    }
                                    if(_S97 != int(7))
                                    {
                                        _S96 = _S97;
                                        break;
                                    }
                                    _S96 = int(6);
                                    break;
                                }
                                if(_S96 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S5)
                                        {
                                            _S101 = int(5);
                                            break;
                                        }
                                        float pj_44 = (*(&kernelContext_0)->ss_0)[4U * _S12 + j_5];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S12 + j_5], _S114, lo_1[4U]);
                                        hi_1[4U] = fma(pj_44, _S115, hi_1[4U]);
                                        _S101 = int(7);
                                        break;
                                    }
                                    if(_S101 != int(7))
                                    {
                                        _S100 = _S101;
                                        break;
                                    }
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S5)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_45 = (*(&kernelContext_0)->ss_0)[5U * _S12 + j_5];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S12 + j_5], _S114, lo_1[5U]);
                                        hi_1[5U] = fma(pj_45, _S115, hi_1[5U]);
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
                                        if(6U >= _S5)
                                        {
                                            _S109 = int(5);
                                            break;
                                        }
                                        float pj_46 = (*(&kernelContext_0)->ss_0)[6U * _S12 + j_5];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S12 + j_5], _S114, lo_1[6U]);
                                        hi_1[6U] = fma(pj_46, _S115, hi_1[6U]);
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
                                        if(7U >= _S5)
                                        {
                                            _S113 = int(5);
                                            break;
                                        }
                                        float pj_47 = (*(&kernelContext_0)->ss_0)[7U * _S12 + j_5];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S12 + j_5], _S114, lo_1[7U]);
                                        hi_1[7U] = fma(pj_47, _S115, hi_1[7U]);
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
                            _S26 = int(4);
                            break;
                        }
                        if(_S26 != int(4))
                        {
                            _S25 = _S26;
                            break;
                        }
                        _S25 = int(3);
                        break;
                    }
                    if(_S25 != int(3))
                    {
                        break;
                    }
                    int _S116;
                    int _S117;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_6 >= _S37)
                            {
                                _S24 = int(2);
                                break;
                            }
                            float _S118 = attnLo_0(vw_0[6U]);
                            float _S119 = attnHi_0(vw_0[6U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S24 = int(5);
                                            break;
                                        }
                                        float pj_48 = (*(&kernelContext_0)->ss_0)[j_6];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_6], _S118, lo_1[0U]);
                                        hi_1[0U] = fma(pj_48, _S119, hi_1[0U]);
                                        _S24 = int(7);
                                        break;
                                    }
                                    if(_S24 != int(7))
                                    {
                                        _S23 = _S24;
                                        break;
                                    }
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S5)
                                        {
                                            _S22 = int(5);
                                            break;
                                        }
                                        float pj_49 = (*(&kernelContext_0)->ss_0)[_S12 + j_6];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S12 + j_6], _S118, lo_1[1U]);
                                        hi_1[1U] = fma(pj_49, _S119, hi_1[1U]);
                                        _S22 = int(7);
                                        break;
                                    }
                                    if(_S22 != int(7))
                                    {
                                        _S21 = _S22;
                                        break;
                                    }
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S5)
                                        {
                                            _S97 = int(5);
                                            break;
                                        }
                                        float pj_50 = (*(&kernelContext_0)->ss_0)[2U * _S12 + j_6];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S12 + j_6], _S118, lo_1[2U]);
                                        hi_1[2U] = fma(pj_50, _S119, hi_1[2U]);
                                        _S97 = int(7);
                                        break;
                                    }
                                    if(_S97 != int(7))
                                    {
                                        _S96 = _S97;
                                        break;
                                    }
                                    _S96 = int(6);
                                    break;
                                }
                                if(_S96 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S5)
                                        {
                                            _S101 = int(5);
                                            break;
                                        }
                                        float pj_51 = (*(&kernelContext_0)->ss_0)[3U * _S12 + j_6];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S12 + j_6], _S118, lo_1[3U]);
                                        hi_1[3U] = fma(pj_51, _S119, hi_1[3U]);
                                        _S101 = int(7);
                                        break;
                                    }
                                    if(_S101 != int(7))
                                    {
                                        _S100 = _S101;
                                        break;
                                    }
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S5)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_52 = (*(&kernelContext_0)->ss_0)[4U * _S12 + j_6];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S12 + j_6], _S118, lo_1[4U]);
                                        hi_1[4U] = fma(pj_52, _S119, hi_1[4U]);
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
                                        if(5U >= _S5)
                                        {
                                            _S109 = int(5);
                                            break;
                                        }
                                        float pj_53 = (*(&kernelContext_0)->ss_0)[5U * _S12 + j_6];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S12 + j_6], _S118, lo_1[5U]);
                                        hi_1[5U] = fma(pj_53, _S119, hi_1[5U]);
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
                                        if(6U >= _S5)
                                        {
                                            _S113 = int(5);
                                            break;
                                        }
                                        float pj_54 = (*(&kernelContext_0)->ss_0)[6U * _S12 + j_6];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S12 + j_6], _S118, lo_1[6U]);
                                        hi_1[6U] = fma(pj_54, _S119, hi_1[6U]);
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
                                        if(7U >= _S5)
                                        {
                                            _S117 = int(5);
                                            break;
                                        }
                                        float pj_55 = (*(&kernelContext_0)->ss_0)[7U * _S12 + j_6];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S12 + j_6], _S118, lo_1[7U]);
                                        hi_1[7U] = fma(pj_55, _S119, hi_1[7U]);
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
                            _S24 = int(4);
                            break;
                        }
                        if(_S24 != int(4))
                        {
                            _S23 = _S24;
                            break;
                        }
                        _S23 = int(3);
                        break;
                    }
                    if(_S23 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_7 >= _S37)
                            {
                                _S22 = int(2);
                                break;
                            }
                            float _S120 = attnLo_0(vw_0[7U]);
                            float _S121 = attnHi_0(vw_0[7U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S22 = int(5);
                                            break;
                                        }
                                        float pj_56 = (*(&kernelContext_0)->ss_0)[j_7];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_7], _S120, lo_1[0U]);
                                        hi_1[0U] = fma(pj_56, _S121, hi_1[0U]);
                                        _S22 = int(7);
                                        break;
                                    }
                                    if(_S22 != int(7))
                                    {
                                        _S21 = _S22;
                                        break;
                                    }
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S5)
                                        {
                                            _S97 = int(5);
                                            break;
                                        }
                                        float pj_57 = (*(&kernelContext_0)->ss_0)[_S12 + j_7];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S12 + j_7], _S120, lo_1[1U]);
                                        hi_1[1U] = fma(pj_57, _S121, hi_1[1U]);
                                        _S97 = int(7);
                                        break;
                                    }
                                    if(_S97 != int(7))
                                    {
                                        _S96 = _S97;
                                        break;
                                    }
                                    _S96 = int(6);
                                    break;
                                }
                                if(_S96 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S5)
                                        {
                                            _S101 = int(5);
                                            break;
                                        }
                                        float pj_58 = (*(&kernelContext_0)->ss_0)[2U * _S12 + j_7];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S12 + j_7], _S120, lo_1[2U]);
                                        hi_1[2U] = fma(pj_58, _S121, hi_1[2U]);
                                        _S101 = int(7);
                                        break;
                                    }
                                    if(_S101 != int(7))
                                    {
                                        _S100 = _S101;
                                        break;
                                    }
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S5)
                                        {
                                            _S105 = int(5);
                                            break;
                                        }
                                        float pj_59 = (*(&kernelContext_0)->ss_0)[3U * _S12 + j_7];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S12 + j_7], _S120, lo_1[3U]);
                                        hi_1[3U] = fma(pj_59, _S121, hi_1[3U]);
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
                                        if(4U >= _S5)
                                        {
                                            _S109 = int(5);
                                            break;
                                        }
                                        float pj_60 = (*(&kernelContext_0)->ss_0)[4U * _S12 + j_7];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S12 + j_7], _S120, lo_1[4U]);
                                        hi_1[4U] = fma(pj_60, _S121, hi_1[4U]);
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
                                        if(5U >= _S5)
                                        {
                                            _S113 = int(5);
                                            break;
                                        }
                                        float pj_61 = (*(&kernelContext_0)->ss_0)[5U * _S12 + j_7];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S12 + j_7], _S120, lo_1[5U]);
                                        hi_1[5U] = fma(pj_61, _S121, hi_1[5U]);
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
                                        if(6U >= _S5)
                                        {
                                            _S117 = int(5);
                                            break;
                                        }
                                        float pj_62 = (*(&kernelContext_0)->ss_0)[6U * _S12 + j_7];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S12 + j_7], _S120, lo_1[6U]);
                                        hi_1[6U] = fma(pj_62, _S121, hi_1[6U]);
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
                                int _S122;
                                for(;;)
                                {
                                    int _S123;
                                    for(;;)
                                    {
                                        if(7U >= _S5)
                                        {
                                            _S123 = int(5);
                                            break;
                                        }
                                        float pj_63 = (*(&kernelContext_0)->ss_0)[7U * _S12 + j_7];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S12 + j_7], _S120, lo_1[7U]);
                                        hi_1[7U] = fma(pj_63, _S121, hi_1[7U]);
                                        _S123 = int(7);
                                        break;
                                    }
                                    if(_S123 != int(7))
                                    {
                                        _S122 = _S123;
                                        break;
                                    }
                                    _S122 = int(6);
                                    break;
                                }
                                if(_S122 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S22 = int(4);
                            break;
                        }
                        if(_S22 != int(4))
                        {
                            _S21 = _S22;
                            break;
                        }
                        _S21 = int(3);
                        break;
                    }
                    if(_S21 != int(3))
                    {
                        break;
                    }
                    break;
                }
                kw_6 = kw_6 + 32U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c0_0 = c0_0 + _S12;
    }
    r_0 = 0U;
    for(;;)
    {
        if(r_0 < _S5)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(0U == r_0)
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
        if(1U == r_0)
        {
            w_3 = lo_1[1U];
            b_0 = hi_1[1U];
        }
        else
        {
            w_3 = m_0;
            b_0 = sum_0;
        }
        float a_8;
        float b_1;
        if(2U == r_0)
        {
            a_8 = lo_1[2U];
            b_1 = hi_1[2U];
        }
        else
        {
            a_8 = w_3;
            b_1 = b_0;
        }
        float a_9;
        float b_2;
        if(3U == r_0)
        {
            a_9 = lo_1[3U];
            b_2 = hi_1[3U];
        }
        else
        {
            a_9 = a_8;
            b_2 = b_1;
        }
        float a_10;
        float b_3;
        if(4U == r_0)
        {
            a_10 = lo_1[4U];
            b_3 = hi_1[4U];
        }
        else
        {
            a_10 = a_9;
            b_3 = b_2;
        }
        float a_11;
        float b_4;
        if(5U == r_0)
        {
            a_11 = lo_1[5U];
            b_4 = hi_1[5U];
        }
        else
        {
            a_11 = a_10;
            b_4 = b_3;
        }
        float a_12;
        float b_5;
        if(6U == r_0)
        {
            a_12 = lo_1[6U];
            b_5 = hi_1[6U];
        }
        else
        {
            a_12 = a_11;
            b_5 = b_4;
        }
        float a_13;
        float b_6;
        if(7U == r_0)
        {
            a_13 = lo_1[7U];
            b_6 = hi_1[7U];
        }
        else
        {
            a_13 = a_12;
            b_6 = b_5;
        }
        uint _S124 = 2U * _S19;
        uint _S125 = _S20 * 64U + _S124;
        (*(&kernelContext_0)->ss_0)[_S125] = a_13;
        (*(&kernelContext_0)->ss_0)[_S125 + 1U] = b_6;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S20 != 0U)
        {
            visible_0 = true;
        }
        else
        {
            visible_0 = _S19 >= _S8;
        }
        if(visible_0)
        {
            r_0 = r_0 + 1U;
            continue;
        }
        uint _S126 = 64U + _S124;
        uint _S127 = 128U + _S124;
        uint _S128 = 192U + _S124;
        float a_14 = (*(&kernelContext_0)->ss_0)[_S124] + (*(&kernelContext_0)->ss_0)[_S126] + (*(&kernelContext_0)->ss_0)[_S127] + (*(&kernelContext_0)->ss_0)[_S128];
        float b_7 = (*(&kernelContext_0)->ss_0)[_S124 + 1U] + (*(&kernelContext_0)->ss_0)[_S126 + 1U] + (*(&kernelContext_0)->ss_0)[_S127 + 1U] + (*(&kernelContext_0)->ss_0)[_S128 + 1U];
        uint _S129 = _S4 * hpk_0;
        uint _S130 = r_0 / (&kernelContext_0)->p_0->queries_0;
        uint h_4 = _S129 + _S130;
        uint i_9 = r_0 % (&kernelContext_0)->p_0->queries_0;
        uint row_0 = _S6 + h_4 * (&kernelContext_0)->p_0->queries_0 + i_9;
        if(((&kernelContext_0)->p_0->splits_0) > 1U)
        {
            uint pr_8 = split_0 * _S7 + row_0;
            if(_S19 == 0U)
            {
                uint _S131 = 2U * pr_8;
                float device* _S132 = (&kernelContext_0)->partialML_0+_S131;
                bool _S133 = keys_1 == 0U;
                float _S134;
                if(_S133)
                {
                    _S134 = dg_negative_infinity_0();
                }
                else
                {
                    _S134 = (*(&kernelContext_0)->rowM_0)[r_0];
                }
                *_S132 = _S134;
                float device* _S135 = (&kernelContext_0)->partialML_0+(_S131 + 1U);
                float _S136;
                if(_S133)
                {
                    _S136 = 0.0;
                }
                else
                {
                    _S136 = (*(&kernelContext_0)->rowL_0)[r_0];
                }
                *_S135 = _S136;
            }
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S124)) = a_14;
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S124 + 1U)) = b_7;
            r_0 = r_0 + 1U;
            continue;
        }
        *((&kernelContext_0)->o_0+(row_0 * _S8 + _S19)) = attnPack_0(a_14 / (*(&kernelContext_0)->rowL_0)[r_0], b_7 / (*(&kernelContext_0)->rowL_0)[r_0]);
        r_0 = r_0 + 1U;
    }
    return;
}

