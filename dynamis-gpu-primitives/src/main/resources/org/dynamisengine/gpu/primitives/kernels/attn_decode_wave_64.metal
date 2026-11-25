#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant uint fc_ROW_TIER_0 [[function_constant(0)]];
constant uint ROW_TIER_0 = is_function_constant_defined(fc_ROW_TIER_0) ? fc_ROW_TIER_0 : 8U;
uint attnKeyEnd_0(uint device* lens_0, uint lengthGroup_0, uint firstHead_0, uint lastHead_0, uint end_0)
{
    if(lengthGroup_0 == 0U)
    {
        return end_0;
    }
    uint first_0 = firstHead_0 / lengthGroup_0;
    uint _S1 = lastHead_0 / lengthGroup_0;
    uint _S2 = first_0 + 1U;
    uint longest_0 = *(lens_0+first_0);
    uint i_0 = _S2;
    for(;;)
    {
        if(i_0 <= _S1)
        {
        }
        else
        {
            break;
        }
        uint _S3 = max(longest_0, *(lens_0+i_0));
        uint _S4 = i_0 + 1U;
        longest_0 = _S3;
        i_0 = _S4;
    }
    return min(end_0, longest_0);
}

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

uint attnAccBase_0(uint splits_0, uint rows_0)
{
    return 2U * splits_0 * rows_0;
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
        uint _S5 = sign_0 | 31744U;
        if(mant_0 != 0U)
        {
            h_0 = 512U | (mant_0 >> 13U);
        }
        else
        {
            h_0 = 0U;
        }
        return _S5 | h_0;
    }
    int exp_0 = int(e_0) - int(112);
    if(exp_0 >= int(31))
    {
        return sign_0 | 31744U;
    }
    bool _S6;
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
            _S6 = true;
        }
        else
        {
            if(rem_0 == halfway_0)
            {
                _S6 = (h_1 & 1U) != 0U;
            }
            else
            {
                _S6 = false;
            }
        }
        if(_S6)
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
        _S6 = true;
    }
    else
    {
        if(rem_1 == 4096U)
        {
            _S6 = (h_2 & 1U) != 0U;
        }
        else
        {
            _S6 = false;
        }
    }
    if(_S6)
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
    uint splits_1;
    uint splitKeys_0;
    uint rowBlocks_0;
    uint pageSize_0;
    uint batch_0;
    uint writeLse_0;
    uint lengthGroup_1;
};

struct KernelContext_0
{
    AttnParams_0 constant* p_0;
    uint device* table_0;
    uint device* lens_1;
    uint device* q_0;
    uint device* k_0;
    uint device* v_0;
    float device* partials_0;
    uint device* o_0;
    uint threadgroup* waves_0;
    array<float, int(8)> threadgroup* rowM_0;
    array<float, int(8)> threadgroup* rowL_0;
    array<uint, int(256)> threadgroup* qs_0;
    array<float, int(1024)> threadgroup* ss_0;
    array<float, int(8)> threadgroup* rowA_0;
};

[[kernel]] void attn_decode_wave_64(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(7)]], uint device* table_1 [[buffer(5)]], uint device* lens_2 [[buffer(6)]], uint device* q_1 [[buffer(0)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partials_1 [[buffer(4)]], uint device* o_1 [[buffer(3)]])
{
    bool _S7;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->table_0 = table_1;
    (&kernelContext_0)->lens_1 = lens_2;
    (&kernelContext_0)->q_0 = q_1;
    (&kernelContext_0)->k_0 = k_1;
    (&kernelContext_0)->v_0 = v_1;
    (&kernelContext_0)->partials_0 = partials_1;
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
    uint kvHead_0 = group_0.y;
    uint split_0 = group_0.z;
    uint seq_0 = group_0.x;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint _S8 = min(hpk_0 * (&kernelContext_0)->p_0->queries_0, ROW_TIER_0);
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
    uint _S9 = seq_0 * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0;
    uint _S10 = max((&kernelContext_0)->p_0->batch_0, 1U) * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0;
    uint _S11 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint _S12 = keys_1 - (&kernelContext_0)->p_0->queries_0;
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S13 = kvHead_0 * hpk_0;
    uint firstHead_1 = seq_0 * p_1->heads_0 + _S13;
    uint kEnd_0 = attnKeyEnd_0((&kernelContext_0)->lens_1, (&kernelContext_0)->p_0->lengthGroup_1, firstHead_1, firstHead_1 + hpk_0 - 1U, min(keys_1, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0));
    uint _S14;
    if(kEnd_0 > kStart_0)
    {
        _S14 = kEnd_0 - kStart_0;
    }
    else
    {
        _S14 = 0U;
    }
    uint _S15 = 1024U / max(_S8, 1U);
    uint _S16 = max(128U, _S15 / 128U * 128U);
    thread array<uint, int(8)> limit_0;
    uint e_1;
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint _S17 = 0U % (&kernelContext_0)->p_0->queries_0;
        e_1 = _S17 + _S12 + 1U;
    }
    else
    {
        e_1 = keys_1;
    }
    limit_0[0U] = e_1;
    bool _S18 = 0U < _S8;
    bool _S19;
    if(_S18)
    {
        _S19 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
    }
    else
    {
        _S19 = false;
    }
    if(_S19)
    {
        uint _S20 = limit_0[0U];
        uint _S21 = firstHead_1 / (&kernelContext_0)->p_0->lengthGroup_1;
        limit_0[0U] = min(_S20, *((&kernelContext_0)->lens_1+_S21));
    }
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint _S22 = 1U % (&kernelContext_0)->p_0->queries_0;
        e_1 = _S22 + _S12 + 1U;
    }
    else
    {
        e_1 = keys_1;
    }
    limit_0[1U] = e_1;
    bool _S23 = 1U < _S8;
    if(_S23)
    {
        _S19 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
    }
    else
    {
        _S19 = false;
    }
    if(_S19)
    {
        uint _S24 = limit_0[1U];
        uint _S25 = 1U / (&kernelContext_0)->p_0->queries_0;
        uint _S26 = (firstHead_1 + _S25) / (&kernelContext_0)->p_0->lengthGroup_1;
        limit_0[1U] = min(_S24, *((&kernelContext_0)->lens_1+_S26));
    }
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint _S27 = 2U % (&kernelContext_0)->p_0->queries_0;
        e_1 = _S27 + _S12 + 1U;
    }
    else
    {
        e_1 = keys_1;
    }
    limit_0[2U] = e_1;
    bool _S28 = 2U < _S8;
    if(_S28)
    {
        _S19 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
    }
    else
    {
        _S19 = false;
    }
    if(_S19)
    {
        uint _S29 = limit_0[2U];
        uint _S30 = 2U / (&kernelContext_0)->p_0->queries_0;
        uint _S31 = (firstHead_1 + _S30) / (&kernelContext_0)->p_0->lengthGroup_1;
        limit_0[2U] = min(_S29, *((&kernelContext_0)->lens_1+_S31));
    }
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint _S32 = 3U % (&kernelContext_0)->p_0->queries_0;
        e_1 = _S32 + _S12 + 1U;
    }
    else
    {
        e_1 = keys_1;
    }
    limit_0[3U] = e_1;
    bool _S33 = 3U < _S8;
    if(_S33)
    {
        _S19 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
    }
    else
    {
        _S19 = false;
    }
    if(_S19)
    {
        uint _S34 = limit_0[3U];
        uint _S35 = 3U / (&kernelContext_0)->p_0->queries_0;
        uint _S36 = (firstHead_1 + _S35) / (&kernelContext_0)->p_0->lengthGroup_1;
        limit_0[3U] = min(_S34, *((&kernelContext_0)->lens_1+_S36));
    }
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint _S37 = 4U % (&kernelContext_0)->p_0->queries_0;
        e_1 = _S37 + _S12 + 1U;
    }
    else
    {
        e_1 = keys_1;
    }
    limit_0[4U] = e_1;
    bool _S38 = 4U < _S8;
    if(_S38)
    {
        _S19 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
    }
    else
    {
        _S19 = false;
    }
    if(_S19)
    {
        uint _S39 = limit_0[4U];
        uint _S40 = 4U / (&kernelContext_0)->p_0->queries_0;
        uint _S41 = (firstHead_1 + _S40) / (&kernelContext_0)->p_0->lengthGroup_1;
        limit_0[4U] = min(_S39, *((&kernelContext_0)->lens_1+_S41));
    }
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint _S42 = 5U % (&kernelContext_0)->p_0->queries_0;
        e_1 = _S42 + _S12 + 1U;
    }
    else
    {
        e_1 = keys_1;
    }
    limit_0[5U] = e_1;
    bool _S43 = 5U < _S8;
    if(_S43)
    {
        _S19 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
    }
    else
    {
        _S19 = false;
    }
    if(_S19)
    {
        uint _S44 = limit_0[5U];
        uint _S45 = 5U / (&kernelContext_0)->p_0->queries_0;
        uint _S46 = (firstHead_1 + _S45) / (&kernelContext_0)->p_0->lengthGroup_1;
        limit_0[5U] = min(_S44, *((&kernelContext_0)->lens_1+_S46));
    }
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint _S47 = 6U % (&kernelContext_0)->p_0->queries_0;
        e_1 = _S47 + _S12 + 1U;
    }
    else
    {
        e_1 = keys_1;
    }
    limit_0[6U] = e_1;
    bool _S48 = 6U < _S8;
    if(_S48)
    {
        _S19 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
    }
    else
    {
        _S19 = false;
    }
    if(_S19)
    {
        uint _S49 = limit_0[6U];
        uint _S50 = 6U / (&kernelContext_0)->p_0->queries_0;
        uint _S51 = (firstHead_1 + _S50) / (&kernelContext_0)->p_0->lengthGroup_1;
        limit_0[6U] = min(_S49, *((&kernelContext_0)->lens_1+_S51));
    }
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint _S52 = 7U % (&kernelContext_0)->p_0->queries_0;
        keys_1 = _S52 + _S12 + 1U;
    }
    limit_0[7U] = keys_1;
    bool _S53 = 7U < _S8;
    if(_S53)
    {
        _S19 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
    }
    else
    {
        _S19 = false;
    }
    if(_S19)
    {
        uint _S54 = limit_0[7U];
        uint _S55 = 7U / (&kernelContext_0)->p_0->queries_0;
        uint _S56 = (firstHead_1 + _S55) / (&kernelContext_0)->p_0->lengthGroup_1;
        limit_0[7U] = min(_S54, *((&kernelContext_0)->lens_1+_S56));
    }
    if(t_0 == 0U)
    {
        *(&kernelContext_0)->waves_0 = 0U;
    }
    if(t_0 < 8U)
    {
        (*(&kernelContext_0)->rowM_0)[t_0] = dg_negative_infinity_0();
        (*(&kernelContext_0)->rowL_0)[t_0] = 0.0;
    }
    uint w_2;
    e_1 = t_0;
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
        if(r_1 < _S8)
        {
            _S19 = pr_0 < _S11;
        }
        else
        {
            _S19 = false;
        }
        if(_S19)
        {
            uint _S57 = r_1 / (&kernelContext_0)->p_0->queries_0;
            uint h_3 = _S13 + _S57;
            uint i_1 = r_1 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((_S9 + h_3 * (&kernelContext_0)->p_0->queries_0 + i_1) * _S11 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_1] = w_2;
        e_1 = e_1 + 128U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S58 = simd_sum(1U);
    uint lane_0 = simd_prefix_exclusive_sum(1U);
    bool _S59 = lane_0 == 0U;
    if(_S59)
    {
        uint _S60 = atomic_fetch_add_explicit(((atomic_uint threadgroup*)((&kernelContext_0)->waves_0)), 1U, memory_order_relaxed);
        w_2 = _S60;
    }
    else
    {
        w_2 = 0U;
    }
    uint _S61 = simd_sum(w_2);
    uint _S62 = t_0 % 32U;
    uint _S63 = t_0 / 32U;
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
        if(c0_0 < kEnd_0)
        {
        }
        else
        {
            break;
        }
        int _S64;
        int _S65;
        int _S66;
        int _S67;
        int _S68;
        int _S69;
        int _S70;
        int _S71;
        int _S72;
        int _S73;
        int _S74;
        int _S75;
        int _S76;
        int _S77;
        int _S78;
        int _S79;
        uint kw_0;
        uint kw_1;
        uint kw_2;
        uint kw_3;
        uint kw_4;
        uint kw_5;
        uint p0_0;
        uint _S80 = min(_S16, kEnd_0 - c0_0);
        uint j_0 = t_0;
        for(;;)
        {
            if(j_0 < _S80)
            {
            }
            else
            {
                break;
            }
            uint key_0 = c0_0 + j_0;
            if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
            {
                keys_1 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * _S11;
            }
            else
            {
                uint _S81 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
                uint _S82 = (*((&kernelContext_0)->table_0+(r_0 + _S81)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                uint _S83 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
                keys_1 = (_S82 + _S83) * _S11;
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
                if(p0_0 < _S11)
                {
                    kw_5 = *((&kernelContext_0)->k_0+(keys_1 + p0_0));
                }
                else
                {
                    kw_5 = 0U;
                }
                float _S84 = attnLo_0(kw_5);
                float _S85 = attnHi_0(kw_5);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[p0_0]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[p0_0]), _S84, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + p0_0]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + p0_0]), _S84, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S84, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + p0_0]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + p0_0]), _S84, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S84, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + p0_0]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + p0_0]), _S84, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S84, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + p0_0]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + p0_0]), _S84, s_0[7U]));
                }
                uint pr_1 = p0_0 + 1U;
                if(pr_1 < _S11)
                {
                    kw_4 = *((&kernelContext_0)->k_0+(keys_1 + pr_1));
                }
                else
                {
                    kw_4 = 0U;
                }
                float _S86 = attnLo_0(kw_4);
                float _S87 = attnHi_0(kw_4);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_1]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_1]), _S86, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_1]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_1]), _S86, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S86, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_1]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_1]), _S86, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S86, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_1]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_1]), _S86, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S86, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_1]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_1]), _S86, s_0[7U]));
                }
                uint pr_2 = p0_0 + 2U;
                if(pr_2 < _S11)
                {
                    kw_3 = *((&kernelContext_0)->k_0+(keys_1 + pr_2));
                }
                else
                {
                    kw_3 = 0U;
                }
                float _S88 = attnLo_0(kw_3);
                float _S89 = attnHi_0(kw_3);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_2]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_2]), _S88, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_2]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_2]), _S88, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S88, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_2]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_2]), _S88, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S88, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_2]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_2]), _S88, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S88, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_2]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_2]), _S88, s_0[7U]));
                }
                uint pr_3 = p0_0 + 3U;
                if(pr_3 < _S11)
                {
                    kw_2 = *((&kernelContext_0)->k_0+(keys_1 + pr_3));
                }
                else
                {
                    kw_2 = 0U;
                }
                float _S90 = attnLo_0(kw_2);
                float _S91 = attnHi_0(kw_2);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_3]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_3]), _S90, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_3]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_3]), _S90, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S90, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_3]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_3]), _S90, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S90, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_3]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_3]), _S90, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S90, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_3]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_3]), _S90, s_0[7U]));
                }
                uint pr_4 = p0_0 + 4U;
                if(pr_4 < _S11)
                {
                    kw_1 = *((&kernelContext_0)->k_0+(keys_1 + pr_4));
                }
                else
                {
                    kw_1 = 0U;
                }
                float _S92 = attnLo_0(kw_1);
                float _S93 = attnHi_0(kw_1);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_4]), _S93, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_4]), _S92, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_4]), _S93, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_4]), _S92, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S93, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S92, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_4]), _S93, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_4]), _S92, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S93, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S92, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_4]), _S93, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_4]), _S92, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S93, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S92, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_4]), _S93, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_4]), _S92, s_0[7U]));
                }
                uint pr_5 = p0_0 + 5U;
                if(pr_5 < _S11)
                {
                    kw_0 = *((&kernelContext_0)->k_0+(keys_1 + pr_5));
                }
                else
                {
                    kw_0 = 0U;
                }
                float _S94 = attnLo_0(kw_0);
                float _S95 = attnHi_0(kw_0);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_5]), _S95, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_5]), _S94, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_5]), _S95, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_5]), _S94, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S95, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S94, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_5]), _S95, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_5]), _S94, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S95, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S94, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_5]), _S95, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_5]), _S94, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S95, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S94, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_5]), _S95, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_5]), _S94, s_0[7U]));
                }
                uint pr_6 = p0_0 + 6U;
                uint kw_6;
                if(pr_6 < _S11)
                {
                    kw_6 = *((&kernelContext_0)->k_0+(keys_1 + pr_6));
                }
                else
                {
                    kw_6 = 0U;
                }
                float _S96 = attnLo_0(kw_6);
                float _S97 = attnHi_0(kw_6);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_6]), _S97, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_6]), _S96, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_6]), _S97, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_6]), _S96, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S97, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S96, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_6]), _S97, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_6]), _S96, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S97, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S96, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_6]), _S97, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_6]), _S96, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S97, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S96, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_6]), _S97, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_6]), _S96, s_0[7U]));
                }
                uint pr_7 = p0_0 + 7U;
                uint kw_7;
                if(pr_7 < _S11)
                {
                    kw_7 = *((&kernelContext_0)->k_0+(keys_1 + pr_7));
                }
                else
                {
                    kw_7 = 0U;
                }
                float _S98 = attnLo_0(kw_7);
                float _S99 = attnHi_0(kw_7);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_7]), _S99, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_7]), _S98, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_7]), _S99, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_7]), _S98, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S99, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S98, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_7]), _S99, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_7]), _S98, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S99, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S98, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_7]), _S99, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_7]), _S98, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S99, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S98, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_7]), _S99, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_7]), _S98, s_0[7U]));
                }
                p0_0 = p0_0 + 8U;
            }
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        if(0U >= _S8)
                        {
                            _S79 = int(2);
                            break;
                        }
                        if(key_0 < limit_0[0U])
                        {
                            m_0 = s_0[0U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[j_0] = m_0;
                        _S79 = int(4);
                        break;
                    }
                    if(_S79 != int(4))
                    {
                        _S78 = _S79;
                        break;
                    }
                    _S78 = int(3);
                    break;
                }
                if(_S78 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(1U >= _S8)
                        {
                            _S77 = int(2);
                            break;
                        }
                        if(key_0 < limit_0[1U])
                        {
                            m_0 = s_0[1U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[_S16 + j_0] = m_0;
                        _S77 = int(4);
                        break;
                    }
                    if(_S77 != int(4))
                    {
                        _S76 = _S77;
                        break;
                    }
                    _S76 = int(3);
                    break;
                }
                if(_S76 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(2U >= _S8)
                        {
                            _S75 = int(2);
                            break;
                        }
                        if(key_0 < limit_0[2U])
                        {
                            m_0 = s_0[2U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[2U * _S16 + j_0] = m_0;
                        _S75 = int(4);
                        break;
                    }
                    if(_S75 != int(4))
                    {
                        _S74 = _S75;
                        break;
                    }
                    _S74 = int(3);
                    break;
                }
                if(_S74 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(3U >= _S8)
                        {
                            _S73 = int(2);
                            break;
                        }
                        if(key_0 < limit_0[3U])
                        {
                            m_0 = s_0[3U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[3U * _S16 + j_0] = m_0;
                        _S73 = int(4);
                        break;
                    }
                    if(_S73 != int(4))
                    {
                        _S72 = _S73;
                        break;
                    }
                    _S72 = int(3);
                    break;
                }
                if(_S72 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(4U >= _S8)
                        {
                            _S71 = int(2);
                            break;
                        }
                        if(key_0 < limit_0[4U])
                        {
                            m_0 = s_0[4U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[4U * _S16 + j_0] = m_0;
                        _S71 = int(4);
                        break;
                    }
                    if(_S71 != int(4))
                    {
                        _S70 = _S71;
                        break;
                    }
                    _S70 = int(3);
                    break;
                }
                if(_S70 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(5U >= _S8)
                        {
                            _S69 = int(2);
                            break;
                        }
                        if(key_0 < limit_0[5U])
                        {
                            m_0 = s_0[5U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[5U * _S16 + j_0] = m_0;
                        _S69 = int(4);
                        break;
                    }
                    if(_S69 != int(4))
                    {
                        _S68 = _S69;
                        break;
                    }
                    _S68 = int(3);
                    break;
                }
                if(_S68 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(6U >= _S8)
                        {
                            _S67 = int(2);
                            break;
                        }
                        if(key_0 < limit_0[6U])
                        {
                            m_0 = s_0[6U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[6U * _S16 + j_0] = m_0;
                        _S67 = int(4);
                        break;
                    }
                    if(_S67 != int(4))
                    {
                        _S66 = _S67;
                        break;
                    }
                    _S66 = int(3);
                    break;
                }
                if(_S66 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(7U >= _S8)
                        {
                            _S65 = int(2);
                            break;
                        }
                        if(key_0 < limit_0[7U])
                        {
                            m_0 = s_0[7U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[7U * _S16 + j_0] = m_0;
                        _S65 = int(4);
                        break;
                    }
                    if(_S65 != int(4))
                    {
                        _S64 = _S65;
                        break;
                    }
                    _S64 = int(3);
                    break;
                }
                if(_S64 != int(3))
                {
                    break;
                }
                break;
            }
            j_0 = j_0 + 128U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        p0_0 = _S61;
        for(;;)
        {
            if(p0_0 < _S8)
            {
            }
            else
            {
                break;
            }
            float _S100 = dg_negative_infinity_0();
            kw_5 = lane_0;
            m_0 = _S100;
            for(;;)
            {
                if(kw_5 < _S80)
                {
                }
                else
                {
                    break;
                }
                float _S101 = max(m_0, (*(&kernelContext_0)->ss_0)[p0_0 * _S16 + kw_5]);
                kw_5 = kw_5 + _S58;
                m_0 = _S101;
            }
            float m_1 = simd_max(m_0);
            float mOld_0 = (*(&kernelContext_0)->rowM_0)[p0_0];
            float _S102 = max((*(&kernelContext_0)->rowM_0)[p0_0], m_1);
            kw_4 = lane_0;
            sum_0 = 0.0;
            for(;;)
            {
                if(kw_4 < _S80)
                {
                }
                else
                {
                    break;
                }
                float sv_0 = (*(&kernelContext_0)->ss_0)[p0_0 * _S16 + kw_4];
                if(((*(&kernelContext_0)->ss_0)[p0_0 * _S16 + kw_4]) == _S100)
                {
                    w_3 = 0.0;
                }
                else
                {
                    w_3 = exp(sv_0 - _S102);
                }
                (*(&kernelContext_0)->ss_0)[p0_0 * _S16 + kw_4] = w_3;
                float sum_1 = sum_0 + w_3;
                kw_4 = kw_4 + _S58;
                sum_0 = sum_1;
            }
            float sum_2 = simd_sum(sum_0);
            if(mOld_0 == _S100)
            {
                w_3 = 0.0;
            }
            else
            {
                w_3 = exp(mOld_0 - _S102);
            }
            if(_S59)
            {
                (*(&kernelContext_0)->rowA_0)[p0_0] = w_3;
                (*(&kernelContext_0)->rowL_0)[p0_0] = (*(&kernelContext_0)->rowL_0)[p0_0] * w_3 + sum_2;
                (*(&kernelContext_0)->rowM_0)[p0_0] = _S102;
            }
            p0_0 = p0_0 + *(&kernelContext_0)->waves_0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S18)
        {
            float a_0 = (*(&kernelContext_0)->rowA_0)[0U];
            lo_1[0U] = lo_1[0U] * (*(&kernelContext_0)->rowA_0)[0U];
            hi_1[0U] = hi_1[0U] * a_0;
        }
        if(_S23)
        {
            float a_1 = (*(&kernelContext_0)->rowA_0)[1U];
            lo_1[1U] = lo_1[1U] * (*(&kernelContext_0)->rowA_0)[1U];
            hi_1[1U] = hi_1[1U] * a_1;
        }
        if(_S28)
        {
            float a_2 = (*(&kernelContext_0)->rowA_0)[2U];
            lo_1[2U] = lo_1[2U] * (*(&kernelContext_0)->rowA_0)[2U];
            hi_1[2U] = hi_1[2U] * a_2;
        }
        if(_S33)
        {
            float a_3 = (*(&kernelContext_0)->rowA_0)[3U];
            lo_1[3U] = lo_1[3U] * (*(&kernelContext_0)->rowA_0)[3U];
            hi_1[3U] = hi_1[3U] * a_3;
        }
        if(_S38)
        {
            float a_4 = (*(&kernelContext_0)->rowA_0)[4U];
            lo_1[4U] = lo_1[4U] * (*(&kernelContext_0)->rowA_0)[4U];
            hi_1[4U] = hi_1[4U] * a_4;
        }
        if(_S43)
        {
            float a_5 = (*(&kernelContext_0)->rowA_0)[5U];
            lo_1[5U] = lo_1[5U] * (*(&kernelContext_0)->rowA_0)[5U];
            hi_1[5U] = hi_1[5U] * a_5;
        }
        if(_S48)
        {
            float a_6 = (*(&kernelContext_0)->rowA_0)[6U];
            lo_1[6U] = lo_1[6U] * (*(&kernelContext_0)->rowA_0)[6U];
            hi_1[6U] = hi_1[6U] * a_6;
        }
        if(_S53)
        {
            float a_7 = (*(&kernelContext_0)->rowA_0)[7U];
            lo_1[7U] = lo_1[7U] * (*(&kernelContext_0)->rowA_0)[7U];
            hi_1[7U] = hi_1[7U] * a_7;
        }
        if(_S62 < _S11)
        {
            kw_5 = _S63;
            for(;;)
            {
                bool _S103 = kw_5 < _S80;
                if(_S103)
                {
                }
                else
                {
                    break;
                }
                thread array<uint, int(8)> vw_0;
                if(_S103)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        keys_1 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + kw_5)) * _S11 + _S62;
                    }
                    else
                    {
                        uint _S104 = c0_0 + kw_5;
                        uint _S105 = _S104 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S106 = (*((&kernelContext_0)->table_0+(r_0 + _S105)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S107 = _S104 % (&kernelContext_0)->p_0->pageSize_0;
                        keys_1 = (_S106 + _S107) * _S11 + _S62;
                    }
                    keys_1 = *((&kernelContext_0)->v_0+keys_1);
                }
                else
                {
                    keys_1 = 0U;
                }
                vw_0[0U] = keys_1;
                uint j_1 = kw_5 + 4U;
                if(j_1 < _S80)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        e_1 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_1)) * _S11 + _S62;
                    }
                    else
                    {
                        uint _S108 = c0_0 + j_1;
                        uint _S109 = _S108 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S110 = (*((&kernelContext_0)->table_0+(r_0 + _S109)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S111 = _S108 % (&kernelContext_0)->p_0->pageSize_0;
                        e_1 = (_S110 + _S111) * _S11 + _S62;
                    }
                    e_1 = *((&kernelContext_0)->v_0+e_1);
                }
                else
                {
                    e_1 = 0U;
                }
                vw_0[1U] = e_1;
                uint j_2 = kw_5 + 8U;
                if(j_2 < _S80)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        w_2 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_2)) * _S11 + _S62;
                    }
                    else
                    {
                        uint _S112 = c0_0 + j_2;
                        uint _S113 = _S112 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S114 = (*((&kernelContext_0)->table_0+(r_0 + _S113)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S115 = _S112 % (&kernelContext_0)->p_0->pageSize_0;
                        w_2 = (_S114 + _S115) * _S11 + _S62;
                    }
                    w_2 = *((&kernelContext_0)->v_0+w_2);
                }
                else
                {
                    w_2 = 0U;
                }
                vw_0[2U] = w_2;
                uint j_3 = kw_5 + 12U;
                if(j_3 < _S80)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_4 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_3)) * _S11 + _S62;
                    }
                    else
                    {
                        uint _S116 = c0_0 + j_3;
                        uint _S117 = _S116 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S118 = (*((&kernelContext_0)->table_0+(r_0 + _S117)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S119 = _S116 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_4 = (_S118 + _S119) * _S11 + _S62;
                    }
                    kw_4 = *((&kernelContext_0)->v_0+kw_4);
                }
                else
                {
                    kw_4 = 0U;
                }
                vw_0[3U] = kw_4;
                uint j_4 = kw_5 + 16U;
                if(j_4 < _S80)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_3 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_4)) * _S11 + _S62;
                    }
                    else
                    {
                        uint _S120 = c0_0 + j_4;
                        uint _S121 = _S120 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S122 = (*((&kernelContext_0)->table_0+(r_0 + _S121)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S123 = _S120 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_3 = (_S122 + _S123) * _S11 + _S62;
                    }
                    kw_3 = *((&kernelContext_0)->v_0+kw_3);
                }
                else
                {
                    kw_3 = 0U;
                }
                vw_0[4U] = kw_3;
                uint j_5 = kw_5 + 20U;
                if(j_5 < _S80)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_2 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_5)) * _S11 + _S62;
                    }
                    else
                    {
                        uint _S124 = c0_0 + j_5;
                        uint _S125 = _S124 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S126 = (*((&kernelContext_0)->table_0+(r_0 + _S125)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S127 = _S124 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_2 = (_S126 + _S127) * _S11 + _S62;
                    }
                    kw_2 = *((&kernelContext_0)->v_0+kw_2);
                }
                else
                {
                    kw_2 = 0U;
                }
                vw_0[5U] = kw_2;
                uint j_6 = kw_5 + 24U;
                if(j_6 < _S80)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_1 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_6)) * _S11 + _S62;
                    }
                    else
                    {
                        uint _S128 = c0_0 + j_6;
                        uint _S129 = _S128 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S130 = (*((&kernelContext_0)->table_0+(r_0 + _S129)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S131 = _S128 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_1 = (_S130 + _S131) * _S11 + _S62;
                    }
                    kw_1 = *((&kernelContext_0)->v_0+kw_1);
                }
                else
                {
                    kw_1 = 0U;
                }
                vw_0[6U] = kw_1;
                uint j_7 = kw_5 + 28U;
                if(j_7 < _S80)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_0 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_7)) * _S11 + _S62;
                    }
                    else
                    {
                        uint _S132 = c0_0 + j_7;
                        uint _S133 = _S132 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S134 = (*((&kernelContext_0)->table_0+(r_0 + _S133)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S135 = _S132 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_0 = (_S134 + _S135) * _S11 + _S62;
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
                            if(kw_5 >= _S80)
                            {
                                _S79 = int(2);
                                _S19 = false;
                                break;
                            }
                            float _S136 = attnLo_0(vw_0[0U]);
                            float _S137 = attnHi_0(vw_0[0U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        bool _S138 = 0U >= _S8;
                                        _S7 = _S138;
                                        if(_S138)
                                        {
                                            _S79 = int(5);
                                            break;
                                        }
                                        float pj_0 = (*(&kernelContext_0)->ss_0)[kw_5];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[kw_5], _S136, lo_1[0U]);
                                        hi_1[0U] = fma(pj_0, _S137, hi_1[0U]);
                                        _S79 = int(7);
                                        break;
                                    }
                                    if(_S79 != int(7))
                                    {
                                        _S78 = _S79;
                                        break;
                                    }
                                    _S78 = int(6);
                                    break;
                                }
                                if(_S78 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S8)
                                        {
                                            _S77 = int(5);
                                            break;
                                        }
                                        float pj_1 = (*(&kernelContext_0)->ss_0)[_S16 + kw_5];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S16 + kw_5], _S136, lo_1[1U]);
                                        hi_1[1U] = fma(pj_1, _S137, hi_1[1U]);
                                        _S77 = int(7);
                                        break;
                                    }
                                    if(_S77 != int(7))
                                    {
                                        _S76 = _S77;
                                        break;
                                    }
                                    _S76 = int(6);
                                    break;
                                }
                                if(_S76 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S8)
                                        {
                                            _S75 = int(5);
                                            break;
                                        }
                                        float pj_2 = (*(&kernelContext_0)->ss_0)[2U * _S16 + kw_5];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S16 + kw_5], _S136, lo_1[2U]);
                                        hi_1[2U] = fma(pj_2, _S137, hi_1[2U]);
                                        _S75 = int(7);
                                        break;
                                    }
                                    if(_S75 != int(7))
                                    {
                                        _S74 = _S75;
                                        break;
                                    }
                                    _S74 = int(6);
                                    break;
                                }
                                if(_S74 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S8)
                                        {
                                            _S73 = int(5);
                                            break;
                                        }
                                        float pj_3 = (*(&kernelContext_0)->ss_0)[3U * _S16 + kw_5];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S16 + kw_5], _S136, lo_1[3U]);
                                        hi_1[3U] = fma(pj_3, _S137, hi_1[3U]);
                                        _S73 = int(7);
                                        break;
                                    }
                                    if(_S73 != int(7))
                                    {
                                        _S72 = _S73;
                                        break;
                                    }
                                    _S72 = int(6);
                                    break;
                                }
                                if(_S72 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S8)
                                        {
                                            _S71 = int(5);
                                            break;
                                        }
                                        float pj_4 = (*(&kernelContext_0)->ss_0)[4U * _S16 + kw_5];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S16 + kw_5], _S136, lo_1[4U]);
                                        hi_1[4U] = fma(pj_4, _S137, hi_1[4U]);
                                        _S71 = int(7);
                                        break;
                                    }
                                    if(_S71 != int(7))
                                    {
                                        _S70 = _S71;
                                        break;
                                    }
                                    _S70 = int(6);
                                    break;
                                }
                                if(_S70 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S8)
                                        {
                                            _S69 = int(5);
                                            break;
                                        }
                                        float pj_5 = (*(&kernelContext_0)->ss_0)[5U * _S16 + kw_5];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S16 + kw_5], _S136, lo_1[5U]);
                                        hi_1[5U] = fma(pj_5, _S137, hi_1[5U]);
                                        _S69 = int(7);
                                        break;
                                    }
                                    if(_S69 != int(7))
                                    {
                                        _S68 = _S69;
                                        break;
                                    }
                                    _S68 = int(6);
                                    break;
                                }
                                if(_S68 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S8)
                                        {
                                            _S67 = int(5);
                                            break;
                                        }
                                        float pj_6 = (*(&kernelContext_0)->ss_0)[6U * _S16 + kw_5];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S16 + kw_5], _S136, lo_1[6U]);
                                        hi_1[6U] = fma(pj_6, _S137, hi_1[6U]);
                                        _S67 = int(7);
                                        break;
                                    }
                                    if(_S67 != int(7))
                                    {
                                        _S66 = _S67;
                                        break;
                                    }
                                    _S66 = int(6);
                                    break;
                                }
                                if(_S66 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S8)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_7 = (*(&kernelContext_0)->ss_0)[7U * _S16 + kw_5];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S16 + kw_5], _S136, lo_1[7U]);
                                        hi_1[7U] = fma(pj_7, _S137, hi_1[7U]);
                                        _S65 = int(7);
                                        break;
                                    }
                                    if(_S65 != int(7))
                                    {
                                        _S64 = _S65;
                                        break;
                                    }
                                    _S64 = int(6);
                                    break;
                                }
                                if(_S64 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S79 = int(4);
                            _S19 = _S7;
                            break;
                        }
                        if(_S79 != int(4))
                        {
                            _S78 = _S79;
                            break;
                        }
                        _S78 = int(3);
                        break;
                    }
                    if(_S78 != int(3))
                    {
                        break;
                    }
                    int _S139;
                    int _S140;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_1 >= _S80)
                            {
                                _S77 = int(2);
                                break;
                            }
                            float _S141 = attnLo_0(vw_0[1U]);
                            float _S142 = attnHi_0(vw_0[1U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(_S19)
                                        {
                                            _S77 = int(5);
                                            break;
                                        }
                                        float pj_8 = (*(&kernelContext_0)->ss_0)[j_1];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_1], _S141, lo_1[0U]);
                                        hi_1[0U] = fma(pj_8, _S142, hi_1[0U]);
                                        _S77 = int(7);
                                        break;
                                    }
                                    if(_S77 != int(7))
                                    {
                                        _S76 = _S77;
                                        break;
                                    }
                                    _S76 = int(6);
                                    break;
                                }
                                if(_S76 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S8)
                                        {
                                            _S75 = int(5);
                                            break;
                                        }
                                        float pj_9 = (*(&kernelContext_0)->ss_0)[_S16 + j_1];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S16 + j_1], _S141, lo_1[1U]);
                                        hi_1[1U] = fma(pj_9, _S142, hi_1[1U]);
                                        _S75 = int(7);
                                        break;
                                    }
                                    if(_S75 != int(7))
                                    {
                                        _S74 = _S75;
                                        break;
                                    }
                                    _S74 = int(6);
                                    break;
                                }
                                if(_S74 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S8)
                                        {
                                            _S73 = int(5);
                                            break;
                                        }
                                        float pj_10 = (*(&kernelContext_0)->ss_0)[2U * _S16 + j_1];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S16 + j_1], _S141, lo_1[2U]);
                                        hi_1[2U] = fma(pj_10, _S142, hi_1[2U]);
                                        _S73 = int(7);
                                        break;
                                    }
                                    if(_S73 != int(7))
                                    {
                                        _S72 = _S73;
                                        break;
                                    }
                                    _S72 = int(6);
                                    break;
                                }
                                if(_S72 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S8)
                                        {
                                            _S71 = int(5);
                                            break;
                                        }
                                        float pj_11 = (*(&kernelContext_0)->ss_0)[3U * _S16 + j_1];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S16 + j_1], _S141, lo_1[3U]);
                                        hi_1[3U] = fma(pj_11, _S142, hi_1[3U]);
                                        _S71 = int(7);
                                        break;
                                    }
                                    if(_S71 != int(7))
                                    {
                                        _S70 = _S71;
                                        break;
                                    }
                                    _S70 = int(6);
                                    break;
                                }
                                if(_S70 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S8)
                                        {
                                            _S69 = int(5);
                                            break;
                                        }
                                        float pj_12 = (*(&kernelContext_0)->ss_0)[4U * _S16 + j_1];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S16 + j_1], _S141, lo_1[4U]);
                                        hi_1[4U] = fma(pj_12, _S142, hi_1[4U]);
                                        _S69 = int(7);
                                        break;
                                    }
                                    if(_S69 != int(7))
                                    {
                                        _S68 = _S69;
                                        break;
                                    }
                                    _S68 = int(6);
                                    break;
                                }
                                if(_S68 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S8)
                                        {
                                            _S67 = int(5);
                                            break;
                                        }
                                        float pj_13 = (*(&kernelContext_0)->ss_0)[5U * _S16 + j_1];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S16 + j_1], _S141, lo_1[5U]);
                                        hi_1[5U] = fma(pj_13, _S142, hi_1[5U]);
                                        _S67 = int(7);
                                        break;
                                    }
                                    if(_S67 != int(7))
                                    {
                                        _S66 = _S67;
                                        break;
                                    }
                                    _S66 = int(6);
                                    break;
                                }
                                if(_S66 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S8)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_14 = (*(&kernelContext_0)->ss_0)[6U * _S16 + j_1];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S16 + j_1], _S141, lo_1[6U]);
                                        hi_1[6U] = fma(pj_14, _S142, hi_1[6U]);
                                        _S65 = int(7);
                                        break;
                                    }
                                    if(_S65 != int(7))
                                    {
                                        _S64 = _S65;
                                        break;
                                    }
                                    _S64 = int(6);
                                    break;
                                }
                                if(_S64 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S8)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_15 = (*(&kernelContext_0)->ss_0)[7U * _S16 + j_1];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S16 + j_1], _S141, lo_1[7U]);
                                        hi_1[7U] = fma(pj_15, _S142, hi_1[7U]);
                                        _S140 = int(7);
                                        break;
                                    }
                                    if(_S140 != int(7))
                                    {
                                        _S139 = _S140;
                                        break;
                                    }
                                    _S139 = int(6);
                                    break;
                                }
                                if(_S139 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S77 = int(4);
                            break;
                        }
                        if(_S77 != int(4))
                        {
                            _S76 = _S77;
                            break;
                        }
                        _S76 = int(3);
                        break;
                    }
                    if(_S76 != int(3))
                    {
                        break;
                    }
                    int _S143;
                    int _S144;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_2 >= _S80)
                            {
                                _S75 = int(2);
                                break;
                            }
                            float _S145 = attnLo_0(vw_0[2U]);
                            float _S146 = attnHi_0(vw_0[2U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(_S19)
                                        {
                                            _S75 = int(5);
                                            break;
                                        }
                                        float pj_16 = (*(&kernelContext_0)->ss_0)[j_2];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_2], _S145, lo_1[0U]);
                                        hi_1[0U] = fma(pj_16, _S146, hi_1[0U]);
                                        _S75 = int(7);
                                        break;
                                    }
                                    if(_S75 != int(7))
                                    {
                                        _S74 = _S75;
                                        break;
                                    }
                                    _S74 = int(6);
                                    break;
                                }
                                if(_S74 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S8)
                                        {
                                            _S73 = int(5);
                                            break;
                                        }
                                        float pj_17 = (*(&kernelContext_0)->ss_0)[_S16 + j_2];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S16 + j_2], _S145, lo_1[1U]);
                                        hi_1[1U] = fma(pj_17, _S146, hi_1[1U]);
                                        _S73 = int(7);
                                        break;
                                    }
                                    if(_S73 != int(7))
                                    {
                                        _S72 = _S73;
                                        break;
                                    }
                                    _S72 = int(6);
                                    break;
                                }
                                if(_S72 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S8)
                                        {
                                            _S71 = int(5);
                                            break;
                                        }
                                        float pj_18 = (*(&kernelContext_0)->ss_0)[2U * _S16 + j_2];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S16 + j_2], _S145, lo_1[2U]);
                                        hi_1[2U] = fma(pj_18, _S146, hi_1[2U]);
                                        _S71 = int(7);
                                        break;
                                    }
                                    if(_S71 != int(7))
                                    {
                                        _S70 = _S71;
                                        break;
                                    }
                                    _S70 = int(6);
                                    break;
                                }
                                if(_S70 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S8)
                                        {
                                            _S69 = int(5);
                                            break;
                                        }
                                        float pj_19 = (*(&kernelContext_0)->ss_0)[3U * _S16 + j_2];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S16 + j_2], _S145, lo_1[3U]);
                                        hi_1[3U] = fma(pj_19, _S146, hi_1[3U]);
                                        _S69 = int(7);
                                        break;
                                    }
                                    if(_S69 != int(7))
                                    {
                                        _S68 = _S69;
                                        break;
                                    }
                                    _S68 = int(6);
                                    break;
                                }
                                if(_S68 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S8)
                                        {
                                            _S67 = int(5);
                                            break;
                                        }
                                        float pj_20 = (*(&kernelContext_0)->ss_0)[4U * _S16 + j_2];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S16 + j_2], _S145, lo_1[4U]);
                                        hi_1[4U] = fma(pj_20, _S146, hi_1[4U]);
                                        _S67 = int(7);
                                        break;
                                    }
                                    if(_S67 != int(7))
                                    {
                                        _S66 = _S67;
                                        break;
                                    }
                                    _S66 = int(6);
                                    break;
                                }
                                if(_S66 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S8)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_21 = (*(&kernelContext_0)->ss_0)[5U * _S16 + j_2];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S16 + j_2], _S145, lo_1[5U]);
                                        hi_1[5U] = fma(pj_21, _S146, hi_1[5U]);
                                        _S65 = int(7);
                                        break;
                                    }
                                    if(_S65 != int(7))
                                    {
                                        _S64 = _S65;
                                        break;
                                    }
                                    _S64 = int(6);
                                    break;
                                }
                                if(_S64 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S8)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_22 = (*(&kernelContext_0)->ss_0)[6U * _S16 + j_2];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S16 + j_2], _S145, lo_1[6U]);
                                        hi_1[6U] = fma(pj_22, _S146, hi_1[6U]);
                                        _S140 = int(7);
                                        break;
                                    }
                                    if(_S140 != int(7))
                                    {
                                        _S139 = _S140;
                                        break;
                                    }
                                    _S139 = int(6);
                                    break;
                                }
                                if(_S139 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S8)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_23 = (*(&kernelContext_0)->ss_0)[7U * _S16 + j_2];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S16 + j_2], _S145, lo_1[7U]);
                                        hi_1[7U] = fma(pj_23, _S146, hi_1[7U]);
                                        _S144 = int(7);
                                        break;
                                    }
                                    if(_S144 != int(7))
                                    {
                                        _S143 = _S144;
                                        break;
                                    }
                                    _S143 = int(6);
                                    break;
                                }
                                if(_S143 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S75 = int(4);
                            break;
                        }
                        if(_S75 != int(4))
                        {
                            _S74 = _S75;
                            break;
                        }
                        _S74 = int(3);
                        break;
                    }
                    if(_S74 != int(3))
                    {
                        break;
                    }
                    int _S147;
                    int _S148;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_3 >= _S80)
                            {
                                _S73 = int(2);
                                break;
                            }
                            float _S149 = attnLo_0(vw_0[3U]);
                            float _S150 = attnHi_0(vw_0[3U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(_S19)
                                        {
                                            _S73 = int(5);
                                            break;
                                        }
                                        float pj_24 = (*(&kernelContext_0)->ss_0)[j_3];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_3], _S149, lo_1[0U]);
                                        hi_1[0U] = fma(pj_24, _S150, hi_1[0U]);
                                        _S73 = int(7);
                                        break;
                                    }
                                    if(_S73 != int(7))
                                    {
                                        _S72 = _S73;
                                        break;
                                    }
                                    _S72 = int(6);
                                    break;
                                }
                                if(_S72 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S8)
                                        {
                                            _S71 = int(5);
                                            break;
                                        }
                                        float pj_25 = (*(&kernelContext_0)->ss_0)[_S16 + j_3];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S16 + j_3], _S149, lo_1[1U]);
                                        hi_1[1U] = fma(pj_25, _S150, hi_1[1U]);
                                        _S71 = int(7);
                                        break;
                                    }
                                    if(_S71 != int(7))
                                    {
                                        _S70 = _S71;
                                        break;
                                    }
                                    _S70 = int(6);
                                    break;
                                }
                                if(_S70 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S8)
                                        {
                                            _S69 = int(5);
                                            break;
                                        }
                                        float pj_26 = (*(&kernelContext_0)->ss_0)[2U * _S16 + j_3];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S16 + j_3], _S149, lo_1[2U]);
                                        hi_1[2U] = fma(pj_26, _S150, hi_1[2U]);
                                        _S69 = int(7);
                                        break;
                                    }
                                    if(_S69 != int(7))
                                    {
                                        _S68 = _S69;
                                        break;
                                    }
                                    _S68 = int(6);
                                    break;
                                }
                                if(_S68 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S8)
                                        {
                                            _S67 = int(5);
                                            break;
                                        }
                                        float pj_27 = (*(&kernelContext_0)->ss_0)[3U * _S16 + j_3];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S16 + j_3], _S149, lo_1[3U]);
                                        hi_1[3U] = fma(pj_27, _S150, hi_1[3U]);
                                        _S67 = int(7);
                                        break;
                                    }
                                    if(_S67 != int(7))
                                    {
                                        _S66 = _S67;
                                        break;
                                    }
                                    _S66 = int(6);
                                    break;
                                }
                                if(_S66 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S8)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_28 = (*(&kernelContext_0)->ss_0)[4U * _S16 + j_3];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S16 + j_3], _S149, lo_1[4U]);
                                        hi_1[4U] = fma(pj_28, _S150, hi_1[4U]);
                                        _S65 = int(7);
                                        break;
                                    }
                                    if(_S65 != int(7))
                                    {
                                        _S64 = _S65;
                                        break;
                                    }
                                    _S64 = int(6);
                                    break;
                                }
                                if(_S64 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S8)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_29 = (*(&kernelContext_0)->ss_0)[5U * _S16 + j_3];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S16 + j_3], _S149, lo_1[5U]);
                                        hi_1[5U] = fma(pj_29, _S150, hi_1[5U]);
                                        _S140 = int(7);
                                        break;
                                    }
                                    if(_S140 != int(7))
                                    {
                                        _S139 = _S140;
                                        break;
                                    }
                                    _S139 = int(6);
                                    break;
                                }
                                if(_S139 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S8)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_30 = (*(&kernelContext_0)->ss_0)[6U * _S16 + j_3];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S16 + j_3], _S149, lo_1[6U]);
                                        hi_1[6U] = fma(pj_30, _S150, hi_1[6U]);
                                        _S144 = int(7);
                                        break;
                                    }
                                    if(_S144 != int(7))
                                    {
                                        _S143 = _S144;
                                        break;
                                    }
                                    _S143 = int(6);
                                    break;
                                }
                                if(_S143 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S8)
                                        {
                                            _S148 = int(5);
                                            break;
                                        }
                                        float pj_31 = (*(&kernelContext_0)->ss_0)[7U * _S16 + j_3];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S16 + j_3], _S149, lo_1[7U]);
                                        hi_1[7U] = fma(pj_31, _S150, hi_1[7U]);
                                        _S148 = int(7);
                                        break;
                                    }
                                    if(_S148 != int(7))
                                    {
                                        _S147 = _S148;
                                        break;
                                    }
                                    _S147 = int(6);
                                    break;
                                }
                                if(_S147 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S73 = int(4);
                            break;
                        }
                        if(_S73 != int(4))
                        {
                            _S72 = _S73;
                            break;
                        }
                        _S72 = int(3);
                        break;
                    }
                    if(_S72 != int(3))
                    {
                        break;
                    }
                    int _S151;
                    int _S152;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_4 >= _S80)
                            {
                                _S71 = int(2);
                                break;
                            }
                            float _S153 = attnLo_0(vw_0[4U]);
                            float _S154 = attnHi_0(vw_0[4U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(_S19)
                                        {
                                            _S71 = int(5);
                                            break;
                                        }
                                        float pj_32 = (*(&kernelContext_0)->ss_0)[j_4];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_4], _S153, lo_1[0U]);
                                        hi_1[0U] = fma(pj_32, _S154, hi_1[0U]);
                                        _S71 = int(7);
                                        break;
                                    }
                                    if(_S71 != int(7))
                                    {
                                        _S70 = _S71;
                                        break;
                                    }
                                    _S70 = int(6);
                                    break;
                                }
                                if(_S70 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S8)
                                        {
                                            _S69 = int(5);
                                            break;
                                        }
                                        float pj_33 = (*(&kernelContext_0)->ss_0)[_S16 + j_4];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S16 + j_4], _S153, lo_1[1U]);
                                        hi_1[1U] = fma(pj_33, _S154, hi_1[1U]);
                                        _S69 = int(7);
                                        break;
                                    }
                                    if(_S69 != int(7))
                                    {
                                        _S68 = _S69;
                                        break;
                                    }
                                    _S68 = int(6);
                                    break;
                                }
                                if(_S68 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S8)
                                        {
                                            _S67 = int(5);
                                            break;
                                        }
                                        float pj_34 = (*(&kernelContext_0)->ss_0)[2U * _S16 + j_4];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S16 + j_4], _S153, lo_1[2U]);
                                        hi_1[2U] = fma(pj_34, _S154, hi_1[2U]);
                                        _S67 = int(7);
                                        break;
                                    }
                                    if(_S67 != int(7))
                                    {
                                        _S66 = _S67;
                                        break;
                                    }
                                    _S66 = int(6);
                                    break;
                                }
                                if(_S66 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S8)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_35 = (*(&kernelContext_0)->ss_0)[3U * _S16 + j_4];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S16 + j_4], _S153, lo_1[3U]);
                                        hi_1[3U] = fma(pj_35, _S154, hi_1[3U]);
                                        _S65 = int(7);
                                        break;
                                    }
                                    if(_S65 != int(7))
                                    {
                                        _S64 = _S65;
                                        break;
                                    }
                                    _S64 = int(6);
                                    break;
                                }
                                if(_S64 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S8)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_36 = (*(&kernelContext_0)->ss_0)[4U * _S16 + j_4];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S16 + j_4], _S153, lo_1[4U]);
                                        hi_1[4U] = fma(pj_36, _S154, hi_1[4U]);
                                        _S140 = int(7);
                                        break;
                                    }
                                    if(_S140 != int(7))
                                    {
                                        _S139 = _S140;
                                        break;
                                    }
                                    _S139 = int(6);
                                    break;
                                }
                                if(_S139 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S8)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_37 = (*(&kernelContext_0)->ss_0)[5U * _S16 + j_4];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S16 + j_4], _S153, lo_1[5U]);
                                        hi_1[5U] = fma(pj_37, _S154, hi_1[5U]);
                                        _S144 = int(7);
                                        break;
                                    }
                                    if(_S144 != int(7))
                                    {
                                        _S143 = _S144;
                                        break;
                                    }
                                    _S143 = int(6);
                                    break;
                                }
                                if(_S143 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S8)
                                        {
                                            _S148 = int(5);
                                            break;
                                        }
                                        float pj_38 = (*(&kernelContext_0)->ss_0)[6U * _S16 + j_4];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S16 + j_4], _S153, lo_1[6U]);
                                        hi_1[6U] = fma(pj_38, _S154, hi_1[6U]);
                                        _S148 = int(7);
                                        break;
                                    }
                                    if(_S148 != int(7))
                                    {
                                        _S147 = _S148;
                                        break;
                                    }
                                    _S147 = int(6);
                                    break;
                                }
                                if(_S147 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S8)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_39 = (*(&kernelContext_0)->ss_0)[7U * _S16 + j_4];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S16 + j_4], _S153, lo_1[7U]);
                                        hi_1[7U] = fma(pj_39, _S154, hi_1[7U]);
                                        _S152 = int(7);
                                        break;
                                    }
                                    if(_S152 != int(7))
                                    {
                                        _S151 = _S152;
                                        break;
                                    }
                                    _S151 = int(6);
                                    break;
                                }
                                if(_S151 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S71 = int(4);
                            break;
                        }
                        if(_S71 != int(4))
                        {
                            _S70 = _S71;
                            break;
                        }
                        _S70 = int(3);
                        break;
                    }
                    if(_S70 != int(3))
                    {
                        break;
                    }
                    int _S155;
                    int _S156;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_5 >= _S80)
                            {
                                _S69 = int(2);
                                break;
                            }
                            float _S157 = attnLo_0(vw_0[5U]);
                            float _S158 = attnHi_0(vw_0[5U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(_S19)
                                        {
                                            _S69 = int(5);
                                            break;
                                        }
                                        float pj_40 = (*(&kernelContext_0)->ss_0)[j_5];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_5], _S157, lo_1[0U]);
                                        hi_1[0U] = fma(pj_40, _S158, hi_1[0U]);
                                        _S69 = int(7);
                                        break;
                                    }
                                    if(_S69 != int(7))
                                    {
                                        _S68 = _S69;
                                        break;
                                    }
                                    _S68 = int(6);
                                    break;
                                }
                                if(_S68 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S8)
                                        {
                                            _S67 = int(5);
                                            break;
                                        }
                                        float pj_41 = (*(&kernelContext_0)->ss_0)[_S16 + j_5];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S16 + j_5], _S157, lo_1[1U]);
                                        hi_1[1U] = fma(pj_41, _S158, hi_1[1U]);
                                        _S67 = int(7);
                                        break;
                                    }
                                    if(_S67 != int(7))
                                    {
                                        _S66 = _S67;
                                        break;
                                    }
                                    _S66 = int(6);
                                    break;
                                }
                                if(_S66 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S8)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_42 = (*(&kernelContext_0)->ss_0)[2U * _S16 + j_5];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S16 + j_5], _S157, lo_1[2U]);
                                        hi_1[2U] = fma(pj_42, _S158, hi_1[2U]);
                                        _S65 = int(7);
                                        break;
                                    }
                                    if(_S65 != int(7))
                                    {
                                        _S64 = _S65;
                                        break;
                                    }
                                    _S64 = int(6);
                                    break;
                                }
                                if(_S64 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S8)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_43 = (*(&kernelContext_0)->ss_0)[3U * _S16 + j_5];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S16 + j_5], _S157, lo_1[3U]);
                                        hi_1[3U] = fma(pj_43, _S158, hi_1[3U]);
                                        _S140 = int(7);
                                        break;
                                    }
                                    if(_S140 != int(7))
                                    {
                                        _S139 = _S140;
                                        break;
                                    }
                                    _S139 = int(6);
                                    break;
                                }
                                if(_S139 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S8)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_44 = (*(&kernelContext_0)->ss_0)[4U * _S16 + j_5];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S16 + j_5], _S157, lo_1[4U]);
                                        hi_1[4U] = fma(pj_44, _S158, hi_1[4U]);
                                        _S144 = int(7);
                                        break;
                                    }
                                    if(_S144 != int(7))
                                    {
                                        _S143 = _S144;
                                        break;
                                    }
                                    _S143 = int(6);
                                    break;
                                }
                                if(_S143 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S8)
                                        {
                                            _S148 = int(5);
                                            break;
                                        }
                                        float pj_45 = (*(&kernelContext_0)->ss_0)[5U * _S16 + j_5];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S16 + j_5], _S157, lo_1[5U]);
                                        hi_1[5U] = fma(pj_45, _S158, hi_1[5U]);
                                        _S148 = int(7);
                                        break;
                                    }
                                    if(_S148 != int(7))
                                    {
                                        _S147 = _S148;
                                        break;
                                    }
                                    _S147 = int(6);
                                    break;
                                }
                                if(_S147 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S8)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_46 = (*(&kernelContext_0)->ss_0)[6U * _S16 + j_5];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S16 + j_5], _S157, lo_1[6U]);
                                        hi_1[6U] = fma(pj_46, _S158, hi_1[6U]);
                                        _S152 = int(7);
                                        break;
                                    }
                                    if(_S152 != int(7))
                                    {
                                        _S151 = _S152;
                                        break;
                                    }
                                    _S151 = int(6);
                                    break;
                                }
                                if(_S151 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S8)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_47 = (*(&kernelContext_0)->ss_0)[7U * _S16 + j_5];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S16 + j_5], _S157, lo_1[7U]);
                                        hi_1[7U] = fma(pj_47, _S158, hi_1[7U]);
                                        _S156 = int(7);
                                        break;
                                    }
                                    if(_S156 != int(7))
                                    {
                                        _S155 = _S156;
                                        break;
                                    }
                                    _S155 = int(6);
                                    break;
                                }
                                if(_S155 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S69 = int(4);
                            break;
                        }
                        if(_S69 != int(4))
                        {
                            _S68 = _S69;
                            break;
                        }
                        _S68 = int(3);
                        break;
                    }
                    if(_S68 != int(3))
                    {
                        break;
                    }
                    int _S159;
                    int _S160;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_6 >= _S80)
                            {
                                _S67 = int(2);
                                break;
                            }
                            float _S161 = attnLo_0(vw_0[6U]);
                            float _S162 = attnHi_0(vw_0[6U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(_S19)
                                        {
                                            _S67 = int(5);
                                            break;
                                        }
                                        float pj_48 = (*(&kernelContext_0)->ss_0)[j_6];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_6], _S161, lo_1[0U]);
                                        hi_1[0U] = fma(pj_48, _S162, hi_1[0U]);
                                        _S67 = int(7);
                                        break;
                                    }
                                    if(_S67 != int(7))
                                    {
                                        _S66 = _S67;
                                        break;
                                    }
                                    _S66 = int(6);
                                    break;
                                }
                                if(_S66 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S8)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_49 = (*(&kernelContext_0)->ss_0)[_S16 + j_6];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S16 + j_6], _S161, lo_1[1U]);
                                        hi_1[1U] = fma(pj_49, _S162, hi_1[1U]);
                                        _S65 = int(7);
                                        break;
                                    }
                                    if(_S65 != int(7))
                                    {
                                        _S64 = _S65;
                                        break;
                                    }
                                    _S64 = int(6);
                                    break;
                                }
                                if(_S64 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S8)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_50 = (*(&kernelContext_0)->ss_0)[2U * _S16 + j_6];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S16 + j_6], _S161, lo_1[2U]);
                                        hi_1[2U] = fma(pj_50, _S162, hi_1[2U]);
                                        _S140 = int(7);
                                        break;
                                    }
                                    if(_S140 != int(7))
                                    {
                                        _S139 = _S140;
                                        break;
                                    }
                                    _S139 = int(6);
                                    break;
                                }
                                if(_S139 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S8)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_51 = (*(&kernelContext_0)->ss_0)[3U * _S16 + j_6];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S16 + j_6], _S161, lo_1[3U]);
                                        hi_1[3U] = fma(pj_51, _S162, hi_1[3U]);
                                        _S144 = int(7);
                                        break;
                                    }
                                    if(_S144 != int(7))
                                    {
                                        _S143 = _S144;
                                        break;
                                    }
                                    _S143 = int(6);
                                    break;
                                }
                                if(_S143 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S8)
                                        {
                                            _S148 = int(5);
                                            break;
                                        }
                                        float pj_52 = (*(&kernelContext_0)->ss_0)[4U * _S16 + j_6];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S16 + j_6], _S161, lo_1[4U]);
                                        hi_1[4U] = fma(pj_52, _S162, hi_1[4U]);
                                        _S148 = int(7);
                                        break;
                                    }
                                    if(_S148 != int(7))
                                    {
                                        _S147 = _S148;
                                        break;
                                    }
                                    _S147 = int(6);
                                    break;
                                }
                                if(_S147 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S8)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_53 = (*(&kernelContext_0)->ss_0)[5U * _S16 + j_6];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S16 + j_6], _S161, lo_1[5U]);
                                        hi_1[5U] = fma(pj_53, _S162, hi_1[5U]);
                                        _S152 = int(7);
                                        break;
                                    }
                                    if(_S152 != int(7))
                                    {
                                        _S151 = _S152;
                                        break;
                                    }
                                    _S151 = int(6);
                                    break;
                                }
                                if(_S151 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S8)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_54 = (*(&kernelContext_0)->ss_0)[6U * _S16 + j_6];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S16 + j_6], _S161, lo_1[6U]);
                                        hi_1[6U] = fma(pj_54, _S162, hi_1[6U]);
                                        _S156 = int(7);
                                        break;
                                    }
                                    if(_S156 != int(7))
                                    {
                                        _S155 = _S156;
                                        break;
                                    }
                                    _S155 = int(6);
                                    break;
                                }
                                if(_S155 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S8)
                                        {
                                            _S160 = int(5);
                                            break;
                                        }
                                        float pj_55 = (*(&kernelContext_0)->ss_0)[7U * _S16 + j_6];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S16 + j_6], _S161, lo_1[7U]);
                                        hi_1[7U] = fma(pj_55, _S162, hi_1[7U]);
                                        _S160 = int(7);
                                        break;
                                    }
                                    if(_S160 != int(7))
                                    {
                                        _S159 = _S160;
                                        break;
                                    }
                                    _S159 = int(6);
                                    break;
                                }
                                if(_S159 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S67 = int(4);
                            break;
                        }
                        if(_S67 != int(4))
                        {
                            _S66 = _S67;
                            break;
                        }
                        _S66 = int(3);
                        break;
                    }
                    if(_S66 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_7 >= _S80)
                            {
                                _S65 = int(2);
                                break;
                            }
                            float _S163 = attnLo_0(vw_0[7U]);
                            float _S164 = attnHi_0(vw_0[7U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(_S19)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_56 = (*(&kernelContext_0)->ss_0)[j_7];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_7], _S163, lo_1[0U]);
                                        hi_1[0U] = fma(pj_56, _S164, hi_1[0U]);
                                        _S65 = int(7);
                                        break;
                                    }
                                    if(_S65 != int(7))
                                    {
                                        _S64 = _S65;
                                        break;
                                    }
                                    _S64 = int(6);
                                    break;
                                }
                                if(_S64 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S8)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_57 = (*(&kernelContext_0)->ss_0)[_S16 + j_7];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S16 + j_7], _S163, lo_1[1U]);
                                        hi_1[1U] = fma(pj_57, _S164, hi_1[1U]);
                                        _S140 = int(7);
                                        break;
                                    }
                                    if(_S140 != int(7))
                                    {
                                        _S139 = _S140;
                                        break;
                                    }
                                    _S139 = int(6);
                                    break;
                                }
                                if(_S139 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S8)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_58 = (*(&kernelContext_0)->ss_0)[2U * _S16 + j_7];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S16 + j_7], _S163, lo_1[2U]);
                                        hi_1[2U] = fma(pj_58, _S164, hi_1[2U]);
                                        _S144 = int(7);
                                        break;
                                    }
                                    if(_S144 != int(7))
                                    {
                                        _S143 = _S144;
                                        break;
                                    }
                                    _S143 = int(6);
                                    break;
                                }
                                if(_S143 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S8)
                                        {
                                            _S148 = int(5);
                                            break;
                                        }
                                        float pj_59 = (*(&kernelContext_0)->ss_0)[3U * _S16 + j_7];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S16 + j_7], _S163, lo_1[3U]);
                                        hi_1[3U] = fma(pj_59, _S164, hi_1[3U]);
                                        _S148 = int(7);
                                        break;
                                    }
                                    if(_S148 != int(7))
                                    {
                                        _S147 = _S148;
                                        break;
                                    }
                                    _S147 = int(6);
                                    break;
                                }
                                if(_S147 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S8)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_60 = (*(&kernelContext_0)->ss_0)[4U * _S16 + j_7];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S16 + j_7], _S163, lo_1[4U]);
                                        hi_1[4U] = fma(pj_60, _S164, hi_1[4U]);
                                        _S152 = int(7);
                                        break;
                                    }
                                    if(_S152 != int(7))
                                    {
                                        _S151 = _S152;
                                        break;
                                    }
                                    _S151 = int(6);
                                    break;
                                }
                                if(_S151 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S8)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_61 = (*(&kernelContext_0)->ss_0)[5U * _S16 + j_7];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S16 + j_7], _S163, lo_1[5U]);
                                        hi_1[5U] = fma(pj_61, _S164, hi_1[5U]);
                                        _S156 = int(7);
                                        break;
                                    }
                                    if(_S156 != int(7))
                                    {
                                        _S155 = _S156;
                                        break;
                                    }
                                    _S155 = int(6);
                                    break;
                                }
                                if(_S155 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S8)
                                        {
                                            _S160 = int(5);
                                            break;
                                        }
                                        float pj_62 = (*(&kernelContext_0)->ss_0)[6U * _S16 + j_7];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S16 + j_7], _S163, lo_1[6U]);
                                        hi_1[6U] = fma(pj_62, _S164, hi_1[6U]);
                                        _S160 = int(7);
                                        break;
                                    }
                                    if(_S160 != int(7))
                                    {
                                        _S159 = _S160;
                                        break;
                                    }
                                    _S159 = int(6);
                                    break;
                                }
                                if(_S159 != int(6))
                                {
                                    break;
                                }
                                int _S165;
                                for(;;)
                                {
                                    int _S166;
                                    for(;;)
                                    {
                                        if(7U >= _S8)
                                        {
                                            _S166 = int(5);
                                            break;
                                        }
                                        float pj_63 = (*(&kernelContext_0)->ss_0)[7U * _S16 + j_7];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S16 + j_7], _S163, lo_1[7U]);
                                        hi_1[7U] = fma(pj_63, _S164, hi_1[7U]);
                                        _S166 = int(7);
                                        break;
                                    }
                                    if(_S166 != int(7))
                                    {
                                        _S165 = _S166;
                                        break;
                                    }
                                    _S165 = int(6);
                                    break;
                                }
                                if(_S165 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S65 = int(4);
                            break;
                        }
                        if(_S65 != int(4))
                        {
                            _S64 = _S65;
                            break;
                        }
                        _S64 = int(3);
                        break;
                    }
                    if(_S64 != int(3))
                    {
                        break;
                    }
                    break;
                }
                kw_5 = kw_5 + 32U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c0_0 = c0_0 + _S16;
    }
    r_0 = 0U;
    for(;;)
    {
        if(r_0 < _S8)
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
        uint _S167 = 2U * _S62;
        uint _S168 = _S63 * 64U + _S167;
        (*(&kernelContext_0)->ss_0)[_S168] = a_13;
        (*(&kernelContext_0)->ss_0)[_S168 + 1U] = b_6;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S63 != 0U)
        {
            _S19 = true;
        }
        else
        {
            _S19 = _S62 >= _S11;
        }
        if(_S19)
        {
            r_0 = r_0 + 1U;
            continue;
        }
        uint _S169 = 64U + _S167;
        uint _S170 = 128U + _S167;
        uint _S171 = 192U + _S167;
        float a_14 = (*(&kernelContext_0)->ss_0)[_S167] + (*(&kernelContext_0)->ss_0)[_S169] + (*(&kernelContext_0)->ss_0)[_S170] + (*(&kernelContext_0)->ss_0)[_S171];
        float b_7 = (*(&kernelContext_0)->ss_0)[_S167 + 1U] + (*(&kernelContext_0)->ss_0)[_S169 + 1U] + (*(&kernelContext_0)->ss_0)[_S170 + 1U] + (*(&kernelContext_0)->ss_0)[_S171 + 1U];
        uint _S172 = r_0 / (&kernelContext_0)->p_0->queries_0;
        uint h_4 = _S13 + _S172;
        uint i_2 = r_0 % (&kernelContext_0)->p_0->queries_0;
        uint row_0 = _S9 + h_4 * (&kernelContext_0)->p_0->queries_0 + i_2;
        float a_15;
        float b_8;
        if(((&kernelContext_0)->p_0->splits_1) > 1U)
        {
            uint pr_8 = split_0 * _S10 + row_0;
            uint at_0 = attnAccBase_0((&kernelContext_0)->p_0->splits_1, _S10) + pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S167;
            if(_S62 == 0U)
            {
                uint _S173 = 2U * pr_8;
                float device* _S174 = (&kernelContext_0)->partials_0+_S173;
                bool _S175 = _S14 == 0U;
                if(_S175)
                {
                    a_15 = dg_negative_infinity_0();
                }
                else
                {
                    a_15 = (*(&kernelContext_0)->rowM_0)[r_0];
                }
                *_S174 = a_15;
                float device* _S176 = (&kernelContext_0)->partials_0+(_S173 + 1U);
                if(_S175)
                {
                    b_8 = 0.0;
                }
                else
                {
                    b_8 = (*(&kernelContext_0)->rowL_0)[r_0];
                }
                *_S176 = b_8;
            }
            float device* _S177 = (&kernelContext_0)->partials_0+at_0;
            if(((*(&kernelContext_0)->rowL_0)[r_0]) == 0.0)
            {
                a_15 = 0.0;
            }
            else
            {
                a_15 = a_14;
            }
            *_S177 = a_15;
            float device* _S178 = (&kernelContext_0)->partials_0+(at_0 + 1U);
            if(((*(&kernelContext_0)->rowL_0)[r_0]) == 0.0)
            {
                b_8 = 0.0;
            }
            else
            {
                b_8 = b_7;
            }
            *_S178 = b_8;
            r_0 = r_0 + 1U;
            continue;
        }
        float l_0 = (*(&kernelContext_0)->rowL_0)[r_0];
        bool _S179 = ((*(&kernelContext_0)->rowL_0)[r_0]) == 0.0;
        if(_S179)
        {
            a_15 = 0.0;
        }
        else
        {
            a_15 = a_14 / l_0;
        }
        if(_S179)
        {
            b_8 = 0.0;
        }
        else
        {
            b_8 = b_7 / l_0;
        }
        *((&kernelContext_0)->o_0+(row_0 * _S11 + _S62)) = attnPack_0(a_15, b_8);
        r_0 = r_0 + 1U;
    }
    return;
}

