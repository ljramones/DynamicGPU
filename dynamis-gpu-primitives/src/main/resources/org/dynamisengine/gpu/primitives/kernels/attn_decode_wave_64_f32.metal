#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant uint fc_USE_LENGTHS_0 [[function_constant(1)]];
constant uint USE_LENGTHS_0 = is_function_constant_defined(fc_USE_LENGTHS_0) ? fc_USE_LENGTHS_0 : 1U;
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
    float device* o_0;
    uint threadgroup* waves_0;
    array<float, int(8)> threadgroup* rowM_0;
    array<float, int(8)> threadgroup* rowL_0;
    array<uint, int(256)> threadgroup* qs_0;
    array<float, int(1024)> threadgroup* ss_0;
    array<float, int(8)> threadgroup* rowA_0;
};

[[kernel]] void attn_decode_wave_64_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(7)]], uint device* table_1 [[buffer(5)]], uint device* lens_2 [[buffer(6)]], uint device* q_1 [[buffer(0)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partials_1 [[buffer(4)]], float device* o_1 [[buffer(3)]])
{
    bool _S5;
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
    uint _S6 = min(hpk_0 * (&kernelContext_0)->p_0->queries_0, ROW_TIER_0);
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
    uint _S7 = seq_0 * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0;
    uint _S8 = max((&kernelContext_0)->p_0->batch_0, 1U) * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0;
    uint _S9 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint _S10 = keys_1 - (&kernelContext_0)->p_0->queries_0;
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S11 = min(keys_1, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    uint _S12 = kvHead_0 * hpk_0;
    uint firstHead_1 = seq_0 * p_1->heads_0 + _S12;
    bool _S13 = USE_LENGTHS_0 != 0U;
    uint kEnd_0;
    if(_S13)
    {
        uint _S14 = attnKeyEnd_0((&kernelContext_0)->lens_1, (&kernelContext_0)->p_0->lengthGroup_1, firstHead_1, firstHead_1 + hpk_0 - 1U, _S11);
        kEnd_0 = _S14;
    }
    else
    {
        kEnd_0 = _S11;
    }
    uint _S15;
    if(kEnd_0 > kStart_0)
    {
        _S15 = kEnd_0 - kStart_0;
    }
    else
    {
        _S15 = 0U;
    }
    uint _S16 = 1024U / max(_S6, 1U);
    uint _S17 = max(128U, _S16 / 128U * 128U);
    thread array<uint, int(8)> limit_0;
    uint e_0;
    bool visible_0;
    if(_S13)
    {
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S18 = 0U % (&kernelContext_0)->p_0->queries_0;
            e_0 = _S18 + _S10 + 1U;
        }
        else
        {
            e_0 = keys_1;
        }
        limit_0[0U] = e_0;
        if(0U < _S6)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S19 = limit_0[0U];
            uint _S20 = firstHead_1 / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[0U] = min(_S19, *((&kernelContext_0)->lens_1+_S20));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S21 = 1U % (&kernelContext_0)->p_0->queries_0;
            e_0 = _S21 + _S10 + 1U;
        }
        else
        {
            e_0 = keys_1;
        }
        limit_0[1U] = e_0;
        if(1U < _S6)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S22 = limit_0[1U];
            uint _S23 = 1U / (&kernelContext_0)->p_0->queries_0;
            uint _S24 = (firstHead_1 + _S23) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[1U] = min(_S22, *((&kernelContext_0)->lens_1+_S24));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S25 = 2U % (&kernelContext_0)->p_0->queries_0;
            e_0 = _S25 + _S10 + 1U;
        }
        else
        {
            e_0 = keys_1;
        }
        limit_0[2U] = e_0;
        if(2U < _S6)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S26 = limit_0[2U];
            uint _S27 = 2U / (&kernelContext_0)->p_0->queries_0;
            uint _S28 = (firstHead_1 + _S27) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[2U] = min(_S26, *((&kernelContext_0)->lens_1+_S28));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S29 = 3U % (&kernelContext_0)->p_0->queries_0;
            e_0 = _S29 + _S10 + 1U;
        }
        else
        {
            e_0 = keys_1;
        }
        limit_0[3U] = e_0;
        if(3U < _S6)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S30 = limit_0[3U];
            uint _S31 = 3U / (&kernelContext_0)->p_0->queries_0;
            uint _S32 = (firstHead_1 + _S31) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[3U] = min(_S30, *((&kernelContext_0)->lens_1+_S32));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S33 = 4U % (&kernelContext_0)->p_0->queries_0;
            e_0 = _S33 + _S10 + 1U;
        }
        else
        {
            e_0 = keys_1;
        }
        limit_0[4U] = e_0;
        if(4U < _S6)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S34 = limit_0[4U];
            uint _S35 = 4U / (&kernelContext_0)->p_0->queries_0;
            uint _S36 = (firstHead_1 + _S35) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[4U] = min(_S34, *((&kernelContext_0)->lens_1+_S36));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S37 = 5U % (&kernelContext_0)->p_0->queries_0;
            e_0 = _S37 + _S10 + 1U;
        }
        else
        {
            e_0 = keys_1;
        }
        limit_0[5U] = e_0;
        if(5U < _S6)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S38 = limit_0[5U];
            uint _S39 = 5U / (&kernelContext_0)->p_0->queries_0;
            uint _S40 = (firstHead_1 + _S39) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[5U] = min(_S38, *((&kernelContext_0)->lens_1+_S40));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S41 = 6U % (&kernelContext_0)->p_0->queries_0;
            e_0 = _S41 + _S10 + 1U;
        }
        else
        {
            e_0 = keys_1;
        }
        limit_0[6U] = e_0;
        if(6U < _S6)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S42 = limit_0[6U];
            uint _S43 = 6U / (&kernelContext_0)->p_0->queries_0;
            uint _S44 = (firstHead_1 + _S43) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[6U] = min(_S42, *((&kernelContext_0)->lens_1+_S44));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S45 = 7U % (&kernelContext_0)->p_0->queries_0;
            keys_1 = _S45 + _S10 + 1U;
        }
        limit_0[7U] = keys_1;
        if(7U < _S6)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S46 = limit_0[7U];
            uint _S47 = 7U / (&kernelContext_0)->p_0->queries_0;
            uint _S48 = (firstHead_1 + _S47) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[7U] = min(_S46, *((&kernelContext_0)->lens_1+_S48));
        }
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
    e_0 = t_0;
    for(;;)
    {
        if(e_0 < (ROW_TIER_0 * 32U))
        {
        }
        else
        {
            break;
        }
        uint r_1 = e_0 / 32U;
        uint pr_0 = e_0 % 32U;
        if(r_1 < _S6)
        {
            visible_0 = pr_0 < _S9;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S49 = r_1 / (&kernelContext_0)->p_0->queries_0;
            uint h_0 = _S12 + _S49;
            uint i_1 = r_1 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((_S7 + h_0 * (&kernelContext_0)->p_0->queries_0 + i_1) * _S9 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_0] = w_2;
        e_0 = e_0 + 128U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S50 = simd_sum(1U);
    uint lane_0 = simd_prefix_exclusive_sum(1U);
    bool _S51 = lane_0 == 0U;
    if(_S51)
    {
        uint _S52 = atomic_fetch_add_explicit(((atomic_uint threadgroup*)((&kernelContext_0)->waves_0)), 1U, memory_order_relaxed);
        w_2 = _S52;
    }
    else
    {
        w_2 = 0U;
    }
    uint _S53 = simd_sum(w_2);
    uint _S54 = t_0 % 32U;
    uint _S55 = t_0 / 32U;
    thread array<float, int(8)> lo_0;
    thread array<float, int(8)> hi_0;
    float w_3;
    float sum_0;
    float m_0;
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
        int _S56;
        int _S57;
        int _S58;
        int _S59;
        int _S60;
        int _S61;
        int _S62;
        int _S63;
        int _S64;
        int _S65;
        int _S66;
        int _S67;
        int _S68;
        int _S69;
        int _S70;
        int _S71;
        uint kw_0;
        uint kw_1;
        uint kw_2;
        uint kw_3;
        uint kw_4;
        uint kw_5;
        uint p0_0;
        uint _S72 = min(_S17, kEnd_0 - c0_0);
        uint j_0 = t_0;
        for(;;)
        {
            if(j_0 < _S72)
            {
            }
            else
            {
                break;
            }
            uint key_0 = c0_0 + j_0;
            if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
            {
                keys_1 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * _S9;
            }
            else
            {
                uint _S73 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
                uint _S74 = (*((&kernelContext_0)->table_0+(r_0 + _S73)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                uint _S75 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
                keys_1 = (_S74 + _S75) * _S9;
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
                if(p0_0 < _S9)
                {
                    kw_5 = *((&kernelContext_0)->k_0+(keys_1 + p0_0));
                }
                else
                {
                    kw_5 = 0U;
                }
                float _S76 = attnLo_0(kw_5);
                float _S77 = attnHi_0(kw_5);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[p0_0]), _S77, fma(attnLo_0((*(&kernelContext_0)->qs_0)[p0_0]), _S76, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + p0_0]), _S77, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + p0_0]), _S76, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S77, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S76, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + p0_0]), _S77, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + p0_0]), _S76, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S77, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S76, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + p0_0]), _S77, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + p0_0]), _S76, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S77, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S76, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + p0_0]), _S77, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + p0_0]), _S76, s_0[7U]));
                }
                uint pr_1 = p0_0 + 1U;
                if(pr_1 < _S9)
                {
                    kw_4 = *((&kernelContext_0)->k_0+(keys_1 + pr_1));
                }
                else
                {
                    kw_4 = 0U;
                }
                float _S78 = attnLo_0(kw_4);
                float _S79 = attnHi_0(kw_4);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_1]), _S79, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_1]), _S78, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_1]), _S79, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_1]), _S78, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S79, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S78, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_1]), _S79, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_1]), _S78, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S79, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S78, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_1]), _S79, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_1]), _S78, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S79, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S78, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_1]), _S79, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_1]), _S78, s_0[7U]));
                }
                uint pr_2 = p0_0 + 2U;
                if(pr_2 < _S9)
                {
                    kw_3 = *((&kernelContext_0)->k_0+(keys_1 + pr_2));
                }
                else
                {
                    kw_3 = 0U;
                }
                float _S80 = attnLo_0(kw_3);
                float _S81 = attnHi_0(kw_3);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_2]), _S81, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_2]), _S80, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_2]), _S81, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_2]), _S80, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S81, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S80, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_2]), _S81, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_2]), _S80, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S81, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S80, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_2]), _S81, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_2]), _S80, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S81, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S80, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_2]), _S81, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_2]), _S80, s_0[7U]));
                }
                uint pr_3 = p0_0 + 3U;
                if(pr_3 < _S9)
                {
                    kw_2 = *((&kernelContext_0)->k_0+(keys_1 + pr_3));
                }
                else
                {
                    kw_2 = 0U;
                }
                float _S82 = attnLo_0(kw_2);
                float _S83 = attnHi_0(kw_2);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_3]), _S83, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_3]), _S82, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_3]), _S83, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_3]), _S82, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S83, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S82, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_3]), _S83, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_3]), _S82, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S83, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S82, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_3]), _S83, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_3]), _S82, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S83, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S82, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_3]), _S83, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_3]), _S82, s_0[7U]));
                }
                uint pr_4 = p0_0 + 4U;
                if(pr_4 < _S9)
                {
                    kw_1 = *((&kernelContext_0)->k_0+(keys_1 + pr_4));
                }
                else
                {
                    kw_1 = 0U;
                }
                float _S84 = attnLo_0(kw_1);
                float _S85 = attnHi_0(kw_1);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_4]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_4]), _S84, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_4]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_4]), _S84, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S84, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_4]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_4]), _S84, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S84, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_4]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_4]), _S84, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S84, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_4]), _S85, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_4]), _S84, s_0[7U]));
                }
                uint pr_5 = p0_0 + 5U;
                if(pr_5 < _S9)
                {
                    kw_0 = *((&kernelContext_0)->k_0+(keys_1 + pr_5));
                }
                else
                {
                    kw_0 = 0U;
                }
                float _S86 = attnLo_0(kw_0);
                float _S87 = attnHi_0(kw_0);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_5]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_5]), _S86, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_5]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_5]), _S86, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S86, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_5]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_5]), _S86, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S86, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_5]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_5]), _S86, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S86, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_5]), _S87, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_5]), _S86, s_0[7U]));
                }
                uint pr_6 = p0_0 + 6U;
                uint kw_6;
                if(pr_6 < _S9)
                {
                    kw_6 = *((&kernelContext_0)->k_0+(keys_1 + pr_6));
                }
                else
                {
                    kw_6 = 0U;
                }
                float _S88 = attnLo_0(kw_6);
                float _S89 = attnHi_0(kw_6);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_6]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_6]), _S88, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_6]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_6]), _S88, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S88, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_6]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_6]), _S88, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S88, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_6]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_6]), _S88, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S88, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_6]), _S89, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_6]), _S88, s_0[7U]));
                }
                uint pr_7 = p0_0 + 7U;
                uint kw_7;
                if(pr_7 < _S9)
                {
                    kw_7 = *((&kernelContext_0)->k_0+(keys_1 + pr_7));
                }
                else
                {
                    kw_7 = 0U;
                }
                float _S90 = attnLo_0(kw_7);
                float _S91 = attnHi_0(kw_7);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_7]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_7]), _S90, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_7]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_7]), _S90, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S90, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_7]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_7]), _S90, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S90, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_7]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_7]), _S90, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S90, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_7]), _S91, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_7]), _S90, s_0[7U]));
                }
                p0_0 = p0_0 + 8U;
            }
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        if(0U >= _S6)
                        {
                            _S71 = int(2);
                            break;
                        }
                        if(USE_LENGTHS_0 != 0U)
                        {
                            visible_0 = key_0 < limit_0[0U];
                        }
                        else
                        {
                            if(((&kernelContext_0)->p_0->causal_0) == 0U)
                            {
                                visible_0 = true;
                            }
                            else
                            {
                                uint _S92 = 0U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S92 + _S10);
                            }
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
                        if(1U >= _S6)
                        {
                            _S69 = int(2);
                            break;
                        }
                        if(USE_LENGTHS_0 != 0U)
                        {
                            visible_0 = key_0 < limit_0[1U];
                        }
                        else
                        {
                            if(((&kernelContext_0)->p_0->causal_0) == 0U)
                            {
                                visible_0 = true;
                            }
                            else
                            {
                                uint _S93 = 1U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S93 + _S10);
                            }
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[1U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[_S17 + j_0] = m_0;
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
                        if(2U >= _S6)
                        {
                            _S67 = int(2);
                            break;
                        }
                        if(USE_LENGTHS_0 != 0U)
                        {
                            visible_0 = key_0 < limit_0[2U];
                        }
                        else
                        {
                            if(((&kernelContext_0)->p_0->causal_0) == 0U)
                            {
                                visible_0 = true;
                            }
                            else
                            {
                                uint _S94 = 2U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S94 + _S10);
                            }
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[2U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[2U * _S17 + j_0] = m_0;
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
                        if(3U >= _S6)
                        {
                            _S65 = int(2);
                            break;
                        }
                        if(USE_LENGTHS_0 != 0U)
                        {
                            visible_0 = key_0 < limit_0[3U];
                        }
                        else
                        {
                            if(((&kernelContext_0)->p_0->causal_0) == 0U)
                            {
                                visible_0 = true;
                            }
                            else
                            {
                                uint _S95 = 3U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S95 + _S10);
                            }
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[3U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[3U * _S17 + j_0] = m_0;
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
                for(;;)
                {
                    for(;;)
                    {
                        if(4U >= _S6)
                        {
                            _S63 = int(2);
                            break;
                        }
                        if(USE_LENGTHS_0 != 0U)
                        {
                            visible_0 = key_0 < limit_0[4U];
                        }
                        else
                        {
                            if(((&kernelContext_0)->p_0->causal_0) == 0U)
                            {
                                visible_0 = true;
                            }
                            else
                            {
                                uint _S96 = 4U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S96 + _S10);
                            }
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[4U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[4U * _S17 + j_0] = m_0;
                        _S63 = int(4);
                        break;
                    }
                    if(_S63 != int(4))
                    {
                        _S62 = _S63;
                        break;
                    }
                    _S62 = int(3);
                    break;
                }
                if(_S62 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(5U >= _S6)
                        {
                            _S61 = int(2);
                            break;
                        }
                        if(USE_LENGTHS_0 != 0U)
                        {
                            visible_0 = key_0 < limit_0[5U];
                        }
                        else
                        {
                            if(((&kernelContext_0)->p_0->causal_0) == 0U)
                            {
                                visible_0 = true;
                            }
                            else
                            {
                                uint _S97 = 5U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S97 + _S10);
                            }
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[5U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[5U * _S17 + j_0] = m_0;
                        _S61 = int(4);
                        break;
                    }
                    if(_S61 != int(4))
                    {
                        _S60 = _S61;
                        break;
                    }
                    _S60 = int(3);
                    break;
                }
                if(_S60 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(6U >= _S6)
                        {
                            _S59 = int(2);
                            break;
                        }
                        if(USE_LENGTHS_0 != 0U)
                        {
                            visible_0 = key_0 < limit_0[6U];
                        }
                        else
                        {
                            if(((&kernelContext_0)->p_0->causal_0) == 0U)
                            {
                                visible_0 = true;
                            }
                            else
                            {
                                uint _S98 = 6U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S98 + _S10);
                            }
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[6U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[6U * _S17 + j_0] = m_0;
                        _S59 = int(4);
                        break;
                    }
                    if(_S59 != int(4))
                    {
                        _S58 = _S59;
                        break;
                    }
                    _S58 = int(3);
                    break;
                }
                if(_S58 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(7U >= _S6)
                        {
                            _S57 = int(2);
                            break;
                        }
                        if(USE_LENGTHS_0 != 0U)
                        {
                            visible_0 = key_0 < limit_0[7U];
                        }
                        else
                        {
                            if(((&kernelContext_0)->p_0->causal_0) == 0U)
                            {
                                visible_0 = true;
                            }
                            else
                            {
                                uint _S99 = 7U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S99 + _S10);
                            }
                        }
                        if(visible_0)
                        {
                            m_0 = s_0[7U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            m_0 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[7U * _S17 + j_0] = m_0;
                        _S57 = int(4);
                        break;
                    }
                    if(_S57 != int(4))
                    {
                        _S56 = _S57;
                        break;
                    }
                    _S56 = int(3);
                    break;
                }
                if(_S56 != int(3))
                {
                    break;
                }
                break;
            }
            j_0 = j_0 + 128U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        p0_0 = _S53;
        for(;;)
        {
            if(p0_0 < _S6)
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
                if(kw_5 < _S72)
                {
                }
                else
                {
                    break;
                }
                float _S101 = max(m_0, (*(&kernelContext_0)->ss_0)[p0_0 * _S17 + kw_5]);
                kw_5 = kw_5 + _S50;
                m_0 = _S101;
            }
            float m_1 = simd_max(m_0);
            float mOld_0 = (*(&kernelContext_0)->rowM_0)[p0_0];
            float _S102 = max((*(&kernelContext_0)->rowM_0)[p0_0], m_1);
            kw_4 = lane_0;
            sum_0 = 0.0;
            for(;;)
            {
                if(kw_4 < _S72)
                {
                }
                else
                {
                    break;
                }
                float sv_0 = (*(&kernelContext_0)->ss_0)[p0_0 * _S17 + kw_4];
                if(((*(&kernelContext_0)->ss_0)[p0_0 * _S17 + kw_4]) == _S100)
                {
                    w_3 = 0.0;
                }
                else
                {
                    w_3 = exp(sv_0 - _S102);
                }
                (*(&kernelContext_0)->ss_0)[p0_0 * _S17 + kw_4] = w_3;
                float sum_1 = sum_0 + w_3;
                kw_4 = kw_4 + _S50;
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
            if(_S51)
            {
                (*(&kernelContext_0)->rowA_0)[p0_0] = w_3;
                (*(&kernelContext_0)->rowL_0)[p0_0] = (*(&kernelContext_0)->rowL_0)[p0_0] * w_3 + sum_2;
                (*(&kernelContext_0)->rowM_0)[p0_0] = _S102;
            }
            p0_0 = p0_0 + *(&kernelContext_0)->waves_0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(0U < _S6)
        {
            float a_0 = (*(&kernelContext_0)->rowA_0)[0U];
            lo_0[0U] = lo_0[0U] * (*(&kernelContext_0)->rowA_0)[0U];
            hi_0[0U] = hi_0[0U] * a_0;
        }
        if(1U < _S6)
        {
            float a_1 = (*(&kernelContext_0)->rowA_0)[1U];
            lo_0[1U] = lo_0[1U] * (*(&kernelContext_0)->rowA_0)[1U];
            hi_0[1U] = hi_0[1U] * a_1;
        }
        if(2U < _S6)
        {
            float a_2 = (*(&kernelContext_0)->rowA_0)[2U];
            lo_0[2U] = lo_0[2U] * (*(&kernelContext_0)->rowA_0)[2U];
            hi_0[2U] = hi_0[2U] * a_2;
        }
        if(3U < _S6)
        {
            float a_3 = (*(&kernelContext_0)->rowA_0)[3U];
            lo_0[3U] = lo_0[3U] * (*(&kernelContext_0)->rowA_0)[3U];
            hi_0[3U] = hi_0[3U] * a_3;
        }
        if(4U < _S6)
        {
            float a_4 = (*(&kernelContext_0)->rowA_0)[4U];
            lo_0[4U] = lo_0[4U] * (*(&kernelContext_0)->rowA_0)[4U];
            hi_0[4U] = hi_0[4U] * a_4;
        }
        if(5U < _S6)
        {
            float a_5 = (*(&kernelContext_0)->rowA_0)[5U];
            lo_0[5U] = lo_0[5U] * (*(&kernelContext_0)->rowA_0)[5U];
            hi_0[5U] = hi_0[5U] * a_5;
        }
        if(6U < _S6)
        {
            float a_6 = (*(&kernelContext_0)->rowA_0)[6U];
            lo_0[6U] = lo_0[6U] * (*(&kernelContext_0)->rowA_0)[6U];
            hi_0[6U] = hi_0[6U] * a_6;
        }
        if(7U < _S6)
        {
            float a_7 = (*(&kernelContext_0)->rowA_0)[7U];
            lo_0[7U] = lo_0[7U] * (*(&kernelContext_0)->rowA_0)[7U];
            hi_0[7U] = hi_0[7U] * a_7;
        }
        if(_S54 < _S9)
        {
            kw_5 = _S55;
            for(;;)
            {
                bool _S103 = kw_5 < _S72;
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
                        keys_1 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + kw_5)) * _S9 + _S54;
                    }
                    else
                    {
                        uint _S104 = c0_0 + kw_5;
                        uint _S105 = _S104 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S106 = (*((&kernelContext_0)->table_0+(r_0 + _S105)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S107 = _S104 % (&kernelContext_0)->p_0->pageSize_0;
                        keys_1 = (_S106 + _S107) * _S9 + _S54;
                    }
                    keys_1 = *((&kernelContext_0)->v_0+keys_1);
                }
                else
                {
                    keys_1 = 0U;
                }
                vw_0[0U] = keys_1;
                uint j_1 = kw_5 + 4U;
                if(j_1 < _S72)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        e_0 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_1)) * _S9 + _S54;
                    }
                    else
                    {
                        uint _S108 = c0_0 + j_1;
                        uint _S109 = _S108 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S110 = (*((&kernelContext_0)->table_0+(r_0 + _S109)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S111 = _S108 % (&kernelContext_0)->p_0->pageSize_0;
                        e_0 = (_S110 + _S111) * _S9 + _S54;
                    }
                    e_0 = *((&kernelContext_0)->v_0+e_0);
                }
                else
                {
                    e_0 = 0U;
                }
                vw_0[1U] = e_0;
                uint j_2 = kw_5 + 8U;
                if(j_2 < _S72)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        w_2 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_2)) * _S9 + _S54;
                    }
                    else
                    {
                        uint _S112 = c0_0 + j_2;
                        uint _S113 = _S112 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S114 = (*((&kernelContext_0)->table_0+(r_0 + _S113)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S115 = _S112 % (&kernelContext_0)->p_0->pageSize_0;
                        w_2 = (_S114 + _S115) * _S9 + _S54;
                    }
                    w_2 = *((&kernelContext_0)->v_0+w_2);
                }
                else
                {
                    w_2 = 0U;
                }
                vw_0[2U] = w_2;
                uint j_3 = kw_5 + 12U;
                if(j_3 < _S72)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_4 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_3)) * _S9 + _S54;
                    }
                    else
                    {
                        uint _S116 = c0_0 + j_3;
                        uint _S117 = _S116 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S118 = (*((&kernelContext_0)->table_0+(r_0 + _S117)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S119 = _S116 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_4 = (_S118 + _S119) * _S9 + _S54;
                    }
                    kw_4 = *((&kernelContext_0)->v_0+kw_4);
                }
                else
                {
                    kw_4 = 0U;
                }
                vw_0[3U] = kw_4;
                uint j_4 = kw_5 + 16U;
                if(j_4 < _S72)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_3 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_4)) * _S9 + _S54;
                    }
                    else
                    {
                        uint _S120 = c0_0 + j_4;
                        uint _S121 = _S120 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S122 = (*((&kernelContext_0)->table_0+(r_0 + _S121)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S123 = _S120 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_3 = (_S122 + _S123) * _S9 + _S54;
                    }
                    kw_3 = *((&kernelContext_0)->v_0+kw_3);
                }
                else
                {
                    kw_3 = 0U;
                }
                vw_0[4U] = kw_3;
                uint j_5 = kw_5 + 20U;
                if(j_5 < _S72)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_2 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_5)) * _S9 + _S54;
                    }
                    else
                    {
                        uint _S124 = c0_0 + j_5;
                        uint _S125 = _S124 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S126 = (*((&kernelContext_0)->table_0+(r_0 + _S125)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S127 = _S124 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_2 = (_S126 + _S127) * _S9 + _S54;
                    }
                    kw_2 = *((&kernelContext_0)->v_0+kw_2);
                }
                else
                {
                    kw_2 = 0U;
                }
                vw_0[5U] = kw_2;
                uint j_6 = kw_5 + 24U;
                if(j_6 < _S72)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_1 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_6)) * _S9 + _S54;
                    }
                    else
                    {
                        uint _S128 = c0_0 + j_6;
                        uint _S129 = _S128 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S130 = (*((&kernelContext_0)->table_0+(r_0 + _S129)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S131 = _S128 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_1 = (_S130 + _S131) * _S9 + _S54;
                    }
                    kw_1 = *((&kernelContext_0)->v_0+kw_1);
                }
                else
                {
                    kw_1 = 0U;
                }
                vw_0[6U] = kw_1;
                uint j_7 = kw_5 + 28U;
                if(j_7 < _S72)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_0 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (c0_0 + j_7)) * _S9 + _S54;
                    }
                    else
                    {
                        uint _S132 = c0_0 + j_7;
                        uint _S133 = _S132 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S134 = (*((&kernelContext_0)->table_0+(r_0 + _S133)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S135 = _S132 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_0 = (_S134 + _S135) * _S9 + _S54;
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
                            if(kw_5 >= _S72)
                            {
                                _S71 = int(2);
                                visible_0 = false;
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
                                        bool _S138 = 0U >= _S6;
                                        _S5 = _S138;
                                        if(_S138)
                                        {
                                            _S71 = int(5);
                                            break;
                                        }
                                        float pj_0 = (*(&kernelContext_0)->ss_0)[kw_5];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[kw_5], _S136, lo_0[0U]);
                                        hi_0[0U] = fma(pj_0, _S137, hi_0[0U]);
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
                                        if(1U >= _S6)
                                        {
                                            _S69 = int(5);
                                            break;
                                        }
                                        float pj_1 = (*(&kernelContext_0)->ss_0)[_S17 + kw_5];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S17 + kw_5], _S136, lo_0[1U]);
                                        hi_0[1U] = fma(pj_1, _S137, hi_0[1U]);
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
                                        if(2U >= _S6)
                                        {
                                            _S67 = int(5);
                                            break;
                                        }
                                        float pj_2 = (*(&kernelContext_0)->ss_0)[2U * _S17 + kw_5];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S17 + kw_5], _S136, lo_0[2U]);
                                        hi_0[2U] = fma(pj_2, _S137, hi_0[2U]);
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
                                        if(3U >= _S6)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_3 = (*(&kernelContext_0)->ss_0)[3U * _S17 + kw_5];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S17 + kw_5], _S136, lo_0[3U]);
                                        hi_0[3U] = fma(pj_3, _S137, hi_0[3U]);
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
                                        if(4U >= _S6)
                                        {
                                            _S63 = int(5);
                                            break;
                                        }
                                        float pj_4 = (*(&kernelContext_0)->ss_0)[4U * _S17 + kw_5];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S17 + kw_5], _S136, lo_0[4U]);
                                        hi_0[4U] = fma(pj_4, _S137, hi_0[4U]);
                                        _S63 = int(7);
                                        break;
                                    }
                                    if(_S63 != int(7))
                                    {
                                        _S62 = _S63;
                                        break;
                                    }
                                    _S62 = int(6);
                                    break;
                                }
                                if(_S62 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S6)
                                        {
                                            _S61 = int(5);
                                            break;
                                        }
                                        float pj_5 = (*(&kernelContext_0)->ss_0)[5U * _S17 + kw_5];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S17 + kw_5], _S136, lo_0[5U]);
                                        hi_0[5U] = fma(pj_5, _S137, hi_0[5U]);
                                        _S61 = int(7);
                                        break;
                                    }
                                    if(_S61 != int(7))
                                    {
                                        _S60 = _S61;
                                        break;
                                    }
                                    _S60 = int(6);
                                    break;
                                }
                                if(_S60 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S6)
                                        {
                                            _S59 = int(5);
                                            break;
                                        }
                                        float pj_6 = (*(&kernelContext_0)->ss_0)[6U * _S17 + kw_5];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S17 + kw_5], _S136, lo_0[6U]);
                                        hi_0[6U] = fma(pj_6, _S137, hi_0[6U]);
                                        _S59 = int(7);
                                        break;
                                    }
                                    if(_S59 != int(7))
                                    {
                                        _S58 = _S59;
                                        break;
                                    }
                                    _S58 = int(6);
                                    break;
                                }
                                if(_S58 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S6)
                                        {
                                            _S57 = int(5);
                                            break;
                                        }
                                        float pj_7 = (*(&kernelContext_0)->ss_0)[7U * _S17 + kw_5];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S17 + kw_5], _S136, lo_0[7U]);
                                        hi_0[7U] = fma(pj_7, _S137, hi_0[7U]);
                                        _S57 = int(7);
                                        break;
                                    }
                                    if(_S57 != int(7))
                                    {
                                        _S56 = _S57;
                                        break;
                                    }
                                    _S56 = int(6);
                                    break;
                                }
                                if(_S56 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S71 = int(4);
                            visible_0 = _S5;
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
                    int _S139;
                    int _S140;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_1 >= _S72)
                            {
                                _S69 = int(2);
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
                                        if(visible_0)
                                        {
                                            _S69 = int(5);
                                            break;
                                        }
                                        float pj_8 = (*(&kernelContext_0)->ss_0)[j_1];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_1], _S141, lo_0[0U]);
                                        hi_0[0U] = fma(pj_8, _S142, hi_0[0U]);
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
                                        if(1U >= _S6)
                                        {
                                            _S67 = int(5);
                                            break;
                                        }
                                        float pj_9 = (*(&kernelContext_0)->ss_0)[_S17 + j_1];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S17 + j_1], _S141, lo_0[1U]);
                                        hi_0[1U] = fma(pj_9, _S142, hi_0[1U]);
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
                                        if(2U >= _S6)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_10 = (*(&kernelContext_0)->ss_0)[2U * _S17 + j_1];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S17 + j_1], _S141, lo_0[2U]);
                                        hi_0[2U] = fma(pj_10, _S142, hi_0[2U]);
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
                                        if(3U >= _S6)
                                        {
                                            _S63 = int(5);
                                            break;
                                        }
                                        float pj_11 = (*(&kernelContext_0)->ss_0)[3U * _S17 + j_1];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S17 + j_1], _S141, lo_0[3U]);
                                        hi_0[3U] = fma(pj_11, _S142, hi_0[3U]);
                                        _S63 = int(7);
                                        break;
                                    }
                                    if(_S63 != int(7))
                                    {
                                        _S62 = _S63;
                                        break;
                                    }
                                    _S62 = int(6);
                                    break;
                                }
                                if(_S62 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S6)
                                        {
                                            _S61 = int(5);
                                            break;
                                        }
                                        float pj_12 = (*(&kernelContext_0)->ss_0)[4U * _S17 + j_1];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S17 + j_1], _S141, lo_0[4U]);
                                        hi_0[4U] = fma(pj_12, _S142, hi_0[4U]);
                                        _S61 = int(7);
                                        break;
                                    }
                                    if(_S61 != int(7))
                                    {
                                        _S60 = _S61;
                                        break;
                                    }
                                    _S60 = int(6);
                                    break;
                                }
                                if(_S60 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S6)
                                        {
                                            _S59 = int(5);
                                            break;
                                        }
                                        float pj_13 = (*(&kernelContext_0)->ss_0)[5U * _S17 + j_1];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S17 + j_1], _S141, lo_0[5U]);
                                        hi_0[5U] = fma(pj_13, _S142, hi_0[5U]);
                                        _S59 = int(7);
                                        break;
                                    }
                                    if(_S59 != int(7))
                                    {
                                        _S58 = _S59;
                                        break;
                                    }
                                    _S58 = int(6);
                                    break;
                                }
                                if(_S58 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S6)
                                        {
                                            _S57 = int(5);
                                            break;
                                        }
                                        float pj_14 = (*(&kernelContext_0)->ss_0)[6U * _S17 + j_1];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S17 + j_1], _S141, lo_0[6U]);
                                        hi_0[6U] = fma(pj_14, _S142, hi_0[6U]);
                                        _S57 = int(7);
                                        break;
                                    }
                                    if(_S57 != int(7))
                                    {
                                        _S56 = _S57;
                                        break;
                                    }
                                    _S56 = int(6);
                                    break;
                                }
                                if(_S56 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S6)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_15 = (*(&kernelContext_0)->ss_0)[7U * _S17 + j_1];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S17 + j_1], _S141, lo_0[7U]);
                                        hi_0[7U] = fma(pj_15, _S142, hi_0[7U]);
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
                    int _S143;
                    int _S144;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_2 >= _S72)
                            {
                                _S67 = int(2);
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
                                        if(visible_0)
                                        {
                                            _S67 = int(5);
                                            break;
                                        }
                                        float pj_16 = (*(&kernelContext_0)->ss_0)[j_2];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_2], _S145, lo_0[0U]);
                                        hi_0[0U] = fma(pj_16, _S146, hi_0[0U]);
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
                                        if(1U >= _S6)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_17 = (*(&kernelContext_0)->ss_0)[_S17 + j_2];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S17 + j_2], _S145, lo_0[1U]);
                                        hi_0[1U] = fma(pj_17, _S146, hi_0[1U]);
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
                                        if(2U >= _S6)
                                        {
                                            _S63 = int(5);
                                            break;
                                        }
                                        float pj_18 = (*(&kernelContext_0)->ss_0)[2U * _S17 + j_2];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S17 + j_2], _S145, lo_0[2U]);
                                        hi_0[2U] = fma(pj_18, _S146, hi_0[2U]);
                                        _S63 = int(7);
                                        break;
                                    }
                                    if(_S63 != int(7))
                                    {
                                        _S62 = _S63;
                                        break;
                                    }
                                    _S62 = int(6);
                                    break;
                                }
                                if(_S62 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S6)
                                        {
                                            _S61 = int(5);
                                            break;
                                        }
                                        float pj_19 = (*(&kernelContext_0)->ss_0)[3U * _S17 + j_2];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S17 + j_2], _S145, lo_0[3U]);
                                        hi_0[3U] = fma(pj_19, _S146, hi_0[3U]);
                                        _S61 = int(7);
                                        break;
                                    }
                                    if(_S61 != int(7))
                                    {
                                        _S60 = _S61;
                                        break;
                                    }
                                    _S60 = int(6);
                                    break;
                                }
                                if(_S60 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S6)
                                        {
                                            _S59 = int(5);
                                            break;
                                        }
                                        float pj_20 = (*(&kernelContext_0)->ss_0)[4U * _S17 + j_2];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S17 + j_2], _S145, lo_0[4U]);
                                        hi_0[4U] = fma(pj_20, _S146, hi_0[4U]);
                                        _S59 = int(7);
                                        break;
                                    }
                                    if(_S59 != int(7))
                                    {
                                        _S58 = _S59;
                                        break;
                                    }
                                    _S58 = int(6);
                                    break;
                                }
                                if(_S58 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S6)
                                        {
                                            _S57 = int(5);
                                            break;
                                        }
                                        float pj_21 = (*(&kernelContext_0)->ss_0)[5U * _S17 + j_2];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S17 + j_2], _S145, lo_0[5U]);
                                        hi_0[5U] = fma(pj_21, _S146, hi_0[5U]);
                                        _S57 = int(7);
                                        break;
                                    }
                                    if(_S57 != int(7))
                                    {
                                        _S56 = _S57;
                                        break;
                                    }
                                    _S56 = int(6);
                                    break;
                                }
                                if(_S56 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S6)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_22 = (*(&kernelContext_0)->ss_0)[6U * _S17 + j_2];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S17 + j_2], _S145, lo_0[6U]);
                                        hi_0[6U] = fma(pj_22, _S146, hi_0[6U]);
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
                                        if(7U >= _S6)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_23 = (*(&kernelContext_0)->ss_0)[7U * _S17 + j_2];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S17 + j_2], _S145, lo_0[7U]);
                                        hi_0[7U] = fma(pj_23, _S146, hi_0[7U]);
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
                    int _S147;
                    int _S148;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_3 >= _S72)
                            {
                                _S65 = int(2);
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
                                        if(visible_0)
                                        {
                                            _S65 = int(5);
                                            break;
                                        }
                                        float pj_24 = (*(&kernelContext_0)->ss_0)[j_3];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_3], _S149, lo_0[0U]);
                                        hi_0[0U] = fma(pj_24, _S150, hi_0[0U]);
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
                                        if(1U >= _S6)
                                        {
                                            _S63 = int(5);
                                            break;
                                        }
                                        float pj_25 = (*(&kernelContext_0)->ss_0)[_S17 + j_3];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S17 + j_3], _S149, lo_0[1U]);
                                        hi_0[1U] = fma(pj_25, _S150, hi_0[1U]);
                                        _S63 = int(7);
                                        break;
                                    }
                                    if(_S63 != int(7))
                                    {
                                        _S62 = _S63;
                                        break;
                                    }
                                    _S62 = int(6);
                                    break;
                                }
                                if(_S62 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S6)
                                        {
                                            _S61 = int(5);
                                            break;
                                        }
                                        float pj_26 = (*(&kernelContext_0)->ss_0)[2U * _S17 + j_3];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S17 + j_3], _S149, lo_0[2U]);
                                        hi_0[2U] = fma(pj_26, _S150, hi_0[2U]);
                                        _S61 = int(7);
                                        break;
                                    }
                                    if(_S61 != int(7))
                                    {
                                        _S60 = _S61;
                                        break;
                                    }
                                    _S60 = int(6);
                                    break;
                                }
                                if(_S60 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S6)
                                        {
                                            _S59 = int(5);
                                            break;
                                        }
                                        float pj_27 = (*(&kernelContext_0)->ss_0)[3U * _S17 + j_3];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S17 + j_3], _S149, lo_0[3U]);
                                        hi_0[3U] = fma(pj_27, _S150, hi_0[3U]);
                                        _S59 = int(7);
                                        break;
                                    }
                                    if(_S59 != int(7))
                                    {
                                        _S58 = _S59;
                                        break;
                                    }
                                    _S58 = int(6);
                                    break;
                                }
                                if(_S58 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S6)
                                        {
                                            _S57 = int(5);
                                            break;
                                        }
                                        float pj_28 = (*(&kernelContext_0)->ss_0)[4U * _S17 + j_3];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S17 + j_3], _S149, lo_0[4U]);
                                        hi_0[4U] = fma(pj_28, _S150, hi_0[4U]);
                                        _S57 = int(7);
                                        break;
                                    }
                                    if(_S57 != int(7))
                                    {
                                        _S56 = _S57;
                                        break;
                                    }
                                    _S56 = int(6);
                                    break;
                                }
                                if(_S56 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S6)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_29 = (*(&kernelContext_0)->ss_0)[5U * _S17 + j_3];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S17 + j_3], _S149, lo_0[5U]);
                                        hi_0[5U] = fma(pj_29, _S150, hi_0[5U]);
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
                                        if(6U >= _S6)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_30 = (*(&kernelContext_0)->ss_0)[6U * _S17 + j_3];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S17 + j_3], _S149, lo_0[6U]);
                                        hi_0[6U] = fma(pj_30, _S150, hi_0[6U]);
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
                                        if(7U >= _S6)
                                        {
                                            _S148 = int(5);
                                            break;
                                        }
                                        float pj_31 = (*(&kernelContext_0)->ss_0)[7U * _S17 + j_3];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S17 + j_3], _S149, lo_0[7U]);
                                        hi_0[7U] = fma(pj_31, _S150, hi_0[7U]);
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
                    int _S151;
                    int _S152;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_4 >= _S72)
                            {
                                _S63 = int(2);
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
                                        if(visible_0)
                                        {
                                            _S63 = int(5);
                                            break;
                                        }
                                        float pj_32 = (*(&kernelContext_0)->ss_0)[j_4];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_4], _S153, lo_0[0U]);
                                        hi_0[0U] = fma(pj_32, _S154, hi_0[0U]);
                                        _S63 = int(7);
                                        break;
                                    }
                                    if(_S63 != int(7))
                                    {
                                        _S62 = _S63;
                                        break;
                                    }
                                    _S62 = int(6);
                                    break;
                                }
                                if(_S62 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S6)
                                        {
                                            _S61 = int(5);
                                            break;
                                        }
                                        float pj_33 = (*(&kernelContext_0)->ss_0)[_S17 + j_4];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S17 + j_4], _S153, lo_0[1U]);
                                        hi_0[1U] = fma(pj_33, _S154, hi_0[1U]);
                                        _S61 = int(7);
                                        break;
                                    }
                                    if(_S61 != int(7))
                                    {
                                        _S60 = _S61;
                                        break;
                                    }
                                    _S60 = int(6);
                                    break;
                                }
                                if(_S60 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S6)
                                        {
                                            _S59 = int(5);
                                            break;
                                        }
                                        float pj_34 = (*(&kernelContext_0)->ss_0)[2U * _S17 + j_4];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S17 + j_4], _S153, lo_0[2U]);
                                        hi_0[2U] = fma(pj_34, _S154, hi_0[2U]);
                                        _S59 = int(7);
                                        break;
                                    }
                                    if(_S59 != int(7))
                                    {
                                        _S58 = _S59;
                                        break;
                                    }
                                    _S58 = int(6);
                                    break;
                                }
                                if(_S58 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S6)
                                        {
                                            _S57 = int(5);
                                            break;
                                        }
                                        float pj_35 = (*(&kernelContext_0)->ss_0)[3U * _S17 + j_4];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S17 + j_4], _S153, lo_0[3U]);
                                        hi_0[3U] = fma(pj_35, _S154, hi_0[3U]);
                                        _S57 = int(7);
                                        break;
                                    }
                                    if(_S57 != int(7))
                                    {
                                        _S56 = _S57;
                                        break;
                                    }
                                    _S56 = int(6);
                                    break;
                                }
                                if(_S56 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S6)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_36 = (*(&kernelContext_0)->ss_0)[4U * _S17 + j_4];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S17 + j_4], _S153, lo_0[4U]);
                                        hi_0[4U] = fma(pj_36, _S154, hi_0[4U]);
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
                                        if(5U >= _S6)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_37 = (*(&kernelContext_0)->ss_0)[5U * _S17 + j_4];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S17 + j_4], _S153, lo_0[5U]);
                                        hi_0[5U] = fma(pj_37, _S154, hi_0[5U]);
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
                                        if(6U >= _S6)
                                        {
                                            _S148 = int(5);
                                            break;
                                        }
                                        float pj_38 = (*(&kernelContext_0)->ss_0)[6U * _S17 + j_4];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S17 + j_4], _S153, lo_0[6U]);
                                        hi_0[6U] = fma(pj_38, _S154, hi_0[6U]);
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
                                        if(7U >= _S6)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_39 = (*(&kernelContext_0)->ss_0)[7U * _S17 + j_4];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S17 + j_4], _S153, lo_0[7U]);
                                        hi_0[7U] = fma(pj_39, _S154, hi_0[7U]);
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
                            _S63 = int(4);
                            break;
                        }
                        if(_S63 != int(4))
                        {
                            _S62 = _S63;
                            break;
                        }
                        _S62 = int(3);
                        break;
                    }
                    if(_S62 != int(3))
                    {
                        break;
                    }
                    int _S155;
                    int _S156;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_5 >= _S72)
                            {
                                _S61 = int(2);
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
                                        if(visible_0)
                                        {
                                            _S61 = int(5);
                                            break;
                                        }
                                        float pj_40 = (*(&kernelContext_0)->ss_0)[j_5];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_5], _S157, lo_0[0U]);
                                        hi_0[0U] = fma(pj_40, _S158, hi_0[0U]);
                                        _S61 = int(7);
                                        break;
                                    }
                                    if(_S61 != int(7))
                                    {
                                        _S60 = _S61;
                                        break;
                                    }
                                    _S60 = int(6);
                                    break;
                                }
                                if(_S60 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S6)
                                        {
                                            _S59 = int(5);
                                            break;
                                        }
                                        float pj_41 = (*(&kernelContext_0)->ss_0)[_S17 + j_5];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S17 + j_5], _S157, lo_0[1U]);
                                        hi_0[1U] = fma(pj_41, _S158, hi_0[1U]);
                                        _S59 = int(7);
                                        break;
                                    }
                                    if(_S59 != int(7))
                                    {
                                        _S58 = _S59;
                                        break;
                                    }
                                    _S58 = int(6);
                                    break;
                                }
                                if(_S58 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S6)
                                        {
                                            _S57 = int(5);
                                            break;
                                        }
                                        float pj_42 = (*(&kernelContext_0)->ss_0)[2U * _S17 + j_5];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S17 + j_5], _S157, lo_0[2U]);
                                        hi_0[2U] = fma(pj_42, _S158, hi_0[2U]);
                                        _S57 = int(7);
                                        break;
                                    }
                                    if(_S57 != int(7))
                                    {
                                        _S56 = _S57;
                                        break;
                                    }
                                    _S56 = int(6);
                                    break;
                                }
                                if(_S56 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S6)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_43 = (*(&kernelContext_0)->ss_0)[3U * _S17 + j_5];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S17 + j_5], _S157, lo_0[3U]);
                                        hi_0[3U] = fma(pj_43, _S158, hi_0[3U]);
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
                                        if(4U >= _S6)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_44 = (*(&kernelContext_0)->ss_0)[4U * _S17 + j_5];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S17 + j_5], _S157, lo_0[4U]);
                                        hi_0[4U] = fma(pj_44, _S158, hi_0[4U]);
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
                                        if(5U >= _S6)
                                        {
                                            _S148 = int(5);
                                            break;
                                        }
                                        float pj_45 = (*(&kernelContext_0)->ss_0)[5U * _S17 + j_5];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S17 + j_5], _S157, lo_0[5U]);
                                        hi_0[5U] = fma(pj_45, _S158, hi_0[5U]);
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
                                        if(6U >= _S6)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_46 = (*(&kernelContext_0)->ss_0)[6U * _S17 + j_5];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S17 + j_5], _S157, lo_0[6U]);
                                        hi_0[6U] = fma(pj_46, _S158, hi_0[6U]);
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
                                        if(7U >= _S6)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_47 = (*(&kernelContext_0)->ss_0)[7U * _S17 + j_5];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S17 + j_5], _S157, lo_0[7U]);
                                        hi_0[7U] = fma(pj_47, _S158, hi_0[7U]);
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
                            _S61 = int(4);
                            break;
                        }
                        if(_S61 != int(4))
                        {
                            _S60 = _S61;
                            break;
                        }
                        _S60 = int(3);
                        break;
                    }
                    if(_S60 != int(3))
                    {
                        break;
                    }
                    int _S159;
                    int _S160;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_6 >= _S72)
                            {
                                _S59 = int(2);
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
                                        if(visible_0)
                                        {
                                            _S59 = int(5);
                                            break;
                                        }
                                        float pj_48 = (*(&kernelContext_0)->ss_0)[j_6];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_6], _S161, lo_0[0U]);
                                        hi_0[0U] = fma(pj_48, _S162, hi_0[0U]);
                                        _S59 = int(7);
                                        break;
                                    }
                                    if(_S59 != int(7))
                                    {
                                        _S58 = _S59;
                                        break;
                                    }
                                    _S58 = int(6);
                                    break;
                                }
                                if(_S58 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S6)
                                        {
                                            _S57 = int(5);
                                            break;
                                        }
                                        float pj_49 = (*(&kernelContext_0)->ss_0)[_S17 + j_6];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S17 + j_6], _S161, lo_0[1U]);
                                        hi_0[1U] = fma(pj_49, _S162, hi_0[1U]);
                                        _S57 = int(7);
                                        break;
                                    }
                                    if(_S57 != int(7))
                                    {
                                        _S56 = _S57;
                                        break;
                                    }
                                    _S56 = int(6);
                                    break;
                                }
                                if(_S56 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S6)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_50 = (*(&kernelContext_0)->ss_0)[2U * _S17 + j_6];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S17 + j_6], _S161, lo_0[2U]);
                                        hi_0[2U] = fma(pj_50, _S162, hi_0[2U]);
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
                                        if(3U >= _S6)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_51 = (*(&kernelContext_0)->ss_0)[3U * _S17 + j_6];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S17 + j_6], _S161, lo_0[3U]);
                                        hi_0[3U] = fma(pj_51, _S162, hi_0[3U]);
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
                                        if(4U >= _S6)
                                        {
                                            _S148 = int(5);
                                            break;
                                        }
                                        float pj_52 = (*(&kernelContext_0)->ss_0)[4U * _S17 + j_6];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S17 + j_6], _S161, lo_0[4U]);
                                        hi_0[4U] = fma(pj_52, _S162, hi_0[4U]);
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
                                        if(5U >= _S6)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_53 = (*(&kernelContext_0)->ss_0)[5U * _S17 + j_6];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S17 + j_6], _S161, lo_0[5U]);
                                        hi_0[5U] = fma(pj_53, _S162, hi_0[5U]);
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
                                        if(6U >= _S6)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_54 = (*(&kernelContext_0)->ss_0)[6U * _S17 + j_6];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S17 + j_6], _S161, lo_0[6U]);
                                        hi_0[6U] = fma(pj_54, _S162, hi_0[6U]);
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
                                        if(7U >= _S6)
                                        {
                                            _S160 = int(5);
                                            break;
                                        }
                                        float pj_55 = (*(&kernelContext_0)->ss_0)[7U * _S17 + j_6];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S17 + j_6], _S161, lo_0[7U]);
                                        hi_0[7U] = fma(pj_55, _S162, hi_0[7U]);
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
                            _S59 = int(4);
                            break;
                        }
                        if(_S59 != int(4))
                        {
                            _S58 = _S59;
                            break;
                        }
                        _S58 = int(3);
                        break;
                    }
                    if(_S58 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_7 >= _S72)
                            {
                                _S57 = int(2);
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
                                        if(visible_0)
                                        {
                                            _S57 = int(5);
                                            break;
                                        }
                                        float pj_56 = (*(&kernelContext_0)->ss_0)[j_7];
                                        lo_0[0U] = fma((*(&kernelContext_0)->ss_0)[j_7], _S163, lo_0[0U]);
                                        hi_0[0U] = fma(pj_56, _S164, hi_0[0U]);
                                        _S57 = int(7);
                                        break;
                                    }
                                    if(_S57 != int(7))
                                    {
                                        _S56 = _S57;
                                        break;
                                    }
                                    _S56 = int(6);
                                    break;
                                }
                                if(_S56 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S6)
                                        {
                                            _S140 = int(5);
                                            break;
                                        }
                                        float pj_57 = (*(&kernelContext_0)->ss_0)[_S17 + j_7];
                                        lo_0[1U] = fma((*(&kernelContext_0)->ss_0)[_S17 + j_7], _S163, lo_0[1U]);
                                        hi_0[1U] = fma(pj_57, _S164, hi_0[1U]);
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
                                        if(2U >= _S6)
                                        {
                                            _S144 = int(5);
                                            break;
                                        }
                                        float pj_58 = (*(&kernelContext_0)->ss_0)[2U * _S17 + j_7];
                                        lo_0[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S17 + j_7], _S163, lo_0[2U]);
                                        hi_0[2U] = fma(pj_58, _S164, hi_0[2U]);
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
                                        if(3U >= _S6)
                                        {
                                            _S148 = int(5);
                                            break;
                                        }
                                        float pj_59 = (*(&kernelContext_0)->ss_0)[3U * _S17 + j_7];
                                        lo_0[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S17 + j_7], _S163, lo_0[3U]);
                                        hi_0[3U] = fma(pj_59, _S164, hi_0[3U]);
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
                                        if(4U >= _S6)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_60 = (*(&kernelContext_0)->ss_0)[4U * _S17 + j_7];
                                        lo_0[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S17 + j_7], _S163, lo_0[4U]);
                                        hi_0[4U] = fma(pj_60, _S164, hi_0[4U]);
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
                                        if(5U >= _S6)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_61 = (*(&kernelContext_0)->ss_0)[5U * _S17 + j_7];
                                        lo_0[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S17 + j_7], _S163, lo_0[5U]);
                                        hi_0[5U] = fma(pj_61, _S164, hi_0[5U]);
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
                                        if(6U >= _S6)
                                        {
                                            _S160 = int(5);
                                            break;
                                        }
                                        float pj_62 = (*(&kernelContext_0)->ss_0)[6U * _S17 + j_7];
                                        lo_0[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S17 + j_7], _S163, lo_0[6U]);
                                        hi_0[6U] = fma(pj_62, _S164, hi_0[6U]);
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
                                        if(7U >= _S6)
                                        {
                                            _S166 = int(5);
                                            break;
                                        }
                                        float pj_63 = (*(&kernelContext_0)->ss_0)[7U * _S17 + j_7];
                                        lo_0[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S17 + j_7], _S163, lo_0[7U]);
                                        hi_0[7U] = fma(pj_63, _S164, hi_0[7U]);
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
                            _S57 = int(4);
                            break;
                        }
                        if(_S57 != int(4))
                        {
                            _S56 = _S57;
                            break;
                        }
                        _S56 = int(3);
                        break;
                    }
                    if(_S56 != int(3))
                    {
                        break;
                    }
                    break;
                }
                kw_5 = kw_5 + 32U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c0_0 = c0_0 + _S17;
    }
    r_0 = 0U;
    for(;;)
    {
        if(r_0 < _S6)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(0U == r_0)
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
        if(1U == r_0)
        {
            w_3 = lo_0[1U];
            b_0 = hi_0[1U];
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
            a_8 = lo_0[2U];
            b_1 = hi_0[2U];
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
            a_9 = lo_0[3U];
            b_2 = hi_0[3U];
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
            a_10 = lo_0[4U];
            b_3 = hi_0[4U];
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
            a_11 = lo_0[5U];
            b_4 = hi_0[5U];
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
            a_12 = lo_0[6U];
            b_5 = hi_0[6U];
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
            a_13 = lo_0[7U];
            b_6 = hi_0[7U];
        }
        else
        {
            a_13 = a_12;
            b_6 = b_5;
        }
        uint _S167 = 2U * _S54;
        uint _S168 = _S55 * 64U + _S167;
        (*(&kernelContext_0)->ss_0)[_S168] = a_13;
        (*(&kernelContext_0)->ss_0)[_S168 + 1U] = b_6;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S55 != 0U)
        {
            visible_0 = true;
        }
        else
        {
            visible_0 = _S54 >= _S9;
        }
        if(visible_0)
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
        uint h_1 = _S12 + _S172;
        uint i_2 = r_0 % (&kernelContext_0)->p_0->queries_0;
        uint row_0 = _S7 + h_1 * (&kernelContext_0)->p_0->queries_0 + i_2;
        float a_15;
        float b_8;
        if(((&kernelContext_0)->p_0->splits_1) > 1U)
        {
            uint pr_8 = split_0 * _S8 + row_0;
            uint at_0 = attnAccBase_0((&kernelContext_0)->p_0->splits_1, _S8) + pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S167;
            if(_S54 == 0U)
            {
                uint _S173 = 2U * pr_8;
                float device* _S174 = (&kernelContext_0)->partials_0+_S173;
                bool _S175 = _S15 == 0U;
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
        *((&kernelContext_0)->o_0+(row_0 * (&kernelContext_0)->p_0->headDim_0 + _S167)) = a_15;
        *((&kernelContext_0)->o_0+(row_0 * (&kernelContext_0)->p_0->headDim_0 + _S167 + 1U)) = b_8;
        r_0 = r_0 + 1U;
    }
    return;
}

