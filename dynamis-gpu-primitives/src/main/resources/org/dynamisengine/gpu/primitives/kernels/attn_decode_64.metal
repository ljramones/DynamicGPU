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

bool dg_is_nan_0(float x_0)
{
    uint _S5 = (as_type<uint>((x_0)));
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
    bool _S7;
    if(a_0 == 0.0)
    {
        _S7 = b_0 == 0.0;
    }
    else
    {
        _S7 = false;
    }
    if(_S7)
    {
        return (as_type<float>(((as_type<uint>((a_0))) & (as_type<uint>((b_0))))));
    }
    float _S8;
    if(a_0 > b_0)
    {
        _S8 = a_0;
    }
    else
    {
        _S8 = b_0;
    }
    return _S8;
}

float dg_op_max_0(float a_1, float b_1)
{
    return dg_max_0(a_1, b_1);
}

float dg_op_add_0(float a_2, float b_2)
{
    return a_2 + b_2;
}

uint attnAccBase_0(uint splits_0, uint rows_0)
{
    return 2U * splits_0 * rows_0;
}

uint qF32ToF16_0(float f_0)
{
    uint x_1 = (as_type<uint>((f_0)));
    uint sign_0 = (x_1 >> 16U) & 32768U;
    uint e_0 = (x_1 >> 23U) & 255U;
    uint mant_0 = x_1 & 8388607U;
    uint h_0;
    if(e_0 == 255U)
    {
        uint _S9 = sign_0 | 31744U;
        if(mant_0 != 0U)
        {
            h_0 = 512U | (mant_0 >> 13U);
        }
        else
        {
            h_0 = 0U;
        }
        return _S9 | h_0;
    }
    int exp_0 = int(e_0) - int(112);
    if(exp_0 >= int(31))
    {
        return sign_0 | 31744U;
    }
    bool _S10;
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
            _S10 = true;
        }
        else
        {
            if(rem_0 == halfway_0)
            {
                _S10 = (h_1 & 1U) != 0U;
            }
            else
            {
                _S10 = false;
            }
        }
        if(_S10)
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
        _S10 = true;
    }
    else
    {
        if(rem_1 == 4096U)
        {
            _S10 = (h_2 & 1U) != 0U;
        }
        else
        {
            _S10 = false;
        }
    }
    if(_S10)
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
    array<float, int(8)> threadgroup* rowM_0;
    array<float, int(8)> threadgroup* rowL_0;
    array<uint, int(256)> threadgroup* qs_0;
    array<float, int(1024)> threadgroup* ss_0;
    array<float, int(128)> threadgroup* red_0;
    array<float, int(8)> threadgroup* rowA_0;
};

[[kernel]] void attn_decode_64(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(7)]], uint device* table_1 [[buffer(5)]], uint device* lens_2 [[buffer(6)]], uint device* q_1 [[buffer(0)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partials_1 [[buffer(4)]], uint device* o_1 [[buffer(3)]])
{
    bool _S11;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->table_0 = table_1;
    (&kernelContext_0)->lens_1 = lens_2;
    (&kernelContext_0)->q_0 = q_1;
    (&kernelContext_0)->k_0 = k_1;
    (&kernelContext_0)->v_0 = v_1;
    (&kernelContext_0)->partials_0 = partials_1;
    (&kernelContext_0)->o_0 = o_1;
    threadgroup array<float, int(8)> rowM_1;
    (&kernelContext_0)->rowM_0 = &rowM_1;
    threadgroup array<float, int(8)> rowL_1;
    (&kernelContext_0)->rowL_0 = &rowL_1;
    threadgroup array<uint, int(256)> qs_1;
    (&kernelContext_0)->qs_0 = &qs_1;
    threadgroup array<float, int(1024)> ss_1;
    (&kernelContext_0)->ss_0 = &ss_1;
    threadgroup array<float, int(128)> red_1;
    (&kernelContext_0)->red_0 = &red_1;
    threadgroup array<float, int(8)> rowA_1;
    (&kernelContext_0)->rowA_0 = &rowA_1;
    uint t_0 = local_0.x;
    uint kvHead_0 = group_0.y;
    uint split_0 = group_0.z;
    uint seq_0 = group_0.x;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint _S12 = min(hpk_0 * (&kernelContext_0)->p_0->queries_0, ROW_TIER_0);
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
    uint _S13 = seq_0 * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0;
    uint _S14 = max((&kernelContext_0)->p_0->batch_0, 1U) * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0;
    uint _S15 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint _S16 = keys_1 - (&kernelContext_0)->p_0->queries_0;
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S17 = min(keys_1, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    uint _S18 = kvHead_0 * hpk_0;
    uint firstHead_1 = seq_0 * p_1->heads_0 + _S18;
    bool _S19 = USE_LENGTHS_0 != 0U;
    uint kEnd_0;
    if(_S19)
    {
        uint _S20 = attnKeyEnd_0((&kernelContext_0)->lens_1, (&kernelContext_0)->p_0->lengthGroup_1, firstHead_1, firstHead_1 + hpk_0 - 1U, _S17);
        kEnd_0 = _S20;
    }
    else
    {
        kEnd_0 = _S17;
    }
    uint _S21;
    if(kEnd_0 > kStart_0)
    {
        _S21 = kEnd_0 - kStart_0;
    }
    else
    {
        _S21 = 0U;
    }
    uint _S22 = 1024U / max(_S12, 1U);
    uint _S23 = max(128U, _S22 / 128U * 128U);
    thread array<uint, int(8)> limit_0;
    uint e_1;
    bool visible_0;
    if(_S19)
    {
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S24 = 0U % (&kernelContext_0)->p_0->queries_0;
            e_1 = _S24 + _S16 + 1U;
        }
        else
        {
            e_1 = keys_1;
        }
        limit_0[0U] = e_1;
        if(0U < _S12)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S25 = limit_0[0U];
            uint _S26 = firstHead_1 / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[0U] = min(_S25, *((&kernelContext_0)->lens_1+_S26));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S27 = 1U % (&kernelContext_0)->p_0->queries_0;
            e_1 = _S27 + _S16 + 1U;
        }
        else
        {
            e_1 = keys_1;
        }
        limit_0[1U] = e_1;
        if(1U < _S12)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S28 = limit_0[1U];
            uint _S29 = 1U / (&kernelContext_0)->p_0->queries_0;
            uint _S30 = (firstHead_1 + _S29) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[1U] = min(_S28, *((&kernelContext_0)->lens_1+_S30));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S31 = 2U % (&kernelContext_0)->p_0->queries_0;
            e_1 = _S31 + _S16 + 1U;
        }
        else
        {
            e_1 = keys_1;
        }
        limit_0[2U] = e_1;
        if(2U < _S12)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S32 = limit_0[2U];
            uint _S33 = 2U / (&kernelContext_0)->p_0->queries_0;
            uint _S34 = (firstHead_1 + _S33) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[2U] = min(_S32, *((&kernelContext_0)->lens_1+_S34));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S35 = 3U % (&kernelContext_0)->p_0->queries_0;
            e_1 = _S35 + _S16 + 1U;
        }
        else
        {
            e_1 = keys_1;
        }
        limit_0[3U] = e_1;
        if(3U < _S12)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S36 = limit_0[3U];
            uint _S37 = 3U / (&kernelContext_0)->p_0->queries_0;
            uint _S38 = (firstHead_1 + _S37) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[3U] = min(_S36, *((&kernelContext_0)->lens_1+_S38));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S39 = 4U % (&kernelContext_0)->p_0->queries_0;
            e_1 = _S39 + _S16 + 1U;
        }
        else
        {
            e_1 = keys_1;
        }
        limit_0[4U] = e_1;
        if(4U < _S12)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S40 = limit_0[4U];
            uint _S41 = 4U / (&kernelContext_0)->p_0->queries_0;
            uint _S42 = (firstHead_1 + _S41) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[4U] = min(_S40, *((&kernelContext_0)->lens_1+_S42));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S43 = 5U % (&kernelContext_0)->p_0->queries_0;
            e_1 = _S43 + _S16 + 1U;
        }
        else
        {
            e_1 = keys_1;
        }
        limit_0[5U] = e_1;
        if(5U < _S12)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S44 = limit_0[5U];
            uint _S45 = 5U / (&kernelContext_0)->p_0->queries_0;
            uint _S46 = (firstHead_1 + _S45) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[5U] = min(_S44, *((&kernelContext_0)->lens_1+_S46));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S47 = 6U % (&kernelContext_0)->p_0->queries_0;
            e_1 = _S47 + _S16 + 1U;
        }
        else
        {
            e_1 = keys_1;
        }
        limit_0[6U] = e_1;
        if(6U < _S12)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S48 = limit_0[6U];
            uint _S49 = 6U / (&kernelContext_0)->p_0->queries_0;
            uint _S50 = (firstHead_1 + _S49) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[6U] = min(_S48, *((&kernelContext_0)->lens_1+_S50));
        }
        if(((&kernelContext_0)->p_0->causal_0) != 0U)
        {
            uint _S51 = 7U % (&kernelContext_0)->p_0->queries_0;
            keys_1 = _S51 + _S16 + 1U;
        }
        limit_0[7U] = keys_1;
        if(7U < _S12)
        {
            visible_0 = ((&kernelContext_0)->p_0->lengthGroup_1) != 0U;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S52 = limit_0[7U];
            uint _S53 = 7U / (&kernelContext_0)->p_0->queries_0;
            uint _S54 = (firstHead_1 + _S53) / (&kernelContext_0)->p_0->lengthGroup_1;
            limit_0[7U] = min(_S52, *((&kernelContext_0)->lens_1+_S54));
        }
    }
    bool _S55 = t_0 < 8U;
    if(_S55)
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
        if(r_1 < _S12)
        {
            visible_0 = pr_0 < _S15;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S56 = r_1 / (&kernelContext_0)->p_0->queries_0;
            uint h_3 = _S18 + _S56;
            uint i_1 = r_1 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((_S13 + h_3 * (&kernelContext_0)->p_0->queries_0 + i_1) * _S15 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_1] = w_2;
        e_1 = e_1 + 128U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S57 = t_0 % 32U;
    uint _S58 = t_0 / 32U;
    thread array<float, int(8)> lo_1;
    thread array<float, int(8)> hi_1;
    float pj_0;
    float acc_0;
    float acc_1;
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
    w_2 = kStart_0;
    for(;;)
    {
        if(w_2 < kEnd_0)
        {
        }
        else
        {
            break;
        }
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
        int _S72;
        int _S73;
        int _S74;
        uint kw_0;
        uint kw_1;
        uint kw_2;
        uint kw_3;
        uint kw_4;
        uint kw_5;
        uint kw_6;
        uint p0_0;
        uint _S75 = min(_S23, kEnd_0 - w_2);
        uint j_0 = t_0;
        for(;;)
        {
            if(j_0 < _S75)
            {
            }
            else
            {
                break;
            }
            uint key_0 = w_2 + j_0;
            if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
            {
                keys_1 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * _S15;
            }
            else
            {
                uint _S76 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
                uint _S77 = (*((&kernelContext_0)->table_0+(r_0 + _S76)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                uint _S78 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
                keys_1 = (_S77 + _S78) * _S15;
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
                if(p0_0 < _S15)
                {
                    kw_6 = *((&kernelContext_0)->k_0+(keys_1 + p0_0));
                }
                else
                {
                    kw_6 = 0U;
                }
                float _S79 = attnLo_0(kw_6);
                float _S80 = attnHi_0(kw_6);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[p0_0]), _S80, fma(attnLo_0((*(&kernelContext_0)->qs_0)[p0_0]), _S79, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + p0_0]), _S80, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + p0_0]), _S79, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S80, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + p0_0]), _S79, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + p0_0]), _S80, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + p0_0]), _S79, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S80, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S79, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + p0_0]), _S80, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + p0_0]), _S79, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S80, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + p0_0]), _S79, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + p0_0]), _S80, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + p0_0]), _S79, s_0[7U]));
                }
                uint pr_1 = p0_0 + 1U;
                if(pr_1 < _S15)
                {
                    kw_5 = *((&kernelContext_0)->k_0+(keys_1 + pr_1));
                }
                else
                {
                    kw_5 = 0U;
                }
                float _S81 = attnLo_0(kw_5);
                float _S82 = attnHi_0(kw_5);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_1]), _S82, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_1]), _S81, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_1]), _S82, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_1]), _S81, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S82, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_1]), _S81, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_1]), _S82, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_1]), _S81, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S82, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S81, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_1]), _S82, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_1]), _S81, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S82, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_1]), _S81, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_1]), _S82, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_1]), _S81, s_0[7U]));
                }
                uint pr_2 = p0_0 + 2U;
                if(pr_2 < _S15)
                {
                    kw_4 = *((&kernelContext_0)->k_0+(keys_1 + pr_2));
                }
                else
                {
                    kw_4 = 0U;
                }
                float _S83 = attnLo_0(kw_4);
                float _S84 = attnHi_0(kw_4);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_2]), _S84, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_2]), _S83, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_2]), _S84, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_2]), _S83, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S84, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_2]), _S83, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_2]), _S84, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_2]), _S83, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S84, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S83, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_2]), _S84, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_2]), _S83, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S84, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_2]), _S83, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_2]), _S84, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_2]), _S83, s_0[7U]));
                }
                uint pr_3 = p0_0 + 3U;
                if(pr_3 < _S15)
                {
                    kw_3 = *((&kernelContext_0)->k_0+(keys_1 + pr_3));
                }
                else
                {
                    kw_3 = 0U;
                }
                float _S85 = attnLo_0(kw_3);
                float _S86 = attnHi_0(kw_3);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_3]), _S86, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_3]), _S85, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_3]), _S86, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_3]), _S85, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S86, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_3]), _S85, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_3]), _S86, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_3]), _S85, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S86, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S85, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_3]), _S86, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_3]), _S85, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S86, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_3]), _S85, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_3]), _S86, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_3]), _S85, s_0[7U]));
                }
                uint pr_4 = p0_0 + 4U;
                if(pr_4 < _S15)
                {
                    kw_2 = *((&kernelContext_0)->k_0+(keys_1 + pr_4));
                }
                else
                {
                    kw_2 = 0U;
                }
                float _S87 = attnLo_0(kw_2);
                float _S88 = attnHi_0(kw_2);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_4]), _S88, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_4]), _S87, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_4]), _S88, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_4]), _S87, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S88, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_4]), _S87, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_4]), _S88, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_4]), _S87, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S88, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S87, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_4]), _S88, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_4]), _S87, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S88, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_4]), _S87, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_4]), _S88, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_4]), _S87, s_0[7U]));
                }
                uint pr_5 = p0_0 + 5U;
                if(pr_5 < _S15)
                {
                    kw_1 = *((&kernelContext_0)->k_0+(keys_1 + pr_5));
                }
                else
                {
                    kw_1 = 0U;
                }
                float _S89 = attnLo_0(kw_1);
                float _S90 = attnHi_0(kw_1);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_5]), _S90, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_5]), _S89, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_5]), _S90, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_5]), _S89, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S90, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_5]), _S89, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_5]), _S90, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_5]), _S89, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S90, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S89, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_5]), _S90, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_5]), _S89, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S90, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_5]), _S89, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_5]), _S90, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_5]), _S89, s_0[7U]));
                }
                uint pr_6 = p0_0 + 6U;
                if(pr_6 < _S15)
                {
                    kw_0 = *((&kernelContext_0)->k_0+(keys_1 + pr_6));
                }
                else
                {
                    kw_0 = 0U;
                }
                float _S91 = attnLo_0(kw_0);
                float _S92 = attnHi_0(kw_0);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_6]), _S92, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_6]), _S91, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_6]), _S92, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_6]), _S91, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S92, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_6]), _S91, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_6]), _S92, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_6]), _S91, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S92, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S91, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_6]), _S92, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_6]), _S91, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S92, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_6]), _S91, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_6]), _S92, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_6]), _S91, s_0[7U]));
                }
                uint pr_7 = p0_0 + 7U;
                uint kw_7;
                if(pr_7 < _S15)
                {
                    kw_7 = *((&kernelContext_0)->k_0+(keys_1 + pr_7));
                }
                else
                {
                    kw_7 = 0U;
                }
                float _S93 = attnLo_0(kw_7);
                float _S94 = attnHi_0(kw_7);
                if(0U < ROW_TIER_0)
                {
                    s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_7]), _S94, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_7]), _S93, s_0[0U]));
                }
                if(1U < ROW_TIER_0)
                {
                    s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[32U + pr_7]), _S94, fma(attnLo_0((*(&kernelContext_0)->qs_0)[32U + pr_7]), _S93, s_0[1U]));
                }
                if(2U < ROW_TIER_0)
                {
                    s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S94, fma(attnLo_0((*(&kernelContext_0)->qs_0)[64U + pr_7]), _S93, s_0[2U]));
                }
                if(3U < ROW_TIER_0)
                {
                    s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[96U + pr_7]), _S94, fma(attnLo_0((*(&kernelContext_0)->qs_0)[96U + pr_7]), _S93, s_0[3U]));
                }
                if(4U < ROW_TIER_0)
                {
                    s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S94, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S93, s_0[4U]));
                }
                if(5U < ROW_TIER_0)
                {
                    s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[160U + pr_7]), _S94, fma(attnLo_0((*(&kernelContext_0)->qs_0)[160U + pr_7]), _S93, s_0[5U]));
                }
                if(6U < ROW_TIER_0)
                {
                    s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S94, fma(attnLo_0((*(&kernelContext_0)->qs_0)[192U + pr_7]), _S93, s_0[6U]));
                }
                if(7U < ROW_TIER_0)
                {
                    s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[224U + pr_7]), _S94, fma(attnLo_0((*(&kernelContext_0)->qs_0)[224U + pr_7]), _S93, s_0[7U]));
                }
                p0_0 = p0_0 + 8U;
            }
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        if(0U >= _S12)
                        {
                            _S74 = int(2);
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
                                uint _S95 = 0U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S95 + _S16);
                            }
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
                        _S74 = int(4);
                        break;
                    }
                    if(_S74 != int(4))
                    {
                        _S73 = _S74;
                        break;
                    }
                    _S73 = int(3);
                    break;
                }
                if(_S73 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(1U >= _S12)
                        {
                            _S72 = int(2);
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
                                uint _S96 = 1U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S96 + _S16);
                            }
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[1U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[_S23 + j_0] = acc_1;
                        _S72 = int(4);
                        break;
                    }
                    if(_S72 != int(4))
                    {
                        _S71 = _S72;
                        break;
                    }
                    _S71 = int(3);
                    break;
                }
                if(_S71 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(2U >= _S12)
                        {
                            _S70 = int(2);
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
                                uint _S97 = 2U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S97 + _S16);
                            }
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[2U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[2U * _S23 + j_0] = acc_1;
                        _S70 = int(4);
                        break;
                    }
                    if(_S70 != int(4))
                    {
                        _S69 = _S70;
                        break;
                    }
                    _S69 = int(3);
                    break;
                }
                if(_S69 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(3U >= _S12)
                        {
                            _S68 = int(2);
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
                                uint _S98 = 3U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S98 + _S16);
                            }
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[3U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[3U * _S23 + j_0] = acc_1;
                        _S68 = int(4);
                        break;
                    }
                    if(_S68 != int(4))
                    {
                        _S67 = _S68;
                        break;
                    }
                    _S67 = int(3);
                    break;
                }
                if(_S67 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(4U >= _S12)
                        {
                            _S66 = int(2);
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
                                uint _S99 = 4U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S99 + _S16);
                            }
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[4U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[4U * _S23 + j_0] = acc_1;
                        _S66 = int(4);
                        break;
                    }
                    if(_S66 != int(4))
                    {
                        _S65 = _S66;
                        break;
                    }
                    _S65 = int(3);
                    break;
                }
                if(_S65 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(5U >= _S12)
                        {
                            _S64 = int(2);
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
                                uint _S100 = 5U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S100 + _S16);
                            }
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[5U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[5U * _S23 + j_0] = acc_1;
                        _S64 = int(4);
                        break;
                    }
                    if(_S64 != int(4))
                    {
                        _S63 = _S64;
                        break;
                    }
                    _S63 = int(3);
                    break;
                }
                if(_S63 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(6U >= _S12)
                        {
                            _S62 = int(2);
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
                                uint _S101 = 6U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S101 + _S16);
                            }
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[6U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[6U * _S23 + j_0] = acc_1;
                        _S62 = int(4);
                        break;
                    }
                    if(_S62 != int(4))
                    {
                        _S61 = _S62;
                        break;
                    }
                    _S61 = int(3);
                    break;
                }
                if(_S61 != int(3))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(7U >= _S12)
                        {
                            _S60 = int(2);
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
                                uint _S102 = 7U % (&kernelContext_0)->p_0->queries_0;
                                visible_0 = key_0 <= (_S102 + _S16);
                            }
                        }
                        if(visible_0)
                        {
                            acc_1 = s_0[7U] * (&kernelContext_0)->p_0->scale_0;
                        }
                        else
                        {
                            acc_1 = dg_negative_infinity_0();
                        }
                        (*(&kernelContext_0)->ss_0)[7U * _S23 + j_0] = acc_1;
                        _S60 = int(4);
                        break;
                    }
                    if(_S60 != int(4))
                    {
                        _S59 = _S60;
                        break;
                    }
                    _S59 = int(3);
                    break;
                }
                if(_S59 != int(3))
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
            if(p0_0 < _S12)
            {
            }
            else
            {
                break;
            }
            float _S103 = dg_negative_infinity_0();
            kw_6 = t_0;
            acc_1 = _S103;
            for(;;)
            {
                if(kw_6 < _S75)
                {
                }
                else
                {
                    break;
                }
                float _S104 = max(acc_1, (*(&kernelContext_0)->ss_0)[p0_0 * _S23 + kw_6]);
                kw_6 = kw_6 + 128U;
                acc_1 = _S104;
            }
            (*(&kernelContext_0)->red_0)[t_0] = acc_1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S105 = t_0 < 64U;
            if(_S105)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S106 = t_0 < 32U;
            if(_S106)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S107 = t_0 < 16U;
            if(_S107)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S55)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S108 = t_0 < 4U;
            if(_S108)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S109 = t_0 < 2U;
            if(_S109)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S110 = t_0 < 1U;
            if(_S110)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool _S111 = t_0 == 0U;
            if(_S111)
            {
                float mOld_0 = (*(&kernelContext_0)->rowM_0)[p0_0];
                float _S112 = max((*(&kernelContext_0)->rowM_0)[p0_0], (*(&kernelContext_0)->red_0)[int(0)]);
                if(((*(&kernelContext_0)->rowM_0)[p0_0]) == _S103)
                {
                    acc_0 = 0.0;
                }
                else
                {
                    acc_0 = exp(mOld_0 - _S112);
                }
                (*(&kernelContext_0)->rowA_0)[p0_0] = acc_0;
                (*(&kernelContext_0)->rowM_0)[p0_0] = _S112;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float _S113 = (*(&kernelContext_0)->rowM_0)[p0_0];
            kw_5 = t_0;
            acc_0 = 0.0;
            for(;;)
            {
                if(kw_5 < _S75)
                {
                }
                else
                {
                    break;
                }
                float sv_0 = (*(&kernelContext_0)->ss_0)[p0_0 * _S23 + kw_5];
                if(((*(&kernelContext_0)->ss_0)[p0_0 * _S23 + kw_5]) == _S103)
                {
                    pj_0 = 0.0;
                }
                else
                {
                    pj_0 = exp(sv_0 - _S113);
                }
                (*(&kernelContext_0)->ss_0)[p0_0 * _S23 + kw_5] = pj_0;
                float _S114 = acc_0 + pj_0;
                kw_5 = kw_5 + 128U;
                acc_0 = _S114;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (*(&kernelContext_0)->red_0)[t_0] = acc_0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S105)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S106)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S107)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S55)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S108)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S109)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S110)
            {
                (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(_S111)
            {
                (*(&kernelContext_0)->rowL_0)[p0_0] = (*(&kernelContext_0)->rowL_0)[p0_0] * (*(&kernelContext_0)->rowA_0)[p0_0] + (*(&kernelContext_0)->red_0)[int(0)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            p0_0 = p0_0 + 1U;
        }
        if(0U < _S12)
        {
            float a_3 = (*(&kernelContext_0)->rowA_0)[0U];
            lo_1[0U] = lo_1[0U] * (*(&kernelContext_0)->rowA_0)[0U];
            hi_1[0U] = hi_1[0U] * a_3;
        }
        if(1U < _S12)
        {
            float a_4 = (*(&kernelContext_0)->rowA_0)[1U];
            lo_1[1U] = lo_1[1U] * (*(&kernelContext_0)->rowA_0)[1U];
            hi_1[1U] = hi_1[1U] * a_4;
        }
        if(2U < _S12)
        {
            float a_5 = (*(&kernelContext_0)->rowA_0)[2U];
            lo_1[2U] = lo_1[2U] * (*(&kernelContext_0)->rowA_0)[2U];
            hi_1[2U] = hi_1[2U] * a_5;
        }
        if(3U < _S12)
        {
            float a_6 = (*(&kernelContext_0)->rowA_0)[3U];
            lo_1[3U] = lo_1[3U] * (*(&kernelContext_0)->rowA_0)[3U];
            hi_1[3U] = hi_1[3U] * a_6;
        }
        if(4U < _S12)
        {
            float a_7 = (*(&kernelContext_0)->rowA_0)[4U];
            lo_1[4U] = lo_1[4U] * (*(&kernelContext_0)->rowA_0)[4U];
            hi_1[4U] = hi_1[4U] * a_7;
        }
        if(5U < _S12)
        {
            float a_8 = (*(&kernelContext_0)->rowA_0)[5U];
            lo_1[5U] = lo_1[5U] * (*(&kernelContext_0)->rowA_0)[5U];
            hi_1[5U] = hi_1[5U] * a_8;
        }
        if(6U < _S12)
        {
            float a_9 = (*(&kernelContext_0)->rowA_0)[6U];
            lo_1[6U] = lo_1[6U] * (*(&kernelContext_0)->rowA_0)[6U];
            hi_1[6U] = hi_1[6U] * a_9;
        }
        if(7U < _S12)
        {
            float a_10 = (*(&kernelContext_0)->rowA_0)[7U];
            lo_1[7U] = lo_1[7U] * (*(&kernelContext_0)->rowA_0)[7U];
            hi_1[7U] = hi_1[7U] * a_10;
        }
        if(_S57 < _S15)
        {
            kw_6 = _S58;
            for(;;)
            {
                bool _S115 = kw_6 < _S75;
                if(_S115)
                {
                }
                else
                {
                    break;
                }
                thread array<uint, int(8)> vw_0;
                if(_S115)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        keys_1 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + kw_6)) * _S15 + _S57;
                    }
                    else
                    {
                        uint _S116 = w_2 + kw_6;
                        uint _S117 = _S116 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S118 = (*((&kernelContext_0)->table_0+(r_0 + _S117)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S119 = _S116 % (&kernelContext_0)->p_0->pageSize_0;
                        keys_1 = (_S118 + _S119) * _S15 + _S57;
                    }
                    keys_1 = *((&kernelContext_0)->v_0+keys_1);
                }
                else
                {
                    keys_1 = 0U;
                }
                vw_0[0U] = keys_1;
                uint j_1 = kw_6 + 4U;
                if(j_1 < _S75)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        e_1 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_1)) * _S15 + _S57;
                    }
                    else
                    {
                        uint _S120 = w_2 + j_1;
                        uint _S121 = _S120 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S122 = (*((&kernelContext_0)->table_0+(r_0 + _S121)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S123 = _S120 % (&kernelContext_0)->p_0->pageSize_0;
                        e_1 = (_S122 + _S123) * _S15 + _S57;
                    }
                    e_1 = *((&kernelContext_0)->v_0+e_1);
                }
                else
                {
                    e_1 = 0U;
                }
                vw_0[1U] = e_1;
                uint j_2 = kw_6 + 8U;
                if(j_2 < _S75)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_5 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_2)) * _S15 + _S57;
                    }
                    else
                    {
                        uint _S124 = w_2 + j_2;
                        uint _S125 = _S124 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S126 = (*((&kernelContext_0)->table_0+(r_0 + _S125)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S127 = _S124 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_5 = (_S126 + _S127) * _S15 + _S57;
                    }
                    kw_5 = *((&kernelContext_0)->v_0+kw_5);
                }
                else
                {
                    kw_5 = 0U;
                }
                vw_0[2U] = kw_5;
                uint j_3 = kw_6 + 12U;
                if(j_3 < _S75)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_4 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_3)) * _S15 + _S57;
                    }
                    else
                    {
                        uint _S128 = w_2 + j_3;
                        uint _S129 = _S128 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S130 = (*((&kernelContext_0)->table_0+(r_0 + _S129)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S131 = _S128 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_4 = (_S130 + _S131) * _S15 + _S57;
                    }
                    kw_4 = *((&kernelContext_0)->v_0+kw_4);
                }
                else
                {
                    kw_4 = 0U;
                }
                vw_0[3U] = kw_4;
                uint j_4 = kw_6 + 16U;
                if(j_4 < _S75)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_3 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_4)) * _S15 + _S57;
                    }
                    else
                    {
                        uint _S132 = w_2 + j_4;
                        uint _S133 = _S132 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S134 = (*((&kernelContext_0)->table_0+(r_0 + _S133)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S135 = _S132 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_3 = (_S134 + _S135) * _S15 + _S57;
                    }
                    kw_3 = *((&kernelContext_0)->v_0+kw_3);
                }
                else
                {
                    kw_3 = 0U;
                }
                vw_0[4U] = kw_3;
                uint j_5 = kw_6 + 20U;
                if(j_5 < _S75)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_2 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_5)) * _S15 + _S57;
                    }
                    else
                    {
                        uint _S136 = w_2 + j_5;
                        uint _S137 = _S136 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S138 = (*((&kernelContext_0)->table_0+(r_0 + _S137)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S139 = _S136 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_2 = (_S138 + _S139) * _S15 + _S57;
                    }
                    kw_2 = *((&kernelContext_0)->v_0+kw_2);
                }
                else
                {
                    kw_2 = 0U;
                }
                vw_0[5U] = kw_2;
                uint j_6 = kw_6 + 24U;
                if(j_6 < _S75)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_1 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_6)) * _S15 + _S57;
                    }
                    else
                    {
                        uint _S140 = w_2 + j_6;
                        uint _S141 = _S140 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S142 = (*((&kernelContext_0)->table_0+(r_0 + _S141)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S143 = _S140 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_1 = (_S142 + _S143) * _S15 + _S57;
                    }
                    kw_1 = *((&kernelContext_0)->v_0+kw_1);
                }
                else
                {
                    kw_1 = 0U;
                }
                vw_0[6U] = kw_1;
                uint j_7 = kw_6 + 28U;
                if(j_7 < _S75)
                {
                    if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                    {
                        kw_0 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + (w_2 + j_7)) * _S15 + _S57;
                    }
                    else
                    {
                        uint _S144 = w_2 + j_7;
                        uint _S145 = _S144 / (&kernelContext_0)->p_0->pageSize_0;
                        uint _S146 = (*((&kernelContext_0)->table_0+(r_0 + _S145)) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                        uint _S147 = _S144 % (&kernelContext_0)->p_0->pageSize_0;
                        kw_0 = (_S146 + _S147) * _S15 + _S57;
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
                            if(kw_6 >= _S75)
                            {
                                _S74 = int(2);
                                visible_0 = false;
                                break;
                            }
                            float _S148 = attnLo_0(vw_0[0U]);
                            float _S149 = attnHi_0(vw_0[0U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        bool _S150 = 0U >= _S12;
                                        _S11 = _S150;
                                        if(_S150)
                                        {
                                            _S74 = int(5);
                                            break;
                                        }
                                        float pj_1 = (*(&kernelContext_0)->ss_0)[kw_6];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[kw_6], _S148, lo_1[0U]);
                                        hi_1[0U] = fma(pj_1, _S149, hi_1[0U]);
                                        _S74 = int(7);
                                        break;
                                    }
                                    if(_S74 != int(7))
                                    {
                                        _S73 = _S74;
                                        break;
                                    }
                                    _S73 = int(6);
                                    break;
                                }
                                if(_S73 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S12)
                                        {
                                            _S72 = int(5);
                                            break;
                                        }
                                        float pj_2 = (*(&kernelContext_0)->ss_0)[_S23 + kw_6];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S23 + kw_6], _S148, lo_1[1U]);
                                        hi_1[1U] = fma(pj_2, _S149, hi_1[1U]);
                                        _S72 = int(7);
                                        break;
                                    }
                                    if(_S72 != int(7))
                                    {
                                        _S71 = _S72;
                                        break;
                                    }
                                    _S71 = int(6);
                                    break;
                                }
                                if(_S71 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S12)
                                        {
                                            _S70 = int(5);
                                            break;
                                        }
                                        float pj_3 = (*(&kernelContext_0)->ss_0)[2U * _S23 + kw_6];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S23 + kw_6], _S148, lo_1[2U]);
                                        hi_1[2U] = fma(pj_3, _S149, hi_1[2U]);
                                        _S70 = int(7);
                                        break;
                                    }
                                    if(_S70 != int(7))
                                    {
                                        _S69 = _S70;
                                        break;
                                    }
                                    _S69 = int(6);
                                    break;
                                }
                                if(_S69 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S12)
                                        {
                                            _S68 = int(5);
                                            break;
                                        }
                                        float pj_4 = (*(&kernelContext_0)->ss_0)[3U * _S23 + kw_6];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S23 + kw_6], _S148, lo_1[3U]);
                                        hi_1[3U] = fma(pj_4, _S149, hi_1[3U]);
                                        _S68 = int(7);
                                        break;
                                    }
                                    if(_S68 != int(7))
                                    {
                                        _S67 = _S68;
                                        break;
                                    }
                                    _S67 = int(6);
                                    break;
                                }
                                if(_S67 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S12)
                                        {
                                            _S66 = int(5);
                                            break;
                                        }
                                        float pj_5 = (*(&kernelContext_0)->ss_0)[4U * _S23 + kw_6];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S23 + kw_6], _S148, lo_1[4U]);
                                        hi_1[4U] = fma(pj_5, _S149, hi_1[4U]);
                                        _S66 = int(7);
                                        break;
                                    }
                                    if(_S66 != int(7))
                                    {
                                        _S65 = _S66;
                                        break;
                                    }
                                    _S65 = int(6);
                                    break;
                                }
                                if(_S65 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S12)
                                        {
                                            _S64 = int(5);
                                            break;
                                        }
                                        float pj_6 = (*(&kernelContext_0)->ss_0)[5U * _S23 + kw_6];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S23 + kw_6], _S148, lo_1[5U]);
                                        hi_1[5U] = fma(pj_6, _S149, hi_1[5U]);
                                        _S64 = int(7);
                                        break;
                                    }
                                    if(_S64 != int(7))
                                    {
                                        _S63 = _S64;
                                        break;
                                    }
                                    _S63 = int(6);
                                    break;
                                }
                                if(_S63 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S12)
                                        {
                                            _S62 = int(5);
                                            break;
                                        }
                                        float pj_7 = (*(&kernelContext_0)->ss_0)[6U * _S23 + kw_6];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S23 + kw_6], _S148, lo_1[6U]);
                                        hi_1[6U] = fma(pj_7, _S149, hi_1[6U]);
                                        _S62 = int(7);
                                        break;
                                    }
                                    if(_S62 != int(7))
                                    {
                                        _S61 = _S62;
                                        break;
                                    }
                                    _S61 = int(6);
                                    break;
                                }
                                if(_S61 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S12)
                                        {
                                            _S60 = int(5);
                                            break;
                                        }
                                        float pj_8 = (*(&kernelContext_0)->ss_0)[7U * _S23 + kw_6];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S23 + kw_6], _S148, lo_1[7U]);
                                        hi_1[7U] = fma(pj_8, _S149, hi_1[7U]);
                                        _S60 = int(7);
                                        break;
                                    }
                                    if(_S60 != int(7))
                                    {
                                        _S59 = _S60;
                                        break;
                                    }
                                    _S59 = int(6);
                                    break;
                                }
                                if(_S59 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S74 = int(4);
                            visible_0 = _S11;
                            break;
                        }
                        if(_S74 != int(4))
                        {
                            _S73 = _S74;
                            break;
                        }
                        _S73 = int(3);
                        break;
                    }
                    if(_S73 != int(3))
                    {
                        break;
                    }
                    int _S151;
                    int _S152;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_1 >= _S75)
                            {
                                _S72 = int(2);
                                break;
                            }
                            float _S153 = attnLo_0(vw_0[1U]);
                            float _S154 = attnHi_0(vw_0[1U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S72 = int(5);
                                            break;
                                        }
                                        float pj_9 = (*(&kernelContext_0)->ss_0)[j_1];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_1], _S153, lo_1[0U]);
                                        hi_1[0U] = fma(pj_9, _S154, hi_1[0U]);
                                        _S72 = int(7);
                                        break;
                                    }
                                    if(_S72 != int(7))
                                    {
                                        _S71 = _S72;
                                        break;
                                    }
                                    _S71 = int(6);
                                    break;
                                }
                                if(_S71 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S12)
                                        {
                                            _S70 = int(5);
                                            break;
                                        }
                                        float pj_10 = (*(&kernelContext_0)->ss_0)[_S23 + j_1];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S23 + j_1], _S153, lo_1[1U]);
                                        hi_1[1U] = fma(pj_10, _S154, hi_1[1U]);
                                        _S70 = int(7);
                                        break;
                                    }
                                    if(_S70 != int(7))
                                    {
                                        _S69 = _S70;
                                        break;
                                    }
                                    _S69 = int(6);
                                    break;
                                }
                                if(_S69 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S12)
                                        {
                                            _S68 = int(5);
                                            break;
                                        }
                                        float pj_11 = (*(&kernelContext_0)->ss_0)[2U * _S23 + j_1];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S23 + j_1], _S153, lo_1[2U]);
                                        hi_1[2U] = fma(pj_11, _S154, hi_1[2U]);
                                        _S68 = int(7);
                                        break;
                                    }
                                    if(_S68 != int(7))
                                    {
                                        _S67 = _S68;
                                        break;
                                    }
                                    _S67 = int(6);
                                    break;
                                }
                                if(_S67 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S12)
                                        {
                                            _S66 = int(5);
                                            break;
                                        }
                                        float pj_12 = (*(&kernelContext_0)->ss_0)[3U * _S23 + j_1];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S23 + j_1], _S153, lo_1[3U]);
                                        hi_1[3U] = fma(pj_12, _S154, hi_1[3U]);
                                        _S66 = int(7);
                                        break;
                                    }
                                    if(_S66 != int(7))
                                    {
                                        _S65 = _S66;
                                        break;
                                    }
                                    _S65 = int(6);
                                    break;
                                }
                                if(_S65 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S12)
                                        {
                                            _S64 = int(5);
                                            break;
                                        }
                                        float pj_13 = (*(&kernelContext_0)->ss_0)[4U * _S23 + j_1];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S23 + j_1], _S153, lo_1[4U]);
                                        hi_1[4U] = fma(pj_13, _S154, hi_1[4U]);
                                        _S64 = int(7);
                                        break;
                                    }
                                    if(_S64 != int(7))
                                    {
                                        _S63 = _S64;
                                        break;
                                    }
                                    _S63 = int(6);
                                    break;
                                }
                                if(_S63 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S12)
                                        {
                                            _S62 = int(5);
                                            break;
                                        }
                                        float pj_14 = (*(&kernelContext_0)->ss_0)[5U * _S23 + j_1];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S23 + j_1], _S153, lo_1[5U]);
                                        hi_1[5U] = fma(pj_14, _S154, hi_1[5U]);
                                        _S62 = int(7);
                                        break;
                                    }
                                    if(_S62 != int(7))
                                    {
                                        _S61 = _S62;
                                        break;
                                    }
                                    _S61 = int(6);
                                    break;
                                }
                                if(_S61 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S12)
                                        {
                                            _S60 = int(5);
                                            break;
                                        }
                                        float pj_15 = (*(&kernelContext_0)->ss_0)[6U * _S23 + j_1];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S23 + j_1], _S153, lo_1[6U]);
                                        hi_1[6U] = fma(pj_15, _S154, hi_1[6U]);
                                        _S60 = int(7);
                                        break;
                                    }
                                    if(_S60 != int(7))
                                    {
                                        _S59 = _S60;
                                        break;
                                    }
                                    _S59 = int(6);
                                    break;
                                }
                                if(_S59 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S12)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_16 = (*(&kernelContext_0)->ss_0)[7U * _S23 + j_1];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S23 + j_1], _S153, lo_1[7U]);
                                        hi_1[7U] = fma(pj_16, _S154, hi_1[7U]);
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
                            _S72 = int(4);
                            break;
                        }
                        if(_S72 != int(4))
                        {
                            _S71 = _S72;
                            break;
                        }
                        _S71 = int(3);
                        break;
                    }
                    if(_S71 != int(3))
                    {
                        break;
                    }
                    int _S155;
                    int _S156;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_2 >= _S75)
                            {
                                _S70 = int(2);
                                break;
                            }
                            float _S157 = attnLo_0(vw_0[2U]);
                            float _S158 = attnHi_0(vw_0[2U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S70 = int(5);
                                            break;
                                        }
                                        float pj_17 = (*(&kernelContext_0)->ss_0)[j_2];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_2], _S157, lo_1[0U]);
                                        hi_1[0U] = fma(pj_17, _S158, hi_1[0U]);
                                        _S70 = int(7);
                                        break;
                                    }
                                    if(_S70 != int(7))
                                    {
                                        _S69 = _S70;
                                        break;
                                    }
                                    _S69 = int(6);
                                    break;
                                }
                                if(_S69 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S12)
                                        {
                                            _S68 = int(5);
                                            break;
                                        }
                                        float pj_18 = (*(&kernelContext_0)->ss_0)[_S23 + j_2];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S23 + j_2], _S157, lo_1[1U]);
                                        hi_1[1U] = fma(pj_18, _S158, hi_1[1U]);
                                        _S68 = int(7);
                                        break;
                                    }
                                    if(_S68 != int(7))
                                    {
                                        _S67 = _S68;
                                        break;
                                    }
                                    _S67 = int(6);
                                    break;
                                }
                                if(_S67 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S12)
                                        {
                                            _S66 = int(5);
                                            break;
                                        }
                                        float pj_19 = (*(&kernelContext_0)->ss_0)[2U * _S23 + j_2];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S23 + j_2], _S157, lo_1[2U]);
                                        hi_1[2U] = fma(pj_19, _S158, hi_1[2U]);
                                        _S66 = int(7);
                                        break;
                                    }
                                    if(_S66 != int(7))
                                    {
                                        _S65 = _S66;
                                        break;
                                    }
                                    _S65 = int(6);
                                    break;
                                }
                                if(_S65 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S12)
                                        {
                                            _S64 = int(5);
                                            break;
                                        }
                                        float pj_20 = (*(&kernelContext_0)->ss_0)[3U * _S23 + j_2];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S23 + j_2], _S157, lo_1[3U]);
                                        hi_1[3U] = fma(pj_20, _S158, hi_1[3U]);
                                        _S64 = int(7);
                                        break;
                                    }
                                    if(_S64 != int(7))
                                    {
                                        _S63 = _S64;
                                        break;
                                    }
                                    _S63 = int(6);
                                    break;
                                }
                                if(_S63 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S12)
                                        {
                                            _S62 = int(5);
                                            break;
                                        }
                                        float pj_21 = (*(&kernelContext_0)->ss_0)[4U * _S23 + j_2];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S23 + j_2], _S157, lo_1[4U]);
                                        hi_1[4U] = fma(pj_21, _S158, hi_1[4U]);
                                        _S62 = int(7);
                                        break;
                                    }
                                    if(_S62 != int(7))
                                    {
                                        _S61 = _S62;
                                        break;
                                    }
                                    _S61 = int(6);
                                    break;
                                }
                                if(_S61 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S12)
                                        {
                                            _S60 = int(5);
                                            break;
                                        }
                                        float pj_22 = (*(&kernelContext_0)->ss_0)[5U * _S23 + j_2];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S23 + j_2], _S157, lo_1[5U]);
                                        hi_1[5U] = fma(pj_22, _S158, hi_1[5U]);
                                        _S60 = int(7);
                                        break;
                                    }
                                    if(_S60 != int(7))
                                    {
                                        _S59 = _S60;
                                        break;
                                    }
                                    _S59 = int(6);
                                    break;
                                }
                                if(_S59 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S12)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_23 = (*(&kernelContext_0)->ss_0)[6U * _S23 + j_2];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S23 + j_2], _S157, lo_1[6U]);
                                        hi_1[6U] = fma(pj_23, _S158, hi_1[6U]);
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
                                        if(7U >= _S12)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_24 = (*(&kernelContext_0)->ss_0)[7U * _S23 + j_2];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S23 + j_2], _S157, lo_1[7U]);
                                        hi_1[7U] = fma(pj_24, _S158, hi_1[7U]);
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
                            _S70 = int(4);
                            break;
                        }
                        if(_S70 != int(4))
                        {
                            _S69 = _S70;
                            break;
                        }
                        _S69 = int(3);
                        break;
                    }
                    if(_S69 != int(3))
                    {
                        break;
                    }
                    int _S159;
                    int _S160;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_3 >= _S75)
                            {
                                _S68 = int(2);
                                break;
                            }
                            float _S161 = attnLo_0(vw_0[3U]);
                            float _S162 = attnHi_0(vw_0[3U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S68 = int(5);
                                            break;
                                        }
                                        float pj_25 = (*(&kernelContext_0)->ss_0)[j_3];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_3], _S161, lo_1[0U]);
                                        hi_1[0U] = fma(pj_25, _S162, hi_1[0U]);
                                        _S68 = int(7);
                                        break;
                                    }
                                    if(_S68 != int(7))
                                    {
                                        _S67 = _S68;
                                        break;
                                    }
                                    _S67 = int(6);
                                    break;
                                }
                                if(_S67 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S12)
                                        {
                                            _S66 = int(5);
                                            break;
                                        }
                                        float pj_26 = (*(&kernelContext_0)->ss_0)[_S23 + j_3];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S23 + j_3], _S161, lo_1[1U]);
                                        hi_1[1U] = fma(pj_26, _S162, hi_1[1U]);
                                        _S66 = int(7);
                                        break;
                                    }
                                    if(_S66 != int(7))
                                    {
                                        _S65 = _S66;
                                        break;
                                    }
                                    _S65 = int(6);
                                    break;
                                }
                                if(_S65 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S12)
                                        {
                                            _S64 = int(5);
                                            break;
                                        }
                                        float pj_27 = (*(&kernelContext_0)->ss_0)[2U * _S23 + j_3];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S23 + j_3], _S161, lo_1[2U]);
                                        hi_1[2U] = fma(pj_27, _S162, hi_1[2U]);
                                        _S64 = int(7);
                                        break;
                                    }
                                    if(_S64 != int(7))
                                    {
                                        _S63 = _S64;
                                        break;
                                    }
                                    _S63 = int(6);
                                    break;
                                }
                                if(_S63 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S12)
                                        {
                                            _S62 = int(5);
                                            break;
                                        }
                                        float pj_28 = (*(&kernelContext_0)->ss_0)[3U * _S23 + j_3];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S23 + j_3], _S161, lo_1[3U]);
                                        hi_1[3U] = fma(pj_28, _S162, hi_1[3U]);
                                        _S62 = int(7);
                                        break;
                                    }
                                    if(_S62 != int(7))
                                    {
                                        _S61 = _S62;
                                        break;
                                    }
                                    _S61 = int(6);
                                    break;
                                }
                                if(_S61 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S12)
                                        {
                                            _S60 = int(5);
                                            break;
                                        }
                                        float pj_29 = (*(&kernelContext_0)->ss_0)[4U * _S23 + j_3];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S23 + j_3], _S161, lo_1[4U]);
                                        hi_1[4U] = fma(pj_29, _S162, hi_1[4U]);
                                        _S60 = int(7);
                                        break;
                                    }
                                    if(_S60 != int(7))
                                    {
                                        _S59 = _S60;
                                        break;
                                    }
                                    _S59 = int(6);
                                    break;
                                }
                                if(_S59 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S12)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_30 = (*(&kernelContext_0)->ss_0)[5U * _S23 + j_3];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S23 + j_3], _S161, lo_1[5U]);
                                        hi_1[5U] = fma(pj_30, _S162, hi_1[5U]);
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
                                        if(6U >= _S12)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_31 = (*(&kernelContext_0)->ss_0)[6U * _S23 + j_3];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S23 + j_3], _S161, lo_1[6U]);
                                        hi_1[6U] = fma(pj_31, _S162, hi_1[6U]);
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
                                        if(7U >= _S12)
                                        {
                                            _S160 = int(5);
                                            break;
                                        }
                                        float pj_32 = (*(&kernelContext_0)->ss_0)[7U * _S23 + j_3];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S23 + j_3], _S161, lo_1[7U]);
                                        hi_1[7U] = fma(pj_32, _S162, hi_1[7U]);
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
                            _S68 = int(4);
                            break;
                        }
                        if(_S68 != int(4))
                        {
                            _S67 = _S68;
                            break;
                        }
                        _S67 = int(3);
                        break;
                    }
                    if(_S67 != int(3))
                    {
                        break;
                    }
                    int _S163;
                    int _S164;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_4 >= _S75)
                            {
                                _S66 = int(2);
                                break;
                            }
                            float _S165 = attnLo_0(vw_0[4U]);
                            float _S166 = attnHi_0(vw_0[4U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S66 = int(5);
                                            break;
                                        }
                                        float pj_33 = (*(&kernelContext_0)->ss_0)[j_4];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_4], _S165, lo_1[0U]);
                                        hi_1[0U] = fma(pj_33, _S166, hi_1[0U]);
                                        _S66 = int(7);
                                        break;
                                    }
                                    if(_S66 != int(7))
                                    {
                                        _S65 = _S66;
                                        break;
                                    }
                                    _S65 = int(6);
                                    break;
                                }
                                if(_S65 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S12)
                                        {
                                            _S64 = int(5);
                                            break;
                                        }
                                        float pj_34 = (*(&kernelContext_0)->ss_0)[_S23 + j_4];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S23 + j_4], _S165, lo_1[1U]);
                                        hi_1[1U] = fma(pj_34, _S166, hi_1[1U]);
                                        _S64 = int(7);
                                        break;
                                    }
                                    if(_S64 != int(7))
                                    {
                                        _S63 = _S64;
                                        break;
                                    }
                                    _S63 = int(6);
                                    break;
                                }
                                if(_S63 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S12)
                                        {
                                            _S62 = int(5);
                                            break;
                                        }
                                        float pj_35 = (*(&kernelContext_0)->ss_0)[2U * _S23 + j_4];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S23 + j_4], _S165, lo_1[2U]);
                                        hi_1[2U] = fma(pj_35, _S166, hi_1[2U]);
                                        _S62 = int(7);
                                        break;
                                    }
                                    if(_S62 != int(7))
                                    {
                                        _S61 = _S62;
                                        break;
                                    }
                                    _S61 = int(6);
                                    break;
                                }
                                if(_S61 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S12)
                                        {
                                            _S60 = int(5);
                                            break;
                                        }
                                        float pj_36 = (*(&kernelContext_0)->ss_0)[3U * _S23 + j_4];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S23 + j_4], _S165, lo_1[3U]);
                                        hi_1[3U] = fma(pj_36, _S166, hi_1[3U]);
                                        _S60 = int(7);
                                        break;
                                    }
                                    if(_S60 != int(7))
                                    {
                                        _S59 = _S60;
                                        break;
                                    }
                                    _S59 = int(6);
                                    break;
                                }
                                if(_S59 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S12)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_37 = (*(&kernelContext_0)->ss_0)[4U * _S23 + j_4];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S23 + j_4], _S165, lo_1[4U]);
                                        hi_1[4U] = fma(pj_37, _S166, hi_1[4U]);
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
                                        if(5U >= _S12)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_38 = (*(&kernelContext_0)->ss_0)[5U * _S23 + j_4];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S23 + j_4], _S165, lo_1[5U]);
                                        hi_1[5U] = fma(pj_38, _S166, hi_1[5U]);
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
                                        if(6U >= _S12)
                                        {
                                            _S160 = int(5);
                                            break;
                                        }
                                        float pj_39 = (*(&kernelContext_0)->ss_0)[6U * _S23 + j_4];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S23 + j_4], _S165, lo_1[6U]);
                                        hi_1[6U] = fma(pj_39, _S166, hi_1[6U]);
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
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S12)
                                        {
                                            _S164 = int(5);
                                            break;
                                        }
                                        float pj_40 = (*(&kernelContext_0)->ss_0)[7U * _S23 + j_4];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S23 + j_4], _S165, lo_1[7U]);
                                        hi_1[7U] = fma(pj_40, _S166, hi_1[7U]);
                                        _S164 = int(7);
                                        break;
                                    }
                                    if(_S164 != int(7))
                                    {
                                        _S163 = _S164;
                                        break;
                                    }
                                    _S163 = int(6);
                                    break;
                                }
                                if(_S163 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S66 = int(4);
                            break;
                        }
                        if(_S66 != int(4))
                        {
                            _S65 = _S66;
                            break;
                        }
                        _S65 = int(3);
                        break;
                    }
                    if(_S65 != int(3))
                    {
                        break;
                    }
                    int _S167;
                    int _S168;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_5 >= _S75)
                            {
                                _S64 = int(2);
                                break;
                            }
                            float _S169 = attnLo_0(vw_0[5U]);
                            float _S170 = attnHi_0(vw_0[5U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S64 = int(5);
                                            break;
                                        }
                                        float pj_41 = (*(&kernelContext_0)->ss_0)[j_5];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_5], _S169, lo_1[0U]);
                                        hi_1[0U] = fma(pj_41, _S170, hi_1[0U]);
                                        _S64 = int(7);
                                        break;
                                    }
                                    if(_S64 != int(7))
                                    {
                                        _S63 = _S64;
                                        break;
                                    }
                                    _S63 = int(6);
                                    break;
                                }
                                if(_S63 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S12)
                                        {
                                            _S62 = int(5);
                                            break;
                                        }
                                        float pj_42 = (*(&kernelContext_0)->ss_0)[_S23 + j_5];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S23 + j_5], _S169, lo_1[1U]);
                                        hi_1[1U] = fma(pj_42, _S170, hi_1[1U]);
                                        _S62 = int(7);
                                        break;
                                    }
                                    if(_S62 != int(7))
                                    {
                                        _S61 = _S62;
                                        break;
                                    }
                                    _S61 = int(6);
                                    break;
                                }
                                if(_S61 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S12)
                                        {
                                            _S60 = int(5);
                                            break;
                                        }
                                        float pj_43 = (*(&kernelContext_0)->ss_0)[2U * _S23 + j_5];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S23 + j_5], _S169, lo_1[2U]);
                                        hi_1[2U] = fma(pj_43, _S170, hi_1[2U]);
                                        _S60 = int(7);
                                        break;
                                    }
                                    if(_S60 != int(7))
                                    {
                                        _S59 = _S60;
                                        break;
                                    }
                                    _S59 = int(6);
                                    break;
                                }
                                if(_S59 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(3U >= _S12)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_44 = (*(&kernelContext_0)->ss_0)[3U * _S23 + j_5];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S23 + j_5], _S169, lo_1[3U]);
                                        hi_1[3U] = fma(pj_44, _S170, hi_1[3U]);
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
                                        if(4U >= _S12)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_45 = (*(&kernelContext_0)->ss_0)[4U * _S23 + j_5];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S23 + j_5], _S169, lo_1[4U]);
                                        hi_1[4U] = fma(pj_45, _S170, hi_1[4U]);
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
                                        if(5U >= _S12)
                                        {
                                            _S160 = int(5);
                                            break;
                                        }
                                        float pj_46 = (*(&kernelContext_0)->ss_0)[5U * _S23 + j_5];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S23 + j_5], _S169, lo_1[5U]);
                                        hi_1[5U] = fma(pj_46, _S170, hi_1[5U]);
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
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S12)
                                        {
                                            _S164 = int(5);
                                            break;
                                        }
                                        float pj_47 = (*(&kernelContext_0)->ss_0)[6U * _S23 + j_5];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S23 + j_5], _S169, lo_1[6U]);
                                        hi_1[6U] = fma(pj_47, _S170, hi_1[6U]);
                                        _S164 = int(7);
                                        break;
                                    }
                                    if(_S164 != int(7))
                                    {
                                        _S163 = _S164;
                                        break;
                                    }
                                    _S163 = int(6);
                                    break;
                                }
                                if(_S163 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S12)
                                        {
                                            _S168 = int(5);
                                            break;
                                        }
                                        float pj_48 = (*(&kernelContext_0)->ss_0)[7U * _S23 + j_5];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S23 + j_5], _S169, lo_1[7U]);
                                        hi_1[7U] = fma(pj_48, _S170, hi_1[7U]);
                                        _S168 = int(7);
                                        break;
                                    }
                                    if(_S168 != int(7))
                                    {
                                        _S167 = _S168;
                                        break;
                                    }
                                    _S167 = int(6);
                                    break;
                                }
                                if(_S167 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S64 = int(4);
                            break;
                        }
                        if(_S64 != int(4))
                        {
                            _S63 = _S64;
                            break;
                        }
                        _S63 = int(3);
                        break;
                    }
                    if(_S63 != int(3))
                    {
                        break;
                    }
                    int _S171;
                    int _S172;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_6 >= _S75)
                            {
                                _S62 = int(2);
                                break;
                            }
                            float _S173 = attnLo_0(vw_0[6U]);
                            float _S174 = attnHi_0(vw_0[6U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S62 = int(5);
                                            break;
                                        }
                                        float pj_49 = (*(&kernelContext_0)->ss_0)[j_6];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_6], _S173, lo_1[0U]);
                                        hi_1[0U] = fma(pj_49, _S174, hi_1[0U]);
                                        _S62 = int(7);
                                        break;
                                    }
                                    if(_S62 != int(7))
                                    {
                                        _S61 = _S62;
                                        break;
                                    }
                                    _S61 = int(6);
                                    break;
                                }
                                if(_S61 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S12)
                                        {
                                            _S60 = int(5);
                                            break;
                                        }
                                        float pj_50 = (*(&kernelContext_0)->ss_0)[_S23 + j_6];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S23 + j_6], _S173, lo_1[1U]);
                                        hi_1[1U] = fma(pj_50, _S174, hi_1[1U]);
                                        _S60 = int(7);
                                        break;
                                    }
                                    if(_S60 != int(7))
                                    {
                                        _S59 = _S60;
                                        break;
                                    }
                                    _S59 = int(6);
                                    break;
                                }
                                if(_S59 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(2U >= _S12)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_51 = (*(&kernelContext_0)->ss_0)[2U * _S23 + j_6];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S23 + j_6], _S173, lo_1[2U]);
                                        hi_1[2U] = fma(pj_51, _S174, hi_1[2U]);
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
                                        if(3U >= _S12)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_52 = (*(&kernelContext_0)->ss_0)[3U * _S23 + j_6];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S23 + j_6], _S173, lo_1[3U]);
                                        hi_1[3U] = fma(pj_52, _S174, hi_1[3U]);
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
                                        if(4U >= _S12)
                                        {
                                            _S160 = int(5);
                                            break;
                                        }
                                        float pj_53 = (*(&kernelContext_0)->ss_0)[4U * _S23 + j_6];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S23 + j_6], _S173, lo_1[4U]);
                                        hi_1[4U] = fma(pj_53, _S174, hi_1[4U]);
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
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S12)
                                        {
                                            _S164 = int(5);
                                            break;
                                        }
                                        float pj_54 = (*(&kernelContext_0)->ss_0)[5U * _S23 + j_6];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S23 + j_6], _S173, lo_1[5U]);
                                        hi_1[5U] = fma(pj_54, _S174, hi_1[5U]);
                                        _S164 = int(7);
                                        break;
                                    }
                                    if(_S164 != int(7))
                                    {
                                        _S163 = _S164;
                                        break;
                                    }
                                    _S163 = int(6);
                                    break;
                                }
                                if(_S163 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S12)
                                        {
                                            _S168 = int(5);
                                            break;
                                        }
                                        float pj_55 = (*(&kernelContext_0)->ss_0)[6U * _S23 + j_6];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S23 + j_6], _S173, lo_1[6U]);
                                        hi_1[6U] = fma(pj_55, _S174, hi_1[6U]);
                                        _S168 = int(7);
                                        break;
                                    }
                                    if(_S168 != int(7))
                                    {
                                        _S167 = _S168;
                                        break;
                                    }
                                    _S167 = int(6);
                                    break;
                                }
                                if(_S167 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(7U >= _S12)
                                        {
                                            _S172 = int(5);
                                            break;
                                        }
                                        float pj_56 = (*(&kernelContext_0)->ss_0)[7U * _S23 + j_6];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S23 + j_6], _S173, lo_1[7U]);
                                        hi_1[7U] = fma(pj_56, _S174, hi_1[7U]);
                                        _S172 = int(7);
                                        break;
                                    }
                                    if(_S172 != int(7))
                                    {
                                        _S171 = _S172;
                                        break;
                                    }
                                    _S171 = int(6);
                                    break;
                                }
                                if(_S171 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S62 = int(4);
                            break;
                        }
                        if(_S62 != int(4))
                        {
                            _S61 = _S62;
                            break;
                        }
                        _S61 = int(3);
                        break;
                    }
                    if(_S61 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(j_7 >= _S75)
                            {
                                _S60 = int(2);
                                break;
                            }
                            float _S175 = attnLo_0(vw_0[7U]);
                            float _S176 = attnHi_0(vw_0[7U]);
                            for(;;)
                            {
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(visible_0)
                                        {
                                            _S60 = int(5);
                                            break;
                                        }
                                        float pj_57 = (*(&kernelContext_0)->ss_0)[j_7];
                                        lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_7], _S175, lo_1[0U]);
                                        hi_1[0U] = fma(pj_57, _S176, hi_1[0U]);
                                        _S60 = int(7);
                                        break;
                                    }
                                    if(_S60 != int(7))
                                    {
                                        _S59 = _S60;
                                        break;
                                    }
                                    _S59 = int(6);
                                    break;
                                }
                                if(_S59 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(1U >= _S12)
                                        {
                                            _S152 = int(5);
                                            break;
                                        }
                                        float pj_58 = (*(&kernelContext_0)->ss_0)[_S23 + j_7];
                                        lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S23 + j_7], _S175, lo_1[1U]);
                                        hi_1[1U] = fma(pj_58, _S176, hi_1[1U]);
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
                                        if(2U >= _S12)
                                        {
                                            _S156 = int(5);
                                            break;
                                        }
                                        float pj_59 = (*(&kernelContext_0)->ss_0)[2U * _S23 + j_7];
                                        lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S23 + j_7], _S175, lo_1[2U]);
                                        hi_1[2U] = fma(pj_59, _S176, hi_1[2U]);
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
                                        if(3U >= _S12)
                                        {
                                            _S160 = int(5);
                                            break;
                                        }
                                        float pj_60 = (*(&kernelContext_0)->ss_0)[3U * _S23 + j_7];
                                        lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S23 + j_7], _S175, lo_1[3U]);
                                        hi_1[3U] = fma(pj_60, _S176, hi_1[3U]);
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
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(4U >= _S12)
                                        {
                                            _S164 = int(5);
                                            break;
                                        }
                                        float pj_61 = (*(&kernelContext_0)->ss_0)[4U * _S23 + j_7];
                                        lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S23 + j_7], _S175, lo_1[4U]);
                                        hi_1[4U] = fma(pj_61, _S176, hi_1[4U]);
                                        _S164 = int(7);
                                        break;
                                    }
                                    if(_S164 != int(7))
                                    {
                                        _S163 = _S164;
                                        break;
                                    }
                                    _S163 = int(6);
                                    break;
                                }
                                if(_S163 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(5U >= _S12)
                                        {
                                            _S168 = int(5);
                                            break;
                                        }
                                        float pj_62 = (*(&kernelContext_0)->ss_0)[5U * _S23 + j_7];
                                        lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S23 + j_7], _S175, lo_1[5U]);
                                        hi_1[5U] = fma(pj_62, _S176, hi_1[5U]);
                                        _S168 = int(7);
                                        break;
                                    }
                                    if(_S168 != int(7))
                                    {
                                        _S167 = _S168;
                                        break;
                                    }
                                    _S167 = int(6);
                                    break;
                                }
                                if(_S167 != int(6))
                                {
                                    break;
                                }
                                for(;;)
                                {
                                    for(;;)
                                    {
                                        if(6U >= _S12)
                                        {
                                            _S172 = int(5);
                                            break;
                                        }
                                        float pj_63 = (*(&kernelContext_0)->ss_0)[6U * _S23 + j_7];
                                        lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S23 + j_7], _S175, lo_1[6U]);
                                        hi_1[6U] = fma(pj_63, _S176, hi_1[6U]);
                                        _S172 = int(7);
                                        break;
                                    }
                                    if(_S172 != int(7))
                                    {
                                        _S171 = _S172;
                                        break;
                                    }
                                    _S171 = int(6);
                                    break;
                                }
                                if(_S171 != int(6))
                                {
                                    break;
                                }
                                int _S177;
                                for(;;)
                                {
                                    int _S178;
                                    for(;;)
                                    {
                                        if(7U >= _S12)
                                        {
                                            _S178 = int(5);
                                            break;
                                        }
                                        float pj_64 = (*(&kernelContext_0)->ss_0)[7U * _S23 + j_7];
                                        lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S23 + j_7], _S175, lo_1[7U]);
                                        hi_1[7U] = fma(pj_64, _S176, hi_1[7U]);
                                        _S178 = int(7);
                                        break;
                                    }
                                    if(_S178 != int(7))
                                    {
                                        _S177 = _S178;
                                        break;
                                    }
                                    _S177 = int(6);
                                    break;
                                }
                                if(_S177 != int(6))
                                {
                                    break;
                                }
                                break;
                            }
                            _S60 = int(4);
                            break;
                        }
                        if(_S60 != int(4))
                        {
                            _S59 = _S60;
                            break;
                        }
                        _S59 = int(3);
                        break;
                    }
                    if(_S59 != int(3))
                    {
                        break;
                    }
                    break;
                }
                kw_6 = kw_6 + 32U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        w_2 = w_2 + _S23;
    }
    r_0 = 0U;
    for(;;)
    {
        if(r_0 < _S12)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(0U == r_0)
        {
            acc_1 = lo_1[0U];
            acc_0 = hi_1[0U];
        }
        else
        {
            acc_1 = 0.0;
            acc_0 = 0.0;
        }
        float b_3;
        if(1U == r_0)
        {
            pj_0 = lo_1[1U];
            b_3 = hi_1[1U];
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
            a_11 = lo_1[2U];
            b_4 = hi_1[2U];
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
            a_12 = lo_1[3U];
            b_5 = hi_1[3U];
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
            a_13 = lo_1[4U];
            b_6 = hi_1[4U];
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
            a_14 = lo_1[5U];
            b_7 = hi_1[5U];
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
            a_15 = lo_1[6U];
            b_8 = hi_1[6U];
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
            a_16 = lo_1[7U];
            b_9 = hi_1[7U];
        }
        else
        {
            a_16 = a_15;
            b_9 = b_8;
        }
        uint _S179 = 2U * _S57;
        uint _S180 = _S58 * 64U + _S179;
        (*(&kernelContext_0)->ss_0)[_S180] = a_16;
        (*(&kernelContext_0)->ss_0)[_S180 + 1U] = b_9;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S58 != 0U)
        {
            visible_0 = true;
        }
        else
        {
            visible_0 = _S57 >= _S15;
        }
        if(visible_0)
        {
            r_0 = r_0 + 1U;
            continue;
        }
        uint _S181 = 64U + _S179;
        uint _S182 = 128U + _S179;
        uint _S183 = 192U + _S179;
        float a_17 = (*(&kernelContext_0)->ss_0)[_S179] + (*(&kernelContext_0)->ss_0)[_S181] + (*(&kernelContext_0)->ss_0)[_S182] + (*(&kernelContext_0)->ss_0)[_S183];
        float b_10 = (*(&kernelContext_0)->ss_0)[_S179 + 1U] + (*(&kernelContext_0)->ss_0)[_S181 + 1U] + (*(&kernelContext_0)->ss_0)[_S182 + 1U] + (*(&kernelContext_0)->ss_0)[_S183 + 1U];
        uint _S184 = r_0 / (&kernelContext_0)->p_0->queries_0;
        uint h_4 = _S18 + _S184;
        uint i_2 = r_0 % (&kernelContext_0)->p_0->queries_0;
        uint row_0 = _S13 + h_4 * (&kernelContext_0)->p_0->queries_0 + i_2;
        float a_18;
        float b_11;
        if(((&kernelContext_0)->p_0->splits_1) > 1U)
        {
            uint pr_8 = split_0 * _S14 + row_0;
            uint at_0 = attnAccBase_0((&kernelContext_0)->p_0->splits_1, _S14) + pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S179;
            if(_S57 == 0U)
            {
                uint _S185 = 2U * pr_8;
                float device* _S186 = (&kernelContext_0)->partials_0+_S185;
                bool _S187 = _S21 == 0U;
                if(_S187)
                {
                    a_18 = dg_negative_infinity_0();
                }
                else
                {
                    a_18 = (*(&kernelContext_0)->rowM_0)[r_0];
                }
                *_S186 = a_18;
                float device* _S188 = (&kernelContext_0)->partials_0+(_S185 + 1U);
                if(_S187)
                {
                    b_11 = 0.0;
                }
                else
                {
                    b_11 = (*(&kernelContext_0)->rowL_0)[r_0];
                }
                *_S188 = b_11;
            }
            float device* _S189 = (&kernelContext_0)->partials_0+at_0;
            if(((*(&kernelContext_0)->rowL_0)[r_0]) == 0.0)
            {
                a_18 = 0.0;
            }
            else
            {
                a_18 = a_17;
            }
            *_S189 = a_18;
            float device* _S190 = (&kernelContext_0)->partials_0+(at_0 + 1U);
            if(((*(&kernelContext_0)->rowL_0)[r_0]) == 0.0)
            {
                b_11 = 0.0;
            }
            else
            {
                b_11 = b_10;
            }
            *_S190 = b_11;
            r_0 = r_0 + 1U;
            continue;
        }
        float l_0 = (*(&kernelContext_0)->rowL_0)[r_0];
        bool _S191 = ((*(&kernelContext_0)->rowL_0)[r_0]) == 0.0;
        if(_S191)
        {
            a_18 = 0.0;
        }
        else
        {
            a_18 = a_17 / l_0;
        }
        if(_S191)
        {
            b_11 = 0.0;
        }
        else
        {
            b_11 = b_10 / l_0;
        }
        *((&kernelContext_0)->o_0+(row_0 * _S15 + _S57)) = attnPack_0(a_18, b_11);
        r_0 = r_0 + 1U;
    }
    return;
}

