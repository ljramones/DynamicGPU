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
    array<uint, int(528)> threadgroup* qs_0;
    array<uint, int(1056)> threadgroup* ks_0;
    array<uint, int(1056)> threadgroup* vs_0;
    array<float, int(512)> threadgroup* ss_0;
    array<float, int(512)> threadgroup* ps_0;
};

[[kernel]] void attn_flash_64(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(7)]], uint device* q_1 [[buffer(0)]], uint device* table_1 [[buffer(6)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partialML_1 [[buffer(4)]], float device* partialAcc_1 [[buffer(5)]], uint device* o_1 [[buffer(3)]])
{
    uint _S3;
    uint _S4;
    uint _S5;
    uint _S6;
    uint _S7;
    uint _S8;
    uint _S9;
    uint _S10;
    uint _S11;
    uint _S12;
    uint _S13;
    uint _S14;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->q_0 = q_1;
    (&kernelContext_0)->table_0 = table_1;
    (&kernelContext_0)->k_0 = k_1;
    (&kernelContext_0)->v_0 = v_1;
    (&kernelContext_0)->partialML_0 = partialML_1;
    (&kernelContext_0)->partialAcc_0 = partialAcc_1;
    (&kernelContext_0)->o_0 = o_1;
    threadgroup array<uint, int(528)> qs_1;
    (&kernelContext_0)->qs_0 = &qs_1;
    threadgroup array<uint, int(1056)> ks_1;
    (&kernelContext_0)->ks_0 = &ks_1;
    threadgroup array<uint, int(1056)> vs_1;
    (&kernelContext_0)->vs_0 = &vs_1;
    threadgroup array<float, int(512)> ss_1;
    (&kernelContext_0)->ss_0 = &ss_1;
    threadgroup array<float, int(512)> ps_1;
    (&kernelContext_0)->ps_0 = &ps_1;
    uint t_0 = local_0.x;
    uint lr_0 = t_0 / 8U;
    uint kk_0 = t_0 % 8U;
    uint kvHead_0 = group_0.y;
    uint split_0 = group_0.z;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint rowsKv_0 = hpk_0 * (&kernelContext_0)->p_0->queries_0;
    uint r0_0 = group_0.x * 16U;
    uint r_0 = r0_0 + lr_0;
    bool live_0 = r_0 < rowsKv_0;
    uint _S15 = kvHead_0 * hpk_0;
    uint qi_0;
    if(live_0)
    {
        uint _S16 = r_0 / (&kernelContext_0)->p_0->queries_0;
        qi_0 = _S16;
    }
    else
    {
        qi_0 = 0U;
    }
    uint head_0 = _S15 + qi_0;
    if(live_0)
    {
        uint _S17 = r_0 % (&kernelContext_0)->p_0->queries_0;
        qi_0 = _S17;
    }
    else
    {
        qi_0 = 0U;
    }
    bool ok_0;
    uint w_2;
    uint _S18 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint offset_0 = (&kernelContext_0)->p_0->keys_0 - (&kernelContext_0)->p_0->queries_0;
    uint e_1 = t_0;
    for(;;)
    {
        if(e_1 < 512U)
        {
        }
        else
        {
            break;
        }
        uint row_0 = r0_0 + e_1 / 32U;
        uint pr_0 = e_1 % 32U;
        if(row_0 < rowsKv_0)
        {
            ok_0 = pr_0 < _S18;
        }
        else
        {
            ok_0 = false;
        }
        if(ok_0)
        {
            uint _S19 = row_0 / (&kernelContext_0)->p_0->queries_0;
            uint h_3 = _S15 + _S19;
            uint i_0 = row_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((h_3 * (&kernelContext_0)->p_0->queries_0 + i_0) * _S18 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_1 / 32U * 33U + pr_0] = w_2;
        e_1 = e_1 + 128U;
    }
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S20 = min((&kernelContext_0)->p_0->keys_0, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint rLast_0 = min(r0_0 + 16U, rowsKv_0) - 1U;
        uint _S21 = r0_0 / (&kernelContext_0)->p_0->queries_0;
        uint _S22 = rLast_0 / (&kernelContext_0)->p_0->queries_0;
        if(_S21 != _S22)
        {
            w_2 = (&kernelContext_0)->p_0->queries_0 - 1U;
        }
        else
        {
            uint _S23 = rLast_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = _S23;
        }
        w_2 = min(_S20, w_2 + offset_0 + 1U);
    }
    else
    {
        w_2 = _S20;
    }
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        e_1 = qi_0 + offset_0 + 1U;
    }
    else
    {
        e_1 = (&kernelContext_0)->p_0->keys_0;
    }
    float _S24 = dg_negative_infinity_0();
    thread array<float, int(4)> accLo_0;
    thread array<float, int(4)> accHi_0;
    accLo_0[0U] = 0.0;
    accHi_0[0U] = 0.0;
    accLo_0[1U] = 0.0;
    accHi_0[1U] = 0.0;
    accLo_0[2U] = 0.0;
    accHi_0[2U] = 0.0;
    accLo_0[3U] = 0.0;
    accHi_0[3U] = 0.0;
    uint b_0 = kStart_0;
    float m_0 = _S24;
    float l_0 = 0.0;
    for(;;)
    {
        if(b_0 < w_2)
        {
        }
        else
        {
            break;
        }
        uint p0_0;
        uint p0_1;
        uint at_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint e_2 = t_0;
        for(;;)
        {
            if(e_2 < 1024U)
            {
            }
            else
            {
                break;
            }
            uint j_0 = e_2 / 32U;
            uint pr_1 = e_2 % 32U;
            uint key_0 = b_0 + j_0;
            if(key_0 < w_2)
            {
                ok_0 = pr_1 < _S18;
            }
            else
            {
                ok_0 = false;
            }
            if(ok_0)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    at_0 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * _S18 + pr_1;
                }
                else
                {
                    uint _S25 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S26 = (*((&kernelContext_0)->table_0+_S25) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S27 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
                    at_0 = (_S26 + _S27) * _S18 + pr_1;
                }
            }
            else
            {
                at_0 = 0U;
            }
            uint _S28 = j_0 * 33U + pr_1;
            if(ok_0)
            {
                p0_1 = *((&kernelContext_0)->k_0+at_0);
            }
            else
            {
                p0_1 = 0U;
            }
            (*(&kernelContext_0)->ks_0)[_S28] = p0_1;
            if(ok_0)
            {
                p0_0 = *((&kernelContext_0)->v_0+at_0);
            }
            else
            {
                p0_0 = 0U;
            }
            (*(&kernelContext_0)->vs_0)[_S28] = p0_0;
            e_2 = e_2 + 128U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(;;)
        {
            for(;;)
            {
                for(;;)
                {
                    uint key_1 = b_0 + kk_0;
                    at_0 = 0U;
                    float s_0 = 0.0;
                    for(;;)
                    {
                        if(at_0 < 32U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S29 = lr_0 * 33U;
                        uint _S30 = kk_0 * 33U;
                        uint pr_2 = at_0 + 1U;
                        uint pr_3 = at_0 + 2U;
                        uint pr_4 = at_0 + 3U;
                        uint pr_5 = at_0 + 4U;
                        uint pr_6 = at_0 + 5U;
                        uint pr_7 = at_0 + 6U;
                        uint pr_8 = at_0 + 7U;
                        float s_1 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S29 + pr_8]), attnHi_0((*(&kernelContext_0)->ks_0)[_S30 + pr_8]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S29 + pr_8]), attnLo_0((*(&kernelContext_0)->ks_0)[_S30 + pr_8]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S29 + pr_7]), attnHi_0((*(&kernelContext_0)->ks_0)[_S30 + pr_7]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S29 + pr_7]), attnLo_0((*(&kernelContext_0)->ks_0)[_S30 + pr_7]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S29 + pr_6]), attnHi_0((*(&kernelContext_0)->ks_0)[_S30 + pr_6]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S29 + pr_6]), attnLo_0((*(&kernelContext_0)->ks_0)[_S30 + pr_6]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S29 + pr_5]), attnHi_0((*(&kernelContext_0)->ks_0)[_S30 + pr_5]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S29 + pr_5]), attnLo_0((*(&kernelContext_0)->ks_0)[_S30 + pr_5]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S29 + pr_4]), attnHi_0((*(&kernelContext_0)->ks_0)[_S30 + pr_4]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S29 + pr_4]), attnLo_0((*(&kernelContext_0)->ks_0)[_S30 + pr_4]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S29 + pr_3]), attnHi_0((*(&kernelContext_0)->ks_0)[_S30 + pr_3]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S29 + pr_3]), attnLo_0((*(&kernelContext_0)->ks_0)[_S30 + pr_3]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S29 + pr_2]), attnHi_0((*(&kernelContext_0)->ks_0)[_S30 + pr_2]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S29 + pr_2]), attnLo_0((*(&kernelContext_0)->ks_0)[_S30 + pr_2]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S29 + at_0]), attnHi_0((*(&kernelContext_0)->ks_0)[_S30 + at_0]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S29 + at_0]), attnLo_0((*(&kernelContext_0)->ks_0)[_S30 + at_0]), s_0))))))))))))))));
                        at_0 = at_0 + 8U;
                        s_0 = s_1;
                    }
                    if(live_0)
                    {
                        ok_0 = key_1 < w_2;
                    }
                    else
                    {
                        ok_0 = false;
                    }
                    bool visible_0;
                    if(ok_0)
                    {
                        visible_0 = key_1 < e_1;
                    }
                    else
                    {
                        visible_0 = false;
                    }
                    uint _S31 = lr_0 * 32U;
                    _S3 = _S31;
                    uint _S32 = _S31 + kk_0;
                    _S11 = _S32;
                    _S4 = _S32;
                    float _S33;
                    if(visible_0)
                    {
                        _S33 = s_0 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S33 = _S24;
                    }
                    (*(&kernelContext_0)->ss_0)[_S32] = _S33;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_1 = kk_0 + 8U;
                    _S5 = j_1;
                    uint key_2 = b_0 + j_1;
                    p0_1 = 0U;
                    float s_2 = 0.0;
                    for(;;)
                    {
                        if(p0_1 < 32U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S34 = lr_0 * 33U;
                        uint _S35 = j_1 * 33U;
                        uint pr_9 = p0_1 + 1U;
                        uint pr_10 = p0_1 + 2U;
                        uint pr_11 = p0_1 + 3U;
                        uint pr_12 = p0_1 + 4U;
                        uint pr_13 = p0_1 + 5U;
                        uint pr_14 = p0_1 + 6U;
                        uint pr_15 = p0_1 + 7U;
                        float s_3 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S34 + pr_15]), attnHi_0((*(&kernelContext_0)->ks_0)[_S35 + pr_15]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S34 + pr_15]), attnLo_0((*(&kernelContext_0)->ks_0)[_S35 + pr_15]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S34 + pr_14]), attnHi_0((*(&kernelContext_0)->ks_0)[_S35 + pr_14]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S34 + pr_14]), attnLo_0((*(&kernelContext_0)->ks_0)[_S35 + pr_14]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S34 + pr_13]), attnHi_0((*(&kernelContext_0)->ks_0)[_S35 + pr_13]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S34 + pr_13]), attnLo_0((*(&kernelContext_0)->ks_0)[_S35 + pr_13]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S34 + pr_12]), attnHi_0((*(&kernelContext_0)->ks_0)[_S35 + pr_12]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S34 + pr_12]), attnLo_0((*(&kernelContext_0)->ks_0)[_S35 + pr_12]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S34 + pr_11]), attnHi_0((*(&kernelContext_0)->ks_0)[_S35 + pr_11]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S34 + pr_11]), attnLo_0((*(&kernelContext_0)->ks_0)[_S35 + pr_11]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S34 + pr_10]), attnHi_0((*(&kernelContext_0)->ks_0)[_S35 + pr_10]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S34 + pr_10]), attnLo_0((*(&kernelContext_0)->ks_0)[_S35 + pr_10]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S34 + pr_9]), attnHi_0((*(&kernelContext_0)->ks_0)[_S35 + pr_9]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S34 + pr_9]), attnLo_0((*(&kernelContext_0)->ks_0)[_S35 + pr_9]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S34 + p0_1]), attnHi_0((*(&kernelContext_0)->ks_0)[_S35 + p0_1]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S34 + p0_1]), attnLo_0((*(&kernelContext_0)->ks_0)[_S35 + p0_1]), s_2))))))))))))))));
                        p0_1 = p0_1 + 8U;
                        s_2 = s_3;
                    }
                    bool _S36;
                    if(live_0)
                    {
                        _S36 = key_2 < w_2;
                    }
                    else
                    {
                        _S36 = false;
                    }
                    bool visible_1;
                    if(_S36)
                    {
                        visible_1 = key_2 < e_1;
                    }
                    else
                    {
                        visible_1 = false;
                    }
                    uint _S37 = _S3 + j_1;
                    _S12 = _S37;
                    _S6 = _S37;
                    float _S38;
                    if(visible_1)
                    {
                        _S38 = s_2 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S38 = _S24;
                    }
                    (*(&kernelContext_0)->ss_0)[_S37] = _S38;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_2 = kk_0 + 16U;
                    _S7 = j_2;
                    uint key_3 = b_0 + j_2;
                    p0_0 = 0U;
                    float s_4 = 0.0;
                    for(;;)
                    {
                        if(p0_0 < 32U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S39 = lr_0 * 33U;
                        uint _S40 = j_2 * 33U;
                        uint pr_16 = p0_0 + 1U;
                        uint pr_17 = p0_0 + 2U;
                        uint pr_18 = p0_0 + 3U;
                        uint pr_19 = p0_0 + 4U;
                        uint pr_20 = p0_0 + 5U;
                        uint pr_21 = p0_0 + 6U;
                        uint pr_22 = p0_0 + 7U;
                        float s_5 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S39 + pr_22]), attnHi_0((*(&kernelContext_0)->ks_0)[_S40 + pr_22]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S39 + pr_22]), attnLo_0((*(&kernelContext_0)->ks_0)[_S40 + pr_22]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S39 + pr_21]), attnHi_0((*(&kernelContext_0)->ks_0)[_S40 + pr_21]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S39 + pr_21]), attnLo_0((*(&kernelContext_0)->ks_0)[_S40 + pr_21]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S39 + pr_20]), attnHi_0((*(&kernelContext_0)->ks_0)[_S40 + pr_20]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S39 + pr_20]), attnLo_0((*(&kernelContext_0)->ks_0)[_S40 + pr_20]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S39 + pr_19]), attnHi_0((*(&kernelContext_0)->ks_0)[_S40 + pr_19]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S39 + pr_19]), attnLo_0((*(&kernelContext_0)->ks_0)[_S40 + pr_19]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S39 + pr_18]), attnHi_0((*(&kernelContext_0)->ks_0)[_S40 + pr_18]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S39 + pr_18]), attnLo_0((*(&kernelContext_0)->ks_0)[_S40 + pr_18]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S39 + pr_17]), attnHi_0((*(&kernelContext_0)->ks_0)[_S40 + pr_17]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S39 + pr_17]), attnLo_0((*(&kernelContext_0)->ks_0)[_S40 + pr_17]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S39 + pr_16]), attnHi_0((*(&kernelContext_0)->ks_0)[_S40 + pr_16]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S39 + pr_16]), attnLo_0((*(&kernelContext_0)->ks_0)[_S40 + pr_16]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S39 + p0_0]), attnHi_0((*(&kernelContext_0)->ks_0)[_S40 + p0_0]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S39 + p0_0]), attnLo_0((*(&kernelContext_0)->ks_0)[_S40 + p0_0]), s_4))))))))))))))));
                        p0_0 = p0_0 + 8U;
                        s_4 = s_5;
                    }
                    bool _S41;
                    if(live_0)
                    {
                        _S41 = key_3 < w_2;
                    }
                    else
                    {
                        _S41 = false;
                    }
                    bool visible_2;
                    if(_S41)
                    {
                        visible_2 = key_3 < e_1;
                    }
                    else
                    {
                        visible_2 = false;
                    }
                    uint _S42 = _S3 + j_2;
                    _S13 = _S42;
                    _S8 = _S42;
                    float _S43;
                    if(visible_2)
                    {
                        _S43 = s_4 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S43 = _S24;
                    }
                    (*(&kernelContext_0)->ss_0)[_S42] = _S43;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_3 = kk_0 + 24U;
                    _S9 = j_3;
                    uint key_4 = b_0 + j_3;
                    uint p0_2 = 0U;
                    float s_6 = 0.0;
                    for(;;)
                    {
                        if(p0_2 < 32U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S44 = lr_0 * 33U;
                        uint _S45 = j_3 * 33U;
                        uint pr_23 = p0_2 + 1U;
                        uint pr_24 = p0_2 + 2U;
                        uint pr_25 = p0_2 + 3U;
                        uint pr_26 = p0_2 + 4U;
                        uint pr_27 = p0_2 + 5U;
                        uint pr_28 = p0_2 + 6U;
                        uint pr_29 = p0_2 + 7U;
                        float s_7 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S44 + pr_29]), attnHi_0((*(&kernelContext_0)->ks_0)[_S45 + pr_29]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S44 + pr_29]), attnLo_0((*(&kernelContext_0)->ks_0)[_S45 + pr_29]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S44 + pr_28]), attnHi_0((*(&kernelContext_0)->ks_0)[_S45 + pr_28]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S44 + pr_28]), attnLo_0((*(&kernelContext_0)->ks_0)[_S45 + pr_28]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S44 + pr_27]), attnHi_0((*(&kernelContext_0)->ks_0)[_S45 + pr_27]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S44 + pr_27]), attnLo_0((*(&kernelContext_0)->ks_0)[_S45 + pr_27]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S44 + pr_26]), attnHi_0((*(&kernelContext_0)->ks_0)[_S45 + pr_26]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S44 + pr_26]), attnLo_0((*(&kernelContext_0)->ks_0)[_S45 + pr_26]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S44 + pr_25]), attnHi_0((*(&kernelContext_0)->ks_0)[_S45 + pr_25]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S44 + pr_25]), attnLo_0((*(&kernelContext_0)->ks_0)[_S45 + pr_25]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S44 + pr_24]), attnHi_0((*(&kernelContext_0)->ks_0)[_S45 + pr_24]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S44 + pr_24]), attnLo_0((*(&kernelContext_0)->ks_0)[_S45 + pr_24]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S44 + pr_23]), attnHi_0((*(&kernelContext_0)->ks_0)[_S45 + pr_23]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S44 + pr_23]), attnLo_0((*(&kernelContext_0)->ks_0)[_S45 + pr_23]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S44 + p0_2]), attnHi_0((*(&kernelContext_0)->ks_0)[_S45 + p0_2]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S44 + p0_2]), attnLo_0((*(&kernelContext_0)->ks_0)[_S45 + p0_2]), s_6))))))))))))))));
                        p0_2 = p0_2 + 8U;
                        s_6 = s_7;
                    }
                    bool _S46;
                    if(live_0)
                    {
                        _S46 = key_4 < w_2;
                    }
                    else
                    {
                        _S46 = false;
                    }
                    bool visible_3;
                    if(_S46)
                    {
                        visible_3 = key_4 < e_1;
                    }
                    else
                    {
                        visible_3 = false;
                    }
                    uint _S47 = _S3 + j_3;
                    _S14 = _S47;
                    _S10 = _S47;
                    float _S48;
                    if(visible_3)
                    {
                        _S48 = s_6 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S48 = _S24;
                    }
                    (*(&kernelContext_0)->ss_0)[_S47] = _S48;
                    break;
                }
                break;
            }
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S49 = _S3 + 1U;
        uint _S50 = _S3 + 2U;
        uint _S51 = _S3 + 3U;
        uint _S52 = _S3 + 4U;
        uint _S53 = _S3 + 5U;
        uint _S54 = _S3 + 6U;
        uint _S55 = _S3 + 7U;
        uint _S56 = _S3 + 8U;
        uint _S57 = _S3 + 9U;
        uint _S58 = _S3 + 10U;
        uint _S59 = _S3 + 11U;
        uint _S60 = _S3 + 12U;
        uint _S61 = _S3 + 13U;
        uint _S62 = _S3 + 14U;
        uint _S63 = _S3 + 15U;
        uint _S64 = _S3 + 16U;
        uint _S65 = _S3 + 17U;
        uint _S66 = _S3 + 18U;
        uint _S67 = _S3 + 19U;
        uint _S68 = _S3 + 20U;
        uint _S69 = _S3 + 21U;
        uint _S70 = _S3 + 22U;
        uint _S71 = _S3 + 23U;
        uint _S72 = _S3 + 24U;
        uint _S73 = _S3 + 25U;
        uint _S74 = _S3 + 26U;
        uint _S75 = _S3 + 27U;
        uint _S76 = _S3 + 28U;
        uint _S77 = _S3 + 29U;
        uint _S78 = _S3 + 30U;
        uint _S79 = _S3 + 31U;
        float _S80 = max(m_0, max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(_S24, (*(&kernelContext_0)->ss_0)[_S3]), (*(&kernelContext_0)->ss_0)[_S49]), (*(&kernelContext_0)->ss_0)[_S50]), (*(&kernelContext_0)->ss_0)[_S51]), (*(&kernelContext_0)->ss_0)[_S52]), (*(&kernelContext_0)->ss_0)[_S53]), (*(&kernelContext_0)->ss_0)[_S54]), (*(&kernelContext_0)->ss_0)[_S55]), (*(&kernelContext_0)->ss_0)[_S56]), (*(&kernelContext_0)->ss_0)[_S57]), (*(&kernelContext_0)->ss_0)[_S58]), (*(&kernelContext_0)->ss_0)[_S59]), (*(&kernelContext_0)->ss_0)[_S60]), (*(&kernelContext_0)->ss_0)[_S61]), (*(&kernelContext_0)->ss_0)[_S62]), (*(&kernelContext_0)->ss_0)[_S63]), (*(&kernelContext_0)->ss_0)[_S64]), (*(&kernelContext_0)->ss_0)[_S65]), (*(&kernelContext_0)->ss_0)[_S66]), (*(&kernelContext_0)->ss_0)[_S67]), (*(&kernelContext_0)->ss_0)[_S68]), (*(&kernelContext_0)->ss_0)[_S69]), (*(&kernelContext_0)->ss_0)[_S70]), (*(&kernelContext_0)->ss_0)[_S71]), (*(&kernelContext_0)->ss_0)[_S72]), (*(&kernelContext_0)->ss_0)[_S73]), (*(&kernelContext_0)->ss_0)[_S74]), (*(&kernelContext_0)->ss_0)[_S75]), (*(&kernelContext_0)->ss_0)[_S76]), (*(&kernelContext_0)->ss_0)[_S77]), (*(&kernelContext_0)->ss_0)[_S78]), (*(&kernelContext_0)->ss_0)[_S79]));
        float alpha_0;
        if(m_0 == _S24)
        {
            alpha_0 = 0.0;
        }
        else
        {
            alpha_0 = exp(m_0 - _S80);
        }
        float s_8 = (*(&kernelContext_0)->ss_0)[_S11];
        float _S81;
        if(((*(&kernelContext_0)->ss_0)[_S11]) == _S24)
        {
            _S81 = 0.0;
        }
        else
        {
            _S81 = exp(s_8 - _S80);
        }
        (*(&kernelContext_0)->ps_0)[_S4] = _S81;
        float s_9 = (*(&kernelContext_0)->ss_0)[_S12];
        float _S82;
        if(((*(&kernelContext_0)->ss_0)[_S12]) == _S24)
        {
            _S82 = 0.0;
        }
        else
        {
            _S82 = exp(s_9 - _S80);
        }
        (*(&kernelContext_0)->ps_0)[_S6] = _S82;
        float s_10 = (*(&kernelContext_0)->ss_0)[_S13];
        float _S83;
        if(((*(&kernelContext_0)->ss_0)[_S13]) == _S24)
        {
            _S83 = 0.0;
        }
        else
        {
            _S83 = exp(s_10 - _S80);
        }
        (*(&kernelContext_0)->ps_0)[_S8] = _S83;
        float s_11 = (*(&kernelContext_0)->ss_0)[_S14];
        float _S84;
        if(((*(&kernelContext_0)->ss_0)[_S14]) == _S24)
        {
            _S84 = 0.0;
        }
        else
        {
            _S84 = exp(s_11 - _S80);
        }
        (*(&kernelContext_0)->ps_0)[_S10] = _S84;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S85 = l_0 * alpha_0;
        float _S86 = accLo_0[0U] * alpha_0;
        accLo_0[0U] = _S86;
        float _S87 = accHi_0[0U] * alpha_0;
        accHi_0[0U] = _S87;
        float _S88 = accLo_0[1U] * alpha_0;
        accLo_0[1U] = _S88;
        float _S89 = accHi_0[1U] * alpha_0;
        accHi_0[1U] = _S89;
        float _S90 = accLo_0[2U] * alpha_0;
        accLo_0[2U] = _S90;
        float _S91 = accHi_0[2U] * alpha_0;
        accHi_0[2U] = _S91;
        float _S92 = accLo_0[3U] * alpha_0;
        accLo_0[3U] = _S92;
        float _S93 = accHi_0[3U] * alpha_0;
        accHi_0[3U] = _S93;
        float pj_0 = (*(&kernelContext_0)->ps_0)[_S3];
        float _S94 = _S85 + (*(&kernelContext_0)->ps_0)[_S3];
        uint vw_0 = (*(&kernelContext_0)->vs_0)[kk_0];
        float _S95 = fma((*(&kernelContext_0)->ps_0)[_S3], attnLo_0((*(&kernelContext_0)->vs_0)[kk_0]), _S86);
        accLo_0[0U] = _S95;
        float _S96 = fma(pj_0, attnHi_0(vw_0), _S87);
        accHi_0[0U] = _S96;
        uint vw_1 = (*(&kernelContext_0)->vs_0)[_S5];
        float _S97 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S5]), _S88);
        accLo_0[1U] = _S97;
        float _S98 = fma(pj_0, attnHi_0(vw_1), _S89);
        accHi_0[1U] = _S98;
        uint vw_2 = (*(&kernelContext_0)->vs_0)[_S7];
        float _S99 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S7]), _S90);
        accLo_0[2U] = _S99;
        float _S100 = fma(pj_0, attnHi_0(vw_2), _S91);
        accHi_0[2U] = _S100;
        uint vw_3 = (*(&kernelContext_0)->vs_0)[_S9];
        float _S101 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S9]), _S92);
        accLo_0[3U] = _S101;
        float _S102 = fma(pj_0, attnHi_0(vw_3), _S93);
        accHi_0[3U] = _S102;
        float pj_1 = (*(&kernelContext_0)->ps_0)[_S49];
        float _S103 = _S94 + (*(&kernelContext_0)->ps_0)[_S49];
        uint _S104 = 33U + kk_0;
        uint vw_4 = (*(&kernelContext_0)->vs_0)[_S104];
        float _S105 = fma((*(&kernelContext_0)->ps_0)[_S49], attnLo_0((*(&kernelContext_0)->vs_0)[_S104]), _S95);
        accLo_0[0U] = _S105;
        float _S106 = fma(pj_1, attnHi_0(vw_4), _S96);
        accHi_0[0U] = _S106;
        uint vw_5 = (*(&kernelContext_0)->vs_0)[_S104 + 8U];
        float _S107 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S104 + 8U]), _S97);
        accLo_0[1U] = _S107;
        float _S108 = fma(pj_1, attnHi_0(vw_5), _S98);
        accHi_0[1U] = _S108;
        uint vw_6 = (*(&kernelContext_0)->vs_0)[_S104 + 16U];
        float _S109 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S104 + 16U]), _S99);
        accLo_0[2U] = _S109;
        float _S110 = fma(pj_1, attnHi_0(vw_6), _S100);
        accHi_0[2U] = _S110;
        uint vw_7 = (*(&kernelContext_0)->vs_0)[_S104 + 24U];
        float _S111 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S104 + 24U]), _S101);
        accLo_0[3U] = _S111;
        float _S112 = fma(pj_1, attnHi_0(vw_7), _S102);
        accHi_0[3U] = _S112;
        float pj_2 = (*(&kernelContext_0)->ps_0)[_S50];
        float _S113 = _S103 + (*(&kernelContext_0)->ps_0)[_S50];
        uint _S114 = 66U + kk_0;
        uint vw_8 = (*(&kernelContext_0)->vs_0)[_S114];
        float _S115 = fma((*(&kernelContext_0)->ps_0)[_S50], attnLo_0((*(&kernelContext_0)->vs_0)[_S114]), _S105);
        accLo_0[0U] = _S115;
        float _S116 = fma(pj_2, attnHi_0(vw_8), _S106);
        accHi_0[0U] = _S116;
        uint vw_9 = (*(&kernelContext_0)->vs_0)[_S114 + 8U];
        float _S117 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S114 + 8U]), _S107);
        accLo_0[1U] = _S117;
        float _S118 = fma(pj_2, attnHi_0(vw_9), _S108);
        accHi_0[1U] = _S118;
        uint vw_10 = (*(&kernelContext_0)->vs_0)[_S114 + 16U];
        float _S119 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S114 + 16U]), _S109);
        accLo_0[2U] = _S119;
        float _S120 = fma(pj_2, attnHi_0(vw_10), _S110);
        accHi_0[2U] = _S120;
        uint vw_11 = (*(&kernelContext_0)->vs_0)[_S114 + 24U];
        float _S121 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S114 + 24U]), _S111);
        accLo_0[3U] = _S121;
        float _S122 = fma(pj_2, attnHi_0(vw_11), _S112);
        accHi_0[3U] = _S122;
        float pj_3 = (*(&kernelContext_0)->ps_0)[_S51];
        float _S123 = _S113 + (*(&kernelContext_0)->ps_0)[_S51];
        uint _S124 = 99U + kk_0;
        uint vw_12 = (*(&kernelContext_0)->vs_0)[_S124];
        float _S125 = fma((*(&kernelContext_0)->ps_0)[_S51], attnLo_0((*(&kernelContext_0)->vs_0)[_S124]), _S115);
        accLo_0[0U] = _S125;
        float _S126 = fma(pj_3, attnHi_0(vw_12), _S116);
        accHi_0[0U] = _S126;
        uint vw_13 = (*(&kernelContext_0)->vs_0)[_S124 + 8U];
        float _S127 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S124 + 8U]), _S117);
        accLo_0[1U] = _S127;
        float _S128 = fma(pj_3, attnHi_0(vw_13), _S118);
        accHi_0[1U] = _S128;
        uint vw_14 = (*(&kernelContext_0)->vs_0)[_S124 + 16U];
        float _S129 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S124 + 16U]), _S119);
        accLo_0[2U] = _S129;
        float _S130 = fma(pj_3, attnHi_0(vw_14), _S120);
        accHi_0[2U] = _S130;
        uint vw_15 = (*(&kernelContext_0)->vs_0)[_S124 + 24U];
        float _S131 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S124 + 24U]), _S121);
        accLo_0[3U] = _S131;
        float _S132 = fma(pj_3, attnHi_0(vw_15), _S122);
        accHi_0[3U] = _S132;
        float pj_4 = (*(&kernelContext_0)->ps_0)[_S52];
        float _S133 = _S123 + (*(&kernelContext_0)->ps_0)[_S52];
        uint _S134 = 132U + kk_0;
        uint vw_16 = (*(&kernelContext_0)->vs_0)[_S134];
        float _S135 = fma((*(&kernelContext_0)->ps_0)[_S52], attnLo_0((*(&kernelContext_0)->vs_0)[_S134]), _S125);
        accLo_0[0U] = _S135;
        float _S136 = fma(pj_4, attnHi_0(vw_16), _S126);
        accHi_0[0U] = _S136;
        uint vw_17 = (*(&kernelContext_0)->vs_0)[_S134 + 8U];
        float _S137 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S134 + 8U]), _S127);
        accLo_0[1U] = _S137;
        float _S138 = fma(pj_4, attnHi_0(vw_17), _S128);
        accHi_0[1U] = _S138;
        uint vw_18 = (*(&kernelContext_0)->vs_0)[_S134 + 16U];
        float _S139 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S134 + 16U]), _S129);
        accLo_0[2U] = _S139;
        float _S140 = fma(pj_4, attnHi_0(vw_18), _S130);
        accHi_0[2U] = _S140;
        uint vw_19 = (*(&kernelContext_0)->vs_0)[_S134 + 24U];
        float _S141 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S134 + 24U]), _S131);
        accLo_0[3U] = _S141;
        float _S142 = fma(pj_4, attnHi_0(vw_19), _S132);
        accHi_0[3U] = _S142;
        float pj_5 = (*(&kernelContext_0)->ps_0)[_S53];
        float _S143 = _S133 + (*(&kernelContext_0)->ps_0)[_S53];
        uint _S144 = 165U + kk_0;
        uint vw_20 = (*(&kernelContext_0)->vs_0)[_S144];
        float _S145 = fma((*(&kernelContext_0)->ps_0)[_S53], attnLo_0((*(&kernelContext_0)->vs_0)[_S144]), _S135);
        accLo_0[0U] = _S145;
        float _S146 = fma(pj_5, attnHi_0(vw_20), _S136);
        accHi_0[0U] = _S146;
        uint vw_21 = (*(&kernelContext_0)->vs_0)[_S144 + 8U];
        float _S147 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S144 + 8U]), _S137);
        accLo_0[1U] = _S147;
        float _S148 = fma(pj_5, attnHi_0(vw_21), _S138);
        accHi_0[1U] = _S148;
        uint vw_22 = (*(&kernelContext_0)->vs_0)[_S144 + 16U];
        float _S149 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S144 + 16U]), _S139);
        accLo_0[2U] = _S149;
        float _S150 = fma(pj_5, attnHi_0(vw_22), _S140);
        accHi_0[2U] = _S150;
        uint vw_23 = (*(&kernelContext_0)->vs_0)[_S144 + 24U];
        float _S151 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S144 + 24U]), _S141);
        accLo_0[3U] = _S151;
        float _S152 = fma(pj_5, attnHi_0(vw_23), _S142);
        accHi_0[3U] = _S152;
        float pj_6 = (*(&kernelContext_0)->ps_0)[_S54];
        float _S153 = _S143 + (*(&kernelContext_0)->ps_0)[_S54];
        uint _S154 = 198U + kk_0;
        uint vw_24 = (*(&kernelContext_0)->vs_0)[_S154];
        float _S155 = fma((*(&kernelContext_0)->ps_0)[_S54], attnLo_0((*(&kernelContext_0)->vs_0)[_S154]), _S145);
        accLo_0[0U] = _S155;
        float _S156 = fma(pj_6, attnHi_0(vw_24), _S146);
        accHi_0[0U] = _S156;
        uint vw_25 = (*(&kernelContext_0)->vs_0)[_S154 + 8U];
        float _S157 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S154 + 8U]), _S147);
        accLo_0[1U] = _S157;
        float _S158 = fma(pj_6, attnHi_0(vw_25), _S148);
        accHi_0[1U] = _S158;
        uint vw_26 = (*(&kernelContext_0)->vs_0)[_S154 + 16U];
        float _S159 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S154 + 16U]), _S149);
        accLo_0[2U] = _S159;
        float _S160 = fma(pj_6, attnHi_0(vw_26), _S150);
        accHi_0[2U] = _S160;
        uint vw_27 = (*(&kernelContext_0)->vs_0)[_S154 + 24U];
        float _S161 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S154 + 24U]), _S151);
        accLo_0[3U] = _S161;
        float _S162 = fma(pj_6, attnHi_0(vw_27), _S152);
        accHi_0[3U] = _S162;
        float pj_7 = (*(&kernelContext_0)->ps_0)[_S55];
        float _S163 = _S153 + (*(&kernelContext_0)->ps_0)[_S55];
        uint _S164 = 231U + kk_0;
        uint vw_28 = (*(&kernelContext_0)->vs_0)[_S164];
        float _S165 = fma((*(&kernelContext_0)->ps_0)[_S55], attnLo_0((*(&kernelContext_0)->vs_0)[_S164]), _S155);
        accLo_0[0U] = _S165;
        float _S166 = fma(pj_7, attnHi_0(vw_28), _S156);
        accHi_0[0U] = _S166;
        uint vw_29 = (*(&kernelContext_0)->vs_0)[_S164 + 8U];
        float _S167 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S164 + 8U]), _S157);
        accLo_0[1U] = _S167;
        float _S168 = fma(pj_7, attnHi_0(vw_29), _S158);
        accHi_0[1U] = _S168;
        uint vw_30 = (*(&kernelContext_0)->vs_0)[_S164 + 16U];
        float _S169 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S164 + 16U]), _S159);
        accLo_0[2U] = _S169;
        float _S170 = fma(pj_7, attnHi_0(vw_30), _S160);
        accHi_0[2U] = _S170;
        uint vw_31 = (*(&kernelContext_0)->vs_0)[_S164 + 24U];
        float _S171 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S164 + 24U]), _S161);
        accLo_0[3U] = _S171;
        float _S172 = fma(pj_7, attnHi_0(vw_31), _S162);
        accHi_0[3U] = _S172;
        float pj_8 = (*(&kernelContext_0)->ps_0)[_S56];
        float _S173 = _S163 + (*(&kernelContext_0)->ps_0)[_S56];
        uint _S174 = 264U + kk_0;
        uint vw_32 = (*(&kernelContext_0)->vs_0)[_S174];
        float _S175 = fma((*(&kernelContext_0)->ps_0)[_S56], attnLo_0((*(&kernelContext_0)->vs_0)[_S174]), _S165);
        accLo_0[0U] = _S175;
        float _S176 = fma(pj_8, attnHi_0(vw_32), _S166);
        accHi_0[0U] = _S176;
        uint vw_33 = (*(&kernelContext_0)->vs_0)[_S174 + 8U];
        float _S177 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S174 + 8U]), _S167);
        accLo_0[1U] = _S177;
        float _S178 = fma(pj_8, attnHi_0(vw_33), _S168);
        accHi_0[1U] = _S178;
        uint vw_34 = (*(&kernelContext_0)->vs_0)[_S174 + 16U];
        float _S179 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S174 + 16U]), _S169);
        accLo_0[2U] = _S179;
        float _S180 = fma(pj_8, attnHi_0(vw_34), _S170);
        accHi_0[2U] = _S180;
        uint vw_35 = (*(&kernelContext_0)->vs_0)[_S174 + 24U];
        float _S181 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S174 + 24U]), _S171);
        accLo_0[3U] = _S181;
        float _S182 = fma(pj_8, attnHi_0(vw_35), _S172);
        accHi_0[3U] = _S182;
        float pj_9 = (*(&kernelContext_0)->ps_0)[_S57];
        float _S183 = _S173 + (*(&kernelContext_0)->ps_0)[_S57];
        uint _S184 = 297U + kk_0;
        uint vw_36 = (*(&kernelContext_0)->vs_0)[_S184];
        float _S185 = fma((*(&kernelContext_0)->ps_0)[_S57], attnLo_0((*(&kernelContext_0)->vs_0)[_S184]), _S175);
        accLo_0[0U] = _S185;
        float _S186 = fma(pj_9, attnHi_0(vw_36), _S176);
        accHi_0[0U] = _S186;
        uint vw_37 = (*(&kernelContext_0)->vs_0)[_S184 + 8U];
        float _S187 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S184 + 8U]), _S177);
        accLo_0[1U] = _S187;
        float _S188 = fma(pj_9, attnHi_0(vw_37), _S178);
        accHi_0[1U] = _S188;
        uint vw_38 = (*(&kernelContext_0)->vs_0)[_S184 + 16U];
        float _S189 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S184 + 16U]), _S179);
        accLo_0[2U] = _S189;
        float _S190 = fma(pj_9, attnHi_0(vw_38), _S180);
        accHi_0[2U] = _S190;
        uint vw_39 = (*(&kernelContext_0)->vs_0)[_S184 + 24U];
        float _S191 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S184 + 24U]), _S181);
        accLo_0[3U] = _S191;
        float _S192 = fma(pj_9, attnHi_0(vw_39), _S182);
        accHi_0[3U] = _S192;
        float pj_10 = (*(&kernelContext_0)->ps_0)[_S58];
        float _S193 = _S183 + (*(&kernelContext_0)->ps_0)[_S58];
        uint _S194 = 330U + kk_0;
        uint vw_40 = (*(&kernelContext_0)->vs_0)[_S194];
        float _S195 = fma((*(&kernelContext_0)->ps_0)[_S58], attnLo_0((*(&kernelContext_0)->vs_0)[_S194]), _S185);
        accLo_0[0U] = _S195;
        float _S196 = fma(pj_10, attnHi_0(vw_40), _S186);
        accHi_0[0U] = _S196;
        uint vw_41 = (*(&kernelContext_0)->vs_0)[_S194 + 8U];
        float _S197 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S194 + 8U]), _S187);
        accLo_0[1U] = _S197;
        float _S198 = fma(pj_10, attnHi_0(vw_41), _S188);
        accHi_0[1U] = _S198;
        uint vw_42 = (*(&kernelContext_0)->vs_0)[_S194 + 16U];
        float _S199 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S194 + 16U]), _S189);
        accLo_0[2U] = _S199;
        float _S200 = fma(pj_10, attnHi_0(vw_42), _S190);
        accHi_0[2U] = _S200;
        uint vw_43 = (*(&kernelContext_0)->vs_0)[_S194 + 24U];
        float _S201 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S194 + 24U]), _S191);
        accLo_0[3U] = _S201;
        float _S202 = fma(pj_10, attnHi_0(vw_43), _S192);
        accHi_0[3U] = _S202;
        float pj_11 = (*(&kernelContext_0)->ps_0)[_S59];
        float _S203 = _S193 + (*(&kernelContext_0)->ps_0)[_S59];
        uint _S204 = 363U + kk_0;
        uint vw_44 = (*(&kernelContext_0)->vs_0)[_S204];
        float _S205 = fma((*(&kernelContext_0)->ps_0)[_S59], attnLo_0((*(&kernelContext_0)->vs_0)[_S204]), _S195);
        accLo_0[0U] = _S205;
        float _S206 = fma(pj_11, attnHi_0(vw_44), _S196);
        accHi_0[0U] = _S206;
        uint vw_45 = (*(&kernelContext_0)->vs_0)[_S204 + 8U];
        float _S207 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S204 + 8U]), _S197);
        accLo_0[1U] = _S207;
        float _S208 = fma(pj_11, attnHi_0(vw_45), _S198);
        accHi_0[1U] = _S208;
        uint vw_46 = (*(&kernelContext_0)->vs_0)[_S204 + 16U];
        float _S209 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S204 + 16U]), _S199);
        accLo_0[2U] = _S209;
        float _S210 = fma(pj_11, attnHi_0(vw_46), _S200);
        accHi_0[2U] = _S210;
        uint vw_47 = (*(&kernelContext_0)->vs_0)[_S204 + 24U];
        float _S211 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S204 + 24U]), _S201);
        accLo_0[3U] = _S211;
        float _S212 = fma(pj_11, attnHi_0(vw_47), _S202);
        accHi_0[3U] = _S212;
        float pj_12 = (*(&kernelContext_0)->ps_0)[_S60];
        float _S213 = _S203 + (*(&kernelContext_0)->ps_0)[_S60];
        uint _S214 = 396U + kk_0;
        uint vw_48 = (*(&kernelContext_0)->vs_0)[_S214];
        float _S215 = fma((*(&kernelContext_0)->ps_0)[_S60], attnLo_0((*(&kernelContext_0)->vs_0)[_S214]), _S205);
        accLo_0[0U] = _S215;
        float _S216 = fma(pj_12, attnHi_0(vw_48), _S206);
        accHi_0[0U] = _S216;
        uint vw_49 = (*(&kernelContext_0)->vs_0)[_S214 + 8U];
        float _S217 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S214 + 8U]), _S207);
        accLo_0[1U] = _S217;
        float _S218 = fma(pj_12, attnHi_0(vw_49), _S208);
        accHi_0[1U] = _S218;
        uint vw_50 = (*(&kernelContext_0)->vs_0)[_S214 + 16U];
        float _S219 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S214 + 16U]), _S209);
        accLo_0[2U] = _S219;
        float _S220 = fma(pj_12, attnHi_0(vw_50), _S210);
        accHi_0[2U] = _S220;
        uint vw_51 = (*(&kernelContext_0)->vs_0)[_S214 + 24U];
        float _S221 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S214 + 24U]), _S211);
        accLo_0[3U] = _S221;
        float _S222 = fma(pj_12, attnHi_0(vw_51), _S212);
        accHi_0[3U] = _S222;
        float pj_13 = (*(&kernelContext_0)->ps_0)[_S61];
        float _S223 = _S213 + (*(&kernelContext_0)->ps_0)[_S61];
        uint _S224 = 429U + kk_0;
        uint vw_52 = (*(&kernelContext_0)->vs_0)[_S224];
        float _S225 = fma((*(&kernelContext_0)->ps_0)[_S61], attnLo_0((*(&kernelContext_0)->vs_0)[_S224]), _S215);
        accLo_0[0U] = _S225;
        float _S226 = fma(pj_13, attnHi_0(vw_52), _S216);
        accHi_0[0U] = _S226;
        uint vw_53 = (*(&kernelContext_0)->vs_0)[_S224 + 8U];
        float _S227 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S224 + 8U]), _S217);
        accLo_0[1U] = _S227;
        float _S228 = fma(pj_13, attnHi_0(vw_53), _S218);
        accHi_0[1U] = _S228;
        uint vw_54 = (*(&kernelContext_0)->vs_0)[_S224 + 16U];
        float _S229 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S224 + 16U]), _S219);
        accLo_0[2U] = _S229;
        float _S230 = fma(pj_13, attnHi_0(vw_54), _S220);
        accHi_0[2U] = _S230;
        uint vw_55 = (*(&kernelContext_0)->vs_0)[_S224 + 24U];
        float _S231 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S224 + 24U]), _S221);
        accLo_0[3U] = _S231;
        float _S232 = fma(pj_13, attnHi_0(vw_55), _S222);
        accHi_0[3U] = _S232;
        float pj_14 = (*(&kernelContext_0)->ps_0)[_S62];
        float _S233 = _S223 + (*(&kernelContext_0)->ps_0)[_S62];
        uint _S234 = 462U + kk_0;
        uint vw_56 = (*(&kernelContext_0)->vs_0)[_S234];
        float _S235 = fma((*(&kernelContext_0)->ps_0)[_S62], attnLo_0((*(&kernelContext_0)->vs_0)[_S234]), _S225);
        accLo_0[0U] = _S235;
        float _S236 = fma(pj_14, attnHi_0(vw_56), _S226);
        accHi_0[0U] = _S236;
        uint vw_57 = (*(&kernelContext_0)->vs_0)[_S234 + 8U];
        float _S237 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S234 + 8U]), _S227);
        accLo_0[1U] = _S237;
        float _S238 = fma(pj_14, attnHi_0(vw_57), _S228);
        accHi_0[1U] = _S238;
        uint vw_58 = (*(&kernelContext_0)->vs_0)[_S234 + 16U];
        float _S239 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S234 + 16U]), _S229);
        accLo_0[2U] = _S239;
        float _S240 = fma(pj_14, attnHi_0(vw_58), _S230);
        accHi_0[2U] = _S240;
        uint vw_59 = (*(&kernelContext_0)->vs_0)[_S234 + 24U];
        float _S241 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S234 + 24U]), _S231);
        accLo_0[3U] = _S241;
        float _S242 = fma(pj_14, attnHi_0(vw_59), _S232);
        accHi_0[3U] = _S242;
        float pj_15 = (*(&kernelContext_0)->ps_0)[_S63];
        float _S243 = _S233 + (*(&kernelContext_0)->ps_0)[_S63];
        uint _S244 = 495U + kk_0;
        uint vw_60 = (*(&kernelContext_0)->vs_0)[_S244];
        float _S245 = fma((*(&kernelContext_0)->ps_0)[_S63], attnLo_0((*(&kernelContext_0)->vs_0)[_S244]), _S235);
        accLo_0[0U] = _S245;
        float _S246 = fma(pj_15, attnHi_0(vw_60), _S236);
        accHi_0[0U] = _S246;
        uint vw_61 = (*(&kernelContext_0)->vs_0)[_S244 + 8U];
        float _S247 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S244 + 8U]), _S237);
        accLo_0[1U] = _S247;
        float _S248 = fma(pj_15, attnHi_0(vw_61), _S238);
        accHi_0[1U] = _S248;
        uint vw_62 = (*(&kernelContext_0)->vs_0)[_S244 + 16U];
        float _S249 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S244 + 16U]), _S239);
        accLo_0[2U] = _S249;
        float _S250 = fma(pj_15, attnHi_0(vw_62), _S240);
        accHi_0[2U] = _S250;
        uint vw_63 = (*(&kernelContext_0)->vs_0)[_S244 + 24U];
        float _S251 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S244 + 24U]), _S241);
        accLo_0[3U] = _S251;
        float _S252 = fma(pj_15, attnHi_0(vw_63), _S242);
        accHi_0[3U] = _S252;
        float pj_16 = (*(&kernelContext_0)->ps_0)[_S64];
        float _S253 = _S243 + (*(&kernelContext_0)->ps_0)[_S64];
        uint _S254 = 528U + kk_0;
        uint vw_64 = (*(&kernelContext_0)->vs_0)[_S254];
        float _S255 = fma((*(&kernelContext_0)->ps_0)[_S64], attnLo_0((*(&kernelContext_0)->vs_0)[_S254]), _S245);
        accLo_0[0U] = _S255;
        float _S256 = fma(pj_16, attnHi_0(vw_64), _S246);
        accHi_0[0U] = _S256;
        uint vw_65 = (*(&kernelContext_0)->vs_0)[_S254 + 8U];
        float _S257 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S254 + 8U]), _S247);
        accLo_0[1U] = _S257;
        float _S258 = fma(pj_16, attnHi_0(vw_65), _S248);
        accHi_0[1U] = _S258;
        uint vw_66 = (*(&kernelContext_0)->vs_0)[_S254 + 16U];
        float _S259 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S254 + 16U]), _S249);
        accLo_0[2U] = _S259;
        float _S260 = fma(pj_16, attnHi_0(vw_66), _S250);
        accHi_0[2U] = _S260;
        uint vw_67 = (*(&kernelContext_0)->vs_0)[_S254 + 24U];
        float _S261 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S254 + 24U]), _S251);
        accLo_0[3U] = _S261;
        float _S262 = fma(pj_16, attnHi_0(vw_67), _S252);
        accHi_0[3U] = _S262;
        float pj_17 = (*(&kernelContext_0)->ps_0)[_S65];
        float _S263 = _S253 + (*(&kernelContext_0)->ps_0)[_S65];
        uint _S264 = 561U + kk_0;
        uint vw_68 = (*(&kernelContext_0)->vs_0)[_S264];
        float _S265 = fma((*(&kernelContext_0)->ps_0)[_S65], attnLo_0((*(&kernelContext_0)->vs_0)[_S264]), _S255);
        accLo_0[0U] = _S265;
        float _S266 = fma(pj_17, attnHi_0(vw_68), _S256);
        accHi_0[0U] = _S266;
        uint vw_69 = (*(&kernelContext_0)->vs_0)[_S264 + 8U];
        float _S267 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S264 + 8U]), _S257);
        accLo_0[1U] = _S267;
        float _S268 = fma(pj_17, attnHi_0(vw_69), _S258);
        accHi_0[1U] = _S268;
        uint vw_70 = (*(&kernelContext_0)->vs_0)[_S264 + 16U];
        float _S269 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S264 + 16U]), _S259);
        accLo_0[2U] = _S269;
        float _S270 = fma(pj_17, attnHi_0(vw_70), _S260);
        accHi_0[2U] = _S270;
        uint vw_71 = (*(&kernelContext_0)->vs_0)[_S264 + 24U];
        float _S271 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S264 + 24U]), _S261);
        accLo_0[3U] = _S271;
        float _S272 = fma(pj_17, attnHi_0(vw_71), _S262);
        accHi_0[3U] = _S272;
        float pj_18 = (*(&kernelContext_0)->ps_0)[_S66];
        float _S273 = _S263 + (*(&kernelContext_0)->ps_0)[_S66];
        uint _S274 = 594U + kk_0;
        uint vw_72 = (*(&kernelContext_0)->vs_0)[_S274];
        float _S275 = fma((*(&kernelContext_0)->ps_0)[_S66], attnLo_0((*(&kernelContext_0)->vs_0)[_S274]), _S265);
        accLo_0[0U] = _S275;
        float _S276 = fma(pj_18, attnHi_0(vw_72), _S266);
        accHi_0[0U] = _S276;
        uint vw_73 = (*(&kernelContext_0)->vs_0)[_S274 + 8U];
        float _S277 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S274 + 8U]), _S267);
        accLo_0[1U] = _S277;
        float _S278 = fma(pj_18, attnHi_0(vw_73), _S268);
        accHi_0[1U] = _S278;
        uint vw_74 = (*(&kernelContext_0)->vs_0)[_S274 + 16U];
        float _S279 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S274 + 16U]), _S269);
        accLo_0[2U] = _S279;
        float _S280 = fma(pj_18, attnHi_0(vw_74), _S270);
        accHi_0[2U] = _S280;
        uint vw_75 = (*(&kernelContext_0)->vs_0)[_S274 + 24U];
        float _S281 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S274 + 24U]), _S271);
        accLo_0[3U] = _S281;
        float _S282 = fma(pj_18, attnHi_0(vw_75), _S272);
        accHi_0[3U] = _S282;
        float pj_19 = (*(&kernelContext_0)->ps_0)[_S67];
        float _S283 = _S273 + (*(&kernelContext_0)->ps_0)[_S67];
        uint _S284 = 627U + kk_0;
        uint vw_76 = (*(&kernelContext_0)->vs_0)[_S284];
        float _S285 = fma((*(&kernelContext_0)->ps_0)[_S67], attnLo_0((*(&kernelContext_0)->vs_0)[_S284]), _S275);
        accLo_0[0U] = _S285;
        float _S286 = fma(pj_19, attnHi_0(vw_76), _S276);
        accHi_0[0U] = _S286;
        uint vw_77 = (*(&kernelContext_0)->vs_0)[_S284 + 8U];
        float _S287 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S284 + 8U]), _S277);
        accLo_0[1U] = _S287;
        float _S288 = fma(pj_19, attnHi_0(vw_77), _S278);
        accHi_0[1U] = _S288;
        uint vw_78 = (*(&kernelContext_0)->vs_0)[_S284 + 16U];
        float _S289 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S284 + 16U]), _S279);
        accLo_0[2U] = _S289;
        float _S290 = fma(pj_19, attnHi_0(vw_78), _S280);
        accHi_0[2U] = _S290;
        uint vw_79 = (*(&kernelContext_0)->vs_0)[_S284 + 24U];
        float _S291 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S284 + 24U]), _S281);
        accLo_0[3U] = _S291;
        float _S292 = fma(pj_19, attnHi_0(vw_79), _S282);
        accHi_0[3U] = _S292;
        float pj_20 = (*(&kernelContext_0)->ps_0)[_S68];
        float _S293 = _S283 + (*(&kernelContext_0)->ps_0)[_S68];
        uint _S294 = 660U + kk_0;
        uint vw_80 = (*(&kernelContext_0)->vs_0)[_S294];
        float _S295 = fma((*(&kernelContext_0)->ps_0)[_S68], attnLo_0((*(&kernelContext_0)->vs_0)[_S294]), _S285);
        accLo_0[0U] = _S295;
        float _S296 = fma(pj_20, attnHi_0(vw_80), _S286);
        accHi_0[0U] = _S296;
        uint vw_81 = (*(&kernelContext_0)->vs_0)[_S294 + 8U];
        float _S297 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S294 + 8U]), _S287);
        accLo_0[1U] = _S297;
        float _S298 = fma(pj_20, attnHi_0(vw_81), _S288);
        accHi_0[1U] = _S298;
        uint vw_82 = (*(&kernelContext_0)->vs_0)[_S294 + 16U];
        float _S299 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S294 + 16U]), _S289);
        accLo_0[2U] = _S299;
        float _S300 = fma(pj_20, attnHi_0(vw_82), _S290);
        accHi_0[2U] = _S300;
        uint vw_83 = (*(&kernelContext_0)->vs_0)[_S294 + 24U];
        float _S301 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S294 + 24U]), _S291);
        accLo_0[3U] = _S301;
        float _S302 = fma(pj_20, attnHi_0(vw_83), _S292);
        accHi_0[3U] = _S302;
        float pj_21 = (*(&kernelContext_0)->ps_0)[_S69];
        float _S303 = _S293 + (*(&kernelContext_0)->ps_0)[_S69];
        uint _S304 = 693U + kk_0;
        uint vw_84 = (*(&kernelContext_0)->vs_0)[_S304];
        float _S305 = fma((*(&kernelContext_0)->ps_0)[_S69], attnLo_0((*(&kernelContext_0)->vs_0)[_S304]), _S295);
        accLo_0[0U] = _S305;
        float _S306 = fma(pj_21, attnHi_0(vw_84), _S296);
        accHi_0[0U] = _S306;
        uint vw_85 = (*(&kernelContext_0)->vs_0)[_S304 + 8U];
        float _S307 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S304 + 8U]), _S297);
        accLo_0[1U] = _S307;
        float _S308 = fma(pj_21, attnHi_0(vw_85), _S298);
        accHi_0[1U] = _S308;
        uint vw_86 = (*(&kernelContext_0)->vs_0)[_S304 + 16U];
        float _S309 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S304 + 16U]), _S299);
        accLo_0[2U] = _S309;
        float _S310 = fma(pj_21, attnHi_0(vw_86), _S300);
        accHi_0[2U] = _S310;
        uint vw_87 = (*(&kernelContext_0)->vs_0)[_S304 + 24U];
        float _S311 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S304 + 24U]), _S301);
        accLo_0[3U] = _S311;
        float _S312 = fma(pj_21, attnHi_0(vw_87), _S302);
        accHi_0[3U] = _S312;
        float pj_22 = (*(&kernelContext_0)->ps_0)[_S70];
        float _S313 = _S303 + (*(&kernelContext_0)->ps_0)[_S70];
        uint _S314 = 726U + kk_0;
        uint vw_88 = (*(&kernelContext_0)->vs_0)[_S314];
        float _S315 = fma((*(&kernelContext_0)->ps_0)[_S70], attnLo_0((*(&kernelContext_0)->vs_0)[_S314]), _S305);
        accLo_0[0U] = _S315;
        float _S316 = fma(pj_22, attnHi_0(vw_88), _S306);
        accHi_0[0U] = _S316;
        uint vw_89 = (*(&kernelContext_0)->vs_0)[_S314 + 8U];
        float _S317 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S314 + 8U]), _S307);
        accLo_0[1U] = _S317;
        float _S318 = fma(pj_22, attnHi_0(vw_89), _S308);
        accHi_0[1U] = _S318;
        uint vw_90 = (*(&kernelContext_0)->vs_0)[_S314 + 16U];
        float _S319 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S314 + 16U]), _S309);
        accLo_0[2U] = _S319;
        float _S320 = fma(pj_22, attnHi_0(vw_90), _S310);
        accHi_0[2U] = _S320;
        uint vw_91 = (*(&kernelContext_0)->vs_0)[_S314 + 24U];
        float _S321 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S314 + 24U]), _S311);
        accLo_0[3U] = _S321;
        float _S322 = fma(pj_22, attnHi_0(vw_91), _S312);
        accHi_0[3U] = _S322;
        float pj_23 = (*(&kernelContext_0)->ps_0)[_S71];
        float _S323 = _S313 + (*(&kernelContext_0)->ps_0)[_S71];
        uint _S324 = 759U + kk_0;
        uint vw_92 = (*(&kernelContext_0)->vs_0)[_S324];
        float _S325 = fma((*(&kernelContext_0)->ps_0)[_S71], attnLo_0((*(&kernelContext_0)->vs_0)[_S324]), _S315);
        accLo_0[0U] = _S325;
        float _S326 = fma(pj_23, attnHi_0(vw_92), _S316);
        accHi_0[0U] = _S326;
        uint vw_93 = (*(&kernelContext_0)->vs_0)[_S324 + 8U];
        float _S327 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S324 + 8U]), _S317);
        accLo_0[1U] = _S327;
        float _S328 = fma(pj_23, attnHi_0(vw_93), _S318);
        accHi_0[1U] = _S328;
        uint vw_94 = (*(&kernelContext_0)->vs_0)[_S324 + 16U];
        float _S329 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S324 + 16U]), _S319);
        accLo_0[2U] = _S329;
        float _S330 = fma(pj_23, attnHi_0(vw_94), _S320);
        accHi_0[2U] = _S330;
        uint vw_95 = (*(&kernelContext_0)->vs_0)[_S324 + 24U];
        float _S331 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S324 + 24U]), _S321);
        accLo_0[3U] = _S331;
        float _S332 = fma(pj_23, attnHi_0(vw_95), _S322);
        accHi_0[3U] = _S332;
        float pj_24 = (*(&kernelContext_0)->ps_0)[_S72];
        float _S333 = _S323 + (*(&kernelContext_0)->ps_0)[_S72];
        uint _S334 = 792U + kk_0;
        uint vw_96 = (*(&kernelContext_0)->vs_0)[_S334];
        float _S335 = fma((*(&kernelContext_0)->ps_0)[_S72], attnLo_0((*(&kernelContext_0)->vs_0)[_S334]), _S325);
        accLo_0[0U] = _S335;
        float _S336 = fma(pj_24, attnHi_0(vw_96), _S326);
        accHi_0[0U] = _S336;
        uint vw_97 = (*(&kernelContext_0)->vs_0)[_S334 + 8U];
        float _S337 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S334 + 8U]), _S327);
        accLo_0[1U] = _S337;
        float _S338 = fma(pj_24, attnHi_0(vw_97), _S328);
        accHi_0[1U] = _S338;
        uint vw_98 = (*(&kernelContext_0)->vs_0)[_S334 + 16U];
        float _S339 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S334 + 16U]), _S329);
        accLo_0[2U] = _S339;
        float _S340 = fma(pj_24, attnHi_0(vw_98), _S330);
        accHi_0[2U] = _S340;
        uint vw_99 = (*(&kernelContext_0)->vs_0)[_S334 + 24U];
        float _S341 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S334 + 24U]), _S331);
        accLo_0[3U] = _S341;
        float _S342 = fma(pj_24, attnHi_0(vw_99), _S332);
        accHi_0[3U] = _S342;
        float pj_25 = (*(&kernelContext_0)->ps_0)[_S73];
        float _S343 = _S333 + (*(&kernelContext_0)->ps_0)[_S73];
        uint _S344 = 825U + kk_0;
        uint vw_100 = (*(&kernelContext_0)->vs_0)[_S344];
        float _S345 = fma((*(&kernelContext_0)->ps_0)[_S73], attnLo_0((*(&kernelContext_0)->vs_0)[_S344]), _S335);
        accLo_0[0U] = _S345;
        float _S346 = fma(pj_25, attnHi_0(vw_100), _S336);
        accHi_0[0U] = _S346;
        uint vw_101 = (*(&kernelContext_0)->vs_0)[_S344 + 8U];
        float _S347 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S344 + 8U]), _S337);
        accLo_0[1U] = _S347;
        float _S348 = fma(pj_25, attnHi_0(vw_101), _S338);
        accHi_0[1U] = _S348;
        uint vw_102 = (*(&kernelContext_0)->vs_0)[_S344 + 16U];
        float _S349 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S344 + 16U]), _S339);
        accLo_0[2U] = _S349;
        float _S350 = fma(pj_25, attnHi_0(vw_102), _S340);
        accHi_0[2U] = _S350;
        uint vw_103 = (*(&kernelContext_0)->vs_0)[_S344 + 24U];
        float _S351 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S344 + 24U]), _S341);
        accLo_0[3U] = _S351;
        float _S352 = fma(pj_25, attnHi_0(vw_103), _S342);
        accHi_0[3U] = _S352;
        float pj_26 = (*(&kernelContext_0)->ps_0)[_S74];
        float _S353 = _S343 + (*(&kernelContext_0)->ps_0)[_S74];
        uint _S354 = 858U + kk_0;
        uint vw_104 = (*(&kernelContext_0)->vs_0)[_S354];
        float _S355 = fma((*(&kernelContext_0)->ps_0)[_S74], attnLo_0((*(&kernelContext_0)->vs_0)[_S354]), _S345);
        accLo_0[0U] = _S355;
        float _S356 = fma(pj_26, attnHi_0(vw_104), _S346);
        accHi_0[0U] = _S356;
        uint vw_105 = (*(&kernelContext_0)->vs_0)[_S354 + 8U];
        float _S357 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S354 + 8U]), _S347);
        accLo_0[1U] = _S357;
        float _S358 = fma(pj_26, attnHi_0(vw_105), _S348);
        accHi_0[1U] = _S358;
        uint vw_106 = (*(&kernelContext_0)->vs_0)[_S354 + 16U];
        float _S359 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S354 + 16U]), _S349);
        accLo_0[2U] = _S359;
        float _S360 = fma(pj_26, attnHi_0(vw_106), _S350);
        accHi_0[2U] = _S360;
        uint vw_107 = (*(&kernelContext_0)->vs_0)[_S354 + 24U];
        float _S361 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S354 + 24U]), _S351);
        accLo_0[3U] = _S361;
        float _S362 = fma(pj_26, attnHi_0(vw_107), _S352);
        accHi_0[3U] = _S362;
        float pj_27 = (*(&kernelContext_0)->ps_0)[_S75];
        float _S363 = _S353 + (*(&kernelContext_0)->ps_0)[_S75];
        uint _S364 = 891U + kk_0;
        uint vw_108 = (*(&kernelContext_0)->vs_0)[_S364];
        float _S365 = fma((*(&kernelContext_0)->ps_0)[_S75], attnLo_0((*(&kernelContext_0)->vs_0)[_S364]), _S355);
        accLo_0[0U] = _S365;
        float _S366 = fma(pj_27, attnHi_0(vw_108), _S356);
        accHi_0[0U] = _S366;
        uint vw_109 = (*(&kernelContext_0)->vs_0)[_S364 + 8U];
        float _S367 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S364 + 8U]), _S357);
        accLo_0[1U] = _S367;
        float _S368 = fma(pj_27, attnHi_0(vw_109), _S358);
        accHi_0[1U] = _S368;
        uint vw_110 = (*(&kernelContext_0)->vs_0)[_S364 + 16U];
        float _S369 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S364 + 16U]), _S359);
        accLo_0[2U] = _S369;
        float _S370 = fma(pj_27, attnHi_0(vw_110), _S360);
        accHi_0[2U] = _S370;
        uint vw_111 = (*(&kernelContext_0)->vs_0)[_S364 + 24U];
        float _S371 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S364 + 24U]), _S361);
        accLo_0[3U] = _S371;
        float _S372 = fma(pj_27, attnHi_0(vw_111), _S362);
        accHi_0[3U] = _S372;
        float pj_28 = (*(&kernelContext_0)->ps_0)[_S76];
        float _S373 = _S363 + (*(&kernelContext_0)->ps_0)[_S76];
        uint _S374 = 924U + kk_0;
        uint vw_112 = (*(&kernelContext_0)->vs_0)[_S374];
        float _S375 = fma((*(&kernelContext_0)->ps_0)[_S76], attnLo_0((*(&kernelContext_0)->vs_0)[_S374]), _S365);
        accLo_0[0U] = _S375;
        float _S376 = fma(pj_28, attnHi_0(vw_112), _S366);
        accHi_0[0U] = _S376;
        uint vw_113 = (*(&kernelContext_0)->vs_0)[_S374 + 8U];
        float _S377 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S374 + 8U]), _S367);
        accLo_0[1U] = _S377;
        float _S378 = fma(pj_28, attnHi_0(vw_113), _S368);
        accHi_0[1U] = _S378;
        uint vw_114 = (*(&kernelContext_0)->vs_0)[_S374 + 16U];
        float _S379 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S374 + 16U]), _S369);
        accLo_0[2U] = _S379;
        float _S380 = fma(pj_28, attnHi_0(vw_114), _S370);
        accHi_0[2U] = _S380;
        uint vw_115 = (*(&kernelContext_0)->vs_0)[_S374 + 24U];
        float _S381 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S374 + 24U]), _S371);
        accLo_0[3U] = _S381;
        float _S382 = fma(pj_28, attnHi_0(vw_115), _S372);
        accHi_0[3U] = _S382;
        float pj_29 = (*(&kernelContext_0)->ps_0)[_S77];
        float _S383 = _S373 + (*(&kernelContext_0)->ps_0)[_S77];
        uint _S384 = 957U + kk_0;
        uint vw_116 = (*(&kernelContext_0)->vs_0)[_S384];
        float _S385 = fma((*(&kernelContext_0)->ps_0)[_S77], attnLo_0((*(&kernelContext_0)->vs_0)[_S384]), _S375);
        accLo_0[0U] = _S385;
        float _S386 = fma(pj_29, attnHi_0(vw_116), _S376);
        accHi_0[0U] = _S386;
        uint vw_117 = (*(&kernelContext_0)->vs_0)[_S384 + 8U];
        float _S387 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S384 + 8U]), _S377);
        accLo_0[1U] = _S387;
        float _S388 = fma(pj_29, attnHi_0(vw_117), _S378);
        accHi_0[1U] = _S388;
        uint vw_118 = (*(&kernelContext_0)->vs_0)[_S384 + 16U];
        float _S389 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S384 + 16U]), _S379);
        accLo_0[2U] = _S389;
        float _S390 = fma(pj_29, attnHi_0(vw_118), _S380);
        accHi_0[2U] = _S390;
        uint vw_119 = (*(&kernelContext_0)->vs_0)[_S384 + 24U];
        float _S391 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S384 + 24U]), _S381);
        accLo_0[3U] = _S391;
        float _S392 = fma(pj_29, attnHi_0(vw_119), _S382);
        accHi_0[3U] = _S392;
        float pj_30 = (*(&kernelContext_0)->ps_0)[_S78];
        float _S393 = _S383 + (*(&kernelContext_0)->ps_0)[_S78];
        uint _S394 = 990U + kk_0;
        uint vw_120 = (*(&kernelContext_0)->vs_0)[_S394];
        float _S395 = fma((*(&kernelContext_0)->ps_0)[_S78], attnLo_0((*(&kernelContext_0)->vs_0)[_S394]), _S385);
        accLo_0[0U] = _S395;
        float _S396 = fma(pj_30, attnHi_0(vw_120), _S386);
        accHi_0[0U] = _S396;
        uint vw_121 = (*(&kernelContext_0)->vs_0)[_S394 + 8U];
        float _S397 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S394 + 8U]), _S387);
        accLo_0[1U] = _S397;
        float _S398 = fma(pj_30, attnHi_0(vw_121), _S388);
        accHi_0[1U] = _S398;
        uint vw_122 = (*(&kernelContext_0)->vs_0)[_S394 + 16U];
        float _S399 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S394 + 16U]), _S389);
        accLo_0[2U] = _S399;
        float _S400 = fma(pj_30, attnHi_0(vw_122), _S390);
        accHi_0[2U] = _S400;
        uint vw_123 = (*(&kernelContext_0)->vs_0)[_S394 + 24U];
        float _S401 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S394 + 24U]), _S391);
        accLo_0[3U] = _S401;
        float _S402 = fma(pj_30, attnHi_0(vw_123), _S392);
        accHi_0[3U] = _S402;
        float pj_31 = (*(&kernelContext_0)->ps_0)[_S79];
        float l_1 = _S393 + (*(&kernelContext_0)->ps_0)[_S79];
        uint _S403 = 1023U + kk_0;
        uint vw_124 = (*(&kernelContext_0)->vs_0)[_S403];
        accLo_0[0U] = fma((*(&kernelContext_0)->ps_0)[_S79], attnLo_0((*(&kernelContext_0)->vs_0)[_S403]), _S395);
        accHi_0[0U] = fma(pj_31, attnHi_0(vw_124), _S396);
        uint vw_125 = (*(&kernelContext_0)->vs_0)[_S403 + 8U];
        accLo_0[1U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S403 + 8U]), _S397);
        accHi_0[1U] = fma(pj_31, attnHi_0(vw_125), _S398);
        uint vw_126 = (*(&kernelContext_0)->vs_0)[_S403 + 16U];
        accLo_0[2U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S403 + 16U]), _S399);
        accHi_0[2U] = fma(pj_31, attnHi_0(vw_126), _S400);
        uint vw_127 = (*(&kernelContext_0)->vs_0)[_S403 + 24U];
        accLo_0[3U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S403 + 24U]), _S401);
        accHi_0[3U] = fma(pj_31, attnHi_0(vw_127), _S402);
        b_0 = b_0 + 32U;
        m_0 = _S80;
        l_0 = l_1;
    }
    if(!live_0)
    {
        return;
    }
    uint row_1 = head_0 * (&kernelContext_0)->p_0->queries_0 + qi_0;
    if(((&kernelContext_0)->p_0->splits_0) > 1U)
    {
        uint pr_30 = split_0 * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0 + row_1;
        if(kk_0 == 0U)
        {
            uint _S404 = 2U * pr_30;
            *((&kernelContext_0)->partialML_0+_S404) = m_0;
            *((&kernelContext_0)->partialML_0+(_S404 + 1U)) = l_0;
        }
        if(kk_0 < _S18)
        {
            uint _S405 = 2U * kk_0;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S405)) = accLo_0[0U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S405 + 1U)) = accHi_0[0U];
        }
        uint pair_0 = kk_0 + 8U;
        if(pair_0 < _S18)
        {
            uint _S406 = 2U * pair_0;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S406)) = accLo_0[1U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S406 + 1U)) = accHi_0[1U];
        }
        uint pair_1 = kk_0 + 16U;
        if(pair_1 < _S18)
        {
            uint _S407 = 2U * pair_1;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S407)) = accLo_0[2U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S407 + 1U)) = accHi_0[2U];
        }
        uint pair_2 = kk_0 + 24U;
        if(pair_2 < _S18)
        {
            uint _S408 = 2U * pair_2;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S408)) = accLo_0[3U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S408 + 1U)) = accHi_0[3U];
        }
        return;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                if(kk_0 >= _S18)
                {
                    break;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S18 + kk_0)) = attnPack_0(accLo_0[0U] / l_0, accHi_0[0U] / l_0);
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_3 = kk_0 + 8U;
                if(pair_3 >= _S18)
                {
                    break;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S18 + pair_3)) = attnPack_0(accLo_0[1U] / l_0, accHi_0[1U] / l_0);
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_4 = kk_0 + 16U;
                if(pair_4 >= _S18)
                {
                    break;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S18 + pair_4)) = attnPack_0(accLo_0[2U] / l_0, accHi_0[2U] / l_0);
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_5 = kk_0 + 24U;
                if(pair_5 >= _S18)
                {
                    break;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S18 + pair_5)) = attnPack_0(accLo_0[3U] / l_0, accHi_0[3U] / l_0);
                break;
            }
            break;
        }
        break;
    }
    return;
}

