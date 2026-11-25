#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
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
    uint device* q_0;
    uint device* lens_1;
    uint device* table_0;
    uint device* k_0;
    uint device* v_0;
    float device* partials_0;
    float device* lse_0;
    uint device* o_0;
    array<uint, int(1040)> threadgroup* qs_0;
    array<uint, int(2080)> threadgroup* ks_0;
    array<uint, int(2080)> threadgroup* vs_0;
    array<float, int(512)> threadgroup* ss_0;
    array<float, int(512)> threadgroup* ps_0;
};

[[kernel]] void attn_flash_128(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(8)]], uint device* q_1 [[buffer(0)]], uint device* lens_2 [[buffer(7)]], uint device* table_1 [[buffer(5)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partials_1 [[buffer(4)]], float device* lse_1 [[buffer(6)]], uint device* o_1 [[buffer(3)]])
{
    uint _S7;
    uint _S8;
    uint _S9;
    uint _S10;
    uint _S11;
    uint _S12;
    uint _S13;
    uint _S14;
    uint _S15;
    uint _S16;
    uint _S17;
    uint _S18;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->q_0 = q_1;
    (&kernelContext_0)->lens_1 = lens_2;
    (&kernelContext_0)->table_0 = table_1;
    (&kernelContext_0)->k_0 = k_1;
    (&kernelContext_0)->v_0 = v_1;
    (&kernelContext_0)->partials_0 = partials_1;
    (&kernelContext_0)->lse_0 = lse_1;
    (&kernelContext_0)->o_0 = o_1;
    threadgroup array<uint, int(1040)> qs_1;
    (&kernelContext_0)->qs_0 = &qs_1;
    threadgroup array<uint, int(2080)> ks_1;
    (&kernelContext_0)->ks_0 = &ks_1;
    threadgroup array<uint, int(2080)> vs_1;
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
    uint _S19 = kvHead_0 * hpk_0;
    uint qi_0;
    if(live_0)
    {
        uint _S20 = r_0 / (&kernelContext_0)->p_0->queries_0;
        qi_0 = _S20;
    }
    else
    {
        qi_0 = 0U;
    }
    uint head_0 = _S19 + qi_0;
    if(live_0)
    {
        uint _S21 = r_0 % (&kernelContext_0)->p_0->queries_0;
        qi_0 = _S21;
    }
    else
    {
        qi_0 = 0U;
    }
    bool ok_0;
    uint w_2;
    uint _S22 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint offset_0 = (&kernelContext_0)->p_0->keys_0 - (&kernelContext_0)->p_0->queries_0;
    uint e_1 = t_0;
    for(;;)
    {
        if(e_1 < 1024U)
        {
        }
        else
        {
            break;
        }
        uint row_0 = r0_0 + e_1 / 64U;
        uint pr_0 = e_1 % 64U;
        if(row_0 < rowsKv_0)
        {
            ok_0 = pr_0 < _S22;
        }
        else
        {
            ok_0 = false;
        }
        if(ok_0)
        {
            uint _S23 = row_0 / (&kernelContext_0)->p_0->queries_0;
            uint h_3 = _S19 + _S23;
            uint i_1 = row_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((h_3 * (&kernelContext_0)->p_0->queries_0 + i_1) * _S22 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_1 / 64U * 65U + pr_0] = w_2;
        e_1 = e_1 + 128U;
    }
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S24 = min((&kernelContext_0)->p_0->keys_0, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    uint rLast_0 = min(r0_0 + 16U, rowsKv_0) - 1U;
    uint _S25 = (&kernelContext_0)->p_0->lengthGroup_1;
    uint _S26 = r0_0 / (&kernelContext_0)->p_0->queries_0;
    uint _S27 = _S19 + _S26;
    uint _S28 = rLast_0 / (&kernelContext_0)->p_0->queries_0;
    uint kEnd_0 = attnKeyEnd_0((&kernelContext_0)->lens_1, _S25, _S27, _S19 + _S28, _S24);
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint _S29 = r0_0 / (&kernelContext_0)->p_0->queries_0;
        uint _S30 = rLast_0 / (&kernelContext_0)->p_0->queries_0;
        if(_S29 != _S30)
        {
            w_2 = (&kernelContext_0)->p_0->queries_0 - 1U;
        }
        else
        {
            uint _S31 = rLast_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = _S31;
        }
        w_2 = min(kEnd_0, w_2 + offset_0 + 1U);
    }
    else
    {
        w_2 = kEnd_0;
    }
    uint limit_0;
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        limit_0 = qi_0 + offset_0 + 1U;
    }
    else
    {
        limit_0 = (&kernelContext_0)->p_0->keys_0;
    }
    if(((&kernelContext_0)->p_0->lengthGroup_1) != 0U)
    {
        uint _S32 = head_0 / (&kernelContext_0)->p_0->lengthGroup_1;
        limit_0 = min(limit_0, *((&kernelContext_0)->lens_1+_S32));
    }
    float _S33 = dg_negative_infinity_0();
    thread array<float, int(8)> accLo_0;
    thread array<float, int(8)> accHi_0;
    float hi_1;
    float s_0;
    accLo_0[0U] = 0.0;
    accHi_0[0U] = 0.0;
    accLo_0[1U] = 0.0;
    accHi_0[1U] = 0.0;
    accLo_0[2U] = 0.0;
    accHi_0[2U] = 0.0;
    accLo_0[3U] = 0.0;
    accHi_0[3U] = 0.0;
    accLo_0[4U] = 0.0;
    accHi_0[4U] = 0.0;
    accLo_0[5U] = 0.0;
    accHi_0[5U] = 0.0;
    accLo_0[6U] = 0.0;
    accHi_0[6U] = 0.0;
    accLo_0[7U] = 0.0;
    accHi_0[7U] = 0.0;
    uint b_0 = kStart_0;
    float m_0 = _S33;
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
        e_1 = t_0;
        for(;;)
        {
            if(e_1 < 2048U)
            {
            }
            else
            {
                break;
            }
            uint j_0 = e_1 / 64U;
            uint pr_1 = e_1 % 64U;
            uint key_0 = b_0 + j_0;
            if(key_0 < w_2)
            {
                ok_0 = pr_1 < _S22;
            }
            else
            {
                ok_0 = false;
            }
            if(ok_0)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    at_0 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * _S22 + pr_1;
                }
                else
                {
                    uint _S34 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S35 = (*((&kernelContext_0)->table_0+_S34) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S36 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
                    at_0 = (_S35 + _S36) * _S22 + pr_1;
                }
            }
            else
            {
                at_0 = 0U;
            }
            uint _S37 = j_0 * 65U + pr_1;
            if(ok_0)
            {
                p0_1 = *((&kernelContext_0)->k_0+at_0);
            }
            else
            {
                p0_1 = 0U;
            }
            (*(&kernelContext_0)->ks_0)[_S37] = p0_1;
            if(ok_0)
            {
                p0_0 = *((&kernelContext_0)->v_0+at_0);
            }
            else
            {
                p0_0 = 0U;
            }
            (*(&kernelContext_0)->vs_0)[_S37] = p0_0;
            e_1 = e_1 + 128U;
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
                    s_0 = 0.0;
                    for(;;)
                    {
                        if(at_0 < 64U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S38 = lr_0 * 65U;
                        uint _S39 = kk_0 * 65U;
                        uint pr_2 = at_0 + 1U;
                        uint pr_3 = at_0 + 2U;
                        uint pr_4 = at_0 + 3U;
                        uint pr_5 = at_0 + 4U;
                        uint pr_6 = at_0 + 5U;
                        uint pr_7 = at_0 + 6U;
                        uint pr_8 = at_0 + 7U;
                        float s_1 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S38 + pr_8]), attnHi_0((*(&kernelContext_0)->ks_0)[_S39 + pr_8]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S38 + pr_8]), attnLo_0((*(&kernelContext_0)->ks_0)[_S39 + pr_8]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S38 + pr_7]), attnHi_0((*(&kernelContext_0)->ks_0)[_S39 + pr_7]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S38 + pr_7]), attnLo_0((*(&kernelContext_0)->ks_0)[_S39 + pr_7]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S38 + pr_6]), attnHi_0((*(&kernelContext_0)->ks_0)[_S39 + pr_6]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S38 + pr_6]), attnLo_0((*(&kernelContext_0)->ks_0)[_S39 + pr_6]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S38 + pr_5]), attnHi_0((*(&kernelContext_0)->ks_0)[_S39 + pr_5]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S38 + pr_5]), attnLo_0((*(&kernelContext_0)->ks_0)[_S39 + pr_5]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S38 + pr_4]), attnHi_0((*(&kernelContext_0)->ks_0)[_S39 + pr_4]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S38 + pr_4]), attnLo_0((*(&kernelContext_0)->ks_0)[_S39 + pr_4]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S38 + pr_3]), attnHi_0((*(&kernelContext_0)->ks_0)[_S39 + pr_3]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S38 + pr_3]), attnLo_0((*(&kernelContext_0)->ks_0)[_S39 + pr_3]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S38 + pr_2]), attnHi_0((*(&kernelContext_0)->ks_0)[_S39 + pr_2]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S38 + pr_2]), attnLo_0((*(&kernelContext_0)->ks_0)[_S39 + pr_2]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S38 + at_0]), attnHi_0((*(&kernelContext_0)->ks_0)[_S39 + at_0]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S38 + at_0]), attnLo_0((*(&kernelContext_0)->ks_0)[_S39 + at_0]), s_0))))))))))))))));
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
                        visible_0 = key_1 < limit_0;
                    }
                    else
                    {
                        visible_0 = false;
                    }
                    uint _S40 = lr_0 * 32U;
                    _S7 = _S40;
                    uint _S41 = _S40 + kk_0;
                    _S15 = _S41;
                    _S8 = _S41;
                    if(visible_0)
                    {
                        hi_1 = s_0 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        hi_1 = _S33;
                    }
                    (*(&kernelContext_0)->ss_0)[_S41] = hi_1;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_1 = kk_0 + 8U;
                    _S9 = j_1;
                    uint key_2 = b_0 + j_1;
                    p0_1 = 0U;
                    float s_2 = 0.0;
                    for(;;)
                    {
                        if(p0_1 < 64U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S42 = lr_0 * 65U;
                        uint _S43 = j_1 * 65U;
                        uint pr_9 = p0_1 + 1U;
                        uint pr_10 = p0_1 + 2U;
                        uint pr_11 = p0_1 + 3U;
                        uint pr_12 = p0_1 + 4U;
                        uint pr_13 = p0_1 + 5U;
                        uint pr_14 = p0_1 + 6U;
                        uint pr_15 = p0_1 + 7U;
                        float s_3 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_15]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_15]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_15]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_15]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_14]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_14]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_14]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_14]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_13]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_13]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_13]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_13]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_12]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_12]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_12]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_12]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_11]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_11]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_11]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_11]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_10]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_10]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_10]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_10]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_9]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_9]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_9]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_9]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + p0_1]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + p0_1]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + p0_1]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + p0_1]), s_2))))))))))))))));
                        p0_1 = p0_1 + 8U;
                        s_2 = s_3;
                    }
                    bool _S44;
                    if(live_0)
                    {
                        _S44 = key_2 < w_2;
                    }
                    else
                    {
                        _S44 = false;
                    }
                    bool visible_1;
                    if(_S44)
                    {
                        visible_1 = key_2 < limit_0;
                    }
                    else
                    {
                        visible_1 = false;
                    }
                    uint _S45 = _S7 + j_1;
                    _S16 = _S45;
                    _S10 = _S45;
                    float _S46;
                    if(visible_1)
                    {
                        _S46 = s_2 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S46 = _S33;
                    }
                    (*(&kernelContext_0)->ss_0)[_S45] = _S46;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_2 = kk_0 + 16U;
                    _S11 = j_2;
                    uint key_3 = b_0 + j_2;
                    p0_0 = 0U;
                    float s_4 = 0.0;
                    for(;;)
                    {
                        if(p0_0 < 64U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S47 = lr_0 * 65U;
                        uint _S48 = j_2 * 65U;
                        uint pr_16 = p0_0 + 1U;
                        uint pr_17 = p0_0 + 2U;
                        uint pr_18 = p0_0 + 3U;
                        uint pr_19 = p0_0 + 4U;
                        uint pr_20 = p0_0 + 5U;
                        uint pr_21 = p0_0 + 6U;
                        uint pr_22 = p0_0 + 7U;
                        float s_5 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S47 + pr_22]), attnHi_0((*(&kernelContext_0)->ks_0)[_S48 + pr_22]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S47 + pr_22]), attnLo_0((*(&kernelContext_0)->ks_0)[_S48 + pr_22]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S47 + pr_21]), attnHi_0((*(&kernelContext_0)->ks_0)[_S48 + pr_21]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S47 + pr_21]), attnLo_0((*(&kernelContext_0)->ks_0)[_S48 + pr_21]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S47 + pr_20]), attnHi_0((*(&kernelContext_0)->ks_0)[_S48 + pr_20]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S47 + pr_20]), attnLo_0((*(&kernelContext_0)->ks_0)[_S48 + pr_20]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S47 + pr_19]), attnHi_0((*(&kernelContext_0)->ks_0)[_S48 + pr_19]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S47 + pr_19]), attnLo_0((*(&kernelContext_0)->ks_0)[_S48 + pr_19]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S47 + pr_18]), attnHi_0((*(&kernelContext_0)->ks_0)[_S48 + pr_18]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S47 + pr_18]), attnLo_0((*(&kernelContext_0)->ks_0)[_S48 + pr_18]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S47 + pr_17]), attnHi_0((*(&kernelContext_0)->ks_0)[_S48 + pr_17]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S47 + pr_17]), attnLo_0((*(&kernelContext_0)->ks_0)[_S48 + pr_17]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S47 + pr_16]), attnHi_0((*(&kernelContext_0)->ks_0)[_S48 + pr_16]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S47 + pr_16]), attnLo_0((*(&kernelContext_0)->ks_0)[_S48 + pr_16]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S47 + p0_0]), attnHi_0((*(&kernelContext_0)->ks_0)[_S48 + p0_0]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S47 + p0_0]), attnLo_0((*(&kernelContext_0)->ks_0)[_S48 + p0_0]), s_4))))))))))))))));
                        p0_0 = p0_0 + 8U;
                        s_4 = s_5;
                    }
                    bool _S49;
                    if(live_0)
                    {
                        _S49 = key_3 < w_2;
                    }
                    else
                    {
                        _S49 = false;
                    }
                    bool visible_2;
                    if(_S49)
                    {
                        visible_2 = key_3 < limit_0;
                    }
                    else
                    {
                        visible_2 = false;
                    }
                    uint _S50 = _S7 + j_2;
                    _S17 = _S50;
                    _S12 = _S50;
                    float _S51;
                    if(visible_2)
                    {
                        _S51 = s_4 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S51 = _S33;
                    }
                    (*(&kernelContext_0)->ss_0)[_S50] = _S51;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_3 = kk_0 + 24U;
                    _S13 = j_3;
                    uint key_4 = b_0 + j_3;
                    uint p0_2 = 0U;
                    float s_6 = 0.0;
                    for(;;)
                    {
                        if(p0_2 < 64U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S52 = lr_0 * 65U;
                        uint _S53 = j_3 * 65U;
                        uint pr_23 = p0_2 + 1U;
                        uint pr_24 = p0_2 + 2U;
                        uint pr_25 = p0_2 + 3U;
                        uint pr_26 = p0_2 + 4U;
                        uint pr_27 = p0_2 + 5U;
                        uint pr_28 = p0_2 + 6U;
                        uint pr_29 = p0_2 + 7U;
                        float s_7 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S52 + pr_29]), attnHi_0((*(&kernelContext_0)->ks_0)[_S53 + pr_29]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S52 + pr_29]), attnLo_0((*(&kernelContext_0)->ks_0)[_S53 + pr_29]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S52 + pr_28]), attnHi_0((*(&kernelContext_0)->ks_0)[_S53 + pr_28]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S52 + pr_28]), attnLo_0((*(&kernelContext_0)->ks_0)[_S53 + pr_28]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S52 + pr_27]), attnHi_0((*(&kernelContext_0)->ks_0)[_S53 + pr_27]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S52 + pr_27]), attnLo_0((*(&kernelContext_0)->ks_0)[_S53 + pr_27]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S52 + pr_26]), attnHi_0((*(&kernelContext_0)->ks_0)[_S53 + pr_26]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S52 + pr_26]), attnLo_0((*(&kernelContext_0)->ks_0)[_S53 + pr_26]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S52 + pr_25]), attnHi_0((*(&kernelContext_0)->ks_0)[_S53 + pr_25]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S52 + pr_25]), attnLo_0((*(&kernelContext_0)->ks_0)[_S53 + pr_25]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S52 + pr_24]), attnHi_0((*(&kernelContext_0)->ks_0)[_S53 + pr_24]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S52 + pr_24]), attnLo_0((*(&kernelContext_0)->ks_0)[_S53 + pr_24]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S52 + pr_23]), attnHi_0((*(&kernelContext_0)->ks_0)[_S53 + pr_23]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S52 + pr_23]), attnLo_0((*(&kernelContext_0)->ks_0)[_S53 + pr_23]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S52 + p0_2]), attnHi_0((*(&kernelContext_0)->ks_0)[_S53 + p0_2]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S52 + p0_2]), attnLo_0((*(&kernelContext_0)->ks_0)[_S53 + p0_2]), s_6))))))))))))))));
                        p0_2 = p0_2 + 8U;
                        s_6 = s_7;
                    }
                    bool _S54;
                    if(live_0)
                    {
                        _S54 = key_4 < w_2;
                    }
                    else
                    {
                        _S54 = false;
                    }
                    bool visible_3;
                    if(_S54)
                    {
                        visible_3 = key_4 < limit_0;
                    }
                    else
                    {
                        visible_3 = false;
                    }
                    uint _S55 = _S7 + j_3;
                    _S18 = _S55;
                    _S14 = _S55;
                    float _S56;
                    if(visible_3)
                    {
                        _S56 = s_6 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S56 = _S33;
                    }
                    (*(&kernelContext_0)->ss_0)[_S55] = _S56;
                    break;
                }
                break;
            }
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S57 = _S7 + 1U;
        uint _S58 = _S7 + 2U;
        uint _S59 = _S7 + 3U;
        uint _S60 = _S7 + 4U;
        uint _S61 = _S7 + 5U;
        uint _S62 = _S7 + 6U;
        uint _S63 = _S7 + 7U;
        uint _S64 = _S7 + 8U;
        uint _S65 = _S7 + 9U;
        uint _S66 = _S7 + 10U;
        uint _S67 = _S7 + 11U;
        uint _S68 = _S7 + 12U;
        uint _S69 = _S7 + 13U;
        uint _S70 = _S7 + 14U;
        uint _S71 = _S7 + 15U;
        uint _S72 = _S7 + 16U;
        uint _S73 = _S7 + 17U;
        uint _S74 = _S7 + 18U;
        uint _S75 = _S7 + 19U;
        uint _S76 = _S7 + 20U;
        uint _S77 = _S7 + 21U;
        uint _S78 = _S7 + 22U;
        uint _S79 = _S7 + 23U;
        uint _S80 = _S7 + 24U;
        uint _S81 = _S7 + 25U;
        uint _S82 = _S7 + 26U;
        uint _S83 = _S7 + 27U;
        uint _S84 = _S7 + 28U;
        uint _S85 = _S7 + 29U;
        uint _S86 = _S7 + 30U;
        uint _S87 = _S7 + 31U;
        float _S88 = max(m_0, max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(_S33, (*(&kernelContext_0)->ss_0)[_S7]), (*(&kernelContext_0)->ss_0)[_S57]), (*(&kernelContext_0)->ss_0)[_S58]), (*(&kernelContext_0)->ss_0)[_S59]), (*(&kernelContext_0)->ss_0)[_S60]), (*(&kernelContext_0)->ss_0)[_S61]), (*(&kernelContext_0)->ss_0)[_S62]), (*(&kernelContext_0)->ss_0)[_S63]), (*(&kernelContext_0)->ss_0)[_S64]), (*(&kernelContext_0)->ss_0)[_S65]), (*(&kernelContext_0)->ss_0)[_S66]), (*(&kernelContext_0)->ss_0)[_S67]), (*(&kernelContext_0)->ss_0)[_S68]), (*(&kernelContext_0)->ss_0)[_S69]), (*(&kernelContext_0)->ss_0)[_S70]), (*(&kernelContext_0)->ss_0)[_S71]), (*(&kernelContext_0)->ss_0)[_S72]), (*(&kernelContext_0)->ss_0)[_S73]), (*(&kernelContext_0)->ss_0)[_S74]), (*(&kernelContext_0)->ss_0)[_S75]), (*(&kernelContext_0)->ss_0)[_S76]), (*(&kernelContext_0)->ss_0)[_S77]), (*(&kernelContext_0)->ss_0)[_S78]), (*(&kernelContext_0)->ss_0)[_S79]), (*(&kernelContext_0)->ss_0)[_S80]), (*(&kernelContext_0)->ss_0)[_S81]), (*(&kernelContext_0)->ss_0)[_S82]), (*(&kernelContext_0)->ss_0)[_S83]), (*(&kernelContext_0)->ss_0)[_S84]), (*(&kernelContext_0)->ss_0)[_S85]), (*(&kernelContext_0)->ss_0)[_S86]), (*(&kernelContext_0)->ss_0)[_S87]));
        float alpha_0;
        if(m_0 == _S33)
        {
            alpha_0 = 0.0;
        }
        else
        {
            alpha_0 = exp(m_0 - _S88);
        }
        float s_8 = (*(&kernelContext_0)->ss_0)[_S15];
        float _S89;
        if(((*(&kernelContext_0)->ss_0)[_S15]) == _S33)
        {
            _S89 = 0.0;
        }
        else
        {
            _S89 = exp(s_8 - _S88);
        }
        (*(&kernelContext_0)->ps_0)[_S8] = _S89;
        float s_9 = (*(&kernelContext_0)->ss_0)[_S16];
        float _S90;
        if(((*(&kernelContext_0)->ss_0)[_S16]) == _S33)
        {
            _S90 = 0.0;
        }
        else
        {
            _S90 = exp(s_9 - _S88);
        }
        (*(&kernelContext_0)->ps_0)[_S10] = _S90;
        float s_10 = (*(&kernelContext_0)->ss_0)[_S17];
        float _S91;
        if(((*(&kernelContext_0)->ss_0)[_S17]) == _S33)
        {
            _S91 = 0.0;
        }
        else
        {
            _S91 = exp(s_10 - _S88);
        }
        (*(&kernelContext_0)->ps_0)[_S12] = _S91;
        float s_11 = (*(&kernelContext_0)->ss_0)[_S18];
        float _S92;
        if(((*(&kernelContext_0)->ss_0)[_S18]) == _S33)
        {
            _S92 = 0.0;
        }
        else
        {
            _S92 = exp(s_11 - _S88);
        }
        (*(&kernelContext_0)->ps_0)[_S14] = _S92;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S93 = l_0 * alpha_0;
        float _S94 = accLo_0[0U] * alpha_0;
        accLo_0[0U] = _S94;
        float _S95 = accHi_0[0U] * alpha_0;
        accHi_0[0U] = _S95;
        float _S96 = accLo_0[1U] * alpha_0;
        accLo_0[1U] = _S96;
        float _S97 = accHi_0[1U] * alpha_0;
        accHi_0[1U] = _S97;
        float _S98 = accLo_0[2U] * alpha_0;
        accLo_0[2U] = _S98;
        float _S99 = accHi_0[2U] * alpha_0;
        accHi_0[2U] = _S99;
        float _S100 = accLo_0[3U] * alpha_0;
        accLo_0[3U] = _S100;
        float _S101 = accHi_0[3U] * alpha_0;
        accHi_0[3U] = _S101;
        float _S102 = accLo_0[4U] * alpha_0;
        accLo_0[4U] = _S102;
        float _S103 = accHi_0[4U] * alpha_0;
        accHi_0[4U] = _S103;
        float _S104 = accLo_0[5U] * alpha_0;
        accLo_0[5U] = _S104;
        float _S105 = accHi_0[5U] * alpha_0;
        accHi_0[5U] = _S105;
        float _S106 = accLo_0[6U] * alpha_0;
        accLo_0[6U] = _S106;
        float _S107 = accHi_0[6U] * alpha_0;
        accHi_0[6U] = _S107;
        float _S108 = accLo_0[7U] * alpha_0;
        accLo_0[7U] = _S108;
        float _S109 = accHi_0[7U] * alpha_0;
        accHi_0[7U] = _S109;
        float pj_0 = (*(&kernelContext_0)->ps_0)[_S7];
        float _S110 = _S93 + (*(&kernelContext_0)->ps_0)[_S7];
        uint vw_0 = (*(&kernelContext_0)->vs_0)[kk_0];
        float _S111 = fma((*(&kernelContext_0)->ps_0)[_S7], attnLo_0((*(&kernelContext_0)->vs_0)[kk_0]), _S94);
        accLo_0[0U] = _S111;
        float _S112 = fma(pj_0, attnHi_0(vw_0), _S95);
        accHi_0[0U] = _S112;
        uint vw_1 = (*(&kernelContext_0)->vs_0)[_S9];
        float _S113 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S9]), _S96);
        accLo_0[1U] = _S113;
        float _S114 = fma(pj_0, attnHi_0(vw_1), _S97);
        accHi_0[1U] = _S114;
        uint vw_2 = (*(&kernelContext_0)->vs_0)[_S11];
        float _S115 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S11]), _S98);
        accLo_0[2U] = _S115;
        float _S116 = fma(pj_0, attnHi_0(vw_2), _S99);
        accHi_0[2U] = _S116;
        uint vw_3 = (*(&kernelContext_0)->vs_0)[_S13];
        float _S117 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S13]), _S100);
        accLo_0[3U] = _S117;
        float _S118 = fma(pj_0, attnHi_0(vw_3), _S101);
        accHi_0[3U] = _S118;
        uint vw_4 = (*(&kernelContext_0)->vs_0)[kk_0 + 32U];
        float _S119 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 32U]), _S102);
        accLo_0[4U] = _S119;
        float _S120 = fma(pj_0, attnHi_0(vw_4), _S103);
        accHi_0[4U] = _S120;
        uint vw_5 = (*(&kernelContext_0)->vs_0)[kk_0 + 40U];
        float _S121 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 40U]), _S104);
        accLo_0[5U] = _S121;
        float _S122 = fma(pj_0, attnHi_0(vw_5), _S105);
        accHi_0[5U] = _S122;
        uint vw_6 = (*(&kernelContext_0)->vs_0)[kk_0 + 48U];
        float _S123 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 48U]), _S106);
        accLo_0[6U] = _S123;
        float _S124 = fma(pj_0, attnHi_0(vw_6), _S107);
        accHi_0[6U] = _S124;
        uint vw_7 = (*(&kernelContext_0)->vs_0)[kk_0 + 56U];
        float _S125 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 56U]), _S108);
        accLo_0[7U] = _S125;
        float _S126 = fma(pj_0, attnHi_0(vw_7), _S109);
        accHi_0[7U] = _S126;
        float pj_1 = (*(&kernelContext_0)->ps_0)[_S57];
        float _S127 = _S110 + (*(&kernelContext_0)->ps_0)[_S57];
        uint _S128 = 65U + kk_0;
        uint vw_8 = (*(&kernelContext_0)->vs_0)[_S128];
        float _S129 = fma((*(&kernelContext_0)->ps_0)[_S57], attnLo_0((*(&kernelContext_0)->vs_0)[_S128]), _S111);
        accLo_0[0U] = _S129;
        float _S130 = fma(pj_1, attnHi_0(vw_8), _S112);
        accHi_0[0U] = _S130;
        uint vw_9 = (*(&kernelContext_0)->vs_0)[_S128 + 8U];
        float _S131 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S128 + 8U]), _S113);
        accLo_0[1U] = _S131;
        float _S132 = fma(pj_1, attnHi_0(vw_9), _S114);
        accHi_0[1U] = _S132;
        uint vw_10 = (*(&kernelContext_0)->vs_0)[_S128 + 16U];
        float _S133 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S128 + 16U]), _S115);
        accLo_0[2U] = _S133;
        float _S134 = fma(pj_1, attnHi_0(vw_10), _S116);
        accHi_0[2U] = _S134;
        uint vw_11 = (*(&kernelContext_0)->vs_0)[_S128 + 24U];
        float _S135 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S128 + 24U]), _S117);
        accLo_0[3U] = _S135;
        float _S136 = fma(pj_1, attnHi_0(vw_11), _S118);
        accHi_0[3U] = _S136;
        uint vw_12 = (*(&kernelContext_0)->vs_0)[_S128 + 32U];
        float _S137 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S128 + 32U]), _S119);
        accLo_0[4U] = _S137;
        float _S138 = fma(pj_1, attnHi_0(vw_12), _S120);
        accHi_0[4U] = _S138;
        uint vw_13 = (*(&kernelContext_0)->vs_0)[_S128 + 40U];
        float _S139 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S128 + 40U]), _S121);
        accLo_0[5U] = _S139;
        float _S140 = fma(pj_1, attnHi_0(vw_13), _S122);
        accHi_0[5U] = _S140;
        uint vw_14 = (*(&kernelContext_0)->vs_0)[_S128 + 48U];
        float _S141 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S128 + 48U]), _S123);
        accLo_0[6U] = _S141;
        float _S142 = fma(pj_1, attnHi_0(vw_14), _S124);
        accHi_0[6U] = _S142;
        uint vw_15 = (*(&kernelContext_0)->vs_0)[_S128 + 56U];
        float _S143 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S128 + 56U]), _S125);
        accLo_0[7U] = _S143;
        float _S144 = fma(pj_1, attnHi_0(vw_15), _S126);
        accHi_0[7U] = _S144;
        float pj_2 = (*(&kernelContext_0)->ps_0)[_S58];
        float _S145 = _S127 + (*(&kernelContext_0)->ps_0)[_S58];
        uint _S146 = 130U + kk_0;
        uint vw_16 = (*(&kernelContext_0)->vs_0)[_S146];
        float _S147 = fma((*(&kernelContext_0)->ps_0)[_S58], attnLo_0((*(&kernelContext_0)->vs_0)[_S146]), _S129);
        accLo_0[0U] = _S147;
        float _S148 = fma(pj_2, attnHi_0(vw_16), _S130);
        accHi_0[0U] = _S148;
        uint vw_17 = (*(&kernelContext_0)->vs_0)[_S146 + 8U];
        float _S149 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S146 + 8U]), _S131);
        accLo_0[1U] = _S149;
        float _S150 = fma(pj_2, attnHi_0(vw_17), _S132);
        accHi_0[1U] = _S150;
        uint vw_18 = (*(&kernelContext_0)->vs_0)[_S146 + 16U];
        float _S151 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S146 + 16U]), _S133);
        accLo_0[2U] = _S151;
        float _S152 = fma(pj_2, attnHi_0(vw_18), _S134);
        accHi_0[2U] = _S152;
        uint vw_19 = (*(&kernelContext_0)->vs_0)[_S146 + 24U];
        float _S153 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S146 + 24U]), _S135);
        accLo_0[3U] = _S153;
        float _S154 = fma(pj_2, attnHi_0(vw_19), _S136);
        accHi_0[3U] = _S154;
        uint vw_20 = (*(&kernelContext_0)->vs_0)[_S146 + 32U];
        float _S155 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S146 + 32U]), _S137);
        accLo_0[4U] = _S155;
        float _S156 = fma(pj_2, attnHi_0(vw_20), _S138);
        accHi_0[4U] = _S156;
        uint vw_21 = (*(&kernelContext_0)->vs_0)[_S146 + 40U];
        float _S157 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S146 + 40U]), _S139);
        accLo_0[5U] = _S157;
        float _S158 = fma(pj_2, attnHi_0(vw_21), _S140);
        accHi_0[5U] = _S158;
        uint vw_22 = (*(&kernelContext_0)->vs_0)[_S146 + 48U];
        float _S159 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S146 + 48U]), _S141);
        accLo_0[6U] = _S159;
        float _S160 = fma(pj_2, attnHi_0(vw_22), _S142);
        accHi_0[6U] = _S160;
        uint vw_23 = (*(&kernelContext_0)->vs_0)[_S146 + 56U];
        float _S161 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S146 + 56U]), _S143);
        accLo_0[7U] = _S161;
        float _S162 = fma(pj_2, attnHi_0(vw_23), _S144);
        accHi_0[7U] = _S162;
        float pj_3 = (*(&kernelContext_0)->ps_0)[_S59];
        float _S163 = _S145 + (*(&kernelContext_0)->ps_0)[_S59];
        uint _S164 = 195U + kk_0;
        uint vw_24 = (*(&kernelContext_0)->vs_0)[_S164];
        float _S165 = fma((*(&kernelContext_0)->ps_0)[_S59], attnLo_0((*(&kernelContext_0)->vs_0)[_S164]), _S147);
        accLo_0[0U] = _S165;
        float _S166 = fma(pj_3, attnHi_0(vw_24), _S148);
        accHi_0[0U] = _S166;
        uint vw_25 = (*(&kernelContext_0)->vs_0)[_S164 + 8U];
        float _S167 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S164 + 8U]), _S149);
        accLo_0[1U] = _S167;
        float _S168 = fma(pj_3, attnHi_0(vw_25), _S150);
        accHi_0[1U] = _S168;
        uint vw_26 = (*(&kernelContext_0)->vs_0)[_S164 + 16U];
        float _S169 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S164 + 16U]), _S151);
        accLo_0[2U] = _S169;
        float _S170 = fma(pj_3, attnHi_0(vw_26), _S152);
        accHi_0[2U] = _S170;
        uint vw_27 = (*(&kernelContext_0)->vs_0)[_S164 + 24U];
        float _S171 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S164 + 24U]), _S153);
        accLo_0[3U] = _S171;
        float _S172 = fma(pj_3, attnHi_0(vw_27), _S154);
        accHi_0[3U] = _S172;
        uint vw_28 = (*(&kernelContext_0)->vs_0)[_S164 + 32U];
        float _S173 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S164 + 32U]), _S155);
        accLo_0[4U] = _S173;
        float _S174 = fma(pj_3, attnHi_0(vw_28), _S156);
        accHi_0[4U] = _S174;
        uint vw_29 = (*(&kernelContext_0)->vs_0)[_S164 + 40U];
        float _S175 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S164 + 40U]), _S157);
        accLo_0[5U] = _S175;
        float _S176 = fma(pj_3, attnHi_0(vw_29), _S158);
        accHi_0[5U] = _S176;
        uint vw_30 = (*(&kernelContext_0)->vs_0)[_S164 + 48U];
        float _S177 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S164 + 48U]), _S159);
        accLo_0[6U] = _S177;
        float _S178 = fma(pj_3, attnHi_0(vw_30), _S160);
        accHi_0[6U] = _S178;
        uint vw_31 = (*(&kernelContext_0)->vs_0)[_S164 + 56U];
        float _S179 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S164 + 56U]), _S161);
        accLo_0[7U] = _S179;
        float _S180 = fma(pj_3, attnHi_0(vw_31), _S162);
        accHi_0[7U] = _S180;
        float pj_4 = (*(&kernelContext_0)->ps_0)[_S60];
        float _S181 = _S163 + (*(&kernelContext_0)->ps_0)[_S60];
        uint _S182 = 260U + kk_0;
        uint vw_32 = (*(&kernelContext_0)->vs_0)[_S182];
        float _S183 = fma((*(&kernelContext_0)->ps_0)[_S60], attnLo_0((*(&kernelContext_0)->vs_0)[_S182]), _S165);
        accLo_0[0U] = _S183;
        float _S184 = fma(pj_4, attnHi_0(vw_32), _S166);
        accHi_0[0U] = _S184;
        uint vw_33 = (*(&kernelContext_0)->vs_0)[_S182 + 8U];
        float _S185 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S182 + 8U]), _S167);
        accLo_0[1U] = _S185;
        float _S186 = fma(pj_4, attnHi_0(vw_33), _S168);
        accHi_0[1U] = _S186;
        uint vw_34 = (*(&kernelContext_0)->vs_0)[_S182 + 16U];
        float _S187 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S182 + 16U]), _S169);
        accLo_0[2U] = _S187;
        float _S188 = fma(pj_4, attnHi_0(vw_34), _S170);
        accHi_0[2U] = _S188;
        uint vw_35 = (*(&kernelContext_0)->vs_0)[_S182 + 24U];
        float _S189 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S182 + 24U]), _S171);
        accLo_0[3U] = _S189;
        float _S190 = fma(pj_4, attnHi_0(vw_35), _S172);
        accHi_0[3U] = _S190;
        uint vw_36 = (*(&kernelContext_0)->vs_0)[_S182 + 32U];
        float _S191 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S182 + 32U]), _S173);
        accLo_0[4U] = _S191;
        float _S192 = fma(pj_4, attnHi_0(vw_36), _S174);
        accHi_0[4U] = _S192;
        uint vw_37 = (*(&kernelContext_0)->vs_0)[_S182 + 40U];
        float _S193 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S182 + 40U]), _S175);
        accLo_0[5U] = _S193;
        float _S194 = fma(pj_4, attnHi_0(vw_37), _S176);
        accHi_0[5U] = _S194;
        uint vw_38 = (*(&kernelContext_0)->vs_0)[_S182 + 48U];
        float _S195 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S182 + 48U]), _S177);
        accLo_0[6U] = _S195;
        float _S196 = fma(pj_4, attnHi_0(vw_38), _S178);
        accHi_0[6U] = _S196;
        uint vw_39 = (*(&kernelContext_0)->vs_0)[_S182 + 56U];
        float _S197 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S182 + 56U]), _S179);
        accLo_0[7U] = _S197;
        float _S198 = fma(pj_4, attnHi_0(vw_39), _S180);
        accHi_0[7U] = _S198;
        float pj_5 = (*(&kernelContext_0)->ps_0)[_S61];
        float _S199 = _S181 + (*(&kernelContext_0)->ps_0)[_S61];
        uint _S200 = 325U + kk_0;
        uint vw_40 = (*(&kernelContext_0)->vs_0)[_S200];
        float _S201 = fma((*(&kernelContext_0)->ps_0)[_S61], attnLo_0((*(&kernelContext_0)->vs_0)[_S200]), _S183);
        accLo_0[0U] = _S201;
        float _S202 = fma(pj_5, attnHi_0(vw_40), _S184);
        accHi_0[0U] = _S202;
        uint vw_41 = (*(&kernelContext_0)->vs_0)[_S200 + 8U];
        float _S203 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S200 + 8U]), _S185);
        accLo_0[1U] = _S203;
        float _S204 = fma(pj_5, attnHi_0(vw_41), _S186);
        accHi_0[1U] = _S204;
        uint vw_42 = (*(&kernelContext_0)->vs_0)[_S200 + 16U];
        float _S205 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S200 + 16U]), _S187);
        accLo_0[2U] = _S205;
        float _S206 = fma(pj_5, attnHi_0(vw_42), _S188);
        accHi_0[2U] = _S206;
        uint vw_43 = (*(&kernelContext_0)->vs_0)[_S200 + 24U];
        float _S207 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S200 + 24U]), _S189);
        accLo_0[3U] = _S207;
        float _S208 = fma(pj_5, attnHi_0(vw_43), _S190);
        accHi_0[3U] = _S208;
        uint vw_44 = (*(&kernelContext_0)->vs_0)[_S200 + 32U];
        float _S209 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S200 + 32U]), _S191);
        accLo_0[4U] = _S209;
        float _S210 = fma(pj_5, attnHi_0(vw_44), _S192);
        accHi_0[4U] = _S210;
        uint vw_45 = (*(&kernelContext_0)->vs_0)[_S200 + 40U];
        float _S211 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S200 + 40U]), _S193);
        accLo_0[5U] = _S211;
        float _S212 = fma(pj_5, attnHi_0(vw_45), _S194);
        accHi_0[5U] = _S212;
        uint vw_46 = (*(&kernelContext_0)->vs_0)[_S200 + 48U];
        float _S213 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S200 + 48U]), _S195);
        accLo_0[6U] = _S213;
        float _S214 = fma(pj_5, attnHi_0(vw_46), _S196);
        accHi_0[6U] = _S214;
        uint vw_47 = (*(&kernelContext_0)->vs_0)[_S200 + 56U];
        float _S215 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S200 + 56U]), _S197);
        accLo_0[7U] = _S215;
        float _S216 = fma(pj_5, attnHi_0(vw_47), _S198);
        accHi_0[7U] = _S216;
        float pj_6 = (*(&kernelContext_0)->ps_0)[_S62];
        float _S217 = _S199 + (*(&kernelContext_0)->ps_0)[_S62];
        uint _S218 = 390U + kk_0;
        uint vw_48 = (*(&kernelContext_0)->vs_0)[_S218];
        float _S219 = fma((*(&kernelContext_0)->ps_0)[_S62], attnLo_0((*(&kernelContext_0)->vs_0)[_S218]), _S201);
        accLo_0[0U] = _S219;
        float _S220 = fma(pj_6, attnHi_0(vw_48), _S202);
        accHi_0[0U] = _S220;
        uint vw_49 = (*(&kernelContext_0)->vs_0)[_S218 + 8U];
        float _S221 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S218 + 8U]), _S203);
        accLo_0[1U] = _S221;
        float _S222 = fma(pj_6, attnHi_0(vw_49), _S204);
        accHi_0[1U] = _S222;
        uint vw_50 = (*(&kernelContext_0)->vs_0)[_S218 + 16U];
        float _S223 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S218 + 16U]), _S205);
        accLo_0[2U] = _S223;
        float _S224 = fma(pj_6, attnHi_0(vw_50), _S206);
        accHi_0[2U] = _S224;
        uint vw_51 = (*(&kernelContext_0)->vs_0)[_S218 + 24U];
        float _S225 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S218 + 24U]), _S207);
        accLo_0[3U] = _S225;
        float _S226 = fma(pj_6, attnHi_0(vw_51), _S208);
        accHi_0[3U] = _S226;
        uint vw_52 = (*(&kernelContext_0)->vs_0)[_S218 + 32U];
        float _S227 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S218 + 32U]), _S209);
        accLo_0[4U] = _S227;
        float _S228 = fma(pj_6, attnHi_0(vw_52), _S210);
        accHi_0[4U] = _S228;
        uint vw_53 = (*(&kernelContext_0)->vs_0)[_S218 + 40U];
        float _S229 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S218 + 40U]), _S211);
        accLo_0[5U] = _S229;
        float _S230 = fma(pj_6, attnHi_0(vw_53), _S212);
        accHi_0[5U] = _S230;
        uint vw_54 = (*(&kernelContext_0)->vs_0)[_S218 + 48U];
        float _S231 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S218 + 48U]), _S213);
        accLo_0[6U] = _S231;
        float _S232 = fma(pj_6, attnHi_0(vw_54), _S214);
        accHi_0[6U] = _S232;
        uint vw_55 = (*(&kernelContext_0)->vs_0)[_S218 + 56U];
        float _S233 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S218 + 56U]), _S215);
        accLo_0[7U] = _S233;
        float _S234 = fma(pj_6, attnHi_0(vw_55), _S216);
        accHi_0[7U] = _S234;
        float pj_7 = (*(&kernelContext_0)->ps_0)[_S63];
        float _S235 = _S217 + (*(&kernelContext_0)->ps_0)[_S63];
        uint _S236 = 455U + kk_0;
        uint vw_56 = (*(&kernelContext_0)->vs_0)[_S236];
        float _S237 = fma((*(&kernelContext_0)->ps_0)[_S63], attnLo_0((*(&kernelContext_0)->vs_0)[_S236]), _S219);
        accLo_0[0U] = _S237;
        float _S238 = fma(pj_7, attnHi_0(vw_56), _S220);
        accHi_0[0U] = _S238;
        uint vw_57 = (*(&kernelContext_0)->vs_0)[_S236 + 8U];
        float _S239 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S236 + 8U]), _S221);
        accLo_0[1U] = _S239;
        float _S240 = fma(pj_7, attnHi_0(vw_57), _S222);
        accHi_0[1U] = _S240;
        uint vw_58 = (*(&kernelContext_0)->vs_0)[_S236 + 16U];
        float _S241 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S236 + 16U]), _S223);
        accLo_0[2U] = _S241;
        float _S242 = fma(pj_7, attnHi_0(vw_58), _S224);
        accHi_0[2U] = _S242;
        uint vw_59 = (*(&kernelContext_0)->vs_0)[_S236 + 24U];
        float _S243 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S236 + 24U]), _S225);
        accLo_0[3U] = _S243;
        float _S244 = fma(pj_7, attnHi_0(vw_59), _S226);
        accHi_0[3U] = _S244;
        uint vw_60 = (*(&kernelContext_0)->vs_0)[_S236 + 32U];
        float _S245 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S236 + 32U]), _S227);
        accLo_0[4U] = _S245;
        float _S246 = fma(pj_7, attnHi_0(vw_60), _S228);
        accHi_0[4U] = _S246;
        uint vw_61 = (*(&kernelContext_0)->vs_0)[_S236 + 40U];
        float _S247 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S236 + 40U]), _S229);
        accLo_0[5U] = _S247;
        float _S248 = fma(pj_7, attnHi_0(vw_61), _S230);
        accHi_0[5U] = _S248;
        uint vw_62 = (*(&kernelContext_0)->vs_0)[_S236 + 48U];
        float _S249 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S236 + 48U]), _S231);
        accLo_0[6U] = _S249;
        float _S250 = fma(pj_7, attnHi_0(vw_62), _S232);
        accHi_0[6U] = _S250;
        uint vw_63 = (*(&kernelContext_0)->vs_0)[_S236 + 56U];
        float _S251 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S236 + 56U]), _S233);
        accLo_0[7U] = _S251;
        float _S252 = fma(pj_7, attnHi_0(vw_63), _S234);
        accHi_0[7U] = _S252;
        float pj_8 = (*(&kernelContext_0)->ps_0)[_S64];
        float _S253 = _S235 + (*(&kernelContext_0)->ps_0)[_S64];
        uint _S254 = 520U + kk_0;
        uint vw_64 = (*(&kernelContext_0)->vs_0)[_S254];
        float _S255 = fma((*(&kernelContext_0)->ps_0)[_S64], attnLo_0((*(&kernelContext_0)->vs_0)[_S254]), _S237);
        accLo_0[0U] = _S255;
        float _S256 = fma(pj_8, attnHi_0(vw_64), _S238);
        accHi_0[0U] = _S256;
        uint vw_65 = (*(&kernelContext_0)->vs_0)[_S254 + 8U];
        float _S257 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S254 + 8U]), _S239);
        accLo_0[1U] = _S257;
        float _S258 = fma(pj_8, attnHi_0(vw_65), _S240);
        accHi_0[1U] = _S258;
        uint vw_66 = (*(&kernelContext_0)->vs_0)[_S254 + 16U];
        float _S259 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S254 + 16U]), _S241);
        accLo_0[2U] = _S259;
        float _S260 = fma(pj_8, attnHi_0(vw_66), _S242);
        accHi_0[2U] = _S260;
        uint vw_67 = (*(&kernelContext_0)->vs_0)[_S254 + 24U];
        float _S261 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S254 + 24U]), _S243);
        accLo_0[3U] = _S261;
        float _S262 = fma(pj_8, attnHi_0(vw_67), _S244);
        accHi_0[3U] = _S262;
        uint vw_68 = (*(&kernelContext_0)->vs_0)[_S254 + 32U];
        float _S263 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S254 + 32U]), _S245);
        accLo_0[4U] = _S263;
        float _S264 = fma(pj_8, attnHi_0(vw_68), _S246);
        accHi_0[4U] = _S264;
        uint vw_69 = (*(&kernelContext_0)->vs_0)[_S254 + 40U];
        float _S265 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S254 + 40U]), _S247);
        accLo_0[5U] = _S265;
        float _S266 = fma(pj_8, attnHi_0(vw_69), _S248);
        accHi_0[5U] = _S266;
        uint vw_70 = (*(&kernelContext_0)->vs_0)[_S254 + 48U];
        float _S267 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S254 + 48U]), _S249);
        accLo_0[6U] = _S267;
        float _S268 = fma(pj_8, attnHi_0(vw_70), _S250);
        accHi_0[6U] = _S268;
        uint vw_71 = (*(&kernelContext_0)->vs_0)[_S254 + 56U];
        float _S269 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S254 + 56U]), _S251);
        accLo_0[7U] = _S269;
        float _S270 = fma(pj_8, attnHi_0(vw_71), _S252);
        accHi_0[7U] = _S270;
        float pj_9 = (*(&kernelContext_0)->ps_0)[_S65];
        float _S271 = _S253 + (*(&kernelContext_0)->ps_0)[_S65];
        uint _S272 = 585U + kk_0;
        uint vw_72 = (*(&kernelContext_0)->vs_0)[_S272];
        float _S273 = fma((*(&kernelContext_0)->ps_0)[_S65], attnLo_0((*(&kernelContext_0)->vs_0)[_S272]), _S255);
        accLo_0[0U] = _S273;
        float _S274 = fma(pj_9, attnHi_0(vw_72), _S256);
        accHi_0[0U] = _S274;
        uint vw_73 = (*(&kernelContext_0)->vs_0)[_S272 + 8U];
        float _S275 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S272 + 8U]), _S257);
        accLo_0[1U] = _S275;
        float _S276 = fma(pj_9, attnHi_0(vw_73), _S258);
        accHi_0[1U] = _S276;
        uint vw_74 = (*(&kernelContext_0)->vs_0)[_S272 + 16U];
        float _S277 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S272 + 16U]), _S259);
        accLo_0[2U] = _S277;
        float _S278 = fma(pj_9, attnHi_0(vw_74), _S260);
        accHi_0[2U] = _S278;
        uint vw_75 = (*(&kernelContext_0)->vs_0)[_S272 + 24U];
        float _S279 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S272 + 24U]), _S261);
        accLo_0[3U] = _S279;
        float _S280 = fma(pj_9, attnHi_0(vw_75), _S262);
        accHi_0[3U] = _S280;
        uint vw_76 = (*(&kernelContext_0)->vs_0)[_S272 + 32U];
        float _S281 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S272 + 32U]), _S263);
        accLo_0[4U] = _S281;
        float _S282 = fma(pj_9, attnHi_0(vw_76), _S264);
        accHi_0[4U] = _S282;
        uint vw_77 = (*(&kernelContext_0)->vs_0)[_S272 + 40U];
        float _S283 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S272 + 40U]), _S265);
        accLo_0[5U] = _S283;
        float _S284 = fma(pj_9, attnHi_0(vw_77), _S266);
        accHi_0[5U] = _S284;
        uint vw_78 = (*(&kernelContext_0)->vs_0)[_S272 + 48U];
        float _S285 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S272 + 48U]), _S267);
        accLo_0[6U] = _S285;
        float _S286 = fma(pj_9, attnHi_0(vw_78), _S268);
        accHi_0[6U] = _S286;
        uint vw_79 = (*(&kernelContext_0)->vs_0)[_S272 + 56U];
        float _S287 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S272 + 56U]), _S269);
        accLo_0[7U] = _S287;
        float _S288 = fma(pj_9, attnHi_0(vw_79), _S270);
        accHi_0[7U] = _S288;
        float pj_10 = (*(&kernelContext_0)->ps_0)[_S66];
        float _S289 = _S271 + (*(&kernelContext_0)->ps_0)[_S66];
        uint _S290 = 650U + kk_0;
        uint vw_80 = (*(&kernelContext_0)->vs_0)[_S290];
        float _S291 = fma((*(&kernelContext_0)->ps_0)[_S66], attnLo_0((*(&kernelContext_0)->vs_0)[_S290]), _S273);
        accLo_0[0U] = _S291;
        float _S292 = fma(pj_10, attnHi_0(vw_80), _S274);
        accHi_0[0U] = _S292;
        uint vw_81 = (*(&kernelContext_0)->vs_0)[_S290 + 8U];
        float _S293 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S290 + 8U]), _S275);
        accLo_0[1U] = _S293;
        float _S294 = fma(pj_10, attnHi_0(vw_81), _S276);
        accHi_0[1U] = _S294;
        uint vw_82 = (*(&kernelContext_0)->vs_0)[_S290 + 16U];
        float _S295 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S290 + 16U]), _S277);
        accLo_0[2U] = _S295;
        float _S296 = fma(pj_10, attnHi_0(vw_82), _S278);
        accHi_0[2U] = _S296;
        uint vw_83 = (*(&kernelContext_0)->vs_0)[_S290 + 24U];
        float _S297 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S290 + 24U]), _S279);
        accLo_0[3U] = _S297;
        float _S298 = fma(pj_10, attnHi_0(vw_83), _S280);
        accHi_0[3U] = _S298;
        uint vw_84 = (*(&kernelContext_0)->vs_0)[_S290 + 32U];
        float _S299 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S290 + 32U]), _S281);
        accLo_0[4U] = _S299;
        float _S300 = fma(pj_10, attnHi_0(vw_84), _S282);
        accHi_0[4U] = _S300;
        uint vw_85 = (*(&kernelContext_0)->vs_0)[_S290 + 40U];
        float _S301 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S290 + 40U]), _S283);
        accLo_0[5U] = _S301;
        float _S302 = fma(pj_10, attnHi_0(vw_85), _S284);
        accHi_0[5U] = _S302;
        uint vw_86 = (*(&kernelContext_0)->vs_0)[_S290 + 48U];
        float _S303 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S290 + 48U]), _S285);
        accLo_0[6U] = _S303;
        float _S304 = fma(pj_10, attnHi_0(vw_86), _S286);
        accHi_0[6U] = _S304;
        uint vw_87 = (*(&kernelContext_0)->vs_0)[_S290 + 56U];
        float _S305 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S290 + 56U]), _S287);
        accLo_0[7U] = _S305;
        float _S306 = fma(pj_10, attnHi_0(vw_87), _S288);
        accHi_0[7U] = _S306;
        float pj_11 = (*(&kernelContext_0)->ps_0)[_S67];
        float _S307 = _S289 + (*(&kernelContext_0)->ps_0)[_S67];
        uint _S308 = 715U + kk_0;
        uint vw_88 = (*(&kernelContext_0)->vs_0)[_S308];
        float _S309 = fma((*(&kernelContext_0)->ps_0)[_S67], attnLo_0((*(&kernelContext_0)->vs_0)[_S308]), _S291);
        accLo_0[0U] = _S309;
        float _S310 = fma(pj_11, attnHi_0(vw_88), _S292);
        accHi_0[0U] = _S310;
        uint vw_89 = (*(&kernelContext_0)->vs_0)[_S308 + 8U];
        float _S311 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S308 + 8U]), _S293);
        accLo_0[1U] = _S311;
        float _S312 = fma(pj_11, attnHi_0(vw_89), _S294);
        accHi_0[1U] = _S312;
        uint vw_90 = (*(&kernelContext_0)->vs_0)[_S308 + 16U];
        float _S313 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S308 + 16U]), _S295);
        accLo_0[2U] = _S313;
        float _S314 = fma(pj_11, attnHi_0(vw_90), _S296);
        accHi_0[2U] = _S314;
        uint vw_91 = (*(&kernelContext_0)->vs_0)[_S308 + 24U];
        float _S315 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S308 + 24U]), _S297);
        accLo_0[3U] = _S315;
        float _S316 = fma(pj_11, attnHi_0(vw_91), _S298);
        accHi_0[3U] = _S316;
        uint vw_92 = (*(&kernelContext_0)->vs_0)[_S308 + 32U];
        float _S317 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S308 + 32U]), _S299);
        accLo_0[4U] = _S317;
        float _S318 = fma(pj_11, attnHi_0(vw_92), _S300);
        accHi_0[4U] = _S318;
        uint vw_93 = (*(&kernelContext_0)->vs_0)[_S308 + 40U];
        float _S319 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S308 + 40U]), _S301);
        accLo_0[5U] = _S319;
        float _S320 = fma(pj_11, attnHi_0(vw_93), _S302);
        accHi_0[5U] = _S320;
        uint vw_94 = (*(&kernelContext_0)->vs_0)[_S308 + 48U];
        float _S321 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S308 + 48U]), _S303);
        accLo_0[6U] = _S321;
        float _S322 = fma(pj_11, attnHi_0(vw_94), _S304);
        accHi_0[6U] = _S322;
        uint vw_95 = (*(&kernelContext_0)->vs_0)[_S308 + 56U];
        float _S323 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S308 + 56U]), _S305);
        accLo_0[7U] = _S323;
        float _S324 = fma(pj_11, attnHi_0(vw_95), _S306);
        accHi_0[7U] = _S324;
        float pj_12 = (*(&kernelContext_0)->ps_0)[_S68];
        float _S325 = _S307 + (*(&kernelContext_0)->ps_0)[_S68];
        uint _S326 = 780U + kk_0;
        uint vw_96 = (*(&kernelContext_0)->vs_0)[_S326];
        float _S327 = fma((*(&kernelContext_0)->ps_0)[_S68], attnLo_0((*(&kernelContext_0)->vs_0)[_S326]), _S309);
        accLo_0[0U] = _S327;
        float _S328 = fma(pj_12, attnHi_0(vw_96), _S310);
        accHi_0[0U] = _S328;
        uint vw_97 = (*(&kernelContext_0)->vs_0)[_S326 + 8U];
        float _S329 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S326 + 8U]), _S311);
        accLo_0[1U] = _S329;
        float _S330 = fma(pj_12, attnHi_0(vw_97), _S312);
        accHi_0[1U] = _S330;
        uint vw_98 = (*(&kernelContext_0)->vs_0)[_S326 + 16U];
        float _S331 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S326 + 16U]), _S313);
        accLo_0[2U] = _S331;
        float _S332 = fma(pj_12, attnHi_0(vw_98), _S314);
        accHi_0[2U] = _S332;
        uint vw_99 = (*(&kernelContext_0)->vs_0)[_S326 + 24U];
        float _S333 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S326 + 24U]), _S315);
        accLo_0[3U] = _S333;
        float _S334 = fma(pj_12, attnHi_0(vw_99), _S316);
        accHi_0[3U] = _S334;
        uint vw_100 = (*(&kernelContext_0)->vs_0)[_S326 + 32U];
        float _S335 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S326 + 32U]), _S317);
        accLo_0[4U] = _S335;
        float _S336 = fma(pj_12, attnHi_0(vw_100), _S318);
        accHi_0[4U] = _S336;
        uint vw_101 = (*(&kernelContext_0)->vs_0)[_S326 + 40U];
        float _S337 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S326 + 40U]), _S319);
        accLo_0[5U] = _S337;
        float _S338 = fma(pj_12, attnHi_0(vw_101), _S320);
        accHi_0[5U] = _S338;
        uint vw_102 = (*(&kernelContext_0)->vs_0)[_S326 + 48U];
        float _S339 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S326 + 48U]), _S321);
        accLo_0[6U] = _S339;
        float _S340 = fma(pj_12, attnHi_0(vw_102), _S322);
        accHi_0[6U] = _S340;
        uint vw_103 = (*(&kernelContext_0)->vs_0)[_S326 + 56U];
        float _S341 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S326 + 56U]), _S323);
        accLo_0[7U] = _S341;
        float _S342 = fma(pj_12, attnHi_0(vw_103), _S324);
        accHi_0[7U] = _S342;
        float pj_13 = (*(&kernelContext_0)->ps_0)[_S69];
        float _S343 = _S325 + (*(&kernelContext_0)->ps_0)[_S69];
        uint _S344 = 845U + kk_0;
        uint vw_104 = (*(&kernelContext_0)->vs_0)[_S344];
        float _S345 = fma((*(&kernelContext_0)->ps_0)[_S69], attnLo_0((*(&kernelContext_0)->vs_0)[_S344]), _S327);
        accLo_0[0U] = _S345;
        float _S346 = fma(pj_13, attnHi_0(vw_104), _S328);
        accHi_0[0U] = _S346;
        uint vw_105 = (*(&kernelContext_0)->vs_0)[_S344 + 8U];
        float _S347 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S344 + 8U]), _S329);
        accLo_0[1U] = _S347;
        float _S348 = fma(pj_13, attnHi_0(vw_105), _S330);
        accHi_0[1U] = _S348;
        uint vw_106 = (*(&kernelContext_0)->vs_0)[_S344 + 16U];
        float _S349 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S344 + 16U]), _S331);
        accLo_0[2U] = _S349;
        float _S350 = fma(pj_13, attnHi_0(vw_106), _S332);
        accHi_0[2U] = _S350;
        uint vw_107 = (*(&kernelContext_0)->vs_0)[_S344 + 24U];
        float _S351 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S344 + 24U]), _S333);
        accLo_0[3U] = _S351;
        float _S352 = fma(pj_13, attnHi_0(vw_107), _S334);
        accHi_0[3U] = _S352;
        uint vw_108 = (*(&kernelContext_0)->vs_0)[_S344 + 32U];
        float _S353 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S344 + 32U]), _S335);
        accLo_0[4U] = _S353;
        float _S354 = fma(pj_13, attnHi_0(vw_108), _S336);
        accHi_0[4U] = _S354;
        uint vw_109 = (*(&kernelContext_0)->vs_0)[_S344 + 40U];
        float _S355 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S344 + 40U]), _S337);
        accLo_0[5U] = _S355;
        float _S356 = fma(pj_13, attnHi_0(vw_109), _S338);
        accHi_0[5U] = _S356;
        uint vw_110 = (*(&kernelContext_0)->vs_0)[_S344 + 48U];
        float _S357 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S344 + 48U]), _S339);
        accLo_0[6U] = _S357;
        float _S358 = fma(pj_13, attnHi_0(vw_110), _S340);
        accHi_0[6U] = _S358;
        uint vw_111 = (*(&kernelContext_0)->vs_0)[_S344 + 56U];
        float _S359 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S344 + 56U]), _S341);
        accLo_0[7U] = _S359;
        float _S360 = fma(pj_13, attnHi_0(vw_111), _S342);
        accHi_0[7U] = _S360;
        float pj_14 = (*(&kernelContext_0)->ps_0)[_S70];
        float _S361 = _S343 + (*(&kernelContext_0)->ps_0)[_S70];
        uint _S362 = 910U + kk_0;
        uint vw_112 = (*(&kernelContext_0)->vs_0)[_S362];
        float _S363 = fma((*(&kernelContext_0)->ps_0)[_S70], attnLo_0((*(&kernelContext_0)->vs_0)[_S362]), _S345);
        accLo_0[0U] = _S363;
        float _S364 = fma(pj_14, attnHi_0(vw_112), _S346);
        accHi_0[0U] = _S364;
        uint vw_113 = (*(&kernelContext_0)->vs_0)[_S362 + 8U];
        float _S365 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S362 + 8U]), _S347);
        accLo_0[1U] = _S365;
        float _S366 = fma(pj_14, attnHi_0(vw_113), _S348);
        accHi_0[1U] = _S366;
        uint vw_114 = (*(&kernelContext_0)->vs_0)[_S362 + 16U];
        float _S367 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S362 + 16U]), _S349);
        accLo_0[2U] = _S367;
        float _S368 = fma(pj_14, attnHi_0(vw_114), _S350);
        accHi_0[2U] = _S368;
        uint vw_115 = (*(&kernelContext_0)->vs_0)[_S362 + 24U];
        float _S369 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S362 + 24U]), _S351);
        accLo_0[3U] = _S369;
        float _S370 = fma(pj_14, attnHi_0(vw_115), _S352);
        accHi_0[3U] = _S370;
        uint vw_116 = (*(&kernelContext_0)->vs_0)[_S362 + 32U];
        float _S371 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S362 + 32U]), _S353);
        accLo_0[4U] = _S371;
        float _S372 = fma(pj_14, attnHi_0(vw_116), _S354);
        accHi_0[4U] = _S372;
        uint vw_117 = (*(&kernelContext_0)->vs_0)[_S362 + 40U];
        float _S373 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S362 + 40U]), _S355);
        accLo_0[5U] = _S373;
        float _S374 = fma(pj_14, attnHi_0(vw_117), _S356);
        accHi_0[5U] = _S374;
        uint vw_118 = (*(&kernelContext_0)->vs_0)[_S362 + 48U];
        float _S375 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S362 + 48U]), _S357);
        accLo_0[6U] = _S375;
        float _S376 = fma(pj_14, attnHi_0(vw_118), _S358);
        accHi_0[6U] = _S376;
        uint vw_119 = (*(&kernelContext_0)->vs_0)[_S362 + 56U];
        float _S377 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S362 + 56U]), _S359);
        accLo_0[7U] = _S377;
        float _S378 = fma(pj_14, attnHi_0(vw_119), _S360);
        accHi_0[7U] = _S378;
        float pj_15 = (*(&kernelContext_0)->ps_0)[_S71];
        float _S379 = _S361 + (*(&kernelContext_0)->ps_0)[_S71];
        uint _S380 = 975U + kk_0;
        uint vw_120 = (*(&kernelContext_0)->vs_0)[_S380];
        float _S381 = fma((*(&kernelContext_0)->ps_0)[_S71], attnLo_0((*(&kernelContext_0)->vs_0)[_S380]), _S363);
        accLo_0[0U] = _S381;
        float _S382 = fma(pj_15, attnHi_0(vw_120), _S364);
        accHi_0[0U] = _S382;
        uint vw_121 = (*(&kernelContext_0)->vs_0)[_S380 + 8U];
        float _S383 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S380 + 8U]), _S365);
        accLo_0[1U] = _S383;
        float _S384 = fma(pj_15, attnHi_0(vw_121), _S366);
        accHi_0[1U] = _S384;
        uint vw_122 = (*(&kernelContext_0)->vs_0)[_S380 + 16U];
        float _S385 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S380 + 16U]), _S367);
        accLo_0[2U] = _S385;
        float _S386 = fma(pj_15, attnHi_0(vw_122), _S368);
        accHi_0[2U] = _S386;
        uint vw_123 = (*(&kernelContext_0)->vs_0)[_S380 + 24U];
        float _S387 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S380 + 24U]), _S369);
        accLo_0[3U] = _S387;
        float _S388 = fma(pj_15, attnHi_0(vw_123), _S370);
        accHi_0[3U] = _S388;
        uint vw_124 = (*(&kernelContext_0)->vs_0)[_S380 + 32U];
        float _S389 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S380 + 32U]), _S371);
        accLo_0[4U] = _S389;
        float _S390 = fma(pj_15, attnHi_0(vw_124), _S372);
        accHi_0[4U] = _S390;
        uint vw_125 = (*(&kernelContext_0)->vs_0)[_S380 + 40U];
        float _S391 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S380 + 40U]), _S373);
        accLo_0[5U] = _S391;
        float _S392 = fma(pj_15, attnHi_0(vw_125), _S374);
        accHi_0[5U] = _S392;
        uint vw_126 = (*(&kernelContext_0)->vs_0)[_S380 + 48U];
        float _S393 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S380 + 48U]), _S375);
        accLo_0[6U] = _S393;
        float _S394 = fma(pj_15, attnHi_0(vw_126), _S376);
        accHi_0[6U] = _S394;
        uint vw_127 = (*(&kernelContext_0)->vs_0)[_S380 + 56U];
        float _S395 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S380 + 56U]), _S377);
        accLo_0[7U] = _S395;
        float _S396 = fma(pj_15, attnHi_0(vw_127), _S378);
        accHi_0[7U] = _S396;
        float pj_16 = (*(&kernelContext_0)->ps_0)[_S72];
        float _S397 = _S379 + (*(&kernelContext_0)->ps_0)[_S72];
        uint _S398 = 1040U + kk_0;
        uint vw_128 = (*(&kernelContext_0)->vs_0)[_S398];
        float _S399 = fma((*(&kernelContext_0)->ps_0)[_S72], attnLo_0((*(&kernelContext_0)->vs_0)[_S398]), _S381);
        accLo_0[0U] = _S399;
        float _S400 = fma(pj_16, attnHi_0(vw_128), _S382);
        accHi_0[0U] = _S400;
        uint vw_129 = (*(&kernelContext_0)->vs_0)[_S398 + 8U];
        float _S401 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S398 + 8U]), _S383);
        accLo_0[1U] = _S401;
        float _S402 = fma(pj_16, attnHi_0(vw_129), _S384);
        accHi_0[1U] = _S402;
        uint vw_130 = (*(&kernelContext_0)->vs_0)[_S398 + 16U];
        float _S403 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S398 + 16U]), _S385);
        accLo_0[2U] = _S403;
        float _S404 = fma(pj_16, attnHi_0(vw_130), _S386);
        accHi_0[2U] = _S404;
        uint vw_131 = (*(&kernelContext_0)->vs_0)[_S398 + 24U];
        float _S405 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S398 + 24U]), _S387);
        accLo_0[3U] = _S405;
        float _S406 = fma(pj_16, attnHi_0(vw_131), _S388);
        accHi_0[3U] = _S406;
        uint vw_132 = (*(&kernelContext_0)->vs_0)[_S398 + 32U];
        float _S407 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S398 + 32U]), _S389);
        accLo_0[4U] = _S407;
        float _S408 = fma(pj_16, attnHi_0(vw_132), _S390);
        accHi_0[4U] = _S408;
        uint vw_133 = (*(&kernelContext_0)->vs_0)[_S398 + 40U];
        float _S409 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S398 + 40U]), _S391);
        accLo_0[5U] = _S409;
        float _S410 = fma(pj_16, attnHi_0(vw_133), _S392);
        accHi_0[5U] = _S410;
        uint vw_134 = (*(&kernelContext_0)->vs_0)[_S398 + 48U];
        float _S411 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S398 + 48U]), _S393);
        accLo_0[6U] = _S411;
        float _S412 = fma(pj_16, attnHi_0(vw_134), _S394);
        accHi_0[6U] = _S412;
        uint vw_135 = (*(&kernelContext_0)->vs_0)[_S398 + 56U];
        float _S413 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S398 + 56U]), _S395);
        accLo_0[7U] = _S413;
        float _S414 = fma(pj_16, attnHi_0(vw_135), _S396);
        accHi_0[7U] = _S414;
        float pj_17 = (*(&kernelContext_0)->ps_0)[_S73];
        float _S415 = _S397 + (*(&kernelContext_0)->ps_0)[_S73];
        uint _S416 = 1105U + kk_0;
        uint vw_136 = (*(&kernelContext_0)->vs_0)[_S416];
        float _S417 = fma((*(&kernelContext_0)->ps_0)[_S73], attnLo_0((*(&kernelContext_0)->vs_0)[_S416]), _S399);
        accLo_0[0U] = _S417;
        float _S418 = fma(pj_17, attnHi_0(vw_136), _S400);
        accHi_0[0U] = _S418;
        uint vw_137 = (*(&kernelContext_0)->vs_0)[_S416 + 8U];
        float _S419 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S416 + 8U]), _S401);
        accLo_0[1U] = _S419;
        float _S420 = fma(pj_17, attnHi_0(vw_137), _S402);
        accHi_0[1U] = _S420;
        uint vw_138 = (*(&kernelContext_0)->vs_0)[_S416 + 16U];
        float _S421 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S416 + 16U]), _S403);
        accLo_0[2U] = _S421;
        float _S422 = fma(pj_17, attnHi_0(vw_138), _S404);
        accHi_0[2U] = _S422;
        uint vw_139 = (*(&kernelContext_0)->vs_0)[_S416 + 24U];
        float _S423 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S416 + 24U]), _S405);
        accLo_0[3U] = _S423;
        float _S424 = fma(pj_17, attnHi_0(vw_139), _S406);
        accHi_0[3U] = _S424;
        uint vw_140 = (*(&kernelContext_0)->vs_0)[_S416 + 32U];
        float _S425 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S416 + 32U]), _S407);
        accLo_0[4U] = _S425;
        float _S426 = fma(pj_17, attnHi_0(vw_140), _S408);
        accHi_0[4U] = _S426;
        uint vw_141 = (*(&kernelContext_0)->vs_0)[_S416 + 40U];
        float _S427 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S416 + 40U]), _S409);
        accLo_0[5U] = _S427;
        float _S428 = fma(pj_17, attnHi_0(vw_141), _S410);
        accHi_0[5U] = _S428;
        uint vw_142 = (*(&kernelContext_0)->vs_0)[_S416 + 48U];
        float _S429 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S416 + 48U]), _S411);
        accLo_0[6U] = _S429;
        float _S430 = fma(pj_17, attnHi_0(vw_142), _S412);
        accHi_0[6U] = _S430;
        uint vw_143 = (*(&kernelContext_0)->vs_0)[_S416 + 56U];
        float _S431 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S416 + 56U]), _S413);
        accLo_0[7U] = _S431;
        float _S432 = fma(pj_17, attnHi_0(vw_143), _S414);
        accHi_0[7U] = _S432;
        float pj_18 = (*(&kernelContext_0)->ps_0)[_S74];
        float _S433 = _S415 + (*(&kernelContext_0)->ps_0)[_S74];
        uint _S434 = 1170U + kk_0;
        uint vw_144 = (*(&kernelContext_0)->vs_0)[_S434];
        float _S435 = fma((*(&kernelContext_0)->ps_0)[_S74], attnLo_0((*(&kernelContext_0)->vs_0)[_S434]), _S417);
        accLo_0[0U] = _S435;
        float _S436 = fma(pj_18, attnHi_0(vw_144), _S418);
        accHi_0[0U] = _S436;
        uint vw_145 = (*(&kernelContext_0)->vs_0)[_S434 + 8U];
        float _S437 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S434 + 8U]), _S419);
        accLo_0[1U] = _S437;
        float _S438 = fma(pj_18, attnHi_0(vw_145), _S420);
        accHi_0[1U] = _S438;
        uint vw_146 = (*(&kernelContext_0)->vs_0)[_S434 + 16U];
        float _S439 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S434 + 16U]), _S421);
        accLo_0[2U] = _S439;
        float _S440 = fma(pj_18, attnHi_0(vw_146), _S422);
        accHi_0[2U] = _S440;
        uint vw_147 = (*(&kernelContext_0)->vs_0)[_S434 + 24U];
        float _S441 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S434 + 24U]), _S423);
        accLo_0[3U] = _S441;
        float _S442 = fma(pj_18, attnHi_0(vw_147), _S424);
        accHi_0[3U] = _S442;
        uint vw_148 = (*(&kernelContext_0)->vs_0)[_S434 + 32U];
        float _S443 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S434 + 32U]), _S425);
        accLo_0[4U] = _S443;
        float _S444 = fma(pj_18, attnHi_0(vw_148), _S426);
        accHi_0[4U] = _S444;
        uint vw_149 = (*(&kernelContext_0)->vs_0)[_S434 + 40U];
        float _S445 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S434 + 40U]), _S427);
        accLo_0[5U] = _S445;
        float _S446 = fma(pj_18, attnHi_0(vw_149), _S428);
        accHi_0[5U] = _S446;
        uint vw_150 = (*(&kernelContext_0)->vs_0)[_S434 + 48U];
        float _S447 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S434 + 48U]), _S429);
        accLo_0[6U] = _S447;
        float _S448 = fma(pj_18, attnHi_0(vw_150), _S430);
        accHi_0[6U] = _S448;
        uint vw_151 = (*(&kernelContext_0)->vs_0)[_S434 + 56U];
        float _S449 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S434 + 56U]), _S431);
        accLo_0[7U] = _S449;
        float _S450 = fma(pj_18, attnHi_0(vw_151), _S432);
        accHi_0[7U] = _S450;
        float pj_19 = (*(&kernelContext_0)->ps_0)[_S75];
        float _S451 = _S433 + (*(&kernelContext_0)->ps_0)[_S75];
        uint _S452 = 1235U + kk_0;
        uint vw_152 = (*(&kernelContext_0)->vs_0)[_S452];
        float _S453 = fma((*(&kernelContext_0)->ps_0)[_S75], attnLo_0((*(&kernelContext_0)->vs_0)[_S452]), _S435);
        accLo_0[0U] = _S453;
        float _S454 = fma(pj_19, attnHi_0(vw_152), _S436);
        accHi_0[0U] = _S454;
        uint vw_153 = (*(&kernelContext_0)->vs_0)[_S452 + 8U];
        float _S455 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S452 + 8U]), _S437);
        accLo_0[1U] = _S455;
        float _S456 = fma(pj_19, attnHi_0(vw_153), _S438);
        accHi_0[1U] = _S456;
        uint vw_154 = (*(&kernelContext_0)->vs_0)[_S452 + 16U];
        float _S457 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S452 + 16U]), _S439);
        accLo_0[2U] = _S457;
        float _S458 = fma(pj_19, attnHi_0(vw_154), _S440);
        accHi_0[2U] = _S458;
        uint vw_155 = (*(&kernelContext_0)->vs_0)[_S452 + 24U];
        float _S459 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S452 + 24U]), _S441);
        accLo_0[3U] = _S459;
        float _S460 = fma(pj_19, attnHi_0(vw_155), _S442);
        accHi_0[3U] = _S460;
        uint vw_156 = (*(&kernelContext_0)->vs_0)[_S452 + 32U];
        float _S461 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S452 + 32U]), _S443);
        accLo_0[4U] = _S461;
        float _S462 = fma(pj_19, attnHi_0(vw_156), _S444);
        accHi_0[4U] = _S462;
        uint vw_157 = (*(&kernelContext_0)->vs_0)[_S452 + 40U];
        float _S463 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S452 + 40U]), _S445);
        accLo_0[5U] = _S463;
        float _S464 = fma(pj_19, attnHi_0(vw_157), _S446);
        accHi_0[5U] = _S464;
        uint vw_158 = (*(&kernelContext_0)->vs_0)[_S452 + 48U];
        float _S465 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S452 + 48U]), _S447);
        accLo_0[6U] = _S465;
        float _S466 = fma(pj_19, attnHi_0(vw_158), _S448);
        accHi_0[6U] = _S466;
        uint vw_159 = (*(&kernelContext_0)->vs_0)[_S452 + 56U];
        float _S467 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S452 + 56U]), _S449);
        accLo_0[7U] = _S467;
        float _S468 = fma(pj_19, attnHi_0(vw_159), _S450);
        accHi_0[7U] = _S468;
        float pj_20 = (*(&kernelContext_0)->ps_0)[_S76];
        float _S469 = _S451 + (*(&kernelContext_0)->ps_0)[_S76];
        uint _S470 = 1300U + kk_0;
        uint vw_160 = (*(&kernelContext_0)->vs_0)[_S470];
        float _S471 = fma((*(&kernelContext_0)->ps_0)[_S76], attnLo_0((*(&kernelContext_0)->vs_0)[_S470]), _S453);
        accLo_0[0U] = _S471;
        float _S472 = fma(pj_20, attnHi_0(vw_160), _S454);
        accHi_0[0U] = _S472;
        uint vw_161 = (*(&kernelContext_0)->vs_0)[_S470 + 8U];
        float _S473 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S470 + 8U]), _S455);
        accLo_0[1U] = _S473;
        float _S474 = fma(pj_20, attnHi_0(vw_161), _S456);
        accHi_0[1U] = _S474;
        uint vw_162 = (*(&kernelContext_0)->vs_0)[_S470 + 16U];
        float _S475 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S470 + 16U]), _S457);
        accLo_0[2U] = _S475;
        float _S476 = fma(pj_20, attnHi_0(vw_162), _S458);
        accHi_0[2U] = _S476;
        uint vw_163 = (*(&kernelContext_0)->vs_0)[_S470 + 24U];
        float _S477 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S470 + 24U]), _S459);
        accLo_0[3U] = _S477;
        float _S478 = fma(pj_20, attnHi_0(vw_163), _S460);
        accHi_0[3U] = _S478;
        uint vw_164 = (*(&kernelContext_0)->vs_0)[_S470 + 32U];
        float _S479 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S470 + 32U]), _S461);
        accLo_0[4U] = _S479;
        float _S480 = fma(pj_20, attnHi_0(vw_164), _S462);
        accHi_0[4U] = _S480;
        uint vw_165 = (*(&kernelContext_0)->vs_0)[_S470 + 40U];
        float _S481 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S470 + 40U]), _S463);
        accLo_0[5U] = _S481;
        float _S482 = fma(pj_20, attnHi_0(vw_165), _S464);
        accHi_0[5U] = _S482;
        uint vw_166 = (*(&kernelContext_0)->vs_0)[_S470 + 48U];
        float _S483 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S470 + 48U]), _S465);
        accLo_0[6U] = _S483;
        float _S484 = fma(pj_20, attnHi_0(vw_166), _S466);
        accHi_0[6U] = _S484;
        uint vw_167 = (*(&kernelContext_0)->vs_0)[_S470 + 56U];
        float _S485 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S470 + 56U]), _S467);
        accLo_0[7U] = _S485;
        float _S486 = fma(pj_20, attnHi_0(vw_167), _S468);
        accHi_0[7U] = _S486;
        float pj_21 = (*(&kernelContext_0)->ps_0)[_S77];
        float _S487 = _S469 + (*(&kernelContext_0)->ps_0)[_S77];
        uint _S488 = 1365U + kk_0;
        uint vw_168 = (*(&kernelContext_0)->vs_0)[_S488];
        float _S489 = fma((*(&kernelContext_0)->ps_0)[_S77], attnLo_0((*(&kernelContext_0)->vs_0)[_S488]), _S471);
        accLo_0[0U] = _S489;
        float _S490 = fma(pj_21, attnHi_0(vw_168), _S472);
        accHi_0[0U] = _S490;
        uint vw_169 = (*(&kernelContext_0)->vs_0)[_S488 + 8U];
        float _S491 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S488 + 8U]), _S473);
        accLo_0[1U] = _S491;
        float _S492 = fma(pj_21, attnHi_0(vw_169), _S474);
        accHi_0[1U] = _S492;
        uint vw_170 = (*(&kernelContext_0)->vs_0)[_S488 + 16U];
        float _S493 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S488 + 16U]), _S475);
        accLo_0[2U] = _S493;
        float _S494 = fma(pj_21, attnHi_0(vw_170), _S476);
        accHi_0[2U] = _S494;
        uint vw_171 = (*(&kernelContext_0)->vs_0)[_S488 + 24U];
        float _S495 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S488 + 24U]), _S477);
        accLo_0[3U] = _S495;
        float _S496 = fma(pj_21, attnHi_0(vw_171), _S478);
        accHi_0[3U] = _S496;
        uint vw_172 = (*(&kernelContext_0)->vs_0)[_S488 + 32U];
        float _S497 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S488 + 32U]), _S479);
        accLo_0[4U] = _S497;
        float _S498 = fma(pj_21, attnHi_0(vw_172), _S480);
        accHi_0[4U] = _S498;
        uint vw_173 = (*(&kernelContext_0)->vs_0)[_S488 + 40U];
        float _S499 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S488 + 40U]), _S481);
        accLo_0[5U] = _S499;
        float _S500 = fma(pj_21, attnHi_0(vw_173), _S482);
        accHi_0[5U] = _S500;
        uint vw_174 = (*(&kernelContext_0)->vs_0)[_S488 + 48U];
        float _S501 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S488 + 48U]), _S483);
        accLo_0[6U] = _S501;
        float _S502 = fma(pj_21, attnHi_0(vw_174), _S484);
        accHi_0[6U] = _S502;
        uint vw_175 = (*(&kernelContext_0)->vs_0)[_S488 + 56U];
        float _S503 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S488 + 56U]), _S485);
        accLo_0[7U] = _S503;
        float _S504 = fma(pj_21, attnHi_0(vw_175), _S486);
        accHi_0[7U] = _S504;
        float pj_22 = (*(&kernelContext_0)->ps_0)[_S78];
        float _S505 = _S487 + (*(&kernelContext_0)->ps_0)[_S78];
        uint _S506 = 1430U + kk_0;
        uint vw_176 = (*(&kernelContext_0)->vs_0)[_S506];
        float _S507 = fma((*(&kernelContext_0)->ps_0)[_S78], attnLo_0((*(&kernelContext_0)->vs_0)[_S506]), _S489);
        accLo_0[0U] = _S507;
        float _S508 = fma(pj_22, attnHi_0(vw_176), _S490);
        accHi_0[0U] = _S508;
        uint vw_177 = (*(&kernelContext_0)->vs_0)[_S506 + 8U];
        float _S509 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S506 + 8U]), _S491);
        accLo_0[1U] = _S509;
        float _S510 = fma(pj_22, attnHi_0(vw_177), _S492);
        accHi_0[1U] = _S510;
        uint vw_178 = (*(&kernelContext_0)->vs_0)[_S506 + 16U];
        float _S511 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S506 + 16U]), _S493);
        accLo_0[2U] = _S511;
        float _S512 = fma(pj_22, attnHi_0(vw_178), _S494);
        accHi_0[2U] = _S512;
        uint vw_179 = (*(&kernelContext_0)->vs_0)[_S506 + 24U];
        float _S513 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S506 + 24U]), _S495);
        accLo_0[3U] = _S513;
        float _S514 = fma(pj_22, attnHi_0(vw_179), _S496);
        accHi_0[3U] = _S514;
        uint vw_180 = (*(&kernelContext_0)->vs_0)[_S506 + 32U];
        float _S515 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S506 + 32U]), _S497);
        accLo_0[4U] = _S515;
        float _S516 = fma(pj_22, attnHi_0(vw_180), _S498);
        accHi_0[4U] = _S516;
        uint vw_181 = (*(&kernelContext_0)->vs_0)[_S506 + 40U];
        float _S517 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S506 + 40U]), _S499);
        accLo_0[5U] = _S517;
        float _S518 = fma(pj_22, attnHi_0(vw_181), _S500);
        accHi_0[5U] = _S518;
        uint vw_182 = (*(&kernelContext_0)->vs_0)[_S506 + 48U];
        float _S519 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S506 + 48U]), _S501);
        accLo_0[6U] = _S519;
        float _S520 = fma(pj_22, attnHi_0(vw_182), _S502);
        accHi_0[6U] = _S520;
        uint vw_183 = (*(&kernelContext_0)->vs_0)[_S506 + 56U];
        float _S521 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S506 + 56U]), _S503);
        accLo_0[7U] = _S521;
        float _S522 = fma(pj_22, attnHi_0(vw_183), _S504);
        accHi_0[7U] = _S522;
        float pj_23 = (*(&kernelContext_0)->ps_0)[_S79];
        float _S523 = _S505 + (*(&kernelContext_0)->ps_0)[_S79];
        uint _S524 = 1495U + kk_0;
        uint vw_184 = (*(&kernelContext_0)->vs_0)[_S524];
        float _S525 = fma((*(&kernelContext_0)->ps_0)[_S79], attnLo_0((*(&kernelContext_0)->vs_0)[_S524]), _S507);
        accLo_0[0U] = _S525;
        float _S526 = fma(pj_23, attnHi_0(vw_184), _S508);
        accHi_0[0U] = _S526;
        uint vw_185 = (*(&kernelContext_0)->vs_0)[_S524 + 8U];
        float _S527 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S524 + 8U]), _S509);
        accLo_0[1U] = _S527;
        float _S528 = fma(pj_23, attnHi_0(vw_185), _S510);
        accHi_0[1U] = _S528;
        uint vw_186 = (*(&kernelContext_0)->vs_0)[_S524 + 16U];
        float _S529 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S524 + 16U]), _S511);
        accLo_0[2U] = _S529;
        float _S530 = fma(pj_23, attnHi_0(vw_186), _S512);
        accHi_0[2U] = _S530;
        uint vw_187 = (*(&kernelContext_0)->vs_0)[_S524 + 24U];
        float _S531 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S524 + 24U]), _S513);
        accLo_0[3U] = _S531;
        float _S532 = fma(pj_23, attnHi_0(vw_187), _S514);
        accHi_0[3U] = _S532;
        uint vw_188 = (*(&kernelContext_0)->vs_0)[_S524 + 32U];
        float _S533 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S524 + 32U]), _S515);
        accLo_0[4U] = _S533;
        float _S534 = fma(pj_23, attnHi_0(vw_188), _S516);
        accHi_0[4U] = _S534;
        uint vw_189 = (*(&kernelContext_0)->vs_0)[_S524 + 40U];
        float _S535 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S524 + 40U]), _S517);
        accLo_0[5U] = _S535;
        float _S536 = fma(pj_23, attnHi_0(vw_189), _S518);
        accHi_0[5U] = _S536;
        uint vw_190 = (*(&kernelContext_0)->vs_0)[_S524 + 48U];
        float _S537 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S524 + 48U]), _S519);
        accLo_0[6U] = _S537;
        float _S538 = fma(pj_23, attnHi_0(vw_190), _S520);
        accHi_0[6U] = _S538;
        uint vw_191 = (*(&kernelContext_0)->vs_0)[_S524 + 56U];
        float _S539 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S524 + 56U]), _S521);
        accLo_0[7U] = _S539;
        float _S540 = fma(pj_23, attnHi_0(vw_191), _S522);
        accHi_0[7U] = _S540;
        float pj_24 = (*(&kernelContext_0)->ps_0)[_S80];
        float _S541 = _S523 + (*(&kernelContext_0)->ps_0)[_S80];
        uint _S542 = 1560U + kk_0;
        uint vw_192 = (*(&kernelContext_0)->vs_0)[_S542];
        float _S543 = fma((*(&kernelContext_0)->ps_0)[_S80], attnLo_0((*(&kernelContext_0)->vs_0)[_S542]), _S525);
        accLo_0[0U] = _S543;
        float _S544 = fma(pj_24, attnHi_0(vw_192), _S526);
        accHi_0[0U] = _S544;
        uint vw_193 = (*(&kernelContext_0)->vs_0)[_S542 + 8U];
        float _S545 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S542 + 8U]), _S527);
        accLo_0[1U] = _S545;
        float _S546 = fma(pj_24, attnHi_0(vw_193), _S528);
        accHi_0[1U] = _S546;
        uint vw_194 = (*(&kernelContext_0)->vs_0)[_S542 + 16U];
        float _S547 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S542 + 16U]), _S529);
        accLo_0[2U] = _S547;
        float _S548 = fma(pj_24, attnHi_0(vw_194), _S530);
        accHi_0[2U] = _S548;
        uint vw_195 = (*(&kernelContext_0)->vs_0)[_S542 + 24U];
        float _S549 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S542 + 24U]), _S531);
        accLo_0[3U] = _S549;
        float _S550 = fma(pj_24, attnHi_0(vw_195), _S532);
        accHi_0[3U] = _S550;
        uint vw_196 = (*(&kernelContext_0)->vs_0)[_S542 + 32U];
        float _S551 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S542 + 32U]), _S533);
        accLo_0[4U] = _S551;
        float _S552 = fma(pj_24, attnHi_0(vw_196), _S534);
        accHi_0[4U] = _S552;
        uint vw_197 = (*(&kernelContext_0)->vs_0)[_S542 + 40U];
        float _S553 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S542 + 40U]), _S535);
        accLo_0[5U] = _S553;
        float _S554 = fma(pj_24, attnHi_0(vw_197), _S536);
        accHi_0[5U] = _S554;
        uint vw_198 = (*(&kernelContext_0)->vs_0)[_S542 + 48U];
        float _S555 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S542 + 48U]), _S537);
        accLo_0[6U] = _S555;
        float _S556 = fma(pj_24, attnHi_0(vw_198), _S538);
        accHi_0[6U] = _S556;
        uint vw_199 = (*(&kernelContext_0)->vs_0)[_S542 + 56U];
        float _S557 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S542 + 56U]), _S539);
        accLo_0[7U] = _S557;
        float _S558 = fma(pj_24, attnHi_0(vw_199), _S540);
        accHi_0[7U] = _S558;
        float pj_25 = (*(&kernelContext_0)->ps_0)[_S81];
        float _S559 = _S541 + (*(&kernelContext_0)->ps_0)[_S81];
        uint _S560 = 1625U + kk_0;
        uint vw_200 = (*(&kernelContext_0)->vs_0)[_S560];
        float _S561 = fma((*(&kernelContext_0)->ps_0)[_S81], attnLo_0((*(&kernelContext_0)->vs_0)[_S560]), _S543);
        accLo_0[0U] = _S561;
        float _S562 = fma(pj_25, attnHi_0(vw_200), _S544);
        accHi_0[0U] = _S562;
        uint vw_201 = (*(&kernelContext_0)->vs_0)[_S560 + 8U];
        float _S563 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S560 + 8U]), _S545);
        accLo_0[1U] = _S563;
        float _S564 = fma(pj_25, attnHi_0(vw_201), _S546);
        accHi_0[1U] = _S564;
        uint vw_202 = (*(&kernelContext_0)->vs_0)[_S560 + 16U];
        float _S565 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S560 + 16U]), _S547);
        accLo_0[2U] = _S565;
        float _S566 = fma(pj_25, attnHi_0(vw_202), _S548);
        accHi_0[2U] = _S566;
        uint vw_203 = (*(&kernelContext_0)->vs_0)[_S560 + 24U];
        float _S567 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S560 + 24U]), _S549);
        accLo_0[3U] = _S567;
        float _S568 = fma(pj_25, attnHi_0(vw_203), _S550);
        accHi_0[3U] = _S568;
        uint vw_204 = (*(&kernelContext_0)->vs_0)[_S560 + 32U];
        float _S569 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S560 + 32U]), _S551);
        accLo_0[4U] = _S569;
        float _S570 = fma(pj_25, attnHi_0(vw_204), _S552);
        accHi_0[4U] = _S570;
        uint vw_205 = (*(&kernelContext_0)->vs_0)[_S560 + 40U];
        float _S571 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S560 + 40U]), _S553);
        accLo_0[5U] = _S571;
        float _S572 = fma(pj_25, attnHi_0(vw_205), _S554);
        accHi_0[5U] = _S572;
        uint vw_206 = (*(&kernelContext_0)->vs_0)[_S560 + 48U];
        float _S573 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S560 + 48U]), _S555);
        accLo_0[6U] = _S573;
        float _S574 = fma(pj_25, attnHi_0(vw_206), _S556);
        accHi_0[6U] = _S574;
        uint vw_207 = (*(&kernelContext_0)->vs_0)[_S560 + 56U];
        float _S575 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S560 + 56U]), _S557);
        accLo_0[7U] = _S575;
        float _S576 = fma(pj_25, attnHi_0(vw_207), _S558);
        accHi_0[7U] = _S576;
        float pj_26 = (*(&kernelContext_0)->ps_0)[_S82];
        float _S577 = _S559 + (*(&kernelContext_0)->ps_0)[_S82];
        uint _S578 = 1690U + kk_0;
        uint vw_208 = (*(&kernelContext_0)->vs_0)[_S578];
        float _S579 = fma((*(&kernelContext_0)->ps_0)[_S82], attnLo_0((*(&kernelContext_0)->vs_0)[_S578]), _S561);
        accLo_0[0U] = _S579;
        float _S580 = fma(pj_26, attnHi_0(vw_208), _S562);
        accHi_0[0U] = _S580;
        uint vw_209 = (*(&kernelContext_0)->vs_0)[_S578 + 8U];
        float _S581 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S578 + 8U]), _S563);
        accLo_0[1U] = _S581;
        float _S582 = fma(pj_26, attnHi_0(vw_209), _S564);
        accHi_0[1U] = _S582;
        uint vw_210 = (*(&kernelContext_0)->vs_0)[_S578 + 16U];
        float _S583 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S578 + 16U]), _S565);
        accLo_0[2U] = _S583;
        float _S584 = fma(pj_26, attnHi_0(vw_210), _S566);
        accHi_0[2U] = _S584;
        uint vw_211 = (*(&kernelContext_0)->vs_0)[_S578 + 24U];
        float _S585 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S578 + 24U]), _S567);
        accLo_0[3U] = _S585;
        float _S586 = fma(pj_26, attnHi_0(vw_211), _S568);
        accHi_0[3U] = _S586;
        uint vw_212 = (*(&kernelContext_0)->vs_0)[_S578 + 32U];
        float _S587 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S578 + 32U]), _S569);
        accLo_0[4U] = _S587;
        float _S588 = fma(pj_26, attnHi_0(vw_212), _S570);
        accHi_0[4U] = _S588;
        uint vw_213 = (*(&kernelContext_0)->vs_0)[_S578 + 40U];
        float _S589 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S578 + 40U]), _S571);
        accLo_0[5U] = _S589;
        float _S590 = fma(pj_26, attnHi_0(vw_213), _S572);
        accHi_0[5U] = _S590;
        uint vw_214 = (*(&kernelContext_0)->vs_0)[_S578 + 48U];
        float _S591 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S578 + 48U]), _S573);
        accLo_0[6U] = _S591;
        float _S592 = fma(pj_26, attnHi_0(vw_214), _S574);
        accHi_0[6U] = _S592;
        uint vw_215 = (*(&kernelContext_0)->vs_0)[_S578 + 56U];
        float _S593 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S578 + 56U]), _S575);
        accLo_0[7U] = _S593;
        float _S594 = fma(pj_26, attnHi_0(vw_215), _S576);
        accHi_0[7U] = _S594;
        float pj_27 = (*(&kernelContext_0)->ps_0)[_S83];
        float _S595 = _S577 + (*(&kernelContext_0)->ps_0)[_S83];
        uint _S596 = 1755U + kk_0;
        uint vw_216 = (*(&kernelContext_0)->vs_0)[_S596];
        float _S597 = fma((*(&kernelContext_0)->ps_0)[_S83], attnLo_0((*(&kernelContext_0)->vs_0)[_S596]), _S579);
        accLo_0[0U] = _S597;
        float _S598 = fma(pj_27, attnHi_0(vw_216), _S580);
        accHi_0[0U] = _S598;
        uint vw_217 = (*(&kernelContext_0)->vs_0)[_S596 + 8U];
        float _S599 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S596 + 8U]), _S581);
        accLo_0[1U] = _S599;
        float _S600 = fma(pj_27, attnHi_0(vw_217), _S582);
        accHi_0[1U] = _S600;
        uint vw_218 = (*(&kernelContext_0)->vs_0)[_S596 + 16U];
        float _S601 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S596 + 16U]), _S583);
        accLo_0[2U] = _S601;
        float _S602 = fma(pj_27, attnHi_0(vw_218), _S584);
        accHi_0[2U] = _S602;
        uint vw_219 = (*(&kernelContext_0)->vs_0)[_S596 + 24U];
        float _S603 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S596 + 24U]), _S585);
        accLo_0[3U] = _S603;
        float _S604 = fma(pj_27, attnHi_0(vw_219), _S586);
        accHi_0[3U] = _S604;
        uint vw_220 = (*(&kernelContext_0)->vs_0)[_S596 + 32U];
        float _S605 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S596 + 32U]), _S587);
        accLo_0[4U] = _S605;
        float _S606 = fma(pj_27, attnHi_0(vw_220), _S588);
        accHi_0[4U] = _S606;
        uint vw_221 = (*(&kernelContext_0)->vs_0)[_S596 + 40U];
        float _S607 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S596 + 40U]), _S589);
        accLo_0[5U] = _S607;
        float _S608 = fma(pj_27, attnHi_0(vw_221), _S590);
        accHi_0[5U] = _S608;
        uint vw_222 = (*(&kernelContext_0)->vs_0)[_S596 + 48U];
        float _S609 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S596 + 48U]), _S591);
        accLo_0[6U] = _S609;
        float _S610 = fma(pj_27, attnHi_0(vw_222), _S592);
        accHi_0[6U] = _S610;
        uint vw_223 = (*(&kernelContext_0)->vs_0)[_S596 + 56U];
        float _S611 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S596 + 56U]), _S593);
        accLo_0[7U] = _S611;
        float _S612 = fma(pj_27, attnHi_0(vw_223), _S594);
        accHi_0[7U] = _S612;
        float pj_28 = (*(&kernelContext_0)->ps_0)[_S84];
        float _S613 = _S595 + (*(&kernelContext_0)->ps_0)[_S84];
        uint _S614 = 1820U + kk_0;
        uint vw_224 = (*(&kernelContext_0)->vs_0)[_S614];
        float _S615 = fma((*(&kernelContext_0)->ps_0)[_S84], attnLo_0((*(&kernelContext_0)->vs_0)[_S614]), _S597);
        accLo_0[0U] = _S615;
        float _S616 = fma(pj_28, attnHi_0(vw_224), _S598);
        accHi_0[0U] = _S616;
        uint vw_225 = (*(&kernelContext_0)->vs_0)[_S614 + 8U];
        float _S617 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S614 + 8U]), _S599);
        accLo_0[1U] = _S617;
        float _S618 = fma(pj_28, attnHi_0(vw_225), _S600);
        accHi_0[1U] = _S618;
        uint vw_226 = (*(&kernelContext_0)->vs_0)[_S614 + 16U];
        float _S619 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S614 + 16U]), _S601);
        accLo_0[2U] = _S619;
        float _S620 = fma(pj_28, attnHi_0(vw_226), _S602);
        accHi_0[2U] = _S620;
        uint vw_227 = (*(&kernelContext_0)->vs_0)[_S614 + 24U];
        float _S621 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S614 + 24U]), _S603);
        accLo_0[3U] = _S621;
        float _S622 = fma(pj_28, attnHi_0(vw_227), _S604);
        accHi_0[3U] = _S622;
        uint vw_228 = (*(&kernelContext_0)->vs_0)[_S614 + 32U];
        float _S623 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S614 + 32U]), _S605);
        accLo_0[4U] = _S623;
        float _S624 = fma(pj_28, attnHi_0(vw_228), _S606);
        accHi_0[4U] = _S624;
        uint vw_229 = (*(&kernelContext_0)->vs_0)[_S614 + 40U];
        float _S625 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S614 + 40U]), _S607);
        accLo_0[5U] = _S625;
        float _S626 = fma(pj_28, attnHi_0(vw_229), _S608);
        accHi_0[5U] = _S626;
        uint vw_230 = (*(&kernelContext_0)->vs_0)[_S614 + 48U];
        float _S627 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S614 + 48U]), _S609);
        accLo_0[6U] = _S627;
        float _S628 = fma(pj_28, attnHi_0(vw_230), _S610);
        accHi_0[6U] = _S628;
        uint vw_231 = (*(&kernelContext_0)->vs_0)[_S614 + 56U];
        float _S629 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S614 + 56U]), _S611);
        accLo_0[7U] = _S629;
        float _S630 = fma(pj_28, attnHi_0(vw_231), _S612);
        accHi_0[7U] = _S630;
        float pj_29 = (*(&kernelContext_0)->ps_0)[_S85];
        float _S631 = _S613 + (*(&kernelContext_0)->ps_0)[_S85];
        uint _S632 = 1885U + kk_0;
        uint vw_232 = (*(&kernelContext_0)->vs_0)[_S632];
        float _S633 = fma((*(&kernelContext_0)->ps_0)[_S85], attnLo_0((*(&kernelContext_0)->vs_0)[_S632]), _S615);
        accLo_0[0U] = _S633;
        float _S634 = fma(pj_29, attnHi_0(vw_232), _S616);
        accHi_0[0U] = _S634;
        uint vw_233 = (*(&kernelContext_0)->vs_0)[_S632 + 8U];
        float _S635 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S632 + 8U]), _S617);
        accLo_0[1U] = _S635;
        float _S636 = fma(pj_29, attnHi_0(vw_233), _S618);
        accHi_0[1U] = _S636;
        uint vw_234 = (*(&kernelContext_0)->vs_0)[_S632 + 16U];
        float _S637 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S632 + 16U]), _S619);
        accLo_0[2U] = _S637;
        float _S638 = fma(pj_29, attnHi_0(vw_234), _S620);
        accHi_0[2U] = _S638;
        uint vw_235 = (*(&kernelContext_0)->vs_0)[_S632 + 24U];
        float _S639 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S632 + 24U]), _S621);
        accLo_0[3U] = _S639;
        float _S640 = fma(pj_29, attnHi_0(vw_235), _S622);
        accHi_0[3U] = _S640;
        uint vw_236 = (*(&kernelContext_0)->vs_0)[_S632 + 32U];
        float _S641 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S632 + 32U]), _S623);
        accLo_0[4U] = _S641;
        float _S642 = fma(pj_29, attnHi_0(vw_236), _S624);
        accHi_0[4U] = _S642;
        uint vw_237 = (*(&kernelContext_0)->vs_0)[_S632 + 40U];
        float _S643 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S632 + 40U]), _S625);
        accLo_0[5U] = _S643;
        float _S644 = fma(pj_29, attnHi_0(vw_237), _S626);
        accHi_0[5U] = _S644;
        uint vw_238 = (*(&kernelContext_0)->vs_0)[_S632 + 48U];
        float _S645 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S632 + 48U]), _S627);
        accLo_0[6U] = _S645;
        float _S646 = fma(pj_29, attnHi_0(vw_238), _S628);
        accHi_0[6U] = _S646;
        uint vw_239 = (*(&kernelContext_0)->vs_0)[_S632 + 56U];
        float _S647 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S632 + 56U]), _S629);
        accLo_0[7U] = _S647;
        float _S648 = fma(pj_29, attnHi_0(vw_239), _S630);
        accHi_0[7U] = _S648;
        float pj_30 = (*(&kernelContext_0)->ps_0)[_S86];
        float _S649 = _S631 + (*(&kernelContext_0)->ps_0)[_S86];
        uint _S650 = 1950U + kk_0;
        uint vw_240 = (*(&kernelContext_0)->vs_0)[_S650];
        float _S651 = fma((*(&kernelContext_0)->ps_0)[_S86], attnLo_0((*(&kernelContext_0)->vs_0)[_S650]), _S633);
        accLo_0[0U] = _S651;
        float _S652 = fma(pj_30, attnHi_0(vw_240), _S634);
        accHi_0[0U] = _S652;
        uint vw_241 = (*(&kernelContext_0)->vs_0)[_S650 + 8U];
        float _S653 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S650 + 8U]), _S635);
        accLo_0[1U] = _S653;
        float _S654 = fma(pj_30, attnHi_0(vw_241), _S636);
        accHi_0[1U] = _S654;
        uint vw_242 = (*(&kernelContext_0)->vs_0)[_S650 + 16U];
        float _S655 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S650 + 16U]), _S637);
        accLo_0[2U] = _S655;
        float _S656 = fma(pj_30, attnHi_0(vw_242), _S638);
        accHi_0[2U] = _S656;
        uint vw_243 = (*(&kernelContext_0)->vs_0)[_S650 + 24U];
        float _S657 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S650 + 24U]), _S639);
        accLo_0[3U] = _S657;
        float _S658 = fma(pj_30, attnHi_0(vw_243), _S640);
        accHi_0[3U] = _S658;
        uint vw_244 = (*(&kernelContext_0)->vs_0)[_S650 + 32U];
        float _S659 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S650 + 32U]), _S641);
        accLo_0[4U] = _S659;
        float _S660 = fma(pj_30, attnHi_0(vw_244), _S642);
        accHi_0[4U] = _S660;
        uint vw_245 = (*(&kernelContext_0)->vs_0)[_S650 + 40U];
        float _S661 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S650 + 40U]), _S643);
        accLo_0[5U] = _S661;
        float _S662 = fma(pj_30, attnHi_0(vw_245), _S644);
        accHi_0[5U] = _S662;
        uint vw_246 = (*(&kernelContext_0)->vs_0)[_S650 + 48U];
        float _S663 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S650 + 48U]), _S645);
        accLo_0[6U] = _S663;
        float _S664 = fma(pj_30, attnHi_0(vw_246), _S646);
        accHi_0[6U] = _S664;
        uint vw_247 = (*(&kernelContext_0)->vs_0)[_S650 + 56U];
        float _S665 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S650 + 56U]), _S647);
        accLo_0[7U] = _S665;
        float _S666 = fma(pj_30, attnHi_0(vw_247), _S648);
        accHi_0[7U] = _S666;
        float pj_31 = (*(&kernelContext_0)->ps_0)[_S87];
        float l_1 = _S649 + (*(&kernelContext_0)->ps_0)[_S87];
        uint _S667 = 2015U + kk_0;
        uint vw_248 = (*(&kernelContext_0)->vs_0)[_S667];
        accLo_0[0U] = fma((*(&kernelContext_0)->ps_0)[_S87], attnLo_0((*(&kernelContext_0)->vs_0)[_S667]), _S651);
        accHi_0[0U] = fma(pj_31, attnHi_0(vw_248), _S652);
        uint vw_249 = (*(&kernelContext_0)->vs_0)[_S667 + 8U];
        accLo_0[1U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S667 + 8U]), _S653);
        accHi_0[1U] = fma(pj_31, attnHi_0(vw_249), _S654);
        uint vw_250 = (*(&kernelContext_0)->vs_0)[_S667 + 16U];
        accLo_0[2U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S667 + 16U]), _S655);
        accHi_0[2U] = fma(pj_31, attnHi_0(vw_250), _S656);
        uint vw_251 = (*(&kernelContext_0)->vs_0)[_S667 + 24U];
        accLo_0[3U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S667 + 24U]), _S657);
        accHi_0[3U] = fma(pj_31, attnHi_0(vw_251), _S658);
        uint vw_252 = (*(&kernelContext_0)->vs_0)[_S667 + 32U];
        accLo_0[4U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S667 + 32U]), _S659);
        accHi_0[4U] = fma(pj_31, attnHi_0(vw_252), _S660);
        uint vw_253 = (*(&kernelContext_0)->vs_0)[_S667 + 40U];
        accLo_0[5U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S667 + 40U]), _S661);
        accHi_0[5U] = fma(pj_31, attnHi_0(vw_253), _S662);
        uint vw_254 = (*(&kernelContext_0)->vs_0)[_S667 + 48U];
        accLo_0[6U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S667 + 48U]), _S663);
        accHi_0[6U] = fma(pj_31, attnHi_0(vw_254), _S664);
        uint vw_255 = (*(&kernelContext_0)->vs_0)[_S667 + 56U];
        accLo_0[7U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S667 + 56U]), _S665);
        accHi_0[7U] = fma(pj_31, attnHi_0(vw_255), _S666);
        b_0 = b_0 + 32U;
        m_0 = _S88;
        l_0 = l_1;
    }
    if(!live_0)
    {
        return;
    }
    uint row_1 = head_0 * (&kernelContext_0)->p_0->queries_0 + qi_0;
    if(((&kernelContext_0)->p_0->splits_1) > 1U)
    {
        uint rows_1 = p_1->heads_0 * (&kernelContext_0)->p_0->queries_0;
        uint pr_30 = split_0 * rows_1 + row_1;
        uint _S668 = attnAccBase_0((&kernelContext_0)->p_0->splits_1, rows_1) + pr_30 * (&kernelContext_0)->p_0->headDim_0;
        if(kk_0 == 0U)
        {
            uint _S669 = 2U * pr_30;
            *((&kernelContext_0)->partials_0+_S669) = m_0;
            *((&kernelContext_0)->partials_0+(_S669 + 1U)) = l_0;
        }
        if(kk_0 < _S22)
        {
            uint _S670 = _S668 + 2U * kk_0;
            float device* _S671 = (&kernelContext_0)->partials_0+_S670;
            bool _S672 = l_0 == 0.0;
            if(_S672)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[0U];
            }
            *_S671 = m_0;
            float device* _S673 = (&kernelContext_0)->partials_0+(_S670 + 1U);
            if(_S672)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[0U];
            }
            *_S673 = m_0;
        }
        uint pair_0 = kk_0 + 8U;
        if(pair_0 < _S22)
        {
            uint _S674 = _S668 + 2U * pair_0;
            float device* _S675 = (&kernelContext_0)->partials_0+_S674;
            bool _S676 = l_0 == 0.0;
            if(_S676)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[1U];
            }
            *_S675 = m_0;
            float device* _S677 = (&kernelContext_0)->partials_0+(_S674 + 1U);
            if(_S676)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[1U];
            }
            *_S677 = m_0;
        }
        uint pair_1 = kk_0 + 16U;
        if(pair_1 < _S22)
        {
            uint _S678 = _S668 + 2U * pair_1;
            float device* _S679 = (&kernelContext_0)->partials_0+_S678;
            bool _S680 = l_0 == 0.0;
            if(_S680)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[2U];
            }
            *_S679 = m_0;
            float device* _S681 = (&kernelContext_0)->partials_0+(_S678 + 1U);
            if(_S680)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[2U];
            }
            *_S681 = m_0;
        }
        uint pair_2 = kk_0 + 24U;
        if(pair_2 < _S22)
        {
            uint _S682 = _S668 + 2U * pair_2;
            float device* _S683 = (&kernelContext_0)->partials_0+_S682;
            bool _S684 = l_0 == 0.0;
            if(_S684)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[3U];
            }
            *_S683 = m_0;
            float device* _S685 = (&kernelContext_0)->partials_0+(_S682 + 1U);
            if(_S684)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[3U];
            }
            *_S685 = m_0;
        }
        uint pair_3 = kk_0 + 32U;
        if(pair_3 < _S22)
        {
            uint _S686 = _S668 + 2U * pair_3;
            float device* _S687 = (&kernelContext_0)->partials_0+_S686;
            bool _S688 = l_0 == 0.0;
            if(_S688)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[4U];
            }
            *_S687 = m_0;
            float device* _S689 = (&kernelContext_0)->partials_0+(_S686 + 1U);
            if(_S688)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[4U];
            }
            *_S689 = m_0;
        }
        uint pair_4 = kk_0 + 40U;
        if(pair_4 < _S22)
        {
            uint _S690 = _S668 + 2U * pair_4;
            float device* _S691 = (&kernelContext_0)->partials_0+_S690;
            bool _S692 = l_0 == 0.0;
            if(_S692)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[5U];
            }
            *_S691 = m_0;
            float device* _S693 = (&kernelContext_0)->partials_0+(_S690 + 1U);
            if(_S692)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[5U];
            }
            *_S693 = m_0;
        }
        uint pair_5 = kk_0 + 48U;
        if(pair_5 < _S22)
        {
            uint _S694 = _S668 + 2U * pair_5;
            float device* _S695 = (&kernelContext_0)->partials_0+_S694;
            bool _S696 = l_0 == 0.0;
            if(_S696)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[6U];
            }
            *_S695 = m_0;
            float device* _S697 = (&kernelContext_0)->partials_0+(_S694 + 1U);
            if(_S696)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[6U];
            }
            *_S697 = m_0;
        }
        uint pair_6 = kk_0 + 56U;
        if(pair_6 < _S22)
        {
            uint _S698 = _S668 + 2U * pair_6;
            float device* _S699 = (&kernelContext_0)->partials_0+_S698;
            bool _S700 = l_0 == 0.0;
            if(_S700)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[7U];
            }
            *_S699 = m_0;
            float device* _S701 = (&kernelContext_0)->partials_0+(_S698 + 1U);
            if(_S700)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[7U];
            }
            *_S701 = m_0;
        }
        return;
    }
    if(((&kernelContext_0)->p_0->writeLse_0) != 0U)
    {
        ok_0 = kk_0 == 0U;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        float device* _S702 = (&kernelContext_0)->lse_0+row_1;
        if(l_0 == 0.0)
        {
            m_0 = _S33;
        }
        else
        {
            m_0 = m_0 + log(l_0);
        }
        *_S702 = m_0;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                if(kk_0 >= _S22)
                {
                    break;
                }
                bool _S703 = l_0 == 0.0;
                if(_S703)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[0U] / l_0;
                }
                if(_S703)
                {
                    hi_1 = 0.0;
                }
                else
                {
                    hi_1 = accHi_0[0U] / l_0;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S22 + kk_0)) = attnPack_0(s_0, hi_1);
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_7 = kk_0 + 8U;
                if(pair_7 >= _S22)
                {
                    break;
                }
                bool _S704 = l_0 == 0.0;
                if(_S704)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[1U] / l_0;
                }
                if(_S704)
                {
                    hi_1 = 0.0;
                }
                else
                {
                    hi_1 = accHi_0[1U] / l_0;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S22 + pair_7)) = attnPack_0(s_0, hi_1);
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_8 = kk_0 + 16U;
                if(pair_8 >= _S22)
                {
                    break;
                }
                bool _S705 = l_0 == 0.0;
                if(_S705)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[2U] / l_0;
                }
                if(_S705)
                {
                    hi_1 = 0.0;
                }
                else
                {
                    hi_1 = accHi_0[2U] / l_0;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S22 + pair_8)) = attnPack_0(s_0, hi_1);
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_9 = kk_0 + 24U;
                if(pair_9 >= _S22)
                {
                    break;
                }
                bool _S706 = l_0 == 0.0;
                if(_S706)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[3U] / l_0;
                }
                if(_S706)
                {
                    hi_1 = 0.0;
                }
                else
                {
                    hi_1 = accHi_0[3U] / l_0;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S22 + pair_9)) = attnPack_0(s_0, hi_1);
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_10 = kk_0 + 32U;
                if(pair_10 >= _S22)
                {
                    break;
                }
                bool _S707 = l_0 == 0.0;
                if(_S707)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[4U] / l_0;
                }
                if(_S707)
                {
                    hi_1 = 0.0;
                }
                else
                {
                    hi_1 = accHi_0[4U] / l_0;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S22 + pair_10)) = attnPack_0(s_0, hi_1);
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_11 = kk_0 + 40U;
                if(pair_11 >= _S22)
                {
                    break;
                }
                bool _S708 = l_0 == 0.0;
                if(_S708)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[5U] / l_0;
                }
                if(_S708)
                {
                    hi_1 = 0.0;
                }
                else
                {
                    hi_1 = accHi_0[5U] / l_0;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S22 + pair_11)) = attnPack_0(s_0, hi_1);
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_12 = kk_0 + 48U;
                if(pair_12 >= _S22)
                {
                    break;
                }
                bool _S709 = l_0 == 0.0;
                if(_S709)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[6U] / l_0;
                }
                if(_S709)
                {
                    hi_1 = 0.0;
                }
                else
                {
                    hi_1 = accHi_0[6U] / l_0;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S22 + pair_12)) = attnPack_0(s_0, hi_1);
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_13 = kk_0 + 56U;
                if(pair_13 >= _S22)
                {
                    break;
                }
                bool _S710 = l_0 == 0.0;
                if(_S710)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[7U] / l_0;
                }
                if(_S710)
                {
                    hi_1 = 0.0;
                }
                else
                {
                    hi_1 = accHi_0[7U] / l_0;
                }
                *((&kernelContext_0)->o_0+(row_1 * _S22 + pair_13)) = attnPack_0(s_0, hi_1);
                break;
            }
            break;
        }
        break;
    }
    return;
}

