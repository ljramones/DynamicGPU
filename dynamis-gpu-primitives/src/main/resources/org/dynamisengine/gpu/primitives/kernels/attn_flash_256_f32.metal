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
    float device* o_0;
    array<uint, int(2064)> threadgroup* qs_0;
    array<uint, int(2064)> threadgroup* ks_0;
    array<uint, int(2064)> threadgroup* vs_0;
    array<float, int(256)> threadgroup* ss_0;
    array<float, int(256)> threadgroup* ps_0;
};

[[kernel]] void attn_flash_256_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(8)]], uint device* q_1 [[buffer(0)]], uint device* lens_2 [[buffer(7)]], uint device* table_1 [[buffer(5)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partials_1 [[buffer(4)]], float device* lse_1 [[buffer(6)]], float device* o_1 [[buffer(3)]])
{
    uint _S5;
    uint _S6;
    uint _S7;
    uint _S8;
    uint _S9;
    uint _S10;
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
    threadgroup array<uint, int(2064)> qs_1;
    (&kernelContext_0)->qs_0 = &qs_1;
    threadgroup array<uint, int(2064)> ks_1;
    (&kernelContext_0)->ks_0 = &ks_1;
    threadgroup array<uint, int(2064)> vs_1;
    (&kernelContext_0)->vs_0 = &vs_1;
    threadgroup array<float, int(256)> ss_1;
    (&kernelContext_0)->ss_0 = &ss_1;
    threadgroup array<float, int(256)> ps_1;
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
    uint _S11 = kvHead_0 * hpk_0;
    uint qi_0;
    if(live_0)
    {
        uint _S12 = r_0 / (&kernelContext_0)->p_0->queries_0;
        qi_0 = _S12;
    }
    else
    {
        qi_0 = 0U;
    }
    uint head_0 = _S11 + qi_0;
    if(live_0)
    {
        uint _S13 = r_0 % (&kernelContext_0)->p_0->queries_0;
        qi_0 = _S13;
    }
    else
    {
        qi_0 = 0U;
    }
    bool ok_0;
    uint w_2;
    uint _S14 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint offset_0 = (&kernelContext_0)->p_0->keys_0 - (&kernelContext_0)->p_0->queries_0;
    uint e_0 = t_0;
    for(;;)
    {
        if(e_0 < 2048U)
        {
        }
        else
        {
            break;
        }
        uint row_0 = r0_0 + e_0 / 128U;
        uint pr_0 = e_0 % 128U;
        if(row_0 < rowsKv_0)
        {
            ok_0 = pr_0 < _S14;
        }
        else
        {
            ok_0 = false;
        }
        if(ok_0)
        {
            uint _S15 = row_0 / (&kernelContext_0)->p_0->queries_0;
            uint h_0 = _S11 + _S15;
            uint i_1 = row_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((h_0 * (&kernelContext_0)->p_0->queries_0 + i_1) * _S14 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_0 / 128U * 129U + pr_0] = w_2;
        e_0 = e_0 + 128U;
    }
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S16 = min((&kernelContext_0)->p_0->keys_0, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    uint rLast_0 = min(r0_0 + 16U, rowsKv_0) - 1U;
    uint _S17 = (&kernelContext_0)->p_0->lengthGroup_1;
    uint _S18 = r0_0 / (&kernelContext_0)->p_0->queries_0;
    uint _S19 = _S11 + _S18;
    uint _S20 = rLast_0 / (&kernelContext_0)->p_0->queries_0;
    uint kEnd_0 = attnKeyEnd_0((&kernelContext_0)->lens_1, _S17, _S19, _S11 + _S20, _S16);
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
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
        uint _S24 = head_0 / (&kernelContext_0)->p_0->lengthGroup_1;
        limit_0 = min(limit_0, *((&kernelContext_0)->lens_1+_S24));
    }
    float _S25 = dg_negative_infinity_0();
    thread array<float, int(16)> accLo_0;
    thread array<float, int(16)> accHi_0;
    float hi_0;
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
    accLo_0[8U] = 0.0;
    accHi_0[8U] = 0.0;
    accLo_0[9U] = 0.0;
    accHi_0[9U] = 0.0;
    accLo_0[10U] = 0.0;
    accHi_0[10U] = 0.0;
    accLo_0[11U] = 0.0;
    accHi_0[11U] = 0.0;
    accLo_0[12U] = 0.0;
    accHi_0[12U] = 0.0;
    accLo_0[13U] = 0.0;
    accHi_0[13U] = 0.0;
    accLo_0[14U] = 0.0;
    accHi_0[14U] = 0.0;
    accLo_0[15U] = 0.0;
    accHi_0[15U] = 0.0;
    uint b_0 = kStart_0;
    float m_0 = _S25;
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
        uint at_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        e_0 = t_0;
        for(;;)
        {
            if(e_0 < 2048U)
            {
            }
            else
            {
                break;
            }
            uint j_0 = e_0 / 128U;
            uint pr_1 = e_0 % 128U;
            uint key_0 = b_0 + j_0;
            if(key_0 < w_2)
            {
                ok_0 = pr_1 < _S14;
            }
            else
            {
                ok_0 = false;
            }
            if(ok_0)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    at_0 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * _S14 + pr_1;
                }
                else
                {
                    uint _S26 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S27 = (*((&kernelContext_0)->table_0+_S26) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S28 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
                    at_0 = (_S27 + _S28) * _S14 + pr_1;
                }
            }
            else
            {
                at_0 = 0U;
            }
            uint _S29 = j_0 * 129U + pr_1;
            if(ok_0)
            {
                p0_0 = *((&kernelContext_0)->k_0+at_0);
            }
            else
            {
                p0_0 = 0U;
            }
            (*(&kernelContext_0)->ks_0)[_S29] = p0_0;
            uint _S30;
            if(ok_0)
            {
                _S30 = *((&kernelContext_0)->v_0+at_0);
            }
            else
            {
                _S30 = 0U;
            }
            (*(&kernelContext_0)->vs_0)[_S29] = _S30;
            e_0 = e_0 + 128U;
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
                        if(at_0 < 128U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S31 = lr_0 * 129U;
                        uint _S32 = kk_0 * 129U;
                        uint pr_2 = at_0 + 1U;
                        uint pr_3 = at_0 + 2U;
                        uint pr_4 = at_0 + 3U;
                        uint pr_5 = at_0 + 4U;
                        uint pr_6 = at_0 + 5U;
                        uint pr_7 = at_0 + 6U;
                        uint pr_8 = at_0 + 7U;
                        float s_1 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S31 + pr_8]), attnHi_0((*(&kernelContext_0)->ks_0)[_S32 + pr_8]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S31 + pr_8]), attnLo_0((*(&kernelContext_0)->ks_0)[_S32 + pr_8]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S31 + pr_7]), attnHi_0((*(&kernelContext_0)->ks_0)[_S32 + pr_7]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S31 + pr_7]), attnLo_0((*(&kernelContext_0)->ks_0)[_S32 + pr_7]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S31 + pr_6]), attnHi_0((*(&kernelContext_0)->ks_0)[_S32 + pr_6]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S31 + pr_6]), attnLo_0((*(&kernelContext_0)->ks_0)[_S32 + pr_6]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S31 + pr_5]), attnHi_0((*(&kernelContext_0)->ks_0)[_S32 + pr_5]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S31 + pr_5]), attnLo_0((*(&kernelContext_0)->ks_0)[_S32 + pr_5]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S31 + pr_4]), attnHi_0((*(&kernelContext_0)->ks_0)[_S32 + pr_4]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S31 + pr_4]), attnLo_0((*(&kernelContext_0)->ks_0)[_S32 + pr_4]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S31 + pr_3]), attnHi_0((*(&kernelContext_0)->ks_0)[_S32 + pr_3]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S31 + pr_3]), attnLo_0((*(&kernelContext_0)->ks_0)[_S32 + pr_3]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S31 + pr_2]), attnHi_0((*(&kernelContext_0)->ks_0)[_S32 + pr_2]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S31 + pr_2]), attnLo_0((*(&kernelContext_0)->ks_0)[_S32 + pr_2]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S31 + at_0]), attnHi_0((*(&kernelContext_0)->ks_0)[_S32 + at_0]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S31 + at_0]), attnLo_0((*(&kernelContext_0)->ks_0)[_S32 + at_0]), s_0))))))))))))))));
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
                    uint _S33 = lr_0 * 16U;
                    _S5 = _S33;
                    uint _S34 = _S33 + kk_0;
                    _S9 = _S34;
                    _S6 = _S34;
                    if(visible_0)
                    {
                        hi_0 = s_0 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        hi_0 = _S25;
                    }
                    (*(&kernelContext_0)->ss_0)[_S34] = hi_0;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_1 = kk_0 + 8U;
                    _S7 = j_1;
                    uint key_2 = b_0 + j_1;
                    p0_0 = 0U;
                    float s_2 = 0.0;
                    for(;;)
                    {
                        if(p0_0 < 128U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S35 = lr_0 * 129U;
                        uint _S36 = j_1 * 129U;
                        uint pr_9 = p0_0 + 1U;
                        uint pr_10 = p0_0 + 2U;
                        uint pr_11 = p0_0 + 3U;
                        uint pr_12 = p0_0 + 4U;
                        uint pr_13 = p0_0 + 5U;
                        uint pr_14 = p0_0 + 6U;
                        uint pr_15 = p0_0 + 7U;
                        float s_3 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S35 + pr_15]), attnHi_0((*(&kernelContext_0)->ks_0)[_S36 + pr_15]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S35 + pr_15]), attnLo_0((*(&kernelContext_0)->ks_0)[_S36 + pr_15]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S35 + pr_14]), attnHi_0((*(&kernelContext_0)->ks_0)[_S36 + pr_14]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S35 + pr_14]), attnLo_0((*(&kernelContext_0)->ks_0)[_S36 + pr_14]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S35 + pr_13]), attnHi_0((*(&kernelContext_0)->ks_0)[_S36 + pr_13]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S35 + pr_13]), attnLo_0((*(&kernelContext_0)->ks_0)[_S36 + pr_13]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S35 + pr_12]), attnHi_0((*(&kernelContext_0)->ks_0)[_S36 + pr_12]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S35 + pr_12]), attnLo_0((*(&kernelContext_0)->ks_0)[_S36 + pr_12]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S35 + pr_11]), attnHi_0((*(&kernelContext_0)->ks_0)[_S36 + pr_11]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S35 + pr_11]), attnLo_0((*(&kernelContext_0)->ks_0)[_S36 + pr_11]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S35 + pr_10]), attnHi_0((*(&kernelContext_0)->ks_0)[_S36 + pr_10]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S35 + pr_10]), attnLo_0((*(&kernelContext_0)->ks_0)[_S36 + pr_10]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S35 + pr_9]), attnHi_0((*(&kernelContext_0)->ks_0)[_S36 + pr_9]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S35 + pr_9]), attnLo_0((*(&kernelContext_0)->ks_0)[_S36 + pr_9]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S35 + p0_0]), attnHi_0((*(&kernelContext_0)->ks_0)[_S36 + p0_0]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S35 + p0_0]), attnLo_0((*(&kernelContext_0)->ks_0)[_S36 + p0_0]), s_2))))))))))))))));
                        p0_0 = p0_0 + 8U;
                        s_2 = s_3;
                    }
                    bool _S37;
                    if(live_0)
                    {
                        _S37 = key_2 < w_2;
                    }
                    else
                    {
                        _S37 = false;
                    }
                    bool visible_1;
                    if(_S37)
                    {
                        visible_1 = key_2 < limit_0;
                    }
                    else
                    {
                        visible_1 = false;
                    }
                    uint _S38 = _S5 + j_1;
                    _S10 = _S38;
                    _S8 = _S38;
                    float _S39;
                    if(visible_1)
                    {
                        _S39 = s_2 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S39 = _S25;
                    }
                    (*(&kernelContext_0)->ss_0)[_S38] = _S39;
                    break;
                }
                break;
            }
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S40 = _S5 + 1U;
        uint _S41 = _S5 + 2U;
        uint _S42 = _S5 + 3U;
        uint _S43 = _S5 + 4U;
        uint _S44 = _S5 + 5U;
        uint _S45 = _S5 + 6U;
        uint _S46 = _S5 + 7U;
        uint _S47 = _S5 + 8U;
        uint _S48 = _S5 + 9U;
        uint _S49 = _S5 + 10U;
        uint _S50 = _S5 + 11U;
        uint _S51 = _S5 + 12U;
        uint _S52 = _S5 + 13U;
        uint _S53 = _S5 + 14U;
        uint _S54 = _S5 + 15U;
        float _S55 = max(m_0, max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(_S25, (*(&kernelContext_0)->ss_0)[_S5]), (*(&kernelContext_0)->ss_0)[_S40]), (*(&kernelContext_0)->ss_0)[_S41]), (*(&kernelContext_0)->ss_0)[_S42]), (*(&kernelContext_0)->ss_0)[_S43]), (*(&kernelContext_0)->ss_0)[_S44]), (*(&kernelContext_0)->ss_0)[_S45]), (*(&kernelContext_0)->ss_0)[_S46]), (*(&kernelContext_0)->ss_0)[_S47]), (*(&kernelContext_0)->ss_0)[_S48]), (*(&kernelContext_0)->ss_0)[_S49]), (*(&kernelContext_0)->ss_0)[_S50]), (*(&kernelContext_0)->ss_0)[_S51]), (*(&kernelContext_0)->ss_0)[_S52]), (*(&kernelContext_0)->ss_0)[_S53]), (*(&kernelContext_0)->ss_0)[_S54]));
        float alpha_0;
        if(m_0 == _S25)
        {
            alpha_0 = 0.0;
        }
        else
        {
            alpha_0 = exp(m_0 - _S55);
        }
        float s_4 = (*(&kernelContext_0)->ss_0)[_S9];
        float _S56;
        if(((*(&kernelContext_0)->ss_0)[_S9]) == _S25)
        {
            _S56 = 0.0;
        }
        else
        {
            _S56 = exp(s_4 - _S55);
        }
        (*(&kernelContext_0)->ps_0)[_S6] = _S56;
        float s_5 = (*(&kernelContext_0)->ss_0)[_S10];
        float _S57;
        if(((*(&kernelContext_0)->ss_0)[_S10]) == _S25)
        {
            _S57 = 0.0;
        }
        else
        {
            _S57 = exp(s_5 - _S55);
        }
        (*(&kernelContext_0)->ps_0)[_S8] = _S57;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S58 = l_0 * alpha_0;
        float _S59 = accLo_0[0U] * alpha_0;
        accLo_0[0U] = _S59;
        float _S60 = accHi_0[0U] * alpha_0;
        accHi_0[0U] = _S60;
        float _S61 = accLo_0[1U] * alpha_0;
        accLo_0[1U] = _S61;
        float _S62 = accHi_0[1U] * alpha_0;
        accHi_0[1U] = _S62;
        float _S63 = accLo_0[2U] * alpha_0;
        accLo_0[2U] = _S63;
        float _S64 = accHi_0[2U] * alpha_0;
        accHi_0[2U] = _S64;
        float _S65 = accLo_0[3U] * alpha_0;
        accLo_0[3U] = _S65;
        float _S66 = accHi_0[3U] * alpha_0;
        accHi_0[3U] = _S66;
        float _S67 = accLo_0[4U] * alpha_0;
        accLo_0[4U] = _S67;
        float _S68 = accHi_0[4U] * alpha_0;
        accHi_0[4U] = _S68;
        float _S69 = accLo_0[5U] * alpha_0;
        accLo_0[5U] = _S69;
        float _S70 = accHi_0[5U] * alpha_0;
        accHi_0[5U] = _S70;
        float _S71 = accLo_0[6U] * alpha_0;
        accLo_0[6U] = _S71;
        float _S72 = accHi_0[6U] * alpha_0;
        accHi_0[6U] = _S72;
        float _S73 = accLo_0[7U] * alpha_0;
        accLo_0[7U] = _S73;
        float _S74 = accHi_0[7U] * alpha_0;
        accHi_0[7U] = _S74;
        float _S75 = accLo_0[8U] * alpha_0;
        accLo_0[8U] = _S75;
        float _S76 = accHi_0[8U] * alpha_0;
        accHi_0[8U] = _S76;
        float _S77 = accLo_0[9U] * alpha_0;
        accLo_0[9U] = _S77;
        float _S78 = accHi_0[9U] * alpha_0;
        accHi_0[9U] = _S78;
        float _S79 = accLo_0[10U] * alpha_0;
        accLo_0[10U] = _S79;
        float _S80 = accHi_0[10U] * alpha_0;
        accHi_0[10U] = _S80;
        float _S81 = accLo_0[11U] * alpha_0;
        accLo_0[11U] = _S81;
        float _S82 = accHi_0[11U] * alpha_0;
        accHi_0[11U] = _S82;
        float _S83 = accLo_0[12U] * alpha_0;
        accLo_0[12U] = _S83;
        float _S84 = accHi_0[12U] * alpha_0;
        accHi_0[12U] = _S84;
        float _S85 = accLo_0[13U] * alpha_0;
        accLo_0[13U] = _S85;
        float _S86 = accHi_0[13U] * alpha_0;
        accHi_0[13U] = _S86;
        float _S87 = accLo_0[14U] * alpha_0;
        accLo_0[14U] = _S87;
        float _S88 = accHi_0[14U] * alpha_0;
        accHi_0[14U] = _S88;
        float _S89 = accLo_0[15U] * alpha_0;
        accLo_0[15U] = _S89;
        float _S90 = accHi_0[15U] * alpha_0;
        accHi_0[15U] = _S90;
        float pj_0 = (*(&kernelContext_0)->ps_0)[_S5];
        float _S91 = _S58 + (*(&kernelContext_0)->ps_0)[_S5];
        uint vw_0 = (*(&kernelContext_0)->vs_0)[kk_0];
        float _S92 = fma((*(&kernelContext_0)->ps_0)[_S5], attnLo_0((*(&kernelContext_0)->vs_0)[kk_0]), _S59);
        accLo_0[0U] = _S92;
        float _S93 = fma(pj_0, attnHi_0(vw_0), _S60);
        accHi_0[0U] = _S93;
        uint vw_1 = (*(&kernelContext_0)->vs_0)[_S7];
        float _S94 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S7]), _S61);
        accLo_0[1U] = _S94;
        float _S95 = fma(pj_0, attnHi_0(vw_1), _S62);
        accHi_0[1U] = _S95;
        uint vw_2 = (*(&kernelContext_0)->vs_0)[kk_0 + 16U];
        float _S96 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 16U]), _S63);
        accLo_0[2U] = _S96;
        float _S97 = fma(pj_0, attnHi_0(vw_2), _S64);
        accHi_0[2U] = _S97;
        uint vw_3 = (*(&kernelContext_0)->vs_0)[kk_0 + 24U];
        float _S98 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 24U]), _S65);
        accLo_0[3U] = _S98;
        float _S99 = fma(pj_0, attnHi_0(vw_3), _S66);
        accHi_0[3U] = _S99;
        uint vw_4 = (*(&kernelContext_0)->vs_0)[kk_0 + 32U];
        float _S100 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 32U]), _S67);
        accLo_0[4U] = _S100;
        float _S101 = fma(pj_0, attnHi_0(vw_4), _S68);
        accHi_0[4U] = _S101;
        uint vw_5 = (*(&kernelContext_0)->vs_0)[kk_0 + 40U];
        float _S102 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 40U]), _S69);
        accLo_0[5U] = _S102;
        float _S103 = fma(pj_0, attnHi_0(vw_5), _S70);
        accHi_0[5U] = _S103;
        uint vw_6 = (*(&kernelContext_0)->vs_0)[kk_0 + 48U];
        float _S104 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 48U]), _S71);
        accLo_0[6U] = _S104;
        float _S105 = fma(pj_0, attnHi_0(vw_6), _S72);
        accHi_0[6U] = _S105;
        uint vw_7 = (*(&kernelContext_0)->vs_0)[kk_0 + 56U];
        float _S106 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 56U]), _S73);
        accLo_0[7U] = _S106;
        float _S107 = fma(pj_0, attnHi_0(vw_7), _S74);
        accHi_0[7U] = _S107;
        uint vw_8 = (*(&kernelContext_0)->vs_0)[kk_0 + 64U];
        float _S108 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 64U]), _S75);
        accLo_0[8U] = _S108;
        float _S109 = fma(pj_0, attnHi_0(vw_8), _S76);
        accHi_0[8U] = _S109;
        uint vw_9 = (*(&kernelContext_0)->vs_0)[kk_0 + 72U];
        float _S110 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 72U]), _S77);
        accLo_0[9U] = _S110;
        float _S111 = fma(pj_0, attnHi_0(vw_9), _S78);
        accHi_0[9U] = _S111;
        uint vw_10 = (*(&kernelContext_0)->vs_0)[kk_0 + 80U];
        float _S112 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 80U]), _S79);
        accLo_0[10U] = _S112;
        float _S113 = fma(pj_0, attnHi_0(vw_10), _S80);
        accHi_0[10U] = _S113;
        uint vw_11 = (*(&kernelContext_0)->vs_0)[kk_0 + 88U];
        float _S114 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 88U]), _S81);
        accLo_0[11U] = _S114;
        float _S115 = fma(pj_0, attnHi_0(vw_11), _S82);
        accHi_0[11U] = _S115;
        uint vw_12 = (*(&kernelContext_0)->vs_0)[kk_0 + 96U];
        float _S116 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 96U]), _S83);
        accLo_0[12U] = _S116;
        float _S117 = fma(pj_0, attnHi_0(vw_12), _S84);
        accHi_0[12U] = _S117;
        uint vw_13 = (*(&kernelContext_0)->vs_0)[kk_0 + 104U];
        float _S118 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 104U]), _S85);
        accLo_0[13U] = _S118;
        float _S119 = fma(pj_0, attnHi_0(vw_13), _S86);
        accHi_0[13U] = _S119;
        uint vw_14 = (*(&kernelContext_0)->vs_0)[kk_0 + 112U];
        float _S120 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 112U]), _S87);
        accLo_0[14U] = _S120;
        float _S121 = fma(pj_0, attnHi_0(vw_14), _S88);
        accHi_0[14U] = _S121;
        uint vw_15 = (*(&kernelContext_0)->vs_0)[kk_0 + 120U];
        float _S122 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 120U]), _S89);
        accLo_0[15U] = _S122;
        float _S123 = fma(pj_0, attnHi_0(vw_15), _S90);
        accHi_0[15U] = _S123;
        float pj_1 = (*(&kernelContext_0)->ps_0)[_S40];
        float _S124 = _S91 + (*(&kernelContext_0)->ps_0)[_S40];
        uint _S125 = 129U + kk_0;
        uint vw_16 = (*(&kernelContext_0)->vs_0)[_S125];
        float _S126 = fma((*(&kernelContext_0)->ps_0)[_S40], attnLo_0((*(&kernelContext_0)->vs_0)[_S125]), _S92);
        accLo_0[0U] = _S126;
        float _S127 = fma(pj_1, attnHi_0(vw_16), _S93);
        accHi_0[0U] = _S127;
        uint vw_17 = (*(&kernelContext_0)->vs_0)[_S125 + 8U];
        float _S128 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 8U]), _S94);
        accLo_0[1U] = _S128;
        float _S129 = fma(pj_1, attnHi_0(vw_17), _S95);
        accHi_0[1U] = _S129;
        uint vw_18 = (*(&kernelContext_0)->vs_0)[_S125 + 16U];
        float _S130 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 16U]), _S96);
        accLo_0[2U] = _S130;
        float _S131 = fma(pj_1, attnHi_0(vw_18), _S97);
        accHi_0[2U] = _S131;
        uint vw_19 = (*(&kernelContext_0)->vs_0)[_S125 + 24U];
        float _S132 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 24U]), _S98);
        accLo_0[3U] = _S132;
        float _S133 = fma(pj_1, attnHi_0(vw_19), _S99);
        accHi_0[3U] = _S133;
        uint vw_20 = (*(&kernelContext_0)->vs_0)[_S125 + 32U];
        float _S134 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 32U]), _S100);
        accLo_0[4U] = _S134;
        float _S135 = fma(pj_1, attnHi_0(vw_20), _S101);
        accHi_0[4U] = _S135;
        uint vw_21 = (*(&kernelContext_0)->vs_0)[_S125 + 40U];
        float _S136 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 40U]), _S102);
        accLo_0[5U] = _S136;
        float _S137 = fma(pj_1, attnHi_0(vw_21), _S103);
        accHi_0[5U] = _S137;
        uint vw_22 = (*(&kernelContext_0)->vs_0)[_S125 + 48U];
        float _S138 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 48U]), _S104);
        accLo_0[6U] = _S138;
        float _S139 = fma(pj_1, attnHi_0(vw_22), _S105);
        accHi_0[6U] = _S139;
        uint vw_23 = (*(&kernelContext_0)->vs_0)[_S125 + 56U];
        float _S140 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 56U]), _S106);
        accLo_0[7U] = _S140;
        float _S141 = fma(pj_1, attnHi_0(vw_23), _S107);
        accHi_0[7U] = _S141;
        uint vw_24 = (*(&kernelContext_0)->vs_0)[_S125 + 64U];
        float _S142 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 64U]), _S108);
        accLo_0[8U] = _S142;
        float _S143 = fma(pj_1, attnHi_0(vw_24), _S109);
        accHi_0[8U] = _S143;
        uint vw_25 = (*(&kernelContext_0)->vs_0)[_S125 + 72U];
        float _S144 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 72U]), _S110);
        accLo_0[9U] = _S144;
        float _S145 = fma(pj_1, attnHi_0(vw_25), _S111);
        accHi_0[9U] = _S145;
        uint vw_26 = (*(&kernelContext_0)->vs_0)[_S125 + 80U];
        float _S146 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 80U]), _S112);
        accLo_0[10U] = _S146;
        float _S147 = fma(pj_1, attnHi_0(vw_26), _S113);
        accHi_0[10U] = _S147;
        uint vw_27 = (*(&kernelContext_0)->vs_0)[_S125 + 88U];
        float _S148 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 88U]), _S114);
        accLo_0[11U] = _S148;
        float _S149 = fma(pj_1, attnHi_0(vw_27), _S115);
        accHi_0[11U] = _S149;
        uint vw_28 = (*(&kernelContext_0)->vs_0)[_S125 + 96U];
        float _S150 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 96U]), _S116);
        accLo_0[12U] = _S150;
        float _S151 = fma(pj_1, attnHi_0(vw_28), _S117);
        accHi_0[12U] = _S151;
        uint vw_29 = (*(&kernelContext_0)->vs_0)[_S125 + 104U];
        float _S152 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 104U]), _S118);
        accLo_0[13U] = _S152;
        float _S153 = fma(pj_1, attnHi_0(vw_29), _S119);
        accHi_0[13U] = _S153;
        uint vw_30 = (*(&kernelContext_0)->vs_0)[_S125 + 112U];
        float _S154 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 112U]), _S120);
        accLo_0[14U] = _S154;
        float _S155 = fma(pj_1, attnHi_0(vw_30), _S121);
        accHi_0[14U] = _S155;
        uint vw_31 = (*(&kernelContext_0)->vs_0)[_S125 + 120U];
        float _S156 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S125 + 120U]), _S122);
        accLo_0[15U] = _S156;
        float _S157 = fma(pj_1, attnHi_0(vw_31), _S123);
        accHi_0[15U] = _S157;
        float pj_2 = (*(&kernelContext_0)->ps_0)[_S41];
        float _S158 = _S124 + (*(&kernelContext_0)->ps_0)[_S41];
        uint _S159 = 258U + kk_0;
        uint vw_32 = (*(&kernelContext_0)->vs_0)[_S159];
        float _S160 = fma((*(&kernelContext_0)->ps_0)[_S41], attnLo_0((*(&kernelContext_0)->vs_0)[_S159]), _S126);
        accLo_0[0U] = _S160;
        float _S161 = fma(pj_2, attnHi_0(vw_32), _S127);
        accHi_0[0U] = _S161;
        uint vw_33 = (*(&kernelContext_0)->vs_0)[_S159 + 8U];
        float _S162 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 8U]), _S128);
        accLo_0[1U] = _S162;
        float _S163 = fma(pj_2, attnHi_0(vw_33), _S129);
        accHi_0[1U] = _S163;
        uint vw_34 = (*(&kernelContext_0)->vs_0)[_S159 + 16U];
        float _S164 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 16U]), _S130);
        accLo_0[2U] = _S164;
        float _S165 = fma(pj_2, attnHi_0(vw_34), _S131);
        accHi_0[2U] = _S165;
        uint vw_35 = (*(&kernelContext_0)->vs_0)[_S159 + 24U];
        float _S166 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 24U]), _S132);
        accLo_0[3U] = _S166;
        float _S167 = fma(pj_2, attnHi_0(vw_35), _S133);
        accHi_0[3U] = _S167;
        uint vw_36 = (*(&kernelContext_0)->vs_0)[_S159 + 32U];
        float _S168 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 32U]), _S134);
        accLo_0[4U] = _S168;
        float _S169 = fma(pj_2, attnHi_0(vw_36), _S135);
        accHi_0[4U] = _S169;
        uint vw_37 = (*(&kernelContext_0)->vs_0)[_S159 + 40U];
        float _S170 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 40U]), _S136);
        accLo_0[5U] = _S170;
        float _S171 = fma(pj_2, attnHi_0(vw_37), _S137);
        accHi_0[5U] = _S171;
        uint vw_38 = (*(&kernelContext_0)->vs_0)[_S159 + 48U];
        float _S172 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 48U]), _S138);
        accLo_0[6U] = _S172;
        float _S173 = fma(pj_2, attnHi_0(vw_38), _S139);
        accHi_0[6U] = _S173;
        uint vw_39 = (*(&kernelContext_0)->vs_0)[_S159 + 56U];
        float _S174 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 56U]), _S140);
        accLo_0[7U] = _S174;
        float _S175 = fma(pj_2, attnHi_0(vw_39), _S141);
        accHi_0[7U] = _S175;
        uint vw_40 = (*(&kernelContext_0)->vs_0)[_S159 + 64U];
        float _S176 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 64U]), _S142);
        accLo_0[8U] = _S176;
        float _S177 = fma(pj_2, attnHi_0(vw_40), _S143);
        accHi_0[8U] = _S177;
        uint vw_41 = (*(&kernelContext_0)->vs_0)[_S159 + 72U];
        float _S178 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 72U]), _S144);
        accLo_0[9U] = _S178;
        float _S179 = fma(pj_2, attnHi_0(vw_41), _S145);
        accHi_0[9U] = _S179;
        uint vw_42 = (*(&kernelContext_0)->vs_0)[_S159 + 80U];
        float _S180 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 80U]), _S146);
        accLo_0[10U] = _S180;
        float _S181 = fma(pj_2, attnHi_0(vw_42), _S147);
        accHi_0[10U] = _S181;
        uint vw_43 = (*(&kernelContext_0)->vs_0)[_S159 + 88U];
        float _S182 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 88U]), _S148);
        accLo_0[11U] = _S182;
        float _S183 = fma(pj_2, attnHi_0(vw_43), _S149);
        accHi_0[11U] = _S183;
        uint vw_44 = (*(&kernelContext_0)->vs_0)[_S159 + 96U];
        float _S184 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 96U]), _S150);
        accLo_0[12U] = _S184;
        float _S185 = fma(pj_2, attnHi_0(vw_44), _S151);
        accHi_0[12U] = _S185;
        uint vw_45 = (*(&kernelContext_0)->vs_0)[_S159 + 104U];
        float _S186 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 104U]), _S152);
        accLo_0[13U] = _S186;
        float _S187 = fma(pj_2, attnHi_0(vw_45), _S153);
        accHi_0[13U] = _S187;
        uint vw_46 = (*(&kernelContext_0)->vs_0)[_S159 + 112U];
        float _S188 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 112U]), _S154);
        accLo_0[14U] = _S188;
        float _S189 = fma(pj_2, attnHi_0(vw_46), _S155);
        accHi_0[14U] = _S189;
        uint vw_47 = (*(&kernelContext_0)->vs_0)[_S159 + 120U];
        float _S190 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S159 + 120U]), _S156);
        accLo_0[15U] = _S190;
        float _S191 = fma(pj_2, attnHi_0(vw_47), _S157);
        accHi_0[15U] = _S191;
        float pj_3 = (*(&kernelContext_0)->ps_0)[_S42];
        float _S192 = _S158 + (*(&kernelContext_0)->ps_0)[_S42];
        uint _S193 = 387U + kk_0;
        uint vw_48 = (*(&kernelContext_0)->vs_0)[_S193];
        float _S194 = fma((*(&kernelContext_0)->ps_0)[_S42], attnLo_0((*(&kernelContext_0)->vs_0)[_S193]), _S160);
        accLo_0[0U] = _S194;
        float _S195 = fma(pj_3, attnHi_0(vw_48), _S161);
        accHi_0[0U] = _S195;
        uint vw_49 = (*(&kernelContext_0)->vs_0)[_S193 + 8U];
        float _S196 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 8U]), _S162);
        accLo_0[1U] = _S196;
        float _S197 = fma(pj_3, attnHi_0(vw_49), _S163);
        accHi_0[1U] = _S197;
        uint vw_50 = (*(&kernelContext_0)->vs_0)[_S193 + 16U];
        float _S198 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 16U]), _S164);
        accLo_0[2U] = _S198;
        float _S199 = fma(pj_3, attnHi_0(vw_50), _S165);
        accHi_0[2U] = _S199;
        uint vw_51 = (*(&kernelContext_0)->vs_0)[_S193 + 24U];
        float _S200 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 24U]), _S166);
        accLo_0[3U] = _S200;
        float _S201 = fma(pj_3, attnHi_0(vw_51), _S167);
        accHi_0[3U] = _S201;
        uint vw_52 = (*(&kernelContext_0)->vs_0)[_S193 + 32U];
        float _S202 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 32U]), _S168);
        accLo_0[4U] = _S202;
        float _S203 = fma(pj_3, attnHi_0(vw_52), _S169);
        accHi_0[4U] = _S203;
        uint vw_53 = (*(&kernelContext_0)->vs_0)[_S193 + 40U];
        float _S204 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 40U]), _S170);
        accLo_0[5U] = _S204;
        float _S205 = fma(pj_3, attnHi_0(vw_53), _S171);
        accHi_0[5U] = _S205;
        uint vw_54 = (*(&kernelContext_0)->vs_0)[_S193 + 48U];
        float _S206 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 48U]), _S172);
        accLo_0[6U] = _S206;
        float _S207 = fma(pj_3, attnHi_0(vw_54), _S173);
        accHi_0[6U] = _S207;
        uint vw_55 = (*(&kernelContext_0)->vs_0)[_S193 + 56U];
        float _S208 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 56U]), _S174);
        accLo_0[7U] = _S208;
        float _S209 = fma(pj_3, attnHi_0(vw_55), _S175);
        accHi_0[7U] = _S209;
        uint vw_56 = (*(&kernelContext_0)->vs_0)[_S193 + 64U];
        float _S210 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 64U]), _S176);
        accLo_0[8U] = _S210;
        float _S211 = fma(pj_3, attnHi_0(vw_56), _S177);
        accHi_0[8U] = _S211;
        uint vw_57 = (*(&kernelContext_0)->vs_0)[_S193 + 72U];
        float _S212 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 72U]), _S178);
        accLo_0[9U] = _S212;
        float _S213 = fma(pj_3, attnHi_0(vw_57), _S179);
        accHi_0[9U] = _S213;
        uint vw_58 = (*(&kernelContext_0)->vs_0)[_S193 + 80U];
        float _S214 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 80U]), _S180);
        accLo_0[10U] = _S214;
        float _S215 = fma(pj_3, attnHi_0(vw_58), _S181);
        accHi_0[10U] = _S215;
        uint vw_59 = (*(&kernelContext_0)->vs_0)[_S193 + 88U];
        float _S216 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 88U]), _S182);
        accLo_0[11U] = _S216;
        float _S217 = fma(pj_3, attnHi_0(vw_59), _S183);
        accHi_0[11U] = _S217;
        uint vw_60 = (*(&kernelContext_0)->vs_0)[_S193 + 96U];
        float _S218 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 96U]), _S184);
        accLo_0[12U] = _S218;
        float _S219 = fma(pj_3, attnHi_0(vw_60), _S185);
        accHi_0[12U] = _S219;
        uint vw_61 = (*(&kernelContext_0)->vs_0)[_S193 + 104U];
        float _S220 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 104U]), _S186);
        accLo_0[13U] = _S220;
        float _S221 = fma(pj_3, attnHi_0(vw_61), _S187);
        accHi_0[13U] = _S221;
        uint vw_62 = (*(&kernelContext_0)->vs_0)[_S193 + 112U];
        float _S222 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 112U]), _S188);
        accLo_0[14U] = _S222;
        float _S223 = fma(pj_3, attnHi_0(vw_62), _S189);
        accHi_0[14U] = _S223;
        uint vw_63 = (*(&kernelContext_0)->vs_0)[_S193 + 120U];
        float _S224 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S193 + 120U]), _S190);
        accLo_0[15U] = _S224;
        float _S225 = fma(pj_3, attnHi_0(vw_63), _S191);
        accHi_0[15U] = _S225;
        float pj_4 = (*(&kernelContext_0)->ps_0)[_S43];
        float _S226 = _S192 + (*(&kernelContext_0)->ps_0)[_S43];
        uint _S227 = 516U + kk_0;
        uint vw_64 = (*(&kernelContext_0)->vs_0)[_S227];
        float _S228 = fma((*(&kernelContext_0)->ps_0)[_S43], attnLo_0((*(&kernelContext_0)->vs_0)[_S227]), _S194);
        accLo_0[0U] = _S228;
        float _S229 = fma(pj_4, attnHi_0(vw_64), _S195);
        accHi_0[0U] = _S229;
        uint vw_65 = (*(&kernelContext_0)->vs_0)[_S227 + 8U];
        float _S230 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 8U]), _S196);
        accLo_0[1U] = _S230;
        float _S231 = fma(pj_4, attnHi_0(vw_65), _S197);
        accHi_0[1U] = _S231;
        uint vw_66 = (*(&kernelContext_0)->vs_0)[_S227 + 16U];
        float _S232 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 16U]), _S198);
        accLo_0[2U] = _S232;
        float _S233 = fma(pj_4, attnHi_0(vw_66), _S199);
        accHi_0[2U] = _S233;
        uint vw_67 = (*(&kernelContext_0)->vs_0)[_S227 + 24U];
        float _S234 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 24U]), _S200);
        accLo_0[3U] = _S234;
        float _S235 = fma(pj_4, attnHi_0(vw_67), _S201);
        accHi_0[3U] = _S235;
        uint vw_68 = (*(&kernelContext_0)->vs_0)[_S227 + 32U];
        float _S236 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 32U]), _S202);
        accLo_0[4U] = _S236;
        float _S237 = fma(pj_4, attnHi_0(vw_68), _S203);
        accHi_0[4U] = _S237;
        uint vw_69 = (*(&kernelContext_0)->vs_0)[_S227 + 40U];
        float _S238 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 40U]), _S204);
        accLo_0[5U] = _S238;
        float _S239 = fma(pj_4, attnHi_0(vw_69), _S205);
        accHi_0[5U] = _S239;
        uint vw_70 = (*(&kernelContext_0)->vs_0)[_S227 + 48U];
        float _S240 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 48U]), _S206);
        accLo_0[6U] = _S240;
        float _S241 = fma(pj_4, attnHi_0(vw_70), _S207);
        accHi_0[6U] = _S241;
        uint vw_71 = (*(&kernelContext_0)->vs_0)[_S227 + 56U];
        float _S242 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 56U]), _S208);
        accLo_0[7U] = _S242;
        float _S243 = fma(pj_4, attnHi_0(vw_71), _S209);
        accHi_0[7U] = _S243;
        uint vw_72 = (*(&kernelContext_0)->vs_0)[_S227 + 64U];
        float _S244 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 64U]), _S210);
        accLo_0[8U] = _S244;
        float _S245 = fma(pj_4, attnHi_0(vw_72), _S211);
        accHi_0[8U] = _S245;
        uint vw_73 = (*(&kernelContext_0)->vs_0)[_S227 + 72U];
        float _S246 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 72U]), _S212);
        accLo_0[9U] = _S246;
        float _S247 = fma(pj_4, attnHi_0(vw_73), _S213);
        accHi_0[9U] = _S247;
        uint vw_74 = (*(&kernelContext_0)->vs_0)[_S227 + 80U];
        float _S248 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 80U]), _S214);
        accLo_0[10U] = _S248;
        float _S249 = fma(pj_4, attnHi_0(vw_74), _S215);
        accHi_0[10U] = _S249;
        uint vw_75 = (*(&kernelContext_0)->vs_0)[_S227 + 88U];
        float _S250 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 88U]), _S216);
        accLo_0[11U] = _S250;
        float _S251 = fma(pj_4, attnHi_0(vw_75), _S217);
        accHi_0[11U] = _S251;
        uint vw_76 = (*(&kernelContext_0)->vs_0)[_S227 + 96U];
        float _S252 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 96U]), _S218);
        accLo_0[12U] = _S252;
        float _S253 = fma(pj_4, attnHi_0(vw_76), _S219);
        accHi_0[12U] = _S253;
        uint vw_77 = (*(&kernelContext_0)->vs_0)[_S227 + 104U];
        float _S254 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 104U]), _S220);
        accLo_0[13U] = _S254;
        float _S255 = fma(pj_4, attnHi_0(vw_77), _S221);
        accHi_0[13U] = _S255;
        uint vw_78 = (*(&kernelContext_0)->vs_0)[_S227 + 112U];
        float _S256 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 112U]), _S222);
        accLo_0[14U] = _S256;
        float _S257 = fma(pj_4, attnHi_0(vw_78), _S223);
        accHi_0[14U] = _S257;
        uint vw_79 = (*(&kernelContext_0)->vs_0)[_S227 + 120U];
        float _S258 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S227 + 120U]), _S224);
        accLo_0[15U] = _S258;
        float _S259 = fma(pj_4, attnHi_0(vw_79), _S225);
        accHi_0[15U] = _S259;
        float pj_5 = (*(&kernelContext_0)->ps_0)[_S44];
        float _S260 = _S226 + (*(&kernelContext_0)->ps_0)[_S44];
        uint _S261 = 645U + kk_0;
        uint vw_80 = (*(&kernelContext_0)->vs_0)[_S261];
        float _S262 = fma((*(&kernelContext_0)->ps_0)[_S44], attnLo_0((*(&kernelContext_0)->vs_0)[_S261]), _S228);
        accLo_0[0U] = _S262;
        float _S263 = fma(pj_5, attnHi_0(vw_80), _S229);
        accHi_0[0U] = _S263;
        uint vw_81 = (*(&kernelContext_0)->vs_0)[_S261 + 8U];
        float _S264 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 8U]), _S230);
        accLo_0[1U] = _S264;
        float _S265 = fma(pj_5, attnHi_0(vw_81), _S231);
        accHi_0[1U] = _S265;
        uint vw_82 = (*(&kernelContext_0)->vs_0)[_S261 + 16U];
        float _S266 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 16U]), _S232);
        accLo_0[2U] = _S266;
        float _S267 = fma(pj_5, attnHi_0(vw_82), _S233);
        accHi_0[2U] = _S267;
        uint vw_83 = (*(&kernelContext_0)->vs_0)[_S261 + 24U];
        float _S268 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 24U]), _S234);
        accLo_0[3U] = _S268;
        float _S269 = fma(pj_5, attnHi_0(vw_83), _S235);
        accHi_0[3U] = _S269;
        uint vw_84 = (*(&kernelContext_0)->vs_0)[_S261 + 32U];
        float _S270 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 32U]), _S236);
        accLo_0[4U] = _S270;
        float _S271 = fma(pj_5, attnHi_0(vw_84), _S237);
        accHi_0[4U] = _S271;
        uint vw_85 = (*(&kernelContext_0)->vs_0)[_S261 + 40U];
        float _S272 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 40U]), _S238);
        accLo_0[5U] = _S272;
        float _S273 = fma(pj_5, attnHi_0(vw_85), _S239);
        accHi_0[5U] = _S273;
        uint vw_86 = (*(&kernelContext_0)->vs_0)[_S261 + 48U];
        float _S274 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 48U]), _S240);
        accLo_0[6U] = _S274;
        float _S275 = fma(pj_5, attnHi_0(vw_86), _S241);
        accHi_0[6U] = _S275;
        uint vw_87 = (*(&kernelContext_0)->vs_0)[_S261 + 56U];
        float _S276 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 56U]), _S242);
        accLo_0[7U] = _S276;
        float _S277 = fma(pj_5, attnHi_0(vw_87), _S243);
        accHi_0[7U] = _S277;
        uint vw_88 = (*(&kernelContext_0)->vs_0)[_S261 + 64U];
        float _S278 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 64U]), _S244);
        accLo_0[8U] = _S278;
        float _S279 = fma(pj_5, attnHi_0(vw_88), _S245);
        accHi_0[8U] = _S279;
        uint vw_89 = (*(&kernelContext_0)->vs_0)[_S261 + 72U];
        float _S280 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 72U]), _S246);
        accLo_0[9U] = _S280;
        float _S281 = fma(pj_5, attnHi_0(vw_89), _S247);
        accHi_0[9U] = _S281;
        uint vw_90 = (*(&kernelContext_0)->vs_0)[_S261 + 80U];
        float _S282 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 80U]), _S248);
        accLo_0[10U] = _S282;
        float _S283 = fma(pj_5, attnHi_0(vw_90), _S249);
        accHi_0[10U] = _S283;
        uint vw_91 = (*(&kernelContext_0)->vs_0)[_S261 + 88U];
        float _S284 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 88U]), _S250);
        accLo_0[11U] = _S284;
        float _S285 = fma(pj_5, attnHi_0(vw_91), _S251);
        accHi_0[11U] = _S285;
        uint vw_92 = (*(&kernelContext_0)->vs_0)[_S261 + 96U];
        float _S286 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 96U]), _S252);
        accLo_0[12U] = _S286;
        float _S287 = fma(pj_5, attnHi_0(vw_92), _S253);
        accHi_0[12U] = _S287;
        uint vw_93 = (*(&kernelContext_0)->vs_0)[_S261 + 104U];
        float _S288 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 104U]), _S254);
        accLo_0[13U] = _S288;
        float _S289 = fma(pj_5, attnHi_0(vw_93), _S255);
        accHi_0[13U] = _S289;
        uint vw_94 = (*(&kernelContext_0)->vs_0)[_S261 + 112U];
        float _S290 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 112U]), _S256);
        accLo_0[14U] = _S290;
        float _S291 = fma(pj_5, attnHi_0(vw_94), _S257);
        accHi_0[14U] = _S291;
        uint vw_95 = (*(&kernelContext_0)->vs_0)[_S261 + 120U];
        float _S292 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S261 + 120U]), _S258);
        accLo_0[15U] = _S292;
        float _S293 = fma(pj_5, attnHi_0(vw_95), _S259);
        accHi_0[15U] = _S293;
        float pj_6 = (*(&kernelContext_0)->ps_0)[_S45];
        float _S294 = _S260 + (*(&kernelContext_0)->ps_0)[_S45];
        uint _S295 = 774U + kk_0;
        uint vw_96 = (*(&kernelContext_0)->vs_0)[_S295];
        float _S296 = fma((*(&kernelContext_0)->ps_0)[_S45], attnLo_0((*(&kernelContext_0)->vs_0)[_S295]), _S262);
        accLo_0[0U] = _S296;
        float _S297 = fma(pj_6, attnHi_0(vw_96), _S263);
        accHi_0[0U] = _S297;
        uint vw_97 = (*(&kernelContext_0)->vs_0)[_S295 + 8U];
        float _S298 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 8U]), _S264);
        accLo_0[1U] = _S298;
        float _S299 = fma(pj_6, attnHi_0(vw_97), _S265);
        accHi_0[1U] = _S299;
        uint vw_98 = (*(&kernelContext_0)->vs_0)[_S295 + 16U];
        float _S300 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 16U]), _S266);
        accLo_0[2U] = _S300;
        float _S301 = fma(pj_6, attnHi_0(vw_98), _S267);
        accHi_0[2U] = _S301;
        uint vw_99 = (*(&kernelContext_0)->vs_0)[_S295 + 24U];
        float _S302 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 24U]), _S268);
        accLo_0[3U] = _S302;
        float _S303 = fma(pj_6, attnHi_0(vw_99), _S269);
        accHi_0[3U] = _S303;
        uint vw_100 = (*(&kernelContext_0)->vs_0)[_S295 + 32U];
        float _S304 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 32U]), _S270);
        accLo_0[4U] = _S304;
        float _S305 = fma(pj_6, attnHi_0(vw_100), _S271);
        accHi_0[4U] = _S305;
        uint vw_101 = (*(&kernelContext_0)->vs_0)[_S295 + 40U];
        float _S306 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 40U]), _S272);
        accLo_0[5U] = _S306;
        float _S307 = fma(pj_6, attnHi_0(vw_101), _S273);
        accHi_0[5U] = _S307;
        uint vw_102 = (*(&kernelContext_0)->vs_0)[_S295 + 48U];
        float _S308 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 48U]), _S274);
        accLo_0[6U] = _S308;
        float _S309 = fma(pj_6, attnHi_0(vw_102), _S275);
        accHi_0[6U] = _S309;
        uint vw_103 = (*(&kernelContext_0)->vs_0)[_S295 + 56U];
        float _S310 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 56U]), _S276);
        accLo_0[7U] = _S310;
        float _S311 = fma(pj_6, attnHi_0(vw_103), _S277);
        accHi_0[7U] = _S311;
        uint vw_104 = (*(&kernelContext_0)->vs_0)[_S295 + 64U];
        float _S312 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 64U]), _S278);
        accLo_0[8U] = _S312;
        float _S313 = fma(pj_6, attnHi_0(vw_104), _S279);
        accHi_0[8U] = _S313;
        uint vw_105 = (*(&kernelContext_0)->vs_0)[_S295 + 72U];
        float _S314 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 72U]), _S280);
        accLo_0[9U] = _S314;
        float _S315 = fma(pj_6, attnHi_0(vw_105), _S281);
        accHi_0[9U] = _S315;
        uint vw_106 = (*(&kernelContext_0)->vs_0)[_S295 + 80U];
        float _S316 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 80U]), _S282);
        accLo_0[10U] = _S316;
        float _S317 = fma(pj_6, attnHi_0(vw_106), _S283);
        accHi_0[10U] = _S317;
        uint vw_107 = (*(&kernelContext_0)->vs_0)[_S295 + 88U];
        float _S318 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 88U]), _S284);
        accLo_0[11U] = _S318;
        float _S319 = fma(pj_6, attnHi_0(vw_107), _S285);
        accHi_0[11U] = _S319;
        uint vw_108 = (*(&kernelContext_0)->vs_0)[_S295 + 96U];
        float _S320 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 96U]), _S286);
        accLo_0[12U] = _S320;
        float _S321 = fma(pj_6, attnHi_0(vw_108), _S287);
        accHi_0[12U] = _S321;
        uint vw_109 = (*(&kernelContext_0)->vs_0)[_S295 + 104U];
        float _S322 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 104U]), _S288);
        accLo_0[13U] = _S322;
        float _S323 = fma(pj_6, attnHi_0(vw_109), _S289);
        accHi_0[13U] = _S323;
        uint vw_110 = (*(&kernelContext_0)->vs_0)[_S295 + 112U];
        float _S324 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 112U]), _S290);
        accLo_0[14U] = _S324;
        float _S325 = fma(pj_6, attnHi_0(vw_110), _S291);
        accHi_0[14U] = _S325;
        uint vw_111 = (*(&kernelContext_0)->vs_0)[_S295 + 120U];
        float _S326 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S295 + 120U]), _S292);
        accLo_0[15U] = _S326;
        float _S327 = fma(pj_6, attnHi_0(vw_111), _S293);
        accHi_0[15U] = _S327;
        float pj_7 = (*(&kernelContext_0)->ps_0)[_S46];
        float _S328 = _S294 + (*(&kernelContext_0)->ps_0)[_S46];
        uint _S329 = 903U + kk_0;
        uint vw_112 = (*(&kernelContext_0)->vs_0)[_S329];
        float _S330 = fma((*(&kernelContext_0)->ps_0)[_S46], attnLo_0((*(&kernelContext_0)->vs_0)[_S329]), _S296);
        accLo_0[0U] = _S330;
        float _S331 = fma(pj_7, attnHi_0(vw_112), _S297);
        accHi_0[0U] = _S331;
        uint vw_113 = (*(&kernelContext_0)->vs_0)[_S329 + 8U];
        float _S332 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 8U]), _S298);
        accLo_0[1U] = _S332;
        float _S333 = fma(pj_7, attnHi_0(vw_113), _S299);
        accHi_0[1U] = _S333;
        uint vw_114 = (*(&kernelContext_0)->vs_0)[_S329 + 16U];
        float _S334 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 16U]), _S300);
        accLo_0[2U] = _S334;
        float _S335 = fma(pj_7, attnHi_0(vw_114), _S301);
        accHi_0[2U] = _S335;
        uint vw_115 = (*(&kernelContext_0)->vs_0)[_S329 + 24U];
        float _S336 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 24U]), _S302);
        accLo_0[3U] = _S336;
        float _S337 = fma(pj_7, attnHi_0(vw_115), _S303);
        accHi_0[3U] = _S337;
        uint vw_116 = (*(&kernelContext_0)->vs_0)[_S329 + 32U];
        float _S338 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 32U]), _S304);
        accLo_0[4U] = _S338;
        float _S339 = fma(pj_7, attnHi_0(vw_116), _S305);
        accHi_0[4U] = _S339;
        uint vw_117 = (*(&kernelContext_0)->vs_0)[_S329 + 40U];
        float _S340 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 40U]), _S306);
        accLo_0[5U] = _S340;
        float _S341 = fma(pj_7, attnHi_0(vw_117), _S307);
        accHi_0[5U] = _S341;
        uint vw_118 = (*(&kernelContext_0)->vs_0)[_S329 + 48U];
        float _S342 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 48U]), _S308);
        accLo_0[6U] = _S342;
        float _S343 = fma(pj_7, attnHi_0(vw_118), _S309);
        accHi_0[6U] = _S343;
        uint vw_119 = (*(&kernelContext_0)->vs_0)[_S329 + 56U];
        float _S344 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 56U]), _S310);
        accLo_0[7U] = _S344;
        float _S345 = fma(pj_7, attnHi_0(vw_119), _S311);
        accHi_0[7U] = _S345;
        uint vw_120 = (*(&kernelContext_0)->vs_0)[_S329 + 64U];
        float _S346 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 64U]), _S312);
        accLo_0[8U] = _S346;
        float _S347 = fma(pj_7, attnHi_0(vw_120), _S313);
        accHi_0[8U] = _S347;
        uint vw_121 = (*(&kernelContext_0)->vs_0)[_S329 + 72U];
        float _S348 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 72U]), _S314);
        accLo_0[9U] = _S348;
        float _S349 = fma(pj_7, attnHi_0(vw_121), _S315);
        accHi_0[9U] = _S349;
        uint vw_122 = (*(&kernelContext_0)->vs_0)[_S329 + 80U];
        float _S350 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 80U]), _S316);
        accLo_0[10U] = _S350;
        float _S351 = fma(pj_7, attnHi_0(vw_122), _S317);
        accHi_0[10U] = _S351;
        uint vw_123 = (*(&kernelContext_0)->vs_0)[_S329 + 88U];
        float _S352 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 88U]), _S318);
        accLo_0[11U] = _S352;
        float _S353 = fma(pj_7, attnHi_0(vw_123), _S319);
        accHi_0[11U] = _S353;
        uint vw_124 = (*(&kernelContext_0)->vs_0)[_S329 + 96U];
        float _S354 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 96U]), _S320);
        accLo_0[12U] = _S354;
        float _S355 = fma(pj_7, attnHi_0(vw_124), _S321);
        accHi_0[12U] = _S355;
        uint vw_125 = (*(&kernelContext_0)->vs_0)[_S329 + 104U];
        float _S356 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 104U]), _S322);
        accLo_0[13U] = _S356;
        float _S357 = fma(pj_7, attnHi_0(vw_125), _S323);
        accHi_0[13U] = _S357;
        uint vw_126 = (*(&kernelContext_0)->vs_0)[_S329 + 112U];
        float _S358 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 112U]), _S324);
        accLo_0[14U] = _S358;
        float _S359 = fma(pj_7, attnHi_0(vw_126), _S325);
        accHi_0[14U] = _S359;
        uint vw_127 = (*(&kernelContext_0)->vs_0)[_S329 + 120U];
        float _S360 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S329 + 120U]), _S326);
        accLo_0[15U] = _S360;
        float _S361 = fma(pj_7, attnHi_0(vw_127), _S327);
        accHi_0[15U] = _S361;
        float pj_8 = (*(&kernelContext_0)->ps_0)[_S47];
        float _S362 = _S328 + (*(&kernelContext_0)->ps_0)[_S47];
        uint _S363 = 1032U + kk_0;
        uint vw_128 = (*(&kernelContext_0)->vs_0)[_S363];
        float _S364 = fma((*(&kernelContext_0)->ps_0)[_S47], attnLo_0((*(&kernelContext_0)->vs_0)[_S363]), _S330);
        accLo_0[0U] = _S364;
        float _S365 = fma(pj_8, attnHi_0(vw_128), _S331);
        accHi_0[0U] = _S365;
        uint vw_129 = (*(&kernelContext_0)->vs_0)[_S363 + 8U];
        float _S366 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 8U]), _S332);
        accLo_0[1U] = _S366;
        float _S367 = fma(pj_8, attnHi_0(vw_129), _S333);
        accHi_0[1U] = _S367;
        uint vw_130 = (*(&kernelContext_0)->vs_0)[_S363 + 16U];
        float _S368 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 16U]), _S334);
        accLo_0[2U] = _S368;
        float _S369 = fma(pj_8, attnHi_0(vw_130), _S335);
        accHi_0[2U] = _S369;
        uint vw_131 = (*(&kernelContext_0)->vs_0)[_S363 + 24U];
        float _S370 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 24U]), _S336);
        accLo_0[3U] = _S370;
        float _S371 = fma(pj_8, attnHi_0(vw_131), _S337);
        accHi_0[3U] = _S371;
        uint vw_132 = (*(&kernelContext_0)->vs_0)[_S363 + 32U];
        float _S372 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 32U]), _S338);
        accLo_0[4U] = _S372;
        float _S373 = fma(pj_8, attnHi_0(vw_132), _S339);
        accHi_0[4U] = _S373;
        uint vw_133 = (*(&kernelContext_0)->vs_0)[_S363 + 40U];
        float _S374 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 40U]), _S340);
        accLo_0[5U] = _S374;
        float _S375 = fma(pj_8, attnHi_0(vw_133), _S341);
        accHi_0[5U] = _S375;
        uint vw_134 = (*(&kernelContext_0)->vs_0)[_S363 + 48U];
        float _S376 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 48U]), _S342);
        accLo_0[6U] = _S376;
        float _S377 = fma(pj_8, attnHi_0(vw_134), _S343);
        accHi_0[6U] = _S377;
        uint vw_135 = (*(&kernelContext_0)->vs_0)[_S363 + 56U];
        float _S378 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 56U]), _S344);
        accLo_0[7U] = _S378;
        float _S379 = fma(pj_8, attnHi_0(vw_135), _S345);
        accHi_0[7U] = _S379;
        uint vw_136 = (*(&kernelContext_0)->vs_0)[_S363 + 64U];
        float _S380 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 64U]), _S346);
        accLo_0[8U] = _S380;
        float _S381 = fma(pj_8, attnHi_0(vw_136), _S347);
        accHi_0[8U] = _S381;
        uint vw_137 = (*(&kernelContext_0)->vs_0)[_S363 + 72U];
        float _S382 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 72U]), _S348);
        accLo_0[9U] = _S382;
        float _S383 = fma(pj_8, attnHi_0(vw_137), _S349);
        accHi_0[9U] = _S383;
        uint vw_138 = (*(&kernelContext_0)->vs_0)[_S363 + 80U];
        float _S384 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 80U]), _S350);
        accLo_0[10U] = _S384;
        float _S385 = fma(pj_8, attnHi_0(vw_138), _S351);
        accHi_0[10U] = _S385;
        uint vw_139 = (*(&kernelContext_0)->vs_0)[_S363 + 88U];
        float _S386 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 88U]), _S352);
        accLo_0[11U] = _S386;
        float _S387 = fma(pj_8, attnHi_0(vw_139), _S353);
        accHi_0[11U] = _S387;
        uint vw_140 = (*(&kernelContext_0)->vs_0)[_S363 + 96U];
        float _S388 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 96U]), _S354);
        accLo_0[12U] = _S388;
        float _S389 = fma(pj_8, attnHi_0(vw_140), _S355);
        accHi_0[12U] = _S389;
        uint vw_141 = (*(&kernelContext_0)->vs_0)[_S363 + 104U];
        float _S390 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 104U]), _S356);
        accLo_0[13U] = _S390;
        float _S391 = fma(pj_8, attnHi_0(vw_141), _S357);
        accHi_0[13U] = _S391;
        uint vw_142 = (*(&kernelContext_0)->vs_0)[_S363 + 112U];
        float _S392 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 112U]), _S358);
        accLo_0[14U] = _S392;
        float _S393 = fma(pj_8, attnHi_0(vw_142), _S359);
        accHi_0[14U] = _S393;
        uint vw_143 = (*(&kernelContext_0)->vs_0)[_S363 + 120U];
        float _S394 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S363 + 120U]), _S360);
        accLo_0[15U] = _S394;
        float _S395 = fma(pj_8, attnHi_0(vw_143), _S361);
        accHi_0[15U] = _S395;
        float pj_9 = (*(&kernelContext_0)->ps_0)[_S48];
        float _S396 = _S362 + (*(&kernelContext_0)->ps_0)[_S48];
        uint _S397 = 1161U + kk_0;
        uint vw_144 = (*(&kernelContext_0)->vs_0)[_S397];
        float _S398 = fma((*(&kernelContext_0)->ps_0)[_S48], attnLo_0((*(&kernelContext_0)->vs_0)[_S397]), _S364);
        accLo_0[0U] = _S398;
        float _S399 = fma(pj_9, attnHi_0(vw_144), _S365);
        accHi_0[0U] = _S399;
        uint vw_145 = (*(&kernelContext_0)->vs_0)[_S397 + 8U];
        float _S400 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 8U]), _S366);
        accLo_0[1U] = _S400;
        float _S401 = fma(pj_9, attnHi_0(vw_145), _S367);
        accHi_0[1U] = _S401;
        uint vw_146 = (*(&kernelContext_0)->vs_0)[_S397 + 16U];
        float _S402 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 16U]), _S368);
        accLo_0[2U] = _S402;
        float _S403 = fma(pj_9, attnHi_0(vw_146), _S369);
        accHi_0[2U] = _S403;
        uint vw_147 = (*(&kernelContext_0)->vs_0)[_S397 + 24U];
        float _S404 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 24U]), _S370);
        accLo_0[3U] = _S404;
        float _S405 = fma(pj_9, attnHi_0(vw_147), _S371);
        accHi_0[3U] = _S405;
        uint vw_148 = (*(&kernelContext_0)->vs_0)[_S397 + 32U];
        float _S406 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 32U]), _S372);
        accLo_0[4U] = _S406;
        float _S407 = fma(pj_9, attnHi_0(vw_148), _S373);
        accHi_0[4U] = _S407;
        uint vw_149 = (*(&kernelContext_0)->vs_0)[_S397 + 40U];
        float _S408 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 40U]), _S374);
        accLo_0[5U] = _S408;
        float _S409 = fma(pj_9, attnHi_0(vw_149), _S375);
        accHi_0[5U] = _S409;
        uint vw_150 = (*(&kernelContext_0)->vs_0)[_S397 + 48U];
        float _S410 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 48U]), _S376);
        accLo_0[6U] = _S410;
        float _S411 = fma(pj_9, attnHi_0(vw_150), _S377);
        accHi_0[6U] = _S411;
        uint vw_151 = (*(&kernelContext_0)->vs_0)[_S397 + 56U];
        float _S412 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 56U]), _S378);
        accLo_0[7U] = _S412;
        float _S413 = fma(pj_9, attnHi_0(vw_151), _S379);
        accHi_0[7U] = _S413;
        uint vw_152 = (*(&kernelContext_0)->vs_0)[_S397 + 64U];
        float _S414 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 64U]), _S380);
        accLo_0[8U] = _S414;
        float _S415 = fma(pj_9, attnHi_0(vw_152), _S381);
        accHi_0[8U] = _S415;
        uint vw_153 = (*(&kernelContext_0)->vs_0)[_S397 + 72U];
        float _S416 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 72U]), _S382);
        accLo_0[9U] = _S416;
        float _S417 = fma(pj_9, attnHi_0(vw_153), _S383);
        accHi_0[9U] = _S417;
        uint vw_154 = (*(&kernelContext_0)->vs_0)[_S397 + 80U];
        float _S418 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 80U]), _S384);
        accLo_0[10U] = _S418;
        float _S419 = fma(pj_9, attnHi_0(vw_154), _S385);
        accHi_0[10U] = _S419;
        uint vw_155 = (*(&kernelContext_0)->vs_0)[_S397 + 88U];
        float _S420 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 88U]), _S386);
        accLo_0[11U] = _S420;
        float _S421 = fma(pj_9, attnHi_0(vw_155), _S387);
        accHi_0[11U] = _S421;
        uint vw_156 = (*(&kernelContext_0)->vs_0)[_S397 + 96U];
        float _S422 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 96U]), _S388);
        accLo_0[12U] = _S422;
        float _S423 = fma(pj_9, attnHi_0(vw_156), _S389);
        accHi_0[12U] = _S423;
        uint vw_157 = (*(&kernelContext_0)->vs_0)[_S397 + 104U];
        float _S424 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 104U]), _S390);
        accLo_0[13U] = _S424;
        float _S425 = fma(pj_9, attnHi_0(vw_157), _S391);
        accHi_0[13U] = _S425;
        uint vw_158 = (*(&kernelContext_0)->vs_0)[_S397 + 112U];
        float _S426 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 112U]), _S392);
        accLo_0[14U] = _S426;
        float _S427 = fma(pj_9, attnHi_0(vw_158), _S393);
        accHi_0[14U] = _S427;
        uint vw_159 = (*(&kernelContext_0)->vs_0)[_S397 + 120U];
        float _S428 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S397 + 120U]), _S394);
        accLo_0[15U] = _S428;
        float _S429 = fma(pj_9, attnHi_0(vw_159), _S395);
        accHi_0[15U] = _S429;
        float pj_10 = (*(&kernelContext_0)->ps_0)[_S49];
        float _S430 = _S396 + (*(&kernelContext_0)->ps_0)[_S49];
        uint _S431 = 1290U + kk_0;
        uint vw_160 = (*(&kernelContext_0)->vs_0)[_S431];
        float _S432 = fma((*(&kernelContext_0)->ps_0)[_S49], attnLo_0((*(&kernelContext_0)->vs_0)[_S431]), _S398);
        accLo_0[0U] = _S432;
        float _S433 = fma(pj_10, attnHi_0(vw_160), _S399);
        accHi_0[0U] = _S433;
        uint vw_161 = (*(&kernelContext_0)->vs_0)[_S431 + 8U];
        float _S434 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 8U]), _S400);
        accLo_0[1U] = _S434;
        float _S435 = fma(pj_10, attnHi_0(vw_161), _S401);
        accHi_0[1U] = _S435;
        uint vw_162 = (*(&kernelContext_0)->vs_0)[_S431 + 16U];
        float _S436 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 16U]), _S402);
        accLo_0[2U] = _S436;
        float _S437 = fma(pj_10, attnHi_0(vw_162), _S403);
        accHi_0[2U] = _S437;
        uint vw_163 = (*(&kernelContext_0)->vs_0)[_S431 + 24U];
        float _S438 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 24U]), _S404);
        accLo_0[3U] = _S438;
        float _S439 = fma(pj_10, attnHi_0(vw_163), _S405);
        accHi_0[3U] = _S439;
        uint vw_164 = (*(&kernelContext_0)->vs_0)[_S431 + 32U];
        float _S440 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 32U]), _S406);
        accLo_0[4U] = _S440;
        float _S441 = fma(pj_10, attnHi_0(vw_164), _S407);
        accHi_0[4U] = _S441;
        uint vw_165 = (*(&kernelContext_0)->vs_0)[_S431 + 40U];
        float _S442 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 40U]), _S408);
        accLo_0[5U] = _S442;
        float _S443 = fma(pj_10, attnHi_0(vw_165), _S409);
        accHi_0[5U] = _S443;
        uint vw_166 = (*(&kernelContext_0)->vs_0)[_S431 + 48U];
        float _S444 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 48U]), _S410);
        accLo_0[6U] = _S444;
        float _S445 = fma(pj_10, attnHi_0(vw_166), _S411);
        accHi_0[6U] = _S445;
        uint vw_167 = (*(&kernelContext_0)->vs_0)[_S431 + 56U];
        float _S446 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 56U]), _S412);
        accLo_0[7U] = _S446;
        float _S447 = fma(pj_10, attnHi_0(vw_167), _S413);
        accHi_0[7U] = _S447;
        uint vw_168 = (*(&kernelContext_0)->vs_0)[_S431 + 64U];
        float _S448 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 64U]), _S414);
        accLo_0[8U] = _S448;
        float _S449 = fma(pj_10, attnHi_0(vw_168), _S415);
        accHi_0[8U] = _S449;
        uint vw_169 = (*(&kernelContext_0)->vs_0)[_S431 + 72U];
        float _S450 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 72U]), _S416);
        accLo_0[9U] = _S450;
        float _S451 = fma(pj_10, attnHi_0(vw_169), _S417);
        accHi_0[9U] = _S451;
        uint vw_170 = (*(&kernelContext_0)->vs_0)[_S431 + 80U];
        float _S452 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 80U]), _S418);
        accLo_0[10U] = _S452;
        float _S453 = fma(pj_10, attnHi_0(vw_170), _S419);
        accHi_0[10U] = _S453;
        uint vw_171 = (*(&kernelContext_0)->vs_0)[_S431 + 88U];
        float _S454 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 88U]), _S420);
        accLo_0[11U] = _S454;
        float _S455 = fma(pj_10, attnHi_0(vw_171), _S421);
        accHi_0[11U] = _S455;
        uint vw_172 = (*(&kernelContext_0)->vs_0)[_S431 + 96U];
        float _S456 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 96U]), _S422);
        accLo_0[12U] = _S456;
        float _S457 = fma(pj_10, attnHi_0(vw_172), _S423);
        accHi_0[12U] = _S457;
        uint vw_173 = (*(&kernelContext_0)->vs_0)[_S431 + 104U];
        float _S458 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 104U]), _S424);
        accLo_0[13U] = _S458;
        float _S459 = fma(pj_10, attnHi_0(vw_173), _S425);
        accHi_0[13U] = _S459;
        uint vw_174 = (*(&kernelContext_0)->vs_0)[_S431 + 112U];
        float _S460 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 112U]), _S426);
        accLo_0[14U] = _S460;
        float _S461 = fma(pj_10, attnHi_0(vw_174), _S427);
        accHi_0[14U] = _S461;
        uint vw_175 = (*(&kernelContext_0)->vs_0)[_S431 + 120U];
        float _S462 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S431 + 120U]), _S428);
        accLo_0[15U] = _S462;
        float _S463 = fma(pj_10, attnHi_0(vw_175), _S429);
        accHi_0[15U] = _S463;
        float pj_11 = (*(&kernelContext_0)->ps_0)[_S50];
        float _S464 = _S430 + (*(&kernelContext_0)->ps_0)[_S50];
        uint _S465 = 1419U + kk_0;
        uint vw_176 = (*(&kernelContext_0)->vs_0)[_S465];
        float _S466 = fma((*(&kernelContext_0)->ps_0)[_S50], attnLo_0((*(&kernelContext_0)->vs_0)[_S465]), _S432);
        accLo_0[0U] = _S466;
        float _S467 = fma(pj_11, attnHi_0(vw_176), _S433);
        accHi_0[0U] = _S467;
        uint vw_177 = (*(&kernelContext_0)->vs_0)[_S465 + 8U];
        float _S468 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 8U]), _S434);
        accLo_0[1U] = _S468;
        float _S469 = fma(pj_11, attnHi_0(vw_177), _S435);
        accHi_0[1U] = _S469;
        uint vw_178 = (*(&kernelContext_0)->vs_0)[_S465 + 16U];
        float _S470 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 16U]), _S436);
        accLo_0[2U] = _S470;
        float _S471 = fma(pj_11, attnHi_0(vw_178), _S437);
        accHi_0[2U] = _S471;
        uint vw_179 = (*(&kernelContext_0)->vs_0)[_S465 + 24U];
        float _S472 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 24U]), _S438);
        accLo_0[3U] = _S472;
        float _S473 = fma(pj_11, attnHi_0(vw_179), _S439);
        accHi_0[3U] = _S473;
        uint vw_180 = (*(&kernelContext_0)->vs_0)[_S465 + 32U];
        float _S474 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 32U]), _S440);
        accLo_0[4U] = _S474;
        float _S475 = fma(pj_11, attnHi_0(vw_180), _S441);
        accHi_0[4U] = _S475;
        uint vw_181 = (*(&kernelContext_0)->vs_0)[_S465 + 40U];
        float _S476 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 40U]), _S442);
        accLo_0[5U] = _S476;
        float _S477 = fma(pj_11, attnHi_0(vw_181), _S443);
        accHi_0[5U] = _S477;
        uint vw_182 = (*(&kernelContext_0)->vs_0)[_S465 + 48U];
        float _S478 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 48U]), _S444);
        accLo_0[6U] = _S478;
        float _S479 = fma(pj_11, attnHi_0(vw_182), _S445);
        accHi_0[6U] = _S479;
        uint vw_183 = (*(&kernelContext_0)->vs_0)[_S465 + 56U];
        float _S480 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 56U]), _S446);
        accLo_0[7U] = _S480;
        float _S481 = fma(pj_11, attnHi_0(vw_183), _S447);
        accHi_0[7U] = _S481;
        uint vw_184 = (*(&kernelContext_0)->vs_0)[_S465 + 64U];
        float _S482 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 64U]), _S448);
        accLo_0[8U] = _S482;
        float _S483 = fma(pj_11, attnHi_0(vw_184), _S449);
        accHi_0[8U] = _S483;
        uint vw_185 = (*(&kernelContext_0)->vs_0)[_S465 + 72U];
        float _S484 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 72U]), _S450);
        accLo_0[9U] = _S484;
        float _S485 = fma(pj_11, attnHi_0(vw_185), _S451);
        accHi_0[9U] = _S485;
        uint vw_186 = (*(&kernelContext_0)->vs_0)[_S465 + 80U];
        float _S486 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 80U]), _S452);
        accLo_0[10U] = _S486;
        float _S487 = fma(pj_11, attnHi_0(vw_186), _S453);
        accHi_0[10U] = _S487;
        uint vw_187 = (*(&kernelContext_0)->vs_0)[_S465 + 88U];
        float _S488 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 88U]), _S454);
        accLo_0[11U] = _S488;
        float _S489 = fma(pj_11, attnHi_0(vw_187), _S455);
        accHi_0[11U] = _S489;
        uint vw_188 = (*(&kernelContext_0)->vs_0)[_S465 + 96U];
        float _S490 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 96U]), _S456);
        accLo_0[12U] = _S490;
        float _S491 = fma(pj_11, attnHi_0(vw_188), _S457);
        accHi_0[12U] = _S491;
        uint vw_189 = (*(&kernelContext_0)->vs_0)[_S465 + 104U];
        float _S492 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 104U]), _S458);
        accLo_0[13U] = _S492;
        float _S493 = fma(pj_11, attnHi_0(vw_189), _S459);
        accHi_0[13U] = _S493;
        uint vw_190 = (*(&kernelContext_0)->vs_0)[_S465 + 112U];
        float _S494 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 112U]), _S460);
        accLo_0[14U] = _S494;
        float _S495 = fma(pj_11, attnHi_0(vw_190), _S461);
        accHi_0[14U] = _S495;
        uint vw_191 = (*(&kernelContext_0)->vs_0)[_S465 + 120U];
        float _S496 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S465 + 120U]), _S462);
        accLo_0[15U] = _S496;
        float _S497 = fma(pj_11, attnHi_0(vw_191), _S463);
        accHi_0[15U] = _S497;
        float pj_12 = (*(&kernelContext_0)->ps_0)[_S51];
        float _S498 = _S464 + (*(&kernelContext_0)->ps_0)[_S51];
        uint _S499 = 1548U + kk_0;
        uint vw_192 = (*(&kernelContext_0)->vs_0)[_S499];
        float _S500 = fma((*(&kernelContext_0)->ps_0)[_S51], attnLo_0((*(&kernelContext_0)->vs_0)[_S499]), _S466);
        accLo_0[0U] = _S500;
        float _S501 = fma(pj_12, attnHi_0(vw_192), _S467);
        accHi_0[0U] = _S501;
        uint vw_193 = (*(&kernelContext_0)->vs_0)[_S499 + 8U];
        float _S502 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 8U]), _S468);
        accLo_0[1U] = _S502;
        float _S503 = fma(pj_12, attnHi_0(vw_193), _S469);
        accHi_0[1U] = _S503;
        uint vw_194 = (*(&kernelContext_0)->vs_0)[_S499 + 16U];
        float _S504 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 16U]), _S470);
        accLo_0[2U] = _S504;
        float _S505 = fma(pj_12, attnHi_0(vw_194), _S471);
        accHi_0[2U] = _S505;
        uint vw_195 = (*(&kernelContext_0)->vs_0)[_S499 + 24U];
        float _S506 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 24U]), _S472);
        accLo_0[3U] = _S506;
        float _S507 = fma(pj_12, attnHi_0(vw_195), _S473);
        accHi_0[3U] = _S507;
        uint vw_196 = (*(&kernelContext_0)->vs_0)[_S499 + 32U];
        float _S508 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 32U]), _S474);
        accLo_0[4U] = _S508;
        float _S509 = fma(pj_12, attnHi_0(vw_196), _S475);
        accHi_0[4U] = _S509;
        uint vw_197 = (*(&kernelContext_0)->vs_0)[_S499 + 40U];
        float _S510 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 40U]), _S476);
        accLo_0[5U] = _S510;
        float _S511 = fma(pj_12, attnHi_0(vw_197), _S477);
        accHi_0[5U] = _S511;
        uint vw_198 = (*(&kernelContext_0)->vs_0)[_S499 + 48U];
        float _S512 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 48U]), _S478);
        accLo_0[6U] = _S512;
        float _S513 = fma(pj_12, attnHi_0(vw_198), _S479);
        accHi_0[6U] = _S513;
        uint vw_199 = (*(&kernelContext_0)->vs_0)[_S499 + 56U];
        float _S514 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 56U]), _S480);
        accLo_0[7U] = _S514;
        float _S515 = fma(pj_12, attnHi_0(vw_199), _S481);
        accHi_0[7U] = _S515;
        uint vw_200 = (*(&kernelContext_0)->vs_0)[_S499 + 64U];
        float _S516 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 64U]), _S482);
        accLo_0[8U] = _S516;
        float _S517 = fma(pj_12, attnHi_0(vw_200), _S483);
        accHi_0[8U] = _S517;
        uint vw_201 = (*(&kernelContext_0)->vs_0)[_S499 + 72U];
        float _S518 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 72U]), _S484);
        accLo_0[9U] = _S518;
        float _S519 = fma(pj_12, attnHi_0(vw_201), _S485);
        accHi_0[9U] = _S519;
        uint vw_202 = (*(&kernelContext_0)->vs_0)[_S499 + 80U];
        float _S520 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 80U]), _S486);
        accLo_0[10U] = _S520;
        float _S521 = fma(pj_12, attnHi_0(vw_202), _S487);
        accHi_0[10U] = _S521;
        uint vw_203 = (*(&kernelContext_0)->vs_0)[_S499 + 88U];
        float _S522 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 88U]), _S488);
        accLo_0[11U] = _S522;
        float _S523 = fma(pj_12, attnHi_0(vw_203), _S489);
        accHi_0[11U] = _S523;
        uint vw_204 = (*(&kernelContext_0)->vs_0)[_S499 + 96U];
        float _S524 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 96U]), _S490);
        accLo_0[12U] = _S524;
        float _S525 = fma(pj_12, attnHi_0(vw_204), _S491);
        accHi_0[12U] = _S525;
        uint vw_205 = (*(&kernelContext_0)->vs_0)[_S499 + 104U];
        float _S526 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 104U]), _S492);
        accLo_0[13U] = _S526;
        float _S527 = fma(pj_12, attnHi_0(vw_205), _S493);
        accHi_0[13U] = _S527;
        uint vw_206 = (*(&kernelContext_0)->vs_0)[_S499 + 112U];
        float _S528 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 112U]), _S494);
        accLo_0[14U] = _S528;
        float _S529 = fma(pj_12, attnHi_0(vw_206), _S495);
        accHi_0[14U] = _S529;
        uint vw_207 = (*(&kernelContext_0)->vs_0)[_S499 + 120U];
        float _S530 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S499 + 120U]), _S496);
        accLo_0[15U] = _S530;
        float _S531 = fma(pj_12, attnHi_0(vw_207), _S497);
        accHi_0[15U] = _S531;
        float pj_13 = (*(&kernelContext_0)->ps_0)[_S52];
        float _S532 = _S498 + (*(&kernelContext_0)->ps_0)[_S52];
        uint _S533 = 1677U + kk_0;
        uint vw_208 = (*(&kernelContext_0)->vs_0)[_S533];
        float _S534 = fma((*(&kernelContext_0)->ps_0)[_S52], attnLo_0((*(&kernelContext_0)->vs_0)[_S533]), _S500);
        accLo_0[0U] = _S534;
        float _S535 = fma(pj_13, attnHi_0(vw_208), _S501);
        accHi_0[0U] = _S535;
        uint vw_209 = (*(&kernelContext_0)->vs_0)[_S533 + 8U];
        float _S536 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 8U]), _S502);
        accLo_0[1U] = _S536;
        float _S537 = fma(pj_13, attnHi_0(vw_209), _S503);
        accHi_0[1U] = _S537;
        uint vw_210 = (*(&kernelContext_0)->vs_0)[_S533 + 16U];
        float _S538 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 16U]), _S504);
        accLo_0[2U] = _S538;
        float _S539 = fma(pj_13, attnHi_0(vw_210), _S505);
        accHi_0[2U] = _S539;
        uint vw_211 = (*(&kernelContext_0)->vs_0)[_S533 + 24U];
        float _S540 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 24U]), _S506);
        accLo_0[3U] = _S540;
        float _S541 = fma(pj_13, attnHi_0(vw_211), _S507);
        accHi_0[3U] = _S541;
        uint vw_212 = (*(&kernelContext_0)->vs_0)[_S533 + 32U];
        float _S542 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 32U]), _S508);
        accLo_0[4U] = _S542;
        float _S543 = fma(pj_13, attnHi_0(vw_212), _S509);
        accHi_0[4U] = _S543;
        uint vw_213 = (*(&kernelContext_0)->vs_0)[_S533 + 40U];
        float _S544 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 40U]), _S510);
        accLo_0[5U] = _S544;
        float _S545 = fma(pj_13, attnHi_0(vw_213), _S511);
        accHi_0[5U] = _S545;
        uint vw_214 = (*(&kernelContext_0)->vs_0)[_S533 + 48U];
        float _S546 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 48U]), _S512);
        accLo_0[6U] = _S546;
        float _S547 = fma(pj_13, attnHi_0(vw_214), _S513);
        accHi_0[6U] = _S547;
        uint vw_215 = (*(&kernelContext_0)->vs_0)[_S533 + 56U];
        float _S548 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 56U]), _S514);
        accLo_0[7U] = _S548;
        float _S549 = fma(pj_13, attnHi_0(vw_215), _S515);
        accHi_0[7U] = _S549;
        uint vw_216 = (*(&kernelContext_0)->vs_0)[_S533 + 64U];
        float _S550 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 64U]), _S516);
        accLo_0[8U] = _S550;
        float _S551 = fma(pj_13, attnHi_0(vw_216), _S517);
        accHi_0[8U] = _S551;
        uint vw_217 = (*(&kernelContext_0)->vs_0)[_S533 + 72U];
        float _S552 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 72U]), _S518);
        accLo_0[9U] = _S552;
        float _S553 = fma(pj_13, attnHi_0(vw_217), _S519);
        accHi_0[9U] = _S553;
        uint vw_218 = (*(&kernelContext_0)->vs_0)[_S533 + 80U];
        float _S554 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 80U]), _S520);
        accLo_0[10U] = _S554;
        float _S555 = fma(pj_13, attnHi_0(vw_218), _S521);
        accHi_0[10U] = _S555;
        uint vw_219 = (*(&kernelContext_0)->vs_0)[_S533 + 88U];
        float _S556 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 88U]), _S522);
        accLo_0[11U] = _S556;
        float _S557 = fma(pj_13, attnHi_0(vw_219), _S523);
        accHi_0[11U] = _S557;
        uint vw_220 = (*(&kernelContext_0)->vs_0)[_S533 + 96U];
        float _S558 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 96U]), _S524);
        accLo_0[12U] = _S558;
        float _S559 = fma(pj_13, attnHi_0(vw_220), _S525);
        accHi_0[12U] = _S559;
        uint vw_221 = (*(&kernelContext_0)->vs_0)[_S533 + 104U];
        float _S560 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 104U]), _S526);
        accLo_0[13U] = _S560;
        float _S561 = fma(pj_13, attnHi_0(vw_221), _S527);
        accHi_0[13U] = _S561;
        uint vw_222 = (*(&kernelContext_0)->vs_0)[_S533 + 112U];
        float _S562 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 112U]), _S528);
        accLo_0[14U] = _S562;
        float _S563 = fma(pj_13, attnHi_0(vw_222), _S529);
        accHi_0[14U] = _S563;
        uint vw_223 = (*(&kernelContext_0)->vs_0)[_S533 + 120U];
        float _S564 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S533 + 120U]), _S530);
        accLo_0[15U] = _S564;
        float _S565 = fma(pj_13, attnHi_0(vw_223), _S531);
        accHi_0[15U] = _S565;
        float pj_14 = (*(&kernelContext_0)->ps_0)[_S53];
        float _S566 = _S532 + (*(&kernelContext_0)->ps_0)[_S53];
        uint _S567 = 1806U + kk_0;
        uint vw_224 = (*(&kernelContext_0)->vs_0)[_S567];
        float _S568 = fma((*(&kernelContext_0)->ps_0)[_S53], attnLo_0((*(&kernelContext_0)->vs_0)[_S567]), _S534);
        accLo_0[0U] = _S568;
        float _S569 = fma(pj_14, attnHi_0(vw_224), _S535);
        accHi_0[0U] = _S569;
        uint vw_225 = (*(&kernelContext_0)->vs_0)[_S567 + 8U];
        float _S570 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 8U]), _S536);
        accLo_0[1U] = _S570;
        float _S571 = fma(pj_14, attnHi_0(vw_225), _S537);
        accHi_0[1U] = _S571;
        uint vw_226 = (*(&kernelContext_0)->vs_0)[_S567 + 16U];
        float _S572 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 16U]), _S538);
        accLo_0[2U] = _S572;
        float _S573 = fma(pj_14, attnHi_0(vw_226), _S539);
        accHi_0[2U] = _S573;
        uint vw_227 = (*(&kernelContext_0)->vs_0)[_S567 + 24U];
        float _S574 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 24U]), _S540);
        accLo_0[3U] = _S574;
        float _S575 = fma(pj_14, attnHi_0(vw_227), _S541);
        accHi_0[3U] = _S575;
        uint vw_228 = (*(&kernelContext_0)->vs_0)[_S567 + 32U];
        float _S576 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 32U]), _S542);
        accLo_0[4U] = _S576;
        float _S577 = fma(pj_14, attnHi_0(vw_228), _S543);
        accHi_0[4U] = _S577;
        uint vw_229 = (*(&kernelContext_0)->vs_0)[_S567 + 40U];
        float _S578 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 40U]), _S544);
        accLo_0[5U] = _S578;
        float _S579 = fma(pj_14, attnHi_0(vw_229), _S545);
        accHi_0[5U] = _S579;
        uint vw_230 = (*(&kernelContext_0)->vs_0)[_S567 + 48U];
        float _S580 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 48U]), _S546);
        accLo_0[6U] = _S580;
        float _S581 = fma(pj_14, attnHi_0(vw_230), _S547);
        accHi_0[6U] = _S581;
        uint vw_231 = (*(&kernelContext_0)->vs_0)[_S567 + 56U];
        float _S582 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 56U]), _S548);
        accLo_0[7U] = _S582;
        float _S583 = fma(pj_14, attnHi_0(vw_231), _S549);
        accHi_0[7U] = _S583;
        uint vw_232 = (*(&kernelContext_0)->vs_0)[_S567 + 64U];
        float _S584 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 64U]), _S550);
        accLo_0[8U] = _S584;
        float _S585 = fma(pj_14, attnHi_0(vw_232), _S551);
        accHi_0[8U] = _S585;
        uint vw_233 = (*(&kernelContext_0)->vs_0)[_S567 + 72U];
        float _S586 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 72U]), _S552);
        accLo_0[9U] = _S586;
        float _S587 = fma(pj_14, attnHi_0(vw_233), _S553);
        accHi_0[9U] = _S587;
        uint vw_234 = (*(&kernelContext_0)->vs_0)[_S567 + 80U];
        float _S588 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 80U]), _S554);
        accLo_0[10U] = _S588;
        float _S589 = fma(pj_14, attnHi_0(vw_234), _S555);
        accHi_0[10U] = _S589;
        uint vw_235 = (*(&kernelContext_0)->vs_0)[_S567 + 88U];
        float _S590 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 88U]), _S556);
        accLo_0[11U] = _S590;
        float _S591 = fma(pj_14, attnHi_0(vw_235), _S557);
        accHi_0[11U] = _S591;
        uint vw_236 = (*(&kernelContext_0)->vs_0)[_S567 + 96U];
        float _S592 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 96U]), _S558);
        accLo_0[12U] = _S592;
        float _S593 = fma(pj_14, attnHi_0(vw_236), _S559);
        accHi_0[12U] = _S593;
        uint vw_237 = (*(&kernelContext_0)->vs_0)[_S567 + 104U];
        float _S594 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 104U]), _S560);
        accLo_0[13U] = _S594;
        float _S595 = fma(pj_14, attnHi_0(vw_237), _S561);
        accHi_0[13U] = _S595;
        uint vw_238 = (*(&kernelContext_0)->vs_0)[_S567 + 112U];
        float _S596 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 112U]), _S562);
        accLo_0[14U] = _S596;
        float _S597 = fma(pj_14, attnHi_0(vw_238), _S563);
        accHi_0[14U] = _S597;
        uint vw_239 = (*(&kernelContext_0)->vs_0)[_S567 + 120U];
        float _S598 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S567 + 120U]), _S564);
        accLo_0[15U] = _S598;
        float _S599 = fma(pj_14, attnHi_0(vw_239), _S565);
        accHi_0[15U] = _S599;
        float pj_15 = (*(&kernelContext_0)->ps_0)[_S54];
        float l_1 = _S566 + (*(&kernelContext_0)->ps_0)[_S54];
        uint _S600 = 1935U + kk_0;
        uint vw_240 = (*(&kernelContext_0)->vs_0)[_S600];
        accLo_0[0U] = fma((*(&kernelContext_0)->ps_0)[_S54], attnLo_0((*(&kernelContext_0)->vs_0)[_S600]), _S568);
        accHi_0[0U] = fma(pj_15, attnHi_0(vw_240), _S569);
        uint vw_241 = (*(&kernelContext_0)->vs_0)[_S600 + 8U];
        accLo_0[1U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 8U]), _S570);
        accHi_0[1U] = fma(pj_15, attnHi_0(vw_241), _S571);
        uint vw_242 = (*(&kernelContext_0)->vs_0)[_S600 + 16U];
        accLo_0[2U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 16U]), _S572);
        accHi_0[2U] = fma(pj_15, attnHi_0(vw_242), _S573);
        uint vw_243 = (*(&kernelContext_0)->vs_0)[_S600 + 24U];
        accLo_0[3U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 24U]), _S574);
        accHi_0[3U] = fma(pj_15, attnHi_0(vw_243), _S575);
        uint vw_244 = (*(&kernelContext_0)->vs_0)[_S600 + 32U];
        accLo_0[4U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 32U]), _S576);
        accHi_0[4U] = fma(pj_15, attnHi_0(vw_244), _S577);
        uint vw_245 = (*(&kernelContext_0)->vs_0)[_S600 + 40U];
        accLo_0[5U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 40U]), _S578);
        accHi_0[5U] = fma(pj_15, attnHi_0(vw_245), _S579);
        uint vw_246 = (*(&kernelContext_0)->vs_0)[_S600 + 48U];
        accLo_0[6U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 48U]), _S580);
        accHi_0[6U] = fma(pj_15, attnHi_0(vw_246), _S581);
        uint vw_247 = (*(&kernelContext_0)->vs_0)[_S600 + 56U];
        accLo_0[7U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 56U]), _S582);
        accHi_0[7U] = fma(pj_15, attnHi_0(vw_247), _S583);
        uint vw_248 = (*(&kernelContext_0)->vs_0)[_S600 + 64U];
        accLo_0[8U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 64U]), _S584);
        accHi_0[8U] = fma(pj_15, attnHi_0(vw_248), _S585);
        uint vw_249 = (*(&kernelContext_0)->vs_0)[_S600 + 72U];
        accLo_0[9U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 72U]), _S586);
        accHi_0[9U] = fma(pj_15, attnHi_0(vw_249), _S587);
        uint vw_250 = (*(&kernelContext_0)->vs_0)[_S600 + 80U];
        accLo_0[10U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 80U]), _S588);
        accHi_0[10U] = fma(pj_15, attnHi_0(vw_250), _S589);
        uint vw_251 = (*(&kernelContext_0)->vs_0)[_S600 + 88U];
        accLo_0[11U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 88U]), _S590);
        accHi_0[11U] = fma(pj_15, attnHi_0(vw_251), _S591);
        uint vw_252 = (*(&kernelContext_0)->vs_0)[_S600 + 96U];
        accLo_0[12U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 96U]), _S592);
        accHi_0[12U] = fma(pj_15, attnHi_0(vw_252), _S593);
        uint vw_253 = (*(&kernelContext_0)->vs_0)[_S600 + 104U];
        accLo_0[13U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 104U]), _S594);
        accHi_0[13U] = fma(pj_15, attnHi_0(vw_253), _S595);
        uint vw_254 = (*(&kernelContext_0)->vs_0)[_S600 + 112U];
        accLo_0[14U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 112U]), _S596);
        accHi_0[14U] = fma(pj_15, attnHi_0(vw_254), _S597);
        uint vw_255 = (*(&kernelContext_0)->vs_0)[_S600 + 120U];
        accLo_0[15U] = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S600 + 120U]), _S598);
        accHi_0[15U] = fma(pj_15, attnHi_0(vw_255), _S599);
        b_0 = b_0 + 16U;
        m_0 = _S55;
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
        uint pr_16 = split_0 * rows_1 + row_1;
        uint _S601 = attnAccBase_0((&kernelContext_0)->p_0->splits_1, rows_1) + pr_16 * (&kernelContext_0)->p_0->headDim_0;
        if(kk_0 == 0U)
        {
            uint _S602 = 2U * pr_16;
            *((&kernelContext_0)->partials_0+_S602) = m_0;
            *((&kernelContext_0)->partials_0+(_S602 + 1U)) = l_0;
        }
        if(kk_0 < _S14)
        {
            uint _S603 = _S601 + 2U * kk_0;
            float device* _S604 = (&kernelContext_0)->partials_0+_S603;
            bool _S605 = l_0 == 0.0;
            if(_S605)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[0U];
            }
            *_S604 = m_0;
            float device* _S606 = (&kernelContext_0)->partials_0+(_S603 + 1U);
            if(_S605)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[0U];
            }
            *_S606 = m_0;
        }
        uint pair_0 = kk_0 + 8U;
        if(pair_0 < _S14)
        {
            uint _S607 = _S601 + 2U * pair_0;
            float device* _S608 = (&kernelContext_0)->partials_0+_S607;
            bool _S609 = l_0 == 0.0;
            if(_S609)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[1U];
            }
            *_S608 = m_0;
            float device* _S610 = (&kernelContext_0)->partials_0+(_S607 + 1U);
            if(_S609)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[1U];
            }
            *_S610 = m_0;
        }
        uint pair_1 = kk_0 + 16U;
        if(pair_1 < _S14)
        {
            uint _S611 = _S601 + 2U * pair_1;
            float device* _S612 = (&kernelContext_0)->partials_0+_S611;
            bool _S613 = l_0 == 0.0;
            if(_S613)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[2U];
            }
            *_S612 = m_0;
            float device* _S614 = (&kernelContext_0)->partials_0+(_S611 + 1U);
            if(_S613)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[2U];
            }
            *_S614 = m_0;
        }
        uint pair_2 = kk_0 + 24U;
        if(pair_2 < _S14)
        {
            uint _S615 = _S601 + 2U * pair_2;
            float device* _S616 = (&kernelContext_0)->partials_0+_S615;
            bool _S617 = l_0 == 0.0;
            if(_S617)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[3U];
            }
            *_S616 = m_0;
            float device* _S618 = (&kernelContext_0)->partials_0+(_S615 + 1U);
            if(_S617)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[3U];
            }
            *_S618 = m_0;
        }
        uint pair_3 = kk_0 + 32U;
        if(pair_3 < _S14)
        {
            uint _S619 = _S601 + 2U * pair_3;
            float device* _S620 = (&kernelContext_0)->partials_0+_S619;
            bool _S621 = l_0 == 0.0;
            if(_S621)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[4U];
            }
            *_S620 = m_0;
            float device* _S622 = (&kernelContext_0)->partials_0+(_S619 + 1U);
            if(_S621)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[4U];
            }
            *_S622 = m_0;
        }
        uint pair_4 = kk_0 + 40U;
        if(pair_4 < _S14)
        {
            uint _S623 = _S601 + 2U * pair_4;
            float device* _S624 = (&kernelContext_0)->partials_0+_S623;
            bool _S625 = l_0 == 0.0;
            if(_S625)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[5U];
            }
            *_S624 = m_0;
            float device* _S626 = (&kernelContext_0)->partials_0+(_S623 + 1U);
            if(_S625)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[5U];
            }
            *_S626 = m_0;
        }
        uint pair_5 = kk_0 + 48U;
        if(pair_5 < _S14)
        {
            uint _S627 = _S601 + 2U * pair_5;
            float device* _S628 = (&kernelContext_0)->partials_0+_S627;
            bool _S629 = l_0 == 0.0;
            if(_S629)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[6U];
            }
            *_S628 = m_0;
            float device* _S630 = (&kernelContext_0)->partials_0+(_S627 + 1U);
            if(_S629)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[6U];
            }
            *_S630 = m_0;
        }
        uint pair_6 = kk_0 + 56U;
        if(pair_6 < _S14)
        {
            uint _S631 = _S601 + 2U * pair_6;
            float device* _S632 = (&kernelContext_0)->partials_0+_S631;
            bool _S633 = l_0 == 0.0;
            if(_S633)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[7U];
            }
            *_S632 = m_0;
            float device* _S634 = (&kernelContext_0)->partials_0+(_S631 + 1U);
            if(_S633)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[7U];
            }
            *_S634 = m_0;
        }
        uint pair_7 = kk_0 + 64U;
        if(pair_7 < _S14)
        {
            uint _S635 = _S601 + 2U * pair_7;
            float device* _S636 = (&kernelContext_0)->partials_0+_S635;
            bool _S637 = l_0 == 0.0;
            if(_S637)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[8U];
            }
            *_S636 = m_0;
            float device* _S638 = (&kernelContext_0)->partials_0+(_S635 + 1U);
            if(_S637)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[8U];
            }
            *_S638 = m_0;
        }
        uint pair_8 = kk_0 + 72U;
        if(pair_8 < _S14)
        {
            uint _S639 = _S601 + 2U * pair_8;
            float device* _S640 = (&kernelContext_0)->partials_0+_S639;
            bool _S641 = l_0 == 0.0;
            if(_S641)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[9U];
            }
            *_S640 = m_0;
            float device* _S642 = (&kernelContext_0)->partials_0+(_S639 + 1U);
            if(_S641)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[9U];
            }
            *_S642 = m_0;
        }
        uint pair_9 = kk_0 + 80U;
        if(pair_9 < _S14)
        {
            uint _S643 = _S601 + 2U * pair_9;
            float device* _S644 = (&kernelContext_0)->partials_0+_S643;
            bool _S645 = l_0 == 0.0;
            if(_S645)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[10U];
            }
            *_S644 = m_0;
            float device* _S646 = (&kernelContext_0)->partials_0+(_S643 + 1U);
            if(_S645)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[10U];
            }
            *_S646 = m_0;
        }
        uint pair_10 = kk_0 + 88U;
        if(pair_10 < _S14)
        {
            uint _S647 = _S601 + 2U * pair_10;
            float device* _S648 = (&kernelContext_0)->partials_0+_S647;
            bool _S649 = l_0 == 0.0;
            if(_S649)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[11U];
            }
            *_S648 = m_0;
            float device* _S650 = (&kernelContext_0)->partials_0+(_S647 + 1U);
            if(_S649)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[11U];
            }
            *_S650 = m_0;
        }
        uint pair_11 = kk_0 + 96U;
        if(pair_11 < _S14)
        {
            uint _S651 = _S601 + 2U * pair_11;
            float device* _S652 = (&kernelContext_0)->partials_0+_S651;
            bool _S653 = l_0 == 0.0;
            if(_S653)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[12U];
            }
            *_S652 = m_0;
            float device* _S654 = (&kernelContext_0)->partials_0+(_S651 + 1U);
            if(_S653)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[12U];
            }
            *_S654 = m_0;
        }
        uint pair_12 = kk_0 + 104U;
        if(pair_12 < _S14)
        {
            uint _S655 = _S601 + 2U * pair_12;
            float device* _S656 = (&kernelContext_0)->partials_0+_S655;
            bool _S657 = l_0 == 0.0;
            if(_S657)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[13U];
            }
            *_S656 = m_0;
            float device* _S658 = (&kernelContext_0)->partials_0+(_S655 + 1U);
            if(_S657)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[13U];
            }
            *_S658 = m_0;
        }
        uint pair_13 = kk_0 + 112U;
        if(pair_13 < _S14)
        {
            uint _S659 = _S601 + 2U * pair_13;
            float device* _S660 = (&kernelContext_0)->partials_0+_S659;
            bool _S661 = l_0 == 0.0;
            if(_S661)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[14U];
            }
            *_S660 = m_0;
            float device* _S662 = (&kernelContext_0)->partials_0+(_S659 + 1U);
            if(_S661)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[14U];
            }
            *_S662 = m_0;
        }
        uint pair_14 = kk_0 + 120U;
        if(pair_14 < _S14)
        {
            uint _S663 = _S601 + 2U * pair_14;
            float device* _S664 = (&kernelContext_0)->partials_0+_S663;
            bool _S665 = l_0 == 0.0;
            if(_S665)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[15U];
            }
            *_S664 = m_0;
            float device* _S666 = (&kernelContext_0)->partials_0+(_S663 + 1U);
            if(_S665)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[15U];
            }
            *_S666 = m_0;
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
        float device* _S667 = (&kernelContext_0)->lse_0+row_1;
        if(l_0 == 0.0)
        {
            m_0 = _S25;
        }
        else
        {
            m_0 = m_0 + log(l_0);
        }
        *_S667 = m_0;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                if(kk_0 >= _S14)
                {
                    break;
                }
                bool _S668 = l_0 == 0.0;
                if(_S668)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[0U] / l_0;
                }
                if(_S668)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[0U] / l_0;
                }
                uint _S669 = 2U * kk_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S669)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S669 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_15 = kk_0 + 8U;
                if(pair_15 >= _S14)
                {
                    break;
                }
                bool _S670 = l_0 == 0.0;
                if(_S670)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[1U] / l_0;
                }
                if(_S670)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[1U] / l_0;
                }
                uint _S671 = 2U * pair_15;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S671)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S671 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_16 = kk_0 + 16U;
                if(pair_16 >= _S14)
                {
                    break;
                }
                bool _S672 = l_0 == 0.0;
                if(_S672)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[2U] / l_0;
                }
                if(_S672)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[2U] / l_0;
                }
                uint _S673 = 2U * pair_16;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S673)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S673 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_17 = kk_0 + 24U;
                if(pair_17 >= _S14)
                {
                    break;
                }
                bool _S674 = l_0 == 0.0;
                if(_S674)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[3U] / l_0;
                }
                if(_S674)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[3U] / l_0;
                }
                uint _S675 = 2U * pair_17;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S675)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S675 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_18 = kk_0 + 32U;
                if(pair_18 >= _S14)
                {
                    break;
                }
                bool _S676 = l_0 == 0.0;
                if(_S676)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[4U] / l_0;
                }
                if(_S676)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[4U] / l_0;
                }
                uint _S677 = 2U * pair_18;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S677)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S677 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_19 = kk_0 + 40U;
                if(pair_19 >= _S14)
                {
                    break;
                }
                bool _S678 = l_0 == 0.0;
                if(_S678)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[5U] / l_0;
                }
                if(_S678)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[5U] / l_0;
                }
                uint _S679 = 2U * pair_19;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S679)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S679 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_20 = kk_0 + 48U;
                if(pair_20 >= _S14)
                {
                    break;
                }
                bool _S680 = l_0 == 0.0;
                if(_S680)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[6U] / l_0;
                }
                if(_S680)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[6U] / l_0;
                }
                uint _S681 = 2U * pair_20;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S681)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S681 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_21 = kk_0 + 56U;
                if(pair_21 >= _S14)
                {
                    break;
                }
                bool _S682 = l_0 == 0.0;
                if(_S682)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[7U] / l_0;
                }
                if(_S682)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[7U] / l_0;
                }
                uint _S683 = 2U * pair_21;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S683)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S683 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_22 = kk_0 + 64U;
                if(pair_22 >= _S14)
                {
                    break;
                }
                bool _S684 = l_0 == 0.0;
                if(_S684)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[8U] / l_0;
                }
                if(_S684)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[8U] / l_0;
                }
                uint _S685 = 2U * pair_22;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S685)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S685 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_23 = kk_0 + 72U;
                if(pair_23 >= _S14)
                {
                    break;
                }
                bool _S686 = l_0 == 0.0;
                if(_S686)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[9U] / l_0;
                }
                if(_S686)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[9U] / l_0;
                }
                uint _S687 = 2U * pair_23;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S687)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S687 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_24 = kk_0 + 80U;
                if(pair_24 >= _S14)
                {
                    break;
                }
                bool _S688 = l_0 == 0.0;
                if(_S688)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[10U] / l_0;
                }
                if(_S688)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[10U] / l_0;
                }
                uint _S689 = 2U * pair_24;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S689)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S689 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_25 = kk_0 + 88U;
                if(pair_25 >= _S14)
                {
                    break;
                }
                bool _S690 = l_0 == 0.0;
                if(_S690)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[11U] / l_0;
                }
                if(_S690)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[11U] / l_0;
                }
                uint _S691 = 2U * pair_25;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S691)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S691 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_26 = kk_0 + 96U;
                if(pair_26 >= _S14)
                {
                    break;
                }
                bool _S692 = l_0 == 0.0;
                if(_S692)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[12U] / l_0;
                }
                if(_S692)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[12U] / l_0;
                }
                uint _S693 = 2U * pair_26;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S693)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S693 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_27 = kk_0 + 104U;
                if(pair_27 >= _S14)
                {
                    break;
                }
                bool _S694 = l_0 == 0.0;
                if(_S694)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[13U] / l_0;
                }
                if(_S694)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[13U] / l_0;
                }
                uint _S695 = 2U * pair_27;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S695)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S695 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_28 = kk_0 + 112U;
                if(pair_28 >= _S14)
                {
                    break;
                }
                bool _S696 = l_0 == 0.0;
                if(_S696)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[14U] / l_0;
                }
                if(_S696)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[14U] / l_0;
                }
                uint _S697 = 2U * pair_28;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S697)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S697 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_29 = kk_0 + 120U;
                if(pair_29 >= _S14)
                {
                    break;
                }
                bool _S698 = l_0 == 0.0;
                if(_S698)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[15U] / l_0;
                }
                if(_S698)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[15U] / l_0;
                }
                uint _S699 = 2U * pair_29;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S699)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S699 + 1U)) = hi_0;
                break;
            }
            break;
        }
        break;
    }
    return;
}

