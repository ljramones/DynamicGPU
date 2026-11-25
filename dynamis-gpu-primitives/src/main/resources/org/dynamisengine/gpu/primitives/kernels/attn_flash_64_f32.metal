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
    array<uint, int(528)> threadgroup* qs_0;
    array<uint, int(1056)> threadgroup* ks_0;
    array<uint, int(1056)> threadgroup* vs_0;
    array<float, int(512)> threadgroup* ss_0;
    array<float, int(512)> threadgroup* ps_0;
};

[[kernel]] void attn_flash_64_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(8)]], uint device* q_1 [[buffer(0)]], uint device* lens_2 [[buffer(7)]], uint device* table_1 [[buffer(5)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partials_1 [[buffer(4)]], float device* lse_1 [[buffer(6)]], float device* o_1 [[buffer(3)]])
{
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
    uint _S15;
    uint _S16;
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
    uint _S17 = kvHead_0 * hpk_0;
    uint qi_0;
    if(live_0)
    {
        uint _S18 = r_0 / (&kernelContext_0)->p_0->queries_0;
        qi_0 = _S18;
    }
    else
    {
        qi_0 = 0U;
    }
    uint head_0 = _S17 + qi_0;
    if(live_0)
    {
        uint _S19 = r_0 % (&kernelContext_0)->p_0->queries_0;
        qi_0 = _S19;
    }
    else
    {
        qi_0 = 0U;
    }
    bool ok_0;
    uint w_2;
    uint _S20 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint offset_0 = (&kernelContext_0)->p_0->keys_0 - (&kernelContext_0)->p_0->queries_0;
    uint e_0 = t_0;
    for(;;)
    {
        if(e_0 < 512U)
        {
        }
        else
        {
            break;
        }
        uint row_0 = r0_0 + e_0 / 32U;
        uint pr_0 = e_0 % 32U;
        if(row_0 < rowsKv_0)
        {
            ok_0 = pr_0 < _S20;
        }
        else
        {
            ok_0 = false;
        }
        if(ok_0)
        {
            uint _S21 = row_0 / (&kernelContext_0)->p_0->queries_0;
            uint h_0 = _S17 + _S21;
            uint i_1 = row_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((h_0 * (&kernelContext_0)->p_0->queries_0 + i_1) * _S20 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_0 / 32U * 33U + pr_0] = w_2;
        e_0 = e_0 + 128U;
    }
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S22 = min((&kernelContext_0)->p_0->keys_0, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    uint rLast_0 = min(r0_0 + 16U, rowsKv_0) - 1U;
    uint _S23 = (&kernelContext_0)->p_0->lengthGroup_1;
    uint _S24 = r0_0 / (&kernelContext_0)->p_0->queries_0;
    uint _S25 = _S17 + _S24;
    uint _S26 = rLast_0 / (&kernelContext_0)->p_0->queries_0;
    uint kEnd_0 = attnKeyEnd_0((&kernelContext_0)->lens_1, _S23, _S25, _S17 + _S26, _S22);
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint _S27 = r0_0 / (&kernelContext_0)->p_0->queries_0;
        uint _S28 = rLast_0 / (&kernelContext_0)->p_0->queries_0;
        if(_S27 != _S28)
        {
            w_2 = (&kernelContext_0)->p_0->queries_0 - 1U;
        }
        else
        {
            uint _S29 = rLast_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = _S29;
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
        uint _S30 = head_0 / (&kernelContext_0)->p_0->lengthGroup_1;
        limit_0 = min(limit_0, *((&kernelContext_0)->lens_1+_S30));
    }
    float _S31 = dg_negative_infinity_0();
    thread array<float, int(4)> accLo_0;
    thread array<float, int(4)> accHi_0;
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
    uint b_0 = kStart_0;
    float m_0 = _S31;
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
        e_0 = t_0;
        for(;;)
        {
            if(e_0 < 1024U)
            {
            }
            else
            {
                break;
            }
            uint j_0 = e_0 / 32U;
            uint pr_1 = e_0 % 32U;
            uint key_0 = b_0 + j_0;
            if(key_0 < w_2)
            {
                ok_0 = pr_1 < _S20;
            }
            else
            {
                ok_0 = false;
            }
            if(ok_0)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    at_0 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * _S20 + pr_1;
                }
                else
                {
                    uint _S32 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S33 = (*((&kernelContext_0)->table_0+_S32) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S34 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
                    at_0 = (_S33 + _S34) * _S20 + pr_1;
                }
            }
            else
            {
                at_0 = 0U;
            }
            uint _S35 = j_0 * 33U + pr_1;
            if(ok_0)
            {
                p0_1 = *((&kernelContext_0)->k_0+at_0);
            }
            else
            {
                p0_1 = 0U;
            }
            (*(&kernelContext_0)->ks_0)[_S35] = p0_1;
            if(ok_0)
            {
                p0_0 = *((&kernelContext_0)->v_0+at_0);
            }
            else
            {
                p0_0 = 0U;
            }
            (*(&kernelContext_0)->vs_0)[_S35] = p0_0;
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
                        if(at_0 < 32U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S36 = lr_0 * 33U;
                        uint _S37 = kk_0 * 33U;
                        uint pr_2 = at_0 + 1U;
                        uint pr_3 = at_0 + 2U;
                        uint pr_4 = at_0 + 3U;
                        uint pr_5 = at_0 + 4U;
                        uint pr_6 = at_0 + 5U;
                        uint pr_7 = at_0 + 6U;
                        uint pr_8 = at_0 + 7U;
                        float s_1 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S36 + pr_8]), attnHi_0((*(&kernelContext_0)->ks_0)[_S37 + pr_8]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S36 + pr_8]), attnLo_0((*(&kernelContext_0)->ks_0)[_S37 + pr_8]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S36 + pr_7]), attnHi_0((*(&kernelContext_0)->ks_0)[_S37 + pr_7]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S36 + pr_7]), attnLo_0((*(&kernelContext_0)->ks_0)[_S37 + pr_7]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S36 + pr_6]), attnHi_0((*(&kernelContext_0)->ks_0)[_S37 + pr_6]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S36 + pr_6]), attnLo_0((*(&kernelContext_0)->ks_0)[_S37 + pr_6]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S36 + pr_5]), attnHi_0((*(&kernelContext_0)->ks_0)[_S37 + pr_5]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S36 + pr_5]), attnLo_0((*(&kernelContext_0)->ks_0)[_S37 + pr_5]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S36 + pr_4]), attnHi_0((*(&kernelContext_0)->ks_0)[_S37 + pr_4]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S36 + pr_4]), attnLo_0((*(&kernelContext_0)->ks_0)[_S37 + pr_4]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S36 + pr_3]), attnHi_0((*(&kernelContext_0)->ks_0)[_S37 + pr_3]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S36 + pr_3]), attnLo_0((*(&kernelContext_0)->ks_0)[_S37 + pr_3]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S36 + pr_2]), attnHi_0((*(&kernelContext_0)->ks_0)[_S37 + pr_2]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S36 + pr_2]), attnLo_0((*(&kernelContext_0)->ks_0)[_S37 + pr_2]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S36 + at_0]), attnHi_0((*(&kernelContext_0)->ks_0)[_S37 + at_0]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S36 + at_0]), attnLo_0((*(&kernelContext_0)->ks_0)[_S37 + at_0]), s_0))))))))))))))));
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
                    uint _S38 = lr_0 * 32U;
                    _S5 = _S38;
                    uint _S39 = _S38 + kk_0;
                    _S13 = _S39;
                    _S6 = _S39;
                    if(visible_0)
                    {
                        hi_0 = s_0 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        hi_0 = _S31;
                    }
                    (*(&kernelContext_0)->ss_0)[_S39] = hi_0;
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
                        uint _S40 = lr_0 * 33U;
                        uint _S41 = j_1 * 33U;
                        uint pr_9 = p0_1 + 1U;
                        uint pr_10 = p0_1 + 2U;
                        uint pr_11 = p0_1 + 3U;
                        uint pr_12 = p0_1 + 4U;
                        uint pr_13 = p0_1 + 5U;
                        uint pr_14 = p0_1 + 6U;
                        uint pr_15 = p0_1 + 7U;
                        float s_3 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S40 + pr_15]), attnHi_0((*(&kernelContext_0)->ks_0)[_S41 + pr_15]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S40 + pr_15]), attnLo_0((*(&kernelContext_0)->ks_0)[_S41 + pr_15]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S40 + pr_14]), attnHi_0((*(&kernelContext_0)->ks_0)[_S41 + pr_14]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S40 + pr_14]), attnLo_0((*(&kernelContext_0)->ks_0)[_S41 + pr_14]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S40 + pr_13]), attnHi_0((*(&kernelContext_0)->ks_0)[_S41 + pr_13]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S40 + pr_13]), attnLo_0((*(&kernelContext_0)->ks_0)[_S41 + pr_13]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S40 + pr_12]), attnHi_0((*(&kernelContext_0)->ks_0)[_S41 + pr_12]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S40 + pr_12]), attnLo_0((*(&kernelContext_0)->ks_0)[_S41 + pr_12]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S40 + pr_11]), attnHi_0((*(&kernelContext_0)->ks_0)[_S41 + pr_11]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S40 + pr_11]), attnLo_0((*(&kernelContext_0)->ks_0)[_S41 + pr_11]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S40 + pr_10]), attnHi_0((*(&kernelContext_0)->ks_0)[_S41 + pr_10]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S40 + pr_10]), attnLo_0((*(&kernelContext_0)->ks_0)[_S41 + pr_10]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S40 + pr_9]), attnHi_0((*(&kernelContext_0)->ks_0)[_S41 + pr_9]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S40 + pr_9]), attnLo_0((*(&kernelContext_0)->ks_0)[_S41 + pr_9]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S40 + p0_1]), attnHi_0((*(&kernelContext_0)->ks_0)[_S41 + p0_1]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S40 + p0_1]), attnLo_0((*(&kernelContext_0)->ks_0)[_S41 + p0_1]), s_2))))))))))))))));
                        p0_1 = p0_1 + 8U;
                        s_2 = s_3;
                    }
                    bool _S42;
                    if(live_0)
                    {
                        _S42 = key_2 < w_2;
                    }
                    else
                    {
                        _S42 = false;
                    }
                    bool visible_1;
                    if(_S42)
                    {
                        visible_1 = key_2 < limit_0;
                    }
                    else
                    {
                        visible_1 = false;
                    }
                    uint _S43 = _S5 + j_1;
                    _S14 = _S43;
                    _S8 = _S43;
                    float _S44;
                    if(visible_1)
                    {
                        _S44 = s_2 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S44 = _S31;
                    }
                    (*(&kernelContext_0)->ss_0)[_S43] = _S44;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_2 = kk_0 + 16U;
                    _S9 = j_2;
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
                        uint _S45 = lr_0 * 33U;
                        uint _S46 = j_2 * 33U;
                        uint pr_16 = p0_0 + 1U;
                        uint pr_17 = p0_0 + 2U;
                        uint pr_18 = p0_0 + 3U;
                        uint pr_19 = p0_0 + 4U;
                        uint pr_20 = p0_0 + 5U;
                        uint pr_21 = p0_0 + 6U;
                        uint pr_22 = p0_0 + 7U;
                        float s_5 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S45 + pr_22]), attnHi_0((*(&kernelContext_0)->ks_0)[_S46 + pr_22]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S45 + pr_22]), attnLo_0((*(&kernelContext_0)->ks_0)[_S46 + pr_22]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S45 + pr_21]), attnHi_0((*(&kernelContext_0)->ks_0)[_S46 + pr_21]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S45 + pr_21]), attnLo_0((*(&kernelContext_0)->ks_0)[_S46 + pr_21]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S45 + pr_20]), attnHi_0((*(&kernelContext_0)->ks_0)[_S46 + pr_20]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S45 + pr_20]), attnLo_0((*(&kernelContext_0)->ks_0)[_S46 + pr_20]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S45 + pr_19]), attnHi_0((*(&kernelContext_0)->ks_0)[_S46 + pr_19]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S45 + pr_19]), attnLo_0((*(&kernelContext_0)->ks_0)[_S46 + pr_19]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S45 + pr_18]), attnHi_0((*(&kernelContext_0)->ks_0)[_S46 + pr_18]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S45 + pr_18]), attnLo_0((*(&kernelContext_0)->ks_0)[_S46 + pr_18]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S45 + pr_17]), attnHi_0((*(&kernelContext_0)->ks_0)[_S46 + pr_17]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S45 + pr_17]), attnLo_0((*(&kernelContext_0)->ks_0)[_S46 + pr_17]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S45 + pr_16]), attnHi_0((*(&kernelContext_0)->ks_0)[_S46 + pr_16]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S45 + pr_16]), attnLo_0((*(&kernelContext_0)->ks_0)[_S46 + pr_16]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S45 + p0_0]), attnHi_0((*(&kernelContext_0)->ks_0)[_S46 + p0_0]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S45 + p0_0]), attnLo_0((*(&kernelContext_0)->ks_0)[_S46 + p0_0]), s_4))))))))))))))));
                        p0_0 = p0_0 + 8U;
                        s_4 = s_5;
                    }
                    bool _S47;
                    if(live_0)
                    {
                        _S47 = key_3 < w_2;
                    }
                    else
                    {
                        _S47 = false;
                    }
                    bool visible_2;
                    if(_S47)
                    {
                        visible_2 = key_3 < limit_0;
                    }
                    else
                    {
                        visible_2 = false;
                    }
                    uint _S48 = _S5 + j_2;
                    _S15 = _S48;
                    _S10 = _S48;
                    float _S49;
                    if(visible_2)
                    {
                        _S49 = s_4 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S49 = _S31;
                    }
                    (*(&kernelContext_0)->ss_0)[_S48] = _S49;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_3 = kk_0 + 24U;
                    _S11 = j_3;
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
                        uint _S50 = lr_0 * 33U;
                        uint _S51 = j_3 * 33U;
                        uint pr_23 = p0_2 + 1U;
                        uint pr_24 = p0_2 + 2U;
                        uint pr_25 = p0_2 + 3U;
                        uint pr_26 = p0_2 + 4U;
                        uint pr_27 = p0_2 + 5U;
                        uint pr_28 = p0_2 + 6U;
                        uint pr_29 = p0_2 + 7U;
                        float s_7 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S50 + pr_29]), attnHi_0((*(&kernelContext_0)->ks_0)[_S51 + pr_29]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S50 + pr_29]), attnLo_0((*(&kernelContext_0)->ks_0)[_S51 + pr_29]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S50 + pr_28]), attnHi_0((*(&kernelContext_0)->ks_0)[_S51 + pr_28]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S50 + pr_28]), attnLo_0((*(&kernelContext_0)->ks_0)[_S51 + pr_28]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S50 + pr_27]), attnHi_0((*(&kernelContext_0)->ks_0)[_S51 + pr_27]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S50 + pr_27]), attnLo_0((*(&kernelContext_0)->ks_0)[_S51 + pr_27]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S50 + pr_26]), attnHi_0((*(&kernelContext_0)->ks_0)[_S51 + pr_26]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S50 + pr_26]), attnLo_0((*(&kernelContext_0)->ks_0)[_S51 + pr_26]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S50 + pr_25]), attnHi_0((*(&kernelContext_0)->ks_0)[_S51 + pr_25]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S50 + pr_25]), attnLo_0((*(&kernelContext_0)->ks_0)[_S51 + pr_25]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S50 + pr_24]), attnHi_0((*(&kernelContext_0)->ks_0)[_S51 + pr_24]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S50 + pr_24]), attnLo_0((*(&kernelContext_0)->ks_0)[_S51 + pr_24]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S50 + pr_23]), attnHi_0((*(&kernelContext_0)->ks_0)[_S51 + pr_23]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S50 + pr_23]), attnLo_0((*(&kernelContext_0)->ks_0)[_S51 + pr_23]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S50 + p0_2]), attnHi_0((*(&kernelContext_0)->ks_0)[_S51 + p0_2]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S50 + p0_2]), attnLo_0((*(&kernelContext_0)->ks_0)[_S51 + p0_2]), s_6))))))))))))))));
                        p0_2 = p0_2 + 8U;
                        s_6 = s_7;
                    }
                    bool _S52;
                    if(live_0)
                    {
                        _S52 = key_4 < w_2;
                    }
                    else
                    {
                        _S52 = false;
                    }
                    bool visible_3;
                    if(_S52)
                    {
                        visible_3 = key_4 < limit_0;
                    }
                    else
                    {
                        visible_3 = false;
                    }
                    uint _S53 = _S5 + j_3;
                    _S16 = _S53;
                    _S12 = _S53;
                    float _S54;
                    if(visible_3)
                    {
                        _S54 = s_6 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S54 = _S31;
                    }
                    (*(&kernelContext_0)->ss_0)[_S53] = _S54;
                    break;
                }
                break;
            }
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S55 = _S5 + 1U;
        uint _S56 = _S5 + 2U;
        uint _S57 = _S5 + 3U;
        uint _S58 = _S5 + 4U;
        uint _S59 = _S5 + 5U;
        uint _S60 = _S5 + 6U;
        uint _S61 = _S5 + 7U;
        uint _S62 = _S5 + 8U;
        uint _S63 = _S5 + 9U;
        uint _S64 = _S5 + 10U;
        uint _S65 = _S5 + 11U;
        uint _S66 = _S5 + 12U;
        uint _S67 = _S5 + 13U;
        uint _S68 = _S5 + 14U;
        uint _S69 = _S5 + 15U;
        uint _S70 = _S5 + 16U;
        uint _S71 = _S5 + 17U;
        uint _S72 = _S5 + 18U;
        uint _S73 = _S5 + 19U;
        uint _S74 = _S5 + 20U;
        uint _S75 = _S5 + 21U;
        uint _S76 = _S5 + 22U;
        uint _S77 = _S5 + 23U;
        uint _S78 = _S5 + 24U;
        uint _S79 = _S5 + 25U;
        uint _S80 = _S5 + 26U;
        uint _S81 = _S5 + 27U;
        uint _S82 = _S5 + 28U;
        uint _S83 = _S5 + 29U;
        uint _S84 = _S5 + 30U;
        uint _S85 = _S5 + 31U;
        float _S86 = max(m_0, max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(_S31, (*(&kernelContext_0)->ss_0)[_S5]), (*(&kernelContext_0)->ss_0)[_S55]), (*(&kernelContext_0)->ss_0)[_S56]), (*(&kernelContext_0)->ss_0)[_S57]), (*(&kernelContext_0)->ss_0)[_S58]), (*(&kernelContext_0)->ss_0)[_S59]), (*(&kernelContext_0)->ss_0)[_S60]), (*(&kernelContext_0)->ss_0)[_S61]), (*(&kernelContext_0)->ss_0)[_S62]), (*(&kernelContext_0)->ss_0)[_S63]), (*(&kernelContext_0)->ss_0)[_S64]), (*(&kernelContext_0)->ss_0)[_S65]), (*(&kernelContext_0)->ss_0)[_S66]), (*(&kernelContext_0)->ss_0)[_S67]), (*(&kernelContext_0)->ss_0)[_S68]), (*(&kernelContext_0)->ss_0)[_S69]), (*(&kernelContext_0)->ss_0)[_S70]), (*(&kernelContext_0)->ss_0)[_S71]), (*(&kernelContext_0)->ss_0)[_S72]), (*(&kernelContext_0)->ss_0)[_S73]), (*(&kernelContext_0)->ss_0)[_S74]), (*(&kernelContext_0)->ss_0)[_S75]), (*(&kernelContext_0)->ss_0)[_S76]), (*(&kernelContext_0)->ss_0)[_S77]), (*(&kernelContext_0)->ss_0)[_S78]), (*(&kernelContext_0)->ss_0)[_S79]), (*(&kernelContext_0)->ss_0)[_S80]), (*(&kernelContext_0)->ss_0)[_S81]), (*(&kernelContext_0)->ss_0)[_S82]), (*(&kernelContext_0)->ss_0)[_S83]), (*(&kernelContext_0)->ss_0)[_S84]), (*(&kernelContext_0)->ss_0)[_S85]));
        float alpha_0;
        if(m_0 == _S31)
        {
            alpha_0 = 0.0;
        }
        else
        {
            alpha_0 = exp(m_0 - _S86);
        }
        float s_8 = (*(&kernelContext_0)->ss_0)[_S13];
        float _S87;
        if(((*(&kernelContext_0)->ss_0)[_S13]) == _S31)
        {
            _S87 = 0.0;
        }
        else
        {
            _S87 = exp(s_8 - _S86);
        }
        (*(&kernelContext_0)->ps_0)[_S6] = _S87;
        float s_9 = (*(&kernelContext_0)->ss_0)[_S14];
        float _S88;
        if(((*(&kernelContext_0)->ss_0)[_S14]) == _S31)
        {
            _S88 = 0.0;
        }
        else
        {
            _S88 = exp(s_9 - _S86);
        }
        (*(&kernelContext_0)->ps_0)[_S8] = _S88;
        float s_10 = (*(&kernelContext_0)->ss_0)[_S15];
        float _S89;
        if(((*(&kernelContext_0)->ss_0)[_S15]) == _S31)
        {
            _S89 = 0.0;
        }
        else
        {
            _S89 = exp(s_10 - _S86);
        }
        (*(&kernelContext_0)->ps_0)[_S10] = _S89;
        float s_11 = (*(&kernelContext_0)->ss_0)[_S16];
        float _S90;
        if(((*(&kernelContext_0)->ss_0)[_S16]) == _S31)
        {
            _S90 = 0.0;
        }
        else
        {
            _S90 = exp(s_11 - _S86);
        }
        (*(&kernelContext_0)->ps_0)[_S12] = _S90;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S91 = l_0 * alpha_0;
        float _S92 = accLo_0[0U] * alpha_0;
        accLo_0[0U] = _S92;
        float _S93 = accHi_0[0U] * alpha_0;
        accHi_0[0U] = _S93;
        float _S94 = accLo_0[1U] * alpha_0;
        accLo_0[1U] = _S94;
        float _S95 = accHi_0[1U] * alpha_0;
        accHi_0[1U] = _S95;
        float _S96 = accLo_0[2U] * alpha_0;
        accLo_0[2U] = _S96;
        float _S97 = accHi_0[2U] * alpha_0;
        accHi_0[2U] = _S97;
        float _S98 = accLo_0[3U] * alpha_0;
        accLo_0[3U] = _S98;
        float _S99 = accHi_0[3U] * alpha_0;
        accHi_0[3U] = _S99;
        float pj_0 = (*(&kernelContext_0)->ps_0)[_S5];
        float _S100 = _S91 + (*(&kernelContext_0)->ps_0)[_S5];
        uint vw_0 = (*(&kernelContext_0)->vs_0)[kk_0];
        float _S101 = fma((*(&kernelContext_0)->ps_0)[_S5], attnLo_0((*(&kernelContext_0)->vs_0)[kk_0]), _S92);
        accLo_0[0U] = _S101;
        float _S102 = fma(pj_0, attnHi_0(vw_0), _S93);
        accHi_0[0U] = _S102;
        uint vw_1 = (*(&kernelContext_0)->vs_0)[_S7];
        float _S103 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S7]), _S94);
        accLo_0[1U] = _S103;
        float _S104 = fma(pj_0, attnHi_0(vw_1), _S95);
        accHi_0[1U] = _S104;
        uint vw_2 = (*(&kernelContext_0)->vs_0)[_S9];
        float _S105 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S9]), _S96);
        accLo_0[2U] = _S105;
        float _S106 = fma(pj_0, attnHi_0(vw_2), _S97);
        accHi_0[2U] = _S106;
        uint vw_3 = (*(&kernelContext_0)->vs_0)[_S11];
        float _S107 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S11]), _S98);
        accLo_0[3U] = _S107;
        float _S108 = fma(pj_0, attnHi_0(vw_3), _S99);
        accHi_0[3U] = _S108;
        float pj_1 = (*(&kernelContext_0)->ps_0)[_S55];
        float _S109 = _S100 + (*(&kernelContext_0)->ps_0)[_S55];
        uint _S110 = 33U + kk_0;
        uint vw_4 = (*(&kernelContext_0)->vs_0)[_S110];
        float _S111 = fma((*(&kernelContext_0)->ps_0)[_S55], attnLo_0((*(&kernelContext_0)->vs_0)[_S110]), _S101);
        accLo_0[0U] = _S111;
        float _S112 = fma(pj_1, attnHi_0(vw_4), _S102);
        accHi_0[0U] = _S112;
        uint vw_5 = (*(&kernelContext_0)->vs_0)[_S110 + 8U];
        float _S113 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S110 + 8U]), _S103);
        accLo_0[1U] = _S113;
        float _S114 = fma(pj_1, attnHi_0(vw_5), _S104);
        accHi_0[1U] = _S114;
        uint vw_6 = (*(&kernelContext_0)->vs_0)[_S110 + 16U];
        float _S115 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S110 + 16U]), _S105);
        accLo_0[2U] = _S115;
        float _S116 = fma(pj_1, attnHi_0(vw_6), _S106);
        accHi_0[2U] = _S116;
        uint vw_7 = (*(&kernelContext_0)->vs_0)[_S110 + 24U];
        float _S117 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S110 + 24U]), _S107);
        accLo_0[3U] = _S117;
        float _S118 = fma(pj_1, attnHi_0(vw_7), _S108);
        accHi_0[3U] = _S118;
        float pj_2 = (*(&kernelContext_0)->ps_0)[_S56];
        float _S119 = _S109 + (*(&kernelContext_0)->ps_0)[_S56];
        uint _S120 = 66U + kk_0;
        uint vw_8 = (*(&kernelContext_0)->vs_0)[_S120];
        float _S121 = fma((*(&kernelContext_0)->ps_0)[_S56], attnLo_0((*(&kernelContext_0)->vs_0)[_S120]), _S111);
        accLo_0[0U] = _S121;
        float _S122 = fma(pj_2, attnHi_0(vw_8), _S112);
        accHi_0[0U] = _S122;
        uint vw_9 = (*(&kernelContext_0)->vs_0)[_S120 + 8U];
        float _S123 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S120 + 8U]), _S113);
        accLo_0[1U] = _S123;
        float _S124 = fma(pj_2, attnHi_0(vw_9), _S114);
        accHi_0[1U] = _S124;
        uint vw_10 = (*(&kernelContext_0)->vs_0)[_S120 + 16U];
        float _S125 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S120 + 16U]), _S115);
        accLo_0[2U] = _S125;
        float _S126 = fma(pj_2, attnHi_0(vw_10), _S116);
        accHi_0[2U] = _S126;
        uint vw_11 = (*(&kernelContext_0)->vs_0)[_S120 + 24U];
        float _S127 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S120 + 24U]), _S117);
        accLo_0[3U] = _S127;
        float _S128 = fma(pj_2, attnHi_0(vw_11), _S118);
        accHi_0[3U] = _S128;
        float pj_3 = (*(&kernelContext_0)->ps_0)[_S57];
        float _S129 = _S119 + (*(&kernelContext_0)->ps_0)[_S57];
        uint _S130 = 99U + kk_0;
        uint vw_12 = (*(&kernelContext_0)->vs_0)[_S130];
        float _S131 = fma((*(&kernelContext_0)->ps_0)[_S57], attnLo_0((*(&kernelContext_0)->vs_0)[_S130]), _S121);
        accLo_0[0U] = _S131;
        float _S132 = fma(pj_3, attnHi_0(vw_12), _S122);
        accHi_0[0U] = _S132;
        uint vw_13 = (*(&kernelContext_0)->vs_0)[_S130 + 8U];
        float _S133 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S130 + 8U]), _S123);
        accLo_0[1U] = _S133;
        float _S134 = fma(pj_3, attnHi_0(vw_13), _S124);
        accHi_0[1U] = _S134;
        uint vw_14 = (*(&kernelContext_0)->vs_0)[_S130 + 16U];
        float _S135 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S130 + 16U]), _S125);
        accLo_0[2U] = _S135;
        float _S136 = fma(pj_3, attnHi_0(vw_14), _S126);
        accHi_0[2U] = _S136;
        uint vw_15 = (*(&kernelContext_0)->vs_0)[_S130 + 24U];
        float _S137 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S130 + 24U]), _S127);
        accLo_0[3U] = _S137;
        float _S138 = fma(pj_3, attnHi_0(vw_15), _S128);
        accHi_0[3U] = _S138;
        float pj_4 = (*(&kernelContext_0)->ps_0)[_S58];
        float _S139 = _S129 + (*(&kernelContext_0)->ps_0)[_S58];
        uint _S140 = 132U + kk_0;
        uint vw_16 = (*(&kernelContext_0)->vs_0)[_S140];
        float _S141 = fma((*(&kernelContext_0)->ps_0)[_S58], attnLo_0((*(&kernelContext_0)->vs_0)[_S140]), _S131);
        accLo_0[0U] = _S141;
        float _S142 = fma(pj_4, attnHi_0(vw_16), _S132);
        accHi_0[0U] = _S142;
        uint vw_17 = (*(&kernelContext_0)->vs_0)[_S140 + 8U];
        float _S143 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S140 + 8U]), _S133);
        accLo_0[1U] = _S143;
        float _S144 = fma(pj_4, attnHi_0(vw_17), _S134);
        accHi_0[1U] = _S144;
        uint vw_18 = (*(&kernelContext_0)->vs_0)[_S140 + 16U];
        float _S145 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S140 + 16U]), _S135);
        accLo_0[2U] = _S145;
        float _S146 = fma(pj_4, attnHi_0(vw_18), _S136);
        accHi_0[2U] = _S146;
        uint vw_19 = (*(&kernelContext_0)->vs_0)[_S140 + 24U];
        float _S147 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S140 + 24U]), _S137);
        accLo_0[3U] = _S147;
        float _S148 = fma(pj_4, attnHi_0(vw_19), _S138);
        accHi_0[3U] = _S148;
        float pj_5 = (*(&kernelContext_0)->ps_0)[_S59];
        float _S149 = _S139 + (*(&kernelContext_0)->ps_0)[_S59];
        uint _S150 = 165U + kk_0;
        uint vw_20 = (*(&kernelContext_0)->vs_0)[_S150];
        float _S151 = fma((*(&kernelContext_0)->ps_0)[_S59], attnLo_0((*(&kernelContext_0)->vs_0)[_S150]), _S141);
        accLo_0[0U] = _S151;
        float _S152 = fma(pj_5, attnHi_0(vw_20), _S142);
        accHi_0[0U] = _S152;
        uint vw_21 = (*(&kernelContext_0)->vs_0)[_S150 + 8U];
        float _S153 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S150 + 8U]), _S143);
        accLo_0[1U] = _S153;
        float _S154 = fma(pj_5, attnHi_0(vw_21), _S144);
        accHi_0[1U] = _S154;
        uint vw_22 = (*(&kernelContext_0)->vs_0)[_S150 + 16U];
        float _S155 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S150 + 16U]), _S145);
        accLo_0[2U] = _S155;
        float _S156 = fma(pj_5, attnHi_0(vw_22), _S146);
        accHi_0[2U] = _S156;
        uint vw_23 = (*(&kernelContext_0)->vs_0)[_S150 + 24U];
        float _S157 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S150 + 24U]), _S147);
        accLo_0[3U] = _S157;
        float _S158 = fma(pj_5, attnHi_0(vw_23), _S148);
        accHi_0[3U] = _S158;
        float pj_6 = (*(&kernelContext_0)->ps_0)[_S60];
        float _S159 = _S149 + (*(&kernelContext_0)->ps_0)[_S60];
        uint _S160 = 198U + kk_0;
        uint vw_24 = (*(&kernelContext_0)->vs_0)[_S160];
        float _S161 = fma((*(&kernelContext_0)->ps_0)[_S60], attnLo_0((*(&kernelContext_0)->vs_0)[_S160]), _S151);
        accLo_0[0U] = _S161;
        float _S162 = fma(pj_6, attnHi_0(vw_24), _S152);
        accHi_0[0U] = _S162;
        uint vw_25 = (*(&kernelContext_0)->vs_0)[_S160 + 8U];
        float _S163 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S160 + 8U]), _S153);
        accLo_0[1U] = _S163;
        float _S164 = fma(pj_6, attnHi_0(vw_25), _S154);
        accHi_0[1U] = _S164;
        uint vw_26 = (*(&kernelContext_0)->vs_0)[_S160 + 16U];
        float _S165 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S160 + 16U]), _S155);
        accLo_0[2U] = _S165;
        float _S166 = fma(pj_6, attnHi_0(vw_26), _S156);
        accHi_0[2U] = _S166;
        uint vw_27 = (*(&kernelContext_0)->vs_0)[_S160 + 24U];
        float _S167 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S160 + 24U]), _S157);
        accLo_0[3U] = _S167;
        float _S168 = fma(pj_6, attnHi_0(vw_27), _S158);
        accHi_0[3U] = _S168;
        float pj_7 = (*(&kernelContext_0)->ps_0)[_S61];
        float _S169 = _S159 + (*(&kernelContext_0)->ps_0)[_S61];
        uint _S170 = 231U + kk_0;
        uint vw_28 = (*(&kernelContext_0)->vs_0)[_S170];
        float _S171 = fma((*(&kernelContext_0)->ps_0)[_S61], attnLo_0((*(&kernelContext_0)->vs_0)[_S170]), _S161);
        accLo_0[0U] = _S171;
        float _S172 = fma(pj_7, attnHi_0(vw_28), _S162);
        accHi_0[0U] = _S172;
        uint vw_29 = (*(&kernelContext_0)->vs_0)[_S170 + 8U];
        float _S173 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S170 + 8U]), _S163);
        accLo_0[1U] = _S173;
        float _S174 = fma(pj_7, attnHi_0(vw_29), _S164);
        accHi_0[1U] = _S174;
        uint vw_30 = (*(&kernelContext_0)->vs_0)[_S170 + 16U];
        float _S175 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S170 + 16U]), _S165);
        accLo_0[2U] = _S175;
        float _S176 = fma(pj_7, attnHi_0(vw_30), _S166);
        accHi_0[2U] = _S176;
        uint vw_31 = (*(&kernelContext_0)->vs_0)[_S170 + 24U];
        float _S177 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S170 + 24U]), _S167);
        accLo_0[3U] = _S177;
        float _S178 = fma(pj_7, attnHi_0(vw_31), _S168);
        accHi_0[3U] = _S178;
        float pj_8 = (*(&kernelContext_0)->ps_0)[_S62];
        float _S179 = _S169 + (*(&kernelContext_0)->ps_0)[_S62];
        uint _S180 = 264U + kk_0;
        uint vw_32 = (*(&kernelContext_0)->vs_0)[_S180];
        float _S181 = fma((*(&kernelContext_0)->ps_0)[_S62], attnLo_0((*(&kernelContext_0)->vs_0)[_S180]), _S171);
        accLo_0[0U] = _S181;
        float _S182 = fma(pj_8, attnHi_0(vw_32), _S172);
        accHi_0[0U] = _S182;
        uint vw_33 = (*(&kernelContext_0)->vs_0)[_S180 + 8U];
        float _S183 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S180 + 8U]), _S173);
        accLo_0[1U] = _S183;
        float _S184 = fma(pj_8, attnHi_0(vw_33), _S174);
        accHi_0[1U] = _S184;
        uint vw_34 = (*(&kernelContext_0)->vs_0)[_S180 + 16U];
        float _S185 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S180 + 16U]), _S175);
        accLo_0[2U] = _S185;
        float _S186 = fma(pj_8, attnHi_0(vw_34), _S176);
        accHi_0[2U] = _S186;
        uint vw_35 = (*(&kernelContext_0)->vs_0)[_S180 + 24U];
        float _S187 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S180 + 24U]), _S177);
        accLo_0[3U] = _S187;
        float _S188 = fma(pj_8, attnHi_0(vw_35), _S178);
        accHi_0[3U] = _S188;
        float pj_9 = (*(&kernelContext_0)->ps_0)[_S63];
        float _S189 = _S179 + (*(&kernelContext_0)->ps_0)[_S63];
        uint _S190 = 297U + kk_0;
        uint vw_36 = (*(&kernelContext_0)->vs_0)[_S190];
        float _S191 = fma((*(&kernelContext_0)->ps_0)[_S63], attnLo_0((*(&kernelContext_0)->vs_0)[_S190]), _S181);
        accLo_0[0U] = _S191;
        float _S192 = fma(pj_9, attnHi_0(vw_36), _S182);
        accHi_0[0U] = _S192;
        uint vw_37 = (*(&kernelContext_0)->vs_0)[_S190 + 8U];
        float _S193 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S190 + 8U]), _S183);
        accLo_0[1U] = _S193;
        float _S194 = fma(pj_9, attnHi_0(vw_37), _S184);
        accHi_0[1U] = _S194;
        uint vw_38 = (*(&kernelContext_0)->vs_0)[_S190 + 16U];
        float _S195 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S190 + 16U]), _S185);
        accLo_0[2U] = _S195;
        float _S196 = fma(pj_9, attnHi_0(vw_38), _S186);
        accHi_0[2U] = _S196;
        uint vw_39 = (*(&kernelContext_0)->vs_0)[_S190 + 24U];
        float _S197 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S190 + 24U]), _S187);
        accLo_0[3U] = _S197;
        float _S198 = fma(pj_9, attnHi_0(vw_39), _S188);
        accHi_0[3U] = _S198;
        float pj_10 = (*(&kernelContext_0)->ps_0)[_S64];
        float _S199 = _S189 + (*(&kernelContext_0)->ps_0)[_S64];
        uint _S200 = 330U + kk_0;
        uint vw_40 = (*(&kernelContext_0)->vs_0)[_S200];
        float _S201 = fma((*(&kernelContext_0)->ps_0)[_S64], attnLo_0((*(&kernelContext_0)->vs_0)[_S200]), _S191);
        accLo_0[0U] = _S201;
        float _S202 = fma(pj_10, attnHi_0(vw_40), _S192);
        accHi_0[0U] = _S202;
        uint vw_41 = (*(&kernelContext_0)->vs_0)[_S200 + 8U];
        float _S203 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S200 + 8U]), _S193);
        accLo_0[1U] = _S203;
        float _S204 = fma(pj_10, attnHi_0(vw_41), _S194);
        accHi_0[1U] = _S204;
        uint vw_42 = (*(&kernelContext_0)->vs_0)[_S200 + 16U];
        float _S205 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S200 + 16U]), _S195);
        accLo_0[2U] = _S205;
        float _S206 = fma(pj_10, attnHi_0(vw_42), _S196);
        accHi_0[2U] = _S206;
        uint vw_43 = (*(&kernelContext_0)->vs_0)[_S200 + 24U];
        float _S207 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S200 + 24U]), _S197);
        accLo_0[3U] = _S207;
        float _S208 = fma(pj_10, attnHi_0(vw_43), _S198);
        accHi_0[3U] = _S208;
        float pj_11 = (*(&kernelContext_0)->ps_0)[_S65];
        float _S209 = _S199 + (*(&kernelContext_0)->ps_0)[_S65];
        uint _S210 = 363U + kk_0;
        uint vw_44 = (*(&kernelContext_0)->vs_0)[_S210];
        float _S211 = fma((*(&kernelContext_0)->ps_0)[_S65], attnLo_0((*(&kernelContext_0)->vs_0)[_S210]), _S201);
        accLo_0[0U] = _S211;
        float _S212 = fma(pj_11, attnHi_0(vw_44), _S202);
        accHi_0[0U] = _S212;
        uint vw_45 = (*(&kernelContext_0)->vs_0)[_S210 + 8U];
        float _S213 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S210 + 8U]), _S203);
        accLo_0[1U] = _S213;
        float _S214 = fma(pj_11, attnHi_0(vw_45), _S204);
        accHi_0[1U] = _S214;
        uint vw_46 = (*(&kernelContext_0)->vs_0)[_S210 + 16U];
        float _S215 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S210 + 16U]), _S205);
        accLo_0[2U] = _S215;
        float _S216 = fma(pj_11, attnHi_0(vw_46), _S206);
        accHi_0[2U] = _S216;
        uint vw_47 = (*(&kernelContext_0)->vs_0)[_S210 + 24U];
        float _S217 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S210 + 24U]), _S207);
        accLo_0[3U] = _S217;
        float _S218 = fma(pj_11, attnHi_0(vw_47), _S208);
        accHi_0[3U] = _S218;
        float pj_12 = (*(&kernelContext_0)->ps_0)[_S66];
        float _S219 = _S209 + (*(&kernelContext_0)->ps_0)[_S66];
        uint _S220 = 396U + kk_0;
        uint vw_48 = (*(&kernelContext_0)->vs_0)[_S220];
        float _S221 = fma((*(&kernelContext_0)->ps_0)[_S66], attnLo_0((*(&kernelContext_0)->vs_0)[_S220]), _S211);
        accLo_0[0U] = _S221;
        float _S222 = fma(pj_12, attnHi_0(vw_48), _S212);
        accHi_0[0U] = _S222;
        uint vw_49 = (*(&kernelContext_0)->vs_0)[_S220 + 8U];
        float _S223 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S220 + 8U]), _S213);
        accLo_0[1U] = _S223;
        float _S224 = fma(pj_12, attnHi_0(vw_49), _S214);
        accHi_0[1U] = _S224;
        uint vw_50 = (*(&kernelContext_0)->vs_0)[_S220 + 16U];
        float _S225 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S220 + 16U]), _S215);
        accLo_0[2U] = _S225;
        float _S226 = fma(pj_12, attnHi_0(vw_50), _S216);
        accHi_0[2U] = _S226;
        uint vw_51 = (*(&kernelContext_0)->vs_0)[_S220 + 24U];
        float _S227 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S220 + 24U]), _S217);
        accLo_0[3U] = _S227;
        float _S228 = fma(pj_12, attnHi_0(vw_51), _S218);
        accHi_0[3U] = _S228;
        float pj_13 = (*(&kernelContext_0)->ps_0)[_S67];
        float _S229 = _S219 + (*(&kernelContext_0)->ps_0)[_S67];
        uint _S230 = 429U + kk_0;
        uint vw_52 = (*(&kernelContext_0)->vs_0)[_S230];
        float _S231 = fma((*(&kernelContext_0)->ps_0)[_S67], attnLo_0((*(&kernelContext_0)->vs_0)[_S230]), _S221);
        accLo_0[0U] = _S231;
        float _S232 = fma(pj_13, attnHi_0(vw_52), _S222);
        accHi_0[0U] = _S232;
        uint vw_53 = (*(&kernelContext_0)->vs_0)[_S230 + 8U];
        float _S233 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S230 + 8U]), _S223);
        accLo_0[1U] = _S233;
        float _S234 = fma(pj_13, attnHi_0(vw_53), _S224);
        accHi_0[1U] = _S234;
        uint vw_54 = (*(&kernelContext_0)->vs_0)[_S230 + 16U];
        float _S235 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S230 + 16U]), _S225);
        accLo_0[2U] = _S235;
        float _S236 = fma(pj_13, attnHi_0(vw_54), _S226);
        accHi_0[2U] = _S236;
        uint vw_55 = (*(&kernelContext_0)->vs_0)[_S230 + 24U];
        float _S237 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S230 + 24U]), _S227);
        accLo_0[3U] = _S237;
        float _S238 = fma(pj_13, attnHi_0(vw_55), _S228);
        accHi_0[3U] = _S238;
        float pj_14 = (*(&kernelContext_0)->ps_0)[_S68];
        float _S239 = _S229 + (*(&kernelContext_0)->ps_0)[_S68];
        uint _S240 = 462U + kk_0;
        uint vw_56 = (*(&kernelContext_0)->vs_0)[_S240];
        float _S241 = fma((*(&kernelContext_0)->ps_0)[_S68], attnLo_0((*(&kernelContext_0)->vs_0)[_S240]), _S231);
        accLo_0[0U] = _S241;
        float _S242 = fma(pj_14, attnHi_0(vw_56), _S232);
        accHi_0[0U] = _S242;
        uint vw_57 = (*(&kernelContext_0)->vs_0)[_S240 + 8U];
        float _S243 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S240 + 8U]), _S233);
        accLo_0[1U] = _S243;
        float _S244 = fma(pj_14, attnHi_0(vw_57), _S234);
        accHi_0[1U] = _S244;
        uint vw_58 = (*(&kernelContext_0)->vs_0)[_S240 + 16U];
        float _S245 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S240 + 16U]), _S235);
        accLo_0[2U] = _S245;
        float _S246 = fma(pj_14, attnHi_0(vw_58), _S236);
        accHi_0[2U] = _S246;
        uint vw_59 = (*(&kernelContext_0)->vs_0)[_S240 + 24U];
        float _S247 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S240 + 24U]), _S237);
        accLo_0[3U] = _S247;
        float _S248 = fma(pj_14, attnHi_0(vw_59), _S238);
        accHi_0[3U] = _S248;
        float pj_15 = (*(&kernelContext_0)->ps_0)[_S69];
        float _S249 = _S239 + (*(&kernelContext_0)->ps_0)[_S69];
        uint _S250 = 495U + kk_0;
        uint vw_60 = (*(&kernelContext_0)->vs_0)[_S250];
        float _S251 = fma((*(&kernelContext_0)->ps_0)[_S69], attnLo_0((*(&kernelContext_0)->vs_0)[_S250]), _S241);
        accLo_0[0U] = _S251;
        float _S252 = fma(pj_15, attnHi_0(vw_60), _S242);
        accHi_0[0U] = _S252;
        uint vw_61 = (*(&kernelContext_0)->vs_0)[_S250 + 8U];
        float _S253 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S250 + 8U]), _S243);
        accLo_0[1U] = _S253;
        float _S254 = fma(pj_15, attnHi_0(vw_61), _S244);
        accHi_0[1U] = _S254;
        uint vw_62 = (*(&kernelContext_0)->vs_0)[_S250 + 16U];
        float _S255 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S250 + 16U]), _S245);
        accLo_0[2U] = _S255;
        float _S256 = fma(pj_15, attnHi_0(vw_62), _S246);
        accHi_0[2U] = _S256;
        uint vw_63 = (*(&kernelContext_0)->vs_0)[_S250 + 24U];
        float _S257 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S250 + 24U]), _S247);
        accLo_0[3U] = _S257;
        float _S258 = fma(pj_15, attnHi_0(vw_63), _S248);
        accHi_0[3U] = _S258;
        float pj_16 = (*(&kernelContext_0)->ps_0)[_S70];
        float _S259 = _S249 + (*(&kernelContext_0)->ps_0)[_S70];
        uint _S260 = 528U + kk_0;
        uint vw_64 = (*(&kernelContext_0)->vs_0)[_S260];
        float _S261 = fma((*(&kernelContext_0)->ps_0)[_S70], attnLo_0((*(&kernelContext_0)->vs_0)[_S260]), _S251);
        accLo_0[0U] = _S261;
        float _S262 = fma(pj_16, attnHi_0(vw_64), _S252);
        accHi_0[0U] = _S262;
        uint vw_65 = (*(&kernelContext_0)->vs_0)[_S260 + 8U];
        float _S263 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S260 + 8U]), _S253);
        accLo_0[1U] = _S263;
        float _S264 = fma(pj_16, attnHi_0(vw_65), _S254);
        accHi_0[1U] = _S264;
        uint vw_66 = (*(&kernelContext_0)->vs_0)[_S260 + 16U];
        float _S265 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S260 + 16U]), _S255);
        accLo_0[2U] = _S265;
        float _S266 = fma(pj_16, attnHi_0(vw_66), _S256);
        accHi_0[2U] = _S266;
        uint vw_67 = (*(&kernelContext_0)->vs_0)[_S260 + 24U];
        float _S267 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S260 + 24U]), _S257);
        accLo_0[3U] = _S267;
        float _S268 = fma(pj_16, attnHi_0(vw_67), _S258);
        accHi_0[3U] = _S268;
        float pj_17 = (*(&kernelContext_0)->ps_0)[_S71];
        float _S269 = _S259 + (*(&kernelContext_0)->ps_0)[_S71];
        uint _S270 = 561U + kk_0;
        uint vw_68 = (*(&kernelContext_0)->vs_0)[_S270];
        float _S271 = fma((*(&kernelContext_0)->ps_0)[_S71], attnLo_0((*(&kernelContext_0)->vs_0)[_S270]), _S261);
        accLo_0[0U] = _S271;
        float _S272 = fma(pj_17, attnHi_0(vw_68), _S262);
        accHi_0[0U] = _S272;
        uint vw_69 = (*(&kernelContext_0)->vs_0)[_S270 + 8U];
        float _S273 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S270 + 8U]), _S263);
        accLo_0[1U] = _S273;
        float _S274 = fma(pj_17, attnHi_0(vw_69), _S264);
        accHi_0[1U] = _S274;
        uint vw_70 = (*(&kernelContext_0)->vs_0)[_S270 + 16U];
        float _S275 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S270 + 16U]), _S265);
        accLo_0[2U] = _S275;
        float _S276 = fma(pj_17, attnHi_0(vw_70), _S266);
        accHi_0[2U] = _S276;
        uint vw_71 = (*(&kernelContext_0)->vs_0)[_S270 + 24U];
        float _S277 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S270 + 24U]), _S267);
        accLo_0[3U] = _S277;
        float _S278 = fma(pj_17, attnHi_0(vw_71), _S268);
        accHi_0[3U] = _S278;
        float pj_18 = (*(&kernelContext_0)->ps_0)[_S72];
        float _S279 = _S269 + (*(&kernelContext_0)->ps_0)[_S72];
        uint _S280 = 594U + kk_0;
        uint vw_72 = (*(&kernelContext_0)->vs_0)[_S280];
        float _S281 = fma((*(&kernelContext_0)->ps_0)[_S72], attnLo_0((*(&kernelContext_0)->vs_0)[_S280]), _S271);
        accLo_0[0U] = _S281;
        float _S282 = fma(pj_18, attnHi_0(vw_72), _S272);
        accHi_0[0U] = _S282;
        uint vw_73 = (*(&kernelContext_0)->vs_0)[_S280 + 8U];
        float _S283 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S280 + 8U]), _S273);
        accLo_0[1U] = _S283;
        float _S284 = fma(pj_18, attnHi_0(vw_73), _S274);
        accHi_0[1U] = _S284;
        uint vw_74 = (*(&kernelContext_0)->vs_0)[_S280 + 16U];
        float _S285 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S280 + 16U]), _S275);
        accLo_0[2U] = _S285;
        float _S286 = fma(pj_18, attnHi_0(vw_74), _S276);
        accHi_0[2U] = _S286;
        uint vw_75 = (*(&kernelContext_0)->vs_0)[_S280 + 24U];
        float _S287 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S280 + 24U]), _S277);
        accLo_0[3U] = _S287;
        float _S288 = fma(pj_18, attnHi_0(vw_75), _S278);
        accHi_0[3U] = _S288;
        float pj_19 = (*(&kernelContext_0)->ps_0)[_S73];
        float _S289 = _S279 + (*(&kernelContext_0)->ps_0)[_S73];
        uint _S290 = 627U + kk_0;
        uint vw_76 = (*(&kernelContext_0)->vs_0)[_S290];
        float _S291 = fma((*(&kernelContext_0)->ps_0)[_S73], attnLo_0((*(&kernelContext_0)->vs_0)[_S290]), _S281);
        accLo_0[0U] = _S291;
        float _S292 = fma(pj_19, attnHi_0(vw_76), _S282);
        accHi_0[0U] = _S292;
        uint vw_77 = (*(&kernelContext_0)->vs_0)[_S290 + 8U];
        float _S293 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S290 + 8U]), _S283);
        accLo_0[1U] = _S293;
        float _S294 = fma(pj_19, attnHi_0(vw_77), _S284);
        accHi_0[1U] = _S294;
        uint vw_78 = (*(&kernelContext_0)->vs_0)[_S290 + 16U];
        float _S295 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S290 + 16U]), _S285);
        accLo_0[2U] = _S295;
        float _S296 = fma(pj_19, attnHi_0(vw_78), _S286);
        accHi_0[2U] = _S296;
        uint vw_79 = (*(&kernelContext_0)->vs_0)[_S290 + 24U];
        float _S297 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S290 + 24U]), _S287);
        accLo_0[3U] = _S297;
        float _S298 = fma(pj_19, attnHi_0(vw_79), _S288);
        accHi_0[3U] = _S298;
        float pj_20 = (*(&kernelContext_0)->ps_0)[_S74];
        float _S299 = _S289 + (*(&kernelContext_0)->ps_0)[_S74];
        uint _S300 = 660U + kk_0;
        uint vw_80 = (*(&kernelContext_0)->vs_0)[_S300];
        float _S301 = fma((*(&kernelContext_0)->ps_0)[_S74], attnLo_0((*(&kernelContext_0)->vs_0)[_S300]), _S291);
        accLo_0[0U] = _S301;
        float _S302 = fma(pj_20, attnHi_0(vw_80), _S292);
        accHi_0[0U] = _S302;
        uint vw_81 = (*(&kernelContext_0)->vs_0)[_S300 + 8U];
        float _S303 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S300 + 8U]), _S293);
        accLo_0[1U] = _S303;
        float _S304 = fma(pj_20, attnHi_0(vw_81), _S294);
        accHi_0[1U] = _S304;
        uint vw_82 = (*(&kernelContext_0)->vs_0)[_S300 + 16U];
        float _S305 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S300 + 16U]), _S295);
        accLo_0[2U] = _S305;
        float _S306 = fma(pj_20, attnHi_0(vw_82), _S296);
        accHi_0[2U] = _S306;
        uint vw_83 = (*(&kernelContext_0)->vs_0)[_S300 + 24U];
        float _S307 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S300 + 24U]), _S297);
        accLo_0[3U] = _S307;
        float _S308 = fma(pj_20, attnHi_0(vw_83), _S298);
        accHi_0[3U] = _S308;
        float pj_21 = (*(&kernelContext_0)->ps_0)[_S75];
        float _S309 = _S299 + (*(&kernelContext_0)->ps_0)[_S75];
        uint _S310 = 693U + kk_0;
        uint vw_84 = (*(&kernelContext_0)->vs_0)[_S310];
        float _S311 = fma((*(&kernelContext_0)->ps_0)[_S75], attnLo_0((*(&kernelContext_0)->vs_0)[_S310]), _S301);
        accLo_0[0U] = _S311;
        float _S312 = fma(pj_21, attnHi_0(vw_84), _S302);
        accHi_0[0U] = _S312;
        uint vw_85 = (*(&kernelContext_0)->vs_0)[_S310 + 8U];
        float _S313 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S310 + 8U]), _S303);
        accLo_0[1U] = _S313;
        float _S314 = fma(pj_21, attnHi_0(vw_85), _S304);
        accHi_0[1U] = _S314;
        uint vw_86 = (*(&kernelContext_0)->vs_0)[_S310 + 16U];
        float _S315 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S310 + 16U]), _S305);
        accLo_0[2U] = _S315;
        float _S316 = fma(pj_21, attnHi_0(vw_86), _S306);
        accHi_0[2U] = _S316;
        uint vw_87 = (*(&kernelContext_0)->vs_0)[_S310 + 24U];
        float _S317 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S310 + 24U]), _S307);
        accLo_0[3U] = _S317;
        float _S318 = fma(pj_21, attnHi_0(vw_87), _S308);
        accHi_0[3U] = _S318;
        float pj_22 = (*(&kernelContext_0)->ps_0)[_S76];
        float _S319 = _S309 + (*(&kernelContext_0)->ps_0)[_S76];
        uint _S320 = 726U + kk_0;
        uint vw_88 = (*(&kernelContext_0)->vs_0)[_S320];
        float _S321 = fma((*(&kernelContext_0)->ps_0)[_S76], attnLo_0((*(&kernelContext_0)->vs_0)[_S320]), _S311);
        accLo_0[0U] = _S321;
        float _S322 = fma(pj_22, attnHi_0(vw_88), _S312);
        accHi_0[0U] = _S322;
        uint vw_89 = (*(&kernelContext_0)->vs_0)[_S320 + 8U];
        float _S323 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S320 + 8U]), _S313);
        accLo_0[1U] = _S323;
        float _S324 = fma(pj_22, attnHi_0(vw_89), _S314);
        accHi_0[1U] = _S324;
        uint vw_90 = (*(&kernelContext_0)->vs_0)[_S320 + 16U];
        float _S325 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S320 + 16U]), _S315);
        accLo_0[2U] = _S325;
        float _S326 = fma(pj_22, attnHi_0(vw_90), _S316);
        accHi_0[2U] = _S326;
        uint vw_91 = (*(&kernelContext_0)->vs_0)[_S320 + 24U];
        float _S327 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S320 + 24U]), _S317);
        accLo_0[3U] = _S327;
        float _S328 = fma(pj_22, attnHi_0(vw_91), _S318);
        accHi_0[3U] = _S328;
        float pj_23 = (*(&kernelContext_0)->ps_0)[_S77];
        float _S329 = _S319 + (*(&kernelContext_0)->ps_0)[_S77];
        uint _S330 = 759U + kk_0;
        uint vw_92 = (*(&kernelContext_0)->vs_0)[_S330];
        float _S331 = fma((*(&kernelContext_0)->ps_0)[_S77], attnLo_0((*(&kernelContext_0)->vs_0)[_S330]), _S321);
        accLo_0[0U] = _S331;
        float _S332 = fma(pj_23, attnHi_0(vw_92), _S322);
        accHi_0[0U] = _S332;
        uint vw_93 = (*(&kernelContext_0)->vs_0)[_S330 + 8U];
        float _S333 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S330 + 8U]), _S323);
        accLo_0[1U] = _S333;
        float _S334 = fma(pj_23, attnHi_0(vw_93), _S324);
        accHi_0[1U] = _S334;
        uint vw_94 = (*(&kernelContext_0)->vs_0)[_S330 + 16U];
        float _S335 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S330 + 16U]), _S325);
        accLo_0[2U] = _S335;
        float _S336 = fma(pj_23, attnHi_0(vw_94), _S326);
        accHi_0[2U] = _S336;
        uint vw_95 = (*(&kernelContext_0)->vs_0)[_S330 + 24U];
        float _S337 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S330 + 24U]), _S327);
        accLo_0[3U] = _S337;
        float _S338 = fma(pj_23, attnHi_0(vw_95), _S328);
        accHi_0[3U] = _S338;
        float pj_24 = (*(&kernelContext_0)->ps_0)[_S78];
        float _S339 = _S329 + (*(&kernelContext_0)->ps_0)[_S78];
        uint _S340 = 792U + kk_0;
        uint vw_96 = (*(&kernelContext_0)->vs_0)[_S340];
        float _S341 = fma((*(&kernelContext_0)->ps_0)[_S78], attnLo_0((*(&kernelContext_0)->vs_0)[_S340]), _S331);
        accLo_0[0U] = _S341;
        float _S342 = fma(pj_24, attnHi_0(vw_96), _S332);
        accHi_0[0U] = _S342;
        uint vw_97 = (*(&kernelContext_0)->vs_0)[_S340 + 8U];
        float _S343 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S340 + 8U]), _S333);
        accLo_0[1U] = _S343;
        float _S344 = fma(pj_24, attnHi_0(vw_97), _S334);
        accHi_0[1U] = _S344;
        uint vw_98 = (*(&kernelContext_0)->vs_0)[_S340 + 16U];
        float _S345 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S340 + 16U]), _S335);
        accLo_0[2U] = _S345;
        float _S346 = fma(pj_24, attnHi_0(vw_98), _S336);
        accHi_0[2U] = _S346;
        uint vw_99 = (*(&kernelContext_0)->vs_0)[_S340 + 24U];
        float _S347 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S340 + 24U]), _S337);
        accLo_0[3U] = _S347;
        float _S348 = fma(pj_24, attnHi_0(vw_99), _S338);
        accHi_0[3U] = _S348;
        float pj_25 = (*(&kernelContext_0)->ps_0)[_S79];
        float _S349 = _S339 + (*(&kernelContext_0)->ps_0)[_S79];
        uint _S350 = 825U + kk_0;
        uint vw_100 = (*(&kernelContext_0)->vs_0)[_S350];
        float _S351 = fma((*(&kernelContext_0)->ps_0)[_S79], attnLo_0((*(&kernelContext_0)->vs_0)[_S350]), _S341);
        accLo_0[0U] = _S351;
        float _S352 = fma(pj_25, attnHi_0(vw_100), _S342);
        accHi_0[0U] = _S352;
        uint vw_101 = (*(&kernelContext_0)->vs_0)[_S350 + 8U];
        float _S353 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S350 + 8U]), _S343);
        accLo_0[1U] = _S353;
        float _S354 = fma(pj_25, attnHi_0(vw_101), _S344);
        accHi_0[1U] = _S354;
        uint vw_102 = (*(&kernelContext_0)->vs_0)[_S350 + 16U];
        float _S355 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S350 + 16U]), _S345);
        accLo_0[2U] = _S355;
        float _S356 = fma(pj_25, attnHi_0(vw_102), _S346);
        accHi_0[2U] = _S356;
        uint vw_103 = (*(&kernelContext_0)->vs_0)[_S350 + 24U];
        float _S357 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S350 + 24U]), _S347);
        accLo_0[3U] = _S357;
        float _S358 = fma(pj_25, attnHi_0(vw_103), _S348);
        accHi_0[3U] = _S358;
        float pj_26 = (*(&kernelContext_0)->ps_0)[_S80];
        float _S359 = _S349 + (*(&kernelContext_0)->ps_0)[_S80];
        uint _S360 = 858U + kk_0;
        uint vw_104 = (*(&kernelContext_0)->vs_0)[_S360];
        float _S361 = fma((*(&kernelContext_0)->ps_0)[_S80], attnLo_0((*(&kernelContext_0)->vs_0)[_S360]), _S351);
        accLo_0[0U] = _S361;
        float _S362 = fma(pj_26, attnHi_0(vw_104), _S352);
        accHi_0[0U] = _S362;
        uint vw_105 = (*(&kernelContext_0)->vs_0)[_S360 + 8U];
        float _S363 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S360 + 8U]), _S353);
        accLo_0[1U] = _S363;
        float _S364 = fma(pj_26, attnHi_0(vw_105), _S354);
        accHi_0[1U] = _S364;
        uint vw_106 = (*(&kernelContext_0)->vs_0)[_S360 + 16U];
        float _S365 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S360 + 16U]), _S355);
        accLo_0[2U] = _S365;
        float _S366 = fma(pj_26, attnHi_0(vw_106), _S356);
        accHi_0[2U] = _S366;
        uint vw_107 = (*(&kernelContext_0)->vs_0)[_S360 + 24U];
        float _S367 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S360 + 24U]), _S357);
        accLo_0[3U] = _S367;
        float _S368 = fma(pj_26, attnHi_0(vw_107), _S358);
        accHi_0[3U] = _S368;
        float pj_27 = (*(&kernelContext_0)->ps_0)[_S81];
        float _S369 = _S359 + (*(&kernelContext_0)->ps_0)[_S81];
        uint _S370 = 891U + kk_0;
        uint vw_108 = (*(&kernelContext_0)->vs_0)[_S370];
        float _S371 = fma((*(&kernelContext_0)->ps_0)[_S81], attnLo_0((*(&kernelContext_0)->vs_0)[_S370]), _S361);
        accLo_0[0U] = _S371;
        float _S372 = fma(pj_27, attnHi_0(vw_108), _S362);
        accHi_0[0U] = _S372;
        uint vw_109 = (*(&kernelContext_0)->vs_0)[_S370 + 8U];
        float _S373 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S370 + 8U]), _S363);
        accLo_0[1U] = _S373;
        float _S374 = fma(pj_27, attnHi_0(vw_109), _S364);
        accHi_0[1U] = _S374;
        uint vw_110 = (*(&kernelContext_0)->vs_0)[_S370 + 16U];
        float _S375 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S370 + 16U]), _S365);
        accLo_0[2U] = _S375;
        float _S376 = fma(pj_27, attnHi_0(vw_110), _S366);
        accHi_0[2U] = _S376;
        uint vw_111 = (*(&kernelContext_0)->vs_0)[_S370 + 24U];
        float _S377 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S370 + 24U]), _S367);
        accLo_0[3U] = _S377;
        float _S378 = fma(pj_27, attnHi_0(vw_111), _S368);
        accHi_0[3U] = _S378;
        float pj_28 = (*(&kernelContext_0)->ps_0)[_S82];
        float _S379 = _S369 + (*(&kernelContext_0)->ps_0)[_S82];
        uint _S380 = 924U + kk_0;
        uint vw_112 = (*(&kernelContext_0)->vs_0)[_S380];
        float _S381 = fma((*(&kernelContext_0)->ps_0)[_S82], attnLo_0((*(&kernelContext_0)->vs_0)[_S380]), _S371);
        accLo_0[0U] = _S381;
        float _S382 = fma(pj_28, attnHi_0(vw_112), _S372);
        accHi_0[0U] = _S382;
        uint vw_113 = (*(&kernelContext_0)->vs_0)[_S380 + 8U];
        float _S383 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S380 + 8U]), _S373);
        accLo_0[1U] = _S383;
        float _S384 = fma(pj_28, attnHi_0(vw_113), _S374);
        accHi_0[1U] = _S384;
        uint vw_114 = (*(&kernelContext_0)->vs_0)[_S380 + 16U];
        float _S385 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S380 + 16U]), _S375);
        accLo_0[2U] = _S385;
        float _S386 = fma(pj_28, attnHi_0(vw_114), _S376);
        accHi_0[2U] = _S386;
        uint vw_115 = (*(&kernelContext_0)->vs_0)[_S380 + 24U];
        float _S387 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S380 + 24U]), _S377);
        accLo_0[3U] = _S387;
        float _S388 = fma(pj_28, attnHi_0(vw_115), _S378);
        accHi_0[3U] = _S388;
        float pj_29 = (*(&kernelContext_0)->ps_0)[_S83];
        float _S389 = _S379 + (*(&kernelContext_0)->ps_0)[_S83];
        uint _S390 = 957U + kk_0;
        uint vw_116 = (*(&kernelContext_0)->vs_0)[_S390];
        float _S391 = fma((*(&kernelContext_0)->ps_0)[_S83], attnLo_0((*(&kernelContext_0)->vs_0)[_S390]), _S381);
        accLo_0[0U] = _S391;
        float _S392 = fma(pj_29, attnHi_0(vw_116), _S382);
        accHi_0[0U] = _S392;
        uint vw_117 = (*(&kernelContext_0)->vs_0)[_S390 + 8U];
        float _S393 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S390 + 8U]), _S383);
        accLo_0[1U] = _S393;
        float _S394 = fma(pj_29, attnHi_0(vw_117), _S384);
        accHi_0[1U] = _S394;
        uint vw_118 = (*(&kernelContext_0)->vs_0)[_S390 + 16U];
        float _S395 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S390 + 16U]), _S385);
        accLo_0[2U] = _S395;
        float _S396 = fma(pj_29, attnHi_0(vw_118), _S386);
        accHi_0[2U] = _S396;
        uint vw_119 = (*(&kernelContext_0)->vs_0)[_S390 + 24U];
        float _S397 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S390 + 24U]), _S387);
        accLo_0[3U] = _S397;
        float _S398 = fma(pj_29, attnHi_0(vw_119), _S388);
        accHi_0[3U] = _S398;
        float pj_30 = (*(&kernelContext_0)->ps_0)[_S84];
        float _S399 = _S389 + (*(&kernelContext_0)->ps_0)[_S84];
        uint _S400 = 990U + kk_0;
        uint vw_120 = (*(&kernelContext_0)->vs_0)[_S400];
        float _S401 = fma((*(&kernelContext_0)->ps_0)[_S84], attnLo_0((*(&kernelContext_0)->vs_0)[_S400]), _S391);
        accLo_0[0U] = _S401;
        float _S402 = fma(pj_30, attnHi_0(vw_120), _S392);
        accHi_0[0U] = _S402;
        uint vw_121 = (*(&kernelContext_0)->vs_0)[_S400 + 8U];
        float _S403 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S400 + 8U]), _S393);
        accLo_0[1U] = _S403;
        float _S404 = fma(pj_30, attnHi_0(vw_121), _S394);
        accHi_0[1U] = _S404;
        uint vw_122 = (*(&kernelContext_0)->vs_0)[_S400 + 16U];
        float _S405 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S400 + 16U]), _S395);
        accLo_0[2U] = _S405;
        float _S406 = fma(pj_30, attnHi_0(vw_122), _S396);
        accHi_0[2U] = _S406;
        uint vw_123 = (*(&kernelContext_0)->vs_0)[_S400 + 24U];
        float _S407 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S400 + 24U]), _S397);
        accLo_0[3U] = _S407;
        float _S408 = fma(pj_30, attnHi_0(vw_123), _S398);
        accHi_0[3U] = _S408;
        float pj_31 = (*(&kernelContext_0)->ps_0)[_S85];
        float l_1 = _S399 + (*(&kernelContext_0)->ps_0)[_S85];
        uint _S409 = 1023U + kk_0;
        uint vw_124 = (*(&kernelContext_0)->vs_0)[_S409];
        accLo_0[0U] = fma((*(&kernelContext_0)->ps_0)[_S85], attnLo_0((*(&kernelContext_0)->vs_0)[_S409]), _S401);
        accHi_0[0U] = fma(pj_31, attnHi_0(vw_124), _S402);
        uint vw_125 = (*(&kernelContext_0)->vs_0)[_S409 + 8U];
        accLo_0[1U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S409 + 8U]), _S403);
        accHi_0[1U] = fma(pj_31, attnHi_0(vw_125), _S404);
        uint vw_126 = (*(&kernelContext_0)->vs_0)[_S409 + 16U];
        accLo_0[2U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S409 + 16U]), _S405);
        accHi_0[2U] = fma(pj_31, attnHi_0(vw_126), _S406);
        uint vw_127 = (*(&kernelContext_0)->vs_0)[_S409 + 24U];
        accLo_0[3U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S409 + 24U]), _S407);
        accHi_0[3U] = fma(pj_31, attnHi_0(vw_127), _S408);
        b_0 = b_0 + 32U;
        m_0 = _S86;
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
        uint _S410 = attnAccBase_0((&kernelContext_0)->p_0->splits_1, rows_1) + pr_30 * (&kernelContext_0)->p_0->headDim_0;
        if(kk_0 == 0U)
        {
            uint _S411 = 2U * pr_30;
            *((&kernelContext_0)->partials_0+_S411) = m_0;
            *((&kernelContext_0)->partials_0+(_S411 + 1U)) = l_0;
        }
        if(kk_0 < _S20)
        {
            uint _S412 = _S410 + 2U * kk_0;
            float device* _S413 = (&kernelContext_0)->partials_0+_S412;
            bool _S414 = l_0 == 0.0;
            if(_S414)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[0U];
            }
            *_S413 = m_0;
            float device* _S415 = (&kernelContext_0)->partials_0+(_S412 + 1U);
            if(_S414)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[0U];
            }
            *_S415 = m_0;
        }
        uint pair_0 = kk_0 + 8U;
        if(pair_0 < _S20)
        {
            uint _S416 = _S410 + 2U * pair_0;
            float device* _S417 = (&kernelContext_0)->partials_0+_S416;
            bool _S418 = l_0 == 0.0;
            if(_S418)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[1U];
            }
            *_S417 = m_0;
            float device* _S419 = (&kernelContext_0)->partials_0+(_S416 + 1U);
            if(_S418)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[1U];
            }
            *_S419 = m_0;
        }
        uint pair_1 = kk_0 + 16U;
        if(pair_1 < _S20)
        {
            uint _S420 = _S410 + 2U * pair_1;
            float device* _S421 = (&kernelContext_0)->partials_0+_S420;
            bool _S422 = l_0 == 0.0;
            if(_S422)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[2U];
            }
            *_S421 = m_0;
            float device* _S423 = (&kernelContext_0)->partials_0+(_S420 + 1U);
            if(_S422)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[2U];
            }
            *_S423 = m_0;
        }
        uint pair_2 = kk_0 + 24U;
        if(pair_2 < _S20)
        {
            uint _S424 = _S410 + 2U * pair_2;
            float device* _S425 = (&kernelContext_0)->partials_0+_S424;
            bool _S426 = l_0 == 0.0;
            if(_S426)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[3U];
            }
            *_S425 = m_0;
            float device* _S427 = (&kernelContext_0)->partials_0+(_S424 + 1U);
            if(_S426)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[3U];
            }
            *_S427 = m_0;
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
        float device* _S428 = (&kernelContext_0)->lse_0+row_1;
        if(l_0 == 0.0)
        {
            m_0 = _S31;
        }
        else
        {
            m_0 = m_0 + log(l_0);
        }
        *_S428 = m_0;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                if(kk_0 >= _S20)
                {
                    break;
                }
                bool _S429 = l_0 == 0.0;
                if(_S429)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[0U] / l_0;
                }
                if(_S429)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[0U] / l_0;
                }
                uint _S430 = 2U * kk_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S430)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S430 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_3 = kk_0 + 8U;
                if(pair_3 >= _S20)
                {
                    break;
                }
                bool _S431 = l_0 == 0.0;
                if(_S431)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[1U] / l_0;
                }
                if(_S431)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[1U] / l_0;
                }
                uint _S432 = 2U * pair_3;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S432)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S432 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_4 = kk_0 + 16U;
                if(pair_4 >= _S20)
                {
                    break;
                }
                bool _S433 = l_0 == 0.0;
                if(_S433)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[2U] / l_0;
                }
                if(_S433)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[2U] / l_0;
                }
                uint _S434 = 2U * pair_4;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S434)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S434 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_5 = kk_0 + 24U;
                if(pair_5 >= _S20)
                {
                    break;
                }
                bool _S435 = l_0 == 0.0;
                if(_S435)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[3U] / l_0;
                }
                if(_S435)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[3U] / l_0;
                }
                uint _S436 = 2U * pair_5;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S436)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S436 + 1U)) = hi_0;
                break;
            }
            break;
        }
        break;
    }
    return;
}

