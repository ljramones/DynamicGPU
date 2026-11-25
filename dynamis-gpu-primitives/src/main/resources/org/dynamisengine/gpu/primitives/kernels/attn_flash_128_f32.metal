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
    array<uint, int(1040)> threadgroup* qs_0;
    array<uint, int(2080)> threadgroup* ks_0;
    array<uint, int(2080)> threadgroup* vs_0;
    array<float, int(512)> threadgroup* ss_0;
    array<float, int(512)> threadgroup* ps_0;
};

[[kernel]] void attn_flash_128_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(8)]], uint device* q_1 [[buffer(0)]], uint device* lens_2 [[buffer(7)]], uint device* table_1 [[buffer(5)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partials_1 [[buffer(4)]], float device* lse_1 [[buffer(6)]], float device* o_1 [[buffer(3)]])
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
        if(e_0 < 1024U)
        {
        }
        else
        {
            break;
        }
        uint row_0 = r0_0 + e_0 / 64U;
        uint pr_0 = e_0 % 64U;
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
        (*(&kernelContext_0)->qs_0)[e_0 / 64U * 65U + pr_0] = w_2;
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
    thread array<float, int(8)> accLo_0;
    thread array<float, int(8)> accHi_0;
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
            if(e_0 < 2048U)
            {
            }
            else
            {
                break;
            }
            uint j_0 = e_0 / 64U;
            uint pr_1 = e_0 % 64U;
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
            uint _S35 = j_0 * 65U + pr_1;
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
                        if(at_0 < 64U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S36 = lr_0 * 65U;
                        uint _S37 = kk_0 * 65U;
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
                        if(p0_1 < 64U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S40 = lr_0 * 65U;
                        uint _S41 = j_1 * 65U;
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
                        if(p0_0 < 64U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S45 = lr_0 * 65U;
                        uint _S46 = j_2 * 65U;
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
                        if(p0_2 < 64U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S50 = lr_0 * 65U;
                        uint _S51 = j_3 * 65U;
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
        float _S100 = accLo_0[4U] * alpha_0;
        accLo_0[4U] = _S100;
        float _S101 = accHi_0[4U] * alpha_0;
        accHi_0[4U] = _S101;
        float _S102 = accLo_0[5U] * alpha_0;
        accLo_0[5U] = _S102;
        float _S103 = accHi_0[5U] * alpha_0;
        accHi_0[5U] = _S103;
        float _S104 = accLo_0[6U] * alpha_0;
        accLo_0[6U] = _S104;
        float _S105 = accHi_0[6U] * alpha_0;
        accHi_0[6U] = _S105;
        float _S106 = accLo_0[7U] * alpha_0;
        accLo_0[7U] = _S106;
        float _S107 = accHi_0[7U] * alpha_0;
        accHi_0[7U] = _S107;
        float pj_0 = (*(&kernelContext_0)->ps_0)[_S5];
        float _S108 = _S91 + (*(&kernelContext_0)->ps_0)[_S5];
        uint vw_0 = (*(&kernelContext_0)->vs_0)[kk_0];
        float _S109 = fma((*(&kernelContext_0)->ps_0)[_S5], attnLo_0((*(&kernelContext_0)->vs_0)[kk_0]), _S92);
        accLo_0[0U] = _S109;
        float _S110 = fma(pj_0, attnHi_0(vw_0), _S93);
        accHi_0[0U] = _S110;
        uint vw_1 = (*(&kernelContext_0)->vs_0)[_S7];
        float _S111 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S7]), _S94);
        accLo_0[1U] = _S111;
        float _S112 = fma(pj_0, attnHi_0(vw_1), _S95);
        accHi_0[1U] = _S112;
        uint vw_2 = (*(&kernelContext_0)->vs_0)[_S9];
        float _S113 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S9]), _S96);
        accLo_0[2U] = _S113;
        float _S114 = fma(pj_0, attnHi_0(vw_2), _S97);
        accHi_0[2U] = _S114;
        uint vw_3 = (*(&kernelContext_0)->vs_0)[_S11];
        float _S115 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S11]), _S98);
        accLo_0[3U] = _S115;
        float _S116 = fma(pj_0, attnHi_0(vw_3), _S99);
        accHi_0[3U] = _S116;
        uint vw_4 = (*(&kernelContext_0)->vs_0)[kk_0 + 32U];
        float _S117 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 32U]), _S100);
        accLo_0[4U] = _S117;
        float _S118 = fma(pj_0, attnHi_0(vw_4), _S101);
        accHi_0[4U] = _S118;
        uint vw_5 = (*(&kernelContext_0)->vs_0)[kk_0 + 40U];
        float _S119 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 40U]), _S102);
        accLo_0[5U] = _S119;
        float _S120 = fma(pj_0, attnHi_0(vw_5), _S103);
        accHi_0[5U] = _S120;
        uint vw_6 = (*(&kernelContext_0)->vs_0)[kk_0 + 48U];
        float _S121 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 48U]), _S104);
        accLo_0[6U] = _S121;
        float _S122 = fma(pj_0, attnHi_0(vw_6), _S105);
        accHi_0[6U] = _S122;
        uint vw_7 = (*(&kernelContext_0)->vs_0)[kk_0 + 56U];
        float _S123 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 56U]), _S106);
        accLo_0[7U] = _S123;
        float _S124 = fma(pj_0, attnHi_0(vw_7), _S107);
        accHi_0[7U] = _S124;
        float pj_1 = (*(&kernelContext_0)->ps_0)[_S55];
        float _S125 = _S108 + (*(&kernelContext_0)->ps_0)[_S55];
        uint _S126 = 65U + kk_0;
        uint vw_8 = (*(&kernelContext_0)->vs_0)[_S126];
        float _S127 = fma((*(&kernelContext_0)->ps_0)[_S55], attnLo_0((*(&kernelContext_0)->vs_0)[_S126]), _S109);
        accLo_0[0U] = _S127;
        float _S128 = fma(pj_1, attnHi_0(vw_8), _S110);
        accHi_0[0U] = _S128;
        uint vw_9 = (*(&kernelContext_0)->vs_0)[_S126 + 8U];
        float _S129 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S126 + 8U]), _S111);
        accLo_0[1U] = _S129;
        float _S130 = fma(pj_1, attnHi_0(vw_9), _S112);
        accHi_0[1U] = _S130;
        uint vw_10 = (*(&kernelContext_0)->vs_0)[_S126 + 16U];
        float _S131 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S126 + 16U]), _S113);
        accLo_0[2U] = _S131;
        float _S132 = fma(pj_1, attnHi_0(vw_10), _S114);
        accHi_0[2U] = _S132;
        uint vw_11 = (*(&kernelContext_0)->vs_0)[_S126 + 24U];
        float _S133 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S126 + 24U]), _S115);
        accLo_0[3U] = _S133;
        float _S134 = fma(pj_1, attnHi_0(vw_11), _S116);
        accHi_0[3U] = _S134;
        uint vw_12 = (*(&kernelContext_0)->vs_0)[_S126 + 32U];
        float _S135 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S126 + 32U]), _S117);
        accLo_0[4U] = _S135;
        float _S136 = fma(pj_1, attnHi_0(vw_12), _S118);
        accHi_0[4U] = _S136;
        uint vw_13 = (*(&kernelContext_0)->vs_0)[_S126 + 40U];
        float _S137 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S126 + 40U]), _S119);
        accLo_0[5U] = _S137;
        float _S138 = fma(pj_1, attnHi_0(vw_13), _S120);
        accHi_0[5U] = _S138;
        uint vw_14 = (*(&kernelContext_0)->vs_0)[_S126 + 48U];
        float _S139 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S126 + 48U]), _S121);
        accLo_0[6U] = _S139;
        float _S140 = fma(pj_1, attnHi_0(vw_14), _S122);
        accHi_0[6U] = _S140;
        uint vw_15 = (*(&kernelContext_0)->vs_0)[_S126 + 56U];
        float _S141 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S126 + 56U]), _S123);
        accLo_0[7U] = _S141;
        float _S142 = fma(pj_1, attnHi_0(vw_15), _S124);
        accHi_0[7U] = _S142;
        float pj_2 = (*(&kernelContext_0)->ps_0)[_S56];
        float _S143 = _S125 + (*(&kernelContext_0)->ps_0)[_S56];
        uint _S144 = 130U + kk_0;
        uint vw_16 = (*(&kernelContext_0)->vs_0)[_S144];
        float _S145 = fma((*(&kernelContext_0)->ps_0)[_S56], attnLo_0((*(&kernelContext_0)->vs_0)[_S144]), _S127);
        accLo_0[0U] = _S145;
        float _S146 = fma(pj_2, attnHi_0(vw_16), _S128);
        accHi_0[0U] = _S146;
        uint vw_17 = (*(&kernelContext_0)->vs_0)[_S144 + 8U];
        float _S147 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S144 + 8U]), _S129);
        accLo_0[1U] = _S147;
        float _S148 = fma(pj_2, attnHi_0(vw_17), _S130);
        accHi_0[1U] = _S148;
        uint vw_18 = (*(&kernelContext_0)->vs_0)[_S144 + 16U];
        float _S149 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S144 + 16U]), _S131);
        accLo_0[2U] = _S149;
        float _S150 = fma(pj_2, attnHi_0(vw_18), _S132);
        accHi_0[2U] = _S150;
        uint vw_19 = (*(&kernelContext_0)->vs_0)[_S144 + 24U];
        float _S151 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S144 + 24U]), _S133);
        accLo_0[3U] = _S151;
        float _S152 = fma(pj_2, attnHi_0(vw_19), _S134);
        accHi_0[3U] = _S152;
        uint vw_20 = (*(&kernelContext_0)->vs_0)[_S144 + 32U];
        float _S153 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S144 + 32U]), _S135);
        accLo_0[4U] = _S153;
        float _S154 = fma(pj_2, attnHi_0(vw_20), _S136);
        accHi_0[4U] = _S154;
        uint vw_21 = (*(&kernelContext_0)->vs_0)[_S144 + 40U];
        float _S155 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S144 + 40U]), _S137);
        accLo_0[5U] = _S155;
        float _S156 = fma(pj_2, attnHi_0(vw_21), _S138);
        accHi_0[5U] = _S156;
        uint vw_22 = (*(&kernelContext_0)->vs_0)[_S144 + 48U];
        float _S157 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S144 + 48U]), _S139);
        accLo_0[6U] = _S157;
        float _S158 = fma(pj_2, attnHi_0(vw_22), _S140);
        accHi_0[6U] = _S158;
        uint vw_23 = (*(&kernelContext_0)->vs_0)[_S144 + 56U];
        float _S159 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S144 + 56U]), _S141);
        accLo_0[7U] = _S159;
        float _S160 = fma(pj_2, attnHi_0(vw_23), _S142);
        accHi_0[7U] = _S160;
        float pj_3 = (*(&kernelContext_0)->ps_0)[_S57];
        float _S161 = _S143 + (*(&kernelContext_0)->ps_0)[_S57];
        uint _S162 = 195U + kk_0;
        uint vw_24 = (*(&kernelContext_0)->vs_0)[_S162];
        float _S163 = fma((*(&kernelContext_0)->ps_0)[_S57], attnLo_0((*(&kernelContext_0)->vs_0)[_S162]), _S145);
        accLo_0[0U] = _S163;
        float _S164 = fma(pj_3, attnHi_0(vw_24), _S146);
        accHi_0[0U] = _S164;
        uint vw_25 = (*(&kernelContext_0)->vs_0)[_S162 + 8U];
        float _S165 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S162 + 8U]), _S147);
        accLo_0[1U] = _S165;
        float _S166 = fma(pj_3, attnHi_0(vw_25), _S148);
        accHi_0[1U] = _S166;
        uint vw_26 = (*(&kernelContext_0)->vs_0)[_S162 + 16U];
        float _S167 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S162 + 16U]), _S149);
        accLo_0[2U] = _S167;
        float _S168 = fma(pj_3, attnHi_0(vw_26), _S150);
        accHi_0[2U] = _S168;
        uint vw_27 = (*(&kernelContext_0)->vs_0)[_S162 + 24U];
        float _S169 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S162 + 24U]), _S151);
        accLo_0[3U] = _S169;
        float _S170 = fma(pj_3, attnHi_0(vw_27), _S152);
        accHi_0[3U] = _S170;
        uint vw_28 = (*(&kernelContext_0)->vs_0)[_S162 + 32U];
        float _S171 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S162 + 32U]), _S153);
        accLo_0[4U] = _S171;
        float _S172 = fma(pj_3, attnHi_0(vw_28), _S154);
        accHi_0[4U] = _S172;
        uint vw_29 = (*(&kernelContext_0)->vs_0)[_S162 + 40U];
        float _S173 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S162 + 40U]), _S155);
        accLo_0[5U] = _S173;
        float _S174 = fma(pj_3, attnHi_0(vw_29), _S156);
        accHi_0[5U] = _S174;
        uint vw_30 = (*(&kernelContext_0)->vs_0)[_S162 + 48U];
        float _S175 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S162 + 48U]), _S157);
        accLo_0[6U] = _S175;
        float _S176 = fma(pj_3, attnHi_0(vw_30), _S158);
        accHi_0[6U] = _S176;
        uint vw_31 = (*(&kernelContext_0)->vs_0)[_S162 + 56U];
        float _S177 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S162 + 56U]), _S159);
        accLo_0[7U] = _S177;
        float _S178 = fma(pj_3, attnHi_0(vw_31), _S160);
        accHi_0[7U] = _S178;
        float pj_4 = (*(&kernelContext_0)->ps_0)[_S58];
        float _S179 = _S161 + (*(&kernelContext_0)->ps_0)[_S58];
        uint _S180 = 260U + kk_0;
        uint vw_32 = (*(&kernelContext_0)->vs_0)[_S180];
        float _S181 = fma((*(&kernelContext_0)->ps_0)[_S58], attnLo_0((*(&kernelContext_0)->vs_0)[_S180]), _S163);
        accLo_0[0U] = _S181;
        float _S182 = fma(pj_4, attnHi_0(vw_32), _S164);
        accHi_0[0U] = _S182;
        uint vw_33 = (*(&kernelContext_0)->vs_0)[_S180 + 8U];
        float _S183 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S180 + 8U]), _S165);
        accLo_0[1U] = _S183;
        float _S184 = fma(pj_4, attnHi_0(vw_33), _S166);
        accHi_0[1U] = _S184;
        uint vw_34 = (*(&kernelContext_0)->vs_0)[_S180 + 16U];
        float _S185 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S180 + 16U]), _S167);
        accLo_0[2U] = _S185;
        float _S186 = fma(pj_4, attnHi_0(vw_34), _S168);
        accHi_0[2U] = _S186;
        uint vw_35 = (*(&kernelContext_0)->vs_0)[_S180 + 24U];
        float _S187 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S180 + 24U]), _S169);
        accLo_0[3U] = _S187;
        float _S188 = fma(pj_4, attnHi_0(vw_35), _S170);
        accHi_0[3U] = _S188;
        uint vw_36 = (*(&kernelContext_0)->vs_0)[_S180 + 32U];
        float _S189 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S180 + 32U]), _S171);
        accLo_0[4U] = _S189;
        float _S190 = fma(pj_4, attnHi_0(vw_36), _S172);
        accHi_0[4U] = _S190;
        uint vw_37 = (*(&kernelContext_0)->vs_0)[_S180 + 40U];
        float _S191 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S180 + 40U]), _S173);
        accLo_0[5U] = _S191;
        float _S192 = fma(pj_4, attnHi_0(vw_37), _S174);
        accHi_0[5U] = _S192;
        uint vw_38 = (*(&kernelContext_0)->vs_0)[_S180 + 48U];
        float _S193 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S180 + 48U]), _S175);
        accLo_0[6U] = _S193;
        float _S194 = fma(pj_4, attnHi_0(vw_38), _S176);
        accHi_0[6U] = _S194;
        uint vw_39 = (*(&kernelContext_0)->vs_0)[_S180 + 56U];
        float _S195 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S180 + 56U]), _S177);
        accLo_0[7U] = _S195;
        float _S196 = fma(pj_4, attnHi_0(vw_39), _S178);
        accHi_0[7U] = _S196;
        float pj_5 = (*(&kernelContext_0)->ps_0)[_S59];
        float _S197 = _S179 + (*(&kernelContext_0)->ps_0)[_S59];
        uint _S198 = 325U + kk_0;
        uint vw_40 = (*(&kernelContext_0)->vs_0)[_S198];
        float _S199 = fma((*(&kernelContext_0)->ps_0)[_S59], attnLo_0((*(&kernelContext_0)->vs_0)[_S198]), _S181);
        accLo_0[0U] = _S199;
        float _S200 = fma(pj_5, attnHi_0(vw_40), _S182);
        accHi_0[0U] = _S200;
        uint vw_41 = (*(&kernelContext_0)->vs_0)[_S198 + 8U];
        float _S201 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S198 + 8U]), _S183);
        accLo_0[1U] = _S201;
        float _S202 = fma(pj_5, attnHi_0(vw_41), _S184);
        accHi_0[1U] = _S202;
        uint vw_42 = (*(&kernelContext_0)->vs_0)[_S198 + 16U];
        float _S203 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S198 + 16U]), _S185);
        accLo_0[2U] = _S203;
        float _S204 = fma(pj_5, attnHi_0(vw_42), _S186);
        accHi_0[2U] = _S204;
        uint vw_43 = (*(&kernelContext_0)->vs_0)[_S198 + 24U];
        float _S205 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S198 + 24U]), _S187);
        accLo_0[3U] = _S205;
        float _S206 = fma(pj_5, attnHi_0(vw_43), _S188);
        accHi_0[3U] = _S206;
        uint vw_44 = (*(&kernelContext_0)->vs_0)[_S198 + 32U];
        float _S207 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S198 + 32U]), _S189);
        accLo_0[4U] = _S207;
        float _S208 = fma(pj_5, attnHi_0(vw_44), _S190);
        accHi_0[4U] = _S208;
        uint vw_45 = (*(&kernelContext_0)->vs_0)[_S198 + 40U];
        float _S209 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S198 + 40U]), _S191);
        accLo_0[5U] = _S209;
        float _S210 = fma(pj_5, attnHi_0(vw_45), _S192);
        accHi_0[5U] = _S210;
        uint vw_46 = (*(&kernelContext_0)->vs_0)[_S198 + 48U];
        float _S211 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S198 + 48U]), _S193);
        accLo_0[6U] = _S211;
        float _S212 = fma(pj_5, attnHi_0(vw_46), _S194);
        accHi_0[6U] = _S212;
        uint vw_47 = (*(&kernelContext_0)->vs_0)[_S198 + 56U];
        float _S213 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S198 + 56U]), _S195);
        accLo_0[7U] = _S213;
        float _S214 = fma(pj_5, attnHi_0(vw_47), _S196);
        accHi_0[7U] = _S214;
        float pj_6 = (*(&kernelContext_0)->ps_0)[_S60];
        float _S215 = _S197 + (*(&kernelContext_0)->ps_0)[_S60];
        uint _S216 = 390U + kk_0;
        uint vw_48 = (*(&kernelContext_0)->vs_0)[_S216];
        float _S217 = fma((*(&kernelContext_0)->ps_0)[_S60], attnLo_0((*(&kernelContext_0)->vs_0)[_S216]), _S199);
        accLo_0[0U] = _S217;
        float _S218 = fma(pj_6, attnHi_0(vw_48), _S200);
        accHi_0[0U] = _S218;
        uint vw_49 = (*(&kernelContext_0)->vs_0)[_S216 + 8U];
        float _S219 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S216 + 8U]), _S201);
        accLo_0[1U] = _S219;
        float _S220 = fma(pj_6, attnHi_0(vw_49), _S202);
        accHi_0[1U] = _S220;
        uint vw_50 = (*(&kernelContext_0)->vs_0)[_S216 + 16U];
        float _S221 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S216 + 16U]), _S203);
        accLo_0[2U] = _S221;
        float _S222 = fma(pj_6, attnHi_0(vw_50), _S204);
        accHi_0[2U] = _S222;
        uint vw_51 = (*(&kernelContext_0)->vs_0)[_S216 + 24U];
        float _S223 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S216 + 24U]), _S205);
        accLo_0[3U] = _S223;
        float _S224 = fma(pj_6, attnHi_0(vw_51), _S206);
        accHi_0[3U] = _S224;
        uint vw_52 = (*(&kernelContext_0)->vs_0)[_S216 + 32U];
        float _S225 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S216 + 32U]), _S207);
        accLo_0[4U] = _S225;
        float _S226 = fma(pj_6, attnHi_0(vw_52), _S208);
        accHi_0[4U] = _S226;
        uint vw_53 = (*(&kernelContext_0)->vs_0)[_S216 + 40U];
        float _S227 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S216 + 40U]), _S209);
        accLo_0[5U] = _S227;
        float _S228 = fma(pj_6, attnHi_0(vw_53), _S210);
        accHi_0[5U] = _S228;
        uint vw_54 = (*(&kernelContext_0)->vs_0)[_S216 + 48U];
        float _S229 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S216 + 48U]), _S211);
        accLo_0[6U] = _S229;
        float _S230 = fma(pj_6, attnHi_0(vw_54), _S212);
        accHi_0[6U] = _S230;
        uint vw_55 = (*(&kernelContext_0)->vs_0)[_S216 + 56U];
        float _S231 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S216 + 56U]), _S213);
        accLo_0[7U] = _S231;
        float _S232 = fma(pj_6, attnHi_0(vw_55), _S214);
        accHi_0[7U] = _S232;
        float pj_7 = (*(&kernelContext_0)->ps_0)[_S61];
        float _S233 = _S215 + (*(&kernelContext_0)->ps_0)[_S61];
        uint _S234 = 455U + kk_0;
        uint vw_56 = (*(&kernelContext_0)->vs_0)[_S234];
        float _S235 = fma((*(&kernelContext_0)->ps_0)[_S61], attnLo_0((*(&kernelContext_0)->vs_0)[_S234]), _S217);
        accLo_0[0U] = _S235;
        float _S236 = fma(pj_7, attnHi_0(vw_56), _S218);
        accHi_0[0U] = _S236;
        uint vw_57 = (*(&kernelContext_0)->vs_0)[_S234 + 8U];
        float _S237 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S234 + 8U]), _S219);
        accLo_0[1U] = _S237;
        float _S238 = fma(pj_7, attnHi_0(vw_57), _S220);
        accHi_0[1U] = _S238;
        uint vw_58 = (*(&kernelContext_0)->vs_0)[_S234 + 16U];
        float _S239 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S234 + 16U]), _S221);
        accLo_0[2U] = _S239;
        float _S240 = fma(pj_7, attnHi_0(vw_58), _S222);
        accHi_0[2U] = _S240;
        uint vw_59 = (*(&kernelContext_0)->vs_0)[_S234 + 24U];
        float _S241 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S234 + 24U]), _S223);
        accLo_0[3U] = _S241;
        float _S242 = fma(pj_7, attnHi_0(vw_59), _S224);
        accHi_0[3U] = _S242;
        uint vw_60 = (*(&kernelContext_0)->vs_0)[_S234 + 32U];
        float _S243 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S234 + 32U]), _S225);
        accLo_0[4U] = _S243;
        float _S244 = fma(pj_7, attnHi_0(vw_60), _S226);
        accHi_0[4U] = _S244;
        uint vw_61 = (*(&kernelContext_0)->vs_0)[_S234 + 40U];
        float _S245 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S234 + 40U]), _S227);
        accLo_0[5U] = _S245;
        float _S246 = fma(pj_7, attnHi_0(vw_61), _S228);
        accHi_0[5U] = _S246;
        uint vw_62 = (*(&kernelContext_0)->vs_0)[_S234 + 48U];
        float _S247 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S234 + 48U]), _S229);
        accLo_0[6U] = _S247;
        float _S248 = fma(pj_7, attnHi_0(vw_62), _S230);
        accHi_0[6U] = _S248;
        uint vw_63 = (*(&kernelContext_0)->vs_0)[_S234 + 56U];
        float _S249 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S234 + 56U]), _S231);
        accLo_0[7U] = _S249;
        float _S250 = fma(pj_7, attnHi_0(vw_63), _S232);
        accHi_0[7U] = _S250;
        float pj_8 = (*(&kernelContext_0)->ps_0)[_S62];
        float _S251 = _S233 + (*(&kernelContext_0)->ps_0)[_S62];
        uint _S252 = 520U + kk_0;
        uint vw_64 = (*(&kernelContext_0)->vs_0)[_S252];
        float _S253 = fma((*(&kernelContext_0)->ps_0)[_S62], attnLo_0((*(&kernelContext_0)->vs_0)[_S252]), _S235);
        accLo_0[0U] = _S253;
        float _S254 = fma(pj_8, attnHi_0(vw_64), _S236);
        accHi_0[0U] = _S254;
        uint vw_65 = (*(&kernelContext_0)->vs_0)[_S252 + 8U];
        float _S255 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S252 + 8U]), _S237);
        accLo_0[1U] = _S255;
        float _S256 = fma(pj_8, attnHi_0(vw_65), _S238);
        accHi_0[1U] = _S256;
        uint vw_66 = (*(&kernelContext_0)->vs_0)[_S252 + 16U];
        float _S257 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S252 + 16U]), _S239);
        accLo_0[2U] = _S257;
        float _S258 = fma(pj_8, attnHi_0(vw_66), _S240);
        accHi_0[2U] = _S258;
        uint vw_67 = (*(&kernelContext_0)->vs_0)[_S252 + 24U];
        float _S259 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S252 + 24U]), _S241);
        accLo_0[3U] = _S259;
        float _S260 = fma(pj_8, attnHi_0(vw_67), _S242);
        accHi_0[3U] = _S260;
        uint vw_68 = (*(&kernelContext_0)->vs_0)[_S252 + 32U];
        float _S261 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S252 + 32U]), _S243);
        accLo_0[4U] = _S261;
        float _S262 = fma(pj_8, attnHi_0(vw_68), _S244);
        accHi_0[4U] = _S262;
        uint vw_69 = (*(&kernelContext_0)->vs_0)[_S252 + 40U];
        float _S263 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S252 + 40U]), _S245);
        accLo_0[5U] = _S263;
        float _S264 = fma(pj_8, attnHi_0(vw_69), _S246);
        accHi_0[5U] = _S264;
        uint vw_70 = (*(&kernelContext_0)->vs_0)[_S252 + 48U];
        float _S265 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S252 + 48U]), _S247);
        accLo_0[6U] = _S265;
        float _S266 = fma(pj_8, attnHi_0(vw_70), _S248);
        accHi_0[6U] = _S266;
        uint vw_71 = (*(&kernelContext_0)->vs_0)[_S252 + 56U];
        float _S267 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S252 + 56U]), _S249);
        accLo_0[7U] = _S267;
        float _S268 = fma(pj_8, attnHi_0(vw_71), _S250);
        accHi_0[7U] = _S268;
        float pj_9 = (*(&kernelContext_0)->ps_0)[_S63];
        float _S269 = _S251 + (*(&kernelContext_0)->ps_0)[_S63];
        uint _S270 = 585U + kk_0;
        uint vw_72 = (*(&kernelContext_0)->vs_0)[_S270];
        float _S271 = fma((*(&kernelContext_0)->ps_0)[_S63], attnLo_0((*(&kernelContext_0)->vs_0)[_S270]), _S253);
        accLo_0[0U] = _S271;
        float _S272 = fma(pj_9, attnHi_0(vw_72), _S254);
        accHi_0[0U] = _S272;
        uint vw_73 = (*(&kernelContext_0)->vs_0)[_S270 + 8U];
        float _S273 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S270 + 8U]), _S255);
        accLo_0[1U] = _S273;
        float _S274 = fma(pj_9, attnHi_0(vw_73), _S256);
        accHi_0[1U] = _S274;
        uint vw_74 = (*(&kernelContext_0)->vs_0)[_S270 + 16U];
        float _S275 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S270 + 16U]), _S257);
        accLo_0[2U] = _S275;
        float _S276 = fma(pj_9, attnHi_0(vw_74), _S258);
        accHi_0[2U] = _S276;
        uint vw_75 = (*(&kernelContext_0)->vs_0)[_S270 + 24U];
        float _S277 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S270 + 24U]), _S259);
        accLo_0[3U] = _S277;
        float _S278 = fma(pj_9, attnHi_0(vw_75), _S260);
        accHi_0[3U] = _S278;
        uint vw_76 = (*(&kernelContext_0)->vs_0)[_S270 + 32U];
        float _S279 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S270 + 32U]), _S261);
        accLo_0[4U] = _S279;
        float _S280 = fma(pj_9, attnHi_0(vw_76), _S262);
        accHi_0[4U] = _S280;
        uint vw_77 = (*(&kernelContext_0)->vs_0)[_S270 + 40U];
        float _S281 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S270 + 40U]), _S263);
        accLo_0[5U] = _S281;
        float _S282 = fma(pj_9, attnHi_0(vw_77), _S264);
        accHi_0[5U] = _S282;
        uint vw_78 = (*(&kernelContext_0)->vs_0)[_S270 + 48U];
        float _S283 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S270 + 48U]), _S265);
        accLo_0[6U] = _S283;
        float _S284 = fma(pj_9, attnHi_0(vw_78), _S266);
        accHi_0[6U] = _S284;
        uint vw_79 = (*(&kernelContext_0)->vs_0)[_S270 + 56U];
        float _S285 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S270 + 56U]), _S267);
        accLo_0[7U] = _S285;
        float _S286 = fma(pj_9, attnHi_0(vw_79), _S268);
        accHi_0[7U] = _S286;
        float pj_10 = (*(&kernelContext_0)->ps_0)[_S64];
        float _S287 = _S269 + (*(&kernelContext_0)->ps_0)[_S64];
        uint _S288 = 650U + kk_0;
        uint vw_80 = (*(&kernelContext_0)->vs_0)[_S288];
        float _S289 = fma((*(&kernelContext_0)->ps_0)[_S64], attnLo_0((*(&kernelContext_0)->vs_0)[_S288]), _S271);
        accLo_0[0U] = _S289;
        float _S290 = fma(pj_10, attnHi_0(vw_80), _S272);
        accHi_0[0U] = _S290;
        uint vw_81 = (*(&kernelContext_0)->vs_0)[_S288 + 8U];
        float _S291 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S288 + 8U]), _S273);
        accLo_0[1U] = _S291;
        float _S292 = fma(pj_10, attnHi_0(vw_81), _S274);
        accHi_0[1U] = _S292;
        uint vw_82 = (*(&kernelContext_0)->vs_0)[_S288 + 16U];
        float _S293 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S288 + 16U]), _S275);
        accLo_0[2U] = _S293;
        float _S294 = fma(pj_10, attnHi_0(vw_82), _S276);
        accHi_0[2U] = _S294;
        uint vw_83 = (*(&kernelContext_0)->vs_0)[_S288 + 24U];
        float _S295 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S288 + 24U]), _S277);
        accLo_0[3U] = _S295;
        float _S296 = fma(pj_10, attnHi_0(vw_83), _S278);
        accHi_0[3U] = _S296;
        uint vw_84 = (*(&kernelContext_0)->vs_0)[_S288 + 32U];
        float _S297 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S288 + 32U]), _S279);
        accLo_0[4U] = _S297;
        float _S298 = fma(pj_10, attnHi_0(vw_84), _S280);
        accHi_0[4U] = _S298;
        uint vw_85 = (*(&kernelContext_0)->vs_0)[_S288 + 40U];
        float _S299 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S288 + 40U]), _S281);
        accLo_0[5U] = _S299;
        float _S300 = fma(pj_10, attnHi_0(vw_85), _S282);
        accHi_0[5U] = _S300;
        uint vw_86 = (*(&kernelContext_0)->vs_0)[_S288 + 48U];
        float _S301 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S288 + 48U]), _S283);
        accLo_0[6U] = _S301;
        float _S302 = fma(pj_10, attnHi_0(vw_86), _S284);
        accHi_0[6U] = _S302;
        uint vw_87 = (*(&kernelContext_0)->vs_0)[_S288 + 56U];
        float _S303 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S288 + 56U]), _S285);
        accLo_0[7U] = _S303;
        float _S304 = fma(pj_10, attnHi_0(vw_87), _S286);
        accHi_0[7U] = _S304;
        float pj_11 = (*(&kernelContext_0)->ps_0)[_S65];
        float _S305 = _S287 + (*(&kernelContext_0)->ps_0)[_S65];
        uint _S306 = 715U + kk_0;
        uint vw_88 = (*(&kernelContext_0)->vs_0)[_S306];
        float _S307 = fma((*(&kernelContext_0)->ps_0)[_S65], attnLo_0((*(&kernelContext_0)->vs_0)[_S306]), _S289);
        accLo_0[0U] = _S307;
        float _S308 = fma(pj_11, attnHi_0(vw_88), _S290);
        accHi_0[0U] = _S308;
        uint vw_89 = (*(&kernelContext_0)->vs_0)[_S306 + 8U];
        float _S309 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S306 + 8U]), _S291);
        accLo_0[1U] = _S309;
        float _S310 = fma(pj_11, attnHi_0(vw_89), _S292);
        accHi_0[1U] = _S310;
        uint vw_90 = (*(&kernelContext_0)->vs_0)[_S306 + 16U];
        float _S311 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S306 + 16U]), _S293);
        accLo_0[2U] = _S311;
        float _S312 = fma(pj_11, attnHi_0(vw_90), _S294);
        accHi_0[2U] = _S312;
        uint vw_91 = (*(&kernelContext_0)->vs_0)[_S306 + 24U];
        float _S313 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S306 + 24U]), _S295);
        accLo_0[3U] = _S313;
        float _S314 = fma(pj_11, attnHi_0(vw_91), _S296);
        accHi_0[3U] = _S314;
        uint vw_92 = (*(&kernelContext_0)->vs_0)[_S306 + 32U];
        float _S315 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S306 + 32U]), _S297);
        accLo_0[4U] = _S315;
        float _S316 = fma(pj_11, attnHi_0(vw_92), _S298);
        accHi_0[4U] = _S316;
        uint vw_93 = (*(&kernelContext_0)->vs_0)[_S306 + 40U];
        float _S317 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S306 + 40U]), _S299);
        accLo_0[5U] = _S317;
        float _S318 = fma(pj_11, attnHi_0(vw_93), _S300);
        accHi_0[5U] = _S318;
        uint vw_94 = (*(&kernelContext_0)->vs_0)[_S306 + 48U];
        float _S319 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S306 + 48U]), _S301);
        accLo_0[6U] = _S319;
        float _S320 = fma(pj_11, attnHi_0(vw_94), _S302);
        accHi_0[6U] = _S320;
        uint vw_95 = (*(&kernelContext_0)->vs_0)[_S306 + 56U];
        float _S321 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S306 + 56U]), _S303);
        accLo_0[7U] = _S321;
        float _S322 = fma(pj_11, attnHi_0(vw_95), _S304);
        accHi_0[7U] = _S322;
        float pj_12 = (*(&kernelContext_0)->ps_0)[_S66];
        float _S323 = _S305 + (*(&kernelContext_0)->ps_0)[_S66];
        uint _S324 = 780U + kk_0;
        uint vw_96 = (*(&kernelContext_0)->vs_0)[_S324];
        float _S325 = fma((*(&kernelContext_0)->ps_0)[_S66], attnLo_0((*(&kernelContext_0)->vs_0)[_S324]), _S307);
        accLo_0[0U] = _S325;
        float _S326 = fma(pj_12, attnHi_0(vw_96), _S308);
        accHi_0[0U] = _S326;
        uint vw_97 = (*(&kernelContext_0)->vs_0)[_S324 + 8U];
        float _S327 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S324 + 8U]), _S309);
        accLo_0[1U] = _S327;
        float _S328 = fma(pj_12, attnHi_0(vw_97), _S310);
        accHi_0[1U] = _S328;
        uint vw_98 = (*(&kernelContext_0)->vs_0)[_S324 + 16U];
        float _S329 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S324 + 16U]), _S311);
        accLo_0[2U] = _S329;
        float _S330 = fma(pj_12, attnHi_0(vw_98), _S312);
        accHi_0[2U] = _S330;
        uint vw_99 = (*(&kernelContext_0)->vs_0)[_S324 + 24U];
        float _S331 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S324 + 24U]), _S313);
        accLo_0[3U] = _S331;
        float _S332 = fma(pj_12, attnHi_0(vw_99), _S314);
        accHi_0[3U] = _S332;
        uint vw_100 = (*(&kernelContext_0)->vs_0)[_S324 + 32U];
        float _S333 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S324 + 32U]), _S315);
        accLo_0[4U] = _S333;
        float _S334 = fma(pj_12, attnHi_0(vw_100), _S316);
        accHi_0[4U] = _S334;
        uint vw_101 = (*(&kernelContext_0)->vs_0)[_S324 + 40U];
        float _S335 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S324 + 40U]), _S317);
        accLo_0[5U] = _S335;
        float _S336 = fma(pj_12, attnHi_0(vw_101), _S318);
        accHi_0[5U] = _S336;
        uint vw_102 = (*(&kernelContext_0)->vs_0)[_S324 + 48U];
        float _S337 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S324 + 48U]), _S319);
        accLo_0[6U] = _S337;
        float _S338 = fma(pj_12, attnHi_0(vw_102), _S320);
        accHi_0[6U] = _S338;
        uint vw_103 = (*(&kernelContext_0)->vs_0)[_S324 + 56U];
        float _S339 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S324 + 56U]), _S321);
        accLo_0[7U] = _S339;
        float _S340 = fma(pj_12, attnHi_0(vw_103), _S322);
        accHi_0[7U] = _S340;
        float pj_13 = (*(&kernelContext_0)->ps_0)[_S67];
        float _S341 = _S323 + (*(&kernelContext_0)->ps_0)[_S67];
        uint _S342 = 845U + kk_0;
        uint vw_104 = (*(&kernelContext_0)->vs_0)[_S342];
        float _S343 = fma((*(&kernelContext_0)->ps_0)[_S67], attnLo_0((*(&kernelContext_0)->vs_0)[_S342]), _S325);
        accLo_0[0U] = _S343;
        float _S344 = fma(pj_13, attnHi_0(vw_104), _S326);
        accHi_0[0U] = _S344;
        uint vw_105 = (*(&kernelContext_0)->vs_0)[_S342 + 8U];
        float _S345 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S342 + 8U]), _S327);
        accLo_0[1U] = _S345;
        float _S346 = fma(pj_13, attnHi_0(vw_105), _S328);
        accHi_0[1U] = _S346;
        uint vw_106 = (*(&kernelContext_0)->vs_0)[_S342 + 16U];
        float _S347 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S342 + 16U]), _S329);
        accLo_0[2U] = _S347;
        float _S348 = fma(pj_13, attnHi_0(vw_106), _S330);
        accHi_0[2U] = _S348;
        uint vw_107 = (*(&kernelContext_0)->vs_0)[_S342 + 24U];
        float _S349 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S342 + 24U]), _S331);
        accLo_0[3U] = _S349;
        float _S350 = fma(pj_13, attnHi_0(vw_107), _S332);
        accHi_0[3U] = _S350;
        uint vw_108 = (*(&kernelContext_0)->vs_0)[_S342 + 32U];
        float _S351 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S342 + 32U]), _S333);
        accLo_0[4U] = _S351;
        float _S352 = fma(pj_13, attnHi_0(vw_108), _S334);
        accHi_0[4U] = _S352;
        uint vw_109 = (*(&kernelContext_0)->vs_0)[_S342 + 40U];
        float _S353 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S342 + 40U]), _S335);
        accLo_0[5U] = _S353;
        float _S354 = fma(pj_13, attnHi_0(vw_109), _S336);
        accHi_0[5U] = _S354;
        uint vw_110 = (*(&kernelContext_0)->vs_0)[_S342 + 48U];
        float _S355 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S342 + 48U]), _S337);
        accLo_0[6U] = _S355;
        float _S356 = fma(pj_13, attnHi_0(vw_110), _S338);
        accHi_0[6U] = _S356;
        uint vw_111 = (*(&kernelContext_0)->vs_0)[_S342 + 56U];
        float _S357 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S342 + 56U]), _S339);
        accLo_0[7U] = _S357;
        float _S358 = fma(pj_13, attnHi_0(vw_111), _S340);
        accHi_0[7U] = _S358;
        float pj_14 = (*(&kernelContext_0)->ps_0)[_S68];
        float _S359 = _S341 + (*(&kernelContext_0)->ps_0)[_S68];
        uint _S360 = 910U + kk_0;
        uint vw_112 = (*(&kernelContext_0)->vs_0)[_S360];
        float _S361 = fma((*(&kernelContext_0)->ps_0)[_S68], attnLo_0((*(&kernelContext_0)->vs_0)[_S360]), _S343);
        accLo_0[0U] = _S361;
        float _S362 = fma(pj_14, attnHi_0(vw_112), _S344);
        accHi_0[0U] = _S362;
        uint vw_113 = (*(&kernelContext_0)->vs_0)[_S360 + 8U];
        float _S363 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S360 + 8U]), _S345);
        accLo_0[1U] = _S363;
        float _S364 = fma(pj_14, attnHi_0(vw_113), _S346);
        accHi_0[1U] = _S364;
        uint vw_114 = (*(&kernelContext_0)->vs_0)[_S360 + 16U];
        float _S365 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S360 + 16U]), _S347);
        accLo_0[2U] = _S365;
        float _S366 = fma(pj_14, attnHi_0(vw_114), _S348);
        accHi_0[2U] = _S366;
        uint vw_115 = (*(&kernelContext_0)->vs_0)[_S360 + 24U];
        float _S367 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S360 + 24U]), _S349);
        accLo_0[3U] = _S367;
        float _S368 = fma(pj_14, attnHi_0(vw_115), _S350);
        accHi_0[3U] = _S368;
        uint vw_116 = (*(&kernelContext_0)->vs_0)[_S360 + 32U];
        float _S369 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S360 + 32U]), _S351);
        accLo_0[4U] = _S369;
        float _S370 = fma(pj_14, attnHi_0(vw_116), _S352);
        accHi_0[4U] = _S370;
        uint vw_117 = (*(&kernelContext_0)->vs_0)[_S360 + 40U];
        float _S371 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S360 + 40U]), _S353);
        accLo_0[5U] = _S371;
        float _S372 = fma(pj_14, attnHi_0(vw_117), _S354);
        accHi_0[5U] = _S372;
        uint vw_118 = (*(&kernelContext_0)->vs_0)[_S360 + 48U];
        float _S373 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S360 + 48U]), _S355);
        accLo_0[6U] = _S373;
        float _S374 = fma(pj_14, attnHi_0(vw_118), _S356);
        accHi_0[6U] = _S374;
        uint vw_119 = (*(&kernelContext_0)->vs_0)[_S360 + 56U];
        float _S375 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S360 + 56U]), _S357);
        accLo_0[7U] = _S375;
        float _S376 = fma(pj_14, attnHi_0(vw_119), _S358);
        accHi_0[7U] = _S376;
        float pj_15 = (*(&kernelContext_0)->ps_0)[_S69];
        float _S377 = _S359 + (*(&kernelContext_0)->ps_0)[_S69];
        uint _S378 = 975U + kk_0;
        uint vw_120 = (*(&kernelContext_0)->vs_0)[_S378];
        float _S379 = fma((*(&kernelContext_0)->ps_0)[_S69], attnLo_0((*(&kernelContext_0)->vs_0)[_S378]), _S361);
        accLo_0[0U] = _S379;
        float _S380 = fma(pj_15, attnHi_0(vw_120), _S362);
        accHi_0[0U] = _S380;
        uint vw_121 = (*(&kernelContext_0)->vs_0)[_S378 + 8U];
        float _S381 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S378 + 8U]), _S363);
        accLo_0[1U] = _S381;
        float _S382 = fma(pj_15, attnHi_0(vw_121), _S364);
        accHi_0[1U] = _S382;
        uint vw_122 = (*(&kernelContext_0)->vs_0)[_S378 + 16U];
        float _S383 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S378 + 16U]), _S365);
        accLo_0[2U] = _S383;
        float _S384 = fma(pj_15, attnHi_0(vw_122), _S366);
        accHi_0[2U] = _S384;
        uint vw_123 = (*(&kernelContext_0)->vs_0)[_S378 + 24U];
        float _S385 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S378 + 24U]), _S367);
        accLo_0[3U] = _S385;
        float _S386 = fma(pj_15, attnHi_0(vw_123), _S368);
        accHi_0[3U] = _S386;
        uint vw_124 = (*(&kernelContext_0)->vs_0)[_S378 + 32U];
        float _S387 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S378 + 32U]), _S369);
        accLo_0[4U] = _S387;
        float _S388 = fma(pj_15, attnHi_0(vw_124), _S370);
        accHi_0[4U] = _S388;
        uint vw_125 = (*(&kernelContext_0)->vs_0)[_S378 + 40U];
        float _S389 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S378 + 40U]), _S371);
        accLo_0[5U] = _S389;
        float _S390 = fma(pj_15, attnHi_0(vw_125), _S372);
        accHi_0[5U] = _S390;
        uint vw_126 = (*(&kernelContext_0)->vs_0)[_S378 + 48U];
        float _S391 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S378 + 48U]), _S373);
        accLo_0[6U] = _S391;
        float _S392 = fma(pj_15, attnHi_0(vw_126), _S374);
        accHi_0[6U] = _S392;
        uint vw_127 = (*(&kernelContext_0)->vs_0)[_S378 + 56U];
        float _S393 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S378 + 56U]), _S375);
        accLo_0[7U] = _S393;
        float _S394 = fma(pj_15, attnHi_0(vw_127), _S376);
        accHi_0[7U] = _S394;
        float pj_16 = (*(&kernelContext_0)->ps_0)[_S70];
        float _S395 = _S377 + (*(&kernelContext_0)->ps_0)[_S70];
        uint _S396 = 1040U + kk_0;
        uint vw_128 = (*(&kernelContext_0)->vs_0)[_S396];
        float _S397 = fma((*(&kernelContext_0)->ps_0)[_S70], attnLo_0((*(&kernelContext_0)->vs_0)[_S396]), _S379);
        accLo_0[0U] = _S397;
        float _S398 = fma(pj_16, attnHi_0(vw_128), _S380);
        accHi_0[0U] = _S398;
        uint vw_129 = (*(&kernelContext_0)->vs_0)[_S396 + 8U];
        float _S399 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S396 + 8U]), _S381);
        accLo_0[1U] = _S399;
        float _S400 = fma(pj_16, attnHi_0(vw_129), _S382);
        accHi_0[1U] = _S400;
        uint vw_130 = (*(&kernelContext_0)->vs_0)[_S396 + 16U];
        float _S401 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S396 + 16U]), _S383);
        accLo_0[2U] = _S401;
        float _S402 = fma(pj_16, attnHi_0(vw_130), _S384);
        accHi_0[2U] = _S402;
        uint vw_131 = (*(&kernelContext_0)->vs_0)[_S396 + 24U];
        float _S403 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S396 + 24U]), _S385);
        accLo_0[3U] = _S403;
        float _S404 = fma(pj_16, attnHi_0(vw_131), _S386);
        accHi_0[3U] = _S404;
        uint vw_132 = (*(&kernelContext_0)->vs_0)[_S396 + 32U];
        float _S405 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S396 + 32U]), _S387);
        accLo_0[4U] = _S405;
        float _S406 = fma(pj_16, attnHi_0(vw_132), _S388);
        accHi_0[4U] = _S406;
        uint vw_133 = (*(&kernelContext_0)->vs_0)[_S396 + 40U];
        float _S407 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S396 + 40U]), _S389);
        accLo_0[5U] = _S407;
        float _S408 = fma(pj_16, attnHi_0(vw_133), _S390);
        accHi_0[5U] = _S408;
        uint vw_134 = (*(&kernelContext_0)->vs_0)[_S396 + 48U];
        float _S409 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S396 + 48U]), _S391);
        accLo_0[6U] = _S409;
        float _S410 = fma(pj_16, attnHi_0(vw_134), _S392);
        accHi_0[6U] = _S410;
        uint vw_135 = (*(&kernelContext_0)->vs_0)[_S396 + 56U];
        float _S411 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S396 + 56U]), _S393);
        accLo_0[7U] = _S411;
        float _S412 = fma(pj_16, attnHi_0(vw_135), _S394);
        accHi_0[7U] = _S412;
        float pj_17 = (*(&kernelContext_0)->ps_0)[_S71];
        float _S413 = _S395 + (*(&kernelContext_0)->ps_0)[_S71];
        uint _S414 = 1105U + kk_0;
        uint vw_136 = (*(&kernelContext_0)->vs_0)[_S414];
        float _S415 = fma((*(&kernelContext_0)->ps_0)[_S71], attnLo_0((*(&kernelContext_0)->vs_0)[_S414]), _S397);
        accLo_0[0U] = _S415;
        float _S416 = fma(pj_17, attnHi_0(vw_136), _S398);
        accHi_0[0U] = _S416;
        uint vw_137 = (*(&kernelContext_0)->vs_0)[_S414 + 8U];
        float _S417 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S414 + 8U]), _S399);
        accLo_0[1U] = _S417;
        float _S418 = fma(pj_17, attnHi_0(vw_137), _S400);
        accHi_0[1U] = _S418;
        uint vw_138 = (*(&kernelContext_0)->vs_0)[_S414 + 16U];
        float _S419 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S414 + 16U]), _S401);
        accLo_0[2U] = _S419;
        float _S420 = fma(pj_17, attnHi_0(vw_138), _S402);
        accHi_0[2U] = _S420;
        uint vw_139 = (*(&kernelContext_0)->vs_0)[_S414 + 24U];
        float _S421 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S414 + 24U]), _S403);
        accLo_0[3U] = _S421;
        float _S422 = fma(pj_17, attnHi_0(vw_139), _S404);
        accHi_0[3U] = _S422;
        uint vw_140 = (*(&kernelContext_0)->vs_0)[_S414 + 32U];
        float _S423 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S414 + 32U]), _S405);
        accLo_0[4U] = _S423;
        float _S424 = fma(pj_17, attnHi_0(vw_140), _S406);
        accHi_0[4U] = _S424;
        uint vw_141 = (*(&kernelContext_0)->vs_0)[_S414 + 40U];
        float _S425 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S414 + 40U]), _S407);
        accLo_0[5U] = _S425;
        float _S426 = fma(pj_17, attnHi_0(vw_141), _S408);
        accHi_0[5U] = _S426;
        uint vw_142 = (*(&kernelContext_0)->vs_0)[_S414 + 48U];
        float _S427 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S414 + 48U]), _S409);
        accLo_0[6U] = _S427;
        float _S428 = fma(pj_17, attnHi_0(vw_142), _S410);
        accHi_0[6U] = _S428;
        uint vw_143 = (*(&kernelContext_0)->vs_0)[_S414 + 56U];
        float _S429 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S414 + 56U]), _S411);
        accLo_0[7U] = _S429;
        float _S430 = fma(pj_17, attnHi_0(vw_143), _S412);
        accHi_0[7U] = _S430;
        float pj_18 = (*(&kernelContext_0)->ps_0)[_S72];
        float _S431 = _S413 + (*(&kernelContext_0)->ps_0)[_S72];
        uint _S432 = 1170U + kk_0;
        uint vw_144 = (*(&kernelContext_0)->vs_0)[_S432];
        float _S433 = fma((*(&kernelContext_0)->ps_0)[_S72], attnLo_0((*(&kernelContext_0)->vs_0)[_S432]), _S415);
        accLo_0[0U] = _S433;
        float _S434 = fma(pj_18, attnHi_0(vw_144), _S416);
        accHi_0[0U] = _S434;
        uint vw_145 = (*(&kernelContext_0)->vs_0)[_S432 + 8U];
        float _S435 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S432 + 8U]), _S417);
        accLo_0[1U] = _S435;
        float _S436 = fma(pj_18, attnHi_0(vw_145), _S418);
        accHi_0[1U] = _S436;
        uint vw_146 = (*(&kernelContext_0)->vs_0)[_S432 + 16U];
        float _S437 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S432 + 16U]), _S419);
        accLo_0[2U] = _S437;
        float _S438 = fma(pj_18, attnHi_0(vw_146), _S420);
        accHi_0[2U] = _S438;
        uint vw_147 = (*(&kernelContext_0)->vs_0)[_S432 + 24U];
        float _S439 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S432 + 24U]), _S421);
        accLo_0[3U] = _S439;
        float _S440 = fma(pj_18, attnHi_0(vw_147), _S422);
        accHi_0[3U] = _S440;
        uint vw_148 = (*(&kernelContext_0)->vs_0)[_S432 + 32U];
        float _S441 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S432 + 32U]), _S423);
        accLo_0[4U] = _S441;
        float _S442 = fma(pj_18, attnHi_0(vw_148), _S424);
        accHi_0[4U] = _S442;
        uint vw_149 = (*(&kernelContext_0)->vs_0)[_S432 + 40U];
        float _S443 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S432 + 40U]), _S425);
        accLo_0[5U] = _S443;
        float _S444 = fma(pj_18, attnHi_0(vw_149), _S426);
        accHi_0[5U] = _S444;
        uint vw_150 = (*(&kernelContext_0)->vs_0)[_S432 + 48U];
        float _S445 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S432 + 48U]), _S427);
        accLo_0[6U] = _S445;
        float _S446 = fma(pj_18, attnHi_0(vw_150), _S428);
        accHi_0[6U] = _S446;
        uint vw_151 = (*(&kernelContext_0)->vs_0)[_S432 + 56U];
        float _S447 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S432 + 56U]), _S429);
        accLo_0[7U] = _S447;
        float _S448 = fma(pj_18, attnHi_0(vw_151), _S430);
        accHi_0[7U] = _S448;
        float pj_19 = (*(&kernelContext_0)->ps_0)[_S73];
        float _S449 = _S431 + (*(&kernelContext_0)->ps_0)[_S73];
        uint _S450 = 1235U + kk_0;
        uint vw_152 = (*(&kernelContext_0)->vs_0)[_S450];
        float _S451 = fma((*(&kernelContext_0)->ps_0)[_S73], attnLo_0((*(&kernelContext_0)->vs_0)[_S450]), _S433);
        accLo_0[0U] = _S451;
        float _S452 = fma(pj_19, attnHi_0(vw_152), _S434);
        accHi_0[0U] = _S452;
        uint vw_153 = (*(&kernelContext_0)->vs_0)[_S450 + 8U];
        float _S453 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S450 + 8U]), _S435);
        accLo_0[1U] = _S453;
        float _S454 = fma(pj_19, attnHi_0(vw_153), _S436);
        accHi_0[1U] = _S454;
        uint vw_154 = (*(&kernelContext_0)->vs_0)[_S450 + 16U];
        float _S455 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S450 + 16U]), _S437);
        accLo_0[2U] = _S455;
        float _S456 = fma(pj_19, attnHi_0(vw_154), _S438);
        accHi_0[2U] = _S456;
        uint vw_155 = (*(&kernelContext_0)->vs_0)[_S450 + 24U];
        float _S457 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S450 + 24U]), _S439);
        accLo_0[3U] = _S457;
        float _S458 = fma(pj_19, attnHi_0(vw_155), _S440);
        accHi_0[3U] = _S458;
        uint vw_156 = (*(&kernelContext_0)->vs_0)[_S450 + 32U];
        float _S459 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S450 + 32U]), _S441);
        accLo_0[4U] = _S459;
        float _S460 = fma(pj_19, attnHi_0(vw_156), _S442);
        accHi_0[4U] = _S460;
        uint vw_157 = (*(&kernelContext_0)->vs_0)[_S450 + 40U];
        float _S461 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S450 + 40U]), _S443);
        accLo_0[5U] = _S461;
        float _S462 = fma(pj_19, attnHi_0(vw_157), _S444);
        accHi_0[5U] = _S462;
        uint vw_158 = (*(&kernelContext_0)->vs_0)[_S450 + 48U];
        float _S463 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S450 + 48U]), _S445);
        accLo_0[6U] = _S463;
        float _S464 = fma(pj_19, attnHi_0(vw_158), _S446);
        accHi_0[6U] = _S464;
        uint vw_159 = (*(&kernelContext_0)->vs_0)[_S450 + 56U];
        float _S465 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S450 + 56U]), _S447);
        accLo_0[7U] = _S465;
        float _S466 = fma(pj_19, attnHi_0(vw_159), _S448);
        accHi_0[7U] = _S466;
        float pj_20 = (*(&kernelContext_0)->ps_0)[_S74];
        float _S467 = _S449 + (*(&kernelContext_0)->ps_0)[_S74];
        uint _S468 = 1300U + kk_0;
        uint vw_160 = (*(&kernelContext_0)->vs_0)[_S468];
        float _S469 = fma((*(&kernelContext_0)->ps_0)[_S74], attnLo_0((*(&kernelContext_0)->vs_0)[_S468]), _S451);
        accLo_0[0U] = _S469;
        float _S470 = fma(pj_20, attnHi_0(vw_160), _S452);
        accHi_0[0U] = _S470;
        uint vw_161 = (*(&kernelContext_0)->vs_0)[_S468 + 8U];
        float _S471 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S468 + 8U]), _S453);
        accLo_0[1U] = _S471;
        float _S472 = fma(pj_20, attnHi_0(vw_161), _S454);
        accHi_0[1U] = _S472;
        uint vw_162 = (*(&kernelContext_0)->vs_0)[_S468 + 16U];
        float _S473 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S468 + 16U]), _S455);
        accLo_0[2U] = _S473;
        float _S474 = fma(pj_20, attnHi_0(vw_162), _S456);
        accHi_0[2U] = _S474;
        uint vw_163 = (*(&kernelContext_0)->vs_0)[_S468 + 24U];
        float _S475 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S468 + 24U]), _S457);
        accLo_0[3U] = _S475;
        float _S476 = fma(pj_20, attnHi_0(vw_163), _S458);
        accHi_0[3U] = _S476;
        uint vw_164 = (*(&kernelContext_0)->vs_0)[_S468 + 32U];
        float _S477 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S468 + 32U]), _S459);
        accLo_0[4U] = _S477;
        float _S478 = fma(pj_20, attnHi_0(vw_164), _S460);
        accHi_0[4U] = _S478;
        uint vw_165 = (*(&kernelContext_0)->vs_0)[_S468 + 40U];
        float _S479 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S468 + 40U]), _S461);
        accLo_0[5U] = _S479;
        float _S480 = fma(pj_20, attnHi_0(vw_165), _S462);
        accHi_0[5U] = _S480;
        uint vw_166 = (*(&kernelContext_0)->vs_0)[_S468 + 48U];
        float _S481 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S468 + 48U]), _S463);
        accLo_0[6U] = _S481;
        float _S482 = fma(pj_20, attnHi_0(vw_166), _S464);
        accHi_0[6U] = _S482;
        uint vw_167 = (*(&kernelContext_0)->vs_0)[_S468 + 56U];
        float _S483 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S468 + 56U]), _S465);
        accLo_0[7U] = _S483;
        float _S484 = fma(pj_20, attnHi_0(vw_167), _S466);
        accHi_0[7U] = _S484;
        float pj_21 = (*(&kernelContext_0)->ps_0)[_S75];
        float _S485 = _S467 + (*(&kernelContext_0)->ps_0)[_S75];
        uint _S486 = 1365U + kk_0;
        uint vw_168 = (*(&kernelContext_0)->vs_0)[_S486];
        float _S487 = fma((*(&kernelContext_0)->ps_0)[_S75], attnLo_0((*(&kernelContext_0)->vs_0)[_S486]), _S469);
        accLo_0[0U] = _S487;
        float _S488 = fma(pj_21, attnHi_0(vw_168), _S470);
        accHi_0[0U] = _S488;
        uint vw_169 = (*(&kernelContext_0)->vs_0)[_S486 + 8U];
        float _S489 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S486 + 8U]), _S471);
        accLo_0[1U] = _S489;
        float _S490 = fma(pj_21, attnHi_0(vw_169), _S472);
        accHi_0[1U] = _S490;
        uint vw_170 = (*(&kernelContext_0)->vs_0)[_S486 + 16U];
        float _S491 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S486 + 16U]), _S473);
        accLo_0[2U] = _S491;
        float _S492 = fma(pj_21, attnHi_0(vw_170), _S474);
        accHi_0[2U] = _S492;
        uint vw_171 = (*(&kernelContext_0)->vs_0)[_S486 + 24U];
        float _S493 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S486 + 24U]), _S475);
        accLo_0[3U] = _S493;
        float _S494 = fma(pj_21, attnHi_0(vw_171), _S476);
        accHi_0[3U] = _S494;
        uint vw_172 = (*(&kernelContext_0)->vs_0)[_S486 + 32U];
        float _S495 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S486 + 32U]), _S477);
        accLo_0[4U] = _S495;
        float _S496 = fma(pj_21, attnHi_0(vw_172), _S478);
        accHi_0[4U] = _S496;
        uint vw_173 = (*(&kernelContext_0)->vs_0)[_S486 + 40U];
        float _S497 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S486 + 40U]), _S479);
        accLo_0[5U] = _S497;
        float _S498 = fma(pj_21, attnHi_0(vw_173), _S480);
        accHi_0[5U] = _S498;
        uint vw_174 = (*(&kernelContext_0)->vs_0)[_S486 + 48U];
        float _S499 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S486 + 48U]), _S481);
        accLo_0[6U] = _S499;
        float _S500 = fma(pj_21, attnHi_0(vw_174), _S482);
        accHi_0[6U] = _S500;
        uint vw_175 = (*(&kernelContext_0)->vs_0)[_S486 + 56U];
        float _S501 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S486 + 56U]), _S483);
        accLo_0[7U] = _S501;
        float _S502 = fma(pj_21, attnHi_0(vw_175), _S484);
        accHi_0[7U] = _S502;
        float pj_22 = (*(&kernelContext_0)->ps_0)[_S76];
        float _S503 = _S485 + (*(&kernelContext_0)->ps_0)[_S76];
        uint _S504 = 1430U + kk_0;
        uint vw_176 = (*(&kernelContext_0)->vs_0)[_S504];
        float _S505 = fma((*(&kernelContext_0)->ps_0)[_S76], attnLo_0((*(&kernelContext_0)->vs_0)[_S504]), _S487);
        accLo_0[0U] = _S505;
        float _S506 = fma(pj_22, attnHi_0(vw_176), _S488);
        accHi_0[0U] = _S506;
        uint vw_177 = (*(&kernelContext_0)->vs_0)[_S504 + 8U];
        float _S507 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S504 + 8U]), _S489);
        accLo_0[1U] = _S507;
        float _S508 = fma(pj_22, attnHi_0(vw_177), _S490);
        accHi_0[1U] = _S508;
        uint vw_178 = (*(&kernelContext_0)->vs_0)[_S504 + 16U];
        float _S509 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S504 + 16U]), _S491);
        accLo_0[2U] = _S509;
        float _S510 = fma(pj_22, attnHi_0(vw_178), _S492);
        accHi_0[2U] = _S510;
        uint vw_179 = (*(&kernelContext_0)->vs_0)[_S504 + 24U];
        float _S511 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S504 + 24U]), _S493);
        accLo_0[3U] = _S511;
        float _S512 = fma(pj_22, attnHi_0(vw_179), _S494);
        accHi_0[3U] = _S512;
        uint vw_180 = (*(&kernelContext_0)->vs_0)[_S504 + 32U];
        float _S513 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S504 + 32U]), _S495);
        accLo_0[4U] = _S513;
        float _S514 = fma(pj_22, attnHi_0(vw_180), _S496);
        accHi_0[4U] = _S514;
        uint vw_181 = (*(&kernelContext_0)->vs_0)[_S504 + 40U];
        float _S515 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S504 + 40U]), _S497);
        accLo_0[5U] = _S515;
        float _S516 = fma(pj_22, attnHi_0(vw_181), _S498);
        accHi_0[5U] = _S516;
        uint vw_182 = (*(&kernelContext_0)->vs_0)[_S504 + 48U];
        float _S517 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S504 + 48U]), _S499);
        accLo_0[6U] = _S517;
        float _S518 = fma(pj_22, attnHi_0(vw_182), _S500);
        accHi_0[6U] = _S518;
        uint vw_183 = (*(&kernelContext_0)->vs_0)[_S504 + 56U];
        float _S519 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S504 + 56U]), _S501);
        accLo_0[7U] = _S519;
        float _S520 = fma(pj_22, attnHi_0(vw_183), _S502);
        accHi_0[7U] = _S520;
        float pj_23 = (*(&kernelContext_0)->ps_0)[_S77];
        float _S521 = _S503 + (*(&kernelContext_0)->ps_0)[_S77];
        uint _S522 = 1495U + kk_0;
        uint vw_184 = (*(&kernelContext_0)->vs_0)[_S522];
        float _S523 = fma((*(&kernelContext_0)->ps_0)[_S77], attnLo_0((*(&kernelContext_0)->vs_0)[_S522]), _S505);
        accLo_0[0U] = _S523;
        float _S524 = fma(pj_23, attnHi_0(vw_184), _S506);
        accHi_0[0U] = _S524;
        uint vw_185 = (*(&kernelContext_0)->vs_0)[_S522 + 8U];
        float _S525 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S522 + 8U]), _S507);
        accLo_0[1U] = _S525;
        float _S526 = fma(pj_23, attnHi_0(vw_185), _S508);
        accHi_0[1U] = _S526;
        uint vw_186 = (*(&kernelContext_0)->vs_0)[_S522 + 16U];
        float _S527 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S522 + 16U]), _S509);
        accLo_0[2U] = _S527;
        float _S528 = fma(pj_23, attnHi_0(vw_186), _S510);
        accHi_0[2U] = _S528;
        uint vw_187 = (*(&kernelContext_0)->vs_0)[_S522 + 24U];
        float _S529 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S522 + 24U]), _S511);
        accLo_0[3U] = _S529;
        float _S530 = fma(pj_23, attnHi_0(vw_187), _S512);
        accHi_0[3U] = _S530;
        uint vw_188 = (*(&kernelContext_0)->vs_0)[_S522 + 32U];
        float _S531 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S522 + 32U]), _S513);
        accLo_0[4U] = _S531;
        float _S532 = fma(pj_23, attnHi_0(vw_188), _S514);
        accHi_0[4U] = _S532;
        uint vw_189 = (*(&kernelContext_0)->vs_0)[_S522 + 40U];
        float _S533 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S522 + 40U]), _S515);
        accLo_0[5U] = _S533;
        float _S534 = fma(pj_23, attnHi_0(vw_189), _S516);
        accHi_0[5U] = _S534;
        uint vw_190 = (*(&kernelContext_0)->vs_0)[_S522 + 48U];
        float _S535 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S522 + 48U]), _S517);
        accLo_0[6U] = _S535;
        float _S536 = fma(pj_23, attnHi_0(vw_190), _S518);
        accHi_0[6U] = _S536;
        uint vw_191 = (*(&kernelContext_0)->vs_0)[_S522 + 56U];
        float _S537 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S522 + 56U]), _S519);
        accLo_0[7U] = _S537;
        float _S538 = fma(pj_23, attnHi_0(vw_191), _S520);
        accHi_0[7U] = _S538;
        float pj_24 = (*(&kernelContext_0)->ps_0)[_S78];
        float _S539 = _S521 + (*(&kernelContext_0)->ps_0)[_S78];
        uint _S540 = 1560U + kk_0;
        uint vw_192 = (*(&kernelContext_0)->vs_0)[_S540];
        float _S541 = fma((*(&kernelContext_0)->ps_0)[_S78], attnLo_0((*(&kernelContext_0)->vs_0)[_S540]), _S523);
        accLo_0[0U] = _S541;
        float _S542 = fma(pj_24, attnHi_0(vw_192), _S524);
        accHi_0[0U] = _S542;
        uint vw_193 = (*(&kernelContext_0)->vs_0)[_S540 + 8U];
        float _S543 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S540 + 8U]), _S525);
        accLo_0[1U] = _S543;
        float _S544 = fma(pj_24, attnHi_0(vw_193), _S526);
        accHi_0[1U] = _S544;
        uint vw_194 = (*(&kernelContext_0)->vs_0)[_S540 + 16U];
        float _S545 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S540 + 16U]), _S527);
        accLo_0[2U] = _S545;
        float _S546 = fma(pj_24, attnHi_0(vw_194), _S528);
        accHi_0[2U] = _S546;
        uint vw_195 = (*(&kernelContext_0)->vs_0)[_S540 + 24U];
        float _S547 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S540 + 24U]), _S529);
        accLo_0[3U] = _S547;
        float _S548 = fma(pj_24, attnHi_0(vw_195), _S530);
        accHi_0[3U] = _S548;
        uint vw_196 = (*(&kernelContext_0)->vs_0)[_S540 + 32U];
        float _S549 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S540 + 32U]), _S531);
        accLo_0[4U] = _S549;
        float _S550 = fma(pj_24, attnHi_0(vw_196), _S532);
        accHi_0[4U] = _S550;
        uint vw_197 = (*(&kernelContext_0)->vs_0)[_S540 + 40U];
        float _S551 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S540 + 40U]), _S533);
        accLo_0[5U] = _S551;
        float _S552 = fma(pj_24, attnHi_0(vw_197), _S534);
        accHi_0[5U] = _S552;
        uint vw_198 = (*(&kernelContext_0)->vs_0)[_S540 + 48U];
        float _S553 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S540 + 48U]), _S535);
        accLo_0[6U] = _S553;
        float _S554 = fma(pj_24, attnHi_0(vw_198), _S536);
        accHi_0[6U] = _S554;
        uint vw_199 = (*(&kernelContext_0)->vs_0)[_S540 + 56U];
        float _S555 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S540 + 56U]), _S537);
        accLo_0[7U] = _S555;
        float _S556 = fma(pj_24, attnHi_0(vw_199), _S538);
        accHi_0[7U] = _S556;
        float pj_25 = (*(&kernelContext_0)->ps_0)[_S79];
        float _S557 = _S539 + (*(&kernelContext_0)->ps_0)[_S79];
        uint _S558 = 1625U + kk_0;
        uint vw_200 = (*(&kernelContext_0)->vs_0)[_S558];
        float _S559 = fma((*(&kernelContext_0)->ps_0)[_S79], attnLo_0((*(&kernelContext_0)->vs_0)[_S558]), _S541);
        accLo_0[0U] = _S559;
        float _S560 = fma(pj_25, attnHi_0(vw_200), _S542);
        accHi_0[0U] = _S560;
        uint vw_201 = (*(&kernelContext_0)->vs_0)[_S558 + 8U];
        float _S561 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S558 + 8U]), _S543);
        accLo_0[1U] = _S561;
        float _S562 = fma(pj_25, attnHi_0(vw_201), _S544);
        accHi_0[1U] = _S562;
        uint vw_202 = (*(&kernelContext_0)->vs_0)[_S558 + 16U];
        float _S563 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S558 + 16U]), _S545);
        accLo_0[2U] = _S563;
        float _S564 = fma(pj_25, attnHi_0(vw_202), _S546);
        accHi_0[2U] = _S564;
        uint vw_203 = (*(&kernelContext_0)->vs_0)[_S558 + 24U];
        float _S565 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S558 + 24U]), _S547);
        accLo_0[3U] = _S565;
        float _S566 = fma(pj_25, attnHi_0(vw_203), _S548);
        accHi_0[3U] = _S566;
        uint vw_204 = (*(&kernelContext_0)->vs_0)[_S558 + 32U];
        float _S567 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S558 + 32U]), _S549);
        accLo_0[4U] = _S567;
        float _S568 = fma(pj_25, attnHi_0(vw_204), _S550);
        accHi_0[4U] = _S568;
        uint vw_205 = (*(&kernelContext_0)->vs_0)[_S558 + 40U];
        float _S569 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S558 + 40U]), _S551);
        accLo_0[5U] = _S569;
        float _S570 = fma(pj_25, attnHi_0(vw_205), _S552);
        accHi_0[5U] = _S570;
        uint vw_206 = (*(&kernelContext_0)->vs_0)[_S558 + 48U];
        float _S571 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S558 + 48U]), _S553);
        accLo_0[6U] = _S571;
        float _S572 = fma(pj_25, attnHi_0(vw_206), _S554);
        accHi_0[6U] = _S572;
        uint vw_207 = (*(&kernelContext_0)->vs_0)[_S558 + 56U];
        float _S573 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S558 + 56U]), _S555);
        accLo_0[7U] = _S573;
        float _S574 = fma(pj_25, attnHi_0(vw_207), _S556);
        accHi_0[7U] = _S574;
        float pj_26 = (*(&kernelContext_0)->ps_0)[_S80];
        float _S575 = _S557 + (*(&kernelContext_0)->ps_0)[_S80];
        uint _S576 = 1690U + kk_0;
        uint vw_208 = (*(&kernelContext_0)->vs_0)[_S576];
        float _S577 = fma((*(&kernelContext_0)->ps_0)[_S80], attnLo_0((*(&kernelContext_0)->vs_0)[_S576]), _S559);
        accLo_0[0U] = _S577;
        float _S578 = fma(pj_26, attnHi_0(vw_208), _S560);
        accHi_0[0U] = _S578;
        uint vw_209 = (*(&kernelContext_0)->vs_0)[_S576 + 8U];
        float _S579 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S576 + 8U]), _S561);
        accLo_0[1U] = _S579;
        float _S580 = fma(pj_26, attnHi_0(vw_209), _S562);
        accHi_0[1U] = _S580;
        uint vw_210 = (*(&kernelContext_0)->vs_0)[_S576 + 16U];
        float _S581 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S576 + 16U]), _S563);
        accLo_0[2U] = _S581;
        float _S582 = fma(pj_26, attnHi_0(vw_210), _S564);
        accHi_0[2U] = _S582;
        uint vw_211 = (*(&kernelContext_0)->vs_0)[_S576 + 24U];
        float _S583 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S576 + 24U]), _S565);
        accLo_0[3U] = _S583;
        float _S584 = fma(pj_26, attnHi_0(vw_211), _S566);
        accHi_0[3U] = _S584;
        uint vw_212 = (*(&kernelContext_0)->vs_0)[_S576 + 32U];
        float _S585 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S576 + 32U]), _S567);
        accLo_0[4U] = _S585;
        float _S586 = fma(pj_26, attnHi_0(vw_212), _S568);
        accHi_0[4U] = _S586;
        uint vw_213 = (*(&kernelContext_0)->vs_0)[_S576 + 40U];
        float _S587 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S576 + 40U]), _S569);
        accLo_0[5U] = _S587;
        float _S588 = fma(pj_26, attnHi_0(vw_213), _S570);
        accHi_0[5U] = _S588;
        uint vw_214 = (*(&kernelContext_0)->vs_0)[_S576 + 48U];
        float _S589 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S576 + 48U]), _S571);
        accLo_0[6U] = _S589;
        float _S590 = fma(pj_26, attnHi_0(vw_214), _S572);
        accHi_0[6U] = _S590;
        uint vw_215 = (*(&kernelContext_0)->vs_0)[_S576 + 56U];
        float _S591 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S576 + 56U]), _S573);
        accLo_0[7U] = _S591;
        float _S592 = fma(pj_26, attnHi_0(vw_215), _S574);
        accHi_0[7U] = _S592;
        float pj_27 = (*(&kernelContext_0)->ps_0)[_S81];
        float _S593 = _S575 + (*(&kernelContext_0)->ps_0)[_S81];
        uint _S594 = 1755U + kk_0;
        uint vw_216 = (*(&kernelContext_0)->vs_0)[_S594];
        float _S595 = fma((*(&kernelContext_0)->ps_0)[_S81], attnLo_0((*(&kernelContext_0)->vs_0)[_S594]), _S577);
        accLo_0[0U] = _S595;
        float _S596 = fma(pj_27, attnHi_0(vw_216), _S578);
        accHi_0[0U] = _S596;
        uint vw_217 = (*(&kernelContext_0)->vs_0)[_S594 + 8U];
        float _S597 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S594 + 8U]), _S579);
        accLo_0[1U] = _S597;
        float _S598 = fma(pj_27, attnHi_0(vw_217), _S580);
        accHi_0[1U] = _S598;
        uint vw_218 = (*(&kernelContext_0)->vs_0)[_S594 + 16U];
        float _S599 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S594 + 16U]), _S581);
        accLo_0[2U] = _S599;
        float _S600 = fma(pj_27, attnHi_0(vw_218), _S582);
        accHi_0[2U] = _S600;
        uint vw_219 = (*(&kernelContext_0)->vs_0)[_S594 + 24U];
        float _S601 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S594 + 24U]), _S583);
        accLo_0[3U] = _S601;
        float _S602 = fma(pj_27, attnHi_0(vw_219), _S584);
        accHi_0[3U] = _S602;
        uint vw_220 = (*(&kernelContext_0)->vs_0)[_S594 + 32U];
        float _S603 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S594 + 32U]), _S585);
        accLo_0[4U] = _S603;
        float _S604 = fma(pj_27, attnHi_0(vw_220), _S586);
        accHi_0[4U] = _S604;
        uint vw_221 = (*(&kernelContext_0)->vs_0)[_S594 + 40U];
        float _S605 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S594 + 40U]), _S587);
        accLo_0[5U] = _S605;
        float _S606 = fma(pj_27, attnHi_0(vw_221), _S588);
        accHi_0[5U] = _S606;
        uint vw_222 = (*(&kernelContext_0)->vs_0)[_S594 + 48U];
        float _S607 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S594 + 48U]), _S589);
        accLo_0[6U] = _S607;
        float _S608 = fma(pj_27, attnHi_0(vw_222), _S590);
        accHi_0[6U] = _S608;
        uint vw_223 = (*(&kernelContext_0)->vs_0)[_S594 + 56U];
        float _S609 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S594 + 56U]), _S591);
        accLo_0[7U] = _S609;
        float _S610 = fma(pj_27, attnHi_0(vw_223), _S592);
        accHi_0[7U] = _S610;
        float pj_28 = (*(&kernelContext_0)->ps_0)[_S82];
        float _S611 = _S593 + (*(&kernelContext_0)->ps_0)[_S82];
        uint _S612 = 1820U + kk_0;
        uint vw_224 = (*(&kernelContext_0)->vs_0)[_S612];
        float _S613 = fma((*(&kernelContext_0)->ps_0)[_S82], attnLo_0((*(&kernelContext_0)->vs_0)[_S612]), _S595);
        accLo_0[0U] = _S613;
        float _S614 = fma(pj_28, attnHi_0(vw_224), _S596);
        accHi_0[0U] = _S614;
        uint vw_225 = (*(&kernelContext_0)->vs_0)[_S612 + 8U];
        float _S615 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S612 + 8U]), _S597);
        accLo_0[1U] = _S615;
        float _S616 = fma(pj_28, attnHi_0(vw_225), _S598);
        accHi_0[1U] = _S616;
        uint vw_226 = (*(&kernelContext_0)->vs_0)[_S612 + 16U];
        float _S617 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S612 + 16U]), _S599);
        accLo_0[2U] = _S617;
        float _S618 = fma(pj_28, attnHi_0(vw_226), _S600);
        accHi_0[2U] = _S618;
        uint vw_227 = (*(&kernelContext_0)->vs_0)[_S612 + 24U];
        float _S619 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S612 + 24U]), _S601);
        accLo_0[3U] = _S619;
        float _S620 = fma(pj_28, attnHi_0(vw_227), _S602);
        accHi_0[3U] = _S620;
        uint vw_228 = (*(&kernelContext_0)->vs_0)[_S612 + 32U];
        float _S621 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S612 + 32U]), _S603);
        accLo_0[4U] = _S621;
        float _S622 = fma(pj_28, attnHi_0(vw_228), _S604);
        accHi_0[4U] = _S622;
        uint vw_229 = (*(&kernelContext_0)->vs_0)[_S612 + 40U];
        float _S623 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S612 + 40U]), _S605);
        accLo_0[5U] = _S623;
        float _S624 = fma(pj_28, attnHi_0(vw_229), _S606);
        accHi_0[5U] = _S624;
        uint vw_230 = (*(&kernelContext_0)->vs_0)[_S612 + 48U];
        float _S625 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S612 + 48U]), _S607);
        accLo_0[6U] = _S625;
        float _S626 = fma(pj_28, attnHi_0(vw_230), _S608);
        accHi_0[6U] = _S626;
        uint vw_231 = (*(&kernelContext_0)->vs_0)[_S612 + 56U];
        float _S627 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S612 + 56U]), _S609);
        accLo_0[7U] = _S627;
        float _S628 = fma(pj_28, attnHi_0(vw_231), _S610);
        accHi_0[7U] = _S628;
        float pj_29 = (*(&kernelContext_0)->ps_0)[_S83];
        float _S629 = _S611 + (*(&kernelContext_0)->ps_0)[_S83];
        uint _S630 = 1885U + kk_0;
        uint vw_232 = (*(&kernelContext_0)->vs_0)[_S630];
        float _S631 = fma((*(&kernelContext_0)->ps_0)[_S83], attnLo_0((*(&kernelContext_0)->vs_0)[_S630]), _S613);
        accLo_0[0U] = _S631;
        float _S632 = fma(pj_29, attnHi_0(vw_232), _S614);
        accHi_0[0U] = _S632;
        uint vw_233 = (*(&kernelContext_0)->vs_0)[_S630 + 8U];
        float _S633 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S630 + 8U]), _S615);
        accLo_0[1U] = _S633;
        float _S634 = fma(pj_29, attnHi_0(vw_233), _S616);
        accHi_0[1U] = _S634;
        uint vw_234 = (*(&kernelContext_0)->vs_0)[_S630 + 16U];
        float _S635 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S630 + 16U]), _S617);
        accLo_0[2U] = _S635;
        float _S636 = fma(pj_29, attnHi_0(vw_234), _S618);
        accHi_0[2U] = _S636;
        uint vw_235 = (*(&kernelContext_0)->vs_0)[_S630 + 24U];
        float _S637 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S630 + 24U]), _S619);
        accLo_0[3U] = _S637;
        float _S638 = fma(pj_29, attnHi_0(vw_235), _S620);
        accHi_0[3U] = _S638;
        uint vw_236 = (*(&kernelContext_0)->vs_0)[_S630 + 32U];
        float _S639 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S630 + 32U]), _S621);
        accLo_0[4U] = _S639;
        float _S640 = fma(pj_29, attnHi_0(vw_236), _S622);
        accHi_0[4U] = _S640;
        uint vw_237 = (*(&kernelContext_0)->vs_0)[_S630 + 40U];
        float _S641 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S630 + 40U]), _S623);
        accLo_0[5U] = _S641;
        float _S642 = fma(pj_29, attnHi_0(vw_237), _S624);
        accHi_0[5U] = _S642;
        uint vw_238 = (*(&kernelContext_0)->vs_0)[_S630 + 48U];
        float _S643 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S630 + 48U]), _S625);
        accLo_0[6U] = _S643;
        float _S644 = fma(pj_29, attnHi_0(vw_238), _S626);
        accHi_0[6U] = _S644;
        uint vw_239 = (*(&kernelContext_0)->vs_0)[_S630 + 56U];
        float _S645 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S630 + 56U]), _S627);
        accLo_0[7U] = _S645;
        float _S646 = fma(pj_29, attnHi_0(vw_239), _S628);
        accHi_0[7U] = _S646;
        float pj_30 = (*(&kernelContext_0)->ps_0)[_S84];
        float _S647 = _S629 + (*(&kernelContext_0)->ps_0)[_S84];
        uint _S648 = 1950U + kk_0;
        uint vw_240 = (*(&kernelContext_0)->vs_0)[_S648];
        float _S649 = fma((*(&kernelContext_0)->ps_0)[_S84], attnLo_0((*(&kernelContext_0)->vs_0)[_S648]), _S631);
        accLo_0[0U] = _S649;
        float _S650 = fma(pj_30, attnHi_0(vw_240), _S632);
        accHi_0[0U] = _S650;
        uint vw_241 = (*(&kernelContext_0)->vs_0)[_S648 + 8U];
        float _S651 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S648 + 8U]), _S633);
        accLo_0[1U] = _S651;
        float _S652 = fma(pj_30, attnHi_0(vw_241), _S634);
        accHi_0[1U] = _S652;
        uint vw_242 = (*(&kernelContext_0)->vs_0)[_S648 + 16U];
        float _S653 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S648 + 16U]), _S635);
        accLo_0[2U] = _S653;
        float _S654 = fma(pj_30, attnHi_0(vw_242), _S636);
        accHi_0[2U] = _S654;
        uint vw_243 = (*(&kernelContext_0)->vs_0)[_S648 + 24U];
        float _S655 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S648 + 24U]), _S637);
        accLo_0[3U] = _S655;
        float _S656 = fma(pj_30, attnHi_0(vw_243), _S638);
        accHi_0[3U] = _S656;
        uint vw_244 = (*(&kernelContext_0)->vs_0)[_S648 + 32U];
        float _S657 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S648 + 32U]), _S639);
        accLo_0[4U] = _S657;
        float _S658 = fma(pj_30, attnHi_0(vw_244), _S640);
        accHi_0[4U] = _S658;
        uint vw_245 = (*(&kernelContext_0)->vs_0)[_S648 + 40U];
        float _S659 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S648 + 40U]), _S641);
        accLo_0[5U] = _S659;
        float _S660 = fma(pj_30, attnHi_0(vw_245), _S642);
        accHi_0[5U] = _S660;
        uint vw_246 = (*(&kernelContext_0)->vs_0)[_S648 + 48U];
        float _S661 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S648 + 48U]), _S643);
        accLo_0[6U] = _S661;
        float _S662 = fma(pj_30, attnHi_0(vw_246), _S644);
        accHi_0[6U] = _S662;
        uint vw_247 = (*(&kernelContext_0)->vs_0)[_S648 + 56U];
        float _S663 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S648 + 56U]), _S645);
        accLo_0[7U] = _S663;
        float _S664 = fma(pj_30, attnHi_0(vw_247), _S646);
        accHi_0[7U] = _S664;
        float pj_31 = (*(&kernelContext_0)->ps_0)[_S85];
        float l_1 = _S647 + (*(&kernelContext_0)->ps_0)[_S85];
        uint _S665 = 2015U + kk_0;
        uint vw_248 = (*(&kernelContext_0)->vs_0)[_S665];
        accLo_0[0U] = fma((*(&kernelContext_0)->ps_0)[_S85], attnLo_0((*(&kernelContext_0)->vs_0)[_S665]), _S649);
        accHi_0[0U] = fma(pj_31, attnHi_0(vw_248), _S650);
        uint vw_249 = (*(&kernelContext_0)->vs_0)[_S665 + 8U];
        accLo_0[1U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S665 + 8U]), _S651);
        accHi_0[1U] = fma(pj_31, attnHi_0(vw_249), _S652);
        uint vw_250 = (*(&kernelContext_0)->vs_0)[_S665 + 16U];
        accLo_0[2U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S665 + 16U]), _S653);
        accHi_0[2U] = fma(pj_31, attnHi_0(vw_250), _S654);
        uint vw_251 = (*(&kernelContext_0)->vs_0)[_S665 + 24U];
        accLo_0[3U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S665 + 24U]), _S655);
        accHi_0[3U] = fma(pj_31, attnHi_0(vw_251), _S656);
        uint vw_252 = (*(&kernelContext_0)->vs_0)[_S665 + 32U];
        accLo_0[4U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S665 + 32U]), _S657);
        accHi_0[4U] = fma(pj_31, attnHi_0(vw_252), _S658);
        uint vw_253 = (*(&kernelContext_0)->vs_0)[_S665 + 40U];
        accLo_0[5U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S665 + 40U]), _S659);
        accHi_0[5U] = fma(pj_31, attnHi_0(vw_253), _S660);
        uint vw_254 = (*(&kernelContext_0)->vs_0)[_S665 + 48U];
        accLo_0[6U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S665 + 48U]), _S661);
        accHi_0[6U] = fma(pj_31, attnHi_0(vw_254), _S662);
        uint vw_255 = (*(&kernelContext_0)->vs_0)[_S665 + 56U];
        accLo_0[7U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S665 + 56U]), _S663);
        accHi_0[7U] = fma(pj_31, attnHi_0(vw_255), _S664);
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
        uint _S666 = attnAccBase_0((&kernelContext_0)->p_0->splits_1, rows_1) + pr_30 * (&kernelContext_0)->p_0->headDim_0;
        if(kk_0 == 0U)
        {
            uint _S667 = 2U * pr_30;
            *((&kernelContext_0)->partials_0+_S667) = m_0;
            *((&kernelContext_0)->partials_0+(_S667 + 1U)) = l_0;
        }
        if(kk_0 < _S20)
        {
            uint _S668 = _S666 + 2U * kk_0;
            float device* _S669 = (&kernelContext_0)->partials_0+_S668;
            bool _S670 = l_0 == 0.0;
            if(_S670)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[0U];
            }
            *_S669 = m_0;
            float device* _S671 = (&kernelContext_0)->partials_0+(_S668 + 1U);
            if(_S670)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[0U];
            }
            *_S671 = m_0;
        }
        uint pair_0 = kk_0 + 8U;
        if(pair_0 < _S20)
        {
            uint _S672 = _S666 + 2U * pair_0;
            float device* _S673 = (&kernelContext_0)->partials_0+_S672;
            bool _S674 = l_0 == 0.0;
            if(_S674)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[1U];
            }
            *_S673 = m_0;
            float device* _S675 = (&kernelContext_0)->partials_0+(_S672 + 1U);
            if(_S674)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[1U];
            }
            *_S675 = m_0;
        }
        uint pair_1 = kk_0 + 16U;
        if(pair_1 < _S20)
        {
            uint _S676 = _S666 + 2U * pair_1;
            float device* _S677 = (&kernelContext_0)->partials_0+_S676;
            bool _S678 = l_0 == 0.0;
            if(_S678)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[2U];
            }
            *_S677 = m_0;
            float device* _S679 = (&kernelContext_0)->partials_0+(_S676 + 1U);
            if(_S678)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[2U];
            }
            *_S679 = m_0;
        }
        uint pair_2 = kk_0 + 24U;
        if(pair_2 < _S20)
        {
            uint _S680 = _S666 + 2U * pair_2;
            float device* _S681 = (&kernelContext_0)->partials_0+_S680;
            bool _S682 = l_0 == 0.0;
            if(_S682)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[3U];
            }
            *_S681 = m_0;
            float device* _S683 = (&kernelContext_0)->partials_0+(_S680 + 1U);
            if(_S682)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[3U];
            }
            *_S683 = m_0;
        }
        uint pair_3 = kk_0 + 32U;
        if(pair_3 < _S20)
        {
            uint _S684 = _S666 + 2U * pair_3;
            float device* _S685 = (&kernelContext_0)->partials_0+_S684;
            bool _S686 = l_0 == 0.0;
            if(_S686)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[4U];
            }
            *_S685 = m_0;
            float device* _S687 = (&kernelContext_0)->partials_0+(_S684 + 1U);
            if(_S686)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[4U];
            }
            *_S687 = m_0;
        }
        uint pair_4 = kk_0 + 40U;
        if(pair_4 < _S20)
        {
            uint _S688 = _S666 + 2U * pair_4;
            float device* _S689 = (&kernelContext_0)->partials_0+_S688;
            bool _S690 = l_0 == 0.0;
            if(_S690)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[5U];
            }
            *_S689 = m_0;
            float device* _S691 = (&kernelContext_0)->partials_0+(_S688 + 1U);
            if(_S690)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[5U];
            }
            *_S691 = m_0;
        }
        uint pair_5 = kk_0 + 48U;
        if(pair_5 < _S20)
        {
            uint _S692 = _S666 + 2U * pair_5;
            float device* _S693 = (&kernelContext_0)->partials_0+_S692;
            bool _S694 = l_0 == 0.0;
            if(_S694)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[6U];
            }
            *_S693 = m_0;
            float device* _S695 = (&kernelContext_0)->partials_0+(_S692 + 1U);
            if(_S694)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[6U];
            }
            *_S695 = m_0;
        }
        uint pair_6 = kk_0 + 56U;
        if(pair_6 < _S20)
        {
            uint _S696 = _S666 + 2U * pair_6;
            float device* _S697 = (&kernelContext_0)->partials_0+_S696;
            bool _S698 = l_0 == 0.0;
            if(_S698)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accLo_0[7U];
            }
            *_S697 = m_0;
            float device* _S699 = (&kernelContext_0)->partials_0+(_S696 + 1U);
            if(_S698)
            {
                m_0 = 0.0;
            }
            else
            {
                m_0 = accHi_0[7U];
            }
            *_S699 = m_0;
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
        float device* _S700 = (&kernelContext_0)->lse_0+row_1;
        if(l_0 == 0.0)
        {
            m_0 = _S31;
        }
        else
        {
            m_0 = m_0 + log(l_0);
        }
        *_S700 = m_0;
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
                bool _S701 = l_0 == 0.0;
                if(_S701)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[0U] / l_0;
                }
                if(_S701)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[0U] / l_0;
                }
                uint _S702 = 2U * kk_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S702)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S702 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_7 = kk_0 + 8U;
                if(pair_7 >= _S20)
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
                    s_0 = accLo_0[1U] / l_0;
                }
                if(_S703)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[1U] / l_0;
                }
                uint _S704 = 2U * pair_7;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S704)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S704 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_8 = kk_0 + 16U;
                if(pair_8 >= _S20)
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
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[2U] / l_0;
                }
                uint _S706 = 2U * pair_8;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S706)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S706 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_9 = kk_0 + 24U;
                if(pair_9 >= _S20)
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
                    s_0 = accLo_0[3U] / l_0;
                }
                if(_S707)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[3U] / l_0;
                }
                uint _S708 = 2U * pair_9;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S708)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S708 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_10 = kk_0 + 32U;
                if(pair_10 >= _S20)
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
                    s_0 = accLo_0[4U] / l_0;
                }
                if(_S709)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[4U] / l_0;
                }
                uint _S710 = 2U * pair_10;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S710)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S710 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_11 = kk_0 + 40U;
                if(pair_11 >= _S20)
                {
                    break;
                }
                bool _S711 = l_0 == 0.0;
                if(_S711)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[5U] / l_0;
                }
                if(_S711)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[5U] / l_0;
                }
                uint _S712 = 2U * pair_11;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S712)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S712 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_12 = kk_0 + 48U;
                if(pair_12 >= _S20)
                {
                    break;
                }
                bool _S713 = l_0 == 0.0;
                if(_S713)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[6U] / l_0;
                }
                if(_S713)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[6U] / l_0;
                }
                uint _S714 = 2U * pair_12;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S714)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S714 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_13 = kk_0 + 56U;
                if(pair_13 >= _S20)
                {
                    break;
                }
                bool _S715 = l_0 == 0.0;
                if(_S715)
                {
                    s_0 = 0.0;
                }
                else
                {
                    s_0 = accLo_0[7U] / l_0;
                }
                if(_S715)
                {
                    hi_0 = 0.0;
                }
                else
                {
                    hi_0 = accHi_0[7U] / l_0;
                }
                uint _S716 = 2U * pair_13;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S716)) = s_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S716 + 1U)) = hi_0;
                break;
            }
            break;
        }
        break;
    }
    return;
}

