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
    uint device* q_0;
    uint device* table_0;
    uint device* k_0;
    uint device* v_0;
    float device* partialML_0;
    float device* partialAcc_0;
    float device* lse_0;
    float device* o_0;
    array<uint, int(1040)> threadgroup* qs_0;
    array<uint, int(2080)> threadgroup* ks_0;
    array<uint, int(2080)> threadgroup* vs_0;
    array<float, int(512)> threadgroup* ss_0;
    array<float, int(512)> threadgroup* ps_0;
};

[[kernel]] void attn_flash_128_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(8)]], uint device* q_1 [[buffer(0)]], uint device* table_1 [[buffer(6)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partialML_1 [[buffer(4)]], float device* partialAcc_1 [[buffer(5)]], float device* lse_1 [[buffer(7)]], float device* o_1 [[buffer(3)]])
{
    uint _S1;
    uint _S2;
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
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->q_0 = q_1;
    (&kernelContext_0)->table_0 = table_1;
    (&kernelContext_0)->k_0 = k_1;
    (&kernelContext_0)->v_0 = v_1;
    (&kernelContext_0)->partialML_0 = partialML_1;
    (&kernelContext_0)->partialAcc_0 = partialAcc_1;
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
    uint _S13 = kvHead_0 * hpk_0;
    uint qi_0;
    if(live_0)
    {
        uint _S14 = r_0 / (&kernelContext_0)->p_0->queries_0;
        qi_0 = _S14;
    }
    else
    {
        qi_0 = 0U;
    }
    uint head_0 = _S13 + qi_0;
    if(live_0)
    {
        uint _S15 = r_0 % (&kernelContext_0)->p_0->queries_0;
        qi_0 = _S15;
    }
    else
    {
        qi_0 = 0U;
    }
    bool ok_0;
    uint w_2;
    uint _S16 = (&kernelContext_0)->p_0->headDim_0 / 2U;
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
            ok_0 = pr_0 < _S16;
        }
        else
        {
            ok_0 = false;
        }
        if(ok_0)
        {
            uint _S17 = row_0 / (&kernelContext_0)->p_0->queries_0;
            uint h_0 = _S13 + _S17;
            uint i_0 = row_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((h_0 * (&kernelContext_0)->p_0->queries_0 + i_0) * _S16 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_0 / 64U * 65U + pr_0] = w_2;
        e_0 = e_0 + 128U;
    }
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S18 = min((&kernelContext_0)->p_0->keys_0, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        uint rLast_0 = min(r0_0 + 16U, rowsKv_0) - 1U;
        uint _S19 = r0_0 / (&kernelContext_0)->p_0->queries_0;
        uint _S20 = rLast_0 / (&kernelContext_0)->p_0->queries_0;
        if(_S19 != _S20)
        {
            w_2 = (&kernelContext_0)->p_0->queries_0 - 1U;
        }
        else
        {
            uint _S21 = rLast_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = _S21;
        }
        w_2 = min(_S18, w_2 + offset_0 + 1U);
    }
    else
    {
        w_2 = _S18;
    }
    if(((&kernelContext_0)->p_0->causal_0) != 0U)
    {
        e_0 = qi_0 + offset_0 + 1U;
    }
    else
    {
        e_0 = (&kernelContext_0)->p_0->keys_0;
    }
    float _S22 = dg_negative_infinity_0();
    thread array<float, int(8)> accLo_0;
    thread array<float, int(8)> accHi_0;
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
    float m_0 = _S22;
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
        uint e_1 = t_0;
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
                ok_0 = pr_1 < _S16;
            }
            else
            {
                ok_0 = false;
            }
            if(ok_0)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    at_0 = (kvHead_0 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * _S16 + pr_1;
                }
                else
                {
                    uint _S23 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S24 = (*((&kernelContext_0)->table_0+_S23) * p_1->kvHeads_0 + kvHead_0) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S25 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
                    at_0 = (_S24 + _S25) * _S16 + pr_1;
                }
            }
            else
            {
                at_0 = 0U;
            }
            uint _S26 = j_0 * 65U + pr_1;
            if(ok_0)
            {
                p0_1 = *((&kernelContext_0)->k_0+at_0);
            }
            else
            {
                p0_1 = 0U;
            }
            (*(&kernelContext_0)->ks_0)[_S26] = p0_1;
            if(ok_0)
            {
                p0_0 = *((&kernelContext_0)->v_0+at_0);
            }
            else
            {
                p0_0 = 0U;
            }
            (*(&kernelContext_0)->vs_0)[_S26] = p0_0;
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
                    float s_0 = 0.0;
                    for(;;)
                    {
                        if(at_0 < 64U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S27 = lr_0 * 65U;
                        uint _S28 = kk_0 * 65U;
                        uint pr_2 = at_0 + 1U;
                        uint pr_3 = at_0 + 2U;
                        uint pr_4 = at_0 + 3U;
                        uint pr_5 = at_0 + 4U;
                        uint pr_6 = at_0 + 5U;
                        uint pr_7 = at_0 + 6U;
                        uint pr_8 = at_0 + 7U;
                        float s_1 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S27 + pr_8]), attnHi_0((*(&kernelContext_0)->ks_0)[_S28 + pr_8]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S27 + pr_8]), attnLo_0((*(&kernelContext_0)->ks_0)[_S28 + pr_8]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S27 + pr_7]), attnHi_0((*(&kernelContext_0)->ks_0)[_S28 + pr_7]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S27 + pr_7]), attnLo_0((*(&kernelContext_0)->ks_0)[_S28 + pr_7]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S27 + pr_6]), attnHi_0((*(&kernelContext_0)->ks_0)[_S28 + pr_6]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S27 + pr_6]), attnLo_0((*(&kernelContext_0)->ks_0)[_S28 + pr_6]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S27 + pr_5]), attnHi_0((*(&kernelContext_0)->ks_0)[_S28 + pr_5]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S27 + pr_5]), attnLo_0((*(&kernelContext_0)->ks_0)[_S28 + pr_5]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S27 + pr_4]), attnHi_0((*(&kernelContext_0)->ks_0)[_S28 + pr_4]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S27 + pr_4]), attnLo_0((*(&kernelContext_0)->ks_0)[_S28 + pr_4]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S27 + pr_3]), attnHi_0((*(&kernelContext_0)->ks_0)[_S28 + pr_3]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S27 + pr_3]), attnLo_0((*(&kernelContext_0)->ks_0)[_S28 + pr_3]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S27 + pr_2]), attnHi_0((*(&kernelContext_0)->ks_0)[_S28 + pr_2]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S27 + pr_2]), attnLo_0((*(&kernelContext_0)->ks_0)[_S28 + pr_2]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S27 + at_0]), attnHi_0((*(&kernelContext_0)->ks_0)[_S28 + at_0]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S27 + at_0]), attnLo_0((*(&kernelContext_0)->ks_0)[_S28 + at_0]), s_0))))))))))))))));
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
                        visible_0 = key_1 < e_0;
                    }
                    else
                    {
                        visible_0 = false;
                    }
                    uint _S29 = lr_0 * 32U;
                    _S1 = _S29;
                    uint _S30 = _S29 + kk_0;
                    _S9 = _S30;
                    _S2 = _S30;
                    float _S31;
                    if(visible_0)
                    {
                        _S31 = s_0 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S31 = _S22;
                    }
                    (*(&kernelContext_0)->ss_0)[_S30] = _S31;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_1 = kk_0 + 8U;
                    _S3 = j_1;
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
                        uint _S32 = lr_0 * 65U;
                        uint _S33 = j_1 * 65U;
                        uint pr_9 = p0_1 + 1U;
                        uint pr_10 = p0_1 + 2U;
                        uint pr_11 = p0_1 + 3U;
                        uint pr_12 = p0_1 + 4U;
                        uint pr_13 = p0_1 + 5U;
                        uint pr_14 = p0_1 + 6U;
                        uint pr_15 = p0_1 + 7U;
                        float s_3 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S32 + pr_15]), attnHi_0((*(&kernelContext_0)->ks_0)[_S33 + pr_15]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S32 + pr_15]), attnLo_0((*(&kernelContext_0)->ks_0)[_S33 + pr_15]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S32 + pr_14]), attnHi_0((*(&kernelContext_0)->ks_0)[_S33 + pr_14]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S32 + pr_14]), attnLo_0((*(&kernelContext_0)->ks_0)[_S33 + pr_14]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S32 + pr_13]), attnHi_0((*(&kernelContext_0)->ks_0)[_S33 + pr_13]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S32 + pr_13]), attnLo_0((*(&kernelContext_0)->ks_0)[_S33 + pr_13]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S32 + pr_12]), attnHi_0((*(&kernelContext_0)->ks_0)[_S33 + pr_12]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S32 + pr_12]), attnLo_0((*(&kernelContext_0)->ks_0)[_S33 + pr_12]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S32 + pr_11]), attnHi_0((*(&kernelContext_0)->ks_0)[_S33 + pr_11]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S32 + pr_11]), attnLo_0((*(&kernelContext_0)->ks_0)[_S33 + pr_11]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S32 + pr_10]), attnHi_0((*(&kernelContext_0)->ks_0)[_S33 + pr_10]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S32 + pr_10]), attnLo_0((*(&kernelContext_0)->ks_0)[_S33 + pr_10]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S32 + pr_9]), attnHi_0((*(&kernelContext_0)->ks_0)[_S33 + pr_9]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S32 + pr_9]), attnLo_0((*(&kernelContext_0)->ks_0)[_S33 + pr_9]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S32 + p0_1]), attnHi_0((*(&kernelContext_0)->ks_0)[_S33 + p0_1]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S32 + p0_1]), attnLo_0((*(&kernelContext_0)->ks_0)[_S33 + p0_1]), s_2))))))))))))))));
                        p0_1 = p0_1 + 8U;
                        s_2 = s_3;
                    }
                    bool _S34;
                    if(live_0)
                    {
                        _S34 = key_2 < w_2;
                    }
                    else
                    {
                        _S34 = false;
                    }
                    bool visible_1;
                    if(_S34)
                    {
                        visible_1 = key_2 < e_0;
                    }
                    else
                    {
                        visible_1 = false;
                    }
                    uint _S35 = _S1 + j_1;
                    _S10 = _S35;
                    _S4 = _S35;
                    float _S36;
                    if(visible_1)
                    {
                        _S36 = s_2 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S36 = _S22;
                    }
                    (*(&kernelContext_0)->ss_0)[_S35] = _S36;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_2 = kk_0 + 16U;
                    _S5 = j_2;
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
                        uint _S37 = lr_0 * 65U;
                        uint _S38 = j_2 * 65U;
                        uint pr_16 = p0_0 + 1U;
                        uint pr_17 = p0_0 + 2U;
                        uint pr_18 = p0_0 + 3U;
                        uint pr_19 = p0_0 + 4U;
                        uint pr_20 = p0_0 + 5U;
                        uint pr_21 = p0_0 + 6U;
                        uint pr_22 = p0_0 + 7U;
                        float s_5 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S37 + pr_22]), attnHi_0((*(&kernelContext_0)->ks_0)[_S38 + pr_22]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S37 + pr_22]), attnLo_0((*(&kernelContext_0)->ks_0)[_S38 + pr_22]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S37 + pr_21]), attnHi_0((*(&kernelContext_0)->ks_0)[_S38 + pr_21]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S37 + pr_21]), attnLo_0((*(&kernelContext_0)->ks_0)[_S38 + pr_21]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S37 + pr_20]), attnHi_0((*(&kernelContext_0)->ks_0)[_S38 + pr_20]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S37 + pr_20]), attnLo_0((*(&kernelContext_0)->ks_0)[_S38 + pr_20]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S37 + pr_19]), attnHi_0((*(&kernelContext_0)->ks_0)[_S38 + pr_19]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S37 + pr_19]), attnLo_0((*(&kernelContext_0)->ks_0)[_S38 + pr_19]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S37 + pr_18]), attnHi_0((*(&kernelContext_0)->ks_0)[_S38 + pr_18]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S37 + pr_18]), attnLo_0((*(&kernelContext_0)->ks_0)[_S38 + pr_18]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S37 + pr_17]), attnHi_0((*(&kernelContext_0)->ks_0)[_S38 + pr_17]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S37 + pr_17]), attnLo_0((*(&kernelContext_0)->ks_0)[_S38 + pr_17]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S37 + pr_16]), attnHi_0((*(&kernelContext_0)->ks_0)[_S38 + pr_16]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S37 + pr_16]), attnLo_0((*(&kernelContext_0)->ks_0)[_S38 + pr_16]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S37 + p0_0]), attnHi_0((*(&kernelContext_0)->ks_0)[_S38 + p0_0]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S37 + p0_0]), attnLo_0((*(&kernelContext_0)->ks_0)[_S38 + p0_0]), s_4))))))))))))))));
                        p0_0 = p0_0 + 8U;
                        s_4 = s_5;
                    }
                    bool _S39;
                    if(live_0)
                    {
                        _S39 = key_3 < w_2;
                    }
                    else
                    {
                        _S39 = false;
                    }
                    bool visible_2;
                    if(_S39)
                    {
                        visible_2 = key_3 < e_0;
                    }
                    else
                    {
                        visible_2 = false;
                    }
                    uint _S40 = _S1 + j_2;
                    _S11 = _S40;
                    _S6 = _S40;
                    float _S41;
                    if(visible_2)
                    {
                        _S41 = s_4 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S41 = _S22;
                    }
                    (*(&kernelContext_0)->ss_0)[_S40] = _S41;
                    break;
                }
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    uint j_3 = kk_0 + 24U;
                    _S7 = j_3;
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
                        uint _S42 = lr_0 * 65U;
                        uint _S43 = j_3 * 65U;
                        uint pr_23 = p0_2 + 1U;
                        uint pr_24 = p0_2 + 2U;
                        uint pr_25 = p0_2 + 3U;
                        uint pr_26 = p0_2 + 4U;
                        uint pr_27 = p0_2 + 5U;
                        uint pr_28 = p0_2 + 6U;
                        uint pr_29 = p0_2 + 7U;
                        float s_7 = fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_29]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_29]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_29]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_29]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_28]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_28]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_28]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_28]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_27]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_27]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_27]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_27]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_26]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_26]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_26]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_26]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_25]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_25]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_25]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_25]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_24]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_24]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_24]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_24]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + pr_23]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + pr_23]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + pr_23]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + pr_23]), fma(attnHi_0((*(&kernelContext_0)->qs_0)[_S42 + p0_2]), attnHi_0((*(&kernelContext_0)->ks_0)[_S43 + p0_2]), fma(attnLo_0((*(&kernelContext_0)->qs_0)[_S42 + p0_2]), attnLo_0((*(&kernelContext_0)->ks_0)[_S43 + p0_2]), s_6))))))))))))))));
                        p0_2 = p0_2 + 8U;
                        s_6 = s_7;
                    }
                    bool _S44;
                    if(live_0)
                    {
                        _S44 = key_4 < w_2;
                    }
                    else
                    {
                        _S44 = false;
                    }
                    bool visible_3;
                    if(_S44)
                    {
                        visible_3 = key_4 < e_0;
                    }
                    else
                    {
                        visible_3 = false;
                    }
                    uint _S45 = _S1 + j_3;
                    _S12 = _S45;
                    _S8 = _S45;
                    float _S46;
                    if(visible_3)
                    {
                        _S46 = s_6 * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        _S46 = _S22;
                    }
                    (*(&kernelContext_0)->ss_0)[_S45] = _S46;
                    break;
                }
                break;
            }
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S47 = _S1 + 1U;
        uint _S48 = _S1 + 2U;
        uint _S49 = _S1 + 3U;
        uint _S50 = _S1 + 4U;
        uint _S51 = _S1 + 5U;
        uint _S52 = _S1 + 6U;
        uint _S53 = _S1 + 7U;
        uint _S54 = _S1 + 8U;
        uint _S55 = _S1 + 9U;
        uint _S56 = _S1 + 10U;
        uint _S57 = _S1 + 11U;
        uint _S58 = _S1 + 12U;
        uint _S59 = _S1 + 13U;
        uint _S60 = _S1 + 14U;
        uint _S61 = _S1 + 15U;
        uint _S62 = _S1 + 16U;
        uint _S63 = _S1 + 17U;
        uint _S64 = _S1 + 18U;
        uint _S65 = _S1 + 19U;
        uint _S66 = _S1 + 20U;
        uint _S67 = _S1 + 21U;
        uint _S68 = _S1 + 22U;
        uint _S69 = _S1 + 23U;
        uint _S70 = _S1 + 24U;
        uint _S71 = _S1 + 25U;
        uint _S72 = _S1 + 26U;
        uint _S73 = _S1 + 27U;
        uint _S74 = _S1 + 28U;
        uint _S75 = _S1 + 29U;
        uint _S76 = _S1 + 30U;
        uint _S77 = _S1 + 31U;
        float _S78 = max(m_0, max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(_S22, (*(&kernelContext_0)->ss_0)[_S1]), (*(&kernelContext_0)->ss_0)[_S47]), (*(&kernelContext_0)->ss_0)[_S48]), (*(&kernelContext_0)->ss_0)[_S49]), (*(&kernelContext_0)->ss_0)[_S50]), (*(&kernelContext_0)->ss_0)[_S51]), (*(&kernelContext_0)->ss_0)[_S52]), (*(&kernelContext_0)->ss_0)[_S53]), (*(&kernelContext_0)->ss_0)[_S54]), (*(&kernelContext_0)->ss_0)[_S55]), (*(&kernelContext_0)->ss_0)[_S56]), (*(&kernelContext_0)->ss_0)[_S57]), (*(&kernelContext_0)->ss_0)[_S58]), (*(&kernelContext_0)->ss_0)[_S59]), (*(&kernelContext_0)->ss_0)[_S60]), (*(&kernelContext_0)->ss_0)[_S61]), (*(&kernelContext_0)->ss_0)[_S62]), (*(&kernelContext_0)->ss_0)[_S63]), (*(&kernelContext_0)->ss_0)[_S64]), (*(&kernelContext_0)->ss_0)[_S65]), (*(&kernelContext_0)->ss_0)[_S66]), (*(&kernelContext_0)->ss_0)[_S67]), (*(&kernelContext_0)->ss_0)[_S68]), (*(&kernelContext_0)->ss_0)[_S69]), (*(&kernelContext_0)->ss_0)[_S70]), (*(&kernelContext_0)->ss_0)[_S71]), (*(&kernelContext_0)->ss_0)[_S72]), (*(&kernelContext_0)->ss_0)[_S73]), (*(&kernelContext_0)->ss_0)[_S74]), (*(&kernelContext_0)->ss_0)[_S75]), (*(&kernelContext_0)->ss_0)[_S76]), (*(&kernelContext_0)->ss_0)[_S77]));
        float alpha_0;
        if(m_0 == _S22)
        {
            alpha_0 = 0.0;
        }
        else
        {
            alpha_0 = exp(m_0 - _S78);
        }
        float s_8 = (*(&kernelContext_0)->ss_0)[_S9];
        float _S79;
        if(((*(&kernelContext_0)->ss_0)[_S9]) == _S22)
        {
            _S79 = 0.0;
        }
        else
        {
            _S79 = exp(s_8 - _S78);
        }
        (*(&kernelContext_0)->ps_0)[_S2] = _S79;
        float s_9 = (*(&kernelContext_0)->ss_0)[_S10];
        float _S80;
        if(((*(&kernelContext_0)->ss_0)[_S10]) == _S22)
        {
            _S80 = 0.0;
        }
        else
        {
            _S80 = exp(s_9 - _S78);
        }
        (*(&kernelContext_0)->ps_0)[_S4] = _S80;
        float s_10 = (*(&kernelContext_0)->ss_0)[_S11];
        float _S81;
        if(((*(&kernelContext_0)->ss_0)[_S11]) == _S22)
        {
            _S81 = 0.0;
        }
        else
        {
            _S81 = exp(s_10 - _S78);
        }
        (*(&kernelContext_0)->ps_0)[_S6] = _S81;
        float s_11 = (*(&kernelContext_0)->ss_0)[_S12];
        float _S82;
        if(((*(&kernelContext_0)->ss_0)[_S12]) == _S22)
        {
            _S82 = 0.0;
        }
        else
        {
            _S82 = exp(s_11 - _S78);
        }
        (*(&kernelContext_0)->ps_0)[_S8] = _S82;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S83 = l_0 * alpha_0;
        float _S84 = accLo_0[0U] * alpha_0;
        accLo_0[0U] = _S84;
        float _S85 = accHi_0[0U] * alpha_0;
        accHi_0[0U] = _S85;
        float _S86 = accLo_0[1U] * alpha_0;
        accLo_0[1U] = _S86;
        float _S87 = accHi_0[1U] * alpha_0;
        accHi_0[1U] = _S87;
        float _S88 = accLo_0[2U] * alpha_0;
        accLo_0[2U] = _S88;
        float _S89 = accHi_0[2U] * alpha_0;
        accHi_0[2U] = _S89;
        float _S90 = accLo_0[3U] * alpha_0;
        accLo_0[3U] = _S90;
        float _S91 = accHi_0[3U] * alpha_0;
        accHi_0[3U] = _S91;
        float _S92 = accLo_0[4U] * alpha_0;
        accLo_0[4U] = _S92;
        float _S93 = accHi_0[4U] * alpha_0;
        accHi_0[4U] = _S93;
        float _S94 = accLo_0[5U] * alpha_0;
        accLo_0[5U] = _S94;
        float _S95 = accHi_0[5U] * alpha_0;
        accHi_0[5U] = _S95;
        float _S96 = accLo_0[6U] * alpha_0;
        accLo_0[6U] = _S96;
        float _S97 = accHi_0[6U] * alpha_0;
        accHi_0[6U] = _S97;
        float _S98 = accLo_0[7U] * alpha_0;
        accLo_0[7U] = _S98;
        float _S99 = accHi_0[7U] * alpha_0;
        accHi_0[7U] = _S99;
        float pj_0 = (*(&kernelContext_0)->ps_0)[_S1];
        float _S100 = _S83 + (*(&kernelContext_0)->ps_0)[_S1];
        uint vw_0 = (*(&kernelContext_0)->vs_0)[kk_0];
        float _S101 = fma((*(&kernelContext_0)->ps_0)[_S1], attnLo_0((*(&kernelContext_0)->vs_0)[kk_0]), _S84);
        accLo_0[0U] = _S101;
        float _S102 = fma(pj_0, attnHi_0(vw_0), _S85);
        accHi_0[0U] = _S102;
        uint vw_1 = (*(&kernelContext_0)->vs_0)[_S3];
        float _S103 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S3]), _S86);
        accLo_0[1U] = _S103;
        float _S104 = fma(pj_0, attnHi_0(vw_1), _S87);
        accHi_0[1U] = _S104;
        uint vw_2 = (*(&kernelContext_0)->vs_0)[_S5];
        float _S105 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S5]), _S88);
        accLo_0[2U] = _S105;
        float _S106 = fma(pj_0, attnHi_0(vw_2), _S89);
        accHi_0[2U] = _S106;
        uint vw_3 = (*(&kernelContext_0)->vs_0)[_S7];
        float _S107 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S7]), _S90);
        accLo_0[3U] = _S107;
        float _S108 = fma(pj_0, attnHi_0(vw_3), _S91);
        accHi_0[3U] = _S108;
        uint vw_4 = (*(&kernelContext_0)->vs_0)[kk_0 + 32U];
        float _S109 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 32U]), _S92);
        accLo_0[4U] = _S109;
        float _S110 = fma(pj_0, attnHi_0(vw_4), _S93);
        accHi_0[4U] = _S110;
        uint vw_5 = (*(&kernelContext_0)->vs_0)[kk_0 + 40U];
        float _S111 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 40U]), _S94);
        accLo_0[5U] = _S111;
        float _S112 = fma(pj_0, attnHi_0(vw_5), _S95);
        accHi_0[5U] = _S112;
        uint vw_6 = (*(&kernelContext_0)->vs_0)[kk_0 + 48U];
        float _S113 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 48U]), _S96);
        accLo_0[6U] = _S113;
        float _S114 = fma(pj_0, attnHi_0(vw_6), _S97);
        accHi_0[6U] = _S114;
        uint vw_7 = (*(&kernelContext_0)->vs_0)[kk_0 + 56U];
        float _S115 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[kk_0 + 56U]), _S98);
        accLo_0[7U] = _S115;
        float _S116 = fma(pj_0, attnHi_0(vw_7), _S99);
        accHi_0[7U] = _S116;
        float pj_1 = (*(&kernelContext_0)->ps_0)[_S47];
        float _S117 = _S100 + (*(&kernelContext_0)->ps_0)[_S47];
        uint _S118 = 65U + kk_0;
        uint vw_8 = (*(&kernelContext_0)->vs_0)[_S118];
        float _S119 = fma((*(&kernelContext_0)->ps_0)[_S47], attnLo_0((*(&kernelContext_0)->vs_0)[_S118]), _S101);
        accLo_0[0U] = _S119;
        float _S120 = fma(pj_1, attnHi_0(vw_8), _S102);
        accHi_0[0U] = _S120;
        uint vw_9 = (*(&kernelContext_0)->vs_0)[_S118 + 8U];
        float _S121 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S118 + 8U]), _S103);
        accLo_0[1U] = _S121;
        float _S122 = fma(pj_1, attnHi_0(vw_9), _S104);
        accHi_0[1U] = _S122;
        uint vw_10 = (*(&kernelContext_0)->vs_0)[_S118 + 16U];
        float _S123 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S118 + 16U]), _S105);
        accLo_0[2U] = _S123;
        float _S124 = fma(pj_1, attnHi_0(vw_10), _S106);
        accHi_0[2U] = _S124;
        uint vw_11 = (*(&kernelContext_0)->vs_0)[_S118 + 24U];
        float _S125 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S118 + 24U]), _S107);
        accLo_0[3U] = _S125;
        float _S126 = fma(pj_1, attnHi_0(vw_11), _S108);
        accHi_0[3U] = _S126;
        uint vw_12 = (*(&kernelContext_0)->vs_0)[_S118 + 32U];
        float _S127 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S118 + 32U]), _S109);
        accLo_0[4U] = _S127;
        float _S128 = fma(pj_1, attnHi_0(vw_12), _S110);
        accHi_0[4U] = _S128;
        uint vw_13 = (*(&kernelContext_0)->vs_0)[_S118 + 40U];
        float _S129 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S118 + 40U]), _S111);
        accLo_0[5U] = _S129;
        float _S130 = fma(pj_1, attnHi_0(vw_13), _S112);
        accHi_0[5U] = _S130;
        uint vw_14 = (*(&kernelContext_0)->vs_0)[_S118 + 48U];
        float _S131 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S118 + 48U]), _S113);
        accLo_0[6U] = _S131;
        float _S132 = fma(pj_1, attnHi_0(vw_14), _S114);
        accHi_0[6U] = _S132;
        uint vw_15 = (*(&kernelContext_0)->vs_0)[_S118 + 56U];
        float _S133 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S118 + 56U]), _S115);
        accLo_0[7U] = _S133;
        float _S134 = fma(pj_1, attnHi_0(vw_15), _S116);
        accHi_0[7U] = _S134;
        float pj_2 = (*(&kernelContext_0)->ps_0)[_S48];
        float _S135 = _S117 + (*(&kernelContext_0)->ps_0)[_S48];
        uint _S136 = 130U + kk_0;
        uint vw_16 = (*(&kernelContext_0)->vs_0)[_S136];
        float _S137 = fma((*(&kernelContext_0)->ps_0)[_S48], attnLo_0((*(&kernelContext_0)->vs_0)[_S136]), _S119);
        accLo_0[0U] = _S137;
        float _S138 = fma(pj_2, attnHi_0(vw_16), _S120);
        accHi_0[0U] = _S138;
        uint vw_17 = (*(&kernelContext_0)->vs_0)[_S136 + 8U];
        float _S139 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S136 + 8U]), _S121);
        accLo_0[1U] = _S139;
        float _S140 = fma(pj_2, attnHi_0(vw_17), _S122);
        accHi_0[1U] = _S140;
        uint vw_18 = (*(&kernelContext_0)->vs_0)[_S136 + 16U];
        float _S141 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S136 + 16U]), _S123);
        accLo_0[2U] = _S141;
        float _S142 = fma(pj_2, attnHi_0(vw_18), _S124);
        accHi_0[2U] = _S142;
        uint vw_19 = (*(&kernelContext_0)->vs_0)[_S136 + 24U];
        float _S143 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S136 + 24U]), _S125);
        accLo_0[3U] = _S143;
        float _S144 = fma(pj_2, attnHi_0(vw_19), _S126);
        accHi_0[3U] = _S144;
        uint vw_20 = (*(&kernelContext_0)->vs_0)[_S136 + 32U];
        float _S145 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S136 + 32U]), _S127);
        accLo_0[4U] = _S145;
        float _S146 = fma(pj_2, attnHi_0(vw_20), _S128);
        accHi_0[4U] = _S146;
        uint vw_21 = (*(&kernelContext_0)->vs_0)[_S136 + 40U];
        float _S147 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S136 + 40U]), _S129);
        accLo_0[5U] = _S147;
        float _S148 = fma(pj_2, attnHi_0(vw_21), _S130);
        accHi_0[5U] = _S148;
        uint vw_22 = (*(&kernelContext_0)->vs_0)[_S136 + 48U];
        float _S149 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S136 + 48U]), _S131);
        accLo_0[6U] = _S149;
        float _S150 = fma(pj_2, attnHi_0(vw_22), _S132);
        accHi_0[6U] = _S150;
        uint vw_23 = (*(&kernelContext_0)->vs_0)[_S136 + 56U];
        float _S151 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S136 + 56U]), _S133);
        accLo_0[7U] = _S151;
        float _S152 = fma(pj_2, attnHi_0(vw_23), _S134);
        accHi_0[7U] = _S152;
        float pj_3 = (*(&kernelContext_0)->ps_0)[_S49];
        float _S153 = _S135 + (*(&kernelContext_0)->ps_0)[_S49];
        uint _S154 = 195U + kk_0;
        uint vw_24 = (*(&kernelContext_0)->vs_0)[_S154];
        float _S155 = fma((*(&kernelContext_0)->ps_0)[_S49], attnLo_0((*(&kernelContext_0)->vs_0)[_S154]), _S137);
        accLo_0[0U] = _S155;
        float _S156 = fma(pj_3, attnHi_0(vw_24), _S138);
        accHi_0[0U] = _S156;
        uint vw_25 = (*(&kernelContext_0)->vs_0)[_S154 + 8U];
        float _S157 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S154 + 8U]), _S139);
        accLo_0[1U] = _S157;
        float _S158 = fma(pj_3, attnHi_0(vw_25), _S140);
        accHi_0[1U] = _S158;
        uint vw_26 = (*(&kernelContext_0)->vs_0)[_S154 + 16U];
        float _S159 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S154 + 16U]), _S141);
        accLo_0[2U] = _S159;
        float _S160 = fma(pj_3, attnHi_0(vw_26), _S142);
        accHi_0[2U] = _S160;
        uint vw_27 = (*(&kernelContext_0)->vs_0)[_S154 + 24U];
        float _S161 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S154 + 24U]), _S143);
        accLo_0[3U] = _S161;
        float _S162 = fma(pj_3, attnHi_0(vw_27), _S144);
        accHi_0[3U] = _S162;
        uint vw_28 = (*(&kernelContext_0)->vs_0)[_S154 + 32U];
        float _S163 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S154 + 32U]), _S145);
        accLo_0[4U] = _S163;
        float _S164 = fma(pj_3, attnHi_0(vw_28), _S146);
        accHi_0[4U] = _S164;
        uint vw_29 = (*(&kernelContext_0)->vs_0)[_S154 + 40U];
        float _S165 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S154 + 40U]), _S147);
        accLo_0[5U] = _S165;
        float _S166 = fma(pj_3, attnHi_0(vw_29), _S148);
        accHi_0[5U] = _S166;
        uint vw_30 = (*(&kernelContext_0)->vs_0)[_S154 + 48U];
        float _S167 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S154 + 48U]), _S149);
        accLo_0[6U] = _S167;
        float _S168 = fma(pj_3, attnHi_0(vw_30), _S150);
        accHi_0[6U] = _S168;
        uint vw_31 = (*(&kernelContext_0)->vs_0)[_S154 + 56U];
        float _S169 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S154 + 56U]), _S151);
        accLo_0[7U] = _S169;
        float _S170 = fma(pj_3, attnHi_0(vw_31), _S152);
        accHi_0[7U] = _S170;
        float pj_4 = (*(&kernelContext_0)->ps_0)[_S50];
        float _S171 = _S153 + (*(&kernelContext_0)->ps_0)[_S50];
        uint _S172 = 260U + kk_0;
        uint vw_32 = (*(&kernelContext_0)->vs_0)[_S172];
        float _S173 = fma((*(&kernelContext_0)->ps_0)[_S50], attnLo_0((*(&kernelContext_0)->vs_0)[_S172]), _S155);
        accLo_0[0U] = _S173;
        float _S174 = fma(pj_4, attnHi_0(vw_32), _S156);
        accHi_0[0U] = _S174;
        uint vw_33 = (*(&kernelContext_0)->vs_0)[_S172 + 8U];
        float _S175 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S172 + 8U]), _S157);
        accLo_0[1U] = _S175;
        float _S176 = fma(pj_4, attnHi_0(vw_33), _S158);
        accHi_0[1U] = _S176;
        uint vw_34 = (*(&kernelContext_0)->vs_0)[_S172 + 16U];
        float _S177 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S172 + 16U]), _S159);
        accLo_0[2U] = _S177;
        float _S178 = fma(pj_4, attnHi_0(vw_34), _S160);
        accHi_0[2U] = _S178;
        uint vw_35 = (*(&kernelContext_0)->vs_0)[_S172 + 24U];
        float _S179 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S172 + 24U]), _S161);
        accLo_0[3U] = _S179;
        float _S180 = fma(pj_4, attnHi_0(vw_35), _S162);
        accHi_0[3U] = _S180;
        uint vw_36 = (*(&kernelContext_0)->vs_0)[_S172 + 32U];
        float _S181 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S172 + 32U]), _S163);
        accLo_0[4U] = _S181;
        float _S182 = fma(pj_4, attnHi_0(vw_36), _S164);
        accHi_0[4U] = _S182;
        uint vw_37 = (*(&kernelContext_0)->vs_0)[_S172 + 40U];
        float _S183 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S172 + 40U]), _S165);
        accLo_0[5U] = _S183;
        float _S184 = fma(pj_4, attnHi_0(vw_37), _S166);
        accHi_0[5U] = _S184;
        uint vw_38 = (*(&kernelContext_0)->vs_0)[_S172 + 48U];
        float _S185 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S172 + 48U]), _S167);
        accLo_0[6U] = _S185;
        float _S186 = fma(pj_4, attnHi_0(vw_38), _S168);
        accHi_0[6U] = _S186;
        uint vw_39 = (*(&kernelContext_0)->vs_0)[_S172 + 56U];
        float _S187 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S172 + 56U]), _S169);
        accLo_0[7U] = _S187;
        float _S188 = fma(pj_4, attnHi_0(vw_39), _S170);
        accHi_0[7U] = _S188;
        float pj_5 = (*(&kernelContext_0)->ps_0)[_S51];
        float _S189 = _S171 + (*(&kernelContext_0)->ps_0)[_S51];
        uint _S190 = 325U + kk_0;
        uint vw_40 = (*(&kernelContext_0)->vs_0)[_S190];
        float _S191 = fma((*(&kernelContext_0)->ps_0)[_S51], attnLo_0((*(&kernelContext_0)->vs_0)[_S190]), _S173);
        accLo_0[0U] = _S191;
        float _S192 = fma(pj_5, attnHi_0(vw_40), _S174);
        accHi_0[0U] = _S192;
        uint vw_41 = (*(&kernelContext_0)->vs_0)[_S190 + 8U];
        float _S193 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S190 + 8U]), _S175);
        accLo_0[1U] = _S193;
        float _S194 = fma(pj_5, attnHi_0(vw_41), _S176);
        accHi_0[1U] = _S194;
        uint vw_42 = (*(&kernelContext_0)->vs_0)[_S190 + 16U];
        float _S195 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S190 + 16U]), _S177);
        accLo_0[2U] = _S195;
        float _S196 = fma(pj_5, attnHi_0(vw_42), _S178);
        accHi_0[2U] = _S196;
        uint vw_43 = (*(&kernelContext_0)->vs_0)[_S190 + 24U];
        float _S197 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S190 + 24U]), _S179);
        accLo_0[3U] = _S197;
        float _S198 = fma(pj_5, attnHi_0(vw_43), _S180);
        accHi_0[3U] = _S198;
        uint vw_44 = (*(&kernelContext_0)->vs_0)[_S190 + 32U];
        float _S199 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S190 + 32U]), _S181);
        accLo_0[4U] = _S199;
        float _S200 = fma(pj_5, attnHi_0(vw_44), _S182);
        accHi_0[4U] = _S200;
        uint vw_45 = (*(&kernelContext_0)->vs_0)[_S190 + 40U];
        float _S201 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S190 + 40U]), _S183);
        accLo_0[5U] = _S201;
        float _S202 = fma(pj_5, attnHi_0(vw_45), _S184);
        accHi_0[5U] = _S202;
        uint vw_46 = (*(&kernelContext_0)->vs_0)[_S190 + 48U];
        float _S203 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S190 + 48U]), _S185);
        accLo_0[6U] = _S203;
        float _S204 = fma(pj_5, attnHi_0(vw_46), _S186);
        accHi_0[6U] = _S204;
        uint vw_47 = (*(&kernelContext_0)->vs_0)[_S190 + 56U];
        float _S205 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S190 + 56U]), _S187);
        accLo_0[7U] = _S205;
        float _S206 = fma(pj_5, attnHi_0(vw_47), _S188);
        accHi_0[7U] = _S206;
        float pj_6 = (*(&kernelContext_0)->ps_0)[_S52];
        float _S207 = _S189 + (*(&kernelContext_0)->ps_0)[_S52];
        uint _S208 = 390U + kk_0;
        uint vw_48 = (*(&kernelContext_0)->vs_0)[_S208];
        float _S209 = fma((*(&kernelContext_0)->ps_0)[_S52], attnLo_0((*(&kernelContext_0)->vs_0)[_S208]), _S191);
        accLo_0[0U] = _S209;
        float _S210 = fma(pj_6, attnHi_0(vw_48), _S192);
        accHi_0[0U] = _S210;
        uint vw_49 = (*(&kernelContext_0)->vs_0)[_S208 + 8U];
        float _S211 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S208 + 8U]), _S193);
        accLo_0[1U] = _S211;
        float _S212 = fma(pj_6, attnHi_0(vw_49), _S194);
        accHi_0[1U] = _S212;
        uint vw_50 = (*(&kernelContext_0)->vs_0)[_S208 + 16U];
        float _S213 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S208 + 16U]), _S195);
        accLo_0[2U] = _S213;
        float _S214 = fma(pj_6, attnHi_0(vw_50), _S196);
        accHi_0[2U] = _S214;
        uint vw_51 = (*(&kernelContext_0)->vs_0)[_S208 + 24U];
        float _S215 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S208 + 24U]), _S197);
        accLo_0[3U] = _S215;
        float _S216 = fma(pj_6, attnHi_0(vw_51), _S198);
        accHi_0[3U] = _S216;
        uint vw_52 = (*(&kernelContext_0)->vs_0)[_S208 + 32U];
        float _S217 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S208 + 32U]), _S199);
        accLo_0[4U] = _S217;
        float _S218 = fma(pj_6, attnHi_0(vw_52), _S200);
        accHi_0[4U] = _S218;
        uint vw_53 = (*(&kernelContext_0)->vs_0)[_S208 + 40U];
        float _S219 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S208 + 40U]), _S201);
        accLo_0[5U] = _S219;
        float _S220 = fma(pj_6, attnHi_0(vw_53), _S202);
        accHi_0[5U] = _S220;
        uint vw_54 = (*(&kernelContext_0)->vs_0)[_S208 + 48U];
        float _S221 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S208 + 48U]), _S203);
        accLo_0[6U] = _S221;
        float _S222 = fma(pj_6, attnHi_0(vw_54), _S204);
        accHi_0[6U] = _S222;
        uint vw_55 = (*(&kernelContext_0)->vs_0)[_S208 + 56U];
        float _S223 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S208 + 56U]), _S205);
        accLo_0[7U] = _S223;
        float _S224 = fma(pj_6, attnHi_0(vw_55), _S206);
        accHi_0[7U] = _S224;
        float pj_7 = (*(&kernelContext_0)->ps_0)[_S53];
        float _S225 = _S207 + (*(&kernelContext_0)->ps_0)[_S53];
        uint _S226 = 455U + kk_0;
        uint vw_56 = (*(&kernelContext_0)->vs_0)[_S226];
        float _S227 = fma((*(&kernelContext_0)->ps_0)[_S53], attnLo_0((*(&kernelContext_0)->vs_0)[_S226]), _S209);
        accLo_0[0U] = _S227;
        float _S228 = fma(pj_7, attnHi_0(vw_56), _S210);
        accHi_0[0U] = _S228;
        uint vw_57 = (*(&kernelContext_0)->vs_0)[_S226 + 8U];
        float _S229 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S226 + 8U]), _S211);
        accLo_0[1U] = _S229;
        float _S230 = fma(pj_7, attnHi_0(vw_57), _S212);
        accHi_0[1U] = _S230;
        uint vw_58 = (*(&kernelContext_0)->vs_0)[_S226 + 16U];
        float _S231 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S226 + 16U]), _S213);
        accLo_0[2U] = _S231;
        float _S232 = fma(pj_7, attnHi_0(vw_58), _S214);
        accHi_0[2U] = _S232;
        uint vw_59 = (*(&kernelContext_0)->vs_0)[_S226 + 24U];
        float _S233 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S226 + 24U]), _S215);
        accLo_0[3U] = _S233;
        float _S234 = fma(pj_7, attnHi_0(vw_59), _S216);
        accHi_0[3U] = _S234;
        uint vw_60 = (*(&kernelContext_0)->vs_0)[_S226 + 32U];
        float _S235 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S226 + 32U]), _S217);
        accLo_0[4U] = _S235;
        float _S236 = fma(pj_7, attnHi_0(vw_60), _S218);
        accHi_0[4U] = _S236;
        uint vw_61 = (*(&kernelContext_0)->vs_0)[_S226 + 40U];
        float _S237 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S226 + 40U]), _S219);
        accLo_0[5U] = _S237;
        float _S238 = fma(pj_7, attnHi_0(vw_61), _S220);
        accHi_0[5U] = _S238;
        uint vw_62 = (*(&kernelContext_0)->vs_0)[_S226 + 48U];
        float _S239 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S226 + 48U]), _S221);
        accLo_0[6U] = _S239;
        float _S240 = fma(pj_7, attnHi_0(vw_62), _S222);
        accHi_0[6U] = _S240;
        uint vw_63 = (*(&kernelContext_0)->vs_0)[_S226 + 56U];
        float _S241 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S226 + 56U]), _S223);
        accLo_0[7U] = _S241;
        float _S242 = fma(pj_7, attnHi_0(vw_63), _S224);
        accHi_0[7U] = _S242;
        float pj_8 = (*(&kernelContext_0)->ps_0)[_S54];
        float _S243 = _S225 + (*(&kernelContext_0)->ps_0)[_S54];
        uint _S244 = 520U + kk_0;
        uint vw_64 = (*(&kernelContext_0)->vs_0)[_S244];
        float _S245 = fma((*(&kernelContext_0)->ps_0)[_S54], attnLo_0((*(&kernelContext_0)->vs_0)[_S244]), _S227);
        accLo_0[0U] = _S245;
        float _S246 = fma(pj_8, attnHi_0(vw_64), _S228);
        accHi_0[0U] = _S246;
        uint vw_65 = (*(&kernelContext_0)->vs_0)[_S244 + 8U];
        float _S247 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S244 + 8U]), _S229);
        accLo_0[1U] = _S247;
        float _S248 = fma(pj_8, attnHi_0(vw_65), _S230);
        accHi_0[1U] = _S248;
        uint vw_66 = (*(&kernelContext_0)->vs_0)[_S244 + 16U];
        float _S249 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S244 + 16U]), _S231);
        accLo_0[2U] = _S249;
        float _S250 = fma(pj_8, attnHi_0(vw_66), _S232);
        accHi_0[2U] = _S250;
        uint vw_67 = (*(&kernelContext_0)->vs_0)[_S244 + 24U];
        float _S251 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S244 + 24U]), _S233);
        accLo_0[3U] = _S251;
        float _S252 = fma(pj_8, attnHi_0(vw_67), _S234);
        accHi_0[3U] = _S252;
        uint vw_68 = (*(&kernelContext_0)->vs_0)[_S244 + 32U];
        float _S253 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S244 + 32U]), _S235);
        accLo_0[4U] = _S253;
        float _S254 = fma(pj_8, attnHi_0(vw_68), _S236);
        accHi_0[4U] = _S254;
        uint vw_69 = (*(&kernelContext_0)->vs_0)[_S244 + 40U];
        float _S255 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S244 + 40U]), _S237);
        accLo_0[5U] = _S255;
        float _S256 = fma(pj_8, attnHi_0(vw_69), _S238);
        accHi_0[5U] = _S256;
        uint vw_70 = (*(&kernelContext_0)->vs_0)[_S244 + 48U];
        float _S257 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S244 + 48U]), _S239);
        accLo_0[6U] = _S257;
        float _S258 = fma(pj_8, attnHi_0(vw_70), _S240);
        accHi_0[6U] = _S258;
        uint vw_71 = (*(&kernelContext_0)->vs_0)[_S244 + 56U];
        float _S259 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S244 + 56U]), _S241);
        accLo_0[7U] = _S259;
        float _S260 = fma(pj_8, attnHi_0(vw_71), _S242);
        accHi_0[7U] = _S260;
        float pj_9 = (*(&kernelContext_0)->ps_0)[_S55];
        float _S261 = _S243 + (*(&kernelContext_0)->ps_0)[_S55];
        uint _S262 = 585U + kk_0;
        uint vw_72 = (*(&kernelContext_0)->vs_0)[_S262];
        float _S263 = fma((*(&kernelContext_0)->ps_0)[_S55], attnLo_0((*(&kernelContext_0)->vs_0)[_S262]), _S245);
        accLo_0[0U] = _S263;
        float _S264 = fma(pj_9, attnHi_0(vw_72), _S246);
        accHi_0[0U] = _S264;
        uint vw_73 = (*(&kernelContext_0)->vs_0)[_S262 + 8U];
        float _S265 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S262 + 8U]), _S247);
        accLo_0[1U] = _S265;
        float _S266 = fma(pj_9, attnHi_0(vw_73), _S248);
        accHi_0[1U] = _S266;
        uint vw_74 = (*(&kernelContext_0)->vs_0)[_S262 + 16U];
        float _S267 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S262 + 16U]), _S249);
        accLo_0[2U] = _S267;
        float _S268 = fma(pj_9, attnHi_0(vw_74), _S250);
        accHi_0[2U] = _S268;
        uint vw_75 = (*(&kernelContext_0)->vs_0)[_S262 + 24U];
        float _S269 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S262 + 24U]), _S251);
        accLo_0[3U] = _S269;
        float _S270 = fma(pj_9, attnHi_0(vw_75), _S252);
        accHi_0[3U] = _S270;
        uint vw_76 = (*(&kernelContext_0)->vs_0)[_S262 + 32U];
        float _S271 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S262 + 32U]), _S253);
        accLo_0[4U] = _S271;
        float _S272 = fma(pj_9, attnHi_0(vw_76), _S254);
        accHi_0[4U] = _S272;
        uint vw_77 = (*(&kernelContext_0)->vs_0)[_S262 + 40U];
        float _S273 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S262 + 40U]), _S255);
        accLo_0[5U] = _S273;
        float _S274 = fma(pj_9, attnHi_0(vw_77), _S256);
        accHi_0[5U] = _S274;
        uint vw_78 = (*(&kernelContext_0)->vs_0)[_S262 + 48U];
        float _S275 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S262 + 48U]), _S257);
        accLo_0[6U] = _S275;
        float _S276 = fma(pj_9, attnHi_0(vw_78), _S258);
        accHi_0[6U] = _S276;
        uint vw_79 = (*(&kernelContext_0)->vs_0)[_S262 + 56U];
        float _S277 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S262 + 56U]), _S259);
        accLo_0[7U] = _S277;
        float _S278 = fma(pj_9, attnHi_0(vw_79), _S260);
        accHi_0[7U] = _S278;
        float pj_10 = (*(&kernelContext_0)->ps_0)[_S56];
        float _S279 = _S261 + (*(&kernelContext_0)->ps_0)[_S56];
        uint _S280 = 650U + kk_0;
        uint vw_80 = (*(&kernelContext_0)->vs_0)[_S280];
        float _S281 = fma((*(&kernelContext_0)->ps_0)[_S56], attnLo_0((*(&kernelContext_0)->vs_0)[_S280]), _S263);
        accLo_0[0U] = _S281;
        float _S282 = fma(pj_10, attnHi_0(vw_80), _S264);
        accHi_0[0U] = _S282;
        uint vw_81 = (*(&kernelContext_0)->vs_0)[_S280 + 8U];
        float _S283 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S280 + 8U]), _S265);
        accLo_0[1U] = _S283;
        float _S284 = fma(pj_10, attnHi_0(vw_81), _S266);
        accHi_0[1U] = _S284;
        uint vw_82 = (*(&kernelContext_0)->vs_0)[_S280 + 16U];
        float _S285 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S280 + 16U]), _S267);
        accLo_0[2U] = _S285;
        float _S286 = fma(pj_10, attnHi_0(vw_82), _S268);
        accHi_0[2U] = _S286;
        uint vw_83 = (*(&kernelContext_0)->vs_0)[_S280 + 24U];
        float _S287 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S280 + 24U]), _S269);
        accLo_0[3U] = _S287;
        float _S288 = fma(pj_10, attnHi_0(vw_83), _S270);
        accHi_0[3U] = _S288;
        uint vw_84 = (*(&kernelContext_0)->vs_0)[_S280 + 32U];
        float _S289 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S280 + 32U]), _S271);
        accLo_0[4U] = _S289;
        float _S290 = fma(pj_10, attnHi_0(vw_84), _S272);
        accHi_0[4U] = _S290;
        uint vw_85 = (*(&kernelContext_0)->vs_0)[_S280 + 40U];
        float _S291 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S280 + 40U]), _S273);
        accLo_0[5U] = _S291;
        float _S292 = fma(pj_10, attnHi_0(vw_85), _S274);
        accHi_0[5U] = _S292;
        uint vw_86 = (*(&kernelContext_0)->vs_0)[_S280 + 48U];
        float _S293 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S280 + 48U]), _S275);
        accLo_0[6U] = _S293;
        float _S294 = fma(pj_10, attnHi_0(vw_86), _S276);
        accHi_0[6U] = _S294;
        uint vw_87 = (*(&kernelContext_0)->vs_0)[_S280 + 56U];
        float _S295 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S280 + 56U]), _S277);
        accLo_0[7U] = _S295;
        float _S296 = fma(pj_10, attnHi_0(vw_87), _S278);
        accHi_0[7U] = _S296;
        float pj_11 = (*(&kernelContext_0)->ps_0)[_S57];
        float _S297 = _S279 + (*(&kernelContext_0)->ps_0)[_S57];
        uint _S298 = 715U + kk_0;
        uint vw_88 = (*(&kernelContext_0)->vs_0)[_S298];
        float _S299 = fma((*(&kernelContext_0)->ps_0)[_S57], attnLo_0((*(&kernelContext_0)->vs_0)[_S298]), _S281);
        accLo_0[0U] = _S299;
        float _S300 = fma(pj_11, attnHi_0(vw_88), _S282);
        accHi_0[0U] = _S300;
        uint vw_89 = (*(&kernelContext_0)->vs_0)[_S298 + 8U];
        float _S301 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S298 + 8U]), _S283);
        accLo_0[1U] = _S301;
        float _S302 = fma(pj_11, attnHi_0(vw_89), _S284);
        accHi_0[1U] = _S302;
        uint vw_90 = (*(&kernelContext_0)->vs_0)[_S298 + 16U];
        float _S303 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S298 + 16U]), _S285);
        accLo_0[2U] = _S303;
        float _S304 = fma(pj_11, attnHi_0(vw_90), _S286);
        accHi_0[2U] = _S304;
        uint vw_91 = (*(&kernelContext_0)->vs_0)[_S298 + 24U];
        float _S305 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S298 + 24U]), _S287);
        accLo_0[3U] = _S305;
        float _S306 = fma(pj_11, attnHi_0(vw_91), _S288);
        accHi_0[3U] = _S306;
        uint vw_92 = (*(&kernelContext_0)->vs_0)[_S298 + 32U];
        float _S307 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S298 + 32U]), _S289);
        accLo_0[4U] = _S307;
        float _S308 = fma(pj_11, attnHi_0(vw_92), _S290);
        accHi_0[4U] = _S308;
        uint vw_93 = (*(&kernelContext_0)->vs_0)[_S298 + 40U];
        float _S309 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S298 + 40U]), _S291);
        accLo_0[5U] = _S309;
        float _S310 = fma(pj_11, attnHi_0(vw_93), _S292);
        accHi_0[5U] = _S310;
        uint vw_94 = (*(&kernelContext_0)->vs_0)[_S298 + 48U];
        float _S311 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S298 + 48U]), _S293);
        accLo_0[6U] = _S311;
        float _S312 = fma(pj_11, attnHi_0(vw_94), _S294);
        accHi_0[6U] = _S312;
        uint vw_95 = (*(&kernelContext_0)->vs_0)[_S298 + 56U];
        float _S313 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S298 + 56U]), _S295);
        accLo_0[7U] = _S313;
        float _S314 = fma(pj_11, attnHi_0(vw_95), _S296);
        accHi_0[7U] = _S314;
        float pj_12 = (*(&kernelContext_0)->ps_0)[_S58];
        float _S315 = _S297 + (*(&kernelContext_0)->ps_0)[_S58];
        uint _S316 = 780U + kk_0;
        uint vw_96 = (*(&kernelContext_0)->vs_0)[_S316];
        float _S317 = fma((*(&kernelContext_0)->ps_0)[_S58], attnLo_0((*(&kernelContext_0)->vs_0)[_S316]), _S299);
        accLo_0[0U] = _S317;
        float _S318 = fma(pj_12, attnHi_0(vw_96), _S300);
        accHi_0[0U] = _S318;
        uint vw_97 = (*(&kernelContext_0)->vs_0)[_S316 + 8U];
        float _S319 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S316 + 8U]), _S301);
        accLo_0[1U] = _S319;
        float _S320 = fma(pj_12, attnHi_0(vw_97), _S302);
        accHi_0[1U] = _S320;
        uint vw_98 = (*(&kernelContext_0)->vs_0)[_S316 + 16U];
        float _S321 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S316 + 16U]), _S303);
        accLo_0[2U] = _S321;
        float _S322 = fma(pj_12, attnHi_0(vw_98), _S304);
        accHi_0[2U] = _S322;
        uint vw_99 = (*(&kernelContext_0)->vs_0)[_S316 + 24U];
        float _S323 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S316 + 24U]), _S305);
        accLo_0[3U] = _S323;
        float _S324 = fma(pj_12, attnHi_0(vw_99), _S306);
        accHi_0[3U] = _S324;
        uint vw_100 = (*(&kernelContext_0)->vs_0)[_S316 + 32U];
        float _S325 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S316 + 32U]), _S307);
        accLo_0[4U] = _S325;
        float _S326 = fma(pj_12, attnHi_0(vw_100), _S308);
        accHi_0[4U] = _S326;
        uint vw_101 = (*(&kernelContext_0)->vs_0)[_S316 + 40U];
        float _S327 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S316 + 40U]), _S309);
        accLo_0[5U] = _S327;
        float _S328 = fma(pj_12, attnHi_0(vw_101), _S310);
        accHi_0[5U] = _S328;
        uint vw_102 = (*(&kernelContext_0)->vs_0)[_S316 + 48U];
        float _S329 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S316 + 48U]), _S311);
        accLo_0[6U] = _S329;
        float _S330 = fma(pj_12, attnHi_0(vw_102), _S312);
        accHi_0[6U] = _S330;
        uint vw_103 = (*(&kernelContext_0)->vs_0)[_S316 + 56U];
        float _S331 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S316 + 56U]), _S313);
        accLo_0[7U] = _S331;
        float _S332 = fma(pj_12, attnHi_0(vw_103), _S314);
        accHi_0[7U] = _S332;
        float pj_13 = (*(&kernelContext_0)->ps_0)[_S59];
        float _S333 = _S315 + (*(&kernelContext_0)->ps_0)[_S59];
        uint _S334 = 845U + kk_0;
        uint vw_104 = (*(&kernelContext_0)->vs_0)[_S334];
        float _S335 = fma((*(&kernelContext_0)->ps_0)[_S59], attnLo_0((*(&kernelContext_0)->vs_0)[_S334]), _S317);
        accLo_0[0U] = _S335;
        float _S336 = fma(pj_13, attnHi_0(vw_104), _S318);
        accHi_0[0U] = _S336;
        uint vw_105 = (*(&kernelContext_0)->vs_0)[_S334 + 8U];
        float _S337 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S334 + 8U]), _S319);
        accLo_0[1U] = _S337;
        float _S338 = fma(pj_13, attnHi_0(vw_105), _S320);
        accHi_0[1U] = _S338;
        uint vw_106 = (*(&kernelContext_0)->vs_0)[_S334 + 16U];
        float _S339 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S334 + 16U]), _S321);
        accLo_0[2U] = _S339;
        float _S340 = fma(pj_13, attnHi_0(vw_106), _S322);
        accHi_0[2U] = _S340;
        uint vw_107 = (*(&kernelContext_0)->vs_0)[_S334 + 24U];
        float _S341 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S334 + 24U]), _S323);
        accLo_0[3U] = _S341;
        float _S342 = fma(pj_13, attnHi_0(vw_107), _S324);
        accHi_0[3U] = _S342;
        uint vw_108 = (*(&kernelContext_0)->vs_0)[_S334 + 32U];
        float _S343 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S334 + 32U]), _S325);
        accLo_0[4U] = _S343;
        float _S344 = fma(pj_13, attnHi_0(vw_108), _S326);
        accHi_0[4U] = _S344;
        uint vw_109 = (*(&kernelContext_0)->vs_0)[_S334 + 40U];
        float _S345 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S334 + 40U]), _S327);
        accLo_0[5U] = _S345;
        float _S346 = fma(pj_13, attnHi_0(vw_109), _S328);
        accHi_0[5U] = _S346;
        uint vw_110 = (*(&kernelContext_0)->vs_0)[_S334 + 48U];
        float _S347 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S334 + 48U]), _S329);
        accLo_0[6U] = _S347;
        float _S348 = fma(pj_13, attnHi_0(vw_110), _S330);
        accHi_0[6U] = _S348;
        uint vw_111 = (*(&kernelContext_0)->vs_0)[_S334 + 56U];
        float _S349 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S334 + 56U]), _S331);
        accLo_0[7U] = _S349;
        float _S350 = fma(pj_13, attnHi_0(vw_111), _S332);
        accHi_0[7U] = _S350;
        float pj_14 = (*(&kernelContext_0)->ps_0)[_S60];
        float _S351 = _S333 + (*(&kernelContext_0)->ps_0)[_S60];
        uint _S352 = 910U + kk_0;
        uint vw_112 = (*(&kernelContext_0)->vs_0)[_S352];
        float _S353 = fma((*(&kernelContext_0)->ps_0)[_S60], attnLo_0((*(&kernelContext_0)->vs_0)[_S352]), _S335);
        accLo_0[0U] = _S353;
        float _S354 = fma(pj_14, attnHi_0(vw_112), _S336);
        accHi_0[0U] = _S354;
        uint vw_113 = (*(&kernelContext_0)->vs_0)[_S352 + 8U];
        float _S355 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S352 + 8U]), _S337);
        accLo_0[1U] = _S355;
        float _S356 = fma(pj_14, attnHi_0(vw_113), _S338);
        accHi_0[1U] = _S356;
        uint vw_114 = (*(&kernelContext_0)->vs_0)[_S352 + 16U];
        float _S357 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S352 + 16U]), _S339);
        accLo_0[2U] = _S357;
        float _S358 = fma(pj_14, attnHi_0(vw_114), _S340);
        accHi_0[2U] = _S358;
        uint vw_115 = (*(&kernelContext_0)->vs_0)[_S352 + 24U];
        float _S359 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S352 + 24U]), _S341);
        accLo_0[3U] = _S359;
        float _S360 = fma(pj_14, attnHi_0(vw_115), _S342);
        accHi_0[3U] = _S360;
        uint vw_116 = (*(&kernelContext_0)->vs_0)[_S352 + 32U];
        float _S361 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S352 + 32U]), _S343);
        accLo_0[4U] = _S361;
        float _S362 = fma(pj_14, attnHi_0(vw_116), _S344);
        accHi_0[4U] = _S362;
        uint vw_117 = (*(&kernelContext_0)->vs_0)[_S352 + 40U];
        float _S363 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S352 + 40U]), _S345);
        accLo_0[5U] = _S363;
        float _S364 = fma(pj_14, attnHi_0(vw_117), _S346);
        accHi_0[5U] = _S364;
        uint vw_118 = (*(&kernelContext_0)->vs_0)[_S352 + 48U];
        float _S365 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S352 + 48U]), _S347);
        accLo_0[6U] = _S365;
        float _S366 = fma(pj_14, attnHi_0(vw_118), _S348);
        accHi_0[6U] = _S366;
        uint vw_119 = (*(&kernelContext_0)->vs_0)[_S352 + 56U];
        float _S367 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S352 + 56U]), _S349);
        accLo_0[7U] = _S367;
        float _S368 = fma(pj_14, attnHi_0(vw_119), _S350);
        accHi_0[7U] = _S368;
        float pj_15 = (*(&kernelContext_0)->ps_0)[_S61];
        float _S369 = _S351 + (*(&kernelContext_0)->ps_0)[_S61];
        uint _S370 = 975U + kk_0;
        uint vw_120 = (*(&kernelContext_0)->vs_0)[_S370];
        float _S371 = fma((*(&kernelContext_0)->ps_0)[_S61], attnLo_0((*(&kernelContext_0)->vs_0)[_S370]), _S353);
        accLo_0[0U] = _S371;
        float _S372 = fma(pj_15, attnHi_0(vw_120), _S354);
        accHi_0[0U] = _S372;
        uint vw_121 = (*(&kernelContext_0)->vs_0)[_S370 + 8U];
        float _S373 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S370 + 8U]), _S355);
        accLo_0[1U] = _S373;
        float _S374 = fma(pj_15, attnHi_0(vw_121), _S356);
        accHi_0[1U] = _S374;
        uint vw_122 = (*(&kernelContext_0)->vs_0)[_S370 + 16U];
        float _S375 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S370 + 16U]), _S357);
        accLo_0[2U] = _S375;
        float _S376 = fma(pj_15, attnHi_0(vw_122), _S358);
        accHi_0[2U] = _S376;
        uint vw_123 = (*(&kernelContext_0)->vs_0)[_S370 + 24U];
        float _S377 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S370 + 24U]), _S359);
        accLo_0[3U] = _S377;
        float _S378 = fma(pj_15, attnHi_0(vw_123), _S360);
        accHi_0[3U] = _S378;
        uint vw_124 = (*(&kernelContext_0)->vs_0)[_S370 + 32U];
        float _S379 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S370 + 32U]), _S361);
        accLo_0[4U] = _S379;
        float _S380 = fma(pj_15, attnHi_0(vw_124), _S362);
        accHi_0[4U] = _S380;
        uint vw_125 = (*(&kernelContext_0)->vs_0)[_S370 + 40U];
        float _S381 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S370 + 40U]), _S363);
        accLo_0[5U] = _S381;
        float _S382 = fma(pj_15, attnHi_0(vw_125), _S364);
        accHi_0[5U] = _S382;
        uint vw_126 = (*(&kernelContext_0)->vs_0)[_S370 + 48U];
        float _S383 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S370 + 48U]), _S365);
        accLo_0[6U] = _S383;
        float _S384 = fma(pj_15, attnHi_0(vw_126), _S366);
        accHi_0[6U] = _S384;
        uint vw_127 = (*(&kernelContext_0)->vs_0)[_S370 + 56U];
        float _S385 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S370 + 56U]), _S367);
        accLo_0[7U] = _S385;
        float _S386 = fma(pj_15, attnHi_0(vw_127), _S368);
        accHi_0[7U] = _S386;
        float pj_16 = (*(&kernelContext_0)->ps_0)[_S62];
        float _S387 = _S369 + (*(&kernelContext_0)->ps_0)[_S62];
        uint _S388 = 1040U + kk_0;
        uint vw_128 = (*(&kernelContext_0)->vs_0)[_S388];
        float _S389 = fma((*(&kernelContext_0)->ps_0)[_S62], attnLo_0((*(&kernelContext_0)->vs_0)[_S388]), _S371);
        accLo_0[0U] = _S389;
        float _S390 = fma(pj_16, attnHi_0(vw_128), _S372);
        accHi_0[0U] = _S390;
        uint vw_129 = (*(&kernelContext_0)->vs_0)[_S388 + 8U];
        float _S391 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S388 + 8U]), _S373);
        accLo_0[1U] = _S391;
        float _S392 = fma(pj_16, attnHi_0(vw_129), _S374);
        accHi_0[1U] = _S392;
        uint vw_130 = (*(&kernelContext_0)->vs_0)[_S388 + 16U];
        float _S393 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S388 + 16U]), _S375);
        accLo_0[2U] = _S393;
        float _S394 = fma(pj_16, attnHi_0(vw_130), _S376);
        accHi_0[2U] = _S394;
        uint vw_131 = (*(&kernelContext_0)->vs_0)[_S388 + 24U];
        float _S395 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S388 + 24U]), _S377);
        accLo_0[3U] = _S395;
        float _S396 = fma(pj_16, attnHi_0(vw_131), _S378);
        accHi_0[3U] = _S396;
        uint vw_132 = (*(&kernelContext_0)->vs_0)[_S388 + 32U];
        float _S397 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S388 + 32U]), _S379);
        accLo_0[4U] = _S397;
        float _S398 = fma(pj_16, attnHi_0(vw_132), _S380);
        accHi_0[4U] = _S398;
        uint vw_133 = (*(&kernelContext_0)->vs_0)[_S388 + 40U];
        float _S399 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S388 + 40U]), _S381);
        accLo_0[5U] = _S399;
        float _S400 = fma(pj_16, attnHi_0(vw_133), _S382);
        accHi_0[5U] = _S400;
        uint vw_134 = (*(&kernelContext_0)->vs_0)[_S388 + 48U];
        float _S401 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S388 + 48U]), _S383);
        accLo_0[6U] = _S401;
        float _S402 = fma(pj_16, attnHi_0(vw_134), _S384);
        accHi_0[6U] = _S402;
        uint vw_135 = (*(&kernelContext_0)->vs_0)[_S388 + 56U];
        float _S403 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S388 + 56U]), _S385);
        accLo_0[7U] = _S403;
        float _S404 = fma(pj_16, attnHi_0(vw_135), _S386);
        accHi_0[7U] = _S404;
        float pj_17 = (*(&kernelContext_0)->ps_0)[_S63];
        float _S405 = _S387 + (*(&kernelContext_0)->ps_0)[_S63];
        uint _S406 = 1105U + kk_0;
        uint vw_136 = (*(&kernelContext_0)->vs_0)[_S406];
        float _S407 = fma((*(&kernelContext_0)->ps_0)[_S63], attnLo_0((*(&kernelContext_0)->vs_0)[_S406]), _S389);
        accLo_0[0U] = _S407;
        float _S408 = fma(pj_17, attnHi_0(vw_136), _S390);
        accHi_0[0U] = _S408;
        uint vw_137 = (*(&kernelContext_0)->vs_0)[_S406 + 8U];
        float _S409 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S406 + 8U]), _S391);
        accLo_0[1U] = _S409;
        float _S410 = fma(pj_17, attnHi_0(vw_137), _S392);
        accHi_0[1U] = _S410;
        uint vw_138 = (*(&kernelContext_0)->vs_0)[_S406 + 16U];
        float _S411 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S406 + 16U]), _S393);
        accLo_0[2U] = _S411;
        float _S412 = fma(pj_17, attnHi_0(vw_138), _S394);
        accHi_0[2U] = _S412;
        uint vw_139 = (*(&kernelContext_0)->vs_0)[_S406 + 24U];
        float _S413 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S406 + 24U]), _S395);
        accLo_0[3U] = _S413;
        float _S414 = fma(pj_17, attnHi_0(vw_139), _S396);
        accHi_0[3U] = _S414;
        uint vw_140 = (*(&kernelContext_0)->vs_0)[_S406 + 32U];
        float _S415 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S406 + 32U]), _S397);
        accLo_0[4U] = _S415;
        float _S416 = fma(pj_17, attnHi_0(vw_140), _S398);
        accHi_0[4U] = _S416;
        uint vw_141 = (*(&kernelContext_0)->vs_0)[_S406 + 40U];
        float _S417 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S406 + 40U]), _S399);
        accLo_0[5U] = _S417;
        float _S418 = fma(pj_17, attnHi_0(vw_141), _S400);
        accHi_0[5U] = _S418;
        uint vw_142 = (*(&kernelContext_0)->vs_0)[_S406 + 48U];
        float _S419 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S406 + 48U]), _S401);
        accLo_0[6U] = _S419;
        float _S420 = fma(pj_17, attnHi_0(vw_142), _S402);
        accHi_0[6U] = _S420;
        uint vw_143 = (*(&kernelContext_0)->vs_0)[_S406 + 56U];
        float _S421 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S406 + 56U]), _S403);
        accLo_0[7U] = _S421;
        float _S422 = fma(pj_17, attnHi_0(vw_143), _S404);
        accHi_0[7U] = _S422;
        float pj_18 = (*(&kernelContext_0)->ps_0)[_S64];
        float _S423 = _S405 + (*(&kernelContext_0)->ps_0)[_S64];
        uint _S424 = 1170U + kk_0;
        uint vw_144 = (*(&kernelContext_0)->vs_0)[_S424];
        float _S425 = fma((*(&kernelContext_0)->ps_0)[_S64], attnLo_0((*(&kernelContext_0)->vs_0)[_S424]), _S407);
        accLo_0[0U] = _S425;
        float _S426 = fma(pj_18, attnHi_0(vw_144), _S408);
        accHi_0[0U] = _S426;
        uint vw_145 = (*(&kernelContext_0)->vs_0)[_S424 + 8U];
        float _S427 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S424 + 8U]), _S409);
        accLo_0[1U] = _S427;
        float _S428 = fma(pj_18, attnHi_0(vw_145), _S410);
        accHi_0[1U] = _S428;
        uint vw_146 = (*(&kernelContext_0)->vs_0)[_S424 + 16U];
        float _S429 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S424 + 16U]), _S411);
        accLo_0[2U] = _S429;
        float _S430 = fma(pj_18, attnHi_0(vw_146), _S412);
        accHi_0[2U] = _S430;
        uint vw_147 = (*(&kernelContext_0)->vs_0)[_S424 + 24U];
        float _S431 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S424 + 24U]), _S413);
        accLo_0[3U] = _S431;
        float _S432 = fma(pj_18, attnHi_0(vw_147), _S414);
        accHi_0[3U] = _S432;
        uint vw_148 = (*(&kernelContext_0)->vs_0)[_S424 + 32U];
        float _S433 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S424 + 32U]), _S415);
        accLo_0[4U] = _S433;
        float _S434 = fma(pj_18, attnHi_0(vw_148), _S416);
        accHi_0[4U] = _S434;
        uint vw_149 = (*(&kernelContext_0)->vs_0)[_S424 + 40U];
        float _S435 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S424 + 40U]), _S417);
        accLo_0[5U] = _S435;
        float _S436 = fma(pj_18, attnHi_0(vw_149), _S418);
        accHi_0[5U] = _S436;
        uint vw_150 = (*(&kernelContext_0)->vs_0)[_S424 + 48U];
        float _S437 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S424 + 48U]), _S419);
        accLo_0[6U] = _S437;
        float _S438 = fma(pj_18, attnHi_0(vw_150), _S420);
        accHi_0[6U] = _S438;
        uint vw_151 = (*(&kernelContext_0)->vs_0)[_S424 + 56U];
        float _S439 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S424 + 56U]), _S421);
        accLo_0[7U] = _S439;
        float _S440 = fma(pj_18, attnHi_0(vw_151), _S422);
        accHi_0[7U] = _S440;
        float pj_19 = (*(&kernelContext_0)->ps_0)[_S65];
        float _S441 = _S423 + (*(&kernelContext_0)->ps_0)[_S65];
        uint _S442 = 1235U + kk_0;
        uint vw_152 = (*(&kernelContext_0)->vs_0)[_S442];
        float _S443 = fma((*(&kernelContext_0)->ps_0)[_S65], attnLo_0((*(&kernelContext_0)->vs_0)[_S442]), _S425);
        accLo_0[0U] = _S443;
        float _S444 = fma(pj_19, attnHi_0(vw_152), _S426);
        accHi_0[0U] = _S444;
        uint vw_153 = (*(&kernelContext_0)->vs_0)[_S442 + 8U];
        float _S445 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S442 + 8U]), _S427);
        accLo_0[1U] = _S445;
        float _S446 = fma(pj_19, attnHi_0(vw_153), _S428);
        accHi_0[1U] = _S446;
        uint vw_154 = (*(&kernelContext_0)->vs_0)[_S442 + 16U];
        float _S447 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S442 + 16U]), _S429);
        accLo_0[2U] = _S447;
        float _S448 = fma(pj_19, attnHi_0(vw_154), _S430);
        accHi_0[2U] = _S448;
        uint vw_155 = (*(&kernelContext_0)->vs_0)[_S442 + 24U];
        float _S449 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S442 + 24U]), _S431);
        accLo_0[3U] = _S449;
        float _S450 = fma(pj_19, attnHi_0(vw_155), _S432);
        accHi_0[3U] = _S450;
        uint vw_156 = (*(&kernelContext_0)->vs_0)[_S442 + 32U];
        float _S451 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S442 + 32U]), _S433);
        accLo_0[4U] = _S451;
        float _S452 = fma(pj_19, attnHi_0(vw_156), _S434);
        accHi_0[4U] = _S452;
        uint vw_157 = (*(&kernelContext_0)->vs_0)[_S442 + 40U];
        float _S453 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S442 + 40U]), _S435);
        accLo_0[5U] = _S453;
        float _S454 = fma(pj_19, attnHi_0(vw_157), _S436);
        accHi_0[5U] = _S454;
        uint vw_158 = (*(&kernelContext_0)->vs_0)[_S442 + 48U];
        float _S455 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S442 + 48U]), _S437);
        accLo_0[6U] = _S455;
        float _S456 = fma(pj_19, attnHi_0(vw_158), _S438);
        accHi_0[6U] = _S456;
        uint vw_159 = (*(&kernelContext_0)->vs_0)[_S442 + 56U];
        float _S457 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S442 + 56U]), _S439);
        accLo_0[7U] = _S457;
        float _S458 = fma(pj_19, attnHi_0(vw_159), _S440);
        accHi_0[7U] = _S458;
        float pj_20 = (*(&kernelContext_0)->ps_0)[_S66];
        float _S459 = _S441 + (*(&kernelContext_0)->ps_0)[_S66];
        uint _S460 = 1300U + kk_0;
        uint vw_160 = (*(&kernelContext_0)->vs_0)[_S460];
        float _S461 = fma((*(&kernelContext_0)->ps_0)[_S66], attnLo_0((*(&kernelContext_0)->vs_0)[_S460]), _S443);
        accLo_0[0U] = _S461;
        float _S462 = fma(pj_20, attnHi_0(vw_160), _S444);
        accHi_0[0U] = _S462;
        uint vw_161 = (*(&kernelContext_0)->vs_0)[_S460 + 8U];
        float _S463 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S460 + 8U]), _S445);
        accLo_0[1U] = _S463;
        float _S464 = fma(pj_20, attnHi_0(vw_161), _S446);
        accHi_0[1U] = _S464;
        uint vw_162 = (*(&kernelContext_0)->vs_0)[_S460 + 16U];
        float _S465 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S460 + 16U]), _S447);
        accLo_0[2U] = _S465;
        float _S466 = fma(pj_20, attnHi_0(vw_162), _S448);
        accHi_0[2U] = _S466;
        uint vw_163 = (*(&kernelContext_0)->vs_0)[_S460 + 24U];
        float _S467 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S460 + 24U]), _S449);
        accLo_0[3U] = _S467;
        float _S468 = fma(pj_20, attnHi_0(vw_163), _S450);
        accHi_0[3U] = _S468;
        uint vw_164 = (*(&kernelContext_0)->vs_0)[_S460 + 32U];
        float _S469 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S460 + 32U]), _S451);
        accLo_0[4U] = _S469;
        float _S470 = fma(pj_20, attnHi_0(vw_164), _S452);
        accHi_0[4U] = _S470;
        uint vw_165 = (*(&kernelContext_0)->vs_0)[_S460 + 40U];
        float _S471 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S460 + 40U]), _S453);
        accLo_0[5U] = _S471;
        float _S472 = fma(pj_20, attnHi_0(vw_165), _S454);
        accHi_0[5U] = _S472;
        uint vw_166 = (*(&kernelContext_0)->vs_0)[_S460 + 48U];
        float _S473 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S460 + 48U]), _S455);
        accLo_0[6U] = _S473;
        float _S474 = fma(pj_20, attnHi_0(vw_166), _S456);
        accHi_0[6U] = _S474;
        uint vw_167 = (*(&kernelContext_0)->vs_0)[_S460 + 56U];
        float _S475 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S460 + 56U]), _S457);
        accLo_0[7U] = _S475;
        float _S476 = fma(pj_20, attnHi_0(vw_167), _S458);
        accHi_0[7U] = _S476;
        float pj_21 = (*(&kernelContext_0)->ps_0)[_S67];
        float _S477 = _S459 + (*(&kernelContext_0)->ps_0)[_S67];
        uint _S478 = 1365U + kk_0;
        uint vw_168 = (*(&kernelContext_0)->vs_0)[_S478];
        float _S479 = fma((*(&kernelContext_0)->ps_0)[_S67], attnLo_0((*(&kernelContext_0)->vs_0)[_S478]), _S461);
        accLo_0[0U] = _S479;
        float _S480 = fma(pj_21, attnHi_0(vw_168), _S462);
        accHi_0[0U] = _S480;
        uint vw_169 = (*(&kernelContext_0)->vs_0)[_S478 + 8U];
        float _S481 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S478 + 8U]), _S463);
        accLo_0[1U] = _S481;
        float _S482 = fma(pj_21, attnHi_0(vw_169), _S464);
        accHi_0[1U] = _S482;
        uint vw_170 = (*(&kernelContext_0)->vs_0)[_S478 + 16U];
        float _S483 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S478 + 16U]), _S465);
        accLo_0[2U] = _S483;
        float _S484 = fma(pj_21, attnHi_0(vw_170), _S466);
        accHi_0[2U] = _S484;
        uint vw_171 = (*(&kernelContext_0)->vs_0)[_S478 + 24U];
        float _S485 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S478 + 24U]), _S467);
        accLo_0[3U] = _S485;
        float _S486 = fma(pj_21, attnHi_0(vw_171), _S468);
        accHi_0[3U] = _S486;
        uint vw_172 = (*(&kernelContext_0)->vs_0)[_S478 + 32U];
        float _S487 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S478 + 32U]), _S469);
        accLo_0[4U] = _S487;
        float _S488 = fma(pj_21, attnHi_0(vw_172), _S470);
        accHi_0[4U] = _S488;
        uint vw_173 = (*(&kernelContext_0)->vs_0)[_S478 + 40U];
        float _S489 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S478 + 40U]), _S471);
        accLo_0[5U] = _S489;
        float _S490 = fma(pj_21, attnHi_0(vw_173), _S472);
        accHi_0[5U] = _S490;
        uint vw_174 = (*(&kernelContext_0)->vs_0)[_S478 + 48U];
        float _S491 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S478 + 48U]), _S473);
        accLo_0[6U] = _S491;
        float _S492 = fma(pj_21, attnHi_0(vw_174), _S474);
        accHi_0[6U] = _S492;
        uint vw_175 = (*(&kernelContext_0)->vs_0)[_S478 + 56U];
        float _S493 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S478 + 56U]), _S475);
        accLo_0[7U] = _S493;
        float _S494 = fma(pj_21, attnHi_0(vw_175), _S476);
        accHi_0[7U] = _S494;
        float pj_22 = (*(&kernelContext_0)->ps_0)[_S68];
        float _S495 = _S477 + (*(&kernelContext_0)->ps_0)[_S68];
        uint _S496 = 1430U + kk_0;
        uint vw_176 = (*(&kernelContext_0)->vs_0)[_S496];
        float _S497 = fma((*(&kernelContext_0)->ps_0)[_S68], attnLo_0((*(&kernelContext_0)->vs_0)[_S496]), _S479);
        accLo_0[0U] = _S497;
        float _S498 = fma(pj_22, attnHi_0(vw_176), _S480);
        accHi_0[0U] = _S498;
        uint vw_177 = (*(&kernelContext_0)->vs_0)[_S496 + 8U];
        float _S499 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S496 + 8U]), _S481);
        accLo_0[1U] = _S499;
        float _S500 = fma(pj_22, attnHi_0(vw_177), _S482);
        accHi_0[1U] = _S500;
        uint vw_178 = (*(&kernelContext_0)->vs_0)[_S496 + 16U];
        float _S501 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S496 + 16U]), _S483);
        accLo_0[2U] = _S501;
        float _S502 = fma(pj_22, attnHi_0(vw_178), _S484);
        accHi_0[2U] = _S502;
        uint vw_179 = (*(&kernelContext_0)->vs_0)[_S496 + 24U];
        float _S503 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S496 + 24U]), _S485);
        accLo_0[3U] = _S503;
        float _S504 = fma(pj_22, attnHi_0(vw_179), _S486);
        accHi_0[3U] = _S504;
        uint vw_180 = (*(&kernelContext_0)->vs_0)[_S496 + 32U];
        float _S505 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S496 + 32U]), _S487);
        accLo_0[4U] = _S505;
        float _S506 = fma(pj_22, attnHi_0(vw_180), _S488);
        accHi_0[4U] = _S506;
        uint vw_181 = (*(&kernelContext_0)->vs_0)[_S496 + 40U];
        float _S507 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S496 + 40U]), _S489);
        accLo_0[5U] = _S507;
        float _S508 = fma(pj_22, attnHi_0(vw_181), _S490);
        accHi_0[5U] = _S508;
        uint vw_182 = (*(&kernelContext_0)->vs_0)[_S496 + 48U];
        float _S509 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S496 + 48U]), _S491);
        accLo_0[6U] = _S509;
        float _S510 = fma(pj_22, attnHi_0(vw_182), _S492);
        accHi_0[6U] = _S510;
        uint vw_183 = (*(&kernelContext_0)->vs_0)[_S496 + 56U];
        float _S511 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S496 + 56U]), _S493);
        accLo_0[7U] = _S511;
        float _S512 = fma(pj_22, attnHi_0(vw_183), _S494);
        accHi_0[7U] = _S512;
        float pj_23 = (*(&kernelContext_0)->ps_0)[_S69];
        float _S513 = _S495 + (*(&kernelContext_0)->ps_0)[_S69];
        uint _S514 = 1495U + kk_0;
        uint vw_184 = (*(&kernelContext_0)->vs_0)[_S514];
        float _S515 = fma((*(&kernelContext_0)->ps_0)[_S69], attnLo_0((*(&kernelContext_0)->vs_0)[_S514]), _S497);
        accLo_0[0U] = _S515;
        float _S516 = fma(pj_23, attnHi_0(vw_184), _S498);
        accHi_0[0U] = _S516;
        uint vw_185 = (*(&kernelContext_0)->vs_0)[_S514 + 8U];
        float _S517 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S514 + 8U]), _S499);
        accLo_0[1U] = _S517;
        float _S518 = fma(pj_23, attnHi_0(vw_185), _S500);
        accHi_0[1U] = _S518;
        uint vw_186 = (*(&kernelContext_0)->vs_0)[_S514 + 16U];
        float _S519 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S514 + 16U]), _S501);
        accLo_0[2U] = _S519;
        float _S520 = fma(pj_23, attnHi_0(vw_186), _S502);
        accHi_0[2U] = _S520;
        uint vw_187 = (*(&kernelContext_0)->vs_0)[_S514 + 24U];
        float _S521 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S514 + 24U]), _S503);
        accLo_0[3U] = _S521;
        float _S522 = fma(pj_23, attnHi_0(vw_187), _S504);
        accHi_0[3U] = _S522;
        uint vw_188 = (*(&kernelContext_0)->vs_0)[_S514 + 32U];
        float _S523 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S514 + 32U]), _S505);
        accLo_0[4U] = _S523;
        float _S524 = fma(pj_23, attnHi_0(vw_188), _S506);
        accHi_0[4U] = _S524;
        uint vw_189 = (*(&kernelContext_0)->vs_0)[_S514 + 40U];
        float _S525 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S514 + 40U]), _S507);
        accLo_0[5U] = _S525;
        float _S526 = fma(pj_23, attnHi_0(vw_189), _S508);
        accHi_0[5U] = _S526;
        uint vw_190 = (*(&kernelContext_0)->vs_0)[_S514 + 48U];
        float _S527 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S514 + 48U]), _S509);
        accLo_0[6U] = _S527;
        float _S528 = fma(pj_23, attnHi_0(vw_190), _S510);
        accHi_0[6U] = _S528;
        uint vw_191 = (*(&kernelContext_0)->vs_0)[_S514 + 56U];
        float _S529 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S514 + 56U]), _S511);
        accLo_0[7U] = _S529;
        float _S530 = fma(pj_23, attnHi_0(vw_191), _S512);
        accHi_0[7U] = _S530;
        float pj_24 = (*(&kernelContext_0)->ps_0)[_S70];
        float _S531 = _S513 + (*(&kernelContext_0)->ps_0)[_S70];
        uint _S532 = 1560U + kk_0;
        uint vw_192 = (*(&kernelContext_0)->vs_0)[_S532];
        float _S533 = fma((*(&kernelContext_0)->ps_0)[_S70], attnLo_0((*(&kernelContext_0)->vs_0)[_S532]), _S515);
        accLo_0[0U] = _S533;
        float _S534 = fma(pj_24, attnHi_0(vw_192), _S516);
        accHi_0[0U] = _S534;
        uint vw_193 = (*(&kernelContext_0)->vs_0)[_S532 + 8U];
        float _S535 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S532 + 8U]), _S517);
        accLo_0[1U] = _S535;
        float _S536 = fma(pj_24, attnHi_0(vw_193), _S518);
        accHi_0[1U] = _S536;
        uint vw_194 = (*(&kernelContext_0)->vs_0)[_S532 + 16U];
        float _S537 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S532 + 16U]), _S519);
        accLo_0[2U] = _S537;
        float _S538 = fma(pj_24, attnHi_0(vw_194), _S520);
        accHi_0[2U] = _S538;
        uint vw_195 = (*(&kernelContext_0)->vs_0)[_S532 + 24U];
        float _S539 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S532 + 24U]), _S521);
        accLo_0[3U] = _S539;
        float _S540 = fma(pj_24, attnHi_0(vw_195), _S522);
        accHi_0[3U] = _S540;
        uint vw_196 = (*(&kernelContext_0)->vs_0)[_S532 + 32U];
        float _S541 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S532 + 32U]), _S523);
        accLo_0[4U] = _S541;
        float _S542 = fma(pj_24, attnHi_0(vw_196), _S524);
        accHi_0[4U] = _S542;
        uint vw_197 = (*(&kernelContext_0)->vs_0)[_S532 + 40U];
        float _S543 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S532 + 40U]), _S525);
        accLo_0[5U] = _S543;
        float _S544 = fma(pj_24, attnHi_0(vw_197), _S526);
        accHi_0[5U] = _S544;
        uint vw_198 = (*(&kernelContext_0)->vs_0)[_S532 + 48U];
        float _S545 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S532 + 48U]), _S527);
        accLo_0[6U] = _S545;
        float _S546 = fma(pj_24, attnHi_0(vw_198), _S528);
        accHi_0[6U] = _S546;
        uint vw_199 = (*(&kernelContext_0)->vs_0)[_S532 + 56U];
        float _S547 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S532 + 56U]), _S529);
        accLo_0[7U] = _S547;
        float _S548 = fma(pj_24, attnHi_0(vw_199), _S530);
        accHi_0[7U] = _S548;
        float pj_25 = (*(&kernelContext_0)->ps_0)[_S71];
        float _S549 = _S531 + (*(&kernelContext_0)->ps_0)[_S71];
        uint _S550 = 1625U + kk_0;
        uint vw_200 = (*(&kernelContext_0)->vs_0)[_S550];
        float _S551 = fma((*(&kernelContext_0)->ps_0)[_S71], attnLo_0((*(&kernelContext_0)->vs_0)[_S550]), _S533);
        accLo_0[0U] = _S551;
        float _S552 = fma(pj_25, attnHi_0(vw_200), _S534);
        accHi_0[0U] = _S552;
        uint vw_201 = (*(&kernelContext_0)->vs_0)[_S550 + 8U];
        float _S553 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S550 + 8U]), _S535);
        accLo_0[1U] = _S553;
        float _S554 = fma(pj_25, attnHi_0(vw_201), _S536);
        accHi_0[1U] = _S554;
        uint vw_202 = (*(&kernelContext_0)->vs_0)[_S550 + 16U];
        float _S555 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S550 + 16U]), _S537);
        accLo_0[2U] = _S555;
        float _S556 = fma(pj_25, attnHi_0(vw_202), _S538);
        accHi_0[2U] = _S556;
        uint vw_203 = (*(&kernelContext_0)->vs_0)[_S550 + 24U];
        float _S557 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S550 + 24U]), _S539);
        accLo_0[3U] = _S557;
        float _S558 = fma(pj_25, attnHi_0(vw_203), _S540);
        accHi_0[3U] = _S558;
        uint vw_204 = (*(&kernelContext_0)->vs_0)[_S550 + 32U];
        float _S559 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S550 + 32U]), _S541);
        accLo_0[4U] = _S559;
        float _S560 = fma(pj_25, attnHi_0(vw_204), _S542);
        accHi_0[4U] = _S560;
        uint vw_205 = (*(&kernelContext_0)->vs_0)[_S550 + 40U];
        float _S561 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S550 + 40U]), _S543);
        accLo_0[5U] = _S561;
        float _S562 = fma(pj_25, attnHi_0(vw_205), _S544);
        accHi_0[5U] = _S562;
        uint vw_206 = (*(&kernelContext_0)->vs_0)[_S550 + 48U];
        float _S563 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S550 + 48U]), _S545);
        accLo_0[6U] = _S563;
        float _S564 = fma(pj_25, attnHi_0(vw_206), _S546);
        accHi_0[6U] = _S564;
        uint vw_207 = (*(&kernelContext_0)->vs_0)[_S550 + 56U];
        float _S565 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S550 + 56U]), _S547);
        accLo_0[7U] = _S565;
        float _S566 = fma(pj_25, attnHi_0(vw_207), _S548);
        accHi_0[7U] = _S566;
        float pj_26 = (*(&kernelContext_0)->ps_0)[_S72];
        float _S567 = _S549 + (*(&kernelContext_0)->ps_0)[_S72];
        uint _S568 = 1690U + kk_0;
        uint vw_208 = (*(&kernelContext_0)->vs_0)[_S568];
        float _S569 = fma((*(&kernelContext_0)->ps_0)[_S72], attnLo_0((*(&kernelContext_0)->vs_0)[_S568]), _S551);
        accLo_0[0U] = _S569;
        float _S570 = fma(pj_26, attnHi_0(vw_208), _S552);
        accHi_0[0U] = _S570;
        uint vw_209 = (*(&kernelContext_0)->vs_0)[_S568 + 8U];
        float _S571 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S568 + 8U]), _S553);
        accLo_0[1U] = _S571;
        float _S572 = fma(pj_26, attnHi_0(vw_209), _S554);
        accHi_0[1U] = _S572;
        uint vw_210 = (*(&kernelContext_0)->vs_0)[_S568 + 16U];
        float _S573 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S568 + 16U]), _S555);
        accLo_0[2U] = _S573;
        float _S574 = fma(pj_26, attnHi_0(vw_210), _S556);
        accHi_0[2U] = _S574;
        uint vw_211 = (*(&kernelContext_0)->vs_0)[_S568 + 24U];
        float _S575 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S568 + 24U]), _S557);
        accLo_0[3U] = _S575;
        float _S576 = fma(pj_26, attnHi_0(vw_211), _S558);
        accHi_0[3U] = _S576;
        uint vw_212 = (*(&kernelContext_0)->vs_0)[_S568 + 32U];
        float _S577 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S568 + 32U]), _S559);
        accLo_0[4U] = _S577;
        float _S578 = fma(pj_26, attnHi_0(vw_212), _S560);
        accHi_0[4U] = _S578;
        uint vw_213 = (*(&kernelContext_0)->vs_0)[_S568 + 40U];
        float _S579 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S568 + 40U]), _S561);
        accLo_0[5U] = _S579;
        float _S580 = fma(pj_26, attnHi_0(vw_213), _S562);
        accHi_0[5U] = _S580;
        uint vw_214 = (*(&kernelContext_0)->vs_0)[_S568 + 48U];
        float _S581 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S568 + 48U]), _S563);
        accLo_0[6U] = _S581;
        float _S582 = fma(pj_26, attnHi_0(vw_214), _S564);
        accHi_0[6U] = _S582;
        uint vw_215 = (*(&kernelContext_0)->vs_0)[_S568 + 56U];
        float _S583 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S568 + 56U]), _S565);
        accLo_0[7U] = _S583;
        float _S584 = fma(pj_26, attnHi_0(vw_215), _S566);
        accHi_0[7U] = _S584;
        float pj_27 = (*(&kernelContext_0)->ps_0)[_S73];
        float _S585 = _S567 + (*(&kernelContext_0)->ps_0)[_S73];
        uint _S586 = 1755U + kk_0;
        uint vw_216 = (*(&kernelContext_0)->vs_0)[_S586];
        float _S587 = fma((*(&kernelContext_0)->ps_0)[_S73], attnLo_0((*(&kernelContext_0)->vs_0)[_S586]), _S569);
        accLo_0[0U] = _S587;
        float _S588 = fma(pj_27, attnHi_0(vw_216), _S570);
        accHi_0[0U] = _S588;
        uint vw_217 = (*(&kernelContext_0)->vs_0)[_S586 + 8U];
        float _S589 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S586 + 8U]), _S571);
        accLo_0[1U] = _S589;
        float _S590 = fma(pj_27, attnHi_0(vw_217), _S572);
        accHi_0[1U] = _S590;
        uint vw_218 = (*(&kernelContext_0)->vs_0)[_S586 + 16U];
        float _S591 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S586 + 16U]), _S573);
        accLo_0[2U] = _S591;
        float _S592 = fma(pj_27, attnHi_0(vw_218), _S574);
        accHi_0[2U] = _S592;
        uint vw_219 = (*(&kernelContext_0)->vs_0)[_S586 + 24U];
        float _S593 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S586 + 24U]), _S575);
        accLo_0[3U] = _S593;
        float _S594 = fma(pj_27, attnHi_0(vw_219), _S576);
        accHi_0[3U] = _S594;
        uint vw_220 = (*(&kernelContext_0)->vs_0)[_S586 + 32U];
        float _S595 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S586 + 32U]), _S577);
        accLo_0[4U] = _S595;
        float _S596 = fma(pj_27, attnHi_0(vw_220), _S578);
        accHi_0[4U] = _S596;
        uint vw_221 = (*(&kernelContext_0)->vs_0)[_S586 + 40U];
        float _S597 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S586 + 40U]), _S579);
        accLo_0[5U] = _S597;
        float _S598 = fma(pj_27, attnHi_0(vw_221), _S580);
        accHi_0[5U] = _S598;
        uint vw_222 = (*(&kernelContext_0)->vs_0)[_S586 + 48U];
        float _S599 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S586 + 48U]), _S581);
        accLo_0[6U] = _S599;
        float _S600 = fma(pj_27, attnHi_0(vw_222), _S582);
        accHi_0[6U] = _S600;
        uint vw_223 = (*(&kernelContext_0)->vs_0)[_S586 + 56U];
        float _S601 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S586 + 56U]), _S583);
        accLo_0[7U] = _S601;
        float _S602 = fma(pj_27, attnHi_0(vw_223), _S584);
        accHi_0[7U] = _S602;
        float pj_28 = (*(&kernelContext_0)->ps_0)[_S74];
        float _S603 = _S585 + (*(&kernelContext_0)->ps_0)[_S74];
        uint _S604 = 1820U + kk_0;
        uint vw_224 = (*(&kernelContext_0)->vs_0)[_S604];
        float _S605 = fma((*(&kernelContext_0)->ps_0)[_S74], attnLo_0((*(&kernelContext_0)->vs_0)[_S604]), _S587);
        accLo_0[0U] = _S605;
        float _S606 = fma(pj_28, attnHi_0(vw_224), _S588);
        accHi_0[0U] = _S606;
        uint vw_225 = (*(&kernelContext_0)->vs_0)[_S604 + 8U];
        float _S607 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S604 + 8U]), _S589);
        accLo_0[1U] = _S607;
        float _S608 = fma(pj_28, attnHi_0(vw_225), _S590);
        accHi_0[1U] = _S608;
        uint vw_226 = (*(&kernelContext_0)->vs_0)[_S604 + 16U];
        float _S609 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S604 + 16U]), _S591);
        accLo_0[2U] = _S609;
        float _S610 = fma(pj_28, attnHi_0(vw_226), _S592);
        accHi_0[2U] = _S610;
        uint vw_227 = (*(&kernelContext_0)->vs_0)[_S604 + 24U];
        float _S611 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S604 + 24U]), _S593);
        accLo_0[3U] = _S611;
        float _S612 = fma(pj_28, attnHi_0(vw_227), _S594);
        accHi_0[3U] = _S612;
        uint vw_228 = (*(&kernelContext_0)->vs_0)[_S604 + 32U];
        float _S613 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S604 + 32U]), _S595);
        accLo_0[4U] = _S613;
        float _S614 = fma(pj_28, attnHi_0(vw_228), _S596);
        accHi_0[4U] = _S614;
        uint vw_229 = (*(&kernelContext_0)->vs_0)[_S604 + 40U];
        float _S615 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S604 + 40U]), _S597);
        accLo_0[5U] = _S615;
        float _S616 = fma(pj_28, attnHi_0(vw_229), _S598);
        accHi_0[5U] = _S616;
        uint vw_230 = (*(&kernelContext_0)->vs_0)[_S604 + 48U];
        float _S617 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S604 + 48U]), _S599);
        accLo_0[6U] = _S617;
        float _S618 = fma(pj_28, attnHi_0(vw_230), _S600);
        accHi_0[6U] = _S618;
        uint vw_231 = (*(&kernelContext_0)->vs_0)[_S604 + 56U];
        float _S619 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S604 + 56U]), _S601);
        accLo_0[7U] = _S619;
        float _S620 = fma(pj_28, attnHi_0(vw_231), _S602);
        accHi_0[7U] = _S620;
        float pj_29 = (*(&kernelContext_0)->ps_0)[_S75];
        float _S621 = _S603 + (*(&kernelContext_0)->ps_0)[_S75];
        uint _S622 = 1885U + kk_0;
        uint vw_232 = (*(&kernelContext_0)->vs_0)[_S622];
        float _S623 = fma((*(&kernelContext_0)->ps_0)[_S75], attnLo_0((*(&kernelContext_0)->vs_0)[_S622]), _S605);
        accLo_0[0U] = _S623;
        float _S624 = fma(pj_29, attnHi_0(vw_232), _S606);
        accHi_0[0U] = _S624;
        uint vw_233 = (*(&kernelContext_0)->vs_0)[_S622 + 8U];
        float _S625 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S622 + 8U]), _S607);
        accLo_0[1U] = _S625;
        float _S626 = fma(pj_29, attnHi_0(vw_233), _S608);
        accHi_0[1U] = _S626;
        uint vw_234 = (*(&kernelContext_0)->vs_0)[_S622 + 16U];
        float _S627 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S622 + 16U]), _S609);
        accLo_0[2U] = _S627;
        float _S628 = fma(pj_29, attnHi_0(vw_234), _S610);
        accHi_0[2U] = _S628;
        uint vw_235 = (*(&kernelContext_0)->vs_0)[_S622 + 24U];
        float _S629 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S622 + 24U]), _S611);
        accLo_0[3U] = _S629;
        float _S630 = fma(pj_29, attnHi_0(vw_235), _S612);
        accHi_0[3U] = _S630;
        uint vw_236 = (*(&kernelContext_0)->vs_0)[_S622 + 32U];
        float _S631 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S622 + 32U]), _S613);
        accLo_0[4U] = _S631;
        float _S632 = fma(pj_29, attnHi_0(vw_236), _S614);
        accHi_0[4U] = _S632;
        uint vw_237 = (*(&kernelContext_0)->vs_0)[_S622 + 40U];
        float _S633 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S622 + 40U]), _S615);
        accLo_0[5U] = _S633;
        float _S634 = fma(pj_29, attnHi_0(vw_237), _S616);
        accHi_0[5U] = _S634;
        uint vw_238 = (*(&kernelContext_0)->vs_0)[_S622 + 48U];
        float _S635 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S622 + 48U]), _S617);
        accLo_0[6U] = _S635;
        float _S636 = fma(pj_29, attnHi_0(vw_238), _S618);
        accHi_0[6U] = _S636;
        uint vw_239 = (*(&kernelContext_0)->vs_0)[_S622 + 56U];
        float _S637 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S622 + 56U]), _S619);
        accLo_0[7U] = _S637;
        float _S638 = fma(pj_29, attnHi_0(vw_239), _S620);
        accHi_0[7U] = _S638;
        float pj_30 = (*(&kernelContext_0)->ps_0)[_S76];
        float _S639 = _S621 + (*(&kernelContext_0)->ps_0)[_S76];
        uint _S640 = 1950U + kk_0;
        uint vw_240 = (*(&kernelContext_0)->vs_0)[_S640];
        float _S641 = fma((*(&kernelContext_0)->ps_0)[_S76], attnLo_0((*(&kernelContext_0)->vs_0)[_S640]), _S623);
        accLo_0[0U] = _S641;
        float _S642 = fma(pj_30, attnHi_0(vw_240), _S624);
        accHi_0[0U] = _S642;
        uint vw_241 = (*(&kernelContext_0)->vs_0)[_S640 + 8U];
        float _S643 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S640 + 8U]), _S625);
        accLo_0[1U] = _S643;
        float _S644 = fma(pj_30, attnHi_0(vw_241), _S626);
        accHi_0[1U] = _S644;
        uint vw_242 = (*(&kernelContext_0)->vs_0)[_S640 + 16U];
        float _S645 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S640 + 16U]), _S627);
        accLo_0[2U] = _S645;
        float _S646 = fma(pj_30, attnHi_0(vw_242), _S628);
        accHi_0[2U] = _S646;
        uint vw_243 = (*(&kernelContext_0)->vs_0)[_S640 + 24U];
        float _S647 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S640 + 24U]), _S629);
        accLo_0[3U] = _S647;
        float _S648 = fma(pj_30, attnHi_0(vw_243), _S630);
        accHi_0[3U] = _S648;
        uint vw_244 = (*(&kernelContext_0)->vs_0)[_S640 + 32U];
        float _S649 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S640 + 32U]), _S631);
        accLo_0[4U] = _S649;
        float _S650 = fma(pj_30, attnHi_0(vw_244), _S632);
        accHi_0[4U] = _S650;
        uint vw_245 = (*(&kernelContext_0)->vs_0)[_S640 + 40U];
        float _S651 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S640 + 40U]), _S633);
        accLo_0[5U] = _S651;
        float _S652 = fma(pj_30, attnHi_0(vw_245), _S634);
        accHi_0[5U] = _S652;
        uint vw_246 = (*(&kernelContext_0)->vs_0)[_S640 + 48U];
        float _S653 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S640 + 48U]), _S635);
        accLo_0[6U] = _S653;
        float _S654 = fma(pj_30, attnHi_0(vw_246), _S636);
        accHi_0[6U] = _S654;
        uint vw_247 = (*(&kernelContext_0)->vs_0)[_S640 + 56U];
        float _S655 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S640 + 56U]), _S637);
        accLo_0[7U] = _S655;
        float _S656 = fma(pj_30, attnHi_0(vw_247), _S638);
        accHi_0[7U] = _S656;
        float pj_31 = (*(&kernelContext_0)->ps_0)[_S77];
        float l_1 = _S639 + (*(&kernelContext_0)->ps_0)[_S77];
        uint _S657 = 2015U + kk_0;
        uint vw_248 = (*(&kernelContext_0)->vs_0)[_S657];
        accLo_0[0U] = fma((*(&kernelContext_0)->ps_0)[_S77], attnLo_0((*(&kernelContext_0)->vs_0)[_S657]), _S641);
        accHi_0[0U] = fma(pj_31, attnHi_0(vw_248), _S642);
        uint vw_249 = (*(&kernelContext_0)->vs_0)[_S657 + 8U];
        accLo_0[1U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S657 + 8U]), _S643);
        accHi_0[1U] = fma(pj_31, attnHi_0(vw_249), _S644);
        uint vw_250 = (*(&kernelContext_0)->vs_0)[_S657 + 16U];
        accLo_0[2U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S657 + 16U]), _S645);
        accHi_0[2U] = fma(pj_31, attnHi_0(vw_250), _S646);
        uint vw_251 = (*(&kernelContext_0)->vs_0)[_S657 + 24U];
        accLo_0[3U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S657 + 24U]), _S647);
        accHi_0[3U] = fma(pj_31, attnHi_0(vw_251), _S648);
        uint vw_252 = (*(&kernelContext_0)->vs_0)[_S657 + 32U];
        accLo_0[4U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S657 + 32U]), _S649);
        accHi_0[4U] = fma(pj_31, attnHi_0(vw_252), _S650);
        uint vw_253 = (*(&kernelContext_0)->vs_0)[_S657 + 40U];
        accLo_0[5U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S657 + 40U]), _S651);
        accHi_0[5U] = fma(pj_31, attnHi_0(vw_253), _S652);
        uint vw_254 = (*(&kernelContext_0)->vs_0)[_S657 + 48U];
        accLo_0[6U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S657 + 48U]), _S653);
        accHi_0[6U] = fma(pj_31, attnHi_0(vw_254), _S654);
        uint vw_255 = (*(&kernelContext_0)->vs_0)[_S657 + 56U];
        accLo_0[7U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S657 + 56U]), _S655);
        accHi_0[7U] = fma(pj_31, attnHi_0(vw_255), _S656);
        b_0 = b_0 + 32U;
        m_0 = _S78;
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
            uint _S658 = 2U * pr_30;
            *((&kernelContext_0)->partialML_0+_S658) = m_0;
            *((&kernelContext_0)->partialML_0+(_S658 + 1U)) = l_0;
        }
        if(kk_0 < _S16)
        {
            uint _S659 = 2U * kk_0;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S659)) = accLo_0[0U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S659 + 1U)) = accHi_0[0U];
        }
        uint pair_0 = kk_0 + 8U;
        if(pair_0 < _S16)
        {
            uint _S660 = 2U * pair_0;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S660)) = accLo_0[1U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S660 + 1U)) = accHi_0[1U];
        }
        uint pair_1 = kk_0 + 16U;
        if(pair_1 < _S16)
        {
            uint _S661 = 2U * pair_1;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S661)) = accLo_0[2U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S661 + 1U)) = accHi_0[2U];
        }
        uint pair_2 = kk_0 + 24U;
        if(pair_2 < _S16)
        {
            uint _S662 = 2U * pair_2;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S662)) = accLo_0[3U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S662 + 1U)) = accHi_0[3U];
        }
        uint pair_3 = kk_0 + 32U;
        if(pair_3 < _S16)
        {
            uint _S663 = 2U * pair_3;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S663)) = accLo_0[4U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S663 + 1U)) = accHi_0[4U];
        }
        uint pair_4 = kk_0 + 40U;
        if(pair_4 < _S16)
        {
            uint _S664 = 2U * pair_4;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S664)) = accLo_0[5U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S664 + 1U)) = accHi_0[5U];
        }
        uint pair_5 = kk_0 + 48U;
        if(pair_5 < _S16)
        {
            uint _S665 = 2U * pair_5;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S665)) = accLo_0[6U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S665 + 1U)) = accHi_0[6U];
        }
        uint pair_6 = kk_0 + 56U;
        if(pair_6 < _S16)
        {
            uint _S666 = 2U * pair_6;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S666)) = accLo_0[7U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S666 + 1U)) = accHi_0[7U];
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
        *((&kernelContext_0)->lse_0+row_1) = m_0 + log(l_0);
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                if(kk_0 >= _S16)
                {
                    break;
                }
                float hi_0 = accHi_0[0U] / l_0;
                uint _S667 = 2U * kk_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S667)) = accLo_0[0U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S667 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_7 = kk_0 + 8U;
                if(pair_7 >= _S16)
                {
                    break;
                }
                float hi_1 = accHi_0[1U] / l_0;
                uint _S668 = 2U * pair_7;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S668)) = accLo_0[1U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S668 + 1U)) = hi_1;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_8 = kk_0 + 16U;
                if(pair_8 >= _S16)
                {
                    break;
                }
                float hi_2 = accHi_0[2U] / l_0;
                uint _S669 = 2U * pair_8;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S669)) = accLo_0[2U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S669 + 1U)) = hi_2;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_9 = kk_0 + 24U;
                if(pair_9 >= _S16)
                {
                    break;
                }
                float hi_3 = accHi_0[3U] / l_0;
                uint _S670 = 2U * pair_9;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S670)) = accLo_0[3U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S670 + 1U)) = hi_3;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_10 = kk_0 + 32U;
                if(pair_10 >= _S16)
                {
                    break;
                }
                float hi_4 = accHi_0[4U] / l_0;
                uint _S671 = 2U * pair_10;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S671)) = accLo_0[4U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S671 + 1U)) = hi_4;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_11 = kk_0 + 40U;
                if(pair_11 >= _S16)
                {
                    break;
                }
                float hi_5 = accHi_0[5U] / l_0;
                uint _S672 = 2U * pair_11;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S672)) = accLo_0[5U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S672 + 1U)) = hi_5;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_12 = kk_0 + 48U;
                if(pair_12 >= _S16)
                {
                    break;
                }
                float hi_6 = accHi_0[6U] / l_0;
                uint _S673 = 2U * pair_12;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S673)) = accLo_0[6U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S673 + 1U)) = hi_6;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_13 = kk_0 + 56U;
                if(pair_13 >= _S16)
                {
                    break;
                }
                float hi_7 = accHi_0[7U] / l_0;
                uint _S674 = 2U * pair_13;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S674)) = accLo_0[7U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S674 + 1U)) = hi_7;
                break;
            }
            break;
        }
        break;
    }
    return;
}

