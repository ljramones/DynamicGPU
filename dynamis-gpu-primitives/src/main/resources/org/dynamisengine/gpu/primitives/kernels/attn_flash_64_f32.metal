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
    float device* o_0;
    array<uint, int(528)> threadgroup* qs_0;
    array<uint, int(1056)> threadgroup* ks_0;
    array<uint, int(1056)> threadgroup* vs_0;
    array<float, int(512)> threadgroup* ss_0;
    array<float, int(512)> threadgroup* ps_0;
};

[[kernel]] void attn_flash_64_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(7)]], uint device* q_1 [[buffer(0)]], uint device* table_1 [[buffer(6)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partialML_1 [[buffer(4)]], float device* partialAcc_1 [[buffer(5)]], float device* o_1 [[buffer(3)]])
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
        (*(&kernelContext_0)->qs_0)[e_0 / 32U * 33U + pr_0] = w_2;
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
            if(e_1 < 1024U)
            {
            }
            else
            {
                break;
            }
            uint j_0 = e_1 / 32U;
            uint pr_1 = e_1 % 32U;
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
            uint _S26 = j_0 * 33U + pr_1;
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
                        if(at_0 < 32U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S27 = lr_0 * 33U;
                        uint _S28 = kk_0 * 33U;
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
                        if(p0_1 < 32U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S32 = lr_0 * 33U;
                        uint _S33 = j_1 * 33U;
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
                        if(p0_0 < 32U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S37 = lr_0 * 33U;
                        uint _S38 = j_2 * 33U;
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
                        if(p0_2 < 32U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint _S42 = lr_0 * 33U;
                        uint _S43 = j_3 * 33U;
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
        float pj_0 = (*(&kernelContext_0)->ps_0)[_S1];
        float _S92 = _S83 + (*(&kernelContext_0)->ps_0)[_S1];
        uint vw_0 = (*(&kernelContext_0)->vs_0)[kk_0];
        float _S93 = fma((*(&kernelContext_0)->ps_0)[_S1], attnLo_0((*(&kernelContext_0)->vs_0)[kk_0]), _S84);
        accLo_0[0U] = _S93;
        float _S94 = fma(pj_0, attnHi_0(vw_0), _S85);
        accHi_0[0U] = _S94;
        uint vw_1 = (*(&kernelContext_0)->vs_0)[_S3];
        float _S95 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S3]), _S86);
        accLo_0[1U] = _S95;
        float _S96 = fma(pj_0, attnHi_0(vw_1), _S87);
        accHi_0[1U] = _S96;
        uint vw_2 = (*(&kernelContext_0)->vs_0)[_S5];
        float _S97 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S5]), _S88);
        accLo_0[2U] = _S97;
        float _S98 = fma(pj_0, attnHi_0(vw_2), _S89);
        accHi_0[2U] = _S98;
        uint vw_3 = (*(&kernelContext_0)->vs_0)[_S7];
        float _S99 = fma(pj_0, attnLo_0((*(&kernelContext_0)->vs_0)[_S7]), _S90);
        accLo_0[3U] = _S99;
        float _S100 = fma(pj_0, attnHi_0(vw_3), _S91);
        accHi_0[3U] = _S100;
        float pj_1 = (*(&kernelContext_0)->ps_0)[_S47];
        float _S101 = _S92 + (*(&kernelContext_0)->ps_0)[_S47];
        uint _S102 = 33U + kk_0;
        uint vw_4 = (*(&kernelContext_0)->vs_0)[_S102];
        float _S103 = fma((*(&kernelContext_0)->ps_0)[_S47], attnLo_0((*(&kernelContext_0)->vs_0)[_S102]), _S93);
        accLo_0[0U] = _S103;
        float _S104 = fma(pj_1, attnHi_0(vw_4), _S94);
        accHi_0[0U] = _S104;
        uint vw_5 = (*(&kernelContext_0)->vs_0)[_S102 + 8U];
        float _S105 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S102 + 8U]), _S95);
        accLo_0[1U] = _S105;
        float _S106 = fma(pj_1, attnHi_0(vw_5), _S96);
        accHi_0[1U] = _S106;
        uint vw_6 = (*(&kernelContext_0)->vs_0)[_S102 + 16U];
        float _S107 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S102 + 16U]), _S97);
        accLo_0[2U] = _S107;
        float _S108 = fma(pj_1, attnHi_0(vw_6), _S98);
        accHi_0[2U] = _S108;
        uint vw_7 = (*(&kernelContext_0)->vs_0)[_S102 + 24U];
        float _S109 = fma(pj_1, attnLo_0((*(&kernelContext_0)->vs_0)[_S102 + 24U]), _S99);
        accLo_0[3U] = _S109;
        float _S110 = fma(pj_1, attnHi_0(vw_7), _S100);
        accHi_0[3U] = _S110;
        float pj_2 = (*(&kernelContext_0)->ps_0)[_S48];
        float _S111 = _S101 + (*(&kernelContext_0)->ps_0)[_S48];
        uint _S112 = 66U + kk_0;
        uint vw_8 = (*(&kernelContext_0)->vs_0)[_S112];
        float _S113 = fma((*(&kernelContext_0)->ps_0)[_S48], attnLo_0((*(&kernelContext_0)->vs_0)[_S112]), _S103);
        accLo_0[0U] = _S113;
        float _S114 = fma(pj_2, attnHi_0(vw_8), _S104);
        accHi_0[0U] = _S114;
        uint vw_9 = (*(&kernelContext_0)->vs_0)[_S112 + 8U];
        float _S115 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S112 + 8U]), _S105);
        accLo_0[1U] = _S115;
        float _S116 = fma(pj_2, attnHi_0(vw_9), _S106);
        accHi_0[1U] = _S116;
        uint vw_10 = (*(&kernelContext_0)->vs_0)[_S112 + 16U];
        float _S117 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S112 + 16U]), _S107);
        accLo_0[2U] = _S117;
        float _S118 = fma(pj_2, attnHi_0(vw_10), _S108);
        accHi_0[2U] = _S118;
        uint vw_11 = (*(&kernelContext_0)->vs_0)[_S112 + 24U];
        float _S119 = fma(pj_2, attnLo_0((*(&kernelContext_0)->vs_0)[_S112 + 24U]), _S109);
        accLo_0[3U] = _S119;
        float _S120 = fma(pj_2, attnHi_0(vw_11), _S110);
        accHi_0[3U] = _S120;
        float pj_3 = (*(&kernelContext_0)->ps_0)[_S49];
        float _S121 = _S111 + (*(&kernelContext_0)->ps_0)[_S49];
        uint _S122 = 99U + kk_0;
        uint vw_12 = (*(&kernelContext_0)->vs_0)[_S122];
        float _S123 = fma((*(&kernelContext_0)->ps_0)[_S49], attnLo_0((*(&kernelContext_0)->vs_0)[_S122]), _S113);
        accLo_0[0U] = _S123;
        float _S124 = fma(pj_3, attnHi_0(vw_12), _S114);
        accHi_0[0U] = _S124;
        uint vw_13 = (*(&kernelContext_0)->vs_0)[_S122 + 8U];
        float _S125 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S122 + 8U]), _S115);
        accLo_0[1U] = _S125;
        float _S126 = fma(pj_3, attnHi_0(vw_13), _S116);
        accHi_0[1U] = _S126;
        uint vw_14 = (*(&kernelContext_0)->vs_0)[_S122 + 16U];
        float _S127 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S122 + 16U]), _S117);
        accLo_0[2U] = _S127;
        float _S128 = fma(pj_3, attnHi_0(vw_14), _S118);
        accHi_0[2U] = _S128;
        uint vw_15 = (*(&kernelContext_0)->vs_0)[_S122 + 24U];
        float _S129 = fma(pj_3, attnLo_0((*(&kernelContext_0)->vs_0)[_S122 + 24U]), _S119);
        accLo_0[3U] = _S129;
        float _S130 = fma(pj_3, attnHi_0(vw_15), _S120);
        accHi_0[3U] = _S130;
        float pj_4 = (*(&kernelContext_0)->ps_0)[_S50];
        float _S131 = _S121 + (*(&kernelContext_0)->ps_0)[_S50];
        uint _S132 = 132U + kk_0;
        uint vw_16 = (*(&kernelContext_0)->vs_0)[_S132];
        float _S133 = fma((*(&kernelContext_0)->ps_0)[_S50], attnLo_0((*(&kernelContext_0)->vs_0)[_S132]), _S123);
        accLo_0[0U] = _S133;
        float _S134 = fma(pj_4, attnHi_0(vw_16), _S124);
        accHi_0[0U] = _S134;
        uint vw_17 = (*(&kernelContext_0)->vs_0)[_S132 + 8U];
        float _S135 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S132 + 8U]), _S125);
        accLo_0[1U] = _S135;
        float _S136 = fma(pj_4, attnHi_0(vw_17), _S126);
        accHi_0[1U] = _S136;
        uint vw_18 = (*(&kernelContext_0)->vs_0)[_S132 + 16U];
        float _S137 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S132 + 16U]), _S127);
        accLo_0[2U] = _S137;
        float _S138 = fma(pj_4, attnHi_0(vw_18), _S128);
        accHi_0[2U] = _S138;
        uint vw_19 = (*(&kernelContext_0)->vs_0)[_S132 + 24U];
        float _S139 = fma(pj_4, attnLo_0((*(&kernelContext_0)->vs_0)[_S132 + 24U]), _S129);
        accLo_0[3U] = _S139;
        float _S140 = fma(pj_4, attnHi_0(vw_19), _S130);
        accHi_0[3U] = _S140;
        float pj_5 = (*(&kernelContext_0)->ps_0)[_S51];
        float _S141 = _S131 + (*(&kernelContext_0)->ps_0)[_S51];
        uint _S142 = 165U + kk_0;
        uint vw_20 = (*(&kernelContext_0)->vs_0)[_S142];
        float _S143 = fma((*(&kernelContext_0)->ps_0)[_S51], attnLo_0((*(&kernelContext_0)->vs_0)[_S142]), _S133);
        accLo_0[0U] = _S143;
        float _S144 = fma(pj_5, attnHi_0(vw_20), _S134);
        accHi_0[0U] = _S144;
        uint vw_21 = (*(&kernelContext_0)->vs_0)[_S142 + 8U];
        float _S145 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S142 + 8U]), _S135);
        accLo_0[1U] = _S145;
        float _S146 = fma(pj_5, attnHi_0(vw_21), _S136);
        accHi_0[1U] = _S146;
        uint vw_22 = (*(&kernelContext_0)->vs_0)[_S142 + 16U];
        float _S147 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S142 + 16U]), _S137);
        accLo_0[2U] = _S147;
        float _S148 = fma(pj_5, attnHi_0(vw_22), _S138);
        accHi_0[2U] = _S148;
        uint vw_23 = (*(&kernelContext_0)->vs_0)[_S142 + 24U];
        float _S149 = fma(pj_5, attnLo_0((*(&kernelContext_0)->vs_0)[_S142 + 24U]), _S139);
        accLo_0[3U] = _S149;
        float _S150 = fma(pj_5, attnHi_0(vw_23), _S140);
        accHi_0[3U] = _S150;
        float pj_6 = (*(&kernelContext_0)->ps_0)[_S52];
        float _S151 = _S141 + (*(&kernelContext_0)->ps_0)[_S52];
        uint _S152 = 198U + kk_0;
        uint vw_24 = (*(&kernelContext_0)->vs_0)[_S152];
        float _S153 = fma((*(&kernelContext_0)->ps_0)[_S52], attnLo_0((*(&kernelContext_0)->vs_0)[_S152]), _S143);
        accLo_0[0U] = _S153;
        float _S154 = fma(pj_6, attnHi_0(vw_24), _S144);
        accHi_0[0U] = _S154;
        uint vw_25 = (*(&kernelContext_0)->vs_0)[_S152 + 8U];
        float _S155 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S152 + 8U]), _S145);
        accLo_0[1U] = _S155;
        float _S156 = fma(pj_6, attnHi_0(vw_25), _S146);
        accHi_0[1U] = _S156;
        uint vw_26 = (*(&kernelContext_0)->vs_0)[_S152 + 16U];
        float _S157 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S152 + 16U]), _S147);
        accLo_0[2U] = _S157;
        float _S158 = fma(pj_6, attnHi_0(vw_26), _S148);
        accHi_0[2U] = _S158;
        uint vw_27 = (*(&kernelContext_0)->vs_0)[_S152 + 24U];
        float _S159 = fma(pj_6, attnLo_0((*(&kernelContext_0)->vs_0)[_S152 + 24U]), _S149);
        accLo_0[3U] = _S159;
        float _S160 = fma(pj_6, attnHi_0(vw_27), _S150);
        accHi_0[3U] = _S160;
        float pj_7 = (*(&kernelContext_0)->ps_0)[_S53];
        float _S161 = _S151 + (*(&kernelContext_0)->ps_0)[_S53];
        uint _S162 = 231U + kk_0;
        uint vw_28 = (*(&kernelContext_0)->vs_0)[_S162];
        float _S163 = fma((*(&kernelContext_0)->ps_0)[_S53], attnLo_0((*(&kernelContext_0)->vs_0)[_S162]), _S153);
        accLo_0[0U] = _S163;
        float _S164 = fma(pj_7, attnHi_0(vw_28), _S154);
        accHi_0[0U] = _S164;
        uint vw_29 = (*(&kernelContext_0)->vs_0)[_S162 + 8U];
        float _S165 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S162 + 8U]), _S155);
        accLo_0[1U] = _S165;
        float _S166 = fma(pj_7, attnHi_0(vw_29), _S156);
        accHi_0[1U] = _S166;
        uint vw_30 = (*(&kernelContext_0)->vs_0)[_S162 + 16U];
        float _S167 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S162 + 16U]), _S157);
        accLo_0[2U] = _S167;
        float _S168 = fma(pj_7, attnHi_0(vw_30), _S158);
        accHi_0[2U] = _S168;
        uint vw_31 = (*(&kernelContext_0)->vs_0)[_S162 + 24U];
        float _S169 = fma(pj_7, attnLo_0((*(&kernelContext_0)->vs_0)[_S162 + 24U]), _S159);
        accLo_0[3U] = _S169;
        float _S170 = fma(pj_7, attnHi_0(vw_31), _S160);
        accHi_0[3U] = _S170;
        float pj_8 = (*(&kernelContext_0)->ps_0)[_S54];
        float _S171 = _S161 + (*(&kernelContext_0)->ps_0)[_S54];
        uint _S172 = 264U + kk_0;
        uint vw_32 = (*(&kernelContext_0)->vs_0)[_S172];
        float _S173 = fma((*(&kernelContext_0)->ps_0)[_S54], attnLo_0((*(&kernelContext_0)->vs_0)[_S172]), _S163);
        accLo_0[0U] = _S173;
        float _S174 = fma(pj_8, attnHi_0(vw_32), _S164);
        accHi_0[0U] = _S174;
        uint vw_33 = (*(&kernelContext_0)->vs_0)[_S172 + 8U];
        float _S175 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S172 + 8U]), _S165);
        accLo_0[1U] = _S175;
        float _S176 = fma(pj_8, attnHi_0(vw_33), _S166);
        accHi_0[1U] = _S176;
        uint vw_34 = (*(&kernelContext_0)->vs_0)[_S172 + 16U];
        float _S177 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S172 + 16U]), _S167);
        accLo_0[2U] = _S177;
        float _S178 = fma(pj_8, attnHi_0(vw_34), _S168);
        accHi_0[2U] = _S178;
        uint vw_35 = (*(&kernelContext_0)->vs_0)[_S172 + 24U];
        float _S179 = fma(pj_8, attnLo_0((*(&kernelContext_0)->vs_0)[_S172 + 24U]), _S169);
        accLo_0[3U] = _S179;
        float _S180 = fma(pj_8, attnHi_0(vw_35), _S170);
        accHi_0[3U] = _S180;
        float pj_9 = (*(&kernelContext_0)->ps_0)[_S55];
        float _S181 = _S171 + (*(&kernelContext_0)->ps_0)[_S55];
        uint _S182 = 297U + kk_0;
        uint vw_36 = (*(&kernelContext_0)->vs_0)[_S182];
        float _S183 = fma((*(&kernelContext_0)->ps_0)[_S55], attnLo_0((*(&kernelContext_0)->vs_0)[_S182]), _S173);
        accLo_0[0U] = _S183;
        float _S184 = fma(pj_9, attnHi_0(vw_36), _S174);
        accHi_0[0U] = _S184;
        uint vw_37 = (*(&kernelContext_0)->vs_0)[_S182 + 8U];
        float _S185 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S182 + 8U]), _S175);
        accLo_0[1U] = _S185;
        float _S186 = fma(pj_9, attnHi_0(vw_37), _S176);
        accHi_0[1U] = _S186;
        uint vw_38 = (*(&kernelContext_0)->vs_0)[_S182 + 16U];
        float _S187 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S182 + 16U]), _S177);
        accLo_0[2U] = _S187;
        float _S188 = fma(pj_9, attnHi_0(vw_38), _S178);
        accHi_0[2U] = _S188;
        uint vw_39 = (*(&kernelContext_0)->vs_0)[_S182 + 24U];
        float _S189 = fma(pj_9, attnLo_0((*(&kernelContext_0)->vs_0)[_S182 + 24U]), _S179);
        accLo_0[3U] = _S189;
        float _S190 = fma(pj_9, attnHi_0(vw_39), _S180);
        accHi_0[3U] = _S190;
        float pj_10 = (*(&kernelContext_0)->ps_0)[_S56];
        float _S191 = _S181 + (*(&kernelContext_0)->ps_0)[_S56];
        uint _S192 = 330U + kk_0;
        uint vw_40 = (*(&kernelContext_0)->vs_0)[_S192];
        float _S193 = fma((*(&kernelContext_0)->ps_0)[_S56], attnLo_0((*(&kernelContext_0)->vs_0)[_S192]), _S183);
        accLo_0[0U] = _S193;
        float _S194 = fma(pj_10, attnHi_0(vw_40), _S184);
        accHi_0[0U] = _S194;
        uint vw_41 = (*(&kernelContext_0)->vs_0)[_S192 + 8U];
        float _S195 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S192 + 8U]), _S185);
        accLo_0[1U] = _S195;
        float _S196 = fma(pj_10, attnHi_0(vw_41), _S186);
        accHi_0[1U] = _S196;
        uint vw_42 = (*(&kernelContext_0)->vs_0)[_S192 + 16U];
        float _S197 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S192 + 16U]), _S187);
        accLo_0[2U] = _S197;
        float _S198 = fma(pj_10, attnHi_0(vw_42), _S188);
        accHi_0[2U] = _S198;
        uint vw_43 = (*(&kernelContext_0)->vs_0)[_S192 + 24U];
        float _S199 = fma(pj_10, attnLo_0((*(&kernelContext_0)->vs_0)[_S192 + 24U]), _S189);
        accLo_0[3U] = _S199;
        float _S200 = fma(pj_10, attnHi_0(vw_43), _S190);
        accHi_0[3U] = _S200;
        float pj_11 = (*(&kernelContext_0)->ps_0)[_S57];
        float _S201 = _S191 + (*(&kernelContext_0)->ps_0)[_S57];
        uint _S202 = 363U + kk_0;
        uint vw_44 = (*(&kernelContext_0)->vs_0)[_S202];
        float _S203 = fma((*(&kernelContext_0)->ps_0)[_S57], attnLo_0((*(&kernelContext_0)->vs_0)[_S202]), _S193);
        accLo_0[0U] = _S203;
        float _S204 = fma(pj_11, attnHi_0(vw_44), _S194);
        accHi_0[0U] = _S204;
        uint vw_45 = (*(&kernelContext_0)->vs_0)[_S202 + 8U];
        float _S205 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S202 + 8U]), _S195);
        accLo_0[1U] = _S205;
        float _S206 = fma(pj_11, attnHi_0(vw_45), _S196);
        accHi_0[1U] = _S206;
        uint vw_46 = (*(&kernelContext_0)->vs_0)[_S202 + 16U];
        float _S207 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S202 + 16U]), _S197);
        accLo_0[2U] = _S207;
        float _S208 = fma(pj_11, attnHi_0(vw_46), _S198);
        accHi_0[2U] = _S208;
        uint vw_47 = (*(&kernelContext_0)->vs_0)[_S202 + 24U];
        float _S209 = fma(pj_11, attnLo_0((*(&kernelContext_0)->vs_0)[_S202 + 24U]), _S199);
        accLo_0[3U] = _S209;
        float _S210 = fma(pj_11, attnHi_0(vw_47), _S200);
        accHi_0[3U] = _S210;
        float pj_12 = (*(&kernelContext_0)->ps_0)[_S58];
        float _S211 = _S201 + (*(&kernelContext_0)->ps_0)[_S58];
        uint _S212 = 396U + kk_0;
        uint vw_48 = (*(&kernelContext_0)->vs_0)[_S212];
        float _S213 = fma((*(&kernelContext_0)->ps_0)[_S58], attnLo_0((*(&kernelContext_0)->vs_0)[_S212]), _S203);
        accLo_0[0U] = _S213;
        float _S214 = fma(pj_12, attnHi_0(vw_48), _S204);
        accHi_0[0U] = _S214;
        uint vw_49 = (*(&kernelContext_0)->vs_0)[_S212 + 8U];
        float _S215 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S212 + 8U]), _S205);
        accLo_0[1U] = _S215;
        float _S216 = fma(pj_12, attnHi_0(vw_49), _S206);
        accHi_0[1U] = _S216;
        uint vw_50 = (*(&kernelContext_0)->vs_0)[_S212 + 16U];
        float _S217 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S212 + 16U]), _S207);
        accLo_0[2U] = _S217;
        float _S218 = fma(pj_12, attnHi_0(vw_50), _S208);
        accHi_0[2U] = _S218;
        uint vw_51 = (*(&kernelContext_0)->vs_0)[_S212 + 24U];
        float _S219 = fma(pj_12, attnLo_0((*(&kernelContext_0)->vs_0)[_S212 + 24U]), _S209);
        accLo_0[3U] = _S219;
        float _S220 = fma(pj_12, attnHi_0(vw_51), _S210);
        accHi_0[3U] = _S220;
        float pj_13 = (*(&kernelContext_0)->ps_0)[_S59];
        float _S221 = _S211 + (*(&kernelContext_0)->ps_0)[_S59];
        uint _S222 = 429U + kk_0;
        uint vw_52 = (*(&kernelContext_0)->vs_0)[_S222];
        float _S223 = fma((*(&kernelContext_0)->ps_0)[_S59], attnLo_0((*(&kernelContext_0)->vs_0)[_S222]), _S213);
        accLo_0[0U] = _S223;
        float _S224 = fma(pj_13, attnHi_0(vw_52), _S214);
        accHi_0[0U] = _S224;
        uint vw_53 = (*(&kernelContext_0)->vs_0)[_S222 + 8U];
        float _S225 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S222 + 8U]), _S215);
        accLo_0[1U] = _S225;
        float _S226 = fma(pj_13, attnHi_0(vw_53), _S216);
        accHi_0[1U] = _S226;
        uint vw_54 = (*(&kernelContext_0)->vs_0)[_S222 + 16U];
        float _S227 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S222 + 16U]), _S217);
        accLo_0[2U] = _S227;
        float _S228 = fma(pj_13, attnHi_0(vw_54), _S218);
        accHi_0[2U] = _S228;
        uint vw_55 = (*(&kernelContext_0)->vs_0)[_S222 + 24U];
        float _S229 = fma(pj_13, attnLo_0((*(&kernelContext_0)->vs_0)[_S222 + 24U]), _S219);
        accLo_0[3U] = _S229;
        float _S230 = fma(pj_13, attnHi_0(vw_55), _S220);
        accHi_0[3U] = _S230;
        float pj_14 = (*(&kernelContext_0)->ps_0)[_S60];
        float _S231 = _S221 + (*(&kernelContext_0)->ps_0)[_S60];
        uint _S232 = 462U + kk_0;
        uint vw_56 = (*(&kernelContext_0)->vs_0)[_S232];
        float _S233 = fma((*(&kernelContext_0)->ps_0)[_S60], attnLo_0((*(&kernelContext_0)->vs_0)[_S232]), _S223);
        accLo_0[0U] = _S233;
        float _S234 = fma(pj_14, attnHi_0(vw_56), _S224);
        accHi_0[0U] = _S234;
        uint vw_57 = (*(&kernelContext_0)->vs_0)[_S232 + 8U];
        float _S235 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S232 + 8U]), _S225);
        accLo_0[1U] = _S235;
        float _S236 = fma(pj_14, attnHi_0(vw_57), _S226);
        accHi_0[1U] = _S236;
        uint vw_58 = (*(&kernelContext_0)->vs_0)[_S232 + 16U];
        float _S237 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S232 + 16U]), _S227);
        accLo_0[2U] = _S237;
        float _S238 = fma(pj_14, attnHi_0(vw_58), _S228);
        accHi_0[2U] = _S238;
        uint vw_59 = (*(&kernelContext_0)->vs_0)[_S232 + 24U];
        float _S239 = fma(pj_14, attnLo_0((*(&kernelContext_0)->vs_0)[_S232 + 24U]), _S229);
        accLo_0[3U] = _S239;
        float _S240 = fma(pj_14, attnHi_0(vw_59), _S230);
        accHi_0[3U] = _S240;
        float pj_15 = (*(&kernelContext_0)->ps_0)[_S61];
        float _S241 = _S231 + (*(&kernelContext_0)->ps_0)[_S61];
        uint _S242 = 495U + kk_0;
        uint vw_60 = (*(&kernelContext_0)->vs_0)[_S242];
        float _S243 = fma((*(&kernelContext_0)->ps_0)[_S61], attnLo_0((*(&kernelContext_0)->vs_0)[_S242]), _S233);
        accLo_0[0U] = _S243;
        float _S244 = fma(pj_15, attnHi_0(vw_60), _S234);
        accHi_0[0U] = _S244;
        uint vw_61 = (*(&kernelContext_0)->vs_0)[_S242 + 8U];
        float _S245 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S242 + 8U]), _S235);
        accLo_0[1U] = _S245;
        float _S246 = fma(pj_15, attnHi_0(vw_61), _S236);
        accHi_0[1U] = _S246;
        uint vw_62 = (*(&kernelContext_0)->vs_0)[_S242 + 16U];
        float _S247 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S242 + 16U]), _S237);
        accLo_0[2U] = _S247;
        float _S248 = fma(pj_15, attnHi_0(vw_62), _S238);
        accHi_0[2U] = _S248;
        uint vw_63 = (*(&kernelContext_0)->vs_0)[_S242 + 24U];
        float _S249 = fma(pj_15, attnLo_0((*(&kernelContext_0)->vs_0)[_S242 + 24U]), _S239);
        accLo_0[3U] = _S249;
        float _S250 = fma(pj_15, attnHi_0(vw_63), _S240);
        accHi_0[3U] = _S250;
        float pj_16 = (*(&kernelContext_0)->ps_0)[_S62];
        float _S251 = _S241 + (*(&kernelContext_0)->ps_0)[_S62];
        uint _S252 = 528U + kk_0;
        uint vw_64 = (*(&kernelContext_0)->vs_0)[_S252];
        float _S253 = fma((*(&kernelContext_0)->ps_0)[_S62], attnLo_0((*(&kernelContext_0)->vs_0)[_S252]), _S243);
        accLo_0[0U] = _S253;
        float _S254 = fma(pj_16, attnHi_0(vw_64), _S244);
        accHi_0[0U] = _S254;
        uint vw_65 = (*(&kernelContext_0)->vs_0)[_S252 + 8U];
        float _S255 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S252 + 8U]), _S245);
        accLo_0[1U] = _S255;
        float _S256 = fma(pj_16, attnHi_0(vw_65), _S246);
        accHi_0[1U] = _S256;
        uint vw_66 = (*(&kernelContext_0)->vs_0)[_S252 + 16U];
        float _S257 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S252 + 16U]), _S247);
        accLo_0[2U] = _S257;
        float _S258 = fma(pj_16, attnHi_0(vw_66), _S248);
        accHi_0[2U] = _S258;
        uint vw_67 = (*(&kernelContext_0)->vs_0)[_S252 + 24U];
        float _S259 = fma(pj_16, attnLo_0((*(&kernelContext_0)->vs_0)[_S252 + 24U]), _S249);
        accLo_0[3U] = _S259;
        float _S260 = fma(pj_16, attnHi_0(vw_67), _S250);
        accHi_0[3U] = _S260;
        float pj_17 = (*(&kernelContext_0)->ps_0)[_S63];
        float _S261 = _S251 + (*(&kernelContext_0)->ps_0)[_S63];
        uint _S262 = 561U + kk_0;
        uint vw_68 = (*(&kernelContext_0)->vs_0)[_S262];
        float _S263 = fma((*(&kernelContext_0)->ps_0)[_S63], attnLo_0((*(&kernelContext_0)->vs_0)[_S262]), _S253);
        accLo_0[0U] = _S263;
        float _S264 = fma(pj_17, attnHi_0(vw_68), _S254);
        accHi_0[0U] = _S264;
        uint vw_69 = (*(&kernelContext_0)->vs_0)[_S262 + 8U];
        float _S265 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S262 + 8U]), _S255);
        accLo_0[1U] = _S265;
        float _S266 = fma(pj_17, attnHi_0(vw_69), _S256);
        accHi_0[1U] = _S266;
        uint vw_70 = (*(&kernelContext_0)->vs_0)[_S262 + 16U];
        float _S267 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S262 + 16U]), _S257);
        accLo_0[2U] = _S267;
        float _S268 = fma(pj_17, attnHi_0(vw_70), _S258);
        accHi_0[2U] = _S268;
        uint vw_71 = (*(&kernelContext_0)->vs_0)[_S262 + 24U];
        float _S269 = fma(pj_17, attnLo_0((*(&kernelContext_0)->vs_0)[_S262 + 24U]), _S259);
        accLo_0[3U] = _S269;
        float _S270 = fma(pj_17, attnHi_0(vw_71), _S260);
        accHi_0[3U] = _S270;
        float pj_18 = (*(&kernelContext_0)->ps_0)[_S64];
        float _S271 = _S261 + (*(&kernelContext_0)->ps_0)[_S64];
        uint _S272 = 594U + kk_0;
        uint vw_72 = (*(&kernelContext_0)->vs_0)[_S272];
        float _S273 = fma((*(&kernelContext_0)->ps_0)[_S64], attnLo_0((*(&kernelContext_0)->vs_0)[_S272]), _S263);
        accLo_0[0U] = _S273;
        float _S274 = fma(pj_18, attnHi_0(vw_72), _S264);
        accHi_0[0U] = _S274;
        uint vw_73 = (*(&kernelContext_0)->vs_0)[_S272 + 8U];
        float _S275 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S272 + 8U]), _S265);
        accLo_0[1U] = _S275;
        float _S276 = fma(pj_18, attnHi_0(vw_73), _S266);
        accHi_0[1U] = _S276;
        uint vw_74 = (*(&kernelContext_0)->vs_0)[_S272 + 16U];
        float _S277 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S272 + 16U]), _S267);
        accLo_0[2U] = _S277;
        float _S278 = fma(pj_18, attnHi_0(vw_74), _S268);
        accHi_0[2U] = _S278;
        uint vw_75 = (*(&kernelContext_0)->vs_0)[_S272 + 24U];
        float _S279 = fma(pj_18, attnLo_0((*(&kernelContext_0)->vs_0)[_S272 + 24U]), _S269);
        accLo_0[3U] = _S279;
        float _S280 = fma(pj_18, attnHi_0(vw_75), _S270);
        accHi_0[3U] = _S280;
        float pj_19 = (*(&kernelContext_0)->ps_0)[_S65];
        float _S281 = _S271 + (*(&kernelContext_0)->ps_0)[_S65];
        uint _S282 = 627U + kk_0;
        uint vw_76 = (*(&kernelContext_0)->vs_0)[_S282];
        float _S283 = fma((*(&kernelContext_0)->ps_0)[_S65], attnLo_0((*(&kernelContext_0)->vs_0)[_S282]), _S273);
        accLo_0[0U] = _S283;
        float _S284 = fma(pj_19, attnHi_0(vw_76), _S274);
        accHi_0[0U] = _S284;
        uint vw_77 = (*(&kernelContext_0)->vs_0)[_S282 + 8U];
        float _S285 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S282 + 8U]), _S275);
        accLo_0[1U] = _S285;
        float _S286 = fma(pj_19, attnHi_0(vw_77), _S276);
        accHi_0[1U] = _S286;
        uint vw_78 = (*(&kernelContext_0)->vs_0)[_S282 + 16U];
        float _S287 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S282 + 16U]), _S277);
        accLo_0[2U] = _S287;
        float _S288 = fma(pj_19, attnHi_0(vw_78), _S278);
        accHi_0[2U] = _S288;
        uint vw_79 = (*(&kernelContext_0)->vs_0)[_S282 + 24U];
        float _S289 = fma(pj_19, attnLo_0((*(&kernelContext_0)->vs_0)[_S282 + 24U]), _S279);
        accLo_0[3U] = _S289;
        float _S290 = fma(pj_19, attnHi_0(vw_79), _S280);
        accHi_0[3U] = _S290;
        float pj_20 = (*(&kernelContext_0)->ps_0)[_S66];
        float _S291 = _S281 + (*(&kernelContext_0)->ps_0)[_S66];
        uint _S292 = 660U + kk_0;
        uint vw_80 = (*(&kernelContext_0)->vs_0)[_S292];
        float _S293 = fma((*(&kernelContext_0)->ps_0)[_S66], attnLo_0((*(&kernelContext_0)->vs_0)[_S292]), _S283);
        accLo_0[0U] = _S293;
        float _S294 = fma(pj_20, attnHi_0(vw_80), _S284);
        accHi_0[0U] = _S294;
        uint vw_81 = (*(&kernelContext_0)->vs_0)[_S292 + 8U];
        float _S295 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S292 + 8U]), _S285);
        accLo_0[1U] = _S295;
        float _S296 = fma(pj_20, attnHi_0(vw_81), _S286);
        accHi_0[1U] = _S296;
        uint vw_82 = (*(&kernelContext_0)->vs_0)[_S292 + 16U];
        float _S297 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S292 + 16U]), _S287);
        accLo_0[2U] = _S297;
        float _S298 = fma(pj_20, attnHi_0(vw_82), _S288);
        accHi_0[2U] = _S298;
        uint vw_83 = (*(&kernelContext_0)->vs_0)[_S292 + 24U];
        float _S299 = fma(pj_20, attnLo_0((*(&kernelContext_0)->vs_0)[_S292 + 24U]), _S289);
        accLo_0[3U] = _S299;
        float _S300 = fma(pj_20, attnHi_0(vw_83), _S290);
        accHi_0[3U] = _S300;
        float pj_21 = (*(&kernelContext_0)->ps_0)[_S67];
        float _S301 = _S291 + (*(&kernelContext_0)->ps_0)[_S67];
        uint _S302 = 693U + kk_0;
        uint vw_84 = (*(&kernelContext_0)->vs_0)[_S302];
        float _S303 = fma((*(&kernelContext_0)->ps_0)[_S67], attnLo_0((*(&kernelContext_0)->vs_0)[_S302]), _S293);
        accLo_0[0U] = _S303;
        float _S304 = fma(pj_21, attnHi_0(vw_84), _S294);
        accHi_0[0U] = _S304;
        uint vw_85 = (*(&kernelContext_0)->vs_0)[_S302 + 8U];
        float _S305 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S302 + 8U]), _S295);
        accLo_0[1U] = _S305;
        float _S306 = fma(pj_21, attnHi_0(vw_85), _S296);
        accHi_0[1U] = _S306;
        uint vw_86 = (*(&kernelContext_0)->vs_0)[_S302 + 16U];
        float _S307 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S302 + 16U]), _S297);
        accLo_0[2U] = _S307;
        float _S308 = fma(pj_21, attnHi_0(vw_86), _S298);
        accHi_0[2U] = _S308;
        uint vw_87 = (*(&kernelContext_0)->vs_0)[_S302 + 24U];
        float _S309 = fma(pj_21, attnLo_0((*(&kernelContext_0)->vs_0)[_S302 + 24U]), _S299);
        accLo_0[3U] = _S309;
        float _S310 = fma(pj_21, attnHi_0(vw_87), _S300);
        accHi_0[3U] = _S310;
        float pj_22 = (*(&kernelContext_0)->ps_0)[_S68];
        float _S311 = _S301 + (*(&kernelContext_0)->ps_0)[_S68];
        uint _S312 = 726U + kk_0;
        uint vw_88 = (*(&kernelContext_0)->vs_0)[_S312];
        float _S313 = fma((*(&kernelContext_0)->ps_0)[_S68], attnLo_0((*(&kernelContext_0)->vs_0)[_S312]), _S303);
        accLo_0[0U] = _S313;
        float _S314 = fma(pj_22, attnHi_0(vw_88), _S304);
        accHi_0[0U] = _S314;
        uint vw_89 = (*(&kernelContext_0)->vs_0)[_S312 + 8U];
        float _S315 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S312 + 8U]), _S305);
        accLo_0[1U] = _S315;
        float _S316 = fma(pj_22, attnHi_0(vw_89), _S306);
        accHi_0[1U] = _S316;
        uint vw_90 = (*(&kernelContext_0)->vs_0)[_S312 + 16U];
        float _S317 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S312 + 16U]), _S307);
        accLo_0[2U] = _S317;
        float _S318 = fma(pj_22, attnHi_0(vw_90), _S308);
        accHi_0[2U] = _S318;
        uint vw_91 = (*(&kernelContext_0)->vs_0)[_S312 + 24U];
        float _S319 = fma(pj_22, attnLo_0((*(&kernelContext_0)->vs_0)[_S312 + 24U]), _S309);
        accLo_0[3U] = _S319;
        float _S320 = fma(pj_22, attnHi_0(vw_91), _S310);
        accHi_0[3U] = _S320;
        float pj_23 = (*(&kernelContext_0)->ps_0)[_S69];
        float _S321 = _S311 + (*(&kernelContext_0)->ps_0)[_S69];
        uint _S322 = 759U + kk_0;
        uint vw_92 = (*(&kernelContext_0)->vs_0)[_S322];
        float _S323 = fma((*(&kernelContext_0)->ps_0)[_S69], attnLo_0((*(&kernelContext_0)->vs_0)[_S322]), _S313);
        accLo_0[0U] = _S323;
        float _S324 = fma(pj_23, attnHi_0(vw_92), _S314);
        accHi_0[0U] = _S324;
        uint vw_93 = (*(&kernelContext_0)->vs_0)[_S322 + 8U];
        float _S325 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S322 + 8U]), _S315);
        accLo_0[1U] = _S325;
        float _S326 = fma(pj_23, attnHi_0(vw_93), _S316);
        accHi_0[1U] = _S326;
        uint vw_94 = (*(&kernelContext_0)->vs_0)[_S322 + 16U];
        float _S327 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S322 + 16U]), _S317);
        accLo_0[2U] = _S327;
        float _S328 = fma(pj_23, attnHi_0(vw_94), _S318);
        accHi_0[2U] = _S328;
        uint vw_95 = (*(&kernelContext_0)->vs_0)[_S322 + 24U];
        float _S329 = fma(pj_23, attnLo_0((*(&kernelContext_0)->vs_0)[_S322 + 24U]), _S319);
        accLo_0[3U] = _S329;
        float _S330 = fma(pj_23, attnHi_0(vw_95), _S320);
        accHi_0[3U] = _S330;
        float pj_24 = (*(&kernelContext_0)->ps_0)[_S70];
        float _S331 = _S321 + (*(&kernelContext_0)->ps_0)[_S70];
        uint _S332 = 792U + kk_0;
        uint vw_96 = (*(&kernelContext_0)->vs_0)[_S332];
        float _S333 = fma((*(&kernelContext_0)->ps_0)[_S70], attnLo_0((*(&kernelContext_0)->vs_0)[_S332]), _S323);
        accLo_0[0U] = _S333;
        float _S334 = fma(pj_24, attnHi_0(vw_96), _S324);
        accHi_0[0U] = _S334;
        uint vw_97 = (*(&kernelContext_0)->vs_0)[_S332 + 8U];
        float _S335 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S332 + 8U]), _S325);
        accLo_0[1U] = _S335;
        float _S336 = fma(pj_24, attnHi_0(vw_97), _S326);
        accHi_0[1U] = _S336;
        uint vw_98 = (*(&kernelContext_0)->vs_0)[_S332 + 16U];
        float _S337 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S332 + 16U]), _S327);
        accLo_0[2U] = _S337;
        float _S338 = fma(pj_24, attnHi_0(vw_98), _S328);
        accHi_0[2U] = _S338;
        uint vw_99 = (*(&kernelContext_0)->vs_0)[_S332 + 24U];
        float _S339 = fma(pj_24, attnLo_0((*(&kernelContext_0)->vs_0)[_S332 + 24U]), _S329);
        accLo_0[3U] = _S339;
        float _S340 = fma(pj_24, attnHi_0(vw_99), _S330);
        accHi_0[3U] = _S340;
        float pj_25 = (*(&kernelContext_0)->ps_0)[_S71];
        float _S341 = _S331 + (*(&kernelContext_0)->ps_0)[_S71];
        uint _S342 = 825U + kk_0;
        uint vw_100 = (*(&kernelContext_0)->vs_0)[_S342];
        float _S343 = fma((*(&kernelContext_0)->ps_0)[_S71], attnLo_0((*(&kernelContext_0)->vs_0)[_S342]), _S333);
        accLo_0[0U] = _S343;
        float _S344 = fma(pj_25, attnHi_0(vw_100), _S334);
        accHi_0[0U] = _S344;
        uint vw_101 = (*(&kernelContext_0)->vs_0)[_S342 + 8U];
        float _S345 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S342 + 8U]), _S335);
        accLo_0[1U] = _S345;
        float _S346 = fma(pj_25, attnHi_0(vw_101), _S336);
        accHi_0[1U] = _S346;
        uint vw_102 = (*(&kernelContext_0)->vs_0)[_S342 + 16U];
        float _S347 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S342 + 16U]), _S337);
        accLo_0[2U] = _S347;
        float _S348 = fma(pj_25, attnHi_0(vw_102), _S338);
        accHi_0[2U] = _S348;
        uint vw_103 = (*(&kernelContext_0)->vs_0)[_S342 + 24U];
        float _S349 = fma(pj_25, attnLo_0((*(&kernelContext_0)->vs_0)[_S342 + 24U]), _S339);
        accLo_0[3U] = _S349;
        float _S350 = fma(pj_25, attnHi_0(vw_103), _S340);
        accHi_0[3U] = _S350;
        float pj_26 = (*(&kernelContext_0)->ps_0)[_S72];
        float _S351 = _S341 + (*(&kernelContext_0)->ps_0)[_S72];
        uint _S352 = 858U + kk_0;
        uint vw_104 = (*(&kernelContext_0)->vs_0)[_S352];
        float _S353 = fma((*(&kernelContext_0)->ps_0)[_S72], attnLo_0((*(&kernelContext_0)->vs_0)[_S352]), _S343);
        accLo_0[0U] = _S353;
        float _S354 = fma(pj_26, attnHi_0(vw_104), _S344);
        accHi_0[0U] = _S354;
        uint vw_105 = (*(&kernelContext_0)->vs_0)[_S352 + 8U];
        float _S355 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S352 + 8U]), _S345);
        accLo_0[1U] = _S355;
        float _S356 = fma(pj_26, attnHi_0(vw_105), _S346);
        accHi_0[1U] = _S356;
        uint vw_106 = (*(&kernelContext_0)->vs_0)[_S352 + 16U];
        float _S357 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S352 + 16U]), _S347);
        accLo_0[2U] = _S357;
        float _S358 = fma(pj_26, attnHi_0(vw_106), _S348);
        accHi_0[2U] = _S358;
        uint vw_107 = (*(&kernelContext_0)->vs_0)[_S352 + 24U];
        float _S359 = fma(pj_26, attnLo_0((*(&kernelContext_0)->vs_0)[_S352 + 24U]), _S349);
        accLo_0[3U] = _S359;
        float _S360 = fma(pj_26, attnHi_0(vw_107), _S350);
        accHi_0[3U] = _S360;
        float pj_27 = (*(&kernelContext_0)->ps_0)[_S73];
        float _S361 = _S351 + (*(&kernelContext_0)->ps_0)[_S73];
        uint _S362 = 891U + kk_0;
        uint vw_108 = (*(&kernelContext_0)->vs_0)[_S362];
        float _S363 = fma((*(&kernelContext_0)->ps_0)[_S73], attnLo_0((*(&kernelContext_0)->vs_0)[_S362]), _S353);
        accLo_0[0U] = _S363;
        float _S364 = fma(pj_27, attnHi_0(vw_108), _S354);
        accHi_0[0U] = _S364;
        uint vw_109 = (*(&kernelContext_0)->vs_0)[_S362 + 8U];
        float _S365 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S362 + 8U]), _S355);
        accLo_0[1U] = _S365;
        float _S366 = fma(pj_27, attnHi_0(vw_109), _S356);
        accHi_0[1U] = _S366;
        uint vw_110 = (*(&kernelContext_0)->vs_0)[_S362 + 16U];
        float _S367 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S362 + 16U]), _S357);
        accLo_0[2U] = _S367;
        float _S368 = fma(pj_27, attnHi_0(vw_110), _S358);
        accHi_0[2U] = _S368;
        uint vw_111 = (*(&kernelContext_0)->vs_0)[_S362 + 24U];
        float _S369 = fma(pj_27, attnLo_0((*(&kernelContext_0)->vs_0)[_S362 + 24U]), _S359);
        accLo_0[3U] = _S369;
        float _S370 = fma(pj_27, attnHi_0(vw_111), _S360);
        accHi_0[3U] = _S370;
        float pj_28 = (*(&kernelContext_0)->ps_0)[_S74];
        float _S371 = _S361 + (*(&kernelContext_0)->ps_0)[_S74];
        uint _S372 = 924U + kk_0;
        uint vw_112 = (*(&kernelContext_0)->vs_0)[_S372];
        float _S373 = fma((*(&kernelContext_0)->ps_0)[_S74], attnLo_0((*(&kernelContext_0)->vs_0)[_S372]), _S363);
        accLo_0[0U] = _S373;
        float _S374 = fma(pj_28, attnHi_0(vw_112), _S364);
        accHi_0[0U] = _S374;
        uint vw_113 = (*(&kernelContext_0)->vs_0)[_S372 + 8U];
        float _S375 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S372 + 8U]), _S365);
        accLo_0[1U] = _S375;
        float _S376 = fma(pj_28, attnHi_0(vw_113), _S366);
        accHi_0[1U] = _S376;
        uint vw_114 = (*(&kernelContext_0)->vs_0)[_S372 + 16U];
        float _S377 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S372 + 16U]), _S367);
        accLo_0[2U] = _S377;
        float _S378 = fma(pj_28, attnHi_0(vw_114), _S368);
        accHi_0[2U] = _S378;
        uint vw_115 = (*(&kernelContext_0)->vs_0)[_S372 + 24U];
        float _S379 = fma(pj_28, attnLo_0((*(&kernelContext_0)->vs_0)[_S372 + 24U]), _S369);
        accLo_0[3U] = _S379;
        float _S380 = fma(pj_28, attnHi_0(vw_115), _S370);
        accHi_0[3U] = _S380;
        float pj_29 = (*(&kernelContext_0)->ps_0)[_S75];
        float _S381 = _S371 + (*(&kernelContext_0)->ps_0)[_S75];
        uint _S382 = 957U + kk_0;
        uint vw_116 = (*(&kernelContext_0)->vs_0)[_S382];
        float _S383 = fma((*(&kernelContext_0)->ps_0)[_S75], attnLo_0((*(&kernelContext_0)->vs_0)[_S382]), _S373);
        accLo_0[0U] = _S383;
        float _S384 = fma(pj_29, attnHi_0(vw_116), _S374);
        accHi_0[0U] = _S384;
        uint vw_117 = (*(&kernelContext_0)->vs_0)[_S382 + 8U];
        float _S385 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S382 + 8U]), _S375);
        accLo_0[1U] = _S385;
        float _S386 = fma(pj_29, attnHi_0(vw_117), _S376);
        accHi_0[1U] = _S386;
        uint vw_118 = (*(&kernelContext_0)->vs_0)[_S382 + 16U];
        float _S387 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S382 + 16U]), _S377);
        accLo_0[2U] = _S387;
        float _S388 = fma(pj_29, attnHi_0(vw_118), _S378);
        accHi_0[2U] = _S388;
        uint vw_119 = (*(&kernelContext_0)->vs_0)[_S382 + 24U];
        float _S389 = fma(pj_29, attnLo_0((*(&kernelContext_0)->vs_0)[_S382 + 24U]), _S379);
        accLo_0[3U] = _S389;
        float _S390 = fma(pj_29, attnHi_0(vw_119), _S380);
        accHi_0[3U] = _S390;
        float pj_30 = (*(&kernelContext_0)->ps_0)[_S76];
        float _S391 = _S381 + (*(&kernelContext_0)->ps_0)[_S76];
        uint _S392 = 990U + kk_0;
        uint vw_120 = (*(&kernelContext_0)->vs_0)[_S392];
        float _S393 = fma((*(&kernelContext_0)->ps_0)[_S76], attnLo_0((*(&kernelContext_0)->vs_0)[_S392]), _S383);
        accLo_0[0U] = _S393;
        float _S394 = fma(pj_30, attnHi_0(vw_120), _S384);
        accHi_0[0U] = _S394;
        uint vw_121 = (*(&kernelContext_0)->vs_0)[_S392 + 8U];
        float _S395 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S392 + 8U]), _S385);
        accLo_0[1U] = _S395;
        float _S396 = fma(pj_30, attnHi_0(vw_121), _S386);
        accHi_0[1U] = _S396;
        uint vw_122 = (*(&kernelContext_0)->vs_0)[_S392 + 16U];
        float _S397 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S392 + 16U]), _S387);
        accLo_0[2U] = _S397;
        float _S398 = fma(pj_30, attnHi_0(vw_122), _S388);
        accHi_0[2U] = _S398;
        uint vw_123 = (*(&kernelContext_0)->vs_0)[_S392 + 24U];
        float _S399 = fma(pj_30, attnLo_0((*(&kernelContext_0)->vs_0)[_S392 + 24U]), _S389);
        accLo_0[3U] = _S399;
        float _S400 = fma(pj_30, attnHi_0(vw_123), _S390);
        accHi_0[3U] = _S400;
        float pj_31 = (*(&kernelContext_0)->ps_0)[_S77];
        float l_1 = _S391 + (*(&kernelContext_0)->ps_0)[_S77];
        uint _S401 = 1023U + kk_0;
        uint vw_124 = (*(&kernelContext_0)->vs_0)[_S401];
        accLo_0[0U] = fma((*(&kernelContext_0)->ps_0)[_S77], attnLo_0((*(&kernelContext_0)->vs_0)[_S401]), _S393);
        accHi_0[0U] = fma(pj_31, attnHi_0(vw_124), _S394);
        uint vw_125 = (*(&kernelContext_0)->vs_0)[_S401 + 8U];
        accLo_0[1U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S401 + 8U]), _S395);
        accHi_0[1U] = fma(pj_31, attnHi_0(vw_125), _S396);
        uint vw_126 = (*(&kernelContext_0)->vs_0)[_S401 + 16U];
        accLo_0[2U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S401 + 16U]), _S397);
        accHi_0[2U] = fma(pj_31, attnHi_0(vw_126), _S398);
        uint vw_127 = (*(&kernelContext_0)->vs_0)[_S401 + 24U];
        accLo_0[3U] = fma(pj_31, attnLo_0((*(&kernelContext_0)->vs_0)[_S401 + 24U]), _S399);
        accHi_0[3U] = fma(pj_31, attnHi_0(vw_127), _S400);
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
            uint _S402 = 2U * pr_30;
            *((&kernelContext_0)->partialML_0+_S402) = m_0;
            *((&kernelContext_0)->partialML_0+(_S402 + 1U)) = l_0;
        }
        if(kk_0 < _S16)
        {
            uint _S403 = 2U * kk_0;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S403)) = accLo_0[0U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S403 + 1U)) = accHi_0[0U];
        }
        uint pair_0 = kk_0 + 8U;
        if(pair_0 < _S16)
        {
            uint _S404 = 2U * pair_0;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S404)) = accLo_0[1U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S404 + 1U)) = accHi_0[1U];
        }
        uint pair_1 = kk_0 + 16U;
        if(pair_1 < _S16)
        {
            uint _S405 = 2U * pair_1;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S405)) = accLo_0[2U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S405 + 1U)) = accHi_0[2U];
        }
        uint pair_2 = kk_0 + 24U;
        if(pair_2 < _S16)
        {
            uint _S406 = 2U * pair_2;
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S406)) = accLo_0[3U];
            *((&kernelContext_0)->partialAcc_0+(pr_30 * (&kernelContext_0)->p_0->headDim_0 + _S406 + 1U)) = accHi_0[3U];
        }
        return;
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
                uint _S407 = 2U * kk_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S407)) = accLo_0[0U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S407 + 1U)) = hi_0;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_3 = kk_0 + 8U;
                if(pair_3 >= _S16)
                {
                    break;
                }
                float hi_1 = accHi_0[1U] / l_0;
                uint _S408 = 2U * pair_3;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S408)) = accLo_0[1U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S408 + 1U)) = hi_1;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_4 = kk_0 + 16U;
                if(pair_4 >= _S16)
                {
                    break;
                }
                float hi_2 = accHi_0[2U] / l_0;
                uint _S409 = 2U * pair_4;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S409)) = accLo_0[2U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S409 + 1U)) = hi_2;
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint pair_5 = kk_0 + 24U;
                if(pair_5 >= _S16)
                {
                    break;
                }
                float hi_3 = accHi_0[3U] / l_0;
                uint _S410 = 2U * pair_5;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S410)) = accLo_0[3U] / l_0;
                *((&kernelContext_0)->o_0+(row_1 * (&kernelContext_0)->p_0->headDim_0 + _S410 + 1U)) = hi_3;
                break;
            }
            break;
        }
        break;
    }
    return;
}

