#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct AttnBwdParams_0
{
    uint heads_0;
    uint kvHeads_0;
    uint queries_0;
    uint keys_0;
    uint headDim_0;
    uint cacheKeys_0;
    float scale_0;
    uint causal_0;
    uint gridX_0;
    uint lengthGroup_0;
    uint pad1_0;
    uint pad2_0;
};

struct KernelContext_0
{
    AttnBwdParams_0 constant* p_0;
    uint device* dD_0;
    uint device* q_0;
    float device* dO_0;
    float device* lse_0;
    uint device* k_0;
    uint device* v_0;
    float device* out0_0;
    array<uint, int(1032)> threadgroup* qs_0;
    array<float, int(2056)> threadgroup* dos_0;
    array<float, int(8)> threadgroup* ls_0;
    array<float, int(8)> threadgroup* ds_0;
    array<uint, int(2064)> threadgroup* ks_0;
    array<uint, int(2064)> threadgroup* vs_0;
    array<float, int(128)> threadgroup* ps_0;
    array<float, int(128)> threadgroup* dss_0;
};

uint validKeys_0(uint row_0, KernelContext_0 thread* kernelContext_0)
{
    if((kernelContext_0->p_0->lengthGroup_0) == 0U)
    {
        return kernelContext_0->p_0->keys_0;
    }
    uint _S1 = kernelContext_0->p_0->heads_0 * kernelContext_0->p_0->queries_0;
    uint _S2 = row_0 / kernelContext_0->p_0->queries_0;
    uint _S3 = _S2 / kernelContext_0->p_0->lengthGroup_0;
    return *(kernelContext_0->dD_0+(_S1 + _S3));
}

void loadRows_0(uint t_0, uint base_0, uint r0_0, uint rows_0, uint d2_0, KernelContext_0 thread* kernelContext_1)
{
    float _S4;
    uint end_0;
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
        uint i_0 = e_0 / 128U;
        uint pr_0 = e_0 % 128U;
        uint r_0 = r0_0 + i_0;
        bool _S5;
        if(r_0 < rows_0)
        {
            _S5 = pr_0 < d2_0;
        }
        else
        {
            _S5 = false;
        }
        bool ok_0;
        if(_S5)
        {
            uint _S6 = validKeys_0(base_0 + r_0, kernelContext_1);
            ok_0 = _S6 != 0U;
        }
        else
        {
            ok_0 = false;
        }
        if(ok_0)
        {
            end_0 = *(kernelContext_1->q_0+((base_0 + r_0) * d2_0 + pr_0));
        }
        else
        {
            end_0 = 0U;
        }
        (*kernelContext_1->qs_0)[i_0 * 129U + pr_0] = end_0;
        uint _S7 = 2U * pr_0;
        uint _S8 = i_0 * 257U + _S7;
        if(ok_0)
        {
            _S4 = *(kernelContext_1->dO_0+((base_0 + r_0) * kernelContext_1->p_0->headDim_0 + _S7));
        }
        else
        {
            _S4 = 0.0;
        }
        (*kernelContext_1->dos_0)[_S8] = _S4;
        float _S9;
        if(ok_0)
        {
            _S9 = *(kernelContext_1->dO_0+((base_0 + r_0) * kernelContext_1->p_0->headDim_0 + _S7 + 1U));
        }
        else
        {
            _S9 = 0.0;
        }
        (*kernelContext_1->dos_0)[_S8 + 1U] = _S9;
        e_0 = e_0 + 128U;
    }
    if(t_0 < 8U)
    {
        uint r_1 = r0_0 + t_0;
        if(r_1 < rows_0)
        {
            uint _S10 = validKeys_0(base_0 + r_1, kernelContext_1);
            end_0 = _S10;
        }
        else
        {
            end_0 = 0U;
        }
        if((kernelContext_1->p_0->causal_0) != 0U)
        {
            uint _S11 = r_1 % kernelContext_1->p_0->queries_0;
            end_0 = min(end_0, _S11 + (kernelContext_1->p_0->keys_0 - kernelContext_1->p_0->queries_0) + 1U);
        }
        (*kernelContext_1->qs_0)[t_0 * 129U + 128U] = end_0;
        bool _S12 = end_0 != 0U;
        if(_S12)
        {
            _S4 = *(kernelContext_1->lse_0+(base_0 + r_1));
        }
        else
        {
            _S4 = 0.0;
        }
        (*kernelContext_1->ls_0)[t_0] = _S4;
        if(_S12)
        {
            _S4 = (as_type<float>((*(kernelContext_1->dD_0+(base_0 + r_1)))));
        }
        else
        {
            _S4 = 0.0;
        }
        (*kernelContext_1->ds_0)[t_0] = _S4;
    }
    return;
}

uint maxValidKeys_0(uint firstHead_0, uint lastHead_0, KernelContext_0 thread* kernelContext_2)
{
    if((kernelContext_2->p_0->lengthGroup_0) == 0U)
    {
        return kernelContext_2->p_0->keys_0;
    }
    uint _S13 = firstHead_0 / kernelContext_2->p_0->lengthGroup_0;
    uint end_1 = 0U;
    uint g_0 = _S13;
    for(;;)
    {
        uint _S14 = lastHead_0 / kernelContext_2->p_0->lengthGroup_0;
        if(g_0 <= _S14)
        {
        }
        else
        {
            break;
        }
        uint _S15 = max(end_1, *(kernelContext_2->dD_0+(kernelContext_2->p_0->heads_0 * kernelContext_2->p_0->queries_0 + g_0)));
        uint _S16 = g_0 + 1U;
        end_1 = _S15;
        g_0 = _S16;
    }
    return end_1;
}

void loadKeys_0(uint t_1, uint kvHead_0, uint k0_0, uint d2_1, uint end_2, KernelContext_0 thread* kernelContext_3)
{
    uint e_1 = t_1;
    for(;;)
    {
        if(e_1 < 2048U)
        {
        }
        else
        {
            break;
        }
        uint kk_0 = e_1 / 128U;
        uint pr_1 = e_1 % 128U;
        uint key_0 = k0_0 + kk_0;
        bool ok_1;
        if(key_0 < end_2)
        {
            ok_1 = pr_1 < d2_1;
        }
        else
        {
            ok_1 = false;
        }
        uint at_0 = (kvHead_0 * kernelContext_3->p_0->cacheKeys_0 + key_0) * d2_1 + pr_1;
        uint _S17 = kk_0 * 129U + pr_1;
        uint _S18;
        if(ok_1)
        {
            _S18 = *(kernelContext_3->k_0+at_0);
        }
        else
        {
            _S18 = 0U;
        }
        (*kernelContext_3->ks_0)[_S17] = _S18;
        uint _S19;
        if(ok_1)
        {
            _S19 = *(kernelContext_3->v_0+at_0);
        }
        else
        {
            _S19 = 0U;
        }
        (*kernelContext_3->vs_0)[_S17] = _S19;
        e_1 = e_1 + 128U;
    }
    return;
}

float attnLo_0(uint w_0)
{
    return (as_type<half>((ushort)((w_0 & 65535U))));
}

float attnHi_0(uint w_1)
{
    return (as_type<half>((ushort)((w_1 >> 16U))));
}

void pairs_0(uint t_2, uint k0_1, uint d2_2, KernelContext_0 thread* kernelContext_4)
{
    thread array<uint, int(1)> ii_0;
    ii_0[0U] = t_2 / 16U;
    thread array<uint, int(1)> kv_0;
    kv_0[0U] = t_2 % 16U;
    thread array<float, int(1)> sv_0;
    sv_0[0U] = 0.0;
    thread array<float, int(1)> dp_0;
    dp_0[0U] = 0.0;
    uint p0_0 = 0U;
    for(;;)
    {
        if(p0_0 < 128U)
        {
        }
        else
        {
            break;
        }
        if(p0_0 < d2_2)
        {
            uint vw_0 = (*kernelContext_4->vs_0)[kv_0[0U] * 129U + p0_0];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + p0_0]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + p0_0]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + p0_0]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + p0_0]), sv_0[0U]));
            uint _S20 = 2U * p0_0;
            float _S21 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S20] * attnLo_0(vw_0);
            dp_0[0U] = _S21;
            dp_0[0U] = _S21 + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S20 + 1U] * attnHi_0(vw_0);
        }
        uint pr_2 = p0_0 + 1U;
        if(pr_2 < d2_2)
        {
            uint vw_1 = (*kernelContext_4->vs_0)[kv_0[0U] * 129U + pr_2];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_2]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_2]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_2]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_2]), sv_0[0U]));
            uint _S22 = 2U * pr_2;
            float _S23 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S22] * attnLo_0(vw_1);
            dp_0[0U] = _S23;
            dp_0[0U] = _S23 + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S22 + 1U] * attnHi_0(vw_1);
        }
        uint pr_3 = p0_0 + 2U;
        if(pr_3 < d2_2)
        {
            uint vw_2 = (*kernelContext_4->vs_0)[kv_0[0U] * 129U + pr_3];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_3]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_3]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_3]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_3]), sv_0[0U]));
            uint _S24 = 2U * pr_3;
            float _S25 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S24] * attnLo_0(vw_2);
            dp_0[0U] = _S25;
            dp_0[0U] = _S25 + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S24 + 1U] * attnHi_0(vw_2);
        }
        uint pr_4 = p0_0 + 3U;
        if(pr_4 < d2_2)
        {
            uint vw_3 = (*kernelContext_4->vs_0)[kv_0[0U] * 129U + pr_4];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_4]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_4]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_4]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_4]), sv_0[0U]));
            uint _S26 = 2U * pr_4;
            float _S27 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S26] * attnLo_0(vw_3);
            dp_0[0U] = _S27;
            dp_0[0U] = _S27 + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S26 + 1U] * attnHi_0(vw_3);
        }
        uint pr_5 = p0_0 + 4U;
        if(pr_5 < d2_2)
        {
            uint vw_4 = (*kernelContext_4->vs_0)[kv_0[0U] * 129U + pr_5];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_5]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_5]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_5]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_5]), sv_0[0U]));
            uint _S28 = 2U * pr_5;
            float _S29 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S28] * attnLo_0(vw_4);
            dp_0[0U] = _S29;
            dp_0[0U] = _S29 + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S28 + 1U] * attnHi_0(vw_4);
        }
        uint pr_6 = p0_0 + 5U;
        if(pr_6 < d2_2)
        {
            uint vw_5 = (*kernelContext_4->vs_0)[kv_0[0U] * 129U + pr_6];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_6]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_6]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_6]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_6]), sv_0[0U]));
            uint _S30 = 2U * pr_6;
            float _S31 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S30] * attnLo_0(vw_5);
            dp_0[0U] = _S31;
            dp_0[0U] = _S31 + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S30 + 1U] * attnHi_0(vw_5);
        }
        uint pr_7 = p0_0 + 6U;
        if(pr_7 < d2_2)
        {
            uint vw_6 = (*kernelContext_4->vs_0)[kv_0[0U] * 129U + pr_7];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_7]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_7]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_7]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_7]), sv_0[0U]));
            uint _S32 = 2U * pr_7;
            float _S33 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S32] * attnLo_0(vw_6);
            dp_0[0U] = _S33;
            dp_0[0U] = _S33 + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S32 + 1U] * attnHi_0(vw_6);
        }
        uint pr_8 = p0_0 + 7U;
        if(pr_8 < d2_2)
        {
            uint vw_7 = (*kernelContext_4->vs_0)[kv_0[0U] * 129U + pr_8];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_8]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_8]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 129U + pr_8]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 129U + pr_8]), sv_0[0U]));
            uint _S34 = 2U * pr_8;
            float _S35 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S34] * attnLo_0(vw_7);
            dp_0[0U] = _S35;
            dp_0[0U] = _S35 + (*kernelContext_4->dos_0)[ii_0[0U] * 257U + _S34 + 1U] * attnHi_0(vw_7);
        }
        p0_0 = p0_0 + 8U;
    }
    uint i_1 = ii_0[0U];
    uint kk_1 = kv_0[0U];
    bool live_0 = (k0_1 + kv_0[0U]) < (*kernelContext_4->qs_0)[ii_0[0U] * 129U + 128U];
    float pv_0;
    if(live_0)
    {
        pv_0 = exp(sv_0[0U] * kernelContext_4->p_0->scale_0 - (*kernelContext_4->ls_0)[i_1]);
    }
    else
    {
        pv_0 = 0.0;
    }
    uint _S36 = i_1 * 16U + kk_1;
    (*kernelContext_4->ps_0)[_S36] = pv_0;
    if(live_0)
    {
        pv_0 = pv_0 * (dp_0[0U] - (*kernelContext_4->ds_0)[i_1]);
    }
    else
    {
        pv_0 = 0.0;
    }
    (*kernelContext_4->dss_0)[_S36] = pv_0;
    return;
}

float half2At_0(uint w_2, uint c_0)
{
    float _S37;
    if((c_0 & 1U) == 0U)
    {
        _S37 = attnLo_0(w_2);
    }
    else
    {
        _S37 = attnHi_0(w_2);
    }
    return _S37;
}

[[kernel]] void attn_bwd_q_256(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnBwdParams_0 constant* p_1 [[buffer(7)]], uint device* dD_1 [[buffer(5)]], uint device* q_1 [[buffer(0)]], float device* dO_1 [[buffer(3)]], float device* lse_1 [[buffer(4)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* out0_1 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->p_0 = p_1;
    (&kernelContext_5)->dD_0 = dD_1;
    (&kernelContext_5)->q_0 = q_1;
    (&kernelContext_5)->dO_0 = dO_1;
    (&kernelContext_5)->lse_0 = lse_1;
    (&kernelContext_5)->k_0 = k_1;
    (&kernelContext_5)->v_0 = v_1;
    (&kernelContext_5)->out0_0 = out0_1;
    threadgroup array<uint, int(1032)> qs_1;
    (&kernelContext_5)->qs_0 = &qs_1;
    threadgroup array<float, int(2056)> dos_1;
    (&kernelContext_5)->dos_0 = &dos_1;
    threadgroup array<float, int(8)> ls_1;
    (&kernelContext_5)->ls_0 = &ls_1;
    threadgroup array<float, int(8)> ds_1;
    (&kernelContext_5)->ds_0 = &ds_1;
    threadgroup array<uint, int(2064)> ks_1;
    (&kernelContext_5)->ks_0 = &ks_1;
    threadgroup array<uint, int(2064)> vs_1;
    (&kernelContext_5)->vs_0 = &vs_1;
    threadgroup array<float, int(128)> ps_1;
    (&kernelContext_5)->ps_0 = &ps_1;
    threadgroup array<float, int(128)> dss_1;
    (&kernelContext_5)->dss_0 = &dss_1;
    uint t_3 = local_0.x;
    uint kvHead_1 = group_0.y;
    uint d2_3 = p_1->headDim_0 / 2U;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint rows_1 = hpk_0 * (&kernelContext_5)->p_0->queries_0;
    uint base_1 = kvHead_1 * rows_1;
    uint offset_0 = (&kernelContext_5)->p_0->keys_0 - (&kernelContext_5)->p_0->queries_0;
    uint owner_0 = t_3 / 8U;
    uint _S38 = t_3 % 8U;
    thread array<float, int(32)> a0_0;
    a0_0[0U] = 0.0;
    thread array<float, int(32)> a1_0;
    a1_0[0U] = 0.0;
    a0_0[1U] = 0.0;
    a1_0[1U] = 0.0;
    a0_0[2U] = 0.0;
    a1_0[2U] = 0.0;
    a0_0[3U] = 0.0;
    a1_0[3U] = 0.0;
    a0_0[4U] = 0.0;
    a1_0[4U] = 0.0;
    a0_0[5U] = 0.0;
    a1_0[5U] = 0.0;
    a0_0[6U] = 0.0;
    a1_0[6U] = 0.0;
    a0_0[7U] = 0.0;
    a1_0[7U] = 0.0;
    a0_0[8U] = 0.0;
    a1_0[8U] = 0.0;
    a0_0[9U] = 0.0;
    a1_0[9U] = 0.0;
    a0_0[10U] = 0.0;
    a1_0[10U] = 0.0;
    a0_0[11U] = 0.0;
    a1_0[11U] = 0.0;
    a0_0[12U] = 0.0;
    a1_0[12U] = 0.0;
    a0_0[13U] = 0.0;
    a1_0[13U] = 0.0;
    a0_0[14U] = 0.0;
    a1_0[14U] = 0.0;
    a0_0[15U] = 0.0;
    a1_0[15U] = 0.0;
    a0_0[16U] = 0.0;
    a1_0[16U] = 0.0;
    a0_0[17U] = 0.0;
    a1_0[17U] = 0.0;
    a0_0[18U] = 0.0;
    a1_0[18U] = 0.0;
    a0_0[19U] = 0.0;
    a1_0[19U] = 0.0;
    a0_0[20U] = 0.0;
    a1_0[20U] = 0.0;
    a0_0[21U] = 0.0;
    a1_0[21U] = 0.0;
    a0_0[22U] = 0.0;
    a1_0[22U] = 0.0;
    a0_0[23U] = 0.0;
    a1_0[23U] = 0.0;
    a0_0[24U] = 0.0;
    a1_0[24U] = 0.0;
    a0_0[25U] = 0.0;
    a1_0[25U] = 0.0;
    a0_0[26U] = 0.0;
    a1_0[26U] = 0.0;
    a0_0[27U] = 0.0;
    a1_0[27U] = 0.0;
    a0_0[28U] = 0.0;
    a1_0[28U] = 0.0;
    a0_0[29U] = 0.0;
    a1_0[29U] = 0.0;
    a0_0[30U] = 0.0;
    a1_0[30U] = 0.0;
    a0_0[31U] = 0.0;
    a1_0[31U] = 0.0;
    uint r0_1 = group_0.x * 8U;
    loadRows_0(t_3, base_1, r0_1, rows_1, d2_3, &kernelContext_5);
    uint rLast_0 = min(r0_1 + 8U, rows_1) - 1U;
    uint _S39 = kvHead_1 * hpk_0;
    uint _S40 = r0_1 / (&kernelContext_5)->p_0->queries_0;
    uint _S41 = _S39 + _S40;
    uint _S42 = rLast_0 / (&kernelContext_5)->p_0->queries_0;
    uint _S43 = maxValidKeys_0(_S41, _S39 + _S42, &kernelContext_5);
    uint kEnd_0;
    if(((&kernelContext_5)->p_0->causal_0) != 0U)
    {
        uint _S44 = r0_1 / (&kernelContext_5)->p_0->queries_0;
        uint _S45 = rLast_0 / (&kernelContext_5)->p_0->queries_0;
        if(_S44 != _S45)
        {
            kEnd_0 = (&kernelContext_5)->p_0->queries_0 - 1U;
        }
        else
        {
            uint _S46 = rLast_0 % (&kernelContext_5)->p_0->queries_0;
            kEnd_0 = _S46;
        }
        kEnd_0 = min(_S43, kEnd_0 + offset_0 + 1U);
    }
    else
    {
        kEnd_0 = _S43;
    }
    uint k0_2 = 0U;
    for(;;)
    {
        if(k0_2 < kEnd_0)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loadKeys_0(t_3, kvHead_1, k0_2, d2_3, kEnd_0, &kernelContext_5);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pairs_0(t_3, k0_2, d2_3, &kernelContext_5);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(owner_0 < 8U)
        {
            uint kk_2 = 0U;
            for(;;)
            {
                if(kk_2 < 16U)
                {
                }
                else
                {
                    break;
                }
                float _S47 = (*(&kernelContext_5)->dss_0)[owner_0 * 16U + kk_2];
                uint _S48 = kk_2 * 129U;
                a0_0[0U] = a0_0[0U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + _S38 / 2U], _S38);
                uint col_0 = _S38 + 8U;
                a0_0[1U] = a0_0[1U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_0 / 2U], col_0);
                uint col_1 = _S38 + 16U;
                a0_0[2U] = a0_0[2U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_1 / 2U], col_1);
                uint col_2 = _S38 + 24U;
                a0_0[3U] = a0_0[3U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_2 / 2U], col_2);
                uint col_3 = _S38 + 32U;
                a0_0[4U] = a0_0[4U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_3 / 2U], col_3);
                uint col_4 = _S38 + 40U;
                a0_0[5U] = a0_0[5U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_4 / 2U], col_4);
                uint col_5 = _S38 + 48U;
                a0_0[6U] = a0_0[6U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_5 / 2U], col_5);
                uint col_6 = _S38 + 56U;
                a0_0[7U] = a0_0[7U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_6 / 2U], col_6);
                uint col_7 = _S38 + 64U;
                a0_0[8U] = a0_0[8U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_7 / 2U], col_7);
                uint col_8 = _S38 + 72U;
                a0_0[9U] = a0_0[9U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_8 / 2U], col_8);
                uint col_9 = _S38 + 80U;
                a0_0[10U] = a0_0[10U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_9 / 2U], col_9);
                uint col_10 = _S38 + 88U;
                a0_0[11U] = a0_0[11U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_10 / 2U], col_10);
                uint col_11 = _S38 + 96U;
                a0_0[12U] = a0_0[12U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_11 / 2U], col_11);
                uint col_12 = _S38 + 104U;
                a0_0[13U] = a0_0[13U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_12 / 2U], col_12);
                uint col_13 = _S38 + 112U;
                a0_0[14U] = a0_0[14U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_13 / 2U], col_13);
                uint col_14 = _S38 + 120U;
                a0_0[15U] = a0_0[15U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_14 / 2U], col_14);
                uint col_15 = _S38 + 128U;
                a0_0[16U] = a0_0[16U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_15 / 2U], col_15);
                uint col_16 = _S38 + 136U;
                a0_0[17U] = a0_0[17U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_16 / 2U], col_16);
                uint col_17 = _S38 + 144U;
                a0_0[18U] = a0_0[18U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_17 / 2U], col_17);
                uint col_18 = _S38 + 152U;
                a0_0[19U] = a0_0[19U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_18 / 2U], col_18);
                uint col_19 = _S38 + 160U;
                a0_0[20U] = a0_0[20U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_19 / 2U], col_19);
                uint col_20 = _S38 + 168U;
                a0_0[21U] = a0_0[21U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_20 / 2U], col_20);
                uint col_21 = _S38 + 176U;
                a0_0[22U] = a0_0[22U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_21 / 2U], col_21);
                uint col_22 = _S38 + 184U;
                a0_0[23U] = a0_0[23U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_22 / 2U], col_22);
                uint col_23 = _S38 + 192U;
                a0_0[24U] = a0_0[24U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_23 / 2U], col_23);
                uint col_24 = _S38 + 200U;
                a0_0[25U] = a0_0[25U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_24 / 2U], col_24);
                uint col_25 = _S38 + 208U;
                a0_0[26U] = a0_0[26U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_25 / 2U], col_25);
                uint col_26 = _S38 + 216U;
                a0_0[27U] = a0_0[27U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_26 / 2U], col_26);
                uint col_27 = _S38 + 224U;
                a0_0[28U] = a0_0[28U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_27 / 2U], col_27);
                uint col_28 = _S38 + 232U;
                a0_0[29U] = a0_0[29U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_28 / 2U], col_28);
                uint col_29 = _S38 + 240U;
                a0_0[30U] = a0_0[30U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_29 / 2U], col_29);
                uint col_30 = _S38 + 248U;
                a0_0[31U] = a0_0[31U] + _S47 * half2At_0((*(&kernelContext_5)->ks_0)[_S48 + col_30 / 2U], col_30);
                kk_2 = kk_2 + 1U;
            }
        }
        k0_2 = k0_2 + 16U;
    }
    uint r_2 = r0_1 + owner_0;
    bool _S49;
    if(owner_0 < 8U)
    {
        _S49 = r_2 < rows_1;
    }
    else
    {
        _S49 = false;
    }
    if(_S49)
    {
        uint _S50 = base_1 + r_2;
        uint _S51 = _S50 * p_1->headDim_0;
        uint _S52 = validKeys_0(_S50, &kernelContext_5);
        bool _S53 = _S52 != 0U;
        float _S54;
        if(_S38 < (p_1->headDim_0))
        {
            float device* _S55 = (&kernelContext_5)->out0_0+(_S51 + _S38);
            if(_S53)
            {
                _S54 = a0_0[0U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S55 = _S54;
        }
        uint col_31 = _S38 + 8U;
        if(col_31 < (p_1->headDim_0))
        {
            float device* _S56 = (&kernelContext_5)->out0_0+(_S51 + col_31);
            if(_S53)
            {
                _S54 = a0_0[1U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S56 = _S54;
        }
        uint col_32 = _S38 + 16U;
        if(col_32 < (p_1->headDim_0))
        {
            float device* _S57 = (&kernelContext_5)->out0_0+(_S51 + col_32);
            if(_S53)
            {
                _S54 = a0_0[2U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S57 = _S54;
        }
        uint col_33 = _S38 + 24U;
        if(col_33 < (p_1->headDim_0))
        {
            float device* _S58 = (&kernelContext_5)->out0_0+(_S51 + col_33);
            if(_S53)
            {
                _S54 = a0_0[3U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S58 = _S54;
        }
        uint col_34 = _S38 + 32U;
        if(col_34 < (p_1->headDim_0))
        {
            float device* _S59 = (&kernelContext_5)->out0_0+(_S51 + col_34);
            if(_S53)
            {
                _S54 = a0_0[4U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S59 = _S54;
        }
        uint col_35 = _S38 + 40U;
        if(col_35 < (p_1->headDim_0))
        {
            float device* _S60 = (&kernelContext_5)->out0_0+(_S51 + col_35);
            if(_S53)
            {
                _S54 = a0_0[5U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S60 = _S54;
        }
        uint col_36 = _S38 + 48U;
        if(col_36 < (p_1->headDim_0))
        {
            float device* _S61 = (&kernelContext_5)->out0_0+(_S51 + col_36);
            if(_S53)
            {
                _S54 = a0_0[6U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S61 = _S54;
        }
        uint col_37 = _S38 + 56U;
        if(col_37 < (p_1->headDim_0))
        {
            float device* _S62 = (&kernelContext_5)->out0_0+(_S51 + col_37);
            if(_S53)
            {
                _S54 = a0_0[7U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S62 = _S54;
        }
        uint col_38 = _S38 + 64U;
        if(col_38 < (p_1->headDim_0))
        {
            float device* _S63 = (&kernelContext_5)->out0_0+(_S51 + col_38);
            if(_S53)
            {
                _S54 = a0_0[8U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S63 = _S54;
        }
        uint col_39 = _S38 + 72U;
        if(col_39 < (p_1->headDim_0))
        {
            float device* _S64 = (&kernelContext_5)->out0_0+(_S51 + col_39);
            if(_S53)
            {
                _S54 = a0_0[9U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S64 = _S54;
        }
        uint col_40 = _S38 + 80U;
        if(col_40 < (p_1->headDim_0))
        {
            float device* _S65 = (&kernelContext_5)->out0_0+(_S51 + col_40);
            if(_S53)
            {
                _S54 = a0_0[10U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S65 = _S54;
        }
        uint col_41 = _S38 + 88U;
        if(col_41 < (p_1->headDim_0))
        {
            float device* _S66 = (&kernelContext_5)->out0_0+(_S51 + col_41);
            if(_S53)
            {
                _S54 = a0_0[11U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S66 = _S54;
        }
        uint col_42 = _S38 + 96U;
        if(col_42 < (p_1->headDim_0))
        {
            float device* _S67 = (&kernelContext_5)->out0_0+(_S51 + col_42);
            if(_S53)
            {
                _S54 = a0_0[12U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S67 = _S54;
        }
        uint col_43 = _S38 + 104U;
        if(col_43 < (p_1->headDim_0))
        {
            float device* _S68 = (&kernelContext_5)->out0_0+(_S51 + col_43);
            if(_S53)
            {
                _S54 = a0_0[13U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S68 = _S54;
        }
        uint col_44 = _S38 + 112U;
        if(col_44 < (p_1->headDim_0))
        {
            float device* _S69 = (&kernelContext_5)->out0_0+(_S51 + col_44);
            if(_S53)
            {
                _S54 = a0_0[14U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S69 = _S54;
        }
        uint col_45 = _S38 + 120U;
        if(col_45 < (p_1->headDim_0))
        {
            float device* _S70 = (&kernelContext_5)->out0_0+(_S51 + col_45);
            if(_S53)
            {
                _S54 = a0_0[15U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S70 = _S54;
        }
        uint col_46 = _S38 + 128U;
        if(col_46 < (p_1->headDim_0))
        {
            float device* _S71 = (&kernelContext_5)->out0_0+(_S51 + col_46);
            if(_S53)
            {
                _S54 = a0_0[16U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S71 = _S54;
        }
        uint col_47 = _S38 + 136U;
        if(col_47 < (p_1->headDim_0))
        {
            float device* _S72 = (&kernelContext_5)->out0_0+(_S51 + col_47);
            if(_S53)
            {
                _S54 = a0_0[17U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S72 = _S54;
        }
        uint col_48 = _S38 + 144U;
        if(col_48 < (p_1->headDim_0))
        {
            float device* _S73 = (&kernelContext_5)->out0_0+(_S51 + col_48);
            if(_S53)
            {
                _S54 = a0_0[18U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S73 = _S54;
        }
        uint col_49 = _S38 + 152U;
        if(col_49 < (p_1->headDim_0))
        {
            float device* _S74 = (&kernelContext_5)->out0_0+(_S51 + col_49);
            if(_S53)
            {
                _S54 = a0_0[19U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S74 = _S54;
        }
        uint col_50 = _S38 + 160U;
        if(col_50 < (p_1->headDim_0))
        {
            float device* _S75 = (&kernelContext_5)->out0_0+(_S51 + col_50);
            if(_S53)
            {
                _S54 = a0_0[20U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S75 = _S54;
        }
        uint col_51 = _S38 + 168U;
        if(col_51 < (p_1->headDim_0))
        {
            float device* _S76 = (&kernelContext_5)->out0_0+(_S51 + col_51);
            if(_S53)
            {
                _S54 = a0_0[21U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S76 = _S54;
        }
        uint col_52 = _S38 + 176U;
        if(col_52 < (p_1->headDim_0))
        {
            float device* _S77 = (&kernelContext_5)->out0_0+(_S51 + col_52);
            if(_S53)
            {
                _S54 = a0_0[22U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S77 = _S54;
        }
        uint col_53 = _S38 + 184U;
        if(col_53 < (p_1->headDim_0))
        {
            float device* _S78 = (&kernelContext_5)->out0_0+(_S51 + col_53);
            if(_S53)
            {
                _S54 = a0_0[23U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S78 = _S54;
        }
        uint col_54 = _S38 + 192U;
        if(col_54 < (p_1->headDim_0))
        {
            float device* _S79 = (&kernelContext_5)->out0_0+(_S51 + col_54);
            if(_S53)
            {
                _S54 = a0_0[24U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S79 = _S54;
        }
        uint col_55 = _S38 + 200U;
        if(col_55 < (p_1->headDim_0))
        {
            float device* _S80 = (&kernelContext_5)->out0_0+(_S51 + col_55);
            if(_S53)
            {
                _S54 = a0_0[25U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S80 = _S54;
        }
        uint col_56 = _S38 + 208U;
        if(col_56 < (p_1->headDim_0))
        {
            float device* _S81 = (&kernelContext_5)->out0_0+(_S51 + col_56);
            if(_S53)
            {
                _S54 = a0_0[26U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S81 = _S54;
        }
        uint col_57 = _S38 + 216U;
        if(col_57 < (p_1->headDim_0))
        {
            float device* _S82 = (&kernelContext_5)->out0_0+(_S51 + col_57);
            if(_S53)
            {
                _S54 = a0_0[27U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S82 = _S54;
        }
        uint col_58 = _S38 + 224U;
        if(col_58 < (p_1->headDim_0))
        {
            float device* _S83 = (&kernelContext_5)->out0_0+(_S51 + col_58);
            if(_S53)
            {
                _S54 = a0_0[28U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S83 = _S54;
        }
        uint col_59 = _S38 + 232U;
        if(col_59 < (p_1->headDim_0))
        {
            float device* _S84 = (&kernelContext_5)->out0_0+(_S51 + col_59);
            if(_S53)
            {
                _S54 = a0_0[29U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S84 = _S54;
        }
        uint col_60 = _S38 + 240U;
        if(col_60 < (p_1->headDim_0))
        {
            float device* _S85 = (&kernelContext_5)->out0_0+(_S51 + col_60);
            if(_S53)
            {
                _S54 = a0_0[30U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S85 = _S54;
        }
        uint col_61 = _S38 + 248U;
        if(col_61 < (p_1->headDim_0))
        {
            float device* _S86 = (&kernelContext_5)->out0_0+(_S51 + col_61);
            if(_S53)
            {
                _S54 = a0_0[31U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S54 = 0.0;
            }
            *_S86 = _S54;
        }
    }
    return;
}

