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
    uint pad0_0;
    uint pad1_0;
    uint pad2_0;
};

struct KernelContext_0
{
    AttnBwdParams_0 constant* p_0;
    uint device* k_0;
    uint device* v_0;
    uint device* q_0;
    float device* dO_0;
    float device* lse_0;
    float device* dD_0;
    float device* out0_0;
    float device* out1_0;
    array<uint, int(528)> threadgroup* ks_0;
    array<uint, int(528)> threadgroup* vs_0;
    array<uint, int(528)> threadgroup* qs_0;
    array<float, int(1040)> threadgroup* dos_0;
    array<float, int(16)> threadgroup* ls_0;
    array<float, int(16)> threadgroup* ds_0;
    array<float, int(256)> threadgroup* ps_0;
    array<float, int(256)> threadgroup* dss_0;
};

void loadKeys_0(uint t_0, uint kvHead_0, uint k0_0, uint d2_0, KernelContext_0 thread* kernelContext_0)
{
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
        uint kk_0 = e_0 / 32U;
        uint pr_0 = e_0 % 32U;
        uint key_0 = k0_0 + kk_0;
        bool ok_0;
        if(key_0 < (kernelContext_0->p_0->keys_0))
        {
            ok_0 = pr_0 < d2_0;
        }
        else
        {
            ok_0 = false;
        }
        uint at_0 = (kvHead_0 * kernelContext_0->p_0->cacheKeys_0 + key_0) * d2_0 + pr_0;
        uint _S1 = kk_0 * 33U + pr_0;
        uint _S2;
        if(ok_0)
        {
            _S2 = *(kernelContext_0->k_0+at_0);
        }
        else
        {
            _S2 = 0U;
        }
        (*kernelContext_0->ks_0)[_S1] = _S2;
        uint _S3;
        if(ok_0)
        {
            _S3 = *(kernelContext_0->v_0+at_0);
        }
        else
        {
            _S3 = 0U;
        }
        (*kernelContext_0->vs_0)[_S1] = _S3;
        e_0 = e_0 + 128U;
    }
    return;
}

void loadRows_0(uint t_1, uint base_0, uint r0_0, uint rows_0, uint d2_1, KernelContext_0 thread* kernelContext_1)
{
    float _S4;
    uint e_1 = t_1;
    for(;;)
    {
        if(e_1 < 512U)
        {
        }
        else
        {
            break;
        }
        uint i_0 = e_1 / 32U;
        uint pr_1 = e_1 % 32U;
        uint r_0 = r0_0 + i_0;
        bool ok_1;
        if(r_0 < rows_0)
        {
            ok_1 = pr_1 < d2_1;
        }
        else
        {
            ok_1 = false;
        }
        uint _S5;
        if(ok_1)
        {
            _S5 = *(kernelContext_1->q_0+((base_0 + r_0) * d2_1 + pr_1));
        }
        else
        {
            _S5 = 0U;
        }
        (*kernelContext_1->qs_0)[i_0 * 33U + pr_1] = _S5;
        uint _S6 = 2U * pr_1;
        uint _S7 = i_0 * 65U + _S6;
        if(ok_1)
        {
            _S4 = *(kernelContext_1->dO_0+((base_0 + r_0) * kernelContext_1->p_0->headDim_0 + _S6));
        }
        else
        {
            _S4 = 0.0;
        }
        (*kernelContext_1->dos_0)[_S7] = _S4;
        float _S8;
        if(ok_1)
        {
            _S8 = *(kernelContext_1->dO_0+((base_0 + r_0) * kernelContext_1->p_0->headDim_0 + _S6 + 1U));
        }
        else
        {
            _S8 = 0.0;
        }
        (*kernelContext_1->dos_0)[_S7 + 1U] = _S8;
        e_1 = e_1 + 128U;
    }
    if(t_1 < 16U)
    {
        uint r_1 = r0_0 + t_1;
        bool _S9 = r_1 < rows_0;
        if(_S9)
        {
            _S4 = *(kernelContext_1->lse_0+(base_0 + r_1));
        }
        else
        {
            _S4 = 0.0;
        }
        (*kernelContext_1->ls_0)[t_1] = _S4;
        if(_S9)
        {
            _S4 = *(kernelContext_1->dD_0+(base_0 + r_1));
        }
        else
        {
            _S4 = 0.0;
        }
        (*kernelContext_1->ds_0)[t_1] = _S4;
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

void pairs_0(uint t_2, uint r0_1, uint rows_1, uint k0_1, uint d2_2, KernelContext_0 thread* kernelContext_2)
{
    uint _S10 = kernelContext_2->p_0->keys_0 - kernelContext_2->p_0->queries_0;
    thread array<uint, int(2)> ii_0;
    ii_0[0U] = t_2 / 16U;
    thread array<uint, int(2)> kv_0;
    kv_0[0U] = t_2 % 16U;
    thread array<float, int(2)> sv_0;
    sv_0[0U] = 0.0;
    thread array<float, int(2)> dp_0;
    dp_0[0U] = 0.0;
    uint e_2 = t_2 + 128U;
    ii_0[1U] = e_2 / 16U;
    kv_0[1U] = e_2 % 16U;
    sv_0[1U] = 0.0;
    dp_0[1U] = 0.0;
    uint p0_0 = 0U;
    for(;;)
    {
        if(p0_0 < 32U)
        {
        }
        else
        {
            break;
        }
        if(p0_0 < d2_2)
        {
            uint vw_0 = (*kernelContext_2->vs_0)[kv_0[0U] * 33U + p0_0];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + p0_0]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + p0_0]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + p0_0]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + p0_0]), sv_0[0U]));
            uint _S11 = 2U * p0_0;
            float _S12 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S11] * attnLo_0(vw_0);
            dp_0[0U] = _S12;
            dp_0[0U] = _S12 + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S11 + 1U] * attnHi_0(vw_0);
            uint vw_1 = (*kernelContext_2->vs_0)[kv_0[1U] * 33U + p0_0];
            sv_0[1U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + p0_0]), attnHi_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + p0_0]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + p0_0]), attnLo_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + p0_0]), sv_0[1U]));
            float _S13 = dp_0[1U] + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S11] * attnLo_0(vw_1);
            dp_0[1U] = _S13;
            dp_0[1U] = _S13 + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S11 + 1U] * attnHi_0(vw_1);
        }
        uint pr_2 = p0_0 + 1U;
        if(pr_2 < d2_2)
        {
            uint vw_2 = (*kernelContext_2->vs_0)[kv_0[0U] * 33U + pr_2];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_2]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_2]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_2]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_2]), sv_0[0U]));
            uint _S14 = 2U * pr_2;
            float _S15 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S14] * attnLo_0(vw_2);
            dp_0[0U] = _S15;
            dp_0[0U] = _S15 + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S14 + 1U] * attnHi_0(vw_2);
            uint vw_3 = (*kernelContext_2->vs_0)[kv_0[1U] * 33U + pr_2];
            sv_0[1U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_2]), attnHi_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_2]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_2]), attnLo_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_2]), sv_0[1U]));
            float _S16 = dp_0[1U] + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S14] * attnLo_0(vw_3);
            dp_0[1U] = _S16;
            dp_0[1U] = _S16 + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S14 + 1U] * attnHi_0(vw_3);
        }
        uint pr_3 = p0_0 + 2U;
        if(pr_3 < d2_2)
        {
            uint vw_4 = (*kernelContext_2->vs_0)[kv_0[0U] * 33U + pr_3];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_3]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_3]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_3]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_3]), sv_0[0U]));
            uint _S17 = 2U * pr_3;
            float _S18 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S17] * attnLo_0(vw_4);
            dp_0[0U] = _S18;
            dp_0[0U] = _S18 + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S17 + 1U] * attnHi_0(vw_4);
            uint vw_5 = (*kernelContext_2->vs_0)[kv_0[1U] * 33U + pr_3];
            sv_0[1U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_3]), attnHi_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_3]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_3]), attnLo_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_3]), sv_0[1U]));
            float _S19 = dp_0[1U] + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S17] * attnLo_0(vw_5);
            dp_0[1U] = _S19;
            dp_0[1U] = _S19 + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S17 + 1U] * attnHi_0(vw_5);
        }
        uint pr_4 = p0_0 + 3U;
        if(pr_4 < d2_2)
        {
            uint vw_6 = (*kernelContext_2->vs_0)[kv_0[0U] * 33U + pr_4];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_4]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_4]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_4]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_4]), sv_0[0U]));
            uint _S20 = 2U * pr_4;
            float _S21 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S20] * attnLo_0(vw_6);
            dp_0[0U] = _S21;
            dp_0[0U] = _S21 + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S20 + 1U] * attnHi_0(vw_6);
            uint vw_7 = (*kernelContext_2->vs_0)[kv_0[1U] * 33U + pr_4];
            sv_0[1U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_4]), attnHi_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_4]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_4]), attnLo_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_4]), sv_0[1U]));
            float _S22 = dp_0[1U] + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S20] * attnLo_0(vw_7);
            dp_0[1U] = _S22;
            dp_0[1U] = _S22 + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S20 + 1U] * attnHi_0(vw_7);
        }
        uint pr_5 = p0_0 + 4U;
        if(pr_5 < d2_2)
        {
            uint vw_8 = (*kernelContext_2->vs_0)[kv_0[0U] * 33U + pr_5];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_5]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_5]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_5]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_5]), sv_0[0U]));
            uint _S23 = 2U * pr_5;
            float _S24 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S23] * attnLo_0(vw_8);
            dp_0[0U] = _S24;
            dp_0[0U] = _S24 + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S23 + 1U] * attnHi_0(vw_8);
            uint vw_9 = (*kernelContext_2->vs_0)[kv_0[1U] * 33U + pr_5];
            sv_0[1U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_5]), attnHi_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_5]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_5]), attnLo_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_5]), sv_0[1U]));
            float _S25 = dp_0[1U] + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S23] * attnLo_0(vw_9);
            dp_0[1U] = _S25;
            dp_0[1U] = _S25 + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S23 + 1U] * attnHi_0(vw_9);
        }
        uint pr_6 = p0_0 + 5U;
        if(pr_6 < d2_2)
        {
            uint vw_10 = (*kernelContext_2->vs_0)[kv_0[0U] * 33U + pr_6];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_6]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_6]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_6]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_6]), sv_0[0U]));
            uint _S26 = 2U * pr_6;
            float _S27 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S26] * attnLo_0(vw_10);
            dp_0[0U] = _S27;
            dp_0[0U] = _S27 + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S26 + 1U] * attnHi_0(vw_10);
            uint vw_11 = (*kernelContext_2->vs_0)[kv_0[1U] * 33U + pr_6];
            sv_0[1U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_6]), attnHi_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_6]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_6]), attnLo_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_6]), sv_0[1U]));
            float _S28 = dp_0[1U] + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S26] * attnLo_0(vw_11);
            dp_0[1U] = _S28;
            dp_0[1U] = _S28 + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S26 + 1U] * attnHi_0(vw_11);
        }
        uint pr_7 = p0_0 + 6U;
        if(pr_7 < d2_2)
        {
            uint vw_12 = (*kernelContext_2->vs_0)[kv_0[0U] * 33U + pr_7];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_7]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_7]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_7]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_7]), sv_0[0U]));
            uint _S29 = 2U * pr_7;
            float _S30 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S29] * attnLo_0(vw_12);
            dp_0[0U] = _S30;
            dp_0[0U] = _S30 + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S29 + 1U] * attnHi_0(vw_12);
            uint vw_13 = (*kernelContext_2->vs_0)[kv_0[1U] * 33U + pr_7];
            sv_0[1U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_7]), attnHi_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_7]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_7]), attnLo_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_7]), sv_0[1U]));
            float _S31 = dp_0[1U] + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S29] * attnLo_0(vw_13);
            dp_0[1U] = _S31;
            dp_0[1U] = _S31 + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S29 + 1U] * attnHi_0(vw_13);
        }
        uint pr_8 = p0_0 + 7U;
        if(pr_8 < d2_2)
        {
            uint vw_14 = (*kernelContext_2->vs_0)[kv_0[0U] * 33U + pr_8];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_8]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_8]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 33U + pr_8]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 33U + pr_8]), sv_0[0U]));
            uint _S32 = 2U * pr_8;
            float _S33 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S32] * attnLo_0(vw_14);
            dp_0[0U] = _S33;
            dp_0[0U] = _S33 + (*kernelContext_2->dos_0)[ii_0[0U] * 65U + _S32 + 1U] * attnHi_0(vw_14);
            uint vw_15 = (*kernelContext_2->vs_0)[kv_0[1U] * 33U + pr_8];
            sv_0[1U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_8]), attnHi_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_8]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[1U] * 33U + pr_8]), attnLo_0((*kernelContext_2->ks_0)[kv_0[1U] * 33U + pr_8]), sv_0[1U]));
            float _S34 = dp_0[1U] + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S32] * attnLo_0(vw_15);
            dp_0[1U] = _S34;
            dp_0[1U] = _S34 + (*kernelContext_2->dos_0)[ii_0[1U] * 65U + _S32 + 1U] * attnHi_0(vw_15);
        }
        p0_0 = p0_0 + 8U;
    }
    uint i_1 = ii_0[0U];
    uint kk_1 = kv_0[0U];
    uint r_2 = r0_1 + ii_0[0U];
    uint key_1 = k0_1 + kv_0[0U];
    bool live_0;
    if(r_2 < rows_1)
    {
        live_0 = key_1 < (kernelContext_2->p_0->keys_0);
    }
    else
    {
        live_0 = false;
    }
    if(live_0)
    {
        if((kernelContext_2->p_0->causal_0) == 0U)
        {
            live_0 = true;
        }
        else
        {
            uint _S35 = r_2 % kernelContext_2->p_0->queries_0;
            live_0 = key_1 <= (_S35 + _S10);
        }
    }
    else
    {
        live_0 = false;
    }
    float pv_0;
    if(live_0)
    {
        pv_0 = exp(sv_0[0U] * kernelContext_2->p_0->scale_0 - (*kernelContext_2->ls_0)[i_1]);
    }
    else
    {
        pv_0 = 0.0;
    }
    uint _S36 = i_1 * 16U + kk_1;
    (*kernelContext_2->ps_0)[_S36] = pv_0;
    if(live_0)
    {
        pv_0 = pv_0 * (dp_0[0U] - (*kernelContext_2->ds_0)[i_1]);
    }
    else
    {
        pv_0 = 0.0;
    }
    (*kernelContext_2->dss_0)[_S36] = pv_0;
    uint i_2 = ii_0[1U];
    uint kk_2 = kv_0[1U];
    uint r_3 = r0_1 + ii_0[1U];
    uint key_2 = k0_1 + kv_0[1U];
    if(r_3 < rows_1)
    {
        live_0 = key_2 < (kernelContext_2->p_0->keys_0);
    }
    else
    {
        live_0 = false;
    }
    if(live_0)
    {
        if((kernelContext_2->p_0->causal_0) == 0U)
        {
            live_0 = true;
        }
        else
        {
            uint _S37 = r_3 % kernelContext_2->p_0->queries_0;
            live_0 = key_2 <= (_S37 + _S10);
        }
    }
    else
    {
        live_0 = false;
    }
    if(live_0)
    {
        pv_0 = exp(sv_0[1U] * kernelContext_2->p_0->scale_0 - (*kernelContext_2->ls_0)[i_2]);
    }
    else
    {
        pv_0 = 0.0;
    }
    uint _S38 = i_2 * 16U + kk_2;
    (*kernelContext_2->ps_0)[_S38] = pv_0;
    if(live_0)
    {
        pv_0 = pv_0 * (dp_0[1U] - (*kernelContext_2->ds_0)[i_2]);
    }
    else
    {
        pv_0 = 0.0;
    }
    (*kernelContext_2->dss_0)[_S38] = pv_0;
    return;
}

float half2At_0(uint w_2, uint c_0)
{
    float _S39;
    if((c_0 & 1U) == 0U)
    {
        _S39 = attnLo_0(w_2);
    }
    else
    {
        _S39 = attnHi_0(w_2);
    }
    return _S39;
}

[[kernel]] void attn_bwd_kv_64(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnBwdParams_0 constant* p_1 [[buffer(8)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], uint device* q_1 [[buffer(0)]], float device* dO_1 [[buffer(3)]], float device* lse_1 [[buffer(4)]], float device* dD_1 [[buffer(5)]], float device* out0_1 [[buffer(6)]], float device* out1_1 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->p_0 = p_1;
    (&kernelContext_3)->k_0 = k_1;
    (&kernelContext_3)->v_0 = v_1;
    (&kernelContext_3)->q_0 = q_1;
    (&kernelContext_3)->dO_0 = dO_1;
    (&kernelContext_3)->lse_0 = lse_1;
    (&kernelContext_3)->dD_0 = dD_1;
    (&kernelContext_3)->out0_0 = out0_1;
    (&kernelContext_3)->out1_0 = out1_1;
    threadgroup array<uint, int(528)> ks_1;
    (&kernelContext_3)->ks_0 = &ks_1;
    threadgroup array<uint, int(528)> vs_1;
    (&kernelContext_3)->vs_0 = &vs_1;
    threadgroup array<uint, int(528)> qs_1;
    (&kernelContext_3)->qs_0 = &qs_1;
    threadgroup array<float, int(1040)> dos_1;
    (&kernelContext_3)->dos_0 = &dos_1;
    threadgroup array<float, int(16)> ls_1;
    (&kernelContext_3)->ls_0 = &ls_1;
    threadgroup array<float, int(16)> ds_1;
    (&kernelContext_3)->ds_0 = &ds_1;
    threadgroup array<float, int(256)> ps_1;
    (&kernelContext_3)->ps_0 = &ps_1;
    threadgroup array<float, int(256)> dss_1;
    (&kernelContext_3)->dss_0 = &dss_1;
    uint t_3 = local_0.x;
    uint kvHead_1 = group_0.y;
    uint d2_3 = p_1->headDim_0 / 2U;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint rows_2 = hpk_0 * (&kernelContext_3)->p_0->queries_0;
    uint _S40 = kvHead_1 * rows_2;
    uint _S41 = (&kernelContext_3)->p_0->keys_0 - (&kernelContext_3)->p_0->queries_0;
    uint owner_0 = t_3 / 8U;
    uint _S42 = t_3 % 8U;
    thread array<float, int(8)> a0_0;
    a0_0[0U] = 0.0;
    thread array<float, int(8)> a1_0;
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
    uint k0_2 = group_0.x * 16U;
    loadKeys_0(t_3, kvHead_1, k0_2, d2_3, &kernelContext_3);
    uint r0_2 = 0U;
    for(;;)
    {
        if(r0_2 < rows_2)
        {
        }
        else
        {
            break;
        }
        uint i_3;
        if(((&kernelContext_3)->p_0->causal_0) != 0U)
        {
            uint rLast_0 = min(r0_2 + 16U, rows_2) - 1U;
            uint _S43 = r0_2 / (&kernelContext_3)->p_0->queries_0;
            uint _S44 = rLast_0 / (&kernelContext_3)->p_0->queries_0;
            if(_S43 != _S44)
            {
                i_3 = (&kernelContext_3)->p_0->queries_0 - 1U;
            }
            else
            {
                uint _S45 = rLast_0 % (&kernelContext_3)->p_0->queries_0;
                i_3 = _S45;
            }
            if((i_3 + _S41) < k0_2)
            {
                r0_2 = r0_2 + 16U;
                continue;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loadRows_0(t_3, _S40, r0_2, rows_2, d2_3, &kernelContext_3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pairs_0(t_3, r0_2, rows_2, k0_2, d2_3, &kernelContext_3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_3 = 0U;
        for(;;)
        {
            if(i_3 < 16U)
            {
            }
            else
            {
                break;
            }
            uint _S46 = i_3 * 16U + owner_0;
            float _S47 = (*(&kernelContext_3)->ps_0)[_S46];
            float _S48 = (*(&kernelContext_3)->dss_0)[_S46];
            uint _S49 = i_3 * 65U;
            a1_0[0U] = a1_0[0U] + _S47 * (*(&kernelContext_3)->dos_0)[_S49 + _S42];
            uint _S50 = i_3 * 33U;
            a0_0[0U] = a0_0[0U] + _S48 * half2At_0((*(&kernelContext_3)->qs_0)[_S50 + _S42 / 2U], _S42);
            uint col_0 = _S42 + 8U;
            a1_0[1U] = a1_0[1U] + _S47 * (*(&kernelContext_3)->dos_0)[_S49 + col_0];
            a0_0[1U] = a0_0[1U] + _S48 * half2At_0((*(&kernelContext_3)->qs_0)[_S50 + col_0 / 2U], col_0);
            uint col_1 = _S42 + 16U;
            a1_0[2U] = a1_0[2U] + _S47 * (*(&kernelContext_3)->dos_0)[_S49 + col_1];
            a0_0[2U] = a0_0[2U] + _S48 * half2At_0((*(&kernelContext_3)->qs_0)[_S50 + col_1 / 2U], col_1);
            uint col_2 = _S42 + 24U;
            a1_0[3U] = a1_0[3U] + _S47 * (*(&kernelContext_3)->dos_0)[_S49 + col_2];
            a0_0[3U] = a0_0[3U] + _S48 * half2At_0((*(&kernelContext_3)->qs_0)[_S50 + col_2 / 2U], col_2);
            uint col_3 = _S42 + 32U;
            a1_0[4U] = a1_0[4U] + _S47 * (*(&kernelContext_3)->dos_0)[_S49 + col_3];
            a0_0[4U] = a0_0[4U] + _S48 * half2At_0((*(&kernelContext_3)->qs_0)[_S50 + col_3 / 2U], col_3);
            uint col_4 = _S42 + 40U;
            a1_0[5U] = a1_0[5U] + _S47 * (*(&kernelContext_3)->dos_0)[_S49 + col_4];
            a0_0[5U] = a0_0[5U] + _S48 * half2At_0((*(&kernelContext_3)->qs_0)[_S50 + col_4 / 2U], col_4);
            uint col_5 = _S42 + 48U;
            a1_0[6U] = a1_0[6U] + _S47 * (*(&kernelContext_3)->dos_0)[_S49 + col_5];
            a0_0[6U] = a0_0[6U] + _S48 * half2At_0((*(&kernelContext_3)->qs_0)[_S50 + col_5 / 2U], col_5);
            uint col_6 = _S42 + 56U;
            a1_0[7U] = a1_0[7U] + _S47 * (*(&kernelContext_3)->dos_0)[_S49 + col_6];
            a0_0[7U] = a0_0[7U] + _S48 * half2At_0((*(&kernelContext_3)->qs_0)[_S50 + col_6 / 2U], col_6);
            i_3 = i_3 + 1U;
        }
        r0_2 = r0_2 + 16U;
    }
    uint key_3 = k0_2 + owner_0;
    if(key_3 < ((&kernelContext_3)->p_0->keys_0))
    {
        uint _S51 = (kvHead_1 * (&kernelContext_3)->p_0->cacheKeys_0 + key_3) * p_1->headDim_0;
        if(_S42 < (p_1->headDim_0))
        {
            uint _S52 = _S51 + _S42;
            *((&kernelContext_3)->out0_0+_S52) = a0_0[0U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S52) = a1_0[0U];
        }
        uint col_7 = _S42 + 8U;
        if(col_7 < (p_1->headDim_0))
        {
            uint _S53 = _S51 + col_7;
            *((&kernelContext_3)->out0_0+_S53) = a0_0[1U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S53) = a1_0[1U];
        }
        uint col_8 = _S42 + 16U;
        if(col_8 < (p_1->headDim_0))
        {
            uint _S54 = _S51 + col_8;
            *((&kernelContext_3)->out0_0+_S54) = a0_0[2U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S54) = a1_0[2U];
        }
        uint col_9 = _S42 + 24U;
        if(col_9 < (p_1->headDim_0))
        {
            uint _S55 = _S51 + col_9;
            *((&kernelContext_3)->out0_0+_S55) = a0_0[3U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S55) = a1_0[3U];
        }
        uint col_10 = _S42 + 32U;
        if(col_10 < (p_1->headDim_0))
        {
            uint _S56 = _S51 + col_10;
            *((&kernelContext_3)->out0_0+_S56) = a0_0[4U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S56) = a1_0[4U];
        }
        uint col_11 = _S42 + 40U;
        if(col_11 < (p_1->headDim_0))
        {
            uint _S57 = _S51 + col_11;
            *((&kernelContext_3)->out0_0+_S57) = a0_0[5U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S57) = a1_0[5U];
        }
        uint col_12 = _S42 + 48U;
        if(col_12 < (p_1->headDim_0))
        {
            uint _S58 = _S51 + col_12;
            *((&kernelContext_3)->out0_0+_S58) = a0_0[6U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S58) = a1_0[6U];
        }
        uint col_13 = _S42 + 56U;
        if(col_13 < (p_1->headDim_0))
        {
            uint _S59 = _S51 + col_13;
            *((&kernelContext_3)->out0_0+_S59) = a0_0[7U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S59) = a1_0[7U];
        }
    }
    return;
}

