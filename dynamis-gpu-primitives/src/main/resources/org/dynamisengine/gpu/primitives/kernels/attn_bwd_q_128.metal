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
    array<uint, int(1040)> threadgroup* qs_0;
    array<float, int(2064)> threadgroup* dos_0;
    array<float, int(16)> threadgroup* ls_0;
    array<float, int(16)> threadgroup* ds_0;
    array<uint, int(1040)> threadgroup* ks_0;
    array<uint, int(1040)> threadgroup* vs_0;
    array<float, int(256)> threadgroup* ps_0;
    array<float, int(256)> threadgroup* dss_0;
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
        uint i_0 = e_0 / 64U;
        uint pr_0 = e_0 % 64U;
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
        (*kernelContext_1->qs_0)[i_0 * 65U + pr_0] = end_0;
        uint _S7 = 2U * pr_0;
        uint _S8 = i_0 * 129U + _S7;
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
    if(t_0 < 16U)
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
        (*kernelContext_1->qs_0)[t_0 * 65U + 64U] = end_0;
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
        if(e_1 < 1024U)
        {
        }
        else
        {
            break;
        }
        uint kk_0 = e_1 / 64U;
        uint pr_1 = e_1 % 64U;
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
        uint _S17 = kk_0 * 65U + pr_1;
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
        if(p0_0 < 64U)
        {
        }
        else
        {
            break;
        }
        if(p0_0 < d2_2)
        {
            uint vw_0 = (*kernelContext_4->vs_0)[kv_0[0U] * 65U + p0_0];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + p0_0]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + p0_0]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + p0_0]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + p0_0]), sv_0[0U]));
            uint _S20 = 2U * p0_0;
            float _S21 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S20] * attnLo_0(vw_0);
            dp_0[0U] = _S21;
            dp_0[0U] = _S21 + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S20 + 1U] * attnHi_0(vw_0);
            uint vw_1 = (*kernelContext_4->vs_0)[kv_0[1U] * 65U + p0_0];
            sv_0[1U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + p0_0]), attnHi_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + p0_0]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + p0_0]), attnLo_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + p0_0]), sv_0[1U]));
            float _S22 = dp_0[1U] + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S20] * attnLo_0(vw_1);
            dp_0[1U] = _S22;
            dp_0[1U] = _S22 + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S20 + 1U] * attnHi_0(vw_1);
        }
        uint pr_2 = p0_0 + 1U;
        if(pr_2 < d2_2)
        {
            uint vw_2 = (*kernelContext_4->vs_0)[kv_0[0U] * 65U + pr_2];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_2]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_2]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_2]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_2]), sv_0[0U]));
            uint _S23 = 2U * pr_2;
            float _S24 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S23] * attnLo_0(vw_2);
            dp_0[0U] = _S24;
            dp_0[0U] = _S24 + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S23 + 1U] * attnHi_0(vw_2);
            uint vw_3 = (*kernelContext_4->vs_0)[kv_0[1U] * 65U + pr_2];
            sv_0[1U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_2]), attnHi_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_2]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_2]), attnLo_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_2]), sv_0[1U]));
            float _S25 = dp_0[1U] + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S23] * attnLo_0(vw_3);
            dp_0[1U] = _S25;
            dp_0[1U] = _S25 + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S23 + 1U] * attnHi_0(vw_3);
        }
        uint pr_3 = p0_0 + 2U;
        if(pr_3 < d2_2)
        {
            uint vw_4 = (*kernelContext_4->vs_0)[kv_0[0U] * 65U + pr_3];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_3]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_3]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_3]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_3]), sv_0[0U]));
            uint _S26 = 2U * pr_3;
            float _S27 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S26] * attnLo_0(vw_4);
            dp_0[0U] = _S27;
            dp_0[0U] = _S27 + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S26 + 1U] * attnHi_0(vw_4);
            uint vw_5 = (*kernelContext_4->vs_0)[kv_0[1U] * 65U + pr_3];
            sv_0[1U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_3]), attnHi_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_3]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_3]), attnLo_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_3]), sv_0[1U]));
            float _S28 = dp_0[1U] + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S26] * attnLo_0(vw_5);
            dp_0[1U] = _S28;
            dp_0[1U] = _S28 + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S26 + 1U] * attnHi_0(vw_5);
        }
        uint pr_4 = p0_0 + 3U;
        if(pr_4 < d2_2)
        {
            uint vw_6 = (*kernelContext_4->vs_0)[kv_0[0U] * 65U + pr_4];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_4]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_4]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_4]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_4]), sv_0[0U]));
            uint _S29 = 2U * pr_4;
            float _S30 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S29] * attnLo_0(vw_6);
            dp_0[0U] = _S30;
            dp_0[0U] = _S30 + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S29 + 1U] * attnHi_0(vw_6);
            uint vw_7 = (*kernelContext_4->vs_0)[kv_0[1U] * 65U + pr_4];
            sv_0[1U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_4]), attnHi_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_4]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_4]), attnLo_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_4]), sv_0[1U]));
            float _S31 = dp_0[1U] + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S29] * attnLo_0(vw_7);
            dp_0[1U] = _S31;
            dp_0[1U] = _S31 + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S29 + 1U] * attnHi_0(vw_7);
        }
        uint pr_5 = p0_0 + 4U;
        if(pr_5 < d2_2)
        {
            uint vw_8 = (*kernelContext_4->vs_0)[kv_0[0U] * 65U + pr_5];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_5]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_5]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_5]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_5]), sv_0[0U]));
            uint _S32 = 2U * pr_5;
            float _S33 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S32] * attnLo_0(vw_8);
            dp_0[0U] = _S33;
            dp_0[0U] = _S33 + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S32 + 1U] * attnHi_0(vw_8);
            uint vw_9 = (*kernelContext_4->vs_0)[kv_0[1U] * 65U + pr_5];
            sv_0[1U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_5]), attnHi_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_5]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_5]), attnLo_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_5]), sv_0[1U]));
            float _S34 = dp_0[1U] + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S32] * attnLo_0(vw_9);
            dp_0[1U] = _S34;
            dp_0[1U] = _S34 + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S32 + 1U] * attnHi_0(vw_9);
        }
        uint pr_6 = p0_0 + 5U;
        if(pr_6 < d2_2)
        {
            uint vw_10 = (*kernelContext_4->vs_0)[kv_0[0U] * 65U + pr_6];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_6]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_6]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_6]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_6]), sv_0[0U]));
            uint _S35 = 2U * pr_6;
            float _S36 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S35] * attnLo_0(vw_10);
            dp_0[0U] = _S36;
            dp_0[0U] = _S36 + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S35 + 1U] * attnHi_0(vw_10);
            uint vw_11 = (*kernelContext_4->vs_0)[kv_0[1U] * 65U + pr_6];
            sv_0[1U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_6]), attnHi_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_6]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_6]), attnLo_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_6]), sv_0[1U]));
            float _S37 = dp_0[1U] + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S35] * attnLo_0(vw_11);
            dp_0[1U] = _S37;
            dp_0[1U] = _S37 + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S35 + 1U] * attnHi_0(vw_11);
        }
        uint pr_7 = p0_0 + 6U;
        if(pr_7 < d2_2)
        {
            uint vw_12 = (*kernelContext_4->vs_0)[kv_0[0U] * 65U + pr_7];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_7]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_7]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_7]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_7]), sv_0[0U]));
            uint _S38 = 2U * pr_7;
            float _S39 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S38] * attnLo_0(vw_12);
            dp_0[0U] = _S39;
            dp_0[0U] = _S39 + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S38 + 1U] * attnHi_0(vw_12);
            uint vw_13 = (*kernelContext_4->vs_0)[kv_0[1U] * 65U + pr_7];
            sv_0[1U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_7]), attnHi_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_7]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_7]), attnLo_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_7]), sv_0[1U]));
            float _S40 = dp_0[1U] + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S38] * attnLo_0(vw_13);
            dp_0[1U] = _S40;
            dp_0[1U] = _S40 + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S38 + 1U] * attnHi_0(vw_13);
        }
        uint pr_8 = p0_0 + 7U;
        if(pr_8 < d2_2)
        {
            uint vw_14 = (*kernelContext_4->vs_0)[kv_0[0U] * 65U + pr_8];
            sv_0[0U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_8]), attnHi_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_8]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[0U] * 65U + pr_8]), attnLo_0((*kernelContext_4->ks_0)[kv_0[0U] * 65U + pr_8]), sv_0[0U]));
            uint _S41 = 2U * pr_8;
            float _S42 = dp_0[0U] + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S41] * attnLo_0(vw_14);
            dp_0[0U] = _S42;
            dp_0[0U] = _S42 + (*kernelContext_4->dos_0)[ii_0[0U] * 129U + _S41 + 1U] * attnHi_0(vw_14);
            uint vw_15 = (*kernelContext_4->vs_0)[kv_0[1U] * 65U + pr_8];
            sv_0[1U] = fma(attnHi_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_8]), attnHi_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_8]), fma(attnLo_0((*kernelContext_4->qs_0)[ii_0[1U] * 65U + pr_8]), attnLo_0((*kernelContext_4->ks_0)[kv_0[1U] * 65U + pr_8]), sv_0[1U]));
            float _S43 = dp_0[1U] + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S41] * attnLo_0(vw_15);
            dp_0[1U] = _S43;
            dp_0[1U] = _S43 + (*kernelContext_4->dos_0)[ii_0[1U] * 129U + _S41 + 1U] * attnHi_0(vw_15);
        }
        p0_0 = p0_0 + 8U;
    }
    uint i_1 = ii_0[0U];
    uint kk_1 = kv_0[0U];
    bool live_0 = (k0_1 + kv_0[0U]) < (*kernelContext_4->qs_0)[ii_0[0U] * 65U + 64U];
    float pv_0;
    if(live_0)
    {
        pv_0 = exp(sv_0[0U] * kernelContext_4->p_0->scale_0 - (*kernelContext_4->ls_0)[i_1]);
    }
    else
    {
        pv_0 = 0.0;
    }
    uint _S44 = i_1 * 16U + kk_1;
    (*kernelContext_4->ps_0)[_S44] = pv_0;
    if(live_0)
    {
        pv_0 = pv_0 * (dp_0[0U] - (*kernelContext_4->ds_0)[i_1]);
    }
    else
    {
        pv_0 = 0.0;
    }
    (*kernelContext_4->dss_0)[_S44] = pv_0;
    uint i_2 = ii_0[1U];
    uint kk_2 = kv_0[1U];
    bool live_1 = (k0_1 + kv_0[1U]) < (*kernelContext_4->qs_0)[ii_0[1U] * 65U + 64U];
    if(live_1)
    {
        pv_0 = exp(sv_0[1U] * kernelContext_4->p_0->scale_0 - (*kernelContext_4->ls_0)[i_2]);
    }
    else
    {
        pv_0 = 0.0;
    }
    uint _S45 = i_2 * 16U + kk_2;
    (*kernelContext_4->ps_0)[_S45] = pv_0;
    if(live_1)
    {
        pv_0 = pv_0 * (dp_0[1U] - (*kernelContext_4->ds_0)[i_2]);
    }
    else
    {
        pv_0 = 0.0;
    }
    (*kernelContext_4->dss_0)[_S45] = pv_0;
    return;
}

float half2At_0(uint w_2, uint c_0)
{
    float _S46;
    if((c_0 & 1U) == 0U)
    {
        _S46 = attnLo_0(w_2);
    }
    else
    {
        _S46 = attnHi_0(w_2);
    }
    return _S46;
}

[[kernel]] void attn_bwd_q_128(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnBwdParams_0 constant* p_1 [[buffer(7)]], uint device* dD_1 [[buffer(5)]], uint device* q_1 [[buffer(0)]], float device* dO_1 [[buffer(3)]], float device* lse_1 [[buffer(4)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* out0_1 [[buffer(6)]])
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
    threadgroup array<uint, int(1040)> qs_1;
    (&kernelContext_5)->qs_0 = &qs_1;
    threadgroup array<float, int(2064)> dos_1;
    (&kernelContext_5)->dos_0 = &dos_1;
    threadgroup array<float, int(16)> ls_1;
    (&kernelContext_5)->ls_0 = &ls_1;
    threadgroup array<float, int(16)> ds_1;
    (&kernelContext_5)->ds_0 = &ds_1;
    threadgroup array<uint, int(1040)> ks_1;
    (&kernelContext_5)->ks_0 = &ks_1;
    threadgroup array<uint, int(1040)> vs_1;
    (&kernelContext_5)->vs_0 = &vs_1;
    threadgroup array<float, int(256)> ps_1;
    (&kernelContext_5)->ps_0 = &ps_1;
    threadgroup array<float, int(256)> dss_1;
    (&kernelContext_5)->dss_0 = &dss_1;
    uint t_3 = local_0.x;
    uint kvHead_1 = group_0.y;
    uint d2_3 = p_1->headDim_0 / 2U;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint rows_1 = hpk_0 * (&kernelContext_5)->p_0->queries_0;
    uint base_1 = kvHead_1 * rows_1;
    uint offset_0 = (&kernelContext_5)->p_0->keys_0 - (&kernelContext_5)->p_0->queries_0;
    uint owner_0 = t_3 / 8U;
    uint _S47 = t_3 % 8U;
    thread array<float, int(16)> a0_0;
    a0_0[0U] = 0.0;
    thread array<float, int(16)> a1_0;
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
    uint r0_1 = group_0.x * 16U;
    loadRows_0(t_3, base_1, r0_1, rows_1, d2_3, &kernelContext_5);
    uint rLast_0 = min(r0_1 + 16U, rows_1) - 1U;
    uint _S48 = kvHead_1 * hpk_0;
    uint _S49 = r0_1 / (&kernelContext_5)->p_0->queries_0;
    uint _S50 = _S48 + _S49;
    uint _S51 = rLast_0 / (&kernelContext_5)->p_0->queries_0;
    uint _S52 = maxValidKeys_0(_S50, _S48 + _S51, &kernelContext_5);
    uint kEnd_0;
    if(((&kernelContext_5)->p_0->causal_0) != 0U)
    {
        uint _S53 = r0_1 / (&kernelContext_5)->p_0->queries_0;
        uint _S54 = rLast_0 / (&kernelContext_5)->p_0->queries_0;
        if(_S53 != _S54)
        {
            kEnd_0 = (&kernelContext_5)->p_0->queries_0 - 1U;
        }
        else
        {
            uint _S55 = rLast_0 % (&kernelContext_5)->p_0->queries_0;
            kEnd_0 = _S55;
        }
        kEnd_0 = min(_S52, kEnd_0 + offset_0 + 1U);
    }
    else
    {
        kEnd_0 = _S52;
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
        if(owner_0 < 16U)
        {
            uint kk_3 = 0U;
            for(;;)
            {
                if(kk_3 < 16U)
                {
                }
                else
                {
                    break;
                }
                float _S56 = (*(&kernelContext_5)->dss_0)[owner_0 * 16U + kk_3];
                uint _S57 = kk_3 * 65U;
                a0_0[0U] = a0_0[0U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + _S47 / 2U], _S47);
                uint col_0 = _S47 + 8U;
                a0_0[1U] = a0_0[1U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_0 / 2U], col_0);
                uint col_1 = _S47 + 16U;
                a0_0[2U] = a0_0[2U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_1 / 2U], col_1);
                uint col_2 = _S47 + 24U;
                a0_0[3U] = a0_0[3U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_2 / 2U], col_2);
                uint col_3 = _S47 + 32U;
                a0_0[4U] = a0_0[4U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_3 / 2U], col_3);
                uint col_4 = _S47 + 40U;
                a0_0[5U] = a0_0[5U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_4 / 2U], col_4);
                uint col_5 = _S47 + 48U;
                a0_0[6U] = a0_0[6U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_5 / 2U], col_5);
                uint col_6 = _S47 + 56U;
                a0_0[7U] = a0_0[7U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_6 / 2U], col_6);
                uint col_7 = _S47 + 64U;
                a0_0[8U] = a0_0[8U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_7 / 2U], col_7);
                uint col_8 = _S47 + 72U;
                a0_0[9U] = a0_0[9U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_8 / 2U], col_8);
                uint col_9 = _S47 + 80U;
                a0_0[10U] = a0_0[10U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_9 / 2U], col_9);
                uint col_10 = _S47 + 88U;
                a0_0[11U] = a0_0[11U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_10 / 2U], col_10);
                uint col_11 = _S47 + 96U;
                a0_0[12U] = a0_0[12U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_11 / 2U], col_11);
                uint col_12 = _S47 + 104U;
                a0_0[13U] = a0_0[13U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_12 / 2U], col_12);
                uint col_13 = _S47 + 112U;
                a0_0[14U] = a0_0[14U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_13 / 2U], col_13);
                uint col_14 = _S47 + 120U;
                a0_0[15U] = a0_0[15U] + _S56 * half2At_0((*(&kernelContext_5)->ks_0)[_S57 + col_14 / 2U], col_14);
                kk_3 = kk_3 + 1U;
            }
        }
        k0_2 = k0_2 + 16U;
    }
    uint r_2 = r0_1 + owner_0;
    bool _S58;
    if(owner_0 < 16U)
    {
        _S58 = r_2 < rows_1;
    }
    else
    {
        _S58 = false;
    }
    if(_S58)
    {
        uint _S59 = base_1 + r_2;
        uint _S60 = _S59 * p_1->headDim_0;
        uint _S61 = validKeys_0(_S59, &kernelContext_5);
        bool _S62 = _S61 != 0U;
        float _S63;
        if(_S47 < (p_1->headDim_0))
        {
            float device* _S64 = (&kernelContext_5)->out0_0+(_S60 + _S47);
            if(_S62)
            {
                _S63 = a0_0[0U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S64 = _S63;
        }
        uint col_15 = _S47 + 8U;
        if(col_15 < (p_1->headDim_0))
        {
            float device* _S65 = (&kernelContext_5)->out0_0+(_S60 + col_15);
            if(_S62)
            {
                _S63 = a0_0[1U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S65 = _S63;
        }
        uint col_16 = _S47 + 16U;
        if(col_16 < (p_1->headDim_0))
        {
            float device* _S66 = (&kernelContext_5)->out0_0+(_S60 + col_16);
            if(_S62)
            {
                _S63 = a0_0[2U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S66 = _S63;
        }
        uint col_17 = _S47 + 24U;
        if(col_17 < (p_1->headDim_0))
        {
            float device* _S67 = (&kernelContext_5)->out0_0+(_S60 + col_17);
            if(_S62)
            {
                _S63 = a0_0[3U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S67 = _S63;
        }
        uint col_18 = _S47 + 32U;
        if(col_18 < (p_1->headDim_0))
        {
            float device* _S68 = (&kernelContext_5)->out0_0+(_S60 + col_18);
            if(_S62)
            {
                _S63 = a0_0[4U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S68 = _S63;
        }
        uint col_19 = _S47 + 40U;
        if(col_19 < (p_1->headDim_0))
        {
            float device* _S69 = (&kernelContext_5)->out0_0+(_S60 + col_19);
            if(_S62)
            {
                _S63 = a0_0[5U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S69 = _S63;
        }
        uint col_20 = _S47 + 48U;
        if(col_20 < (p_1->headDim_0))
        {
            float device* _S70 = (&kernelContext_5)->out0_0+(_S60 + col_20);
            if(_S62)
            {
                _S63 = a0_0[6U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S70 = _S63;
        }
        uint col_21 = _S47 + 56U;
        if(col_21 < (p_1->headDim_0))
        {
            float device* _S71 = (&kernelContext_5)->out0_0+(_S60 + col_21);
            if(_S62)
            {
                _S63 = a0_0[7U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S71 = _S63;
        }
        uint col_22 = _S47 + 64U;
        if(col_22 < (p_1->headDim_0))
        {
            float device* _S72 = (&kernelContext_5)->out0_0+(_S60 + col_22);
            if(_S62)
            {
                _S63 = a0_0[8U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S72 = _S63;
        }
        uint col_23 = _S47 + 72U;
        if(col_23 < (p_1->headDim_0))
        {
            float device* _S73 = (&kernelContext_5)->out0_0+(_S60 + col_23);
            if(_S62)
            {
                _S63 = a0_0[9U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S73 = _S63;
        }
        uint col_24 = _S47 + 80U;
        if(col_24 < (p_1->headDim_0))
        {
            float device* _S74 = (&kernelContext_5)->out0_0+(_S60 + col_24);
            if(_S62)
            {
                _S63 = a0_0[10U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S74 = _S63;
        }
        uint col_25 = _S47 + 88U;
        if(col_25 < (p_1->headDim_0))
        {
            float device* _S75 = (&kernelContext_5)->out0_0+(_S60 + col_25);
            if(_S62)
            {
                _S63 = a0_0[11U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S75 = _S63;
        }
        uint col_26 = _S47 + 96U;
        if(col_26 < (p_1->headDim_0))
        {
            float device* _S76 = (&kernelContext_5)->out0_0+(_S60 + col_26);
            if(_S62)
            {
                _S63 = a0_0[12U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S76 = _S63;
        }
        uint col_27 = _S47 + 104U;
        if(col_27 < (p_1->headDim_0))
        {
            float device* _S77 = (&kernelContext_5)->out0_0+(_S60 + col_27);
            if(_S62)
            {
                _S63 = a0_0[13U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S77 = _S63;
        }
        uint col_28 = _S47 + 112U;
        if(col_28 < (p_1->headDim_0))
        {
            float device* _S78 = (&kernelContext_5)->out0_0+(_S60 + col_28);
            if(_S62)
            {
                _S63 = a0_0[14U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S78 = _S63;
        }
        uint col_29 = _S47 + 120U;
        if(col_29 < (p_1->headDim_0))
        {
            float device* _S79 = (&kernelContext_5)->out0_0+(_S60 + col_29);
            if(_S62)
            {
                _S63 = a0_0[15U] * (&kernelContext_5)->p_0->scale_0;
            }
            else
            {
                _S63 = 0.0;
            }
            *_S79 = _S63;
        }
    }
    return;
}

