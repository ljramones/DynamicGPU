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
    uint i_1 = t_2 / 16U;
    uint kk_1 = t_2 % 16U;
    uint r_2 = r0_1 + i_1;
    uint key_1 = k0_1 + kk_1;
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
            uint _S11 = r_2 % kernelContext_2->p_0->queries_0;
            live_0 = key_1 <= (_S11 + _S10);
        }
    }
    else
    {
        live_0 = false;
    }
    uint p0_0;
    float pv_0;
    float dsv_0;
    float s_0;
    float dp_0;
    float s_1;
    float dp_1;
    float s_2;
    float dp_2;
    float s_3;
    float dp_3;
    float s_4;
    float dp_4;
    float s_5;
    float dp_5;
    float s_6;
    float dp_6;
    if(live_0)
    {
        p0_0 = 0U;
        pv_0 = 0.0;
        dsv_0 = 0.0;
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
                uint _S12 = kk_1 * 33U + p0_0;
                uint _S13 = i_1 * 65U + 2U * p0_0;
                float _S14 = dsv_0 + (*kernelContext_2->dos_0)[_S13] * attnLo_0((*kernelContext_2->vs_0)[_S12]) + (*kernelContext_2->dos_0)[_S13 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S12]);
                s_0 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 33U + p0_0]), attnHi_0((*kernelContext_2->ks_0)[_S12]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 33U + p0_0]), attnLo_0((*kernelContext_2->ks_0)[_S12]), pv_0));
                dp_0 = _S14;
            }
            else
            {
                s_0 = pv_0;
                dp_0 = dsv_0;
            }
            uint pr_2 = p0_0 + 1U;
            if(pr_2 < d2_2)
            {
                uint _S15 = kk_1 * 33U + pr_2;
                uint _S16 = i_1 * 65U + 2U * pr_2;
                float _S17 = dp_0 + (*kernelContext_2->dos_0)[_S16] * attnLo_0((*kernelContext_2->vs_0)[_S15]) + (*kernelContext_2->dos_0)[_S16 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S15]);
                s_1 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_2]), attnHi_0((*kernelContext_2->ks_0)[_S15]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_2]), attnLo_0((*kernelContext_2->ks_0)[_S15]), s_0));
                dp_1 = _S17;
            }
            else
            {
                s_1 = s_0;
                dp_1 = dp_0;
            }
            uint pr_3 = p0_0 + 2U;
            if(pr_3 < d2_2)
            {
                uint _S18 = kk_1 * 33U + pr_3;
                uint _S19 = i_1 * 65U + 2U * pr_3;
                float _S20 = dp_1 + (*kernelContext_2->dos_0)[_S19] * attnLo_0((*kernelContext_2->vs_0)[_S18]) + (*kernelContext_2->dos_0)[_S19 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S18]);
                s_2 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_3]), attnHi_0((*kernelContext_2->ks_0)[_S18]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_3]), attnLo_0((*kernelContext_2->ks_0)[_S18]), s_1));
                dp_2 = _S20;
            }
            else
            {
                s_2 = s_1;
                dp_2 = dp_1;
            }
            uint pr_4 = p0_0 + 3U;
            if(pr_4 < d2_2)
            {
                uint _S21 = kk_1 * 33U + pr_4;
                uint _S22 = i_1 * 65U + 2U * pr_4;
                float _S23 = dp_2 + (*kernelContext_2->dos_0)[_S22] * attnLo_0((*kernelContext_2->vs_0)[_S21]) + (*kernelContext_2->dos_0)[_S22 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S21]);
                s_3 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_4]), attnHi_0((*kernelContext_2->ks_0)[_S21]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_4]), attnLo_0((*kernelContext_2->ks_0)[_S21]), s_2));
                dp_3 = _S23;
            }
            else
            {
                s_3 = s_2;
                dp_3 = dp_2;
            }
            uint pr_5 = p0_0 + 4U;
            if(pr_5 < d2_2)
            {
                uint _S24 = kk_1 * 33U + pr_5;
                uint _S25 = i_1 * 65U + 2U * pr_5;
                float _S26 = dp_3 + (*kernelContext_2->dos_0)[_S25] * attnLo_0((*kernelContext_2->vs_0)[_S24]) + (*kernelContext_2->dos_0)[_S25 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S24]);
                s_4 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_5]), attnHi_0((*kernelContext_2->ks_0)[_S24]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_5]), attnLo_0((*kernelContext_2->ks_0)[_S24]), s_3));
                dp_4 = _S26;
            }
            else
            {
                s_4 = s_3;
                dp_4 = dp_3;
            }
            uint pr_6 = p0_0 + 5U;
            if(pr_6 < d2_2)
            {
                uint _S27 = kk_1 * 33U + pr_6;
                uint _S28 = i_1 * 65U + 2U * pr_6;
                float _S29 = dp_4 + (*kernelContext_2->dos_0)[_S28] * attnLo_0((*kernelContext_2->vs_0)[_S27]) + (*kernelContext_2->dos_0)[_S28 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S27]);
                s_5 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_6]), attnHi_0((*kernelContext_2->ks_0)[_S27]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_6]), attnLo_0((*kernelContext_2->ks_0)[_S27]), s_4));
                dp_5 = _S29;
            }
            else
            {
                s_5 = s_4;
                dp_5 = dp_4;
            }
            uint pr_7 = p0_0 + 6U;
            if(pr_7 < d2_2)
            {
                uint _S30 = kk_1 * 33U + pr_7;
                uint _S31 = i_1 * 65U + 2U * pr_7;
                float _S32 = dp_5 + (*kernelContext_2->dos_0)[_S31] * attnLo_0((*kernelContext_2->vs_0)[_S30]) + (*kernelContext_2->dos_0)[_S31 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S30]);
                s_6 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_7]), attnHi_0((*kernelContext_2->ks_0)[_S30]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_7]), attnLo_0((*kernelContext_2->ks_0)[_S30]), s_5));
                dp_6 = _S32;
            }
            else
            {
                s_6 = s_5;
                dp_6 = dp_5;
            }
            uint pr_8 = p0_0 + 7U;
            if(pr_8 < d2_2)
            {
                uint _S33 = kk_1 * 33U + pr_8;
                uint _S34 = i_1 * 65U + 2U * pr_8;
                float _S35 = dp_6 + (*kernelContext_2->dos_0)[_S34] * attnLo_0((*kernelContext_2->vs_0)[_S33]) + (*kernelContext_2->dos_0)[_S34 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S33]);
                pv_0 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_8]), attnHi_0((*kernelContext_2->ks_0)[_S33]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 33U + pr_8]), attnLo_0((*kernelContext_2->ks_0)[_S33]), s_6));
                dsv_0 = _S35;
            }
            else
            {
                pv_0 = s_6;
                dsv_0 = dp_6;
            }
            p0_0 = p0_0 + 8U;
        }
        float pv_1 = exp(pv_0 * kernelContext_2->p_0->scale_0 - (*kernelContext_2->ls_0)[i_1]);
        float _S36 = pv_1 * (dsv_0 - (*kernelContext_2->ds_0)[i_1]);
        pv_0 = pv_1;
        dsv_0 = _S36;
    }
    else
    {
        pv_0 = 0.0;
        dsv_0 = 0.0;
    }
    uint _S37 = i_1 * 16U + kk_1;
    (*kernelContext_2->ps_0)[_S37] = pv_0;
    (*kernelContext_2->dss_0)[_S37] = dsv_0;
    uint e_2 = t_2 + 128U;
    uint i_2 = e_2 / 16U;
    uint kk_2 = e_2 % 16U;
    uint r_3 = r0_1 + i_2;
    uint key_2 = k0_1 + kk_2;
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
            uint _S38 = r_3 % kernelContext_2->p_0->queries_0;
            live_0 = key_2 <= (_S38 + _S10);
        }
    }
    else
    {
        live_0 = false;
    }
    if(live_0)
    {
        p0_0 = 0U;
        s_0 = 0.0;
        dp_0 = 0.0;
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
                uint _S39 = kk_2 * 33U + p0_0;
                uint _S40 = i_2 * 65U + 2U * p0_0;
                float _S41 = dp_0 + (*kernelContext_2->dos_0)[_S40] * attnLo_0((*kernelContext_2->vs_0)[_S39]) + (*kernelContext_2->dos_0)[_S40 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S39]);
                s_1 = fma(attnHi_0((*kernelContext_2->qs_0)[i_2 * 33U + p0_0]), attnHi_0((*kernelContext_2->ks_0)[_S39]), fma(attnLo_0((*kernelContext_2->qs_0)[i_2 * 33U + p0_0]), attnLo_0((*kernelContext_2->ks_0)[_S39]), s_0));
                dp_1 = _S41;
            }
            else
            {
                s_1 = s_0;
                dp_1 = dp_0;
            }
            uint pr_9 = p0_0 + 1U;
            if(pr_9 < d2_2)
            {
                uint _S42 = kk_2 * 33U + pr_9;
                uint _S43 = i_2 * 65U + 2U * pr_9;
                float _S44 = dp_1 + (*kernelContext_2->dos_0)[_S43] * attnLo_0((*kernelContext_2->vs_0)[_S42]) + (*kernelContext_2->dos_0)[_S43 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S42]);
                s_2 = fma(attnHi_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_9]), attnHi_0((*kernelContext_2->ks_0)[_S42]), fma(attnLo_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_9]), attnLo_0((*kernelContext_2->ks_0)[_S42]), s_1));
                dp_2 = _S44;
            }
            else
            {
                s_2 = s_1;
                dp_2 = dp_1;
            }
            uint pr_10 = p0_0 + 2U;
            if(pr_10 < d2_2)
            {
                uint _S45 = kk_2 * 33U + pr_10;
                uint _S46 = i_2 * 65U + 2U * pr_10;
                float _S47 = dp_2 + (*kernelContext_2->dos_0)[_S46] * attnLo_0((*kernelContext_2->vs_0)[_S45]) + (*kernelContext_2->dos_0)[_S46 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S45]);
                s_3 = fma(attnHi_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_10]), attnHi_0((*kernelContext_2->ks_0)[_S45]), fma(attnLo_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_10]), attnLo_0((*kernelContext_2->ks_0)[_S45]), s_2));
                dp_3 = _S47;
            }
            else
            {
                s_3 = s_2;
                dp_3 = dp_2;
            }
            uint pr_11 = p0_0 + 3U;
            if(pr_11 < d2_2)
            {
                uint _S48 = kk_2 * 33U + pr_11;
                uint _S49 = i_2 * 65U + 2U * pr_11;
                float _S50 = dp_3 + (*kernelContext_2->dos_0)[_S49] * attnLo_0((*kernelContext_2->vs_0)[_S48]) + (*kernelContext_2->dos_0)[_S49 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S48]);
                s_4 = fma(attnHi_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_11]), attnHi_0((*kernelContext_2->ks_0)[_S48]), fma(attnLo_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_11]), attnLo_0((*kernelContext_2->ks_0)[_S48]), s_3));
                dp_4 = _S50;
            }
            else
            {
                s_4 = s_3;
                dp_4 = dp_3;
            }
            uint pr_12 = p0_0 + 4U;
            if(pr_12 < d2_2)
            {
                uint _S51 = kk_2 * 33U + pr_12;
                uint _S52 = i_2 * 65U + 2U * pr_12;
                float _S53 = dp_4 + (*kernelContext_2->dos_0)[_S52] * attnLo_0((*kernelContext_2->vs_0)[_S51]) + (*kernelContext_2->dos_0)[_S52 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S51]);
                s_5 = fma(attnHi_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_12]), attnHi_0((*kernelContext_2->ks_0)[_S51]), fma(attnLo_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_12]), attnLo_0((*kernelContext_2->ks_0)[_S51]), s_4));
                dp_5 = _S53;
            }
            else
            {
                s_5 = s_4;
                dp_5 = dp_4;
            }
            uint pr_13 = p0_0 + 5U;
            if(pr_13 < d2_2)
            {
                uint _S54 = kk_2 * 33U + pr_13;
                uint _S55 = i_2 * 65U + 2U * pr_13;
                float _S56 = dp_5 + (*kernelContext_2->dos_0)[_S55] * attnLo_0((*kernelContext_2->vs_0)[_S54]) + (*kernelContext_2->dos_0)[_S55 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S54]);
                s_6 = fma(attnHi_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_13]), attnHi_0((*kernelContext_2->ks_0)[_S54]), fma(attnLo_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_13]), attnLo_0((*kernelContext_2->ks_0)[_S54]), s_5));
                dp_6 = _S56;
            }
            else
            {
                s_6 = s_5;
                dp_6 = dp_5;
            }
            uint pr_14 = p0_0 + 6U;
            float s_7;
            float dp_7;
            if(pr_14 < d2_2)
            {
                uint _S57 = kk_2 * 33U + pr_14;
                uint _S58 = i_2 * 65U + 2U * pr_14;
                float _S59 = dp_6 + (*kernelContext_2->dos_0)[_S58] * attnLo_0((*kernelContext_2->vs_0)[_S57]) + (*kernelContext_2->dos_0)[_S58 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S57]);
                s_7 = fma(attnHi_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_14]), attnHi_0((*kernelContext_2->ks_0)[_S57]), fma(attnLo_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_14]), attnLo_0((*kernelContext_2->ks_0)[_S57]), s_6));
                dp_7 = _S59;
            }
            else
            {
                s_7 = s_6;
                dp_7 = dp_6;
            }
            uint pr_15 = p0_0 + 7U;
            if(pr_15 < d2_2)
            {
                uint _S60 = kk_2 * 33U + pr_15;
                uint _S61 = i_2 * 65U + 2U * pr_15;
                float _S62 = dp_7 + (*kernelContext_2->dos_0)[_S61] * attnLo_0((*kernelContext_2->vs_0)[_S60]) + (*kernelContext_2->dos_0)[_S61 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S60]);
                s_0 = fma(attnHi_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_15]), attnHi_0((*kernelContext_2->ks_0)[_S60]), fma(attnLo_0((*kernelContext_2->qs_0)[i_2 * 33U + pr_15]), attnLo_0((*kernelContext_2->ks_0)[_S60]), s_7));
                dp_0 = _S62;
            }
            else
            {
                s_0 = s_7;
                dp_0 = dp_7;
            }
            p0_0 = p0_0 + 8U;
        }
        float pv_2 = exp(s_0 * kernelContext_2->p_0->scale_0 - (*kernelContext_2->ls_0)[i_2]);
        float _S63 = pv_2 * (dp_0 - (*kernelContext_2->ds_0)[i_2]);
        pv_0 = pv_2;
        dsv_0 = _S63;
    }
    else
    {
        pv_0 = 0.0;
        dsv_0 = 0.0;
    }
    uint _S64 = i_2 * 16U + kk_2;
    (*kernelContext_2->ps_0)[_S64] = pv_0;
    (*kernelContext_2->dss_0)[_S64] = dsv_0;
    return;
}

float half2At_0(uint w_2, uint c_0)
{
    float _S65;
    if((c_0 & 1U) == 0U)
    {
        _S65 = attnLo_0(w_2);
    }
    else
    {
        _S65 = attnHi_0(w_2);
    }
    return _S65;
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
    uint _S66 = kvHead_1 * rows_2;
    uint _S67 = (&kernelContext_3)->p_0->keys_0 - (&kernelContext_3)->p_0->queries_0;
    uint owner_0 = t_3 / 8U;
    uint _S68 = t_3 % 8U;
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
            uint _S69 = r0_2 / (&kernelContext_3)->p_0->queries_0;
            uint _S70 = rLast_0 / (&kernelContext_3)->p_0->queries_0;
            if(_S69 != _S70)
            {
                i_3 = (&kernelContext_3)->p_0->queries_0 - 1U;
            }
            else
            {
                uint _S71 = rLast_0 % (&kernelContext_3)->p_0->queries_0;
                i_3 = _S71;
            }
            if((i_3 + _S67) < k0_2)
            {
                r0_2 = r0_2 + 16U;
                continue;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loadRows_0(t_3, _S66, r0_2, rows_2, d2_3, &kernelContext_3);
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
            uint _S72 = i_3 * 16U + owner_0;
            float _S73 = (*(&kernelContext_3)->ps_0)[_S72];
            float _S74 = (*(&kernelContext_3)->dss_0)[_S72];
            uint _S75 = i_3 * 65U;
            a1_0[0U] = a1_0[0U] + _S73 * (*(&kernelContext_3)->dos_0)[_S75 + _S68];
            uint _S76 = i_3 * 33U;
            a0_0[0U] = a0_0[0U] + _S74 * half2At_0((*(&kernelContext_3)->qs_0)[_S76 + _S68 / 2U], _S68);
            uint col_0 = _S68 + 8U;
            a1_0[1U] = a1_0[1U] + _S73 * (*(&kernelContext_3)->dos_0)[_S75 + col_0];
            a0_0[1U] = a0_0[1U] + _S74 * half2At_0((*(&kernelContext_3)->qs_0)[_S76 + col_0 / 2U], col_0);
            uint col_1 = _S68 + 16U;
            a1_0[2U] = a1_0[2U] + _S73 * (*(&kernelContext_3)->dos_0)[_S75 + col_1];
            a0_0[2U] = a0_0[2U] + _S74 * half2At_0((*(&kernelContext_3)->qs_0)[_S76 + col_1 / 2U], col_1);
            uint col_2 = _S68 + 24U;
            a1_0[3U] = a1_0[3U] + _S73 * (*(&kernelContext_3)->dos_0)[_S75 + col_2];
            a0_0[3U] = a0_0[3U] + _S74 * half2At_0((*(&kernelContext_3)->qs_0)[_S76 + col_2 / 2U], col_2);
            uint col_3 = _S68 + 32U;
            a1_0[4U] = a1_0[4U] + _S73 * (*(&kernelContext_3)->dos_0)[_S75 + col_3];
            a0_0[4U] = a0_0[4U] + _S74 * half2At_0((*(&kernelContext_3)->qs_0)[_S76 + col_3 / 2U], col_3);
            uint col_4 = _S68 + 40U;
            a1_0[5U] = a1_0[5U] + _S73 * (*(&kernelContext_3)->dos_0)[_S75 + col_4];
            a0_0[5U] = a0_0[5U] + _S74 * half2At_0((*(&kernelContext_3)->qs_0)[_S76 + col_4 / 2U], col_4);
            uint col_5 = _S68 + 48U;
            a1_0[6U] = a1_0[6U] + _S73 * (*(&kernelContext_3)->dos_0)[_S75 + col_5];
            a0_0[6U] = a0_0[6U] + _S74 * half2At_0((*(&kernelContext_3)->qs_0)[_S76 + col_5 / 2U], col_5);
            uint col_6 = _S68 + 56U;
            a1_0[7U] = a1_0[7U] + _S73 * (*(&kernelContext_3)->dos_0)[_S75 + col_6];
            a0_0[7U] = a0_0[7U] + _S74 * half2At_0((*(&kernelContext_3)->qs_0)[_S76 + col_6 / 2U], col_6);
            i_3 = i_3 + 1U;
        }
        r0_2 = r0_2 + 16U;
    }
    uint key_3 = k0_2 + owner_0;
    if(key_3 < ((&kernelContext_3)->p_0->keys_0))
    {
        uint _S77 = (kvHead_1 * (&kernelContext_3)->p_0->cacheKeys_0 + key_3) * p_1->headDim_0;
        if(_S68 < (p_1->headDim_0))
        {
            uint _S78 = _S77 + _S68;
            *((&kernelContext_3)->out0_0+_S78) = a0_0[0U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S78) = a1_0[0U];
        }
        uint col_7 = _S68 + 8U;
        if(col_7 < (p_1->headDim_0))
        {
            uint _S79 = _S77 + col_7;
            *((&kernelContext_3)->out0_0+_S79) = a0_0[1U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S79) = a1_0[1U];
        }
        uint col_8 = _S68 + 16U;
        if(col_8 < (p_1->headDim_0))
        {
            uint _S80 = _S77 + col_8;
            *((&kernelContext_3)->out0_0+_S80) = a0_0[2U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S80) = a1_0[2U];
        }
        uint col_9 = _S68 + 24U;
        if(col_9 < (p_1->headDim_0))
        {
            uint _S81 = _S77 + col_9;
            *((&kernelContext_3)->out0_0+_S81) = a0_0[3U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S81) = a1_0[3U];
        }
        uint col_10 = _S68 + 32U;
        if(col_10 < (p_1->headDim_0))
        {
            uint _S82 = _S77 + col_10;
            *((&kernelContext_3)->out0_0+_S82) = a0_0[4U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S82) = a1_0[4U];
        }
        uint col_11 = _S68 + 40U;
        if(col_11 < (p_1->headDim_0))
        {
            uint _S83 = _S77 + col_11;
            *((&kernelContext_3)->out0_0+_S83) = a0_0[5U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S83) = a1_0[5U];
        }
        uint col_12 = _S68 + 48U;
        if(col_12 < (p_1->headDim_0))
        {
            uint _S84 = _S77 + col_12;
            *((&kernelContext_3)->out0_0+_S84) = a0_0[6U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S84) = a1_0[6U];
        }
        uint col_13 = _S68 + 56U;
        if(col_13 < (p_1->headDim_0))
        {
            uint _S85 = _S77 + col_13;
            *((&kernelContext_3)->out0_0+_S85) = a0_0[7U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S85) = a1_0[7U];
        }
    }
    return;
}

