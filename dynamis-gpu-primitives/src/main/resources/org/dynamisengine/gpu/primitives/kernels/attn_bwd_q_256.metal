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
    uint device* q_0;
    float device* dO_0;
    float device* lse_0;
    float device* dD_0;
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

void loadRows_0(uint t_0, uint base_0, uint r0_0, uint rows_0, uint d2_0, KernelContext_0 thread* kernelContext_0)
{
    float _S1;
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
        bool ok_0;
        if(r_0 < rows_0)
        {
            ok_0 = pr_0 < d2_0;
        }
        else
        {
            ok_0 = false;
        }
        uint _S2;
        if(ok_0)
        {
            _S2 = *(kernelContext_0->q_0+((base_0 + r_0) * d2_0 + pr_0));
        }
        else
        {
            _S2 = 0U;
        }
        (*kernelContext_0->qs_0)[i_0 * 129U + pr_0] = _S2;
        uint _S3 = 2U * pr_0;
        uint _S4 = i_0 * 257U + _S3;
        if(ok_0)
        {
            _S1 = *(kernelContext_0->dO_0+((base_0 + r_0) * kernelContext_0->p_0->headDim_0 + _S3));
        }
        else
        {
            _S1 = 0.0;
        }
        (*kernelContext_0->dos_0)[_S4] = _S1;
        float _S5;
        if(ok_0)
        {
            _S5 = *(kernelContext_0->dO_0+((base_0 + r_0) * kernelContext_0->p_0->headDim_0 + _S3 + 1U));
        }
        else
        {
            _S5 = 0.0;
        }
        (*kernelContext_0->dos_0)[_S4 + 1U] = _S5;
        e_0 = e_0 + 128U;
    }
    if(t_0 < 8U)
    {
        uint r_1 = r0_0 + t_0;
        bool _S6 = r_1 < rows_0;
        if(_S6)
        {
            _S1 = *(kernelContext_0->lse_0+(base_0 + r_1));
        }
        else
        {
            _S1 = 0.0;
        }
        (*kernelContext_0->ls_0)[t_0] = _S1;
        if(_S6)
        {
            _S1 = *(kernelContext_0->dD_0+(base_0 + r_1));
        }
        else
        {
            _S1 = 0.0;
        }
        (*kernelContext_0->ds_0)[t_0] = _S1;
    }
    return;
}

void loadKeys_0(uint t_1, uint kvHead_0, uint k0_0, uint d2_1, KernelContext_0 thread* kernelContext_1)
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
        if(key_0 < (kernelContext_1->p_0->keys_0))
        {
            ok_1 = pr_1 < d2_1;
        }
        else
        {
            ok_1 = false;
        }
        uint at_0 = (kvHead_0 * kernelContext_1->p_0->cacheKeys_0 + key_0) * d2_1 + pr_1;
        uint _S7 = kk_0 * 129U + pr_1;
        uint _S8;
        if(ok_1)
        {
            _S8 = *(kernelContext_1->k_0+at_0);
        }
        else
        {
            _S8 = 0U;
        }
        (*kernelContext_1->ks_0)[_S7] = _S8;
        uint _S9;
        if(ok_1)
        {
            _S9 = *(kernelContext_1->v_0+at_0);
        }
        else
        {
            _S9 = 0U;
        }
        (*kernelContext_1->vs_0)[_S7] = _S9;
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

void pairs_0(uint t_2, uint r0_1, uint rows_1, uint k0_1, uint d2_2, KernelContext_0 thread* kernelContext_2)
{
    uint _S10 = kernelContext_2->p_0->keys_0 - kernelContext_2->p_0->queries_0;
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
            uint vw_0 = (*kernelContext_2->vs_0)[kv_0[0U] * 129U + p0_0];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + p0_0]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + p0_0]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + p0_0]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + p0_0]), sv_0[0U]));
            uint _S11 = 2U * p0_0;
            float _S12 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S11] * attnLo_0(vw_0);
            dp_0[0U] = _S12;
            dp_0[0U] = _S12 + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S11 + 1U] * attnHi_0(vw_0);
        }
        uint pr_2 = p0_0 + 1U;
        if(pr_2 < d2_2)
        {
            uint vw_1 = (*kernelContext_2->vs_0)[kv_0[0U] * 129U + pr_2];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_2]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_2]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_2]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_2]), sv_0[0U]));
            uint _S13 = 2U * pr_2;
            float _S14 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S13] * attnLo_0(vw_1);
            dp_0[0U] = _S14;
            dp_0[0U] = _S14 + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S13 + 1U] * attnHi_0(vw_1);
        }
        uint pr_3 = p0_0 + 2U;
        if(pr_3 < d2_2)
        {
            uint vw_2 = (*kernelContext_2->vs_0)[kv_0[0U] * 129U + pr_3];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_3]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_3]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_3]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_3]), sv_0[0U]));
            uint _S15 = 2U * pr_3;
            float _S16 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S15] * attnLo_0(vw_2);
            dp_0[0U] = _S16;
            dp_0[0U] = _S16 + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S15 + 1U] * attnHi_0(vw_2);
        }
        uint pr_4 = p0_0 + 3U;
        if(pr_4 < d2_2)
        {
            uint vw_3 = (*kernelContext_2->vs_0)[kv_0[0U] * 129U + pr_4];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_4]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_4]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_4]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_4]), sv_0[0U]));
            uint _S17 = 2U * pr_4;
            float _S18 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S17] * attnLo_0(vw_3);
            dp_0[0U] = _S18;
            dp_0[0U] = _S18 + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S17 + 1U] * attnHi_0(vw_3);
        }
        uint pr_5 = p0_0 + 4U;
        if(pr_5 < d2_2)
        {
            uint vw_4 = (*kernelContext_2->vs_0)[kv_0[0U] * 129U + pr_5];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_5]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_5]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_5]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_5]), sv_0[0U]));
            uint _S19 = 2U * pr_5;
            float _S20 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S19] * attnLo_0(vw_4);
            dp_0[0U] = _S20;
            dp_0[0U] = _S20 + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S19 + 1U] * attnHi_0(vw_4);
        }
        uint pr_6 = p0_0 + 5U;
        if(pr_6 < d2_2)
        {
            uint vw_5 = (*kernelContext_2->vs_0)[kv_0[0U] * 129U + pr_6];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_6]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_6]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_6]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_6]), sv_0[0U]));
            uint _S21 = 2U * pr_6;
            float _S22 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S21] * attnLo_0(vw_5);
            dp_0[0U] = _S22;
            dp_0[0U] = _S22 + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S21 + 1U] * attnHi_0(vw_5);
        }
        uint pr_7 = p0_0 + 6U;
        if(pr_7 < d2_2)
        {
            uint vw_6 = (*kernelContext_2->vs_0)[kv_0[0U] * 129U + pr_7];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_7]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_7]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_7]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_7]), sv_0[0U]));
            uint _S23 = 2U * pr_7;
            float _S24 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S23] * attnLo_0(vw_6);
            dp_0[0U] = _S24;
            dp_0[0U] = _S24 + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S23 + 1U] * attnHi_0(vw_6);
        }
        uint pr_8 = p0_0 + 7U;
        if(pr_8 < d2_2)
        {
            uint vw_7 = (*kernelContext_2->vs_0)[kv_0[0U] * 129U + pr_8];
            sv_0[0U] = fma(attnHi_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_8]), attnHi_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_8]), fma(attnLo_0((*kernelContext_2->qs_0)[ii_0[0U] * 129U + pr_8]), attnLo_0((*kernelContext_2->ks_0)[kv_0[0U] * 129U + pr_8]), sv_0[0U]));
            uint _S25 = 2U * pr_8;
            float _S26 = dp_0[0U] + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S25] * attnLo_0(vw_7);
            dp_0[0U] = _S26;
            dp_0[0U] = _S26 + (*kernelContext_2->dos_0)[ii_0[0U] * 257U + _S25 + 1U] * attnHi_0(vw_7);
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
            uint _S27 = r_2 % kernelContext_2->p_0->queries_0;
            live_0 = key_1 <= (_S27 + _S10);
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
    uint _S28 = i_1 * 16U + kk_1;
    (*kernelContext_2->ps_0)[_S28] = pv_0;
    if(live_0)
    {
        pv_0 = pv_0 * (dp_0[0U] - (*kernelContext_2->ds_0)[i_1]);
    }
    else
    {
        pv_0 = 0.0;
    }
    (*kernelContext_2->dss_0)[_S28] = pv_0;
    return;
}

float half2At_0(uint w_2, uint c_0)
{
    float _S29;
    if((c_0 & 1U) == 0U)
    {
        _S29 = attnLo_0(w_2);
    }
    else
    {
        _S29 = attnHi_0(w_2);
    }
    return _S29;
}

[[kernel]] void attn_bwd_q_256(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnBwdParams_0 constant* p_1 [[buffer(7)]], uint device* q_1 [[buffer(0)]], float device* dO_1 [[buffer(3)]], float device* lse_1 [[buffer(4)]], float device* dD_1 [[buffer(5)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* out0_1 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->p_0 = p_1;
    (&kernelContext_3)->q_0 = q_1;
    (&kernelContext_3)->dO_0 = dO_1;
    (&kernelContext_3)->lse_0 = lse_1;
    (&kernelContext_3)->dD_0 = dD_1;
    (&kernelContext_3)->k_0 = k_1;
    (&kernelContext_3)->v_0 = v_1;
    (&kernelContext_3)->out0_0 = out0_1;
    threadgroup array<uint, int(1032)> qs_1;
    (&kernelContext_3)->qs_0 = &qs_1;
    threadgroup array<float, int(2056)> dos_1;
    (&kernelContext_3)->dos_0 = &dos_1;
    threadgroup array<float, int(8)> ls_1;
    (&kernelContext_3)->ls_0 = &ls_1;
    threadgroup array<float, int(8)> ds_1;
    (&kernelContext_3)->ds_0 = &ds_1;
    threadgroup array<uint, int(2064)> ks_1;
    (&kernelContext_3)->ks_0 = &ks_1;
    threadgroup array<uint, int(2064)> vs_1;
    (&kernelContext_3)->vs_0 = &vs_1;
    threadgroup array<float, int(128)> ps_1;
    (&kernelContext_3)->ps_0 = &ps_1;
    threadgroup array<float, int(128)> dss_1;
    (&kernelContext_3)->dss_0 = &dss_1;
    uint t_3 = local_0.x;
    uint kvHead_1 = group_0.y;
    uint d2_3 = p_1->headDim_0 / 2U;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint rows_2 = hpk_0 * (&kernelContext_3)->p_0->queries_0;
    uint base_1 = kvHead_1 * rows_2;
    uint offset_0 = (&kernelContext_3)->p_0->keys_0 - (&kernelContext_3)->p_0->queries_0;
    uint owner_0 = t_3 / 8U;
    uint _S30 = t_3 % 8U;
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
    uint r0_2 = group_0.x * 8U;
    loadRows_0(t_3, base_1, r0_2, rows_2, d2_3, &kernelContext_3);
    uint kEnd_0 = (&kernelContext_3)->p_0->keys_0;
    uint kEnd_1;
    if(((&kernelContext_3)->p_0->causal_0) != 0U)
    {
        uint rLast_0 = min(r0_2 + 8U, rows_2) - 1U;
        uint _S31 = r0_2 / (&kernelContext_3)->p_0->queries_0;
        uint _S32 = rLast_0 / (&kernelContext_3)->p_0->queries_0;
        if(_S31 != _S32)
        {
            kEnd_1 = (&kernelContext_3)->p_0->queries_0 - 1U;
        }
        else
        {
            uint _S33 = rLast_0 % (&kernelContext_3)->p_0->queries_0;
            kEnd_1 = _S33;
        }
        kEnd_1 = min(kEnd_0, kEnd_1 + offset_0 + 1U);
    }
    else
    {
        kEnd_1 = kEnd_0;
    }
    uint k0_2 = 0U;
    for(;;)
    {
        if(k0_2 < kEnd_1)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loadKeys_0(t_3, kvHead_1, k0_2, d2_3, &kernelContext_3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pairs_0(t_3, r0_2, rows_2, k0_2, d2_3, &kernelContext_3);
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
                float _S34 = (*(&kernelContext_3)->dss_0)[owner_0 * 16U + kk_2];
                uint _S35 = kk_2 * 129U;
                a0_0[0U] = a0_0[0U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + _S30 / 2U], _S30);
                uint col_0 = _S30 + 8U;
                a0_0[1U] = a0_0[1U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_0 / 2U], col_0);
                uint col_1 = _S30 + 16U;
                a0_0[2U] = a0_0[2U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_1 / 2U], col_1);
                uint col_2 = _S30 + 24U;
                a0_0[3U] = a0_0[3U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_2 / 2U], col_2);
                uint col_3 = _S30 + 32U;
                a0_0[4U] = a0_0[4U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_3 / 2U], col_3);
                uint col_4 = _S30 + 40U;
                a0_0[5U] = a0_0[5U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_4 / 2U], col_4);
                uint col_5 = _S30 + 48U;
                a0_0[6U] = a0_0[6U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_5 / 2U], col_5);
                uint col_6 = _S30 + 56U;
                a0_0[7U] = a0_0[7U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_6 / 2U], col_6);
                uint col_7 = _S30 + 64U;
                a0_0[8U] = a0_0[8U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_7 / 2U], col_7);
                uint col_8 = _S30 + 72U;
                a0_0[9U] = a0_0[9U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_8 / 2U], col_8);
                uint col_9 = _S30 + 80U;
                a0_0[10U] = a0_0[10U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_9 / 2U], col_9);
                uint col_10 = _S30 + 88U;
                a0_0[11U] = a0_0[11U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_10 / 2U], col_10);
                uint col_11 = _S30 + 96U;
                a0_0[12U] = a0_0[12U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_11 / 2U], col_11);
                uint col_12 = _S30 + 104U;
                a0_0[13U] = a0_0[13U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_12 / 2U], col_12);
                uint col_13 = _S30 + 112U;
                a0_0[14U] = a0_0[14U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_13 / 2U], col_13);
                uint col_14 = _S30 + 120U;
                a0_0[15U] = a0_0[15U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_14 / 2U], col_14);
                uint col_15 = _S30 + 128U;
                a0_0[16U] = a0_0[16U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_15 / 2U], col_15);
                uint col_16 = _S30 + 136U;
                a0_0[17U] = a0_0[17U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_16 / 2U], col_16);
                uint col_17 = _S30 + 144U;
                a0_0[18U] = a0_0[18U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_17 / 2U], col_17);
                uint col_18 = _S30 + 152U;
                a0_0[19U] = a0_0[19U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_18 / 2U], col_18);
                uint col_19 = _S30 + 160U;
                a0_0[20U] = a0_0[20U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_19 / 2U], col_19);
                uint col_20 = _S30 + 168U;
                a0_0[21U] = a0_0[21U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_20 / 2U], col_20);
                uint col_21 = _S30 + 176U;
                a0_0[22U] = a0_0[22U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_21 / 2U], col_21);
                uint col_22 = _S30 + 184U;
                a0_0[23U] = a0_0[23U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_22 / 2U], col_22);
                uint col_23 = _S30 + 192U;
                a0_0[24U] = a0_0[24U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_23 / 2U], col_23);
                uint col_24 = _S30 + 200U;
                a0_0[25U] = a0_0[25U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_24 / 2U], col_24);
                uint col_25 = _S30 + 208U;
                a0_0[26U] = a0_0[26U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_25 / 2U], col_25);
                uint col_26 = _S30 + 216U;
                a0_0[27U] = a0_0[27U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_26 / 2U], col_26);
                uint col_27 = _S30 + 224U;
                a0_0[28U] = a0_0[28U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_27 / 2U], col_27);
                uint col_28 = _S30 + 232U;
                a0_0[29U] = a0_0[29U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_28 / 2U], col_28);
                uint col_29 = _S30 + 240U;
                a0_0[30U] = a0_0[30U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_29 / 2U], col_29);
                uint col_30 = _S30 + 248U;
                a0_0[31U] = a0_0[31U] + _S34 * half2At_0((*(&kernelContext_3)->ks_0)[_S35 + col_30 / 2U], col_30);
                kk_2 = kk_2 + 1U;
            }
        }
        k0_2 = k0_2 + 16U;
    }
    uint r_3 = r0_2 + owner_0;
    bool _S36;
    if(owner_0 < 8U)
    {
        _S36 = r_3 < rows_2;
    }
    else
    {
        _S36 = false;
    }
    if(_S36)
    {
        uint _S37 = (base_1 + r_3) * p_1->headDim_0;
        if(_S30 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + _S30)) = a0_0[0U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_31 = _S30 + 8U;
        if(col_31 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_31)) = a0_0[1U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_32 = _S30 + 16U;
        if(col_32 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_32)) = a0_0[2U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_33 = _S30 + 24U;
        if(col_33 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_33)) = a0_0[3U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_34 = _S30 + 32U;
        if(col_34 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_34)) = a0_0[4U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_35 = _S30 + 40U;
        if(col_35 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_35)) = a0_0[5U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_36 = _S30 + 48U;
        if(col_36 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_36)) = a0_0[6U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_37 = _S30 + 56U;
        if(col_37 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_37)) = a0_0[7U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_38 = _S30 + 64U;
        if(col_38 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_38)) = a0_0[8U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_39 = _S30 + 72U;
        if(col_39 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_39)) = a0_0[9U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_40 = _S30 + 80U;
        if(col_40 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_40)) = a0_0[10U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_41 = _S30 + 88U;
        if(col_41 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_41)) = a0_0[11U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_42 = _S30 + 96U;
        if(col_42 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_42)) = a0_0[12U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_43 = _S30 + 104U;
        if(col_43 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_43)) = a0_0[13U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_44 = _S30 + 112U;
        if(col_44 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_44)) = a0_0[14U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_45 = _S30 + 120U;
        if(col_45 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_45)) = a0_0[15U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_46 = _S30 + 128U;
        if(col_46 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_46)) = a0_0[16U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_47 = _S30 + 136U;
        if(col_47 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_47)) = a0_0[17U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_48 = _S30 + 144U;
        if(col_48 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_48)) = a0_0[18U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_49 = _S30 + 152U;
        if(col_49 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_49)) = a0_0[19U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_50 = _S30 + 160U;
        if(col_50 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_50)) = a0_0[20U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_51 = _S30 + 168U;
        if(col_51 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_51)) = a0_0[21U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_52 = _S30 + 176U;
        if(col_52 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_52)) = a0_0[22U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_53 = _S30 + 184U;
        if(col_53 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_53)) = a0_0[23U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_54 = _S30 + 192U;
        if(col_54 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_54)) = a0_0[24U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_55 = _S30 + 200U;
        if(col_55 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_55)) = a0_0[25U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_56 = _S30 + 208U;
        if(col_56 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_56)) = a0_0[26U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_57 = _S30 + 216U;
        if(col_57 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_57)) = a0_0[27U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_58 = _S30 + 224U;
        if(col_58 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_58)) = a0_0[28U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_59 = _S30 + 232U;
        if(col_59 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_59)) = a0_0[29U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_60 = _S30 + 240U;
        if(col_60 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_60)) = a0_0[30U] * (&kernelContext_3)->p_0->scale_0;
        }
        uint col_61 = _S30 + 248U;
        if(col_61 < (p_1->headDim_0))
        {
            *((&kernelContext_3)->out0_0+(_S37 + col_61)) = a0_0[31U] * (&kernelContext_3)->p_0->scale_0;
        }
    }
    return;
}

