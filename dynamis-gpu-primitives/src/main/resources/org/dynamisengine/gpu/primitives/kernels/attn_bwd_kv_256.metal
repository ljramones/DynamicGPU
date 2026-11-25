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
    array<uint, int(2064)> threadgroup* ks_0;
    array<uint, int(2064)> threadgroup* vs_0;
    array<uint, int(1032)> threadgroup* qs_0;
    array<float, int(2056)> threadgroup* dos_0;
    array<float, int(8)> threadgroup* ls_0;
    array<float, int(8)> threadgroup* ds_0;
    array<float, int(128)> threadgroup* ps_0;
    array<float, int(128)> threadgroup* dss_0;
};

void loadKeys_0(uint t_0, uint kvHead_0, uint k0_0, uint d2_0, KernelContext_0 thread* kernelContext_0)
{
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
        uint kk_0 = e_0 / 128U;
        uint pr_0 = e_0 % 128U;
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
        uint _S1 = kk_0 * 129U + pr_0;
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
        if(e_1 < 1024U)
        {
        }
        else
        {
            break;
        }
        uint i_0 = e_1 / 128U;
        uint pr_1 = e_1 % 128U;
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
        (*kernelContext_1->qs_0)[i_0 * 129U + pr_1] = _S5;
        uint _S6 = 2U * pr_1;
        uint _S7 = i_0 * 257U + _S6;
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
    if(t_1 < 8U)
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
    float pv_0;
    float dsv_0;
    if(live_0)
    {
        uint p0_0 = 0U;
        pv_0 = 0.0;
        dsv_0 = 0.0;
        for(;;)
        {
            if(p0_0 < 128U)
            {
            }
            else
            {
                break;
            }
            float s_0;
            float dp_0;
            if(p0_0 < d2_2)
            {
                uint _S12 = kk_1 * 129U + p0_0;
                uint _S13 = i_1 * 257U + 2U * p0_0;
                float _S14 = dsv_0 + (*kernelContext_2->dos_0)[_S13] * attnLo_0((*kernelContext_2->vs_0)[_S12]) + (*kernelContext_2->dos_0)[_S13 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S12]);
                s_0 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 129U + p0_0]), attnHi_0((*kernelContext_2->ks_0)[_S12]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 129U + p0_0]), attnLo_0((*kernelContext_2->ks_0)[_S12]), pv_0));
                dp_0 = _S14;
            }
            else
            {
                s_0 = pv_0;
                dp_0 = dsv_0;
            }
            uint pr_2 = p0_0 + 1U;
            float s_1;
            float dp_1;
            if(pr_2 < d2_2)
            {
                uint _S15 = kk_1 * 129U + pr_2;
                uint _S16 = i_1 * 257U + 2U * pr_2;
                float _S17 = dp_0 + (*kernelContext_2->dos_0)[_S16] * attnLo_0((*kernelContext_2->vs_0)[_S15]) + (*kernelContext_2->dos_0)[_S16 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S15]);
                s_1 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_2]), attnHi_0((*kernelContext_2->ks_0)[_S15]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_2]), attnLo_0((*kernelContext_2->ks_0)[_S15]), s_0));
                dp_1 = _S17;
            }
            else
            {
                s_1 = s_0;
                dp_1 = dp_0;
            }
            uint pr_3 = p0_0 + 2U;
            float s_2;
            float dp_2;
            if(pr_3 < d2_2)
            {
                uint _S18 = kk_1 * 129U + pr_3;
                uint _S19 = i_1 * 257U + 2U * pr_3;
                float _S20 = dp_1 + (*kernelContext_2->dos_0)[_S19] * attnLo_0((*kernelContext_2->vs_0)[_S18]) + (*kernelContext_2->dos_0)[_S19 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S18]);
                s_2 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_3]), attnHi_0((*kernelContext_2->ks_0)[_S18]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_3]), attnLo_0((*kernelContext_2->ks_0)[_S18]), s_1));
                dp_2 = _S20;
            }
            else
            {
                s_2 = s_1;
                dp_2 = dp_1;
            }
            uint pr_4 = p0_0 + 3U;
            float s_3;
            float dp_3;
            if(pr_4 < d2_2)
            {
                uint _S21 = kk_1 * 129U + pr_4;
                uint _S22 = i_1 * 257U + 2U * pr_4;
                float _S23 = dp_2 + (*kernelContext_2->dos_0)[_S22] * attnLo_0((*kernelContext_2->vs_0)[_S21]) + (*kernelContext_2->dos_0)[_S22 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S21]);
                s_3 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_4]), attnHi_0((*kernelContext_2->ks_0)[_S21]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_4]), attnLo_0((*kernelContext_2->ks_0)[_S21]), s_2));
                dp_3 = _S23;
            }
            else
            {
                s_3 = s_2;
                dp_3 = dp_2;
            }
            uint pr_5 = p0_0 + 4U;
            float s_4;
            float dp_4;
            if(pr_5 < d2_2)
            {
                uint _S24 = kk_1 * 129U + pr_5;
                uint _S25 = i_1 * 257U + 2U * pr_5;
                float _S26 = dp_3 + (*kernelContext_2->dos_0)[_S25] * attnLo_0((*kernelContext_2->vs_0)[_S24]) + (*kernelContext_2->dos_0)[_S25 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S24]);
                s_4 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_5]), attnHi_0((*kernelContext_2->ks_0)[_S24]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_5]), attnLo_0((*kernelContext_2->ks_0)[_S24]), s_3));
                dp_4 = _S26;
            }
            else
            {
                s_4 = s_3;
                dp_4 = dp_3;
            }
            uint pr_6 = p0_0 + 5U;
            float s_5;
            float dp_5;
            if(pr_6 < d2_2)
            {
                uint _S27 = kk_1 * 129U + pr_6;
                uint _S28 = i_1 * 257U + 2U * pr_6;
                float _S29 = dp_4 + (*kernelContext_2->dos_0)[_S28] * attnLo_0((*kernelContext_2->vs_0)[_S27]) + (*kernelContext_2->dos_0)[_S28 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S27]);
                s_5 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_6]), attnHi_0((*kernelContext_2->ks_0)[_S27]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_6]), attnLo_0((*kernelContext_2->ks_0)[_S27]), s_4));
                dp_5 = _S29;
            }
            else
            {
                s_5 = s_4;
                dp_5 = dp_4;
            }
            uint pr_7 = p0_0 + 6U;
            float s_6;
            float dp_6;
            if(pr_7 < d2_2)
            {
                uint _S30 = kk_1 * 129U + pr_7;
                uint _S31 = i_1 * 257U + 2U * pr_7;
                float _S32 = dp_5 + (*kernelContext_2->dos_0)[_S31] * attnLo_0((*kernelContext_2->vs_0)[_S30]) + (*kernelContext_2->dos_0)[_S31 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S30]);
                s_6 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_7]), attnHi_0((*kernelContext_2->ks_0)[_S30]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_7]), attnLo_0((*kernelContext_2->ks_0)[_S30]), s_5));
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
                uint _S33 = kk_1 * 129U + pr_8;
                uint _S34 = i_1 * 257U + 2U * pr_8;
                float _S35 = dp_6 + (*kernelContext_2->dos_0)[_S34] * attnLo_0((*kernelContext_2->vs_0)[_S33]) + (*kernelContext_2->dos_0)[_S34 + 1U] * attnHi_0((*kernelContext_2->vs_0)[_S33]);
                pv_0 = fma(attnHi_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_8]), attnHi_0((*kernelContext_2->ks_0)[_S33]), fma(attnLo_0((*kernelContext_2->qs_0)[i_1 * 129U + pr_8]), attnLo_0((*kernelContext_2->ks_0)[_S33]), s_6));
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
    return;
}

float half2At_0(uint w_2, uint c_0)
{
    float _S38;
    if((c_0 & 1U) == 0U)
    {
        _S38 = attnLo_0(w_2);
    }
    else
    {
        _S38 = attnHi_0(w_2);
    }
    return _S38;
}

[[kernel]] void attn_bwd_kv_256(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnBwdParams_0 constant* p_1 [[buffer(8)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], uint device* q_1 [[buffer(0)]], float device* dO_1 [[buffer(3)]], float device* lse_1 [[buffer(4)]], float device* dD_1 [[buffer(5)]], float device* out0_1 [[buffer(6)]], float device* out1_1 [[buffer(7)]])
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
    threadgroup array<uint, int(2064)> ks_1;
    (&kernelContext_3)->ks_0 = &ks_1;
    threadgroup array<uint, int(2064)> vs_1;
    (&kernelContext_3)->vs_0 = &vs_1;
    threadgroup array<uint, int(1032)> qs_1;
    (&kernelContext_3)->qs_0 = &qs_1;
    threadgroup array<float, int(2056)> dos_1;
    (&kernelContext_3)->dos_0 = &dos_1;
    threadgroup array<float, int(8)> ls_1;
    (&kernelContext_3)->ls_0 = &ls_1;
    threadgroup array<float, int(8)> ds_1;
    (&kernelContext_3)->ds_0 = &ds_1;
    threadgroup array<float, int(128)> ps_1;
    (&kernelContext_3)->ps_0 = &ps_1;
    threadgroup array<float, int(128)> dss_1;
    (&kernelContext_3)->dss_0 = &dss_1;
    uint t_3 = local_0.x;
    uint kvHead_1 = group_0.y;
    uint d2_3 = p_1->headDim_0 / 2U;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint rows_2 = hpk_0 * (&kernelContext_3)->p_0->queries_0;
    uint _S39 = kvHead_1 * rows_2;
    uint _S40 = (&kernelContext_3)->p_0->keys_0 - (&kernelContext_3)->p_0->queries_0;
    uint owner_0 = t_3 / 8U;
    uint _S41 = t_3 % 8U;
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
        uint i_2;
        if(((&kernelContext_3)->p_0->causal_0) != 0U)
        {
            uint rLast_0 = min(r0_2 + 8U, rows_2) - 1U;
            uint _S42 = r0_2 / (&kernelContext_3)->p_0->queries_0;
            uint _S43 = rLast_0 / (&kernelContext_3)->p_0->queries_0;
            if(_S42 != _S43)
            {
                i_2 = (&kernelContext_3)->p_0->queries_0 - 1U;
            }
            else
            {
                uint _S44 = rLast_0 % (&kernelContext_3)->p_0->queries_0;
                i_2 = _S44;
            }
            if((i_2 + _S40) < k0_2)
            {
                r0_2 = r0_2 + 8U;
                continue;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loadRows_0(t_3, _S39, r0_2, rows_2, d2_3, &kernelContext_3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pairs_0(t_3, r0_2, rows_2, k0_2, d2_3, &kernelContext_3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_2 = 0U;
        for(;;)
        {
            if(i_2 < 8U)
            {
            }
            else
            {
                break;
            }
            uint _S45 = i_2 * 16U + owner_0;
            float _S46 = (*(&kernelContext_3)->ps_0)[_S45];
            float _S47 = (*(&kernelContext_3)->dss_0)[_S45];
            uint _S48 = i_2 * 257U;
            a1_0[0U] = a1_0[0U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + _S41];
            uint _S49 = i_2 * 129U;
            a0_0[0U] = a0_0[0U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + _S41 / 2U], _S41);
            uint col_0 = _S41 + 8U;
            a1_0[1U] = a1_0[1U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_0];
            a0_0[1U] = a0_0[1U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_0 / 2U], col_0);
            uint col_1 = _S41 + 16U;
            a1_0[2U] = a1_0[2U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_1];
            a0_0[2U] = a0_0[2U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_1 / 2U], col_1);
            uint col_2 = _S41 + 24U;
            a1_0[3U] = a1_0[3U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_2];
            a0_0[3U] = a0_0[3U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_2 / 2U], col_2);
            uint col_3 = _S41 + 32U;
            a1_0[4U] = a1_0[4U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_3];
            a0_0[4U] = a0_0[4U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_3 / 2U], col_3);
            uint col_4 = _S41 + 40U;
            a1_0[5U] = a1_0[5U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_4];
            a0_0[5U] = a0_0[5U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_4 / 2U], col_4);
            uint col_5 = _S41 + 48U;
            a1_0[6U] = a1_0[6U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_5];
            a0_0[6U] = a0_0[6U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_5 / 2U], col_5);
            uint col_6 = _S41 + 56U;
            a1_0[7U] = a1_0[7U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_6];
            a0_0[7U] = a0_0[7U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_6 / 2U], col_6);
            uint col_7 = _S41 + 64U;
            a1_0[8U] = a1_0[8U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_7];
            a0_0[8U] = a0_0[8U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_7 / 2U], col_7);
            uint col_8 = _S41 + 72U;
            a1_0[9U] = a1_0[9U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_8];
            a0_0[9U] = a0_0[9U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_8 / 2U], col_8);
            uint col_9 = _S41 + 80U;
            a1_0[10U] = a1_0[10U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_9];
            a0_0[10U] = a0_0[10U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_9 / 2U], col_9);
            uint col_10 = _S41 + 88U;
            a1_0[11U] = a1_0[11U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_10];
            a0_0[11U] = a0_0[11U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_10 / 2U], col_10);
            uint col_11 = _S41 + 96U;
            a1_0[12U] = a1_0[12U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_11];
            a0_0[12U] = a0_0[12U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_11 / 2U], col_11);
            uint col_12 = _S41 + 104U;
            a1_0[13U] = a1_0[13U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_12];
            a0_0[13U] = a0_0[13U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_12 / 2U], col_12);
            uint col_13 = _S41 + 112U;
            a1_0[14U] = a1_0[14U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_13];
            a0_0[14U] = a0_0[14U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_13 / 2U], col_13);
            uint col_14 = _S41 + 120U;
            a1_0[15U] = a1_0[15U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_14];
            a0_0[15U] = a0_0[15U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_14 / 2U], col_14);
            uint col_15 = _S41 + 128U;
            a1_0[16U] = a1_0[16U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_15];
            a0_0[16U] = a0_0[16U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_15 / 2U], col_15);
            uint col_16 = _S41 + 136U;
            a1_0[17U] = a1_0[17U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_16];
            a0_0[17U] = a0_0[17U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_16 / 2U], col_16);
            uint col_17 = _S41 + 144U;
            a1_0[18U] = a1_0[18U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_17];
            a0_0[18U] = a0_0[18U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_17 / 2U], col_17);
            uint col_18 = _S41 + 152U;
            a1_0[19U] = a1_0[19U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_18];
            a0_0[19U] = a0_0[19U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_18 / 2U], col_18);
            uint col_19 = _S41 + 160U;
            a1_0[20U] = a1_0[20U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_19];
            a0_0[20U] = a0_0[20U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_19 / 2U], col_19);
            uint col_20 = _S41 + 168U;
            a1_0[21U] = a1_0[21U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_20];
            a0_0[21U] = a0_0[21U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_20 / 2U], col_20);
            uint col_21 = _S41 + 176U;
            a1_0[22U] = a1_0[22U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_21];
            a0_0[22U] = a0_0[22U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_21 / 2U], col_21);
            uint col_22 = _S41 + 184U;
            a1_0[23U] = a1_0[23U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_22];
            a0_0[23U] = a0_0[23U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_22 / 2U], col_22);
            uint col_23 = _S41 + 192U;
            a1_0[24U] = a1_0[24U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_23];
            a0_0[24U] = a0_0[24U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_23 / 2U], col_23);
            uint col_24 = _S41 + 200U;
            a1_0[25U] = a1_0[25U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_24];
            a0_0[25U] = a0_0[25U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_24 / 2U], col_24);
            uint col_25 = _S41 + 208U;
            a1_0[26U] = a1_0[26U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_25];
            a0_0[26U] = a0_0[26U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_25 / 2U], col_25);
            uint col_26 = _S41 + 216U;
            a1_0[27U] = a1_0[27U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_26];
            a0_0[27U] = a0_0[27U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_26 / 2U], col_26);
            uint col_27 = _S41 + 224U;
            a1_0[28U] = a1_0[28U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_27];
            a0_0[28U] = a0_0[28U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_27 / 2U], col_27);
            uint col_28 = _S41 + 232U;
            a1_0[29U] = a1_0[29U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_28];
            a0_0[29U] = a0_0[29U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_28 / 2U], col_28);
            uint col_29 = _S41 + 240U;
            a1_0[30U] = a1_0[30U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_29];
            a0_0[30U] = a0_0[30U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_29 / 2U], col_29);
            uint col_30 = _S41 + 248U;
            a1_0[31U] = a1_0[31U] + _S46 * (*(&kernelContext_3)->dos_0)[_S48 + col_30];
            a0_0[31U] = a0_0[31U] + _S47 * half2At_0((*(&kernelContext_3)->qs_0)[_S49 + col_30 / 2U], col_30);
            i_2 = i_2 + 1U;
        }
        r0_2 = r0_2 + 8U;
    }
    uint key_2 = k0_2 + owner_0;
    if(key_2 < ((&kernelContext_3)->p_0->keys_0))
    {
        uint _S50 = (kvHead_1 * (&kernelContext_3)->p_0->cacheKeys_0 + key_2) * p_1->headDim_0;
        if(_S41 < (p_1->headDim_0))
        {
            uint _S51 = _S50 + _S41;
            *((&kernelContext_3)->out0_0+_S51) = a0_0[0U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S51) = a1_0[0U];
        }
        uint col_31 = _S41 + 8U;
        if(col_31 < (p_1->headDim_0))
        {
            uint _S52 = _S50 + col_31;
            *((&kernelContext_3)->out0_0+_S52) = a0_0[1U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S52) = a1_0[1U];
        }
        uint col_32 = _S41 + 16U;
        if(col_32 < (p_1->headDim_0))
        {
            uint _S53 = _S50 + col_32;
            *((&kernelContext_3)->out0_0+_S53) = a0_0[2U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S53) = a1_0[2U];
        }
        uint col_33 = _S41 + 24U;
        if(col_33 < (p_1->headDim_0))
        {
            uint _S54 = _S50 + col_33;
            *((&kernelContext_3)->out0_0+_S54) = a0_0[3U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S54) = a1_0[3U];
        }
        uint col_34 = _S41 + 32U;
        if(col_34 < (p_1->headDim_0))
        {
            uint _S55 = _S50 + col_34;
            *((&kernelContext_3)->out0_0+_S55) = a0_0[4U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S55) = a1_0[4U];
        }
        uint col_35 = _S41 + 40U;
        if(col_35 < (p_1->headDim_0))
        {
            uint _S56 = _S50 + col_35;
            *((&kernelContext_3)->out0_0+_S56) = a0_0[5U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S56) = a1_0[5U];
        }
        uint col_36 = _S41 + 48U;
        if(col_36 < (p_1->headDim_0))
        {
            uint _S57 = _S50 + col_36;
            *((&kernelContext_3)->out0_0+_S57) = a0_0[6U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S57) = a1_0[6U];
        }
        uint col_37 = _S41 + 56U;
        if(col_37 < (p_1->headDim_0))
        {
            uint _S58 = _S50 + col_37;
            *((&kernelContext_3)->out0_0+_S58) = a0_0[7U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S58) = a1_0[7U];
        }
        uint col_38 = _S41 + 64U;
        if(col_38 < (p_1->headDim_0))
        {
            uint _S59 = _S50 + col_38;
            *((&kernelContext_3)->out0_0+_S59) = a0_0[8U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S59) = a1_0[8U];
        }
        uint col_39 = _S41 + 72U;
        if(col_39 < (p_1->headDim_0))
        {
            uint _S60 = _S50 + col_39;
            *((&kernelContext_3)->out0_0+_S60) = a0_0[9U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S60) = a1_0[9U];
        }
        uint col_40 = _S41 + 80U;
        if(col_40 < (p_1->headDim_0))
        {
            uint _S61 = _S50 + col_40;
            *((&kernelContext_3)->out0_0+_S61) = a0_0[10U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S61) = a1_0[10U];
        }
        uint col_41 = _S41 + 88U;
        if(col_41 < (p_1->headDim_0))
        {
            uint _S62 = _S50 + col_41;
            *((&kernelContext_3)->out0_0+_S62) = a0_0[11U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S62) = a1_0[11U];
        }
        uint col_42 = _S41 + 96U;
        if(col_42 < (p_1->headDim_0))
        {
            uint _S63 = _S50 + col_42;
            *((&kernelContext_3)->out0_0+_S63) = a0_0[12U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S63) = a1_0[12U];
        }
        uint col_43 = _S41 + 104U;
        if(col_43 < (p_1->headDim_0))
        {
            uint _S64 = _S50 + col_43;
            *((&kernelContext_3)->out0_0+_S64) = a0_0[13U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S64) = a1_0[13U];
        }
        uint col_44 = _S41 + 112U;
        if(col_44 < (p_1->headDim_0))
        {
            uint _S65 = _S50 + col_44;
            *((&kernelContext_3)->out0_0+_S65) = a0_0[14U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S65) = a1_0[14U];
        }
        uint col_45 = _S41 + 120U;
        if(col_45 < (p_1->headDim_0))
        {
            uint _S66 = _S50 + col_45;
            *((&kernelContext_3)->out0_0+_S66) = a0_0[15U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S66) = a1_0[15U];
        }
        uint col_46 = _S41 + 128U;
        if(col_46 < (p_1->headDim_0))
        {
            uint _S67 = _S50 + col_46;
            *((&kernelContext_3)->out0_0+_S67) = a0_0[16U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S67) = a1_0[16U];
        }
        uint col_47 = _S41 + 136U;
        if(col_47 < (p_1->headDim_0))
        {
            uint _S68 = _S50 + col_47;
            *((&kernelContext_3)->out0_0+_S68) = a0_0[17U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S68) = a1_0[17U];
        }
        uint col_48 = _S41 + 144U;
        if(col_48 < (p_1->headDim_0))
        {
            uint _S69 = _S50 + col_48;
            *((&kernelContext_3)->out0_0+_S69) = a0_0[18U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S69) = a1_0[18U];
        }
        uint col_49 = _S41 + 152U;
        if(col_49 < (p_1->headDim_0))
        {
            uint _S70 = _S50 + col_49;
            *((&kernelContext_3)->out0_0+_S70) = a0_0[19U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S70) = a1_0[19U];
        }
        uint col_50 = _S41 + 160U;
        if(col_50 < (p_1->headDim_0))
        {
            uint _S71 = _S50 + col_50;
            *((&kernelContext_3)->out0_0+_S71) = a0_0[20U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S71) = a1_0[20U];
        }
        uint col_51 = _S41 + 168U;
        if(col_51 < (p_1->headDim_0))
        {
            uint _S72 = _S50 + col_51;
            *((&kernelContext_3)->out0_0+_S72) = a0_0[21U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S72) = a1_0[21U];
        }
        uint col_52 = _S41 + 176U;
        if(col_52 < (p_1->headDim_0))
        {
            uint _S73 = _S50 + col_52;
            *((&kernelContext_3)->out0_0+_S73) = a0_0[22U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S73) = a1_0[22U];
        }
        uint col_53 = _S41 + 184U;
        if(col_53 < (p_1->headDim_0))
        {
            uint _S74 = _S50 + col_53;
            *((&kernelContext_3)->out0_0+_S74) = a0_0[23U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S74) = a1_0[23U];
        }
        uint col_54 = _S41 + 192U;
        if(col_54 < (p_1->headDim_0))
        {
            uint _S75 = _S50 + col_54;
            *((&kernelContext_3)->out0_0+_S75) = a0_0[24U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S75) = a1_0[24U];
        }
        uint col_55 = _S41 + 200U;
        if(col_55 < (p_1->headDim_0))
        {
            uint _S76 = _S50 + col_55;
            *((&kernelContext_3)->out0_0+_S76) = a0_0[25U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S76) = a1_0[25U];
        }
        uint col_56 = _S41 + 208U;
        if(col_56 < (p_1->headDim_0))
        {
            uint _S77 = _S50 + col_56;
            *((&kernelContext_3)->out0_0+_S77) = a0_0[26U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S77) = a1_0[26U];
        }
        uint col_57 = _S41 + 216U;
        if(col_57 < (p_1->headDim_0))
        {
            uint _S78 = _S50 + col_57;
            *((&kernelContext_3)->out0_0+_S78) = a0_0[27U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S78) = a1_0[27U];
        }
        uint col_58 = _S41 + 224U;
        if(col_58 < (p_1->headDim_0))
        {
            uint _S79 = _S50 + col_58;
            *((&kernelContext_3)->out0_0+_S79) = a0_0[28U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S79) = a1_0[28U];
        }
        uint col_59 = _S41 + 232U;
        if(col_59 < (p_1->headDim_0))
        {
            uint _S80 = _S50 + col_59;
            *((&kernelContext_3)->out0_0+_S80) = a0_0[29U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S80) = a1_0[29U];
        }
        uint col_60 = _S41 + 240U;
        if(col_60 < (p_1->headDim_0))
        {
            uint _S81 = _S50 + col_60;
            *((&kernelContext_3)->out0_0+_S81) = a0_0[30U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S81) = a1_0[30U];
        }
        uint col_61 = _S41 + 248U;
        if(col_61 < (p_1->headDim_0))
        {
            uint _S82 = _S50 + col_61;
            *((&kernelContext_3)->out0_0+_S82) = a0_0[31U] * (&kernelContext_3)->p_0->scale_0;
            *((&kernelContext_3)->out1_0+_S82) = a1_0[31U];
        }
    }
    return;
}

