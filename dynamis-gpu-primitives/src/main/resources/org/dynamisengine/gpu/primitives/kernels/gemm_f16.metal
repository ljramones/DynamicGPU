#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct GemmParams_0
{
    uint m_0;
    uint n_0;
    uint k_0;
    uint lda_0;
    uint ldb_0;
    uint ldc_0;
    float alpha_0;
    float beta_0;
};

struct KernelContext_0
{
    GemmParams_0 constant* p_0;
    half device* a_0;
    half device* b_0;
    float device* c_0;
    array<array<float, int(64)>, int(16)> threadgroup* tileA_0;
    array<array<float, int(64)>, int(16)> threadgroup* tileB_0;
};

[[kernel]] void gemm_f16(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(3)]], half device* a_1 [[buffer(0)]], half device* b_1 [[buffer(1)]], float device* c_1 [[buffer(2)]])
{
    float _S1;
    uint _S2;
    uint _S3;
    uint _S4;
    uint _S5;
    uint _S6;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->a_0 = a_1;
    (&kernelContext_0)->b_0 = b_1;
    (&kernelContext_0)->c_0 = c_1;
    threadgroup array<array<float, int(64)>, int(16)> tileA_1;
    (&kernelContext_0)->tileA_0 = &tileA_1;
    threadgroup array<array<float, int(64)>, int(16)> tileB_1;
    (&kernelContext_0)->tileB_0 = &tileB_1;
    uint tx_0 = local_0.x;
    uint ty_0 = local_0.y;
    uint _S7 = ty_0 * 16U + tx_0;
    uint _S8 = group_0.y * 64U;
    uint _S9 = group_0.x * 64U;
    thread array<array<float, int(4)>, int(4)> acc_0;
    acc_0[0U][0U] = 0.0;
    acc_0[0U][1U] = 0.0;
    acc_0[0U][2U] = 0.0;
    acc_0[0U][3U] = 0.0;
    acc_0[1U][0U] = 0.0;
    acc_0[1U][1U] = 0.0;
    acc_0[1U][2U] = 0.0;
    acc_0[1U][3U] = 0.0;
    acc_0[2U][0U] = 0.0;
    acc_0[2U][1U] = 0.0;
    acc_0[2U][2U] = 0.0;
    acc_0[2U][3U] = 0.0;
    acc_0[3U][0U] = 0.0;
    acc_0[3U][1U] = 0.0;
    acc_0[3U][2U] = 0.0;
    acc_0[3U][3U] = 0.0;
    uint k0_0 = 0U;
    for(;;)
    {
        if(k0_0 < ((&kernelContext_0)->p_0->k_0))
        {
        }
        else
        {
            break;
        }
        uint _S10 = _S7 * 4U;
        uint ar_0 = _S10 / 16U;
        uint ak_0 = _S10 % 16U;
        uint gr_0 = _S8 + ar_0;
        uint gk_0 = k0_0 + ak_0;
        bool _S11;
        if(gr_0 < ((&kernelContext_0)->p_0->m_0))
        {
            _S11 = gk_0 < ((&kernelContext_0)->p_0->k_0);
        }
        else
        {
            _S11 = false;
        }
        if(_S11)
        {
            _S1 = float(*((&kernelContext_0)->a_0+(gr_0 * (&kernelContext_0)->p_0->lda_0 + gk_0)));
        }
        else
        {
            _S1 = 0.0;
        }
        (*(&kernelContext_0)->tileA_0)[ak_0][ar_0] = _S1;
        uint br_0 = _S10 / 64U;
        uint bc_0 = _S10 % 64U;
        uint hk_0 = k0_0 + br_0;
        uint gc_0 = _S9 + bc_0;
        bool _S12;
        if(hk_0 < ((&kernelContext_0)->p_0->k_0))
        {
            _S12 = gc_0 < ((&kernelContext_0)->p_0->n_0);
        }
        else
        {
            _S12 = false;
        }
        float _S13;
        if(_S12)
        {
            _S13 = float(*((&kernelContext_0)->b_0+(hk_0 * (&kernelContext_0)->p_0->ldb_0 + gc_0)));
        }
        else
        {
            _S13 = 0.0;
        }
        (*(&kernelContext_0)->tileB_0)[br_0][bc_0] = _S13;
        uint e_0 = _S10 + 1U;
        uint ar_1 = e_0 / 16U;
        uint ak_1 = e_0 % 16U;
        uint gr_1 = _S8 + ar_1;
        uint gk_1 = k0_0 + ak_1;
        bool _S14;
        if(gr_1 < ((&kernelContext_0)->p_0->m_0))
        {
            _S14 = gk_1 < ((&kernelContext_0)->p_0->k_0);
        }
        else
        {
            _S14 = false;
        }
        float _S15;
        if(_S14)
        {
            _S15 = float(*((&kernelContext_0)->a_0+(gr_1 * (&kernelContext_0)->p_0->lda_0 + gk_1)));
        }
        else
        {
            _S15 = 0.0;
        }
        (*(&kernelContext_0)->tileA_0)[ak_1][ar_1] = _S15;
        uint br_1 = e_0 / 64U;
        uint bc_1 = e_0 % 64U;
        uint hk_1 = k0_0 + br_1;
        uint gc_1 = _S9 + bc_1;
        bool _S16;
        if(hk_1 < ((&kernelContext_0)->p_0->k_0))
        {
            _S16 = gc_1 < ((&kernelContext_0)->p_0->n_0);
        }
        else
        {
            _S16 = false;
        }
        float _S17;
        if(_S16)
        {
            _S17 = float(*((&kernelContext_0)->b_0+(hk_1 * (&kernelContext_0)->p_0->ldb_0 + gc_1)));
        }
        else
        {
            _S17 = 0.0;
        }
        (*(&kernelContext_0)->tileB_0)[br_1][bc_1] = _S17;
        uint e_1 = _S10 + 2U;
        uint ar_2 = e_1 / 16U;
        uint ak_2 = e_1 % 16U;
        uint gr_2 = _S8 + ar_2;
        uint gk_2 = k0_0 + ak_2;
        bool _S18;
        if(gr_2 < ((&kernelContext_0)->p_0->m_0))
        {
            _S18 = gk_2 < ((&kernelContext_0)->p_0->k_0);
        }
        else
        {
            _S18 = false;
        }
        float _S19;
        if(_S18)
        {
            _S19 = float(*((&kernelContext_0)->a_0+(gr_2 * (&kernelContext_0)->p_0->lda_0 + gk_2)));
        }
        else
        {
            _S19 = 0.0;
        }
        (*(&kernelContext_0)->tileA_0)[ak_2][ar_2] = _S19;
        uint br_2 = e_1 / 64U;
        uint bc_2 = e_1 % 64U;
        uint hk_2 = k0_0 + br_2;
        uint gc_2 = _S9 + bc_2;
        bool _S20;
        if(hk_2 < ((&kernelContext_0)->p_0->k_0))
        {
            _S20 = gc_2 < ((&kernelContext_0)->p_0->n_0);
        }
        else
        {
            _S20 = false;
        }
        float _S21;
        if(_S20)
        {
            _S21 = float(*((&kernelContext_0)->b_0+(hk_2 * (&kernelContext_0)->p_0->ldb_0 + gc_2)));
        }
        else
        {
            _S21 = 0.0;
        }
        (*(&kernelContext_0)->tileB_0)[br_2][bc_2] = _S21;
        uint e_2 = _S10 + 3U;
        uint ar_3 = e_2 / 16U;
        uint ak_3 = e_2 % 16U;
        uint gr_3 = _S8 + ar_3;
        uint gk_3 = k0_0 + ak_3;
        bool _S22;
        if(gr_3 < ((&kernelContext_0)->p_0->m_0))
        {
            _S22 = gk_3 < ((&kernelContext_0)->p_0->k_0);
        }
        else
        {
            _S22 = false;
        }
        float _S23;
        if(_S22)
        {
            _S23 = float(*((&kernelContext_0)->a_0+(gr_3 * (&kernelContext_0)->p_0->lda_0 + gk_3)));
        }
        else
        {
            _S23 = 0.0;
        }
        (*(&kernelContext_0)->tileA_0)[ak_3][ar_3] = _S23;
        uint br_3 = e_2 / 64U;
        uint bc_3 = e_2 % 64U;
        uint hk_3 = k0_0 + br_3;
        uint gc_3 = _S9 + bc_3;
        bool _S24;
        if(hk_3 < ((&kernelContext_0)->p_0->k_0))
        {
            _S24 = gc_3 < ((&kernelContext_0)->p_0->n_0);
        }
        else
        {
            _S24 = false;
        }
        float _S25;
        if(_S24)
        {
            _S25 = float(*((&kernelContext_0)->b_0+(hk_3 * (&kernelContext_0)->p_0->ldb_0 + gc_3)));
        }
        else
        {
            _S25 = 0.0;
        }
        (*(&kernelContext_0)->tileB_0)[br_3][bc_3] = _S25;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        thread array<float, int(4)> av_0;
        float _S26 = (*(&kernelContext_0)->tileA_0)[0U][ty_0];
        av_0[0U] = (*(&kernelContext_0)->tileA_0)[0U][ty_0];
        uint _S27 = ty_0 + 16U;
        float _S28 = (*(&kernelContext_0)->tileA_0)[0U][_S27];
        av_0[1U] = (*(&kernelContext_0)->tileA_0)[0U][_S27];
        uint _S29 = ty_0 + 32U;
        float _S30 = (*(&kernelContext_0)->tileA_0)[0U][_S29];
        av_0[2U] = (*(&kernelContext_0)->tileA_0)[0U][_S29];
        uint _S31 = ty_0 + 48U;
        float _S32 = (*(&kernelContext_0)->tileA_0)[0U][_S31];
        av_0[3U] = (*(&kernelContext_0)->tileA_0)[0U][_S31];
        thread array<float, int(4)> bv_0;
        float _S33 = (*(&kernelContext_0)->tileB_0)[0U][tx_0];
        bv_0[0U] = (*(&kernelContext_0)->tileB_0)[0U][tx_0];
        uint _S34 = tx_0 + 16U;
        float _S35 = (*(&kernelContext_0)->tileB_0)[0U][_S34];
        bv_0[1U] = (*(&kernelContext_0)->tileB_0)[0U][_S34];
        uint _S36 = tx_0 + 32U;
        float _S37 = (*(&kernelContext_0)->tileB_0)[0U][_S36];
        bv_0[2U] = (*(&kernelContext_0)->tileB_0)[0U][_S36];
        uint _S38 = tx_0 + 48U;
        float _S39 = (*(&kernelContext_0)->tileB_0)[0U][_S38];
        bv_0[3U] = (*(&kernelContext_0)->tileB_0)[0U][_S38];
        acc_0[0U][0U] = fma(_S26, _S33, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S26, _S35, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S26, _S37, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S26, _S39, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S28, _S33, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S28, _S35, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S28, _S37, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S28, _S39, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S30, _S33, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S30, _S35, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S30, _S37, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S30, _S39, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S32, _S33, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S32, _S35, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S32, _S37, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S32, _S39, acc_0[3U][3U]);
        thread array<float, int(4)> av_1;
        float _S40 = (*(&kernelContext_0)->tileA_0)[1U][ty_0];
        av_1[0U] = (*(&kernelContext_0)->tileA_0)[1U][ty_0];
        float _S41 = (*(&kernelContext_0)->tileA_0)[1U][_S27];
        av_1[1U] = (*(&kernelContext_0)->tileA_0)[1U][_S27];
        float _S42 = (*(&kernelContext_0)->tileA_0)[1U][_S29];
        av_1[2U] = (*(&kernelContext_0)->tileA_0)[1U][_S29];
        float _S43 = (*(&kernelContext_0)->tileA_0)[1U][_S31];
        av_1[3U] = (*(&kernelContext_0)->tileA_0)[1U][_S31];
        thread array<float, int(4)> bv_1;
        float _S44 = (*(&kernelContext_0)->tileB_0)[1U][tx_0];
        bv_1[0U] = (*(&kernelContext_0)->tileB_0)[1U][tx_0];
        float _S45 = (*(&kernelContext_0)->tileB_0)[1U][_S34];
        bv_1[1U] = (*(&kernelContext_0)->tileB_0)[1U][_S34];
        float _S46 = (*(&kernelContext_0)->tileB_0)[1U][_S36];
        bv_1[2U] = (*(&kernelContext_0)->tileB_0)[1U][_S36];
        float _S47 = (*(&kernelContext_0)->tileB_0)[1U][_S38];
        bv_1[3U] = (*(&kernelContext_0)->tileB_0)[1U][_S38];
        acc_0[0U][0U] = fma(_S40, _S44, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S40, _S45, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S40, _S46, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S40, _S47, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S41, _S44, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S41, _S45, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S41, _S46, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S41, _S47, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S42, _S44, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S42, _S45, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S42, _S46, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S42, _S47, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S43, _S44, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S43, _S45, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S43, _S46, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S43, _S47, acc_0[3U][3U]);
        thread array<float, int(4)> av_2;
        float _S48 = (*(&kernelContext_0)->tileA_0)[2U][ty_0];
        av_2[0U] = (*(&kernelContext_0)->tileA_0)[2U][ty_0];
        float _S49 = (*(&kernelContext_0)->tileA_0)[2U][_S27];
        av_2[1U] = (*(&kernelContext_0)->tileA_0)[2U][_S27];
        float _S50 = (*(&kernelContext_0)->tileA_0)[2U][_S29];
        av_2[2U] = (*(&kernelContext_0)->tileA_0)[2U][_S29];
        float _S51 = (*(&kernelContext_0)->tileA_0)[2U][_S31];
        av_2[3U] = (*(&kernelContext_0)->tileA_0)[2U][_S31];
        thread array<float, int(4)> bv_2;
        float _S52 = (*(&kernelContext_0)->tileB_0)[2U][tx_0];
        bv_2[0U] = (*(&kernelContext_0)->tileB_0)[2U][tx_0];
        float _S53 = (*(&kernelContext_0)->tileB_0)[2U][_S34];
        bv_2[1U] = (*(&kernelContext_0)->tileB_0)[2U][_S34];
        float _S54 = (*(&kernelContext_0)->tileB_0)[2U][_S36];
        bv_2[2U] = (*(&kernelContext_0)->tileB_0)[2U][_S36];
        float _S55 = (*(&kernelContext_0)->tileB_0)[2U][_S38];
        bv_2[3U] = (*(&kernelContext_0)->tileB_0)[2U][_S38];
        acc_0[0U][0U] = fma(_S48, _S52, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S48, _S53, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S48, _S54, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S48, _S55, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S49, _S52, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S49, _S53, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S49, _S54, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S49, _S55, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S50, _S52, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S50, _S53, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S50, _S54, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S50, _S55, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S51, _S52, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S51, _S53, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S51, _S54, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S51, _S55, acc_0[3U][3U]);
        thread array<float, int(4)> av_3;
        float _S56 = (*(&kernelContext_0)->tileA_0)[3U][ty_0];
        av_3[0U] = (*(&kernelContext_0)->tileA_0)[3U][ty_0];
        float _S57 = (*(&kernelContext_0)->tileA_0)[3U][_S27];
        av_3[1U] = (*(&kernelContext_0)->tileA_0)[3U][_S27];
        float _S58 = (*(&kernelContext_0)->tileA_0)[3U][_S29];
        av_3[2U] = (*(&kernelContext_0)->tileA_0)[3U][_S29];
        float _S59 = (*(&kernelContext_0)->tileA_0)[3U][_S31];
        av_3[3U] = (*(&kernelContext_0)->tileA_0)[3U][_S31];
        thread array<float, int(4)> bv_3;
        float _S60 = (*(&kernelContext_0)->tileB_0)[3U][tx_0];
        bv_3[0U] = (*(&kernelContext_0)->tileB_0)[3U][tx_0];
        float _S61 = (*(&kernelContext_0)->tileB_0)[3U][_S34];
        bv_3[1U] = (*(&kernelContext_0)->tileB_0)[3U][_S34];
        float _S62 = (*(&kernelContext_0)->tileB_0)[3U][_S36];
        bv_3[2U] = (*(&kernelContext_0)->tileB_0)[3U][_S36];
        float _S63 = (*(&kernelContext_0)->tileB_0)[3U][_S38];
        bv_3[3U] = (*(&kernelContext_0)->tileB_0)[3U][_S38];
        acc_0[0U][0U] = fma(_S56, _S60, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S56, _S61, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S56, _S62, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S56, _S63, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S57, _S60, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S57, _S61, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S57, _S62, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S57, _S63, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S58, _S60, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S58, _S61, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S58, _S62, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S58, _S63, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S59, _S60, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S59, _S61, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S59, _S62, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S59, _S63, acc_0[3U][3U]);
        thread array<float, int(4)> av_4;
        float _S64 = (*(&kernelContext_0)->tileA_0)[4U][ty_0];
        av_4[0U] = (*(&kernelContext_0)->tileA_0)[4U][ty_0];
        float _S65 = (*(&kernelContext_0)->tileA_0)[4U][_S27];
        av_4[1U] = (*(&kernelContext_0)->tileA_0)[4U][_S27];
        float _S66 = (*(&kernelContext_0)->tileA_0)[4U][_S29];
        av_4[2U] = (*(&kernelContext_0)->tileA_0)[4U][_S29];
        float _S67 = (*(&kernelContext_0)->tileA_0)[4U][_S31];
        av_4[3U] = (*(&kernelContext_0)->tileA_0)[4U][_S31];
        thread array<float, int(4)> bv_4;
        float _S68 = (*(&kernelContext_0)->tileB_0)[4U][tx_0];
        bv_4[0U] = (*(&kernelContext_0)->tileB_0)[4U][tx_0];
        float _S69 = (*(&kernelContext_0)->tileB_0)[4U][_S34];
        bv_4[1U] = (*(&kernelContext_0)->tileB_0)[4U][_S34];
        float _S70 = (*(&kernelContext_0)->tileB_0)[4U][_S36];
        bv_4[2U] = (*(&kernelContext_0)->tileB_0)[4U][_S36];
        float _S71 = (*(&kernelContext_0)->tileB_0)[4U][_S38];
        bv_4[3U] = (*(&kernelContext_0)->tileB_0)[4U][_S38];
        acc_0[0U][0U] = fma(_S64, _S68, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S64, _S69, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S64, _S70, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S64, _S71, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S65, _S68, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S65, _S69, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S65, _S70, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S65, _S71, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S66, _S68, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S66, _S69, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S66, _S70, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S66, _S71, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S67, _S68, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S67, _S69, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S67, _S70, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S67, _S71, acc_0[3U][3U]);
        thread array<float, int(4)> av_5;
        float _S72 = (*(&kernelContext_0)->tileA_0)[5U][ty_0];
        av_5[0U] = (*(&kernelContext_0)->tileA_0)[5U][ty_0];
        float _S73 = (*(&kernelContext_0)->tileA_0)[5U][_S27];
        av_5[1U] = (*(&kernelContext_0)->tileA_0)[5U][_S27];
        float _S74 = (*(&kernelContext_0)->tileA_0)[5U][_S29];
        av_5[2U] = (*(&kernelContext_0)->tileA_0)[5U][_S29];
        float _S75 = (*(&kernelContext_0)->tileA_0)[5U][_S31];
        av_5[3U] = (*(&kernelContext_0)->tileA_0)[5U][_S31];
        thread array<float, int(4)> bv_5;
        float _S76 = (*(&kernelContext_0)->tileB_0)[5U][tx_0];
        bv_5[0U] = (*(&kernelContext_0)->tileB_0)[5U][tx_0];
        float _S77 = (*(&kernelContext_0)->tileB_0)[5U][_S34];
        bv_5[1U] = (*(&kernelContext_0)->tileB_0)[5U][_S34];
        float _S78 = (*(&kernelContext_0)->tileB_0)[5U][_S36];
        bv_5[2U] = (*(&kernelContext_0)->tileB_0)[5U][_S36];
        float _S79 = (*(&kernelContext_0)->tileB_0)[5U][_S38];
        bv_5[3U] = (*(&kernelContext_0)->tileB_0)[5U][_S38];
        acc_0[0U][0U] = fma(_S72, _S76, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S72, _S77, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S72, _S78, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S72, _S79, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S73, _S76, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S73, _S77, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S73, _S78, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S73, _S79, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S74, _S76, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S74, _S77, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S74, _S78, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S74, _S79, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S75, _S76, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S75, _S77, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S75, _S78, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S75, _S79, acc_0[3U][3U]);
        thread array<float, int(4)> av_6;
        float _S80 = (*(&kernelContext_0)->tileA_0)[6U][ty_0];
        av_6[0U] = (*(&kernelContext_0)->tileA_0)[6U][ty_0];
        float _S81 = (*(&kernelContext_0)->tileA_0)[6U][_S27];
        av_6[1U] = (*(&kernelContext_0)->tileA_0)[6U][_S27];
        float _S82 = (*(&kernelContext_0)->tileA_0)[6U][_S29];
        av_6[2U] = (*(&kernelContext_0)->tileA_0)[6U][_S29];
        float _S83 = (*(&kernelContext_0)->tileA_0)[6U][_S31];
        av_6[3U] = (*(&kernelContext_0)->tileA_0)[6U][_S31];
        thread array<float, int(4)> bv_6;
        float _S84 = (*(&kernelContext_0)->tileB_0)[6U][tx_0];
        bv_6[0U] = (*(&kernelContext_0)->tileB_0)[6U][tx_0];
        float _S85 = (*(&kernelContext_0)->tileB_0)[6U][_S34];
        bv_6[1U] = (*(&kernelContext_0)->tileB_0)[6U][_S34];
        float _S86 = (*(&kernelContext_0)->tileB_0)[6U][_S36];
        bv_6[2U] = (*(&kernelContext_0)->tileB_0)[6U][_S36];
        float _S87 = (*(&kernelContext_0)->tileB_0)[6U][_S38];
        bv_6[3U] = (*(&kernelContext_0)->tileB_0)[6U][_S38];
        acc_0[0U][0U] = fma(_S80, _S84, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S80, _S85, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S80, _S86, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S80, _S87, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S81, _S84, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S81, _S85, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S81, _S86, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S81, _S87, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S82, _S84, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S82, _S85, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S82, _S86, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S82, _S87, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S83, _S84, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S83, _S85, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S83, _S86, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S83, _S87, acc_0[3U][3U]);
        thread array<float, int(4)> av_7;
        float _S88 = (*(&kernelContext_0)->tileA_0)[7U][ty_0];
        av_7[0U] = (*(&kernelContext_0)->tileA_0)[7U][ty_0];
        float _S89 = (*(&kernelContext_0)->tileA_0)[7U][_S27];
        av_7[1U] = (*(&kernelContext_0)->tileA_0)[7U][_S27];
        float _S90 = (*(&kernelContext_0)->tileA_0)[7U][_S29];
        av_7[2U] = (*(&kernelContext_0)->tileA_0)[7U][_S29];
        float _S91 = (*(&kernelContext_0)->tileA_0)[7U][_S31];
        av_7[3U] = (*(&kernelContext_0)->tileA_0)[7U][_S31];
        thread array<float, int(4)> bv_7;
        float _S92 = (*(&kernelContext_0)->tileB_0)[7U][tx_0];
        bv_7[0U] = (*(&kernelContext_0)->tileB_0)[7U][tx_0];
        float _S93 = (*(&kernelContext_0)->tileB_0)[7U][_S34];
        bv_7[1U] = (*(&kernelContext_0)->tileB_0)[7U][_S34];
        float _S94 = (*(&kernelContext_0)->tileB_0)[7U][_S36];
        bv_7[2U] = (*(&kernelContext_0)->tileB_0)[7U][_S36];
        float _S95 = (*(&kernelContext_0)->tileB_0)[7U][_S38];
        bv_7[3U] = (*(&kernelContext_0)->tileB_0)[7U][_S38];
        acc_0[0U][0U] = fma(_S88, _S92, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S88, _S93, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S88, _S94, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S88, _S95, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S89, _S92, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S89, _S93, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S89, _S94, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S89, _S95, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S90, _S92, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S90, _S93, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S90, _S94, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S90, _S95, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S91, _S92, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S91, _S93, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S91, _S94, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S91, _S95, acc_0[3U][3U]);
        thread array<float, int(4)> av_8;
        float _S96 = (*(&kernelContext_0)->tileA_0)[8U][ty_0];
        av_8[0U] = (*(&kernelContext_0)->tileA_0)[8U][ty_0];
        float _S97 = (*(&kernelContext_0)->tileA_0)[8U][_S27];
        av_8[1U] = (*(&kernelContext_0)->tileA_0)[8U][_S27];
        float _S98 = (*(&kernelContext_0)->tileA_0)[8U][_S29];
        av_8[2U] = (*(&kernelContext_0)->tileA_0)[8U][_S29];
        float _S99 = (*(&kernelContext_0)->tileA_0)[8U][_S31];
        av_8[3U] = (*(&kernelContext_0)->tileA_0)[8U][_S31];
        thread array<float, int(4)> bv_8;
        float _S100 = (*(&kernelContext_0)->tileB_0)[8U][tx_0];
        bv_8[0U] = (*(&kernelContext_0)->tileB_0)[8U][tx_0];
        float _S101 = (*(&kernelContext_0)->tileB_0)[8U][_S34];
        bv_8[1U] = (*(&kernelContext_0)->tileB_0)[8U][_S34];
        float _S102 = (*(&kernelContext_0)->tileB_0)[8U][_S36];
        bv_8[2U] = (*(&kernelContext_0)->tileB_0)[8U][_S36];
        float _S103 = (*(&kernelContext_0)->tileB_0)[8U][_S38];
        bv_8[3U] = (*(&kernelContext_0)->tileB_0)[8U][_S38];
        acc_0[0U][0U] = fma(_S96, _S100, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S96, _S101, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S96, _S102, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S96, _S103, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S97, _S100, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S97, _S101, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S97, _S102, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S97, _S103, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S98, _S100, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S98, _S101, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S98, _S102, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S98, _S103, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S99, _S100, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S99, _S101, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S99, _S102, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S99, _S103, acc_0[3U][3U]);
        thread array<float, int(4)> av_9;
        float _S104 = (*(&kernelContext_0)->tileA_0)[9U][ty_0];
        av_9[0U] = (*(&kernelContext_0)->tileA_0)[9U][ty_0];
        float _S105 = (*(&kernelContext_0)->tileA_0)[9U][_S27];
        av_9[1U] = (*(&kernelContext_0)->tileA_0)[9U][_S27];
        float _S106 = (*(&kernelContext_0)->tileA_0)[9U][_S29];
        av_9[2U] = (*(&kernelContext_0)->tileA_0)[9U][_S29];
        float _S107 = (*(&kernelContext_0)->tileA_0)[9U][_S31];
        av_9[3U] = (*(&kernelContext_0)->tileA_0)[9U][_S31];
        thread array<float, int(4)> bv_9;
        float _S108 = (*(&kernelContext_0)->tileB_0)[9U][tx_0];
        bv_9[0U] = (*(&kernelContext_0)->tileB_0)[9U][tx_0];
        float _S109 = (*(&kernelContext_0)->tileB_0)[9U][_S34];
        bv_9[1U] = (*(&kernelContext_0)->tileB_0)[9U][_S34];
        float _S110 = (*(&kernelContext_0)->tileB_0)[9U][_S36];
        bv_9[2U] = (*(&kernelContext_0)->tileB_0)[9U][_S36];
        float _S111 = (*(&kernelContext_0)->tileB_0)[9U][_S38];
        bv_9[3U] = (*(&kernelContext_0)->tileB_0)[9U][_S38];
        acc_0[0U][0U] = fma(_S104, _S108, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S104, _S109, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S104, _S110, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S104, _S111, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S105, _S108, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S105, _S109, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S105, _S110, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S105, _S111, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S106, _S108, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S106, _S109, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S106, _S110, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S106, _S111, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S107, _S108, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S107, _S109, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S107, _S110, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S107, _S111, acc_0[3U][3U]);
        thread array<float, int(4)> av_10;
        float _S112 = (*(&kernelContext_0)->tileA_0)[10U][ty_0];
        av_10[0U] = (*(&kernelContext_0)->tileA_0)[10U][ty_0];
        float _S113 = (*(&kernelContext_0)->tileA_0)[10U][_S27];
        av_10[1U] = (*(&kernelContext_0)->tileA_0)[10U][_S27];
        float _S114 = (*(&kernelContext_0)->tileA_0)[10U][_S29];
        av_10[2U] = (*(&kernelContext_0)->tileA_0)[10U][_S29];
        float _S115 = (*(&kernelContext_0)->tileA_0)[10U][_S31];
        av_10[3U] = (*(&kernelContext_0)->tileA_0)[10U][_S31];
        thread array<float, int(4)> bv_10;
        float _S116 = (*(&kernelContext_0)->tileB_0)[10U][tx_0];
        bv_10[0U] = (*(&kernelContext_0)->tileB_0)[10U][tx_0];
        float _S117 = (*(&kernelContext_0)->tileB_0)[10U][_S34];
        bv_10[1U] = (*(&kernelContext_0)->tileB_0)[10U][_S34];
        float _S118 = (*(&kernelContext_0)->tileB_0)[10U][_S36];
        bv_10[2U] = (*(&kernelContext_0)->tileB_0)[10U][_S36];
        float _S119 = (*(&kernelContext_0)->tileB_0)[10U][_S38];
        bv_10[3U] = (*(&kernelContext_0)->tileB_0)[10U][_S38];
        acc_0[0U][0U] = fma(_S112, _S116, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S112, _S117, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S112, _S118, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S112, _S119, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S113, _S116, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S113, _S117, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S113, _S118, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S113, _S119, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S114, _S116, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S114, _S117, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S114, _S118, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S114, _S119, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S115, _S116, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S115, _S117, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S115, _S118, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S115, _S119, acc_0[3U][3U]);
        thread array<float, int(4)> av_11;
        float _S120 = (*(&kernelContext_0)->tileA_0)[11U][ty_0];
        av_11[0U] = (*(&kernelContext_0)->tileA_0)[11U][ty_0];
        float _S121 = (*(&kernelContext_0)->tileA_0)[11U][_S27];
        av_11[1U] = (*(&kernelContext_0)->tileA_0)[11U][_S27];
        float _S122 = (*(&kernelContext_0)->tileA_0)[11U][_S29];
        av_11[2U] = (*(&kernelContext_0)->tileA_0)[11U][_S29];
        float _S123 = (*(&kernelContext_0)->tileA_0)[11U][_S31];
        av_11[3U] = (*(&kernelContext_0)->tileA_0)[11U][_S31];
        thread array<float, int(4)> bv_11;
        float _S124 = (*(&kernelContext_0)->tileB_0)[11U][tx_0];
        bv_11[0U] = (*(&kernelContext_0)->tileB_0)[11U][tx_0];
        float _S125 = (*(&kernelContext_0)->tileB_0)[11U][_S34];
        bv_11[1U] = (*(&kernelContext_0)->tileB_0)[11U][_S34];
        float _S126 = (*(&kernelContext_0)->tileB_0)[11U][_S36];
        bv_11[2U] = (*(&kernelContext_0)->tileB_0)[11U][_S36];
        float _S127 = (*(&kernelContext_0)->tileB_0)[11U][_S38];
        bv_11[3U] = (*(&kernelContext_0)->tileB_0)[11U][_S38];
        acc_0[0U][0U] = fma(_S120, _S124, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S120, _S125, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S120, _S126, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S120, _S127, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S121, _S124, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S121, _S125, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S121, _S126, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S121, _S127, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S122, _S124, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S122, _S125, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S122, _S126, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S122, _S127, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S123, _S124, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S123, _S125, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S123, _S126, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S123, _S127, acc_0[3U][3U]);
        thread array<float, int(4)> av_12;
        float _S128 = (*(&kernelContext_0)->tileA_0)[12U][ty_0];
        av_12[0U] = (*(&kernelContext_0)->tileA_0)[12U][ty_0];
        float _S129 = (*(&kernelContext_0)->tileA_0)[12U][_S27];
        av_12[1U] = (*(&kernelContext_0)->tileA_0)[12U][_S27];
        float _S130 = (*(&kernelContext_0)->tileA_0)[12U][_S29];
        av_12[2U] = (*(&kernelContext_0)->tileA_0)[12U][_S29];
        float _S131 = (*(&kernelContext_0)->tileA_0)[12U][_S31];
        av_12[3U] = (*(&kernelContext_0)->tileA_0)[12U][_S31];
        thread array<float, int(4)> bv_12;
        float _S132 = (*(&kernelContext_0)->tileB_0)[12U][tx_0];
        bv_12[0U] = (*(&kernelContext_0)->tileB_0)[12U][tx_0];
        float _S133 = (*(&kernelContext_0)->tileB_0)[12U][_S34];
        bv_12[1U] = (*(&kernelContext_0)->tileB_0)[12U][_S34];
        float _S134 = (*(&kernelContext_0)->tileB_0)[12U][_S36];
        bv_12[2U] = (*(&kernelContext_0)->tileB_0)[12U][_S36];
        float _S135 = (*(&kernelContext_0)->tileB_0)[12U][_S38];
        bv_12[3U] = (*(&kernelContext_0)->tileB_0)[12U][_S38];
        acc_0[0U][0U] = fma(_S128, _S132, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S128, _S133, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S128, _S134, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S128, _S135, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S129, _S132, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S129, _S133, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S129, _S134, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S129, _S135, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S130, _S132, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S130, _S133, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S130, _S134, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S130, _S135, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S131, _S132, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S131, _S133, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S131, _S134, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S131, _S135, acc_0[3U][3U]);
        thread array<float, int(4)> av_13;
        float _S136 = (*(&kernelContext_0)->tileA_0)[13U][ty_0];
        av_13[0U] = (*(&kernelContext_0)->tileA_0)[13U][ty_0];
        float _S137 = (*(&kernelContext_0)->tileA_0)[13U][_S27];
        av_13[1U] = (*(&kernelContext_0)->tileA_0)[13U][_S27];
        float _S138 = (*(&kernelContext_0)->tileA_0)[13U][_S29];
        av_13[2U] = (*(&kernelContext_0)->tileA_0)[13U][_S29];
        float _S139 = (*(&kernelContext_0)->tileA_0)[13U][_S31];
        av_13[3U] = (*(&kernelContext_0)->tileA_0)[13U][_S31];
        thread array<float, int(4)> bv_13;
        float _S140 = (*(&kernelContext_0)->tileB_0)[13U][tx_0];
        bv_13[0U] = (*(&kernelContext_0)->tileB_0)[13U][tx_0];
        float _S141 = (*(&kernelContext_0)->tileB_0)[13U][_S34];
        bv_13[1U] = (*(&kernelContext_0)->tileB_0)[13U][_S34];
        float _S142 = (*(&kernelContext_0)->tileB_0)[13U][_S36];
        bv_13[2U] = (*(&kernelContext_0)->tileB_0)[13U][_S36];
        float _S143 = (*(&kernelContext_0)->tileB_0)[13U][_S38];
        bv_13[3U] = (*(&kernelContext_0)->tileB_0)[13U][_S38];
        acc_0[0U][0U] = fma(_S136, _S140, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S136, _S141, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S136, _S142, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S136, _S143, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S137, _S140, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S137, _S141, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S137, _S142, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S137, _S143, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S138, _S140, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S138, _S141, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S138, _S142, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S138, _S143, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S139, _S140, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S139, _S141, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S139, _S142, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S139, _S143, acc_0[3U][3U]);
        thread array<float, int(4)> av_14;
        float _S144 = (*(&kernelContext_0)->tileA_0)[14U][ty_0];
        av_14[0U] = (*(&kernelContext_0)->tileA_0)[14U][ty_0];
        float _S145 = (*(&kernelContext_0)->tileA_0)[14U][_S27];
        av_14[1U] = (*(&kernelContext_0)->tileA_0)[14U][_S27];
        float _S146 = (*(&kernelContext_0)->tileA_0)[14U][_S29];
        av_14[2U] = (*(&kernelContext_0)->tileA_0)[14U][_S29];
        float _S147 = (*(&kernelContext_0)->tileA_0)[14U][_S31];
        av_14[3U] = (*(&kernelContext_0)->tileA_0)[14U][_S31];
        thread array<float, int(4)> bv_14;
        float _S148 = (*(&kernelContext_0)->tileB_0)[14U][tx_0];
        bv_14[0U] = (*(&kernelContext_0)->tileB_0)[14U][tx_0];
        float _S149 = (*(&kernelContext_0)->tileB_0)[14U][_S34];
        bv_14[1U] = (*(&kernelContext_0)->tileB_0)[14U][_S34];
        float _S150 = (*(&kernelContext_0)->tileB_0)[14U][_S36];
        bv_14[2U] = (*(&kernelContext_0)->tileB_0)[14U][_S36];
        float _S151 = (*(&kernelContext_0)->tileB_0)[14U][_S38];
        bv_14[3U] = (*(&kernelContext_0)->tileB_0)[14U][_S38];
        acc_0[0U][0U] = fma(_S144, _S148, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S144, _S149, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S144, _S150, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S144, _S151, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S145, _S148, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S145, _S149, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S145, _S150, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S145, _S151, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S146, _S148, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S146, _S149, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S146, _S150, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S146, _S151, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S147, _S148, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S147, _S149, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S147, _S150, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S147, _S151, acc_0[3U][3U]);
        thread array<float, int(4)> av_15;
        float _S152 = (*(&kernelContext_0)->tileA_0)[15U][ty_0];
        av_15[0U] = (*(&kernelContext_0)->tileA_0)[15U][ty_0];
        float _S153 = (*(&kernelContext_0)->tileA_0)[15U][_S27];
        av_15[1U] = (*(&kernelContext_0)->tileA_0)[15U][_S27];
        float _S154 = (*(&kernelContext_0)->tileA_0)[15U][_S29];
        av_15[2U] = (*(&kernelContext_0)->tileA_0)[15U][_S29];
        float _S155 = (*(&kernelContext_0)->tileA_0)[15U][_S31];
        av_15[3U] = (*(&kernelContext_0)->tileA_0)[15U][_S31];
        thread array<float, int(4)> bv_15;
        float _S156 = (*(&kernelContext_0)->tileB_0)[15U][tx_0];
        bv_15[0U] = (*(&kernelContext_0)->tileB_0)[15U][tx_0];
        float _S157 = (*(&kernelContext_0)->tileB_0)[15U][_S34];
        bv_15[1U] = (*(&kernelContext_0)->tileB_0)[15U][_S34];
        float _S158 = (*(&kernelContext_0)->tileB_0)[15U][_S36];
        bv_15[2U] = (*(&kernelContext_0)->tileB_0)[15U][_S36];
        float _S159 = (*(&kernelContext_0)->tileB_0)[15U][_S38];
        bv_15[3U] = (*(&kernelContext_0)->tileB_0)[15U][_S38];
        acc_0[0U][0U] = fma(_S152, _S156, acc_0[0U][0U]);
        acc_0[0U][1U] = fma(_S152, _S157, acc_0[0U][1U]);
        acc_0[0U][2U] = fma(_S152, _S158, acc_0[0U][2U]);
        acc_0[0U][3U] = fma(_S152, _S159, acc_0[0U][3U]);
        acc_0[1U][0U] = fma(_S153, _S156, acc_0[1U][0U]);
        acc_0[1U][1U] = fma(_S153, _S157, acc_0[1U][1U]);
        acc_0[1U][2U] = fma(_S153, _S158, acc_0[1U][2U]);
        acc_0[1U][3U] = fma(_S153, _S159, acc_0[1U][3U]);
        acc_0[2U][0U] = fma(_S154, _S156, acc_0[2U][0U]);
        acc_0[2U][1U] = fma(_S154, _S157, acc_0[2U][1U]);
        acc_0[2U][2U] = fma(_S154, _S158, acc_0[2U][2U]);
        acc_0[2U][3U] = fma(_S154, _S159, acc_0[2U][3U]);
        acc_0[3U][0U] = fma(_S155, _S156, acc_0[3U][0U]);
        acc_0[3U][1U] = fma(_S155, _S157, acc_0[3U][1U]);
        acc_0[3U][2U] = fma(_S155, _S158, acc_0[3U][2U]);
        acc_0[3U][3U] = fma(_S155, _S159, acc_0[3U][3U]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k0_0 = k0_0 + 16U;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                uint _S160 = _S8 + ty_0;
                _S2 = _S160;
                if(_S160 >= ((&kernelContext_0)->p_0->m_0))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S161 = _S9 + tx_0;
                            _S3 = _S161;
                            if(_S161 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_0 = _S160 * (&kernelContext_0)->p_0->ldc_0 + _S161;
                            float device* _S162 = (&kernelContext_0)->c_0+idx_0;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[0U][0U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[0U][0U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_0);
                            }
                            *_S162 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_0 = _S3 + 16U;
                            if(col_0 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_1 = _S160 * (&kernelContext_0)->p_0->ldc_0 + col_0;
                            float device* _S163 = (&kernelContext_0)->c_0+idx_1;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[0U][1U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[0U][1U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_1);
                            }
                            *_S163 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_1 = _S3 + 32U;
                            if(col_1 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_2 = _S160 * (&kernelContext_0)->p_0->ldc_0 + col_1;
                            float device* _S164 = (&kernelContext_0)->c_0+idx_2;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[0U][2U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[0U][2U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_2);
                            }
                            *_S164 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_2 = _S3 + 48U;
                            if(col_2 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_3 = _S160 * (&kernelContext_0)->p_0->ldc_0 + col_2;
                            float device* _S165 = (&kernelContext_0)->c_0+idx_3;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[0U][3U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[0U][3U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_3);
                            }
                            *_S165 = _S1;
                            break;
                        }
                        break;
                    }
                    break;
                }
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint r_0 = _S2 + 16U;
                if(r_0 >= ((&kernelContext_0)->p_0->m_0))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S166 = _S9 + tx_0;
                            _S4 = _S166;
                            if(_S166 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_4 = r_0 * (&kernelContext_0)->p_0->ldc_0 + _S166;
                            float device* _S167 = (&kernelContext_0)->c_0+idx_4;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[1U][0U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[1U][0U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_4);
                            }
                            *_S167 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_3 = _S4 + 16U;
                            if(col_3 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_5 = r_0 * (&kernelContext_0)->p_0->ldc_0 + col_3;
                            float device* _S168 = (&kernelContext_0)->c_0+idx_5;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[1U][1U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[1U][1U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_5);
                            }
                            *_S168 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_4 = _S4 + 32U;
                            if(col_4 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_6 = r_0 * (&kernelContext_0)->p_0->ldc_0 + col_4;
                            float device* _S169 = (&kernelContext_0)->c_0+idx_6;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[1U][2U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[1U][2U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_6);
                            }
                            *_S169 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_5 = _S4 + 48U;
                            if(col_5 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_7 = r_0 * (&kernelContext_0)->p_0->ldc_0 + col_5;
                            float device* _S170 = (&kernelContext_0)->c_0+idx_7;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[1U][3U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[1U][3U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_7);
                            }
                            *_S170 = _S1;
                            break;
                        }
                        break;
                    }
                    break;
                }
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint r_1 = _S2 + 32U;
                if(r_1 >= ((&kernelContext_0)->p_0->m_0))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S171 = _S9 + tx_0;
                            _S5 = _S171;
                            if(_S171 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_8 = r_1 * (&kernelContext_0)->p_0->ldc_0 + _S171;
                            float device* _S172 = (&kernelContext_0)->c_0+idx_8;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[2U][0U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[2U][0U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_8);
                            }
                            *_S172 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_6 = _S5 + 16U;
                            if(col_6 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_9 = r_1 * (&kernelContext_0)->p_0->ldc_0 + col_6;
                            float device* _S173 = (&kernelContext_0)->c_0+idx_9;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[2U][1U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[2U][1U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_9);
                            }
                            *_S173 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_7 = _S5 + 32U;
                            if(col_7 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_10 = r_1 * (&kernelContext_0)->p_0->ldc_0 + col_7;
                            float device* _S174 = (&kernelContext_0)->c_0+idx_10;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[2U][2U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[2U][2U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_10);
                            }
                            *_S174 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_8 = _S5 + 48U;
                            if(col_8 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_11 = r_1 * (&kernelContext_0)->p_0->ldc_0 + col_8;
                            float device* _S175 = (&kernelContext_0)->c_0+idx_11;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[2U][3U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[2U][3U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_11);
                            }
                            *_S175 = _S1;
                            break;
                        }
                        break;
                    }
                    break;
                }
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint r_2 = _S2 + 48U;
                if(r_2 >= ((&kernelContext_0)->p_0->m_0))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S176 = _S9 + tx_0;
                            _S6 = _S176;
                            if(_S176 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_12 = r_2 * (&kernelContext_0)->p_0->ldc_0 + _S176;
                            float device* _S177 = (&kernelContext_0)->c_0+idx_12;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[3U][0U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[3U][0U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_12);
                            }
                            *_S177 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_9 = _S6 + 16U;
                            if(col_9 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_13 = r_2 * (&kernelContext_0)->p_0->ldc_0 + col_9;
                            float device* _S178 = (&kernelContext_0)->c_0+idx_13;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[3U][1U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[3U][1U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_13);
                            }
                            *_S178 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_10 = _S6 + 32U;
                            if(col_10 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_14 = r_2 * (&kernelContext_0)->p_0->ldc_0 + col_10;
                            float device* _S179 = (&kernelContext_0)->c_0+idx_14;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[3U][2U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[3U][2U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_14);
                            }
                            *_S179 = _S1;
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_11 = _S6 + 48U;
                            if(col_11 >= ((&kernelContext_0)->p_0->n_0))
                            {
                                break;
                            }
                            uint idx_15 = r_2 * (&kernelContext_0)->p_0->ldc_0 + col_11;
                            float device* _S180 = (&kernelContext_0)->c_0+idx_15;
                            if(((&kernelContext_0)->p_0->beta_0) == 0.0)
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[3U][3U];
                            }
                            else
                            {
                                _S1 = (&kernelContext_0)->p_0->alpha_0 * acc_0[3U][3U] + (&kernelContext_0)->p_0->beta_0 * *((&kernelContext_0)->c_0+idx_15);
                            }
                            *_S180 = _S1;
                            break;
                        }
                        break;
                    }
                    break;
                }
                break;
            }
            break;
        }
        break;
    }
    return;
}

