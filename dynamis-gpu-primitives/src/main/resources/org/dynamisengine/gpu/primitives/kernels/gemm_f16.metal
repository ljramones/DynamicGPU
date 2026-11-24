#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
bool dg_is_nan_0(float x_0)
{
    uint _S1 = (as_type<uint>((x_0)));
    bool _S2;
    if((_S1 & 2139095040U) == 2139095040U)
    {
        _S2 = (_S1 & 8388607U) != 0U;
    }
    else
    {
        _S2 = false;
    }
    return _S2;
}

float dg_max_0(float a_0, float b_0)
{
    if(dg_is_nan_0(a_0))
    {
        return a_0;
    }
    if(dg_is_nan_0(b_0))
    {
        return b_0;
    }
    bool _S3;
    if(a_0 == 0.0)
    {
        _S3 = b_0 == 0.0;
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        return (as_type<float>(((as_type<uint>((a_0))) & (as_type<uint>((b_0))))));
    }
    float _S4;
    if(a_0 > b_0)
    {
        _S4 = a_0;
    }
    else
    {
        _S4 = b_0;
    }
    return _S4;
}

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
    uint transB_0;
    uint strideA_0;
    uint strideB_0;
    uint strideC_0;
    uint transA_0;
    uint bias_0;
    uint act_0;
    uint gated_0;
    uint residual_0;
};

struct KernelContext_0
{
    GemmParams_0 constant* p_0;
    half device* a_1;
    half device* b_1;
    float device* c_0;
    float device* epBias_0;
    float device* epGate_0;
    float device* epResidual_0;
    array<array<float, int(64)>, int(16)> threadgroup* tileA_0;
    array<array<float, int(64)>, int(16)> threadgroup* tileB_0;
};

float gemmEpilogue_0(float r_0, uint i_0, uint j_0, uint idx_0, KernelContext_0 thread* kernelContext_0)
{
    float _S5;
    if((kernelContext_0->p_0->bias_0) == 1U)
    {
        _S5 = r_0 + *(kernelContext_0->epBias_0+j_0);
    }
    else
    {
        if((kernelContext_0->p_0->bias_0) == 2U)
        {
            _S5 = r_0 + *(kernelContext_0->epBias_0+i_0);
        }
        else
        {
            _S5 = r_0;
        }
    }
    if((kernelContext_0->p_0->act_0) == 1U)
    {
        _S5 = dg_max_0(_S5, 0.0);
    }
    else
    {
        if((kernelContext_0->p_0->act_0) == 2U)
        {
            _S5 = 0.5 * _S5 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S5 + 0.04471499845385551 * _S5 * _S5 * _S5))) + 1.0)));
        }
        else
        {
            if((kernelContext_0->p_0->act_0) == 3U)
            {
                _S5 = _S5 / (1.0 + exp(- _S5));
            }
        }
    }
    if((kernelContext_0->p_0->gated_0) != 0U)
    {
        _S5 = _S5 * *(kernelContext_0->epGate_0+idx_0);
    }
    if((kernelContext_0->p_0->residual_0) != 0U)
    {
        _S5 = _S5 + *(kernelContext_0->epResidual_0+idx_0);
    }
    return _S5;
}

void gemmStore_0(uint i_1, uint j_1, uint idx_1, float acc_0, KernelContext_0 thread* kernelContext_1)
{
    float r_1;
    if((kernelContext_1->p_0->beta_0) == 0.0)
    {
        r_1 = kernelContext_1->p_0->alpha_0 * acc_0;
    }
    else
    {
        r_1 = kernelContext_1->p_0->alpha_0 * acc_0 + kernelContext_1->p_0->beta_0 * *(kernelContext_1->c_0+idx_1);
    }
    float device* _S6 = kernelContext_1->c_0+idx_1;
    float _S7 = gemmEpilogue_0(r_1, i_1, j_1, idx_1, kernelContext_1);
    *_S6 = _S7;
    return;
}

[[kernel]] void gemm_f16(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], half device* a_2 [[buffer(0)]], half device* b_2 [[buffer(1)]], float device* c_1 [[buffer(2)]], float device* epBias_1 [[buffer(3)]], float device* epGate_1 [[buffer(4)]], float device* epResidual_1 [[buffer(5)]])
{
    uint _S8;
    uint _S9;
    uint _S10;
    uint _S11;
    uint _S12;
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->p_0 = p_1;
    (&kernelContext_2)->a_1 = a_2;
    (&kernelContext_2)->b_1 = b_2;
    (&kernelContext_2)->c_0 = c_1;
    (&kernelContext_2)->epBias_0 = epBias_1;
    (&kernelContext_2)->epGate_0 = epGate_1;
    (&kernelContext_2)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(64)>, int(16)> tileA_1;
    (&kernelContext_2)->tileA_0 = &tileA_1;
    threadgroup array<array<float, int(64)>, int(16)> tileB_1;
    (&kernelContext_2)->tileB_0 = &tileB_1;
    uint tx_0 = local_0.x;
    uint ty_0 = local_0.y;
    uint _S13 = ty_0 * 16U + tx_0;
    uint _S14 = group_0.y * 64U;
    uint _S15 = group_0.x * 64U;
    uint _S16 = group_0.z;
    uint _S17 = _S16 * p_1->strideA_0;
    uint _S18 = _S16 * p_1->strideB_0;
    uint _S19 = _S16 * p_1->strideC_0;
    thread array<array<float, int(4)>, int(4)> acc_1;
    acc_1[0U][0U] = 0.0;
    acc_1[0U][1U] = 0.0;
    acc_1[0U][2U] = 0.0;
    acc_1[0U][3U] = 0.0;
    acc_1[1U][0U] = 0.0;
    acc_1[1U][1U] = 0.0;
    acc_1[1U][2U] = 0.0;
    acc_1[1U][3U] = 0.0;
    acc_1[2U][0U] = 0.0;
    acc_1[2U][1U] = 0.0;
    acc_1[2U][2U] = 0.0;
    acc_1[2U][3U] = 0.0;
    acc_1[3U][0U] = 0.0;
    acc_1[3U][1U] = 0.0;
    acc_1[3U][2U] = 0.0;
    acc_1[3U][3U] = 0.0;
    uint k0_0 = 0U;
    for(;;)
    {
        if(k0_0 < ((&kernelContext_2)->p_0->k_0))
        {
        }
        else
        {
            break;
        }
        uint _S20 = _S13 * 4U;
        uint ar_0 = _S20 / 16U;
        uint ak_0 = _S20 % 16U;
        uint gr_0 = _S14 + ar_0;
        uint gk_0 = k0_0 + ak_0;
        uint ai_0;
        if(((&kernelContext_2)->p_0->transA_0) != 0U)
        {
            ai_0 = gk_0 * (&kernelContext_2)->p_0->lda_0 + gr_0;
        }
        else
        {
            ai_0 = gr_0 * (&kernelContext_2)->p_0->lda_0 + gk_0;
        }
        bool _S21;
        if(gr_0 < ((&kernelContext_2)->p_0->m_0))
        {
            _S21 = gk_0 < ((&kernelContext_2)->p_0->k_0);
        }
        else
        {
            _S21 = false;
        }
        float _S22;
        if(_S21)
        {
            _S22 = float(*((&kernelContext_2)->a_1+(_S17 + ai_0)));
        }
        else
        {
            _S22 = 0.0;
        }
        (*(&kernelContext_2)->tileA_0)[ak_0][ar_0] = _S22;
        uint br_0 = _S20 / 64U;
        uint bc_0 = _S20 % 64U;
        uint hk_0 = k0_0 + br_0;
        uint gc_0 = _S15 + bc_0;
        uint bi_0;
        if(((&kernelContext_2)->p_0->transB_0) != 0U)
        {
            bi_0 = gc_0 * (&kernelContext_2)->p_0->ldb_0 + hk_0;
        }
        else
        {
            bi_0 = hk_0 * (&kernelContext_2)->p_0->ldb_0 + gc_0;
        }
        bool _S23;
        if(hk_0 < ((&kernelContext_2)->p_0->k_0))
        {
            _S23 = gc_0 < ((&kernelContext_2)->p_0->n_0);
        }
        else
        {
            _S23 = false;
        }
        float _S24;
        if(_S23)
        {
            _S24 = float(*((&kernelContext_2)->b_1+(_S18 + bi_0)));
        }
        else
        {
            _S24 = 0.0;
        }
        (*(&kernelContext_2)->tileB_0)[br_0][bc_0] = _S24;
        uint e_0 = _S20 + 1U;
        uint ar_1 = e_0 / 16U;
        uint ak_1 = e_0 % 16U;
        uint gr_1 = _S14 + ar_1;
        uint gk_1 = k0_0 + ak_1;
        uint ai_1;
        if(((&kernelContext_2)->p_0->transA_0) != 0U)
        {
            ai_1 = gk_1 * (&kernelContext_2)->p_0->lda_0 + gr_1;
        }
        else
        {
            ai_1 = gr_1 * (&kernelContext_2)->p_0->lda_0 + gk_1;
        }
        bool _S25;
        if(gr_1 < ((&kernelContext_2)->p_0->m_0))
        {
            _S25 = gk_1 < ((&kernelContext_2)->p_0->k_0);
        }
        else
        {
            _S25 = false;
        }
        float _S26;
        if(_S25)
        {
            _S26 = float(*((&kernelContext_2)->a_1+(_S17 + ai_1)));
        }
        else
        {
            _S26 = 0.0;
        }
        (*(&kernelContext_2)->tileA_0)[ak_1][ar_1] = _S26;
        uint br_1 = e_0 / 64U;
        uint bc_1 = e_0 % 64U;
        uint hk_1 = k0_0 + br_1;
        uint gc_1 = _S15 + bc_1;
        uint bi_1;
        if(((&kernelContext_2)->p_0->transB_0) != 0U)
        {
            bi_1 = gc_1 * (&kernelContext_2)->p_0->ldb_0 + hk_1;
        }
        else
        {
            bi_1 = hk_1 * (&kernelContext_2)->p_0->ldb_0 + gc_1;
        }
        bool _S27;
        if(hk_1 < ((&kernelContext_2)->p_0->k_0))
        {
            _S27 = gc_1 < ((&kernelContext_2)->p_0->n_0);
        }
        else
        {
            _S27 = false;
        }
        float _S28;
        if(_S27)
        {
            _S28 = float(*((&kernelContext_2)->b_1+(_S18 + bi_1)));
        }
        else
        {
            _S28 = 0.0;
        }
        (*(&kernelContext_2)->tileB_0)[br_1][bc_1] = _S28;
        uint e_1 = _S20 + 2U;
        uint ar_2 = e_1 / 16U;
        uint ak_2 = e_1 % 16U;
        uint gr_2 = _S14 + ar_2;
        uint gk_2 = k0_0 + ak_2;
        uint ai_2;
        if(((&kernelContext_2)->p_0->transA_0) != 0U)
        {
            ai_2 = gk_2 * (&kernelContext_2)->p_0->lda_0 + gr_2;
        }
        else
        {
            ai_2 = gr_2 * (&kernelContext_2)->p_0->lda_0 + gk_2;
        }
        bool _S29;
        if(gr_2 < ((&kernelContext_2)->p_0->m_0))
        {
            _S29 = gk_2 < ((&kernelContext_2)->p_0->k_0);
        }
        else
        {
            _S29 = false;
        }
        float _S30;
        if(_S29)
        {
            _S30 = float(*((&kernelContext_2)->a_1+(_S17 + ai_2)));
        }
        else
        {
            _S30 = 0.0;
        }
        (*(&kernelContext_2)->tileA_0)[ak_2][ar_2] = _S30;
        uint br_2 = e_1 / 64U;
        uint bc_2 = e_1 % 64U;
        uint hk_2 = k0_0 + br_2;
        uint gc_2 = _S15 + bc_2;
        uint bi_2;
        if(((&kernelContext_2)->p_0->transB_0) != 0U)
        {
            bi_2 = gc_2 * (&kernelContext_2)->p_0->ldb_0 + hk_2;
        }
        else
        {
            bi_2 = hk_2 * (&kernelContext_2)->p_0->ldb_0 + gc_2;
        }
        bool _S31;
        if(hk_2 < ((&kernelContext_2)->p_0->k_0))
        {
            _S31 = gc_2 < ((&kernelContext_2)->p_0->n_0);
        }
        else
        {
            _S31 = false;
        }
        float _S32;
        if(_S31)
        {
            _S32 = float(*((&kernelContext_2)->b_1+(_S18 + bi_2)));
        }
        else
        {
            _S32 = 0.0;
        }
        (*(&kernelContext_2)->tileB_0)[br_2][bc_2] = _S32;
        uint e_2 = _S20 + 3U;
        uint ar_3 = e_2 / 16U;
        uint ak_3 = e_2 % 16U;
        uint gr_3 = _S14 + ar_3;
        uint gk_3 = k0_0 + ak_3;
        uint ai_3;
        if(((&kernelContext_2)->p_0->transA_0) != 0U)
        {
            ai_3 = gk_3 * (&kernelContext_2)->p_0->lda_0 + gr_3;
        }
        else
        {
            ai_3 = gr_3 * (&kernelContext_2)->p_0->lda_0 + gk_3;
        }
        bool _S33;
        if(gr_3 < ((&kernelContext_2)->p_0->m_0))
        {
            _S33 = gk_3 < ((&kernelContext_2)->p_0->k_0);
        }
        else
        {
            _S33 = false;
        }
        float _S34;
        if(_S33)
        {
            _S34 = float(*((&kernelContext_2)->a_1+(_S17 + ai_3)));
        }
        else
        {
            _S34 = 0.0;
        }
        (*(&kernelContext_2)->tileA_0)[ak_3][ar_3] = _S34;
        uint br_3 = e_2 / 64U;
        uint bc_3 = e_2 % 64U;
        uint hk_3 = k0_0 + br_3;
        uint gc_3 = _S15 + bc_3;
        uint bi_3;
        if(((&kernelContext_2)->p_0->transB_0) != 0U)
        {
            bi_3 = gc_3 * (&kernelContext_2)->p_0->ldb_0 + hk_3;
        }
        else
        {
            bi_3 = hk_3 * (&kernelContext_2)->p_0->ldb_0 + gc_3;
        }
        bool _S35;
        if(hk_3 < ((&kernelContext_2)->p_0->k_0))
        {
            _S35 = gc_3 < ((&kernelContext_2)->p_0->n_0);
        }
        else
        {
            _S35 = false;
        }
        float _S36;
        if(_S35)
        {
            _S36 = float(*((&kernelContext_2)->b_1+(_S18 + bi_3)));
        }
        else
        {
            _S36 = 0.0;
        }
        (*(&kernelContext_2)->tileB_0)[br_3][bc_3] = _S36;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        thread array<float, int(4)> av_0;
        float _S37 = (*(&kernelContext_2)->tileA_0)[0U][ty_0];
        av_0[0U] = (*(&kernelContext_2)->tileA_0)[0U][ty_0];
        uint _S38 = ty_0 + 16U;
        float _S39 = (*(&kernelContext_2)->tileA_0)[0U][_S38];
        av_0[1U] = (*(&kernelContext_2)->tileA_0)[0U][_S38];
        uint _S40 = ty_0 + 32U;
        float _S41 = (*(&kernelContext_2)->tileA_0)[0U][_S40];
        av_0[2U] = (*(&kernelContext_2)->tileA_0)[0U][_S40];
        uint _S42 = ty_0 + 48U;
        float _S43 = (*(&kernelContext_2)->tileA_0)[0U][_S42];
        av_0[3U] = (*(&kernelContext_2)->tileA_0)[0U][_S42];
        thread array<float, int(4)> bv_0;
        float _S44 = (*(&kernelContext_2)->tileB_0)[0U][tx_0];
        bv_0[0U] = (*(&kernelContext_2)->tileB_0)[0U][tx_0];
        uint _S45 = tx_0 + 16U;
        float _S46 = (*(&kernelContext_2)->tileB_0)[0U][_S45];
        bv_0[1U] = (*(&kernelContext_2)->tileB_0)[0U][_S45];
        uint _S47 = tx_0 + 32U;
        float _S48 = (*(&kernelContext_2)->tileB_0)[0U][_S47];
        bv_0[2U] = (*(&kernelContext_2)->tileB_0)[0U][_S47];
        uint _S49 = tx_0 + 48U;
        float _S50 = (*(&kernelContext_2)->tileB_0)[0U][_S49];
        bv_0[3U] = (*(&kernelContext_2)->tileB_0)[0U][_S49];
        acc_1[0U][0U] = fma(_S37, _S44, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S37, _S46, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S37, _S48, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S37, _S50, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S39, _S44, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S39, _S46, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S39, _S48, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S39, _S50, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S41, _S44, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S41, _S46, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S41, _S48, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S41, _S50, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S43, _S44, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S43, _S46, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S43, _S48, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S43, _S50, acc_1[3U][3U]);
        thread array<float, int(4)> av_1;
        float _S51 = (*(&kernelContext_2)->tileA_0)[1U][ty_0];
        av_1[0U] = (*(&kernelContext_2)->tileA_0)[1U][ty_0];
        float _S52 = (*(&kernelContext_2)->tileA_0)[1U][_S38];
        av_1[1U] = (*(&kernelContext_2)->tileA_0)[1U][_S38];
        float _S53 = (*(&kernelContext_2)->tileA_0)[1U][_S40];
        av_1[2U] = (*(&kernelContext_2)->tileA_0)[1U][_S40];
        float _S54 = (*(&kernelContext_2)->tileA_0)[1U][_S42];
        av_1[3U] = (*(&kernelContext_2)->tileA_0)[1U][_S42];
        thread array<float, int(4)> bv_1;
        float _S55 = (*(&kernelContext_2)->tileB_0)[1U][tx_0];
        bv_1[0U] = (*(&kernelContext_2)->tileB_0)[1U][tx_0];
        float _S56 = (*(&kernelContext_2)->tileB_0)[1U][_S45];
        bv_1[1U] = (*(&kernelContext_2)->tileB_0)[1U][_S45];
        float _S57 = (*(&kernelContext_2)->tileB_0)[1U][_S47];
        bv_1[2U] = (*(&kernelContext_2)->tileB_0)[1U][_S47];
        float _S58 = (*(&kernelContext_2)->tileB_0)[1U][_S49];
        bv_1[3U] = (*(&kernelContext_2)->tileB_0)[1U][_S49];
        acc_1[0U][0U] = fma(_S51, _S55, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S51, _S56, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S51, _S57, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S51, _S58, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S52, _S55, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S52, _S56, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S52, _S57, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S52, _S58, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S53, _S55, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S53, _S56, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S53, _S57, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S53, _S58, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S54, _S55, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S54, _S56, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S54, _S57, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S54, _S58, acc_1[3U][3U]);
        thread array<float, int(4)> av_2;
        float _S59 = (*(&kernelContext_2)->tileA_0)[2U][ty_0];
        av_2[0U] = (*(&kernelContext_2)->tileA_0)[2U][ty_0];
        float _S60 = (*(&kernelContext_2)->tileA_0)[2U][_S38];
        av_2[1U] = (*(&kernelContext_2)->tileA_0)[2U][_S38];
        float _S61 = (*(&kernelContext_2)->tileA_0)[2U][_S40];
        av_2[2U] = (*(&kernelContext_2)->tileA_0)[2U][_S40];
        float _S62 = (*(&kernelContext_2)->tileA_0)[2U][_S42];
        av_2[3U] = (*(&kernelContext_2)->tileA_0)[2U][_S42];
        thread array<float, int(4)> bv_2;
        float _S63 = (*(&kernelContext_2)->tileB_0)[2U][tx_0];
        bv_2[0U] = (*(&kernelContext_2)->tileB_0)[2U][tx_0];
        float _S64 = (*(&kernelContext_2)->tileB_0)[2U][_S45];
        bv_2[1U] = (*(&kernelContext_2)->tileB_0)[2U][_S45];
        float _S65 = (*(&kernelContext_2)->tileB_0)[2U][_S47];
        bv_2[2U] = (*(&kernelContext_2)->tileB_0)[2U][_S47];
        float _S66 = (*(&kernelContext_2)->tileB_0)[2U][_S49];
        bv_2[3U] = (*(&kernelContext_2)->tileB_0)[2U][_S49];
        acc_1[0U][0U] = fma(_S59, _S63, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S59, _S64, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S59, _S65, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S59, _S66, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S60, _S63, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S60, _S64, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S60, _S65, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S60, _S66, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S61, _S63, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S61, _S64, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S61, _S65, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S61, _S66, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S62, _S63, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S62, _S64, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S62, _S65, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S62, _S66, acc_1[3U][3U]);
        thread array<float, int(4)> av_3;
        float _S67 = (*(&kernelContext_2)->tileA_0)[3U][ty_0];
        av_3[0U] = (*(&kernelContext_2)->tileA_0)[3U][ty_0];
        float _S68 = (*(&kernelContext_2)->tileA_0)[3U][_S38];
        av_3[1U] = (*(&kernelContext_2)->tileA_0)[3U][_S38];
        float _S69 = (*(&kernelContext_2)->tileA_0)[3U][_S40];
        av_3[2U] = (*(&kernelContext_2)->tileA_0)[3U][_S40];
        float _S70 = (*(&kernelContext_2)->tileA_0)[3U][_S42];
        av_3[3U] = (*(&kernelContext_2)->tileA_0)[3U][_S42];
        thread array<float, int(4)> bv_3;
        float _S71 = (*(&kernelContext_2)->tileB_0)[3U][tx_0];
        bv_3[0U] = (*(&kernelContext_2)->tileB_0)[3U][tx_0];
        float _S72 = (*(&kernelContext_2)->tileB_0)[3U][_S45];
        bv_3[1U] = (*(&kernelContext_2)->tileB_0)[3U][_S45];
        float _S73 = (*(&kernelContext_2)->tileB_0)[3U][_S47];
        bv_3[2U] = (*(&kernelContext_2)->tileB_0)[3U][_S47];
        float _S74 = (*(&kernelContext_2)->tileB_0)[3U][_S49];
        bv_3[3U] = (*(&kernelContext_2)->tileB_0)[3U][_S49];
        acc_1[0U][0U] = fma(_S67, _S71, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S67, _S72, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S67, _S73, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S67, _S74, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S68, _S71, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S68, _S72, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S68, _S73, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S68, _S74, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S69, _S71, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S69, _S72, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S69, _S73, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S69, _S74, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S70, _S71, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S70, _S72, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S70, _S73, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S70, _S74, acc_1[3U][3U]);
        thread array<float, int(4)> av_4;
        float _S75 = (*(&kernelContext_2)->tileA_0)[4U][ty_0];
        av_4[0U] = (*(&kernelContext_2)->tileA_0)[4U][ty_0];
        float _S76 = (*(&kernelContext_2)->tileA_0)[4U][_S38];
        av_4[1U] = (*(&kernelContext_2)->tileA_0)[4U][_S38];
        float _S77 = (*(&kernelContext_2)->tileA_0)[4U][_S40];
        av_4[2U] = (*(&kernelContext_2)->tileA_0)[4U][_S40];
        float _S78 = (*(&kernelContext_2)->tileA_0)[4U][_S42];
        av_4[3U] = (*(&kernelContext_2)->tileA_0)[4U][_S42];
        thread array<float, int(4)> bv_4;
        float _S79 = (*(&kernelContext_2)->tileB_0)[4U][tx_0];
        bv_4[0U] = (*(&kernelContext_2)->tileB_0)[4U][tx_0];
        float _S80 = (*(&kernelContext_2)->tileB_0)[4U][_S45];
        bv_4[1U] = (*(&kernelContext_2)->tileB_0)[4U][_S45];
        float _S81 = (*(&kernelContext_2)->tileB_0)[4U][_S47];
        bv_4[2U] = (*(&kernelContext_2)->tileB_0)[4U][_S47];
        float _S82 = (*(&kernelContext_2)->tileB_0)[4U][_S49];
        bv_4[3U] = (*(&kernelContext_2)->tileB_0)[4U][_S49];
        acc_1[0U][0U] = fma(_S75, _S79, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S75, _S80, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S75, _S81, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S75, _S82, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S76, _S79, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S76, _S80, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S76, _S81, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S76, _S82, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S77, _S79, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S77, _S80, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S77, _S81, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S77, _S82, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S78, _S79, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S78, _S80, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S78, _S81, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S78, _S82, acc_1[3U][3U]);
        thread array<float, int(4)> av_5;
        float _S83 = (*(&kernelContext_2)->tileA_0)[5U][ty_0];
        av_5[0U] = (*(&kernelContext_2)->tileA_0)[5U][ty_0];
        float _S84 = (*(&kernelContext_2)->tileA_0)[5U][_S38];
        av_5[1U] = (*(&kernelContext_2)->tileA_0)[5U][_S38];
        float _S85 = (*(&kernelContext_2)->tileA_0)[5U][_S40];
        av_5[2U] = (*(&kernelContext_2)->tileA_0)[5U][_S40];
        float _S86 = (*(&kernelContext_2)->tileA_0)[5U][_S42];
        av_5[3U] = (*(&kernelContext_2)->tileA_0)[5U][_S42];
        thread array<float, int(4)> bv_5;
        float _S87 = (*(&kernelContext_2)->tileB_0)[5U][tx_0];
        bv_5[0U] = (*(&kernelContext_2)->tileB_0)[5U][tx_0];
        float _S88 = (*(&kernelContext_2)->tileB_0)[5U][_S45];
        bv_5[1U] = (*(&kernelContext_2)->tileB_0)[5U][_S45];
        float _S89 = (*(&kernelContext_2)->tileB_0)[5U][_S47];
        bv_5[2U] = (*(&kernelContext_2)->tileB_0)[5U][_S47];
        float _S90 = (*(&kernelContext_2)->tileB_0)[5U][_S49];
        bv_5[3U] = (*(&kernelContext_2)->tileB_0)[5U][_S49];
        acc_1[0U][0U] = fma(_S83, _S87, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S83, _S88, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S83, _S89, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S83, _S90, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S84, _S87, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S84, _S88, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S84, _S89, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S84, _S90, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S85, _S87, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S85, _S88, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S85, _S89, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S85, _S90, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S86, _S87, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S86, _S88, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S86, _S89, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S86, _S90, acc_1[3U][3U]);
        thread array<float, int(4)> av_6;
        float _S91 = (*(&kernelContext_2)->tileA_0)[6U][ty_0];
        av_6[0U] = (*(&kernelContext_2)->tileA_0)[6U][ty_0];
        float _S92 = (*(&kernelContext_2)->tileA_0)[6U][_S38];
        av_6[1U] = (*(&kernelContext_2)->tileA_0)[6U][_S38];
        float _S93 = (*(&kernelContext_2)->tileA_0)[6U][_S40];
        av_6[2U] = (*(&kernelContext_2)->tileA_0)[6U][_S40];
        float _S94 = (*(&kernelContext_2)->tileA_0)[6U][_S42];
        av_6[3U] = (*(&kernelContext_2)->tileA_0)[6U][_S42];
        thread array<float, int(4)> bv_6;
        float _S95 = (*(&kernelContext_2)->tileB_0)[6U][tx_0];
        bv_6[0U] = (*(&kernelContext_2)->tileB_0)[6U][tx_0];
        float _S96 = (*(&kernelContext_2)->tileB_0)[6U][_S45];
        bv_6[1U] = (*(&kernelContext_2)->tileB_0)[6U][_S45];
        float _S97 = (*(&kernelContext_2)->tileB_0)[6U][_S47];
        bv_6[2U] = (*(&kernelContext_2)->tileB_0)[6U][_S47];
        float _S98 = (*(&kernelContext_2)->tileB_0)[6U][_S49];
        bv_6[3U] = (*(&kernelContext_2)->tileB_0)[6U][_S49];
        acc_1[0U][0U] = fma(_S91, _S95, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S91, _S96, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S91, _S97, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S91, _S98, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S92, _S95, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S92, _S96, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S92, _S97, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S92, _S98, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S93, _S95, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S93, _S96, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S93, _S97, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S93, _S98, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S94, _S95, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S94, _S96, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S94, _S97, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S94, _S98, acc_1[3U][3U]);
        thread array<float, int(4)> av_7;
        float _S99 = (*(&kernelContext_2)->tileA_0)[7U][ty_0];
        av_7[0U] = (*(&kernelContext_2)->tileA_0)[7U][ty_0];
        float _S100 = (*(&kernelContext_2)->tileA_0)[7U][_S38];
        av_7[1U] = (*(&kernelContext_2)->tileA_0)[7U][_S38];
        float _S101 = (*(&kernelContext_2)->tileA_0)[7U][_S40];
        av_7[2U] = (*(&kernelContext_2)->tileA_0)[7U][_S40];
        float _S102 = (*(&kernelContext_2)->tileA_0)[7U][_S42];
        av_7[3U] = (*(&kernelContext_2)->tileA_0)[7U][_S42];
        thread array<float, int(4)> bv_7;
        float _S103 = (*(&kernelContext_2)->tileB_0)[7U][tx_0];
        bv_7[0U] = (*(&kernelContext_2)->tileB_0)[7U][tx_0];
        float _S104 = (*(&kernelContext_2)->tileB_0)[7U][_S45];
        bv_7[1U] = (*(&kernelContext_2)->tileB_0)[7U][_S45];
        float _S105 = (*(&kernelContext_2)->tileB_0)[7U][_S47];
        bv_7[2U] = (*(&kernelContext_2)->tileB_0)[7U][_S47];
        float _S106 = (*(&kernelContext_2)->tileB_0)[7U][_S49];
        bv_7[3U] = (*(&kernelContext_2)->tileB_0)[7U][_S49];
        acc_1[0U][0U] = fma(_S99, _S103, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S99, _S104, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S99, _S105, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S99, _S106, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S100, _S103, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S100, _S104, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S100, _S105, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S100, _S106, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S101, _S103, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S101, _S104, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S101, _S105, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S101, _S106, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S102, _S103, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S102, _S104, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S102, _S105, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S102, _S106, acc_1[3U][3U]);
        thread array<float, int(4)> av_8;
        float _S107 = (*(&kernelContext_2)->tileA_0)[8U][ty_0];
        av_8[0U] = (*(&kernelContext_2)->tileA_0)[8U][ty_0];
        float _S108 = (*(&kernelContext_2)->tileA_0)[8U][_S38];
        av_8[1U] = (*(&kernelContext_2)->tileA_0)[8U][_S38];
        float _S109 = (*(&kernelContext_2)->tileA_0)[8U][_S40];
        av_8[2U] = (*(&kernelContext_2)->tileA_0)[8U][_S40];
        float _S110 = (*(&kernelContext_2)->tileA_0)[8U][_S42];
        av_8[3U] = (*(&kernelContext_2)->tileA_0)[8U][_S42];
        thread array<float, int(4)> bv_8;
        float _S111 = (*(&kernelContext_2)->tileB_0)[8U][tx_0];
        bv_8[0U] = (*(&kernelContext_2)->tileB_0)[8U][tx_0];
        float _S112 = (*(&kernelContext_2)->tileB_0)[8U][_S45];
        bv_8[1U] = (*(&kernelContext_2)->tileB_0)[8U][_S45];
        float _S113 = (*(&kernelContext_2)->tileB_0)[8U][_S47];
        bv_8[2U] = (*(&kernelContext_2)->tileB_0)[8U][_S47];
        float _S114 = (*(&kernelContext_2)->tileB_0)[8U][_S49];
        bv_8[3U] = (*(&kernelContext_2)->tileB_0)[8U][_S49];
        acc_1[0U][0U] = fma(_S107, _S111, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S107, _S112, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S107, _S113, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S107, _S114, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S108, _S111, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S108, _S112, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S108, _S113, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S108, _S114, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S109, _S111, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S109, _S112, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S109, _S113, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S109, _S114, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S110, _S111, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S110, _S112, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S110, _S113, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S110, _S114, acc_1[3U][3U]);
        thread array<float, int(4)> av_9;
        float _S115 = (*(&kernelContext_2)->tileA_0)[9U][ty_0];
        av_9[0U] = (*(&kernelContext_2)->tileA_0)[9U][ty_0];
        float _S116 = (*(&kernelContext_2)->tileA_0)[9U][_S38];
        av_9[1U] = (*(&kernelContext_2)->tileA_0)[9U][_S38];
        float _S117 = (*(&kernelContext_2)->tileA_0)[9U][_S40];
        av_9[2U] = (*(&kernelContext_2)->tileA_0)[9U][_S40];
        float _S118 = (*(&kernelContext_2)->tileA_0)[9U][_S42];
        av_9[3U] = (*(&kernelContext_2)->tileA_0)[9U][_S42];
        thread array<float, int(4)> bv_9;
        float _S119 = (*(&kernelContext_2)->tileB_0)[9U][tx_0];
        bv_9[0U] = (*(&kernelContext_2)->tileB_0)[9U][tx_0];
        float _S120 = (*(&kernelContext_2)->tileB_0)[9U][_S45];
        bv_9[1U] = (*(&kernelContext_2)->tileB_0)[9U][_S45];
        float _S121 = (*(&kernelContext_2)->tileB_0)[9U][_S47];
        bv_9[2U] = (*(&kernelContext_2)->tileB_0)[9U][_S47];
        float _S122 = (*(&kernelContext_2)->tileB_0)[9U][_S49];
        bv_9[3U] = (*(&kernelContext_2)->tileB_0)[9U][_S49];
        acc_1[0U][0U] = fma(_S115, _S119, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S115, _S120, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S115, _S121, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S115, _S122, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S116, _S119, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S116, _S120, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S116, _S121, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S116, _S122, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S117, _S119, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S117, _S120, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S117, _S121, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S117, _S122, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S118, _S119, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S118, _S120, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S118, _S121, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S118, _S122, acc_1[3U][3U]);
        thread array<float, int(4)> av_10;
        float _S123 = (*(&kernelContext_2)->tileA_0)[10U][ty_0];
        av_10[0U] = (*(&kernelContext_2)->tileA_0)[10U][ty_0];
        float _S124 = (*(&kernelContext_2)->tileA_0)[10U][_S38];
        av_10[1U] = (*(&kernelContext_2)->tileA_0)[10U][_S38];
        float _S125 = (*(&kernelContext_2)->tileA_0)[10U][_S40];
        av_10[2U] = (*(&kernelContext_2)->tileA_0)[10U][_S40];
        float _S126 = (*(&kernelContext_2)->tileA_0)[10U][_S42];
        av_10[3U] = (*(&kernelContext_2)->tileA_0)[10U][_S42];
        thread array<float, int(4)> bv_10;
        float _S127 = (*(&kernelContext_2)->tileB_0)[10U][tx_0];
        bv_10[0U] = (*(&kernelContext_2)->tileB_0)[10U][tx_0];
        float _S128 = (*(&kernelContext_2)->tileB_0)[10U][_S45];
        bv_10[1U] = (*(&kernelContext_2)->tileB_0)[10U][_S45];
        float _S129 = (*(&kernelContext_2)->tileB_0)[10U][_S47];
        bv_10[2U] = (*(&kernelContext_2)->tileB_0)[10U][_S47];
        float _S130 = (*(&kernelContext_2)->tileB_0)[10U][_S49];
        bv_10[3U] = (*(&kernelContext_2)->tileB_0)[10U][_S49];
        acc_1[0U][0U] = fma(_S123, _S127, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S123, _S128, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S123, _S129, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S123, _S130, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S124, _S127, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S124, _S128, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S124, _S129, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S124, _S130, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S125, _S127, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S125, _S128, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S125, _S129, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S125, _S130, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S126, _S127, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S126, _S128, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S126, _S129, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S126, _S130, acc_1[3U][3U]);
        thread array<float, int(4)> av_11;
        float _S131 = (*(&kernelContext_2)->tileA_0)[11U][ty_0];
        av_11[0U] = (*(&kernelContext_2)->tileA_0)[11U][ty_0];
        float _S132 = (*(&kernelContext_2)->tileA_0)[11U][_S38];
        av_11[1U] = (*(&kernelContext_2)->tileA_0)[11U][_S38];
        float _S133 = (*(&kernelContext_2)->tileA_0)[11U][_S40];
        av_11[2U] = (*(&kernelContext_2)->tileA_0)[11U][_S40];
        float _S134 = (*(&kernelContext_2)->tileA_0)[11U][_S42];
        av_11[3U] = (*(&kernelContext_2)->tileA_0)[11U][_S42];
        thread array<float, int(4)> bv_11;
        float _S135 = (*(&kernelContext_2)->tileB_0)[11U][tx_0];
        bv_11[0U] = (*(&kernelContext_2)->tileB_0)[11U][tx_0];
        float _S136 = (*(&kernelContext_2)->tileB_0)[11U][_S45];
        bv_11[1U] = (*(&kernelContext_2)->tileB_0)[11U][_S45];
        float _S137 = (*(&kernelContext_2)->tileB_0)[11U][_S47];
        bv_11[2U] = (*(&kernelContext_2)->tileB_0)[11U][_S47];
        float _S138 = (*(&kernelContext_2)->tileB_0)[11U][_S49];
        bv_11[3U] = (*(&kernelContext_2)->tileB_0)[11U][_S49];
        acc_1[0U][0U] = fma(_S131, _S135, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S131, _S136, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S131, _S137, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S131, _S138, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S132, _S135, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S132, _S136, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S132, _S137, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S132, _S138, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S133, _S135, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S133, _S136, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S133, _S137, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S133, _S138, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S134, _S135, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S134, _S136, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S134, _S137, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S134, _S138, acc_1[3U][3U]);
        thread array<float, int(4)> av_12;
        float _S139 = (*(&kernelContext_2)->tileA_0)[12U][ty_0];
        av_12[0U] = (*(&kernelContext_2)->tileA_0)[12U][ty_0];
        float _S140 = (*(&kernelContext_2)->tileA_0)[12U][_S38];
        av_12[1U] = (*(&kernelContext_2)->tileA_0)[12U][_S38];
        float _S141 = (*(&kernelContext_2)->tileA_0)[12U][_S40];
        av_12[2U] = (*(&kernelContext_2)->tileA_0)[12U][_S40];
        float _S142 = (*(&kernelContext_2)->tileA_0)[12U][_S42];
        av_12[3U] = (*(&kernelContext_2)->tileA_0)[12U][_S42];
        thread array<float, int(4)> bv_12;
        float _S143 = (*(&kernelContext_2)->tileB_0)[12U][tx_0];
        bv_12[0U] = (*(&kernelContext_2)->tileB_0)[12U][tx_0];
        float _S144 = (*(&kernelContext_2)->tileB_0)[12U][_S45];
        bv_12[1U] = (*(&kernelContext_2)->tileB_0)[12U][_S45];
        float _S145 = (*(&kernelContext_2)->tileB_0)[12U][_S47];
        bv_12[2U] = (*(&kernelContext_2)->tileB_0)[12U][_S47];
        float _S146 = (*(&kernelContext_2)->tileB_0)[12U][_S49];
        bv_12[3U] = (*(&kernelContext_2)->tileB_0)[12U][_S49];
        acc_1[0U][0U] = fma(_S139, _S143, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S139, _S144, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S139, _S145, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S139, _S146, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S140, _S143, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S140, _S144, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S140, _S145, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S140, _S146, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S141, _S143, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S141, _S144, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S141, _S145, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S141, _S146, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S142, _S143, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S142, _S144, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S142, _S145, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S142, _S146, acc_1[3U][3U]);
        thread array<float, int(4)> av_13;
        float _S147 = (*(&kernelContext_2)->tileA_0)[13U][ty_0];
        av_13[0U] = (*(&kernelContext_2)->tileA_0)[13U][ty_0];
        float _S148 = (*(&kernelContext_2)->tileA_0)[13U][_S38];
        av_13[1U] = (*(&kernelContext_2)->tileA_0)[13U][_S38];
        float _S149 = (*(&kernelContext_2)->tileA_0)[13U][_S40];
        av_13[2U] = (*(&kernelContext_2)->tileA_0)[13U][_S40];
        float _S150 = (*(&kernelContext_2)->tileA_0)[13U][_S42];
        av_13[3U] = (*(&kernelContext_2)->tileA_0)[13U][_S42];
        thread array<float, int(4)> bv_13;
        float _S151 = (*(&kernelContext_2)->tileB_0)[13U][tx_0];
        bv_13[0U] = (*(&kernelContext_2)->tileB_0)[13U][tx_0];
        float _S152 = (*(&kernelContext_2)->tileB_0)[13U][_S45];
        bv_13[1U] = (*(&kernelContext_2)->tileB_0)[13U][_S45];
        float _S153 = (*(&kernelContext_2)->tileB_0)[13U][_S47];
        bv_13[2U] = (*(&kernelContext_2)->tileB_0)[13U][_S47];
        float _S154 = (*(&kernelContext_2)->tileB_0)[13U][_S49];
        bv_13[3U] = (*(&kernelContext_2)->tileB_0)[13U][_S49];
        acc_1[0U][0U] = fma(_S147, _S151, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S147, _S152, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S147, _S153, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S147, _S154, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S148, _S151, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S148, _S152, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S148, _S153, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S148, _S154, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S149, _S151, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S149, _S152, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S149, _S153, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S149, _S154, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S150, _S151, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S150, _S152, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S150, _S153, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S150, _S154, acc_1[3U][3U]);
        thread array<float, int(4)> av_14;
        float _S155 = (*(&kernelContext_2)->tileA_0)[14U][ty_0];
        av_14[0U] = (*(&kernelContext_2)->tileA_0)[14U][ty_0];
        float _S156 = (*(&kernelContext_2)->tileA_0)[14U][_S38];
        av_14[1U] = (*(&kernelContext_2)->tileA_0)[14U][_S38];
        float _S157 = (*(&kernelContext_2)->tileA_0)[14U][_S40];
        av_14[2U] = (*(&kernelContext_2)->tileA_0)[14U][_S40];
        float _S158 = (*(&kernelContext_2)->tileA_0)[14U][_S42];
        av_14[3U] = (*(&kernelContext_2)->tileA_0)[14U][_S42];
        thread array<float, int(4)> bv_14;
        float _S159 = (*(&kernelContext_2)->tileB_0)[14U][tx_0];
        bv_14[0U] = (*(&kernelContext_2)->tileB_0)[14U][tx_0];
        float _S160 = (*(&kernelContext_2)->tileB_0)[14U][_S45];
        bv_14[1U] = (*(&kernelContext_2)->tileB_0)[14U][_S45];
        float _S161 = (*(&kernelContext_2)->tileB_0)[14U][_S47];
        bv_14[2U] = (*(&kernelContext_2)->tileB_0)[14U][_S47];
        float _S162 = (*(&kernelContext_2)->tileB_0)[14U][_S49];
        bv_14[3U] = (*(&kernelContext_2)->tileB_0)[14U][_S49];
        acc_1[0U][0U] = fma(_S155, _S159, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S155, _S160, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S155, _S161, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S155, _S162, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S156, _S159, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S156, _S160, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S156, _S161, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S156, _S162, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S157, _S159, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S157, _S160, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S157, _S161, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S157, _S162, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S158, _S159, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S158, _S160, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S158, _S161, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S158, _S162, acc_1[3U][3U]);
        thread array<float, int(4)> av_15;
        float _S163 = (*(&kernelContext_2)->tileA_0)[15U][ty_0];
        av_15[0U] = (*(&kernelContext_2)->tileA_0)[15U][ty_0];
        float _S164 = (*(&kernelContext_2)->tileA_0)[15U][_S38];
        av_15[1U] = (*(&kernelContext_2)->tileA_0)[15U][_S38];
        float _S165 = (*(&kernelContext_2)->tileA_0)[15U][_S40];
        av_15[2U] = (*(&kernelContext_2)->tileA_0)[15U][_S40];
        float _S166 = (*(&kernelContext_2)->tileA_0)[15U][_S42];
        av_15[3U] = (*(&kernelContext_2)->tileA_0)[15U][_S42];
        thread array<float, int(4)> bv_15;
        float _S167 = (*(&kernelContext_2)->tileB_0)[15U][tx_0];
        bv_15[0U] = (*(&kernelContext_2)->tileB_0)[15U][tx_0];
        float _S168 = (*(&kernelContext_2)->tileB_0)[15U][_S45];
        bv_15[1U] = (*(&kernelContext_2)->tileB_0)[15U][_S45];
        float _S169 = (*(&kernelContext_2)->tileB_0)[15U][_S47];
        bv_15[2U] = (*(&kernelContext_2)->tileB_0)[15U][_S47];
        float _S170 = (*(&kernelContext_2)->tileB_0)[15U][_S49];
        bv_15[3U] = (*(&kernelContext_2)->tileB_0)[15U][_S49];
        acc_1[0U][0U] = fma(_S163, _S167, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S163, _S168, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S163, _S169, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S163, _S170, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S164, _S167, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S164, _S168, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S164, _S169, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S164, _S170, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S165, _S167, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S165, _S168, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S165, _S169, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S165, _S170, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S166, _S167, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S166, _S168, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S166, _S169, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S166, _S170, acc_1[3U][3U]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k0_0 = k0_0 + 16U;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                uint _S171 = _S14 + ty_0;
                _S8 = _S171;
                if(_S171 >= ((&kernelContext_2)->p_0->m_0))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S172 = _S15 + tx_0;
                            _S9 = _S172;
                            if(_S172 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(_S171, _S172, _S19 + _S171 * (&kernelContext_2)->p_0->ldc_0 + _S172, acc_1[0U][0U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_0 = _S9 + 16U;
                            if(col_0 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(_S171, col_0, _S19 + _S171 * (&kernelContext_2)->p_0->ldc_0 + col_0, acc_1[0U][1U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_1 = _S9 + 32U;
                            if(col_1 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(_S171, col_1, _S19 + _S171 * (&kernelContext_2)->p_0->ldc_0 + col_1, acc_1[0U][2U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_2 = _S9 + 48U;
                            if(col_2 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(_S171, col_2, _S19 + _S171 * (&kernelContext_2)->p_0->ldc_0 + col_2, acc_1[0U][3U], &kernelContext_2);
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
                uint r_2 = _S8 + 16U;
                if(r_2 >= ((&kernelContext_2)->p_0->m_0))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S173 = _S15 + tx_0;
                            _S10 = _S173;
                            if(_S173 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_2, _S173, _S19 + r_2 * (&kernelContext_2)->p_0->ldc_0 + _S173, acc_1[1U][0U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_3 = _S10 + 16U;
                            if(col_3 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_2, col_3, _S19 + r_2 * (&kernelContext_2)->p_0->ldc_0 + col_3, acc_1[1U][1U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_4 = _S10 + 32U;
                            if(col_4 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_2, col_4, _S19 + r_2 * (&kernelContext_2)->p_0->ldc_0 + col_4, acc_1[1U][2U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_5 = _S10 + 48U;
                            if(col_5 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_2, col_5, _S19 + r_2 * (&kernelContext_2)->p_0->ldc_0 + col_5, acc_1[1U][3U], &kernelContext_2);
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
                uint r_3 = _S8 + 32U;
                if(r_3 >= ((&kernelContext_2)->p_0->m_0))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S174 = _S15 + tx_0;
                            _S11 = _S174;
                            if(_S174 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_3, _S174, _S19 + r_3 * (&kernelContext_2)->p_0->ldc_0 + _S174, acc_1[2U][0U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_6 = _S11 + 16U;
                            if(col_6 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_3, col_6, _S19 + r_3 * (&kernelContext_2)->p_0->ldc_0 + col_6, acc_1[2U][1U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_7 = _S11 + 32U;
                            if(col_7 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_3, col_7, _S19 + r_3 * (&kernelContext_2)->p_0->ldc_0 + col_7, acc_1[2U][2U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_8 = _S11 + 48U;
                            if(col_8 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_3, col_8, _S19 + r_3 * (&kernelContext_2)->p_0->ldc_0 + col_8, acc_1[2U][3U], &kernelContext_2);
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
                uint r_4 = _S8 + 48U;
                if(r_4 >= ((&kernelContext_2)->p_0->m_0))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S175 = _S15 + tx_0;
                            _S12 = _S175;
                            if(_S175 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_4, _S175, _S19 + r_4 * (&kernelContext_2)->p_0->ldc_0 + _S175, acc_1[3U][0U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_9 = _S12 + 16U;
                            if(col_9 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_4, col_9, _S19 + r_4 * (&kernelContext_2)->p_0->ldc_0 + col_9, acc_1[3U][1U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_10 = _S12 + 32U;
                            if(col_10 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_4, col_10, _S19 + r_4 * (&kernelContext_2)->p_0->ldc_0 + col_10, acc_1[3U][2U], &kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_11 = _S12 + 48U;
                            if(col_11 >= ((&kernelContext_2)->p_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(r_4, col_11, _S19 + r_4 * (&kernelContext_2)->p_0->ldc_0 + col_11, acc_1[3U][3U], &kernelContext_2);
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

