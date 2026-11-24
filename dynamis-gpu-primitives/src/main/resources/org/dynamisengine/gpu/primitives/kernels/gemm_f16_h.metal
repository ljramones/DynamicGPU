#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct GemmProblem_0
{
    uint m_0;
    uint n_0;
    uint k_0;
    uint lda_0;
    uint ldb_0;
    uint ldc_0;
    uint baseA_0;
    uint baseB_0;
    uint baseC_0;
    uint baseBias_0;
};

struct GemmParams_0
{
    uint m_1;
    uint n_1;
    uint k_1;
    uint lda_1;
    uint ldb_1;
    uint ldc_1;
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
    half device* a_0;
    half device* b_0;
    half device* c_0;
    half device* epBias_0;
    half device* epGate_0;
    half device* epResidual_0;
    array<array<float, int(64)>, int(16)> threadgroup* tileA_0;
    array<array<float, int(64)>, int(16)> threadgroup* tileB_0;
};

GemmProblem_0 gemmBatchProblem_0(uint z_0, KernelContext_0 thread* kernelContext_0)
{
    thread GemmProblem_0 q_0;
    (&q_0)->m_0 = kernelContext_0->p_0->m_1;
    (&q_0)->n_0 = kernelContext_0->p_0->n_1;
    (&q_0)->k_0 = kernelContext_0->p_0->k_1;
    (&q_0)->lda_0 = kernelContext_0->p_0->lda_1;
    (&q_0)->ldb_0 = kernelContext_0->p_0->ldb_1;
    (&q_0)->ldc_0 = kernelContext_0->p_0->ldc_1;
    (&q_0)->baseA_0 = z_0 * kernelContext_0->p_0->strideA_0;
    (&q_0)->baseB_0 = z_0 * kernelContext_0->p_0->strideB_0;
    (&q_0)->baseC_0 = z_0 * kernelContext_0->p_0->strideC_0;
    (&q_0)->baseBias_0 = 0U;
    return q_0;
}

float loadA_0(uint e_0, KernelContext_0 thread* kernelContext_1)
{
    return float(*(kernelContext_1->a_0+e_0));
}

float loadB_0(uint e_1, KernelContext_0 thread* kernelContext_2)
{
    return float(*(kernelContext_2->b_0+e_1));
}

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

float dg_max_0(float a_1, float b_1)
{
    if(dg_is_nan_0(a_1))
    {
        return a_1;
    }
    if(dg_is_nan_0(b_1))
    {
        return b_1;
    }
    bool _S3;
    if(a_1 == 0.0)
    {
        _S3 = b_1 == 0.0;
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        return (as_type<float>(((as_type<uint>((a_1))) & (as_type<uint>((b_1))))));
    }
    float _S4;
    if(a_1 > b_1)
    {
        _S4 = a_1;
    }
    else
    {
        _S4 = b_1;
    }
    return _S4;
}

float gemmEpilogue_0(float r_0, uint i_0, uint j_0, uint idx_0, uint biasBase_0, KernelContext_0 thread* kernelContext_3)
{
    float _S5;
    if((kernelContext_3->p_0->bias_0) == 1U)
    {
        _S5 = r_0 + float(*(kernelContext_3->epBias_0+(biasBase_0 + j_0)));
    }
    else
    {
        if((kernelContext_3->p_0->bias_0) == 2U)
        {
            _S5 = r_0 + float(*(kernelContext_3->epBias_0+(biasBase_0 + i_0)));
        }
        else
        {
            _S5 = r_0;
        }
    }
    if((kernelContext_3->p_0->act_0) == 1U)
    {
        _S5 = dg_max_0(_S5, 0.0);
    }
    else
    {
        if((kernelContext_3->p_0->act_0) == 2U)
        {
            _S5 = 0.5 * _S5 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S5 + 0.04471499845385551 * _S5 * _S5 * _S5))) + 1.0)));
        }
        else
        {
            if((kernelContext_3->p_0->act_0) == 3U)
            {
                _S5 = _S5 / (1.0 + exp(- _S5));
            }
        }
    }
    if((kernelContext_3->p_0->gated_0) != 0U)
    {
        _S5 = _S5 * float(*(kernelContext_3->epGate_0+idx_0));
    }
    if((kernelContext_3->p_0->residual_0) != 0U)
    {
        _S5 = _S5 + float(*(kernelContext_3->epResidual_0+idx_0));
    }
    return _S5;
}

void gemmStore_0(const GemmProblem_0 thread* q_1, uint i_1, uint j_1, float acc_0, KernelContext_0 thread* kernelContext_4)
{
    uint idx_1 = q_1->baseC_0 + i_1 * q_1->ldc_0 + j_1;
    float r_1;
    if((kernelContext_4->p_0->beta_0) == 0.0)
    {
        r_1 = kernelContext_4->p_0->alpha_0 * acc_0;
    }
    else
    {
        r_1 = kernelContext_4->p_0->alpha_0 * acc_0 + kernelContext_4->p_0->beta_0 * float(*(kernelContext_4->c_0+idx_1));
    }
    half device* _S6 = kernelContext_4->c_0+idx_1;
    float _S7 = gemmEpilogue_0(r_1, i_1, j_1, idx_1, q_1->baseBias_0, kernelContext_4);
    *_S6 = half(_S7);
    return;
}

void gemmTile_0(const GemmProblem_0 thread* pr_0, uint2 tile_0, uint3 local_0, KernelContext_0 thread* kernelContext_5)
{
    uint _S8;
    uint _S9;
    uint _S10;
    uint _S11;
    uint _S12;
    uint _S13;
    uint _S14;
    uint _S15;
    uint _S16;
    uint _S17;
    uint tx_0 = local_0.x;
    uint ty_0 = local_0.y;
    uint _S18 = ty_0 * 16U + tx_0;
    uint _S19 = tile_0.y * 64U;
    uint _S20 = tile_0.x * 64U;
    uint _S21 = pr_0->baseA_0;
    uint _S22 = pr_0->baseB_0;
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
        uint _S23 = pr_0->k_0;
        if(k0_0 < (pr_0->k_0))
        {
        }
        else
        {
            break;
        }
        uint _S24 = _S18 * 4U;
        uint ar_0 = _S24 / 16U;
        uint ak_0 = _S24 % 16U;
        uint gr_0 = _S19 + ar_0;
        uint gk_0 = k0_0 + ak_0;
        uint ai_0;
        if((kernelContext_5->p_0->transA_0) != 0U)
        {
            ai_0 = gk_0 * pr_0->lda_0 + gr_0;
        }
        else
        {
            ai_0 = gr_0 * pr_0->lda_0 + gk_0;
        }
        uint _S25 = pr_0->m_0;
        bool _S26;
        if(gr_0 < (pr_0->m_0))
        {
            _S26 = gk_0 < _S23;
        }
        else
        {
            _S26 = false;
        }
        float _S27;
        if(_S26)
        {
            float _S28 = loadA_0(_S21 + ai_0, kernelContext_5);
            _S27 = _S28;
        }
        else
        {
            _S27 = 0.0;
        }
        (*kernelContext_5->tileA_0)[ak_0][ar_0] = _S27;
        uint br_0 = _S24 / 64U;
        uint bc_0 = _S24 % 64U;
        uint hk_0 = k0_0 + br_0;
        uint gc_0 = _S20 + bc_0;
        uint bi_0;
        if((kernelContext_5->p_0->transB_0) != 0U)
        {
            bi_0 = gc_0 * pr_0->ldb_0 + hk_0;
        }
        else
        {
            bi_0 = hk_0 * pr_0->ldb_0 + gc_0;
        }
        bool _S29;
        if(hk_0 < _S23)
        {
            _S29 = gc_0 < (pr_0->n_0);
        }
        else
        {
            _S29 = false;
        }
        float _S30;
        if(_S29)
        {
            float _S31 = loadB_0(_S22 + bi_0, kernelContext_5);
            _S30 = _S31;
        }
        else
        {
            _S30 = 0.0;
        }
        (*kernelContext_5->tileB_0)[br_0][bc_0] = _S30;
        uint e_2 = _S24 + 1U;
        uint ar_1 = e_2 / 16U;
        uint ak_1 = e_2 % 16U;
        uint gr_1 = _S19 + ar_1;
        uint gk_1 = k0_0 + ak_1;
        uint ai_1;
        if((kernelContext_5->p_0->transA_0) != 0U)
        {
            ai_1 = gk_1 * pr_0->lda_0 + gr_1;
        }
        else
        {
            ai_1 = gr_1 * pr_0->lda_0 + gk_1;
        }
        bool _S32;
        if(gr_1 < _S25)
        {
            _S32 = gk_1 < _S23;
        }
        else
        {
            _S32 = false;
        }
        float _S33;
        if(_S32)
        {
            float _S34 = loadA_0(_S21 + ai_1, kernelContext_5);
            _S33 = _S34;
        }
        else
        {
            _S33 = 0.0;
        }
        (*kernelContext_5->tileA_0)[ak_1][ar_1] = _S33;
        uint br_1 = e_2 / 64U;
        uint bc_1 = e_2 % 64U;
        uint hk_1 = k0_0 + br_1;
        uint gc_1 = _S20 + bc_1;
        uint bi_1;
        if((kernelContext_5->p_0->transB_0) != 0U)
        {
            bi_1 = gc_1 * pr_0->ldb_0 + hk_1;
        }
        else
        {
            bi_1 = hk_1 * pr_0->ldb_0 + gc_1;
        }
        bool _S35;
        if(hk_1 < _S23)
        {
            _S35 = gc_1 < (pr_0->n_0);
        }
        else
        {
            _S35 = false;
        }
        float _S36;
        if(_S35)
        {
            float _S37 = loadB_0(_S22 + bi_1, kernelContext_5);
            _S36 = _S37;
        }
        else
        {
            _S36 = 0.0;
        }
        (*kernelContext_5->tileB_0)[br_1][bc_1] = _S36;
        uint e_3 = _S24 + 2U;
        uint ar_2 = e_3 / 16U;
        uint ak_2 = e_3 % 16U;
        uint gr_2 = _S19 + ar_2;
        uint gk_2 = k0_0 + ak_2;
        uint ai_2;
        if((kernelContext_5->p_0->transA_0) != 0U)
        {
            ai_2 = gk_2 * pr_0->lda_0 + gr_2;
        }
        else
        {
            ai_2 = gr_2 * pr_0->lda_0 + gk_2;
        }
        bool _S38;
        if(gr_2 < _S25)
        {
            _S38 = gk_2 < _S23;
        }
        else
        {
            _S38 = false;
        }
        float _S39;
        if(_S38)
        {
            float _S40 = loadA_0(_S21 + ai_2, kernelContext_5);
            _S39 = _S40;
        }
        else
        {
            _S39 = 0.0;
        }
        (*kernelContext_5->tileA_0)[ak_2][ar_2] = _S39;
        uint br_2 = e_3 / 64U;
        uint bc_2 = e_3 % 64U;
        uint hk_2 = k0_0 + br_2;
        uint gc_2 = _S20 + bc_2;
        uint bi_2;
        if((kernelContext_5->p_0->transB_0) != 0U)
        {
            bi_2 = gc_2 * pr_0->ldb_0 + hk_2;
        }
        else
        {
            bi_2 = hk_2 * pr_0->ldb_0 + gc_2;
        }
        bool _S41;
        if(hk_2 < _S23)
        {
            _S41 = gc_2 < (pr_0->n_0);
        }
        else
        {
            _S41 = false;
        }
        float _S42;
        if(_S41)
        {
            float _S43 = loadB_0(_S22 + bi_2, kernelContext_5);
            _S42 = _S43;
        }
        else
        {
            _S42 = 0.0;
        }
        (*kernelContext_5->tileB_0)[br_2][bc_2] = _S42;
        uint e_4 = _S24 + 3U;
        uint ar_3 = e_4 / 16U;
        uint ak_3 = e_4 % 16U;
        uint gr_3 = _S19 + ar_3;
        uint gk_3 = k0_0 + ak_3;
        uint ai_3;
        if((kernelContext_5->p_0->transA_0) != 0U)
        {
            ai_3 = gk_3 * pr_0->lda_0 + gr_3;
        }
        else
        {
            ai_3 = gr_3 * pr_0->lda_0 + gk_3;
        }
        bool _S44;
        if(gr_3 < _S25)
        {
            _S44 = gk_3 < _S23;
        }
        else
        {
            _S44 = false;
        }
        float _S45;
        if(_S44)
        {
            float _S46 = loadA_0(_S21 + ai_3, kernelContext_5);
            _S45 = _S46;
        }
        else
        {
            _S45 = 0.0;
        }
        (*kernelContext_5->tileA_0)[ak_3][ar_3] = _S45;
        uint br_3 = e_4 / 64U;
        uint bc_3 = e_4 % 64U;
        uint hk_3 = k0_0 + br_3;
        uint gc_3 = _S20 + bc_3;
        uint bi_3;
        if((kernelContext_5->p_0->transB_0) != 0U)
        {
            bi_3 = gc_3 * pr_0->ldb_0 + hk_3;
        }
        else
        {
            bi_3 = hk_3 * pr_0->ldb_0 + gc_3;
        }
        bool _S47;
        if(hk_3 < _S23)
        {
            _S47 = gc_3 < (pr_0->n_0);
        }
        else
        {
            _S47 = false;
        }
        float _S48;
        if(_S47)
        {
            float _S49 = loadB_0(_S22 + bi_3, kernelContext_5);
            _S48 = _S49;
        }
        else
        {
            _S48 = 0.0;
        }
        (*kernelContext_5->tileB_0)[br_3][bc_3] = _S48;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        thread array<float, int(4)> av_0;
        float _S50 = (*kernelContext_5->tileA_0)[0U][ty_0];
        av_0[0U] = (*kernelContext_5->tileA_0)[0U][ty_0];
        uint _S51 = ty_0 + 16U;
        float _S52 = (*kernelContext_5->tileA_0)[0U][_S51];
        av_0[1U] = (*kernelContext_5->tileA_0)[0U][_S51];
        uint _S53 = ty_0 + 32U;
        float _S54 = (*kernelContext_5->tileA_0)[0U][_S53];
        av_0[2U] = (*kernelContext_5->tileA_0)[0U][_S53];
        uint _S55 = ty_0 + 48U;
        float _S56 = (*kernelContext_5->tileA_0)[0U][_S55];
        av_0[3U] = (*kernelContext_5->tileA_0)[0U][_S55];
        thread array<float, int(4)> bv_0;
        float _S57 = (*kernelContext_5->tileB_0)[0U][tx_0];
        bv_0[0U] = (*kernelContext_5->tileB_0)[0U][tx_0];
        uint _S58 = tx_0 + 16U;
        float _S59 = (*kernelContext_5->tileB_0)[0U][_S58];
        bv_0[1U] = (*kernelContext_5->tileB_0)[0U][_S58];
        uint _S60 = tx_0 + 32U;
        float _S61 = (*kernelContext_5->tileB_0)[0U][_S60];
        bv_0[2U] = (*kernelContext_5->tileB_0)[0U][_S60];
        uint _S62 = tx_0 + 48U;
        float _S63 = (*kernelContext_5->tileB_0)[0U][_S62];
        bv_0[3U] = (*kernelContext_5->tileB_0)[0U][_S62];
        acc_1[0U][0U] = fma(_S50, _S57, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S50, _S59, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S50, _S61, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S50, _S63, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S52, _S57, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S52, _S59, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S52, _S61, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S52, _S63, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S54, _S57, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S54, _S59, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S54, _S61, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S54, _S63, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S56, _S57, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S56, _S59, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S56, _S61, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S56, _S63, acc_1[3U][3U]);
        thread array<float, int(4)> av_1;
        float _S64 = (*kernelContext_5->tileA_0)[1U][ty_0];
        av_1[0U] = (*kernelContext_5->tileA_0)[1U][ty_0];
        float _S65 = (*kernelContext_5->tileA_0)[1U][_S51];
        av_1[1U] = (*kernelContext_5->tileA_0)[1U][_S51];
        float _S66 = (*kernelContext_5->tileA_0)[1U][_S53];
        av_1[2U] = (*kernelContext_5->tileA_0)[1U][_S53];
        float _S67 = (*kernelContext_5->tileA_0)[1U][_S55];
        av_1[3U] = (*kernelContext_5->tileA_0)[1U][_S55];
        thread array<float, int(4)> bv_1;
        float _S68 = (*kernelContext_5->tileB_0)[1U][tx_0];
        bv_1[0U] = (*kernelContext_5->tileB_0)[1U][tx_0];
        float _S69 = (*kernelContext_5->tileB_0)[1U][_S58];
        bv_1[1U] = (*kernelContext_5->tileB_0)[1U][_S58];
        float _S70 = (*kernelContext_5->tileB_0)[1U][_S60];
        bv_1[2U] = (*kernelContext_5->tileB_0)[1U][_S60];
        float _S71 = (*kernelContext_5->tileB_0)[1U][_S62];
        bv_1[3U] = (*kernelContext_5->tileB_0)[1U][_S62];
        acc_1[0U][0U] = fma(_S64, _S68, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S64, _S69, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S64, _S70, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S64, _S71, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S65, _S68, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S65, _S69, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S65, _S70, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S65, _S71, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S66, _S68, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S66, _S69, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S66, _S70, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S66, _S71, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S67, _S68, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S67, _S69, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S67, _S70, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S67, _S71, acc_1[3U][3U]);
        thread array<float, int(4)> av_2;
        float _S72 = (*kernelContext_5->tileA_0)[2U][ty_0];
        av_2[0U] = (*kernelContext_5->tileA_0)[2U][ty_0];
        float _S73 = (*kernelContext_5->tileA_0)[2U][_S51];
        av_2[1U] = (*kernelContext_5->tileA_0)[2U][_S51];
        float _S74 = (*kernelContext_5->tileA_0)[2U][_S53];
        av_2[2U] = (*kernelContext_5->tileA_0)[2U][_S53];
        float _S75 = (*kernelContext_5->tileA_0)[2U][_S55];
        av_2[3U] = (*kernelContext_5->tileA_0)[2U][_S55];
        thread array<float, int(4)> bv_2;
        float _S76 = (*kernelContext_5->tileB_0)[2U][tx_0];
        bv_2[0U] = (*kernelContext_5->tileB_0)[2U][tx_0];
        float _S77 = (*kernelContext_5->tileB_0)[2U][_S58];
        bv_2[1U] = (*kernelContext_5->tileB_0)[2U][_S58];
        float _S78 = (*kernelContext_5->tileB_0)[2U][_S60];
        bv_2[2U] = (*kernelContext_5->tileB_0)[2U][_S60];
        float _S79 = (*kernelContext_5->tileB_0)[2U][_S62];
        bv_2[3U] = (*kernelContext_5->tileB_0)[2U][_S62];
        acc_1[0U][0U] = fma(_S72, _S76, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S72, _S77, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S72, _S78, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S72, _S79, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S73, _S76, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S73, _S77, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S73, _S78, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S73, _S79, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S74, _S76, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S74, _S77, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S74, _S78, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S74, _S79, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S75, _S76, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S75, _S77, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S75, _S78, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S75, _S79, acc_1[3U][3U]);
        thread array<float, int(4)> av_3;
        float _S80 = (*kernelContext_5->tileA_0)[3U][ty_0];
        av_3[0U] = (*kernelContext_5->tileA_0)[3U][ty_0];
        float _S81 = (*kernelContext_5->tileA_0)[3U][_S51];
        av_3[1U] = (*kernelContext_5->tileA_0)[3U][_S51];
        float _S82 = (*kernelContext_5->tileA_0)[3U][_S53];
        av_3[2U] = (*kernelContext_5->tileA_0)[3U][_S53];
        float _S83 = (*kernelContext_5->tileA_0)[3U][_S55];
        av_3[3U] = (*kernelContext_5->tileA_0)[3U][_S55];
        thread array<float, int(4)> bv_3;
        float _S84 = (*kernelContext_5->tileB_0)[3U][tx_0];
        bv_3[0U] = (*kernelContext_5->tileB_0)[3U][tx_0];
        float _S85 = (*kernelContext_5->tileB_0)[3U][_S58];
        bv_3[1U] = (*kernelContext_5->tileB_0)[3U][_S58];
        float _S86 = (*kernelContext_5->tileB_0)[3U][_S60];
        bv_3[2U] = (*kernelContext_5->tileB_0)[3U][_S60];
        float _S87 = (*kernelContext_5->tileB_0)[3U][_S62];
        bv_3[3U] = (*kernelContext_5->tileB_0)[3U][_S62];
        acc_1[0U][0U] = fma(_S80, _S84, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S80, _S85, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S80, _S86, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S80, _S87, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S81, _S84, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S81, _S85, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S81, _S86, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S81, _S87, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S82, _S84, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S82, _S85, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S82, _S86, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S82, _S87, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S83, _S84, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S83, _S85, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S83, _S86, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S83, _S87, acc_1[3U][3U]);
        thread array<float, int(4)> av_4;
        float _S88 = (*kernelContext_5->tileA_0)[4U][ty_0];
        av_4[0U] = (*kernelContext_5->tileA_0)[4U][ty_0];
        float _S89 = (*kernelContext_5->tileA_0)[4U][_S51];
        av_4[1U] = (*kernelContext_5->tileA_0)[4U][_S51];
        float _S90 = (*kernelContext_5->tileA_0)[4U][_S53];
        av_4[2U] = (*kernelContext_5->tileA_0)[4U][_S53];
        float _S91 = (*kernelContext_5->tileA_0)[4U][_S55];
        av_4[3U] = (*kernelContext_5->tileA_0)[4U][_S55];
        thread array<float, int(4)> bv_4;
        float _S92 = (*kernelContext_5->tileB_0)[4U][tx_0];
        bv_4[0U] = (*kernelContext_5->tileB_0)[4U][tx_0];
        float _S93 = (*kernelContext_5->tileB_0)[4U][_S58];
        bv_4[1U] = (*kernelContext_5->tileB_0)[4U][_S58];
        float _S94 = (*kernelContext_5->tileB_0)[4U][_S60];
        bv_4[2U] = (*kernelContext_5->tileB_0)[4U][_S60];
        float _S95 = (*kernelContext_5->tileB_0)[4U][_S62];
        bv_4[3U] = (*kernelContext_5->tileB_0)[4U][_S62];
        acc_1[0U][0U] = fma(_S88, _S92, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S88, _S93, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S88, _S94, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S88, _S95, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S89, _S92, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S89, _S93, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S89, _S94, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S89, _S95, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S90, _S92, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S90, _S93, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S90, _S94, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S90, _S95, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S91, _S92, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S91, _S93, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S91, _S94, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S91, _S95, acc_1[3U][3U]);
        thread array<float, int(4)> av_5;
        float _S96 = (*kernelContext_5->tileA_0)[5U][ty_0];
        av_5[0U] = (*kernelContext_5->tileA_0)[5U][ty_0];
        float _S97 = (*kernelContext_5->tileA_0)[5U][_S51];
        av_5[1U] = (*kernelContext_5->tileA_0)[5U][_S51];
        float _S98 = (*kernelContext_5->tileA_0)[5U][_S53];
        av_5[2U] = (*kernelContext_5->tileA_0)[5U][_S53];
        float _S99 = (*kernelContext_5->tileA_0)[5U][_S55];
        av_5[3U] = (*kernelContext_5->tileA_0)[5U][_S55];
        thread array<float, int(4)> bv_5;
        float _S100 = (*kernelContext_5->tileB_0)[5U][tx_0];
        bv_5[0U] = (*kernelContext_5->tileB_0)[5U][tx_0];
        float _S101 = (*kernelContext_5->tileB_0)[5U][_S58];
        bv_5[1U] = (*kernelContext_5->tileB_0)[5U][_S58];
        float _S102 = (*kernelContext_5->tileB_0)[5U][_S60];
        bv_5[2U] = (*kernelContext_5->tileB_0)[5U][_S60];
        float _S103 = (*kernelContext_5->tileB_0)[5U][_S62];
        bv_5[3U] = (*kernelContext_5->tileB_0)[5U][_S62];
        acc_1[0U][0U] = fma(_S96, _S100, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S96, _S101, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S96, _S102, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S96, _S103, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S97, _S100, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S97, _S101, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S97, _S102, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S97, _S103, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S98, _S100, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S98, _S101, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S98, _S102, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S98, _S103, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S99, _S100, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S99, _S101, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S99, _S102, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S99, _S103, acc_1[3U][3U]);
        thread array<float, int(4)> av_6;
        float _S104 = (*kernelContext_5->tileA_0)[6U][ty_0];
        av_6[0U] = (*kernelContext_5->tileA_0)[6U][ty_0];
        float _S105 = (*kernelContext_5->tileA_0)[6U][_S51];
        av_6[1U] = (*kernelContext_5->tileA_0)[6U][_S51];
        float _S106 = (*kernelContext_5->tileA_0)[6U][_S53];
        av_6[2U] = (*kernelContext_5->tileA_0)[6U][_S53];
        float _S107 = (*kernelContext_5->tileA_0)[6U][_S55];
        av_6[3U] = (*kernelContext_5->tileA_0)[6U][_S55];
        thread array<float, int(4)> bv_6;
        float _S108 = (*kernelContext_5->tileB_0)[6U][tx_0];
        bv_6[0U] = (*kernelContext_5->tileB_0)[6U][tx_0];
        float _S109 = (*kernelContext_5->tileB_0)[6U][_S58];
        bv_6[1U] = (*kernelContext_5->tileB_0)[6U][_S58];
        float _S110 = (*kernelContext_5->tileB_0)[6U][_S60];
        bv_6[2U] = (*kernelContext_5->tileB_0)[6U][_S60];
        float _S111 = (*kernelContext_5->tileB_0)[6U][_S62];
        bv_6[3U] = (*kernelContext_5->tileB_0)[6U][_S62];
        acc_1[0U][0U] = fma(_S104, _S108, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S104, _S109, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S104, _S110, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S104, _S111, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S105, _S108, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S105, _S109, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S105, _S110, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S105, _S111, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S106, _S108, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S106, _S109, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S106, _S110, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S106, _S111, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S107, _S108, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S107, _S109, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S107, _S110, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S107, _S111, acc_1[3U][3U]);
        thread array<float, int(4)> av_7;
        float _S112 = (*kernelContext_5->tileA_0)[7U][ty_0];
        av_7[0U] = (*kernelContext_5->tileA_0)[7U][ty_0];
        float _S113 = (*kernelContext_5->tileA_0)[7U][_S51];
        av_7[1U] = (*kernelContext_5->tileA_0)[7U][_S51];
        float _S114 = (*kernelContext_5->tileA_0)[7U][_S53];
        av_7[2U] = (*kernelContext_5->tileA_0)[7U][_S53];
        float _S115 = (*kernelContext_5->tileA_0)[7U][_S55];
        av_7[3U] = (*kernelContext_5->tileA_0)[7U][_S55];
        thread array<float, int(4)> bv_7;
        float _S116 = (*kernelContext_5->tileB_0)[7U][tx_0];
        bv_7[0U] = (*kernelContext_5->tileB_0)[7U][tx_0];
        float _S117 = (*kernelContext_5->tileB_0)[7U][_S58];
        bv_7[1U] = (*kernelContext_5->tileB_0)[7U][_S58];
        float _S118 = (*kernelContext_5->tileB_0)[7U][_S60];
        bv_7[2U] = (*kernelContext_5->tileB_0)[7U][_S60];
        float _S119 = (*kernelContext_5->tileB_0)[7U][_S62];
        bv_7[3U] = (*kernelContext_5->tileB_0)[7U][_S62];
        acc_1[0U][0U] = fma(_S112, _S116, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S112, _S117, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S112, _S118, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S112, _S119, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S113, _S116, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S113, _S117, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S113, _S118, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S113, _S119, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S114, _S116, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S114, _S117, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S114, _S118, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S114, _S119, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S115, _S116, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S115, _S117, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S115, _S118, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S115, _S119, acc_1[3U][3U]);
        thread array<float, int(4)> av_8;
        float _S120 = (*kernelContext_5->tileA_0)[8U][ty_0];
        av_8[0U] = (*kernelContext_5->tileA_0)[8U][ty_0];
        float _S121 = (*kernelContext_5->tileA_0)[8U][_S51];
        av_8[1U] = (*kernelContext_5->tileA_0)[8U][_S51];
        float _S122 = (*kernelContext_5->tileA_0)[8U][_S53];
        av_8[2U] = (*kernelContext_5->tileA_0)[8U][_S53];
        float _S123 = (*kernelContext_5->tileA_0)[8U][_S55];
        av_8[3U] = (*kernelContext_5->tileA_0)[8U][_S55];
        thread array<float, int(4)> bv_8;
        float _S124 = (*kernelContext_5->tileB_0)[8U][tx_0];
        bv_8[0U] = (*kernelContext_5->tileB_0)[8U][tx_0];
        float _S125 = (*kernelContext_5->tileB_0)[8U][_S58];
        bv_8[1U] = (*kernelContext_5->tileB_0)[8U][_S58];
        float _S126 = (*kernelContext_5->tileB_0)[8U][_S60];
        bv_8[2U] = (*kernelContext_5->tileB_0)[8U][_S60];
        float _S127 = (*kernelContext_5->tileB_0)[8U][_S62];
        bv_8[3U] = (*kernelContext_5->tileB_0)[8U][_S62];
        acc_1[0U][0U] = fma(_S120, _S124, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S120, _S125, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S120, _S126, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S120, _S127, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S121, _S124, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S121, _S125, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S121, _S126, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S121, _S127, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S122, _S124, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S122, _S125, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S122, _S126, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S122, _S127, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S123, _S124, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S123, _S125, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S123, _S126, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S123, _S127, acc_1[3U][3U]);
        thread array<float, int(4)> av_9;
        float _S128 = (*kernelContext_5->tileA_0)[9U][ty_0];
        av_9[0U] = (*kernelContext_5->tileA_0)[9U][ty_0];
        float _S129 = (*kernelContext_5->tileA_0)[9U][_S51];
        av_9[1U] = (*kernelContext_5->tileA_0)[9U][_S51];
        float _S130 = (*kernelContext_5->tileA_0)[9U][_S53];
        av_9[2U] = (*kernelContext_5->tileA_0)[9U][_S53];
        float _S131 = (*kernelContext_5->tileA_0)[9U][_S55];
        av_9[3U] = (*kernelContext_5->tileA_0)[9U][_S55];
        thread array<float, int(4)> bv_9;
        float _S132 = (*kernelContext_5->tileB_0)[9U][tx_0];
        bv_9[0U] = (*kernelContext_5->tileB_0)[9U][tx_0];
        float _S133 = (*kernelContext_5->tileB_0)[9U][_S58];
        bv_9[1U] = (*kernelContext_5->tileB_0)[9U][_S58];
        float _S134 = (*kernelContext_5->tileB_0)[9U][_S60];
        bv_9[2U] = (*kernelContext_5->tileB_0)[9U][_S60];
        float _S135 = (*kernelContext_5->tileB_0)[9U][_S62];
        bv_9[3U] = (*kernelContext_5->tileB_0)[9U][_S62];
        acc_1[0U][0U] = fma(_S128, _S132, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S128, _S133, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S128, _S134, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S128, _S135, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S129, _S132, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S129, _S133, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S129, _S134, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S129, _S135, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S130, _S132, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S130, _S133, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S130, _S134, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S130, _S135, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S131, _S132, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S131, _S133, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S131, _S134, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S131, _S135, acc_1[3U][3U]);
        thread array<float, int(4)> av_10;
        float _S136 = (*kernelContext_5->tileA_0)[10U][ty_0];
        av_10[0U] = (*kernelContext_5->tileA_0)[10U][ty_0];
        float _S137 = (*kernelContext_5->tileA_0)[10U][_S51];
        av_10[1U] = (*kernelContext_5->tileA_0)[10U][_S51];
        float _S138 = (*kernelContext_5->tileA_0)[10U][_S53];
        av_10[2U] = (*kernelContext_5->tileA_0)[10U][_S53];
        float _S139 = (*kernelContext_5->tileA_0)[10U][_S55];
        av_10[3U] = (*kernelContext_5->tileA_0)[10U][_S55];
        thread array<float, int(4)> bv_10;
        float _S140 = (*kernelContext_5->tileB_0)[10U][tx_0];
        bv_10[0U] = (*kernelContext_5->tileB_0)[10U][tx_0];
        float _S141 = (*kernelContext_5->tileB_0)[10U][_S58];
        bv_10[1U] = (*kernelContext_5->tileB_0)[10U][_S58];
        float _S142 = (*kernelContext_5->tileB_0)[10U][_S60];
        bv_10[2U] = (*kernelContext_5->tileB_0)[10U][_S60];
        float _S143 = (*kernelContext_5->tileB_0)[10U][_S62];
        bv_10[3U] = (*kernelContext_5->tileB_0)[10U][_S62];
        acc_1[0U][0U] = fma(_S136, _S140, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S136, _S141, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S136, _S142, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S136, _S143, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S137, _S140, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S137, _S141, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S137, _S142, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S137, _S143, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S138, _S140, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S138, _S141, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S138, _S142, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S138, _S143, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S139, _S140, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S139, _S141, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S139, _S142, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S139, _S143, acc_1[3U][3U]);
        thread array<float, int(4)> av_11;
        float _S144 = (*kernelContext_5->tileA_0)[11U][ty_0];
        av_11[0U] = (*kernelContext_5->tileA_0)[11U][ty_0];
        float _S145 = (*kernelContext_5->tileA_0)[11U][_S51];
        av_11[1U] = (*kernelContext_5->tileA_0)[11U][_S51];
        float _S146 = (*kernelContext_5->tileA_0)[11U][_S53];
        av_11[2U] = (*kernelContext_5->tileA_0)[11U][_S53];
        float _S147 = (*kernelContext_5->tileA_0)[11U][_S55];
        av_11[3U] = (*kernelContext_5->tileA_0)[11U][_S55];
        thread array<float, int(4)> bv_11;
        float _S148 = (*kernelContext_5->tileB_0)[11U][tx_0];
        bv_11[0U] = (*kernelContext_5->tileB_0)[11U][tx_0];
        float _S149 = (*kernelContext_5->tileB_0)[11U][_S58];
        bv_11[1U] = (*kernelContext_5->tileB_0)[11U][_S58];
        float _S150 = (*kernelContext_5->tileB_0)[11U][_S60];
        bv_11[2U] = (*kernelContext_5->tileB_0)[11U][_S60];
        float _S151 = (*kernelContext_5->tileB_0)[11U][_S62];
        bv_11[3U] = (*kernelContext_5->tileB_0)[11U][_S62];
        acc_1[0U][0U] = fma(_S144, _S148, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S144, _S149, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S144, _S150, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S144, _S151, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S145, _S148, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S145, _S149, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S145, _S150, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S145, _S151, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S146, _S148, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S146, _S149, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S146, _S150, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S146, _S151, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S147, _S148, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S147, _S149, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S147, _S150, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S147, _S151, acc_1[3U][3U]);
        thread array<float, int(4)> av_12;
        float _S152 = (*kernelContext_5->tileA_0)[12U][ty_0];
        av_12[0U] = (*kernelContext_5->tileA_0)[12U][ty_0];
        float _S153 = (*kernelContext_5->tileA_0)[12U][_S51];
        av_12[1U] = (*kernelContext_5->tileA_0)[12U][_S51];
        float _S154 = (*kernelContext_5->tileA_0)[12U][_S53];
        av_12[2U] = (*kernelContext_5->tileA_0)[12U][_S53];
        float _S155 = (*kernelContext_5->tileA_0)[12U][_S55];
        av_12[3U] = (*kernelContext_5->tileA_0)[12U][_S55];
        thread array<float, int(4)> bv_12;
        float _S156 = (*kernelContext_5->tileB_0)[12U][tx_0];
        bv_12[0U] = (*kernelContext_5->tileB_0)[12U][tx_0];
        float _S157 = (*kernelContext_5->tileB_0)[12U][_S58];
        bv_12[1U] = (*kernelContext_5->tileB_0)[12U][_S58];
        float _S158 = (*kernelContext_5->tileB_0)[12U][_S60];
        bv_12[2U] = (*kernelContext_5->tileB_0)[12U][_S60];
        float _S159 = (*kernelContext_5->tileB_0)[12U][_S62];
        bv_12[3U] = (*kernelContext_5->tileB_0)[12U][_S62];
        acc_1[0U][0U] = fma(_S152, _S156, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S152, _S157, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S152, _S158, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S152, _S159, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S153, _S156, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S153, _S157, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S153, _S158, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S153, _S159, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S154, _S156, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S154, _S157, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S154, _S158, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S154, _S159, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S155, _S156, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S155, _S157, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S155, _S158, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S155, _S159, acc_1[3U][3U]);
        thread array<float, int(4)> av_13;
        float _S160 = (*kernelContext_5->tileA_0)[13U][ty_0];
        av_13[0U] = (*kernelContext_5->tileA_0)[13U][ty_0];
        float _S161 = (*kernelContext_5->tileA_0)[13U][_S51];
        av_13[1U] = (*kernelContext_5->tileA_0)[13U][_S51];
        float _S162 = (*kernelContext_5->tileA_0)[13U][_S53];
        av_13[2U] = (*kernelContext_5->tileA_0)[13U][_S53];
        float _S163 = (*kernelContext_5->tileA_0)[13U][_S55];
        av_13[3U] = (*kernelContext_5->tileA_0)[13U][_S55];
        thread array<float, int(4)> bv_13;
        float _S164 = (*kernelContext_5->tileB_0)[13U][tx_0];
        bv_13[0U] = (*kernelContext_5->tileB_0)[13U][tx_0];
        float _S165 = (*kernelContext_5->tileB_0)[13U][_S58];
        bv_13[1U] = (*kernelContext_5->tileB_0)[13U][_S58];
        float _S166 = (*kernelContext_5->tileB_0)[13U][_S60];
        bv_13[2U] = (*kernelContext_5->tileB_0)[13U][_S60];
        float _S167 = (*kernelContext_5->tileB_0)[13U][_S62];
        bv_13[3U] = (*kernelContext_5->tileB_0)[13U][_S62];
        acc_1[0U][0U] = fma(_S160, _S164, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S160, _S165, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S160, _S166, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S160, _S167, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S161, _S164, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S161, _S165, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S161, _S166, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S161, _S167, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S162, _S164, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S162, _S165, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S162, _S166, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S162, _S167, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S163, _S164, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S163, _S165, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S163, _S166, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S163, _S167, acc_1[3U][3U]);
        thread array<float, int(4)> av_14;
        float _S168 = (*kernelContext_5->tileA_0)[14U][ty_0];
        av_14[0U] = (*kernelContext_5->tileA_0)[14U][ty_0];
        float _S169 = (*kernelContext_5->tileA_0)[14U][_S51];
        av_14[1U] = (*kernelContext_5->tileA_0)[14U][_S51];
        float _S170 = (*kernelContext_5->tileA_0)[14U][_S53];
        av_14[2U] = (*kernelContext_5->tileA_0)[14U][_S53];
        float _S171 = (*kernelContext_5->tileA_0)[14U][_S55];
        av_14[3U] = (*kernelContext_5->tileA_0)[14U][_S55];
        thread array<float, int(4)> bv_14;
        float _S172 = (*kernelContext_5->tileB_0)[14U][tx_0];
        bv_14[0U] = (*kernelContext_5->tileB_0)[14U][tx_0];
        float _S173 = (*kernelContext_5->tileB_0)[14U][_S58];
        bv_14[1U] = (*kernelContext_5->tileB_0)[14U][_S58];
        float _S174 = (*kernelContext_5->tileB_0)[14U][_S60];
        bv_14[2U] = (*kernelContext_5->tileB_0)[14U][_S60];
        float _S175 = (*kernelContext_5->tileB_0)[14U][_S62];
        bv_14[3U] = (*kernelContext_5->tileB_0)[14U][_S62];
        acc_1[0U][0U] = fma(_S168, _S172, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S168, _S173, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S168, _S174, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S168, _S175, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S169, _S172, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S169, _S173, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S169, _S174, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S169, _S175, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S170, _S172, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S170, _S173, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S170, _S174, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S170, _S175, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S171, _S172, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S171, _S173, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S171, _S174, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S171, _S175, acc_1[3U][3U]);
        thread array<float, int(4)> av_15;
        float _S176 = (*kernelContext_5->tileA_0)[15U][ty_0];
        av_15[0U] = (*kernelContext_5->tileA_0)[15U][ty_0];
        float _S177 = (*kernelContext_5->tileA_0)[15U][_S51];
        av_15[1U] = (*kernelContext_5->tileA_0)[15U][_S51];
        float _S178 = (*kernelContext_5->tileA_0)[15U][_S53];
        av_15[2U] = (*kernelContext_5->tileA_0)[15U][_S53];
        float _S179 = (*kernelContext_5->tileA_0)[15U][_S55];
        av_15[3U] = (*kernelContext_5->tileA_0)[15U][_S55];
        thread array<float, int(4)> bv_15;
        float _S180 = (*kernelContext_5->tileB_0)[15U][tx_0];
        bv_15[0U] = (*kernelContext_5->tileB_0)[15U][tx_0];
        float _S181 = (*kernelContext_5->tileB_0)[15U][_S58];
        bv_15[1U] = (*kernelContext_5->tileB_0)[15U][_S58];
        float _S182 = (*kernelContext_5->tileB_0)[15U][_S60];
        bv_15[2U] = (*kernelContext_5->tileB_0)[15U][_S60];
        float _S183 = (*kernelContext_5->tileB_0)[15U][_S62];
        bv_15[3U] = (*kernelContext_5->tileB_0)[15U][_S62];
        acc_1[0U][0U] = fma(_S176, _S180, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S176, _S181, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S176, _S182, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S176, _S183, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S177, _S180, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S177, _S181, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S177, _S182, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S177, _S183, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S178, _S180, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S178, _S181, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S178, _S182, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S178, _S183, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S179, _S180, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S179, _S181, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S179, _S182, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S179, _S183, acc_1[3U][3U]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k0_0 = k0_0 + 16U;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                uint _S184 = _S19 + ty_0;
                _S8 = _S184;
                _S9 = pr_0->m_0;
                if(_S184 >= (pr_0->m_0))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S185 = _S20 + tx_0;
                            _S10 = _S185;
                            _S11 = pr_0->n_0;
                            if(_S185 >= (pr_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S184, _S185, acc_1[0U][0U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_0 = _S10 + 16U;
                            if(col_0 >= _S11)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S184, col_0, acc_1[0U][1U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_1 = _S10 + 32U;
                            if(col_1 >= _S11)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S184, col_1, acc_1[0U][2U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_2 = _S10 + 48U;
                            if(col_2 >= _S11)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S184, col_2, acc_1[0U][3U], kernelContext_5);
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
                if(r_2 >= _S9)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S186 = _S20 + tx_0;
                            _S12 = _S186;
                            _S13 = pr_0->n_0;
                            if(_S186 >= (pr_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, _S186, acc_1[1U][0U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_3 = _S12 + 16U;
                            if(col_3 >= _S13)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, col_3, acc_1[1U][1U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_4 = _S12 + 32U;
                            if(col_4 >= _S13)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, col_4, acc_1[1U][2U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_5 = _S12 + 48U;
                            if(col_5 >= _S13)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, col_5, acc_1[1U][3U], kernelContext_5);
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
                if(r_3 >= _S9)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S187 = _S20 + tx_0;
                            _S14 = _S187;
                            _S15 = pr_0->n_0;
                            if(_S187 >= (pr_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, _S187, acc_1[2U][0U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_6 = _S14 + 16U;
                            if(col_6 >= _S15)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, col_6, acc_1[2U][1U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_7 = _S14 + 32U;
                            if(col_7 >= _S15)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, col_7, acc_1[2U][2U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_8 = _S14 + 48U;
                            if(col_8 >= _S15)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, col_8, acc_1[2U][3U], kernelContext_5);
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
                if(r_4 >= _S9)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S188 = _S20 + tx_0;
                            _S16 = _S188;
                            _S17 = pr_0->n_0;
                            if(_S188 >= (pr_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, _S188, acc_1[3U][0U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_9 = _S16 + 16U;
                            if(col_9 >= _S17)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, col_9, acc_1[3U][1U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_10 = _S16 + 32U;
                            if(col_10 >= _S17)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, col_10, acc_1[3U][2U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_11 = _S16 + 48U;
                            if(col_11 >= _S17)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, col_11, acc_1[3U][3U], kernelContext_5);
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

[[kernel]] void gemm_f16_h(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], half device* a_2 [[buffer(0)]], half device* b_2 [[buffer(1)]], half device* c_1 [[buffer(2)]], half device* epBias_1 [[buffer(3)]], half device* epGate_1 [[buffer(4)]], half device* epResidual_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->p_0 = p_1;
    (&kernelContext_6)->a_0 = a_2;
    (&kernelContext_6)->b_0 = b_2;
    (&kernelContext_6)->c_0 = c_1;
    (&kernelContext_6)->epBias_0 = epBias_1;
    (&kernelContext_6)->epGate_0 = epGate_1;
    (&kernelContext_6)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(64)>, int(16)> tileA_1;
    (&kernelContext_6)->tileA_0 = &tileA_1;
    threadgroup array<array<float, int(64)>, int(16)> tileB_1;
    (&kernelContext_6)->tileB_0 = &tileB_1;
    GemmProblem_0 _S189 = gemmBatchProblem_0(group_0.z, &kernelContext_6);
    uint2 _S190 = group_0.xy;
    thread GemmProblem_0 _S191 = _S189;
    gemmTile_0(&_S191, _S190, local_1, &kernelContext_6);
    return;
}

