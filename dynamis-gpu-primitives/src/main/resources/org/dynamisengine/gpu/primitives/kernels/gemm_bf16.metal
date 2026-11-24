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
    uint device* a_0;
    uint device* b_0;
    float device* c_0;
    float device* epBias_0;
    float device* epGate_0;
    float device* epResidual_0;
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

float bf16At_0(uint w_0, uint e_0)
{
    uint _S1;
    if((e_0 & 1U) != 0U)
    {
        _S1 = w_0 & 4294901760U;
    }
    else
    {
        _S1 = w_0 << 16U;
    }
    return (as_type<float>((_S1)));
}

float loadA_0(uint e_1, KernelContext_0 thread* kernelContext_1)
{
    return bf16At_0(*(kernelContext_1->a_0+(e_1 >> 1U)), e_1);
}

float loadB_0(uint e_2, KernelContext_0 thread* kernelContext_2)
{
    return bf16At_0(*(kernelContext_2->b_0+(e_2 >> 1U)), e_2);
}

bool dg_is_nan_0(float x_0)
{
    uint _S2 = (as_type<uint>((x_0)));
    bool _S3;
    if((_S2 & 2139095040U) == 2139095040U)
    {
        _S3 = (_S2 & 8388607U) != 0U;
    }
    else
    {
        _S3 = false;
    }
    return _S3;
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
    bool _S4;
    if(a_1 == 0.0)
    {
        _S4 = b_1 == 0.0;
    }
    else
    {
        _S4 = false;
    }
    if(_S4)
    {
        return (as_type<float>(((as_type<uint>((a_1))) & (as_type<uint>((b_1))))));
    }
    float _S5;
    if(a_1 > b_1)
    {
        _S5 = a_1;
    }
    else
    {
        _S5 = b_1;
    }
    return _S5;
}

float gemmEpilogue_0(float r_0, uint i_0, uint j_0, uint idx_0, uint biasBase_0, KernelContext_0 thread* kernelContext_3)
{
    float _S6;
    if((kernelContext_3->p_0->bias_0) == 1U)
    {
        _S6 = r_0 + *(kernelContext_3->epBias_0+(biasBase_0 + j_0));
    }
    else
    {
        if((kernelContext_3->p_0->bias_0) == 2U)
        {
            _S6 = r_0 + *(kernelContext_3->epBias_0+(biasBase_0 + i_0));
        }
        else
        {
            _S6 = r_0;
        }
    }
    if((kernelContext_3->p_0->act_0) == 1U)
    {
        _S6 = dg_max_0(_S6, 0.0);
    }
    else
    {
        if((kernelContext_3->p_0->act_0) == 2U)
        {
            _S6 = 0.5 * _S6 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S6 + 0.04471499845385551 * _S6 * _S6 * _S6))) + 1.0)));
        }
        else
        {
            if((kernelContext_3->p_0->act_0) == 3U)
            {
                _S6 = _S6 / (1.0 + exp(- _S6));
            }
        }
    }
    if((kernelContext_3->p_0->gated_0) != 0U)
    {
        _S6 = _S6 * *(kernelContext_3->epGate_0+idx_0);
    }
    if((kernelContext_3->p_0->residual_0) != 0U)
    {
        _S6 = _S6 + *(kernelContext_3->epResidual_0+idx_0);
    }
    return _S6;
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
        r_1 = kernelContext_4->p_0->alpha_0 * acc_0 + kernelContext_4->p_0->beta_0 * *(kernelContext_4->c_0+idx_1);
    }
    float device* _S7 = kernelContext_4->c_0+idx_1;
    float _S8 = gemmEpilogue_0(r_1, i_1, j_1, idx_1, q_1->baseBias_0, kernelContext_4);
    *_S7 = _S8;
    return;
}

void gemmTile_0(const GemmProblem_0 thread* pr_0, uint2 tile_0, uint3 local_0, KernelContext_0 thread* kernelContext_5)
{
    uint _S9;
    uint _S10;
    uint _S11;
    uint _S12;
    uint _S13;
    uint _S14;
    uint _S15;
    uint _S16;
    uint _S17;
    uint _S18;
    uint tx_0 = local_0.x;
    uint ty_0 = local_0.y;
    uint _S19 = ty_0 * 16U + tx_0;
    uint _S20 = tile_0.y * 64U;
    uint _S21 = tile_0.x * 64U;
    uint _S22 = pr_0->baseA_0;
    uint _S23 = pr_0->baseB_0;
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
        uint _S24 = pr_0->k_0;
        if(k0_0 < (pr_0->k_0))
        {
        }
        else
        {
            break;
        }
        uint _S25 = _S19 * 4U;
        uint ar_0 = _S25 / 16U;
        uint ak_0 = _S25 % 16U;
        uint gr_0 = _S20 + ar_0;
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
        uint _S26 = pr_0->m_0;
        bool _S27;
        if(gr_0 < (pr_0->m_0))
        {
            _S27 = gk_0 < _S24;
        }
        else
        {
            _S27 = false;
        }
        float _S28;
        if(_S27)
        {
            float _S29 = loadA_0(_S22 + ai_0, kernelContext_5);
            _S28 = _S29;
        }
        else
        {
            _S28 = 0.0;
        }
        (*kernelContext_5->tileA_0)[ak_0][ar_0] = _S28;
        uint br_0 = _S25 / 64U;
        uint bc_0 = _S25 % 64U;
        uint hk_0 = k0_0 + br_0;
        uint gc_0 = _S21 + bc_0;
        uint bi_0;
        if((kernelContext_5->p_0->transB_0) != 0U)
        {
            bi_0 = gc_0 * pr_0->ldb_0 + hk_0;
        }
        else
        {
            bi_0 = hk_0 * pr_0->ldb_0 + gc_0;
        }
        bool _S30;
        if(hk_0 < _S24)
        {
            _S30 = gc_0 < (pr_0->n_0);
        }
        else
        {
            _S30 = false;
        }
        float _S31;
        if(_S30)
        {
            float _S32 = loadB_0(_S23 + bi_0, kernelContext_5);
            _S31 = _S32;
        }
        else
        {
            _S31 = 0.0;
        }
        (*kernelContext_5->tileB_0)[br_0][bc_0] = _S31;
        uint e_3 = _S25 + 1U;
        uint ar_1 = e_3 / 16U;
        uint ak_1 = e_3 % 16U;
        uint gr_1 = _S20 + ar_1;
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
        bool _S33;
        if(gr_1 < _S26)
        {
            _S33 = gk_1 < _S24;
        }
        else
        {
            _S33 = false;
        }
        float _S34;
        if(_S33)
        {
            float _S35 = loadA_0(_S22 + ai_1, kernelContext_5);
            _S34 = _S35;
        }
        else
        {
            _S34 = 0.0;
        }
        (*kernelContext_5->tileA_0)[ak_1][ar_1] = _S34;
        uint br_1 = e_3 / 64U;
        uint bc_1 = e_3 % 64U;
        uint hk_1 = k0_0 + br_1;
        uint gc_1 = _S21 + bc_1;
        uint bi_1;
        if((kernelContext_5->p_0->transB_0) != 0U)
        {
            bi_1 = gc_1 * pr_0->ldb_0 + hk_1;
        }
        else
        {
            bi_1 = hk_1 * pr_0->ldb_0 + gc_1;
        }
        bool _S36;
        if(hk_1 < _S24)
        {
            _S36 = gc_1 < (pr_0->n_0);
        }
        else
        {
            _S36 = false;
        }
        float _S37;
        if(_S36)
        {
            float _S38 = loadB_0(_S23 + bi_1, kernelContext_5);
            _S37 = _S38;
        }
        else
        {
            _S37 = 0.0;
        }
        (*kernelContext_5->tileB_0)[br_1][bc_1] = _S37;
        uint e_4 = _S25 + 2U;
        uint ar_2 = e_4 / 16U;
        uint ak_2 = e_4 % 16U;
        uint gr_2 = _S20 + ar_2;
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
        bool _S39;
        if(gr_2 < _S26)
        {
            _S39 = gk_2 < _S24;
        }
        else
        {
            _S39 = false;
        }
        float _S40;
        if(_S39)
        {
            float _S41 = loadA_0(_S22 + ai_2, kernelContext_5);
            _S40 = _S41;
        }
        else
        {
            _S40 = 0.0;
        }
        (*kernelContext_5->tileA_0)[ak_2][ar_2] = _S40;
        uint br_2 = e_4 / 64U;
        uint bc_2 = e_4 % 64U;
        uint hk_2 = k0_0 + br_2;
        uint gc_2 = _S21 + bc_2;
        uint bi_2;
        if((kernelContext_5->p_0->transB_0) != 0U)
        {
            bi_2 = gc_2 * pr_0->ldb_0 + hk_2;
        }
        else
        {
            bi_2 = hk_2 * pr_0->ldb_0 + gc_2;
        }
        bool _S42;
        if(hk_2 < _S24)
        {
            _S42 = gc_2 < (pr_0->n_0);
        }
        else
        {
            _S42 = false;
        }
        float _S43;
        if(_S42)
        {
            float _S44 = loadB_0(_S23 + bi_2, kernelContext_5);
            _S43 = _S44;
        }
        else
        {
            _S43 = 0.0;
        }
        (*kernelContext_5->tileB_0)[br_2][bc_2] = _S43;
        uint e_5 = _S25 + 3U;
        uint ar_3 = e_5 / 16U;
        uint ak_3 = e_5 % 16U;
        uint gr_3 = _S20 + ar_3;
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
        bool _S45;
        if(gr_3 < _S26)
        {
            _S45 = gk_3 < _S24;
        }
        else
        {
            _S45 = false;
        }
        float _S46;
        if(_S45)
        {
            float _S47 = loadA_0(_S22 + ai_3, kernelContext_5);
            _S46 = _S47;
        }
        else
        {
            _S46 = 0.0;
        }
        (*kernelContext_5->tileA_0)[ak_3][ar_3] = _S46;
        uint br_3 = e_5 / 64U;
        uint bc_3 = e_5 % 64U;
        uint hk_3 = k0_0 + br_3;
        uint gc_3 = _S21 + bc_3;
        uint bi_3;
        if((kernelContext_5->p_0->transB_0) != 0U)
        {
            bi_3 = gc_3 * pr_0->ldb_0 + hk_3;
        }
        else
        {
            bi_3 = hk_3 * pr_0->ldb_0 + gc_3;
        }
        bool _S48;
        if(hk_3 < _S24)
        {
            _S48 = gc_3 < (pr_0->n_0);
        }
        else
        {
            _S48 = false;
        }
        float _S49;
        if(_S48)
        {
            float _S50 = loadB_0(_S23 + bi_3, kernelContext_5);
            _S49 = _S50;
        }
        else
        {
            _S49 = 0.0;
        }
        (*kernelContext_5->tileB_0)[br_3][bc_3] = _S49;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        thread array<float, int(4)> av_0;
        float _S51 = (*kernelContext_5->tileA_0)[0U][ty_0];
        av_0[0U] = (*kernelContext_5->tileA_0)[0U][ty_0];
        uint _S52 = ty_0 + 16U;
        float _S53 = (*kernelContext_5->tileA_0)[0U][_S52];
        av_0[1U] = (*kernelContext_5->tileA_0)[0U][_S52];
        uint _S54 = ty_0 + 32U;
        float _S55 = (*kernelContext_5->tileA_0)[0U][_S54];
        av_0[2U] = (*kernelContext_5->tileA_0)[0U][_S54];
        uint _S56 = ty_0 + 48U;
        float _S57 = (*kernelContext_5->tileA_0)[0U][_S56];
        av_0[3U] = (*kernelContext_5->tileA_0)[0U][_S56];
        thread array<float, int(4)> bv_0;
        float _S58 = (*kernelContext_5->tileB_0)[0U][tx_0];
        bv_0[0U] = (*kernelContext_5->tileB_0)[0U][tx_0];
        uint _S59 = tx_0 + 16U;
        float _S60 = (*kernelContext_5->tileB_0)[0U][_S59];
        bv_0[1U] = (*kernelContext_5->tileB_0)[0U][_S59];
        uint _S61 = tx_0 + 32U;
        float _S62 = (*kernelContext_5->tileB_0)[0U][_S61];
        bv_0[2U] = (*kernelContext_5->tileB_0)[0U][_S61];
        uint _S63 = tx_0 + 48U;
        float _S64 = (*kernelContext_5->tileB_0)[0U][_S63];
        bv_0[3U] = (*kernelContext_5->tileB_0)[0U][_S63];
        acc_1[0U][0U] = fma(_S51, _S58, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S51, _S60, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S51, _S62, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S51, _S64, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S53, _S58, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S53, _S60, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S53, _S62, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S53, _S64, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S55, _S58, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S55, _S60, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S55, _S62, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S55, _S64, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S57, _S58, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S57, _S60, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S57, _S62, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S57, _S64, acc_1[3U][3U]);
        thread array<float, int(4)> av_1;
        float _S65 = (*kernelContext_5->tileA_0)[1U][ty_0];
        av_1[0U] = (*kernelContext_5->tileA_0)[1U][ty_0];
        float _S66 = (*kernelContext_5->tileA_0)[1U][_S52];
        av_1[1U] = (*kernelContext_5->tileA_0)[1U][_S52];
        float _S67 = (*kernelContext_5->tileA_0)[1U][_S54];
        av_1[2U] = (*kernelContext_5->tileA_0)[1U][_S54];
        float _S68 = (*kernelContext_5->tileA_0)[1U][_S56];
        av_1[3U] = (*kernelContext_5->tileA_0)[1U][_S56];
        thread array<float, int(4)> bv_1;
        float _S69 = (*kernelContext_5->tileB_0)[1U][tx_0];
        bv_1[0U] = (*kernelContext_5->tileB_0)[1U][tx_0];
        float _S70 = (*kernelContext_5->tileB_0)[1U][_S59];
        bv_1[1U] = (*kernelContext_5->tileB_0)[1U][_S59];
        float _S71 = (*kernelContext_5->tileB_0)[1U][_S61];
        bv_1[2U] = (*kernelContext_5->tileB_0)[1U][_S61];
        float _S72 = (*kernelContext_5->tileB_0)[1U][_S63];
        bv_1[3U] = (*kernelContext_5->tileB_0)[1U][_S63];
        acc_1[0U][0U] = fma(_S65, _S69, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S65, _S70, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S65, _S71, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S65, _S72, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S66, _S69, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S66, _S70, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S66, _S71, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S66, _S72, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S67, _S69, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S67, _S70, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S67, _S71, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S67, _S72, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S68, _S69, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S68, _S70, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S68, _S71, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S68, _S72, acc_1[3U][3U]);
        thread array<float, int(4)> av_2;
        float _S73 = (*kernelContext_5->tileA_0)[2U][ty_0];
        av_2[0U] = (*kernelContext_5->tileA_0)[2U][ty_0];
        float _S74 = (*kernelContext_5->tileA_0)[2U][_S52];
        av_2[1U] = (*kernelContext_5->tileA_0)[2U][_S52];
        float _S75 = (*kernelContext_5->tileA_0)[2U][_S54];
        av_2[2U] = (*kernelContext_5->tileA_0)[2U][_S54];
        float _S76 = (*kernelContext_5->tileA_0)[2U][_S56];
        av_2[3U] = (*kernelContext_5->tileA_0)[2U][_S56];
        thread array<float, int(4)> bv_2;
        float _S77 = (*kernelContext_5->tileB_0)[2U][tx_0];
        bv_2[0U] = (*kernelContext_5->tileB_0)[2U][tx_0];
        float _S78 = (*kernelContext_5->tileB_0)[2U][_S59];
        bv_2[1U] = (*kernelContext_5->tileB_0)[2U][_S59];
        float _S79 = (*kernelContext_5->tileB_0)[2U][_S61];
        bv_2[2U] = (*kernelContext_5->tileB_0)[2U][_S61];
        float _S80 = (*kernelContext_5->tileB_0)[2U][_S63];
        bv_2[3U] = (*kernelContext_5->tileB_0)[2U][_S63];
        acc_1[0U][0U] = fma(_S73, _S77, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S73, _S78, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S73, _S79, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S73, _S80, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S74, _S77, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S74, _S78, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S74, _S79, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S74, _S80, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S75, _S77, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S75, _S78, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S75, _S79, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S75, _S80, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S76, _S77, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S76, _S78, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S76, _S79, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S76, _S80, acc_1[3U][3U]);
        thread array<float, int(4)> av_3;
        float _S81 = (*kernelContext_5->tileA_0)[3U][ty_0];
        av_3[0U] = (*kernelContext_5->tileA_0)[3U][ty_0];
        float _S82 = (*kernelContext_5->tileA_0)[3U][_S52];
        av_3[1U] = (*kernelContext_5->tileA_0)[3U][_S52];
        float _S83 = (*kernelContext_5->tileA_0)[3U][_S54];
        av_3[2U] = (*kernelContext_5->tileA_0)[3U][_S54];
        float _S84 = (*kernelContext_5->tileA_0)[3U][_S56];
        av_3[3U] = (*kernelContext_5->tileA_0)[3U][_S56];
        thread array<float, int(4)> bv_3;
        float _S85 = (*kernelContext_5->tileB_0)[3U][tx_0];
        bv_3[0U] = (*kernelContext_5->tileB_0)[3U][tx_0];
        float _S86 = (*kernelContext_5->tileB_0)[3U][_S59];
        bv_3[1U] = (*kernelContext_5->tileB_0)[3U][_S59];
        float _S87 = (*kernelContext_5->tileB_0)[3U][_S61];
        bv_3[2U] = (*kernelContext_5->tileB_0)[3U][_S61];
        float _S88 = (*kernelContext_5->tileB_0)[3U][_S63];
        bv_3[3U] = (*kernelContext_5->tileB_0)[3U][_S63];
        acc_1[0U][0U] = fma(_S81, _S85, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S81, _S86, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S81, _S87, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S81, _S88, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S82, _S85, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S82, _S86, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S82, _S87, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S82, _S88, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S83, _S85, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S83, _S86, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S83, _S87, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S83, _S88, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S84, _S85, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S84, _S86, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S84, _S87, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S84, _S88, acc_1[3U][3U]);
        thread array<float, int(4)> av_4;
        float _S89 = (*kernelContext_5->tileA_0)[4U][ty_0];
        av_4[0U] = (*kernelContext_5->tileA_0)[4U][ty_0];
        float _S90 = (*kernelContext_5->tileA_0)[4U][_S52];
        av_4[1U] = (*kernelContext_5->tileA_0)[4U][_S52];
        float _S91 = (*kernelContext_5->tileA_0)[4U][_S54];
        av_4[2U] = (*kernelContext_5->tileA_0)[4U][_S54];
        float _S92 = (*kernelContext_5->tileA_0)[4U][_S56];
        av_4[3U] = (*kernelContext_5->tileA_0)[4U][_S56];
        thread array<float, int(4)> bv_4;
        float _S93 = (*kernelContext_5->tileB_0)[4U][tx_0];
        bv_4[0U] = (*kernelContext_5->tileB_0)[4U][tx_0];
        float _S94 = (*kernelContext_5->tileB_0)[4U][_S59];
        bv_4[1U] = (*kernelContext_5->tileB_0)[4U][_S59];
        float _S95 = (*kernelContext_5->tileB_0)[4U][_S61];
        bv_4[2U] = (*kernelContext_5->tileB_0)[4U][_S61];
        float _S96 = (*kernelContext_5->tileB_0)[4U][_S63];
        bv_4[3U] = (*kernelContext_5->tileB_0)[4U][_S63];
        acc_1[0U][0U] = fma(_S89, _S93, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S89, _S94, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S89, _S95, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S89, _S96, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S90, _S93, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S90, _S94, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S90, _S95, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S90, _S96, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S91, _S93, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S91, _S94, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S91, _S95, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S91, _S96, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S92, _S93, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S92, _S94, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S92, _S95, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S92, _S96, acc_1[3U][3U]);
        thread array<float, int(4)> av_5;
        float _S97 = (*kernelContext_5->tileA_0)[5U][ty_0];
        av_5[0U] = (*kernelContext_5->tileA_0)[5U][ty_0];
        float _S98 = (*kernelContext_5->tileA_0)[5U][_S52];
        av_5[1U] = (*kernelContext_5->tileA_0)[5U][_S52];
        float _S99 = (*kernelContext_5->tileA_0)[5U][_S54];
        av_5[2U] = (*kernelContext_5->tileA_0)[5U][_S54];
        float _S100 = (*kernelContext_5->tileA_0)[5U][_S56];
        av_5[3U] = (*kernelContext_5->tileA_0)[5U][_S56];
        thread array<float, int(4)> bv_5;
        float _S101 = (*kernelContext_5->tileB_0)[5U][tx_0];
        bv_5[0U] = (*kernelContext_5->tileB_0)[5U][tx_0];
        float _S102 = (*kernelContext_5->tileB_0)[5U][_S59];
        bv_5[1U] = (*kernelContext_5->tileB_0)[5U][_S59];
        float _S103 = (*kernelContext_5->tileB_0)[5U][_S61];
        bv_5[2U] = (*kernelContext_5->tileB_0)[5U][_S61];
        float _S104 = (*kernelContext_5->tileB_0)[5U][_S63];
        bv_5[3U] = (*kernelContext_5->tileB_0)[5U][_S63];
        acc_1[0U][0U] = fma(_S97, _S101, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S97, _S102, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S97, _S103, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S97, _S104, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S98, _S101, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S98, _S102, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S98, _S103, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S98, _S104, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S99, _S101, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S99, _S102, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S99, _S103, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S99, _S104, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S100, _S101, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S100, _S102, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S100, _S103, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S100, _S104, acc_1[3U][3U]);
        thread array<float, int(4)> av_6;
        float _S105 = (*kernelContext_5->tileA_0)[6U][ty_0];
        av_6[0U] = (*kernelContext_5->tileA_0)[6U][ty_0];
        float _S106 = (*kernelContext_5->tileA_0)[6U][_S52];
        av_6[1U] = (*kernelContext_5->tileA_0)[6U][_S52];
        float _S107 = (*kernelContext_5->tileA_0)[6U][_S54];
        av_6[2U] = (*kernelContext_5->tileA_0)[6U][_S54];
        float _S108 = (*kernelContext_5->tileA_0)[6U][_S56];
        av_6[3U] = (*kernelContext_5->tileA_0)[6U][_S56];
        thread array<float, int(4)> bv_6;
        float _S109 = (*kernelContext_5->tileB_0)[6U][tx_0];
        bv_6[0U] = (*kernelContext_5->tileB_0)[6U][tx_0];
        float _S110 = (*kernelContext_5->tileB_0)[6U][_S59];
        bv_6[1U] = (*kernelContext_5->tileB_0)[6U][_S59];
        float _S111 = (*kernelContext_5->tileB_0)[6U][_S61];
        bv_6[2U] = (*kernelContext_5->tileB_0)[6U][_S61];
        float _S112 = (*kernelContext_5->tileB_0)[6U][_S63];
        bv_6[3U] = (*kernelContext_5->tileB_0)[6U][_S63];
        acc_1[0U][0U] = fma(_S105, _S109, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S105, _S110, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S105, _S111, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S105, _S112, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S106, _S109, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S106, _S110, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S106, _S111, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S106, _S112, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S107, _S109, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S107, _S110, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S107, _S111, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S107, _S112, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S108, _S109, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S108, _S110, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S108, _S111, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S108, _S112, acc_1[3U][3U]);
        thread array<float, int(4)> av_7;
        float _S113 = (*kernelContext_5->tileA_0)[7U][ty_0];
        av_7[0U] = (*kernelContext_5->tileA_0)[7U][ty_0];
        float _S114 = (*kernelContext_5->tileA_0)[7U][_S52];
        av_7[1U] = (*kernelContext_5->tileA_0)[7U][_S52];
        float _S115 = (*kernelContext_5->tileA_0)[7U][_S54];
        av_7[2U] = (*kernelContext_5->tileA_0)[7U][_S54];
        float _S116 = (*kernelContext_5->tileA_0)[7U][_S56];
        av_7[3U] = (*kernelContext_5->tileA_0)[7U][_S56];
        thread array<float, int(4)> bv_7;
        float _S117 = (*kernelContext_5->tileB_0)[7U][tx_0];
        bv_7[0U] = (*kernelContext_5->tileB_0)[7U][tx_0];
        float _S118 = (*kernelContext_5->tileB_0)[7U][_S59];
        bv_7[1U] = (*kernelContext_5->tileB_0)[7U][_S59];
        float _S119 = (*kernelContext_5->tileB_0)[7U][_S61];
        bv_7[2U] = (*kernelContext_5->tileB_0)[7U][_S61];
        float _S120 = (*kernelContext_5->tileB_0)[7U][_S63];
        bv_7[3U] = (*kernelContext_5->tileB_0)[7U][_S63];
        acc_1[0U][0U] = fma(_S113, _S117, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S113, _S118, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S113, _S119, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S113, _S120, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S114, _S117, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S114, _S118, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S114, _S119, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S114, _S120, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S115, _S117, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S115, _S118, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S115, _S119, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S115, _S120, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S116, _S117, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S116, _S118, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S116, _S119, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S116, _S120, acc_1[3U][3U]);
        thread array<float, int(4)> av_8;
        float _S121 = (*kernelContext_5->tileA_0)[8U][ty_0];
        av_8[0U] = (*kernelContext_5->tileA_0)[8U][ty_0];
        float _S122 = (*kernelContext_5->tileA_0)[8U][_S52];
        av_8[1U] = (*kernelContext_5->tileA_0)[8U][_S52];
        float _S123 = (*kernelContext_5->tileA_0)[8U][_S54];
        av_8[2U] = (*kernelContext_5->tileA_0)[8U][_S54];
        float _S124 = (*kernelContext_5->tileA_0)[8U][_S56];
        av_8[3U] = (*kernelContext_5->tileA_0)[8U][_S56];
        thread array<float, int(4)> bv_8;
        float _S125 = (*kernelContext_5->tileB_0)[8U][tx_0];
        bv_8[0U] = (*kernelContext_5->tileB_0)[8U][tx_0];
        float _S126 = (*kernelContext_5->tileB_0)[8U][_S59];
        bv_8[1U] = (*kernelContext_5->tileB_0)[8U][_S59];
        float _S127 = (*kernelContext_5->tileB_0)[8U][_S61];
        bv_8[2U] = (*kernelContext_5->tileB_0)[8U][_S61];
        float _S128 = (*kernelContext_5->tileB_0)[8U][_S63];
        bv_8[3U] = (*kernelContext_5->tileB_0)[8U][_S63];
        acc_1[0U][0U] = fma(_S121, _S125, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S121, _S126, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S121, _S127, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S121, _S128, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S122, _S125, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S122, _S126, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S122, _S127, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S122, _S128, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S123, _S125, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S123, _S126, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S123, _S127, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S123, _S128, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S124, _S125, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S124, _S126, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S124, _S127, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S124, _S128, acc_1[3U][3U]);
        thread array<float, int(4)> av_9;
        float _S129 = (*kernelContext_5->tileA_0)[9U][ty_0];
        av_9[0U] = (*kernelContext_5->tileA_0)[9U][ty_0];
        float _S130 = (*kernelContext_5->tileA_0)[9U][_S52];
        av_9[1U] = (*kernelContext_5->tileA_0)[9U][_S52];
        float _S131 = (*kernelContext_5->tileA_0)[9U][_S54];
        av_9[2U] = (*kernelContext_5->tileA_0)[9U][_S54];
        float _S132 = (*kernelContext_5->tileA_0)[9U][_S56];
        av_9[3U] = (*kernelContext_5->tileA_0)[9U][_S56];
        thread array<float, int(4)> bv_9;
        float _S133 = (*kernelContext_5->tileB_0)[9U][tx_0];
        bv_9[0U] = (*kernelContext_5->tileB_0)[9U][tx_0];
        float _S134 = (*kernelContext_5->tileB_0)[9U][_S59];
        bv_9[1U] = (*kernelContext_5->tileB_0)[9U][_S59];
        float _S135 = (*kernelContext_5->tileB_0)[9U][_S61];
        bv_9[2U] = (*kernelContext_5->tileB_0)[9U][_S61];
        float _S136 = (*kernelContext_5->tileB_0)[9U][_S63];
        bv_9[3U] = (*kernelContext_5->tileB_0)[9U][_S63];
        acc_1[0U][0U] = fma(_S129, _S133, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S129, _S134, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S129, _S135, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S129, _S136, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S130, _S133, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S130, _S134, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S130, _S135, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S130, _S136, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S131, _S133, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S131, _S134, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S131, _S135, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S131, _S136, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S132, _S133, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S132, _S134, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S132, _S135, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S132, _S136, acc_1[3U][3U]);
        thread array<float, int(4)> av_10;
        float _S137 = (*kernelContext_5->tileA_0)[10U][ty_0];
        av_10[0U] = (*kernelContext_5->tileA_0)[10U][ty_0];
        float _S138 = (*kernelContext_5->tileA_0)[10U][_S52];
        av_10[1U] = (*kernelContext_5->tileA_0)[10U][_S52];
        float _S139 = (*kernelContext_5->tileA_0)[10U][_S54];
        av_10[2U] = (*kernelContext_5->tileA_0)[10U][_S54];
        float _S140 = (*kernelContext_5->tileA_0)[10U][_S56];
        av_10[3U] = (*kernelContext_5->tileA_0)[10U][_S56];
        thread array<float, int(4)> bv_10;
        float _S141 = (*kernelContext_5->tileB_0)[10U][tx_0];
        bv_10[0U] = (*kernelContext_5->tileB_0)[10U][tx_0];
        float _S142 = (*kernelContext_5->tileB_0)[10U][_S59];
        bv_10[1U] = (*kernelContext_5->tileB_0)[10U][_S59];
        float _S143 = (*kernelContext_5->tileB_0)[10U][_S61];
        bv_10[2U] = (*kernelContext_5->tileB_0)[10U][_S61];
        float _S144 = (*kernelContext_5->tileB_0)[10U][_S63];
        bv_10[3U] = (*kernelContext_5->tileB_0)[10U][_S63];
        acc_1[0U][0U] = fma(_S137, _S141, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S137, _S142, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S137, _S143, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S137, _S144, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S138, _S141, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S138, _S142, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S138, _S143, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S138, _S144, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S139, _S141, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S139, _S142, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S139, _S143, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S139, _S144, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S140, _S141, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S140, _S142, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S140, _S143, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S140, _S144, acc_1[3U][3U]);
        thread array<float, int(4)> av_11;
        float _S145 = (*kernelContext_5->tileA_0)[11U][ty_0];
        av_11[0U] = (*kernelContext_5->tileA_0)[11U][ty_0];
        float _S146 = (*kernelContext_5->tileA_0)[11U][_S52];
        av_11[1U] = (*kernelContext_5->tileA_0)[11U][_S52];
        float _S147 = (*kernelContext_5->tileA_0)[11U][_S54];
        av_11[2U] = (*kernelContext_5->tileA_0)[11U][_S54];
        float _S148 = (*kernelContext_5->tileA_0)[11U][_S56];
        av_11[3U] = (*kernelContext_5->tileA_0)[11U][_S56];
        thread array<float, int(4)> bv_11;
        float _S149 = (*kernelContext_5->tileB_0)[11U][tx_0];
        bv_11[0U] = (*kernelContext_5->tileB_0)[11U][tx_0];
        float _S150 = (*kernelContext_5->tileB_0)[11U][_S59];
        bv_11[1U] = (*kernelContext_5->tileB_0)[11U][_S59];
        float _S151 = (*kernelContext_5->tileB_0)[11U][_S61];
        bv_11[2U] = (*kernelContext_5->tileB_0)[11U][_S61];
        float _S152 = (*kernelContext_5->tileB_0)[11U][_S63];
        bv_11[3U] = (*kernelContext_5->tileB_0)[11U][_S63];
        acc_1[0U][0U] = fma(_S145, _S149, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S145, _S150, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S145, _S151, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S145, _S152, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S146, _S149, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S146, _S150, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S146, _S151, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S146, _S152, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S147, _S149, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S147, _S150, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S147, _S151, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S147, _S152, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S148, _S149, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S148, _S150, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S148, _S151, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S148, _S152, acc_1[3U][3U]);
        thread array<float, int(4)> av_12;
        float _S153 = (*kernelContext_5->tileA_0)[12U][ty_0];
        av_12[0U] = (*kernelContext_5->tileA_0)[12U][ty_0];
        float _S154 = (*kernelContext_5->tileA_0)[12U][_S52];
        av_12[1U] = (*kernelContext_5->tileA_0)[12U][_S52];
        float _S155 = (*kernelContext_5->tileA_0)[12U][_S54];
        av_12[2U] = (*kernelContext_5->tileA_0)[12U][_S54];
        float _S156 = (*kernelContext_5->tileA_0)[12U][_S56];
        av_12[3U] = (*kernelContext_5->tileA_0)[12U][_S56];
        thread array<float, int(4)> bv_12;
        float _S157 = (*kernelContext_5->tileB_0)[12U][tx_0];
        bv_12[0U] = (*kernelContext_5->tileB_0)[12U][tx_0];
        float _S158 = (*kernelContext_5->tileB_0)[12U][_S59];
        bv_12[1U] = (*kernelContext_5->tileB_0)[12U][_S59];
        float _S159 = (*kernelContext_5->tileB_0)[12U][_S61];
        bv_12[2U] = (*kernelContext_5->tileB_0)[12U][_S61];
        float _S160 = (*kernelContext_5->tileB_0)[12U][_S63];
        bv_12[3U] = (*kernelContext_5->tileB_0)[12U][_S63];
        acc_1[0U][0U] = fma(_S153, _S157, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S153, _S158, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S153, _S159, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S153, _S160, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S154, _S157, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S154, _S158, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S154, _S159, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S154, _S160, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S155, _S157, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S155, _S158, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S155, _S159, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S155, _S160, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S156, _S157, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S156, _S158, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S156, _S159, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S156, _S160, acc_1[3U][3U]);
        thread array<float, int(4)> av_13;
        float _S161 = (*kernelContext_5->tileA_0)[13U][ty_0];
        av_13[0U] = (*kernelContext_5->tileA_0)[13U][ty_0];
        float _S162 = (*kernelContext_5->tileA_0)[13U][_S52];
        av_13[1U] = (*kernelContext_5->tileA_0)[13U][_S52];
        float _S163 = (*kernelContext_5->tileA_0)[13U][_S54];
        av_13[2U] = (*kernelContext_5->tileA_0)[13U][_S54];
        float _S164 = (*kernelContext_5->tileA_0)[13U][_S56];
        av_13[3U] = (*kernelContext_5->tileA_0)[13U][_S56];
        thread array<float, int(4)> bv_13;
        float _S165 = (*kernelContext_5->tileB_0)[13U][tx_0];
        bv_13[0U] = (*kernelContext_5->tileB_0)[13U][tx_0];
        float _S166 = (*kernelContext_5->tileB_0)[13U][_S59];
        bv_13[1U] = (*kernelContext_5->tileB_0)[13U][_S59];
        float _S167 = (*kernelContext_5->tileB_0)[13U][_S61];
        bv_13[2U] = (*kernelContext_5->tileB_0)[13U][_S61];
        float _S168 = (*kernelContext_5->tileB_0)[13U][_S63];
        bv_13[3U] = (*kernelContext_5->tileB_0)[13U][_S63];
        acc_1[0U][0U] = fma(_S161, _S165, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S161, _S166, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S161, _S167, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S161, _S168, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S162, _S165, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S162, _S166, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S162, _S167, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S162, _S168, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S163, _S165, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S163, _S166, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S163, _S167, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S163, _S168, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S164, _S165, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S164, _S166, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S164, _S167, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S164, _S168, acc_1[3U][3U]);
        thread array<float, int(4)> av_14;
        float _S169 = (*kernelContext_5->tileA_0)[14U][ty_0];
        av_14[0U] = (*kernelContext_5->tileA_0)[14U][ty_0];
        float _S170 = (*kernelContext_5->tileA_0)[14U][_S52];
        av_14[1U] = (*kernelContext_5->tileA_0)[14U][_S52];
        float _S171 = (*kernelContext_5->tileA_0)[14U][_S54];
        av_14[2U] = (*kernelContext_5->tileA_0)[14U][_S54];
        float _S172 = (*kernelContext_5->tileA_0)[14U][_S56];
        av_14[3U] = (*kernelContext_5->tileA_0)[14U][_S56];
        thread array<float, int(4)> bv_14;
        float _S173 = (*kernelContext_5->tileB_0)[14U][tx_0];
        bv_14[0U] = (*kernelContext_5->tileB_0)[14U][tx_0];
        float _S174 = (*kernelContext_5->tileB_0)[14U][_S59];
        bv_14[1U] = (*kernelContext_5->tileB_0)[14U][_S59];
        float _S175 = (*kernelContext_5->tileB_0)[14U][_S61];
        bv_14[2U] = (*kernelContext_5->tileB_0)[14U][_S61];
        float _S176 = (*kernelContext_5->tileB_0)[14U][_S63];
        bv_14[3U] = (*kernelContext_5->tileB_0)[14U][_S63];
        acc_1[0U][0U] = fma(_S169, _S173, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S169, _S174, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S169, _S175, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S169, _S176, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S170, _S173, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S170, _S174, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S170, _S175, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S170, _S176, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S171, _S173, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S171, _S174, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S171, _S175, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S171, _S176, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S172, _S173, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S172, _S174, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S172, _S175, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S172, _S176, acc_1[3U][3U]);
        thread array<float, int(4)> av_15;
        float _S177 = (*kernelContext_5->tileA_0)[15U][ty_0];
        av_15[0U] = (*kernelContext_5->tileA_0)[15U][ty_0];
        float _S178 = (*kernelContext_5->tileA_0)[15U][_S52];
        av_15[1U] = (*kernelContext_5->tileA_0)[15U][_S52];
        float _S179 = (*kernelContext_5->tileA_0)[15U][_S54];
        av_15[2U] = (*kernelContext_5->tileA_0)[15U][_S54];
        float _S180 = (*kernelContext_5->tileA_0)[15U][_S56];
        av_15[3U] = (*kernelContext_5->tileA_0)[15U][_S56];
        thread array<float, int(4)> bv_15;
        float _S181 = (*kernelContext_5->tileB_0)[15U][tx_0];
        bv_15[0U] = (*kernelContext_5->tileB_0)[15U][tx_0];
        float _S182 = (*kernelContext_5->tileB_0)[15U][_S59];
        bv_15[1U] = (*kernelContext_5->tileB_0)[15U][_S59];
        float _S183 = (*kernelContext_5->tileB_0)[15U][_S61];
        bv_15[2U] = (*kernelContext_5->tileB_0)[15U][_S61];
        float _S184 = (*kernelContext_5->tileB_0)[15U][_S63];
        bv_15[3U] = (*kernelContext_5->tileB_0)[15U][_S63];
        acc_1[0U][0U] = fma(_S177, _S181, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S177, _S182, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S177, _S183, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S177, _S184, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S178, _S181, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S178, _S182, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S178, _S183, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S178, _S184, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S179, _S181, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S179, _S182, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S179, _S183, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S179, _S184, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S180, _S181, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S180, _S182, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S180, _S183, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S180, _S184, acc_1[3U][3U]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k0_0 = k0_0 + 16U;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                uint _S185 = _S20 + ty_0;
                _S9 = _S185;
                _S10 = pr_0->m_0;
                if(_S185 >= (pr_0->m_0))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S186 = _S21 + tx_0;
                            _S11 = _S186;
                            _S12 = pr_0->n_0;
                            if(_S186 >= (pr_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S185, _S186, acc_1[0U][0U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_0 = _S11 + 16U;
                            if(col_0 >= _S12)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S185, col_0, acc_1[0U][1U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_1 = _S11 + 32U;
                            if(col_1 >= _S12)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S185, col_1, acc_1[0U][2U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_2 = _S11 + 48U;
                            if(col_2 >= _S12)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S185, col_2, acc_1[0U][3U], kernelContext_5);
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
                uint r_2 = _S9 + 16U;
                if(r_2 >= _S10)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S187 = _S21 + tx_0;
                            _S13 = _S187;
                            _S14 = pr_0->n_0;
                            if(_S187 >= (pr_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, _S187, acc_1[1U][0U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_3 = _S13 + 16U;
                            if(col_3 >= _S14)
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
                            uint col_4 = _S13 + 32U;
                            if(col_4 >= _S14)
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
                            uint col_5 = _S13 + 48U;
                            if(col_5 >= _S14)
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
                uint r_3 = _S9 + 32U;
                if(r_3 >= _S10)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S188 = _S21 + tx_0;
                            _S15 = _S188;
                            _S16 = pr_0->n_0;
                            if(_S188 >= (pr_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, _S188, acc_1[2U][0U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_6 = _S15 + 16U;
                            if(col_6 >= _S16)
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
                            uint col_7 = _S15 + 32U;
                            if(col_7 >= _S16)
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
                            uint col_8 = _S15 + 48U;
                            if(col_8 >= _S16)
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
                uint r_4 = _S9 + 48U;
                if(r_4 >= _S10)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S189 = _S21 + tx_0;
                            _S17 = _S189;
                            _S18 = pr_0->n_0;
                            if(_S189 >= (pr_0->n_0))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, _S189, acc_1[3U][0U], kernelContext_5);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_9 = _S17 + 16U;
                            if(col_9 >= _S18)
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
                            uint col_10 = _S17 + 32U;
                            if(col_10 >= _S18)
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
                            uint col_11 = _S17 + 48U;
                            if(col_11 >= _S18)
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

[[kernel]] void gemm_bf16(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], uint device* a_2 [[buffer(0)]], uint device* b_2 [[buffer(1)]], float device* c_1 [[buffer(2)]], float device* epBias_1 [[buffer(3)]], float device* epGate_1 [[buffer(4)]], float device* epResidual_1 [[buffer(5)]])
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
    GemmProblem_0 _S190 = gemmBatchProblem_0(group_0.z, &kernelContext_6);
    uint2 _S191 = group_0.xy;
    thread GemmProblem_0 _S192 = _S190;
    gemmTile_0(&_S192, _S191, local_1, &kernelContext_6);
    return;
}

