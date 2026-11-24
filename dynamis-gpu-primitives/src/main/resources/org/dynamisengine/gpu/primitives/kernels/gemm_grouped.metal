#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
bool fits_0(uint off_0, uint rows_0, uint rowLength_0, uint ld_0, uint length_0)
{
    bool _S1;
    if(rows_0 == 0U)
    {
        _S1 = true;
    }
    else
    {
        _S1 = rowLength_0 == 0U;
    }
    if(_S1)
    {
        return true;
    }
    if(ld_0 < rowLength_0)
    {
        _S1 = true;
    }
    else
    {
        _S1 = off_0 > length_0;
    }
    if(_S1)
    {
        _S1 = true;
    }
    else
    {
        _S1 = rowLength_0 > (length_0 - off_0);
    }
    if(_S1)
    {
        return false;
    }
    uint _S2 = rows_0 - 1U;
    uint _S3 = (length_0 - off_0 - rowLength_0) / ld_0;
    return _S2 <= _S3;
}

void dg_report_error_0(uint device* errors_0, uint code_0, uint index_0)
{
    uint _S4 = atomic_fetch_or_explicit(((atomic_uint device*)(errors_0+int(0))), code_0, memory_order_relaxed);
    uint _S5 = atomic_fetch_max_explicit(((atomic_uint device*)(errors_0+int(1))), ~index_0, memory_order_relaxed);
    uint _S6 = atomic_fetch_add_explicit(((atomic_uint device*)(errors_0+int(2))), 1U, memory_order_relaxed);
    return;
}

bool dg_is_nan_0(float x_0)
{
    uint _S7 = (as_type<uint>((x_0)));
    bool _S8;
    if((_S7 & 2139095040U) == 2139095040U)
    {
        _S8 = (_S7 & 8388607U) != 0U;
    }
    else
    {
        _S8 = false;
    }
    return _S8;
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
    bool _S9;
    if(a_0 == 0.0)
    {
        _S9 = b_0 == 0.0;
    }
    else
    {
        _S9 = false;
    }
    if(_S9)
    {
        return (as_type<float>(((as_type<uint>((a_0))) & (as_type<uint>((b_0))))));
    }
    float _S10;
    if(a_0 > b_0)
    {
        _S10 = a_0;
    }
    else
    {
        _S10 = b_0;
    }
    return _S10;
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
    uint count_0;
    uint aLength_0;
    uint bLength_0;
    uint cLength_0;
    uint biasLength_0;
};

struct ProblemRecord_0
{
    uint m_1;
    uint n_1;
    uint k_1;
    uint lda_1;
    uint ldb_1;
    uint ldc_1;
    uint offsetA_0;
    uint offsetB_0;
    uint offsetC_0;
    uint offsetBias_0;
    uint reserved0_0;
    uint reserved1_0;
};

struct KernelContext_0
{
    GemmParams_0 constant* p_0;
    ProblemRecord_0 device* problems_0;
    uint device* errors_1;
    half device* a_1;
    half device* b_1;
    float device* c_0;
    float device* epBias_0;
    float device* epGate_0;
    float device* epResidual_0;
    array<array<float, int(64)>, int(16)> threadgroup* tileA_0;
    array<array<float, int(64)>, int(16)> threadgroup* tileB_0;
};

float gemmEpilogue_0(float r_0, uint i_0, uint j_0, uint idx_0, uint biasBase_0, KernelContext_0 thread* kernelContext_0)
{
    float _S11;
    if((kernelContext_0->p_0->bias_0) == 1U)
    {
        _S11 = r_0 + *(kernelContext_0->epBias_0+(biasBase_0 + j_0));
    }
    else
    {
        if((kernelContext_0->p_0->bias_0) == 2U)
        {
            _S11 = r_0 + *(kernelContext_0->epBias_0+(biasBase_0 + i_0));
        }
        else
        {
            _S11 = r_0;
        }
    }
    if((kernelContext_0->p_0->act_0) == 1U)
    {
        _S11 = dg_max_0(_S11, 0.0);
    }
    else
    {
        if((kernelContext_0->p_0->act_0) == 2U)
        {
            _S11 = 0.5 * _S11 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S11 + 0.04471499845385551 * _S11 * _S11 * _S11))) + 1.0)));
        }
        else
        {
            if((kernelContext_0->p_0->act_0) == 3U)
            {
                _S11 = _S11 / (1.0 + exp(- _S11));
            }
        }
    }
    if((kernelContext_0->p_0->gated_0) != 0U)
    {
        _S11 = _S11 * *(kernelContext_0->epGate_0+idx_0);
    }
    if((kernelContext_0->p_0->residual_0) != 0U)
    {
        _S11 = _S11 + *(kernelContext_0->epResidual_0+idx_0);
    }
    return _S11;
}

struct GemmProblem_0
{
    uint m_2;
    uint n_2;
    uint k_2;
    uint lda_2;
    uint ldb_2;
    uint ldc_2;
    uint baseA_0;
    uint baseB_0;
    uint baseC_0;
    uint baseBias_0;
};

void gemmStore_0(const GemmProblem_0 thread* q_0, uint i_1, uint j_1, float acc_0, KernelContext_0 thread* kernelContext_1)
{
    uint idx_1 = q_0->baseC_0 + i_1 * q_0->ldc_2 + j_1;
    float r_1;
    if((kernelContext_1->p_0->beta_0) == 0.0)
    {
        r_1 = kernelContext_1->p_0->alpha_0 * acc_0;
    }
    else
    {
        r_1 = kernelContext_1->p_0->alpha_0 * acc_0 + kernelContext_1->p_0->beta_0 * *(kernelContext_1->c_0+idx_1);
    }
    float device* _S12 = kernelContext_1->c_0+idx_1;
    float _S13 = gemmEpilogue_0(r_1, i_1, j_1, idx_1, q_0->baseBias_0, kernelContext_1);
    *_S12 = _S13;
    return;
}

void gemmTile_0(const GemmProblem_0 thread* pr_0, uint2 tile_0, uint3 local_0, KernelContext_0 thread* kernelContext_2)
{
    uint _S14;
    uint _S15;
    uint _S16;
    uint _S17;
    uint _S18;
    uint _S19;
    uint _S20;
    uint _S21;
    uint _S22;
    uint _S23;
    uint tx_0 = local_0.x;
    uint ty_0 = local_0.y;
    uint _S24 = ty_0 * 16U + tx_0;
    uint _S25 = tile_0.y * 64U;
    uint _S26 = tile_0.x * 64U;
    uint _S27 = pr_0->baseA_0;
    uint _S28 = pr_0->baseB_0;
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
        uint _S29 = pr_0->k_2;
        if(k0_0 < (pr_0->k_2))
        {
        }
        else
        {
            break;
        }
        uint _S30 = _S24 * 4U;
        uint ar_0 = _S30 / 16U;
        uint ak_0 = _S30 % 16U;
        uint gr_0 = _S25 + ar_0;
        uint gk_0 = k0_0 + ak_0;
        uint ai_0;
        if((kernelContext_2->p_0->transA_0) != 0U)
        {
            ai_0 = gk_0 * pr_0->lda_2 + gr_0;
        }
        else
        {
            ai_0 = gr_0 * pr_0->lda_2 + gk_0;
        }
        uint _S31 = pr_0->m_2;
        bool _S32;
        if(gr_0 < (pr_0->m_2))
        {
            _S32 = gk_0 < _S29;
        }
        else
        {
            _S32 = false;
        }
        float _S33;
        if(_S32)
        {
            _S33 = float(*(kernelContext_2->a_1+(_S27 + ai_0)));
        }
        else
        {
            _S33 = 0.0;
        }
        (*kernelContext_2->tileA_0)[ak_0][ar_0] = _S33;
        uint br_0 = _S30 / 64U;
        uint bc_0 = _S30 % 64U;
        uint hk_0 = k0_0 + br_0;
        uint gc_0 = _S26 + bc_0;
        uint bi_0;
        if((kernelContext_2->p_0->transB_0) != 0U)
        {
            bi_0 = gc_0 * pr_0->ldb_2 + hk_0;
        }
        else
        {
            bi_0 = hk_0 * pr_0->ldb_2 + gc_0;
        }
        bool _S34;
        if(hk_0 < _S29)
        {
            _S34 = gc_0 < (pr_0->n_2);
        }
        else
        {
            _S34 = false;
        }
        float _S35;
        if(_S34)
        {
            _S35 = float(*(kernelContext_2->b_1+(_S28 + bi_0)));
        }
        else
        {
            _S35 = 0.0;
        }
        (*kernelContext_2->tileB_0)[br_0][bc_0] = _S35;
        uint e_0 = _S30 + 1U;
        uint ar_1 = e_0 / 16U;
        uint ak_1 = e_0 % 16U;
        uint gr_1 = _S25 + ar_1;
        uint gk_1 = k0_0 + ak_1;
        uint ai_1;
        if((kernelContext_2->p_0->transA_0) != 0U)
        {
            ai_1 = gk_1 * pr_0->lda_2 + gr_1;
        }
        else
        {
            ai_1 = gr_1 * pr_0->lda_2 + gk_1;
        }
        bool _S36;
        if(gr_1 < _S31)
        {
            _S36 = gk_1 < _S29;
        }
        else
        {
            _S36 = false;
        }
        float _S37;
        if(_S36)
        {
            _S37 = float(*(kernelContext_2->a_1+(_S27 + ai_1)));
        }
        else
        {
            _S37 = 0.0;
        }
        (*kernelContext_2->tileA_0)[ak_1][ar_1] = _S37;
        uint br_1 = e_0 / 64U;
        uint bc_1 = e_0 % 64U;
        uint hk_1 = k0_0 + br_1;
        uint gc_1 = _S26 + bc_1;
        uint bi_1;
        if((kernelContext_2->p_0->transB_0) != 0U)
        {
            bi_1 = gc_1 * pr_0->ldb_2 + hk_1;
        }
        else
        {
            bi_1 = hk_1 * pr_0->ldb_2 + gc_1;
        }
        bool _S38;
        if(hk_1 < _S29)
        {
            _S38 = gc_1 < (pr_0->n_2);
        }
        else
        {
            _S38 = false;
        }
        float _S39;
        if(_S38)
        {
            _S39 = float(*(kernelContext_2->b_1+(_S28 + bi_1)));
        }
        else
        {
            _S39 = 0.0;
        }
        (*kernelContext_2->tileB_0)[br_1][bc_1] = _S39;
        uint e_1 = _S30 + 2U;
        uint ar_2 = e_1 / 16U;
        uint ak_2 = e_1 % 16U;
        uint gr_2 = _S25 + ar_2;
        uint gk_2 = k0_0 + ak_2;
        uint ai_2;
        if((kernelContext_2->p_0->transA_0) != 0U)
        {
            ai_2 = gk_2 * pr_0->lda_2 + gr_2;
        }
        else
        {
            ai_2 = gr_2 * pr_0->lda_2 + gk_2;
        }
        bool _S40;
        if(gr_2 < _S31)
        {
            _S40 = gk_2 < _S29;
        }
        else
        {
            _S40 = false;
        }
        float _S41;
        if(_S40)
        {
            _S41 = float(*(kernelContext_2->a_1+(_S27 + ai_2)));
        }
        else
        {
            _S41 = 0.0;
        }
        (*kernelContext_2->tileA_0)[ak_2][ar_2] = _S41;
        uint br_2 = e_1 / 64U;
        uint bc_2 = e_1 % 64U;
        uint hk_2 = k0_0 + br_2;
        uint gc_2 = _S26 + bc_2;
        uint bi_2;
        if((kernelContext_2->p_0->transB_0) != 0U)
        {
            bi_2 = gc_2 * pr_0->ldb_2 + hk_2;
        }
        else
        {
            bi_2 = hk_2 * pr_0->ldb_2 + gc_2;
        }
        bool _S42;
        if(hk_2 < _S29)
        {
            _S42 = gc_2 < (pr_0->n_2);
        }
        else
        {
            _S42 = false;
        }
        float _S43;
        if(_S42)
        {
            _S43 = float(*(kernelContext_2->b_1+(_S28 + bi_2)));
        }
        else
        {
            _S43 = 0.0;
        }
        (*kernelContext_2->tileB_0)[br_2][bc_2] = _S43;
        uint e_2 = _S30 + 3U;
        uint ar_3 = e_2 / 16U;
        uint ak_3 = e_2 % 16U;
        uint gr_3 = _S25 + ar_3;
        uint gk_3 = k0_0 + ak_3;
        uint ai_3;
        if((kernelContext_2->p_0->transA_0) != 0U)
        {
            ai_3 = gk_3 * pr_0->lda_2 + gr_3;
        }
        else
        {
            ai_3 = gr_3 * pr_0->lda_2 + gk_3;
        }
        bool _S44;
        if(gr_3 < _S31)
        {
            _S44 = gk_3 < _S29;
        }
        else
        {
            _S44 = false;
        }
        float _S45;
        if(_S44)
        {
            _S45 = float(*(kernelContext_2->a_1+(_S27 + ai_3)));
        }
        else
        {
            _S45 = 0.0;
        }
        (*kernelContext_2->tileA_0)[ak_3][ar_3] = _S45;
        uint br_3 = e_2 / 64U;
        uint bc_3 = e_2 % 64U;
        uint hk_3 = k0_0 + br_3;
        uint gc_3 = _S26 + bc_3;
        uint bi_3;
        if((kernelContext_2->p_0->transB_0) != 0U)
        {
            bi_3 = gc_3 * pr_0->ldb_2 + hk_3;
        }
        else
        {
            bi_3 = hk_3 * pr_0->ldb_2 + gc_3;
        }
        bool _S46;
        if(hk_3 < _S29)
        {
            _S46 = gc_3 < (pr_0->n_2);
        }
        else
        {
            _S46 = false;
        }
        float _S47;
        if(_S46)
        {
            _S47 = float(*(kernelContext_2->b_1+(_S28 + bi_3)));
        }
        else
        {
            _S47 = 0.0;
        }
        (*kernelContext_2->tileB_0)[br_3][bc_3] = _S47;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        thread array<float, int(4)> av_0;
        float _S48 = (*kernelContext_2->tileA_0)[0U][ty_0];
        av_0[0U] = (*kernelContext_2->tileA_0)[0U][ty_0];
        uint _S49 = ty_0 + 16U;
        float _S50 = (*kernelContext_2->tileA_0)[0U][_S49];
        av_0[1U] = (*kernelContext_2->tileA_0)[0U][_S49];
        uint _S51 = ty_0 + 32U;
        float _S52 = (*kernelContext_2->tileA_0)[0U][_S51];
        av_0[2U] = (*kernelContext_2->tileA_0)[0U][_S51];
        uint _S53 = ty_0 + 48U;
        float _S54 = (*kernelContext_2->tileA_0)[0U][_S53];
        av_0[3U] = (*kernelContext_2->tileA_0)[0U][_S53];
        thread array<float, int(4)> bv_0;
        float _S55 = (*kernelContext_2->tileB_0)[0U][tx_0];
        bv_0[0U] = (*kernelContext_2->tileB_0)[0U][tx_0];
        uint _S56 = tx_0 + 16U;
        float _S57 = (*kernelContext_2->tileB_0)[0U][_S56];
        bv_0[1U] = (*kernelContext_2->tileB_0)[0U][_S56];
        uint _S58 = tx_0 + 32U;
        float _S59 = (*kernelContext_2->tileB_0)[0U][_S58];
        bv_0[2U] = (*kernelContext_2->tileB_0)[0U][_S58];
        uint _S60 = tx_0 + 48U;
        float _S61 = (*kernelContext_2->tileB_0)[0U][_S60];
        bv_0[3U] = (*kernelContext_2->tileB_0)[0U][_S60];
        acc_1[0U][0U] = fma(_S48, _S55, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S48, _S57, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S48, _S59, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S48, _S61, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S50, _S55, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S50, _S57, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S50, _S59, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S50, _S61, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S52, _S55, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S52, _S57, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S52, _S59, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S52, _S61, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S54, _S55, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S54, _S57, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S54, _S59, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S54, _S61, acc_1[3U][3U]);
        thread array<float, int(4)> av_1;
        float _S62 = (*kernelContext_2->tileA_0)[1U][ty_0];
        av_1[0U] = (*kernelContext_2->tileA_0)[1U][ty_0];
        float _S63 = (*kernelContext_2->tileA_0)[1U][_S49];
        av_1[1U] = (*kernelContext_2->tileA_0)[1U][_S49];
        float _S64 = (*kernelContext_2->tileA_0)[1U][_S51];
        av_1[2U] = (*kernelContext_2->tileA_0)[1U][_S51];
        float _S65 = (*kernelContext_2->tileA_0)[1U][_S53];
        av_1[3U] = (*kernelContext_2->tileA_0)[1U][_S53];
        thread array<float, int(4)> bv_1;
        float _S66 = (*kernelContext_2->tileB_0)[1U][tx_0];
        bv_1[0U] = (*kernelContext_2->tileB_0)[1U][tx_0];
        float _S67 = (*kernelContext_2->tileB_0)[1U][_S56];
        bv_1[1U] = (*kernelContext_2->tileB_0)[1U][_S56];
        float _S68 = (*kernelContext_2->tileB_0)[1U][_S58];
        bv_1[2U] = (*kernelContext_2->tileB_0)[1U][_S58];
        float _S69 = (*kernelContext_2->tileB_0)[1U][_S60];
        bv_1[3U] = (*kernelContext_2->tileB_0)[1U][_S60];
        acc_1[0U][0U] = fma(_S62, _S66, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S62, _S67, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S62, _S68, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S62, _S69, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S63, _S66, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S63, _S67, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S63, _S68, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S63, _S69, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S64, _S66, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S64, _S67, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S64, _S68, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S64, _S69, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S65, _S66, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S65, _S67, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S65, _S68, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S65, _S69, acc_1[3U][3U]);
        thread array<float, int(4)> av_2;
        float _S70 = (*kernelContext_2->tileA_0)[2U][ty_0];
        av_2[0U] = (*kernelContext_2->tileA_0)[2U][ty_0];
        float _S71 = (*kernelContext_2->tileA_0)[2U][_S49];
        av_2[1U] = (*kernelContext_2->tileA_0)[2U][_S49];
        float _S72 = (*kernelContext_2->tileA_0)[2U][_S51];
        av_2[2U] = (*kernelContext_2->tileA_0)[2U][_S51];
        float _S73 = (*kernelContext_2->tileA_0)[2U][_S53];
        av_2[3U] = (*kernelContext_2->tileA_0)[2U][_S53];
        thread array<float, int(4)> bv_2;
        float _S74 = (*kernelContext_2->tileB_0)[2U][tx_0];
        bv_2[0U] = (*kernelContext_2->tileB_0)[2U][tx_0];
        float _S75 = (*kernelContext_2->tileB_0)[2U][_S56];
        bv_2[1U] = (*kernelContext_2->tileB_0)[2U][_S56];
        float _S76 = (*kernelContext_2->tileB_0)[2U][_S58];
        bv_2[2U] = (*kernelContext_2->tileB_0)[2U][_S58];
        float _S77 = (*kernelContext_2->tileB_0)[2U][_S60];
        bv_2[3U] = (*kernelContext_2->tileB_0)[2U][_S60];
        acc_1[0U][0U] = fma(_S70, _S74, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S70, _S75, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S70, _S76, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S70, _S77, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S71, _S74, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S71, _S75, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S71, _S76, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S71, _S77, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S72, _S74, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S72, _S75, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S72, _S76, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S72, _S77, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S73, _S74, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S73, _S75, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S73, _S76, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S73, _S77, acc_1[3U][3U]);
        thread array<float, int(4)> av_3;
        float _S78 = (*kernelContext_2->tileA_0)[3U][ty_0];
        av_3[0U] = (*kernelContext_2->tileA_0)[3U][ty_0];
        float _S79 = (*kernelContext_2->tileA_0)[3U][_S49];
        av_3[1U] = (*kernelContext_2->tileA_0)[3U][_S49];
        float _S80 = (*kernelContext_2->tileA_0)[3U][_S51];
        av_3[2U] = (*kernelContext_2->tileA_0)[3U][_S51];
        float _S81 = (*kernelContext_2->tileA_0)[3U][_S53];
        av_3[3U] = (*kernelContext_2->tileA_0)[3U][_S53];
        thread array<float, int(4)> bv_3;
        float _S82 = (*kernelContext_2->tileB_0)[3U][tx_0];
        bv_3[0U] = (*kernelContext_2->tileB_0)[3U][tx_0];
        float _S83 = (*kernelContext_2->tileB_0)[3U][_S56];
        bv_3[1U] = (*kernelContext_2->tileB_0)[3U][_S56];
        float _S84 = (*kernelContext_2->tileB_0)[3U][_S58];
        bv_3[2U] = (*kernelContext_2->tileB_0)[3U][_S58];
        float _S85 = (*kernelContext_2->tileB_0)[3U][_S60];
        bv_3[3U] = (*kernelContext_2->tileB_0)[3U][_S60];
        acc_1[0U][0U] = fma(_S78, _S82, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S78, _S83, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S78, _S84, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S78, _S85, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S79, _S82, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S79, _S83, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S79, _S84, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S79, _S85, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S80, _S82, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S80, _S83, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S80, _S84, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S80, _S85, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S81, _S82, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S81, _S83, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S81, _S84, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S81, _S85, acc_1[3U][3U]);
        thread array<float, int(4)> av_4;
        float _S86 = (*kernelContext_2->tileA_0)[4U][ty_0];
        av_4[0U] = (*kernelContext_2->tileA_0)[4U][ty_0];
        float _S87 = (*kernelContext_2->tileA_0)[4U][_S49];
        av_4[1U] = (*kernelContext_2->tileA_0)[4U][_S49];
        float _S88 = (*kernelContext_2->tileA_0)[4U][_S51];
        av_4[2U] = (*kernelContext_2->tileA_0)[4U][_S51];
        float _S89 = (*kernelContext_2->tileA_0)[4U][_S53];
        av_4[3U] = (*kernelContext_2->tileA_0)[4U][_S53];
        thread array<float, int(4)> bv_4;
        float _S90 = (*kernelContext_2->tileB_0)[4U][tx_0];
        bv_4[0U] = (*kernelContext_2->tileB_0)[4U][tx_0];
        float _S91 = (*kernelContext_2->tileB_0)[4U][_S56];
        bv_4[1U] = (*kernelContext_2->tileB_0)[4U][_S56];
        float _S92 = (*kernelContext_2->tileB_0)[4U][_S58];
        bv_4[2U] = (*kernelContext_2->tileB_0)[4U][_S58];
        float _S93 = (*kernelContext_2->tileB_0)[4U][_S60];
        bv_4[3U] = (*kernelContext_2->tileB_0)[4U][_S60];
        acc_1[0U][0U] = fma(_S86, _S90, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S86, _S91, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S86, _S92, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S86, _S93, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S87, _S90, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S87, _S91, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S87, _S92, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S87, _S93, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S88, _S90, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S88, _S91, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S88, _S92, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S88, _S93, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S89, _S90, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S89, _S91, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S89, _S92, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S89, _S93, acc_1[3U][3U]);
        thread array<float, int(4)> av_5;
        float _S94 = (*kernelContext_2->tileA_0)[5U][ty_0];
        av_5[0U] = (*kernelContext_2->tileA_0)[5U][ty_0];
        float _S95 = (*kernelContext_2->tileA_0)[5U][_S49];
        av_5[1U] = (*kernelContext_2->tileA_0)[5U][_S49];
        float _S96 = (*kernelContext_2->tileA_0)[5U][_S51];
        av_5[2U] = (*kernelContext_2->tileA_0)[5U][_S51];
        float _S97 = (*kernelContext_2->tileA_0)[5U][_S53];
        av_5[3U] = (*kernelContext_2->tileA_0)[5U][_S53];
        thread array<float, int(4)> bv_5;
        float _S98 = (*kernelContext_2->tileB_0)[5U][tx_0];
        bv_5[0U] = (*kernelContext_2->tileB_0)[5U][tx_0];
        float _S99 = (*kernelContext_2->tileB_0)[5U][_S56];
        bv_5[1U] = (*kernelContext_2->tileB_0)[5U][_S56];
        float _S100 = (*kernelContext_2->tileB_0)[5U][_S58];
        bv_5[2U] = (*kernelContext_2->tileB_0)[5U][_S58];
        float _S101 = (*kernelContext_2->tileB_0)[5U][_S60];
        bv_5[3U] = (*kernelContext_2->tileB_0)[5U][_S60];
        acc_1[0U][0U] = fma(_S94, _S98, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S94, _S99, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S94, _S100, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S94, _S101, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S95, _S98, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S95, _S99, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S95, _S100, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S95, _S101, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S96, _S98, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S96, _S99, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S96, _S100, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S96, _S101, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S97, _S98, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S97, _S99, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S97, _S100, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S97, _S101, acc_1[3U][3U]);
        thread array<float, int(4)> av_6;
        float _S102 = (*kernelContext_2->tileA_0)[6U][ty_0];
        av_6[0U] = (*kernelContext_2->tileA_0)[6U][ty_0];
        float _S103 = (*kernelContext_2->tileA_0)[6U][_S49];
        av_6[1U] = (*kernelContext_2->tileA_0)[6U][_S49];
        float _S104 = (*kernelContext_2->tileA_0)[6U][_S51];
        av_6[2U] = (*kernelContext_2->tileA_0)[6U][_S51];
        float _S105 = (*kernelContext_2->tileA_0)[6U][_S53];
        av_6[3U] = (*kernelContext_2->tileA_0)[6U][_S53];
        thread array<float, int(4)> bv_6;
        float _S106 = (*kernelContext_2->tileB_0)[6U][tx_0];
        bv_6[0U] = (*kernelContext_2->tileB_0)[6U][tx_0];
        float _S107 = (*kernelContext_2->tileB_0)[6U][_S56];
        bv_6[1U] = (*kernelContext_2->tileB_0)[6U][_S56];
        float _S108 = (*kernelContext_2->tileB_0)[6U][_S58];
        bv_6[2U] = (*kernelContext_2->tileB_0)[6U][_S58];
        float _S109 = (*kernelContext_2->tileB_0)[6U][_S60];
        bv_6[3U] = (*kernelContext_2->tileB_0)[6U][_S60];
        acc_1[0U][0U] = fma(_S102, _S106, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S102, _S107, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S102, _S108, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S102, _S109, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S103, _S106, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S103, _S107, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S103, _S108, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S103, _S109, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S104, _S106, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S104, _S107, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S104, _S108, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S104, _S109, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S105, _S106, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S105, _S107, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S105, _S108, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S105, _S109, acc_1[3U][3U]);
        thread array<float, int(4)> av_7;
        float _S110 = (*kernelContext_2->tileA_0)[7U][ty_0];
        av_7[0U] = (*kernelContext_2->tileA_0)[7U][ty_0];
        float _S111 = (*kernelContext_2->tileA_0)[7U][_S49];
        av_7[1U] = (*kernelContext_2->tileA_0)[7U][_S49];
        float _S112 = (*kernelContext_2->tileA_0)[7U][_S51];
        av_7[2U] = (*kernelContext_2->tileA_0)[7U][_S51];
        float _S113 = (*kernelContext_2->tileA_0)[7U][_S53];
        av_7[3U] = (*kernelContext_2->tileA_0)[7U][_S53];
        thread array<float, int(4)> bv_7;
        float _S114 = (*kernelContext_2->tileB_0)[7U][tx_0];
        bv_7[0U] = (*kernelContext_2->tileB_0)[7U][tx_0];
        float _S115 = (*kernelContext_2->tileB_0)[7U][_S56];
        bv_7[1U] = (*kernelContext_2->tileB_0)[7U][_S56];
        float _S116 = (*kernelContext_2->tileB_0)[7U][_S58];
        bv_7[2U] = (*kernelContext_2->tileB_0)[7U][_S58];
        float _S117 = (*kernelContext_2->tileB_0)[7U][_S60];
        bv_7[3U] = (*kernelContext_2->tileB_0)[7U][_S60];
        acc_1[0U][0U] = fma(_S110, _S114, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S110, _S115, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S110, _S116, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S110, _S117, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S111, _S114, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S111, _S115, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S111, _S116, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S111, _S117, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S112, _S114, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S112, _S115, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S112, _S116, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S112, _S117, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S113, _S114, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S113, _S115, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S113, _S116, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S113, _S117, acc_1[3U][3U]);
        thread array<float, int(4)> av_8;
        float _S118 = (*kernelContext_2->tileA_0)[8U][ty_0];
        av_8[0U] = (*kernelContext_2->tileA_0)[8U][ty_0];
        float _S119 = (*kernelContext_2->tileA_0)[8U][_S49];
        av_8[1U] = (*kernelContext_2->tileA_0)[8U][_S49];
        float _S120 = (*kernelContext_2->tileA_0)[8U][_S51];
        av_8[2U] = (*kernelContext_2->tileA_0)[8U][_S51];
        float _S121 = (*kernelContext_2->tileA_0)[8U][_S53];
        av_8[3U] = (*kernelContext_2->tileA_0)[8U][_S53];
        thread array<float, int(4)> bv_8;
        float _S122 = (*kernelContext_2->tileB_0)[8U][tx_0];
        bv_8[0U] = (*kernelContext_2->tileB_0)[8U][tx_0];
        float _S123 = (*kernelContext_2->tileB_0)[8U][_S56];
        bv_8[1U] = (*kernelContext_2->tileB_0)[8U][_S56];
        float _S124 = (*kernelContext_2->tileB_0)[8U][_S58];
        bv_8[2U] = (*kernelContext_2->tileB_0)[8U][_S58];
        float _S125 = (*kernelContext_2->tileB_0)[8U][_S60];
        bv_8[3U] = (*kernelContext_2->tileB_0)[8U][_S60];
        acc_1[0U][0U] = fma(_S118, _S122, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S118, _S123, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S118, _S124, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S118, _S125, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S119, _S122, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S119, _S123, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S119, _S124, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S119, _S125, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S120, _S122, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S120, _S123, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S120, _S124, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S120, _S125, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S121, _S122, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S121, _S123, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S121, _S124, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S121, _S125, acc_1[3U][3U]);
        thread array<float, int(4)> av_9;
        float _S126 = (*kernelContext_2->tileA_0)[9U][ty_0];
        av_9[0U] = (*kernelContext_2->tileA_0)[9U][ty_0];
        float _S127 = (*kernelContext_2->tileA_0)[9U][_S49];
        av_9[1U] = (*kernelContext_2->tileA_0)[9U][_S49];
        float _S128 = (*kernelContext_2->tileA_0)[9U][_S51];
        av_9[2U] = (*kernelContext_2->tileA_0)[9U][_S51];
        float _S129 = (*kernelContext_2->tileA_0)[9U][_S53];
        av_9[3U] = (*kernelContext_2->tileA_0)[9U][_S53];
        thread array<float, int(4)> bv_9;
        float _S130 = (*kernelContext_2->tileB_0)[9U][tx_0];
        bv_9[0U] = (*kernelContext_2->tileB_0)[9U][tx_0];
        float _S131 = (*kernelContext_2->tileB_0)[9U][_S56];
        bv_9[1U] = (*kernelContext_2->tileB_0)[9U][_S56];
        float _S132 = (*kernelContext_2->tileB_0)[9U][_S58];
        bv_9[2U] = (*kernelContext_2->tileB_0)[9U][_S58];
        float _S133 = (*kernelContext_2->tileB_0)[9U][_S60];
        bv_9[3U] = (*kernelContext_2->tileB_0)[9U][_S60];
        acc_1[0U][0U] = fma(_S126, _S130, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S126, _S131, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S126, _S132, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S126, _S133, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S127, _S130, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S127, _S131, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S127, _S132, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S127, _S133, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S128, _S130, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S128, _S131, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S128, _S132, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S128, _S133, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S129, _S130, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S129, _S131, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S129, _S132, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S129, _S133, acc_1[3U][3U]);
        thread array<float, int(4)> av_10;
        float _S134 = (*kernelContext_2->tileA_0)[10U][ty_0];
        av_10[0U] = (*kernelContext_2->tileA_0)[10U][ty_0];
        float _S135 = (*kernelContext_2->tileA_0)[10U][_S49];
        av_10[1U] = (*kernelContext_2->tileA_0)[10U][_S49];
        float _S136 = (*kernelContext_2->tileA_0)[10U][_S51];
        av_10[2U] = (*kernelContext_2->tileA_0)[10U][_S51];
        float _S137 = (*kernelContext_2->tileA_0)[10U][_S53];
        av_10[3U] = (*kernelContext_2->tileA_0)[10U][_S53];
        thread array<float, int(4)> bv_10;
        float _S138 = (*kernelContext_2->tileB_0)[10U][tx_0];
        bv_10[0U] = (*kernelContext_2->tileB_0)[10U][tx_0];
        float _S139 = (*kernelContext_2->tileB_0)[10U][_S56];
        bv_10[1U] = (*kernelContext_2->tileB_0)[10U][_S56];
        float _S140 = (*kernelContext_2->tileB_0)[10U][_S58];
        bv_10[2U] = (*kernelContext_2->tileB_0)[10U][_S58];
        float _S141 = (*kernelContext_2->tileB_0)[10U][_S60];
        bv_10[3U] = (*kernelContext_2->tileB_0)[10U][_S60];
        acc_1[0U][0U] = fma(_S134, _S138, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S134, _S139, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S134, _S140, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S134, _S141, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S135, _S138, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S135, _S139, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S135, _S140, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S135, _S141, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S136, _S138, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S136, _S139, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S136, _S140, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S136, _S141, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S137, _S138, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S137, _S139, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S137, _S140, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S137, _S141, acc_1[3U][3U]);
        thread array<float, int(4)> av_11;
        float _S142 = (*kernelContext_2->tileA_0)[11U][ty_0];
        av_11[0U] = (*kernelContext_2->tileA_0)[11U][ty_0];
        float _S143 = (*kernelContext_2->tileA_0)[11U][_S49];
        av_11[1U] = (*kernelContext_2->tileA_0)[11U][_S49];
        float _S144 = (*kernelContext_2->tileA_0)[11U][_S51];
        av_11[2U] = (*kernelContext_2->tileA_0)[11U][_S51];
        float _S145 = (*kernelContext_2->tileA_0)[11U][_S53];
        av_11[3U] = (*kernelContext_2->tileA_0)[11U][_S53];
        thread array<float, int(4)> bv_11;
        float _S146 = (*kernelContext_2->tileB_0)[11U][tx_0];
        bv_11[0U] = (*kernelContext_2->tileB_0)[11U][tx_0];
        float _S147 = (*kernelContext_2->tileB_0)[11U][_S56];
        bv_11[1U] = (*kernelContext_2->tileB_0)[11U][_S56];
        float _S148 = (*kernelContext_2->tileB_0)[11U][_S58];
        bv_11[2U] = (*kernelContext_2->tileB_0)[11U][_S58];
        float _S149 = (*kernelContext_2->tileB_0)[11U][_S60];
        bv_11[3U] = (*kernelContext_2->tileB_0)[11U][_S60];
        acc_1[0U][0U] = fma(_S142, _S146, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S142, _S147, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S142, _S148, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S142, _S149, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S143, _S146, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S143, _S147, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S143, _S148, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S143, _S149, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S144, _S146, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S144, _S147, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S144, _S148, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S144, _S149, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S145, _S146, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S145, _S147, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S145, _S148, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S145, _S149, acc_1[3U][3U]);
        thread array<float, int(4)> av_12;
        float _S150 = (*kernelContext_2->tileA_0)[12U][ty_0];
        av_12[0U] = (*kernelContext_2->tileA_0)[12U][ty_0];
        float _S151 = (*kernelContext_2->tileA_0)[12U][_S49];
        av_12[1U] = (*kernelContext_2->tileA_0)[12U][_S49];
        float _S152 = (*kernelContext_2->tileA_0)[12U][_S51];
        av_12[2U] = (*kernelContext_2->tileA_0)[12U][_S51];
        float _S153 = (*kernelContext_2->tileA_0)[12U][_S53];
        av_12[3U] = (*kernelContext_2->tileA_0)[12U][_S53];
        thread array<float, int(4)> bv_12;
        float _S154 = (*kernelContext_2->tileB_0)[12U][tx_0];
        bv_12[0U] = (*kernelContext_2->tileB_0)[12U][tx_0];
        float _S155 = (*kernelContext_2->tileB_0)[12U][_S56];
        bv_12[1U] = (*kernelContext_2->tileB_0)[12U][_S56];
        float _S156 = (*kernelContext_2->tileB_0)[12U][_S58];
        bv_12[2U] = (*kernelContext_2->tileB_0)[12U][_S58];
        float _S157 = (*kernelContext_2->tileB_0)[12U][_S60];
        bv_12[3U] = (*kernelContext_2->tileB_0)[12U][_S60];
        acc_1[0U][0U] = fma(_S150, _S154, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S150, _S155, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S150, _S156, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S150, _S157, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S151, _S154, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S151, _S155, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S151, _S156, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S151, _S157, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S152, _S154, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S152, _S155, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S152, _S156, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S152, _S157, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S153, _S154, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S153, _S155, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S153, _S156, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S153, _S157, acc_1[3U][3U]);
        thread array<float, int(4)> av_13;
        float _S158 = (*kernelContext_2->tileA_0)[13U][ty_0];
        av_13[0U] = (*kernelContext_2->tileA_0)[13U][ty_0];
        float _S159 = (*kernelContext_2->tileA_0)[13U][_S49];
        av_13[1U] = (*kernelContext_2->tileA_0)[13U][_S49];
        float _S160 = (*kernelContext_2->tileA_0)[13U][_S51];
        av_13[2U] = (*kernelContext_2->tileA_0)[13U][_S51];
        float _S161 = (*kernelContext_2->tileA_0)[13U][_S53];
        av_13[3U] = (*kernelContext_2->tileA_0)[13U][_S53];
        thread array<float, int(4)> bv_13;
        float _S162 = (*kernelContext_2->tileB_0)[13U][tx_0];
        bv_13[0U] = (*kernelContext_2->tileB_0)[13U][tx_0];
        float _S163 = (*kernelContext_2->tileB_0)[13U][_S56];
        bv_13[1U] = (*kernelContext_2->tileB_0)[13U][_S56];
        float _S164 = (*kernelContext_2->tileB_0)[13U][_S58];
        bv_13[2U] = (*kernelContext_2->tileB_0)[13U][_S58];
        float _S165 = (*kernelContext_2->tileB_0)[13U][_S60];
        bv_13[3U] = (*kernelContext_2->tileB_0)[13U][_S60];
        acc_1[0U][0U] = fma(_S158, _S162, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S158, _S163, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S158, _S164, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S158, _S165, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S159, _S162, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S159, _S163, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S159, _S164, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S159, _S165, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S160, _S162, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S160, _S163, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S160, _S164, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S160, _S165, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S161, _S162, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S161, _S163, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S161, _S164, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S161, _S165, acc_1[3U][3U]);
        thread array<float, int(4)> av_14;
        float _S166 = (*kernelContext_2->tileA_0)[14U][ty_0];
        av_14[0U] = (*kernelContext_2->tileA_0)[14U][ty_0];
        float _S167 = (*kernelContext_2->tileA_0)[14U][_S49];
        av_14[1U] = (*kernelContext_2->tileA_0)[14U][_S49];
        float _S168 = (*kernelContext_2->tileA_0)[14U][_S51];
        av_14[2U] = (*kernelContext_2->tileA_0)[14U][_S51];
        float _S169 = (*kernelContext_2->tileA_0)[14U][_S53];
        av_14[3U] = (*kernelContext_2->tileA_0)[14U][_S53];
        thread array<float, int(4)> bv_14;
        float _S170 = (*kernelContext_2->tileB_0)[14U][tx_0];
        bv_14[0U] = (*kernelContext_2->tileB_0)[14U][tx_0];
        float _S171 = (*kernelContext_2->tileB_0)[14U][_S56];
        bv_14[1U] = (*kernelContext_2->tileB_0)[14U][_S56];
        float _S172 = (*kernelContext_2->tileB_0)[14U][_S58];
        bv_14[2U] = (*kernelContext_2->tileB_0)[14U][_S58];
        float _S173 = (*kernelContext_2->tileB_0)[14U][_S60];
        bv_14[3U] = (*kernelContext_2->tileB_0)[14U][_S60];
        acc_1[0U][0U] = fma(_S166, _S170, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S166, _S171, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S166, _S172, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S166, _S173, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S167, _S170, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S167, _S171, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S167, _S172, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S167, _S173, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S168, _S170, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S168, _S171, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S168, _S172, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S168, _S173, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S169, _S170, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S169, _S171, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S169, _S172, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S169, _S173, acc_1[3U][3U]);
        thread array<float, int(4)> av_15;
        float _S174 = (*kernelContext_2->tileA_0)[15U][ty_0];
        av_15[0U] = (*kernelContext_2->tileA_0)[15U][ty_0];
        float _S175 = (*kernelContext_2->tileA_0)[15U][_S49];
        av_15[1U] = (*kernelContext_2->tileA_0)[15U][_S49];
        float _S176 = (*kernelContext_2->tileA_0)[15U][_S51];
        av_15[2U] = (*kernelContext_2->tileA_0)[15U][_S51];
        float _S177 = (*kernelContext_2->tileA_0)[15U][_S53];
        av_15[3U] = (*kernelContext_2->tileA_0)[15U][_S53];
        thread array<float, int(4)> bv_15;
        float _S178 = (*kernelContext_2->tileB_0)[15U][tx_0];
        bv_15[0U] = (*kernelContext_2->tileB_0)[15U][tx_0];
        float _S179 = (*kernelContext_2->tileB_0)[15U][_S56];
        bv_15[1U] = (*kernelContext_2->tileB_0)[15U][_S56];
        float _S180 = (*kernelContext_2->tileB_0)[15U][_S58];
        bv_15[2U] = (*kernelContext_2->tileB_0)[15U][_S58];
        float _S181 = (*kernelContext_2->tileB_0)[15U][_S60];
        bv_15[3U] = (*kernelContext_2->tileB_0)[15U][_S60];
        acc_1[0U][0U] = fma(_S174, _S178, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S174, _S179, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S174, _S180, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S174, _S181, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S175, _S178, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S175, _S179, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S175, _S180, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S175, _S181, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S176, _S178, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S176, _S179, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S176, _S180, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S176, _S181, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S177, _S178, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S177, _S179, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S177, _S180, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S177, _S181, acc_1[3U][3U]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k0_0 = k0_0 + 16U;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                uint _S182 = _S25 + ty_0;
                _S14 = _S182;
                _S15 = pr_0->m_2;
                if(_S182 >= (pr_0->m_2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S183 = _S26 + tx_0;
                            _S16 = _S183;
                            _S17 = pr_0->n_2;
                            if(_S183 >= (pr_0->n_2))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S182, _S183, acc_1[0U][0U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_0 = _S16 + 16U;
                            if(col_0 >= _S17)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S182, col_0, acc_1[0U][1U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_1 = _S16 + 32U;
                            if(col_1 >= _S17)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S182, col_1, acc_1[0U][2U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_2 = _S16 + 48U;
                            if(col_2 >= _S17)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S182, col_2, acc_1[0U][3U], kernelContext_2);
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
                uint r_2 = _S14 + 16U;
                if(r_2 >= _S15)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S184 = _S26 + tx_0;
                            _S18 = _S184;
                            _S19 = pr_0->n_2;
                            if(_S184 >= (pr_0->n_2))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, _S184, acc_1[1U][0U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_3 = _S18 + 16U;
                            if(col_3 >= _S19)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, col_3, acc_1[1U][1U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_4 = _S18 + 32U;
                            if(col_4 >= _S19)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, col_4, acc_1[1U][2U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_5 = _S18 + 48U;
                            if(col_5 >= _S19)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, col_5, acc_1[1U][3U], kernelContext_2);
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
                uint r_3 = _S14 + 32U;
                if(r_3 >= _S15)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S185 = _S26 + tx_0;
                            _S20 = _S185;
                            _S21 = pr_0->n_2;
                            if(_S185 >= (pr_0->n_2))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, _S185, acc_1[2U][0U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_6 = _S20 + 16U;
                            if(col_6 >= _S21)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, col_6, acc_1[2U][1U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_7 = _S20 + 32U;
                            if(col_7 >= _S21)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, col_7, acc_1[2U][2U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_8 = _S20 + 48U;
                            if(col_8 >= _S21)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, col_8, acc_1[2U][3U], kernelContext_2);
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
                uint r_4 = _S14 + 48U;
                if(r_4 >= _S15)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S186 = _S26 + tx_0;
                            _S22 = _S186;
                            _S23 = pr_0->n_2;
                            if(_S186 >= (pr_0->n_2))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, _S186, acc_1[3U][0U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_9 = _S22 + 16U;
                            if(col_9 >= _S23)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, col_9, acc_1[3U][1U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_10 = _S22 + 32U;
                            if(col_10 >= _S23)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, col_10, acc_1[3U][2U], kernelContext_2);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_11 = _S22 + 48U;
                            if(col_11 >= _S23)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, col_11, acc_1[3U][3U], kernelContext_2);
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

[[kernel]] void gemm_grouped(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(8)]], ProblemRecord_0 device* problems_1 [[buffer(6)]], uint device* errors_2 [[buffer(7)]], half device* a_2 [[buffer(0)]], half device* b_2 [[buffer(1)]], float device* c_1 [[buffer(2)]], float device* epBias_1 [[buffer(3)]], float device* epGate_1 [[buffer(4)]], float device* epResidual_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->p_0 = p_1;
    (&kernelContext_3)->problems_0 = problems_1;
    (&kernelContext_3)->errors_1 = errors_2;
    (&kernelContext_3)->a_1 = a_2;
    (&kernelContext_3)->b_1 = b_2;
    (&kernelContext_3)->c_0 = c_1;
    (&kernelContext_3)->epBias_0 = epBias_1;
    (&kernelContext_3)->epGate_0 = epGate_1;
    (&kernelContext_3)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(64)>, int(16)> tileA_1;
    (&kernelContext_3)->tileA_0 = &tileA_1;
    threadgroup array<array<float, int(64)>, int(16)> tileB_1;
    (&kernelContext_3)->tileB_0 = &tileB_1;
    uint z_0 = group_0.z;
    if(z_0 >= (p_1->count_0))
    {
        return;
    }
    ProblemRecord_0 device* _S187 = (&kernelContext_3)->problems_0+z_0;
    ProblemRecord_0 r_5 = *_S187;
    uint _S188 = group_0.x;
    bool ok_0;
    if((_S188 * 64U) >= ((*_S187).n_1))
    {
        ok_0 = true;
    }
    else
    {
        ok_0 = (group_0.y * 64U) >= (r_5.m_1);
    }
    if(ok_0)
    {
        return;
    }
    bool transA_1 = ((&kernelContext_3)->p_0->transA_0) != 0U;
    bool transB_1 = ((&kernelContext_3)->p_0->transB_0) != 0U;
    uint _S189;
    if(transA_1)
    {
        _S189 = r_5.k_1;
    }
    else
    {
        _S189 = r_5.m_1;
    }
    uint _S190;
    if(transA_1)
    {
        _S190 = r_5.m_1;
    }
    else
    {
        _S190 = r_5.k_1;
    }
    bool _S191 = fits_0(r_5.offsetA_0, _S189, _S190, r_5.lda_1, (&kernelContext_3)->p_0->aLength_0);
    if(_S191)
    {
        if(transB_1)
        {
            _S189 = r_5.n_1;
        }
        else
        {
            _S189 = r_5.k_1;
        }
        if(transB_1)
        {
            _S190 = r_5.k_1;
        }
        else
        {
            _S190 = r_5.n_1;
        }
        bool _S192 = fits_0(r_5.offsetB_0, _S189, _S190, r_5.ldb_1, (&kernelContext_3)->p_0->bLength_0);
        ok_0 = _S192;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        bool _S193 = fits_0(r_5.offsetC_0, r_5.m_1, r_5.n_1, r_5.ldc_1, (&kernelContext_3)->p_0->cLength_0);
        ok_0 = _S193;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        ok_0 = (r_5.m_1) <= ((&kernelContext_3)->p_0->m_0);
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        ok_0 = (r_5.n_1) <= ((&kernelContext_3)->p_0->n_0);
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        if(((&kernelContext_3)->p_0->bias_0) == 0U)
        {
            ok_0 = true;
        }
        else
        {
            if(((&kernelContext_3)->p_0->bias_0) == 1U)
            {
                _S189 = r_5.n_1;
            }
            else
            {
                _S189 = r_5.m_1;
            }
            if(((&kernelContext_3)->p_0->bias_0) == 1U)
            {
                _S190 = r_5.n_1;
            }
            else
            {
                _S190 = r_5.m_1;
            }
            bool _S194 = fits_0(r_5.offsetBias_0, 1U, _S189, _S190, (&kernelContext_3)->p_0->biasLength_0);
            ok_0 = _S194;
        }
    }
    else
    {
        ok_0 = false;
    }
    if(!ok_0)
    {
        if(_S188 == 0U)
        {
            ok_0 = (group_0.y) == 0U;
        }
        else
        {
            ok_0 = false;
        }
        if(ok_0)
        {
            ok_0 = (local_1.x) == 0U;
        }
        else
        {
            ok_0 = false;
        }
        if(ok_0)
        {
            ok_0 = (local_1.y) == 0U;
        }
        else
        {
            ok_0 = false;
        }
        if(ok_0)
        {
            dg_report_error_0((&kernelContext_3)->errors_1, 1U, z_0);
        }
        return;
    }
    thread GemmProblem_0 pr_1;
    (&pr_1)->m_2 = r_5.m_1;
    (&pr_1)->n_2 = r_5.n_1;
    (&pr_1)->k_2 = r_5.k_1;
    (&pr_1)->lda_2 = r_5.lda_1;
    (&pr_1)->ldb_2 = r_5.ldb_1;
    (&pr_1)->ldc_2 = r_5.ldc_1;
    (&pr_1)->baseA_0 = r_5.offsetA_0;
    (&pr_1)->baseB_0 = r_5.offsetB_0;
    (&pr_1)->baseC_0 = r_5.offsetC_0;
    (&pr_1)->baseBias_0 = r_5.offsetBias_0;
    uint2 _S195 = group_0.xy;
    thread GemmProblem_0 _S196 = pr_1;
    gemmTile_0(&_S196, _S195, local_1, &kernelContext_3);
    return;
}

