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

float bf16At_0(uint w_0, uint e_0)
{
    uint _S7;
    if((e_0 & 1U) != 0U)
    {
        _S7 = w_0 & 4294901760U;
    }
    else
    {
        _S7 = w_0 << 16U;
    }
    return (as_type<float>((_S7)));
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
    uint device* a_0;
    uint device* b_0;
    float device* c_0;
    float device* epBias_0;
    float device* epGate_0;
    float device* epResidual_0;
    array<array<float, int(64)>, int(16)> threadgroup* tileA_0;
    array<array<float, int(64)>, int(16)> threadgroup* tileB_0;
};

float loadA_0(uint e_1, KernelContext_0 thread* kernelContext_0)
{
    return bf16At_0(*(kernelContext_0->a_0+(e_1 >> 1U)), e_1);
}

float loadB_0(uint e_2, KernelContext_0 thread* kernelContext_1)
{
    return bf16At_0(*(kernelContext_1->b_0+(e_2 >> 1U)), e_2);
}

bool dg_is_nan_0(float x_0)
{
    uint _S8 = (as_type<uint>((x_0)));
    bool _S9;
    if((_S8 & 2139095040U) == 2139095040U)
    {
        _S9 = (_S8 & 8388607U) != 0U;
    }
    else
    {
        _S9 = false;
    }
    return _S9;
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
    bool _S10;
    if(a_1 == 0.0)
    {
        _S10 = b_1 == 0.0;
    }
    else
    {
        _S10 = false;
    }
    if(_S10)
    {
        return (as_type<float>(((as_type<uint>((a_1))) & (as_type<uint>((b_1))))));
    }
    float _S11;
    if(a_1 > b_1)
    {
        _S11 = a_1;
    }
    else
    {
        _S11 = b_1;
    }
    return _S11;
}

float gemmEpilogue_0(float r_0, uint i_0, uint j_0, uint idx_0, uint biasBase_0, KernelContext_0 thread* kernelContext_2)
{
    float _S12;
    if((kernelContext_2->p_0->bias_0) == 1U)
    {
        _S12 = r_0 + *(kernelContext_2->epBias_0+(biasBase_0 + j_0));
    }
    else
    {
        if((kernelContext_2->p_0->bias_0) == 2U)
        {
            _S12 = r_0 + *(kernelContext_2->epBias_0+(biasBase_0 + i_0));
        }
        else
        {
            _S12 = r_0;
        }
    }
    if((kernelContext_2->p_0->act_0) == 1U)
    {
        _S12 = dg_max_0(_S12, 0.0);
    }
    else
    {
        if((kernelContext_2->p_0->act_0) == 2U)
        {
            _S12 = 0.5 * _S12 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S12 + 0.04471499845385551 * _S12 * _S12 * _S12))) + 1.0)));
        }
        else
        {
            if((kernelContext_2->p_0->act_0) == 3U)
            {
                _S12 = _S12 / (1.0 + exp(- _S12));
            }
        }
    }
    if((kernelContext_2->p_0->gated_0) != 0U)
    {
        _S12 = _S12 * *(kernelContext_2->epGate_0+idx_0);
    }
    if((kernelContext_2->p_0->residual_0) != 0U)
    {
        _S12 = _S12 + *(kernelContext_2->epResidual_0+idx_0);
    }
    return _S12;
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

void gemmStore_0(const GemmProblem_0 thread* q_0, uint i_1, uint j_1, float acc_0, KernelContext_0 thread* kernelContext_3)
{
    uint idx_1 = q_0->baseC_0 + i_1 * q_0->ldc_2 + j_1;
    float r_1;
    if((kernelContext_3->p_0->beta_0) == 0.0)
    {
        r_1 = kernelContext_3->p_0->alpha_0 * acc_0;
    }
    else
    {
        r_1 = kernelContext_3->p_0->alpha_0 * acc_0 + kernelContext_3->p_0->beta_0 * *(kernelContext_3->c_0+idx_1);
    }
    float device* _S13 = kernelContext_3->c_0+idx_1;
    float _S14 = gemmEpilogue_0(r_1, i_1, j_1, idx_1, q_0->baseBias_0, kernelContext_3);
    *_S13 = _S14;
    return;
}

void gemmTile_0(const GemmProblem_0 thread* pr_0, uint2 tile_0, uint3 local_0, KernelContext_0 thread* kernelContext_4)
{
    uint _S15;
    uint _S16;
    uint _S17;
    uint _S18;
    uint _S19;
    uint _S20;
    uint _S21;
    uint _S22;
    uint _S23;
    uint _S24;
    uint tx_0 = local_0.x;
    uint ty_0 = local_0.y;
    uint _S25 = ty_0 * 16U + tx_0;
    uint _S26 = tile_0.y * 64U;
    uint _S27 = tile_0.x * 64U;
    uint _S28 = pr_0->baseA_0;
    uint _S29 = pr_0->baseB_0;
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
        uint _S30 = pr_0->k_2;
        if(k0_0 < (pr_0->k_2))
        {
        }
        else
        {
            break;
        }
        uint _S31 = _S25 * 4U;
        uint ar_0 = _S31 / 16U;
        uint ak_0 = _S31 % 16U;
        uint gr_0 = _S26 + ar_0;
        uint gk_0 = k0_0 + ak_0;
        uint ai_0;
        if((kernelContext_4->p_0->transA_0) != 0U)
        {
            ai_0 = gk_0 * pr_0->lda_2 + gr_0;
        }
        else
        {
            ai_0 = gr_0 * pr_0->lda_2 + gk_0;
        }
        uint _S32 = pr_0->m_2;
        bool _S33;
        if(gr_0 < (pr_0->m_2))
        {
            _S33 = gk_0 < _S30;
        }
        else
        {
            _S33 = false;
        }
        float _S34;
        if(_S33)
        {
            float _S35 = loadA_0(_S28 + ai_0, kernelContext_4);
            _S34 = _S35;
        }
        else
        {
            _S34 = 0.0;
        }
        (*kernelContext_4->tileA_0)[ak_0][ar_0] = _S34;
        uint br_0 = _S31 / 64U;
        uint bc_0 = _S31 % 64U;
        uint hk_0 = k0_0 + br_0;
        uint gc_0 = _S27 + bc_0;
        uint bi_0;
        if((kernelContext_4->p_0->transB_0) != 0U)
        {
            bi_0 = gc_0 * pr_0->ldb_2 + hk_0;
        }
        else
        {
            bi_0 = hk_0 * pr_0->ldb_2 + gc_0;
        }
        bool _S36;
        if(hk_0 < _S30)
        {
            _S36 = gc_0 < (pr_0->n_2);
        }
        else
        {
            _S36 = false;
        }
        float _S37;
        if(_S36)
        {
            float _S38 = loadB_0(_S29 + bi_0, kernelContext_4);
            _S37 = _S38;
        }
        else
        {
            _S37 = 0.0;
        }
        (*kernelContext_4->tileB_0)[br_0][bc_0] = _S37;
        uint e_3 = _S31 + 1U;
        uint ar_1 = e_3 / 16U;
        uint ak_1 = e_3 % 16U;
        uint gr_1 = _S26 + ar_1;
        uint gk_1 = k0_0 + ak_1;
        uint ai_1;
        if((kernelContext_4->p_0->transA_0) != 0U)
        {
            ai_1 = gk_1 * pr_0->lda_2 + gr_1;
        }
        else
        {
            ai_1 = gr_1 * pr_0->lda_2 + gk_1;
        }
        bool _S39;
        if(gr_1 < _S32)
        {
            _S39 = gk_1 < _S30;
        }
        else
        {
            _S39 = false;
        }
        float _S40;
        if(_S39)
        {
            float _S41 = loadA_0(_S28 + ai_1, kernelContext_4);
            _S40 = _S41;
        }
        else
        {
            _S40 = 0.0;
        }
        (*kernelContext_4->tileA_0)[ak_1][ar_1] = _S40;
        uint br_1 = e_3 / 64U;
        uint bc_1 = e_3 % 64U;
        uint hk_1 = k0_0 + br_1;
        uint gc_1 = _S27 + bc_1;
        uint bi_1;
        if((kernelContext_4->p_0->transB_0) != 0U)
        {
            bi_1 = gc_1 * pr_0->ldb_2 + hk_1;
        }
        else
        {
            bi_1 = hk_1 * pr_0->ldb_2 + gc_1;
        }
        bool _S42;
        if(hk_1 < _S30)
        {
            _S42 = gc_1 < (pr_0->n_2);
        }
        else
        {
            _S42 = false;
        }
        float _S43;
        if(_S42)
        {
            float _S44 = loadB_0(_S29 + bi_1, kernelContext_4);
            _S43 = _S44;
        }
        else
        {
            _S43 = 0.0;
        }
        (*kernelContext_4->tileB_0)[br_1][bc_1] = _S43;
        uint e_4 = _S31 + 2U;
        uint ar_2 = e_4 / 16U;
        uint ak_2 = e_4 % 16U;
        uint gr_2 = _S26 + ar_2;
        uint gk_2 = k0_0 + ak_2;
        uint ai_2;
        if((kernelContext_4->p_0->transA_0) != 0U)
        {
            ai_2 = gk_2 * pr_0->lda_2 + gr_2;
        }
        else
        {
            ai_2 = gr_2 * pr_0->lda_2 + gk_2;
        }
        bool _S45;
        if(gr_2 < _S32)
        {
            _S45 = gk_2 < _S30;
        }
        else
        {
            _S45 = false;
        }
        float _S46;
        if(_S45)
        {
            float _S47 = loadA_0(_S28 + ai_2, kernelContext_4);
            _S46 = _S47;
        }
        else
        {
            _S46 = 0.0;
        }
        (*kernelContext_4->tileA_0)[ak_2][ar_2] = _S46;
        uint br_2 = e_4 / 64U;
        uint bc_2 = e_4 % 64U;
        uint hk_2 = k0_0 + br_2;
        uint gc_2 = _S27 + bc_2;
        uint bi_2;
        if((kernelContext_4->p_0->transB_0) != 0U)
        {
            bi_2 = gc_2 * pr_0->ldb_2 + hk_2;
        }
        else
        {
            bi_2 = hk_2 * pr_0->ldb_2 + gc_2;
        }
        bool _S48;
        if(hk_2 < _S30)
        {
            _S48 = gc_2 < (pr_0->n_2);
        }
        else
        {
            _S48 = false;
        }
        float _S49;
        if(_S48)
        {
            float _S50 = loadB_0(_S29 + bi_2, kernelContext_4);
            _S49 = _S50;
        }
        else
        {
            _S49 = 0.0;
        }
        (*kernelContext_4->tileB_0)[br_2][bc_2] = _S49;
        uint e_5 = _S31 + 3U;
        uint ar_3 = e_5 / 16U;
        uint ak_3 = e_5 % 16U;
        uint gr_3 = _S26 + ar_3;
        uint gk_3 = k0_0 + ak_3;
        uint ai_3;
        if((kernelContext_4->p_0->transA_0) != 0U)
        {
            ai_3 = gk_3 * pr_0->lda_2 + gr_3;
        }
        else
        {
            ai_3 = gr_3 * pr_0->lda_2 + gk_3;
        }
        bool _S51;
        if(gr_3 < _S32)
        {
            _S51 = gk_3 < _S30;
        }
        else
        {
            _S51 = false;
        }
        float _S52;
        if(_S51)
        {
            float _S53 = loadA_0(_S28 + ai_3, kernelContext_4);
            _S52 = _S53;
        }
        else
        {
            _S52 = 0.0;
        }
        (*kernelContext_4->tileA_0)[ak_3][ar_3] = _S52;
        uint br_3 = e_5 / 64U;
        uint bc_3 = e_5 % 64U;
        uint hk_3 = k0_0 + br_3;
        uint gc_3 = _S27 + bc_3;
        uint bi_3;
        if((kernelContext_4->p_0->transB_0) != 0U)
        {
            bi_3 = gc_3 * pr_0->ldb_2 + hk_3;
        }
        else
        {
            bi_3 = hk_3 * pr_0->ldb_2 + gc_3;
        }
        bool _S54;
        if(hk_3 < _S30)
        {
            _S54 = gc_3 < (pr_0->n_2);
        }
        else
        {
            _S54 = false;
        }
        float _S55;
        if(_S54)
        {
            float _S56 = loadB_0(_S29 + bi_3, kernelContext_4);
            _S55 = _S56;
        }
        else
        {
            _S55 = 0.0;
        }
        (*kernelContext_4->tileB_0)[br_3][bc_3] = _S55;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        thread array<float, int(4)> av_0;
        float _S57 = (*kernelContext_4->tileA_0)[0U][ty_0];
        av_0[0U] = (*kernelContext_4->tileA_0)[0U][ty_0];
        uint _S58 = ty_0 + 16U;
        float _S59 = (*kernelContext_4->tileA_0)[0U][_S58];
        av_0[1U] = (*kernelContext_4->tileA_0)[0U][_S58];
        uint _S60 = ty_0 + 32U;
        float _S61 = (*kernelContext_4->tileA_0)[0U][_S60];
        av_0[2U] = (*kernelContext_4->tileA_0)[0U][_S60];
        uint _S62 = ty_0 + 48U;
        float _S63 = (*kernelContext_4->tileA_0)[0U][_S62];
        av_0[3U] = (*kernelContext_4->tileA_0)[0U][_S62];
        thread array<float, int(4)> bv_0;
        float _S64 = (*kernelContext_4->tileB_0)[0U][tx_0];
        bv_0[0U] = (*kernelContext_4->tileB_0)[0U][tx_0];
        uint _S65 = tx_0 + 16U;
        float _S66 = (*kernelContext_4->tileB_0)[0U][_S65];
        bv_0[1U] = (*kernelContext_4->tileB_0)[0U][_S65];
        uint _S67 = tx_0 + 32U;
        float _S68 = (*kernelContext_4->tileB_0)[0U][_S67];
        bv_0[2U] = (*kernelContext_4->tileB_0)[0U][_S67];
        uint _S69 = tx_0 + 48U;
        float _S70 = (*kernelContext_4->tileB_0)[0U][_S69];
        bv_0[3U] = (*kernelContext_4->tileB_0)[0U][_S69];
        acc_1[0U][0U] = fma(_S57, _S64, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S57, _S66, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S57, _S68, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S57, _S70, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S59, _S64, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S59, _S66, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S59, _S68, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S59, _S70, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S61, _S64, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S61, _S66, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S61, _S68, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S61, _S70, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S63, _S64, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S63, _S66, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S63, _S68, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S63, _S70, acc_1[3U][3U]);
        thread array<float, int(4)> av_1;
        float _S71 = (*kernelContext_4->tileA_0)[1U][ty_0];
        av_1[0U] = (*kernelContext_4->tileA_0)[1U][ty_0];
        float _S72 = (*kernelContext_4->tileA_0)[1U][_S58];
        av_1[1U] = (*kernelContext_4->tileA_0)[1U][_S58];
        float _S73 = (*kernelContext_4->tileA_0)[1U][_S60];
        av_1[2U] = (*kernelContext_4->tileA_0)[1U][_S60];
        float _S74 = (*kernelContext_4->tileA_0)[1U][_S62];
        av_1[3U] = (*kernelContext_4->tileA_0)[1U][_S62];
        thread array<float, int(4)> bv_1;
        float _S75 = (*kernelContext_4->tileB_0)[1U][tx_0];
        bv_1[0U] = (*kernelContext_4->tileB_0)[1U][tx_0];
        float _S76 = (*kernelContext_4->tileB_0)[1U][_S65];
        bv_1[1U] = (*kernelContext_4->tileB_0)[1U][_S65];
        float _S77 = (*kernelContext_4->tileB_0)[1U][_S67];
        bv_1[2U] = (*kernelContext_4->tileB_0)[1U][_S67];
        float _S78 = (*kernelContext_4->tileB_0)[1U][_S69];
        bv_1[3U] = (*kernelContext_4->tileB_0)[1U][_S69];
        acc_1[0U][0U] = fma(_S71, _S75, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S71, _S76, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S71, _S77, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S71, _S78, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S72, _S75, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S72, _S76, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S72, _S77, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S72, _S78, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S73, _S75, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S73, _S76, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S73, _S77, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S73, _S78, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S74, _S75, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S74, _S76, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S74, _S77, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S74, _S78, acc_1[3U][3U]);
        thread array<float, int(4)> av_2;
        float _S79 = (*kernelContext_4->tileA_0)[2U][ty_0];
        av_2[0U] = (*kernelContext_4->tileA_0)[2U][ty_0];
        float _S80 = (*kernelContext_4->tileA_0)[2U][_S58];
        av_2[1U] = (*kernelContext_4->tileA_0)[2U][_S58];
        float _S81 = (*kernelContext_4->tileA_0)[2U][_S60];
        av_2[2U] = (*kernelContext_4->tileA_0)[2U][_S60];
        float _S82 = (*kernelContext_4->tileA_0)[2U][_S62];
        av_2[3U] = (*kernelContext_4->tileA_0)[2U][_S62];
        thread array<float, int(4)> bv_2;
        float _S83 = (*kernelContext_4->tileB_0)[2U][tx_0];
        bv_2[0U] = (*kernelContext_4->tileB_0)[2U][tx_0];
        float _S84 = (*kernelContext_4->tileB_0)[2U][_S65];
        bv_2[1U] = (*kernelContext_4->tileB_0)[2U][_S65];
        float _S85 = (*kernelContext_4->tileB_0)[2U][_S67];
        bv_2[2U] = (*kernelContext_4->tileB_0)[2U][_S67];
        float _S86 = (*kernelContext_4->tileB_0)[2U][_S69];
        bv_2[3U] = (*kernelContext_4->tileB_0)[2U][_S69];
        acc_1[0U][0U] = fma(_S79, _S83, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S79, _S84, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S79, _S85, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S79, _S86, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S80, _S83, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S80, _S84, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S80, _S85, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S80, _S86, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S81, _S83, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S81, _S84, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S81, _S85, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S81, _S86, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S82, _S83, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S82, _S84, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S82, _S85, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S82, _S86, acc_1[3U][3U]);
        thread array<float, int(4)> av_3;
        float _S87 = (*kernelContext_4->tileA_0)[3U][ty_0];
        av_3[0U] = (*kernelContext_4->tileA_0)[3U][ty_0];
        float _S88 = (*kernelContext_4->tileA_0)[3U][_S58];
        av_3[1U] = (*kernelContext_4->tileA_0)[3U][_S58];
        float _S89 = (*kernelContext_4->tileA_0)[3U][_S60];
        av_3[2U] = (*kernelContext_4->tileA_0)[3U][_S60];
        float _S90 = (*kernelContext_4->tileA_0)[3U][_S62];
        av_3[3U] = (*kernelContext_4->tileA_0)[3U][_S62];
        thread array<float, int(4)> bv_3;
        float _S91 = (*kernelContext_4->tileB_0)[3U][tx_0];
        bv_3[0U] = (*kernelContext_4->tileB_0)[3U][tx_0];
        float _S92 = (*kernelContext_4->tileB_0)[3U][_S65];
        bv_3[1U] = (*kernelContext_4->tileB_0)[3U][_S65];
        float _S93 = (*kernelContext_4->tileB_0)[3U][_S67];
        bv_3[2U] = (*kernelContext_4->tileB_0)[3U][_S67];
        float _S94 = (*kernelContext_4->tileB_0)[3U][_S69];
        bv_3[3U] = (*kernelContext_4->tileB_0)[3U][_S69];
        acc_1[0U][0U] = fma(_S87, _S91, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S87, _S92, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S87, _S93, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S87, _S94, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S88, _S91, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S88, _S92, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S88, _S93, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S88, _S94, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S89, _S91, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S89, _S92, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S89, _S93, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S89, _S94, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S90, _S91, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S90, _S92, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S90, _S93, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S90, _S94, acc_1[3U][3U]);
        thread array<float, int(4)> av_4;
        float _S95 = (*kernelContext_4->tileA_0)[4U][ty_0];
        av_4[0U] = (*kernelContext_4->tileA_0)[4U][ty_0];
        float _S96 = (*kernelContext_4->tileA_0)[4U][_S58];
        av_4[1U] = (*kernelContext_4->tileA_0)[4U][_S58];
        float _S97 = (*kernelContext_4->tileA_0)[4U][_S60];
        av_4[2U] = (*kernelContext_4->tileA_0)[4U][_S60];
        float _S98 = (*kernelContext_4->tileA_0)[4U][_S62];
        av_4[3U] = (*kernelContext_4->tileA_0)[4U][_S62];
        thread array<float, int(4)> bv_4;
        float _S99 = (*kernelContext_4->tileB_0)[4U][tx_0];
        bv_4[0U] = (*kernelContext_4->tileB_0)[4U][tx_0];
        float _S100 = (*kernelContext_4->tileB_0)[4U][_S65];
        bv_4[1U] = (*kernelContext_4->tileB_0)[4U][_S65];
        float _S101 = (*kernelContext_4->tileB_0)[4U][_S67];
        bv_4[2U] = (*kernelContext_4->tileB_0)[4U][_S67];
        float _S102 = (*kernelContext_4->tileB_0)[4U][_S69];
        bv_4[3U] = (*kernelContext_4->tileB_0)[4U][_S69];
        acc_1[0U][0U] = fma(_S95, _S99, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S95, _S100, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S95, _S101, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S95, _S102, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S96, _S99, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S96, _S100, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S96, _S101, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S96, _S102, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S97, _S99, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S97, _S100, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S97, _S101, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S97, _S102, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S98, _S99, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S98, _S100, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S98, _S101, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S98, _S102, acc_1[3U][3U]);
        thread array<float, int(4)> av_5;
        float _S103 = (*kernelContext_4->tileA_0)[5U][ty_0];
        av_5[0U] = (*kernelContext_4->tileA_0)[5U][ty_0];
        float _S104 = (*kernelContext_4->tileA_0)[5U][_S58];
        av_5[1U] = (*kernelContext_4->tileA_0)[5U][_S58];
        float _S105 = (*kernelContext_4->tileA_0)[5U][_S60];
        av_5[2U] = (*kernelContext_4->tileA_0)[5U][_S60];
        float _S106 = (*kernelContext_4->tileA_0)[5U][_S62];
        av_5[3U] = (*kernelContext_4->tileA_0)[5U][_S62];
        thread array<float, int(4)> bv_5;
        float _S107 = (*kernelContext_4->tileB_0)[5U][tx_0];
        bv_5[0U] = (*kernelContext_4->tileB_0)[5U][tx_0];
        float _S108 = (*kernelContext_4->tileB_0)[5U][_S65];
        bv_5[1U] = (*kernelContext_4->tileB_0)[5U][_S65];
        float _S109 = (*kernelContext_4->tileB_0)[5U][_S67];
        bv_5[2U] = (*kernelContext_4->tileB_0)[5U][_S67];
        float _S110 = (*kernelContext_4->tileB_0)[5U][_S69];
        bv_5[3U] = (*kernelContext_4->tileB_0)[5U][_S69];
        acc_1[0U][0U] = fma(_S103, _S107, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S103, _S108, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S103, _S109, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S103, _S110, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S104, _S107, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S104, _S108, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S104, _S109, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S104, _S110, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S105, _S107, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S105, _S108, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S105, _S109, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S105, _S110, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S106, _S107, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S106, _S108, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S106, _S109, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S106, _S110, acc_1[3U][3U]);
        thread array<float, int(4)> av_6;
        float _S111 = (*kernelContext_4->tileA_0)[6U][ty_0];
        av_6[0U] = (*kernelContext_4->tileA_0)[6U][ty_0];
        float _S112 = (*kernelContext_4->tileA_0)[6U][_S58];
        av_6[1U] = (*kernelContext_4->tileA_0)[6U][_S58];
        float _S113 = (*kernelContext_4->tileA_0)[6U][_S60];
        av_6[2U] = (*kernelContext_4->tileA_0)[6U][_S60];
        float _S114 = (*kernelContext_4->tileA_0)[6U][_S62];
        av_6[3U] = (*kernelContext_4->tileA_0)[6U][_S62];
        thread array<float, int(4)> bv_6;
        float _S115 = (*kernelContext_4->tileB_0)[6U][tx_0];
        bv_6[0U] = (*kernelContext_4->tileB_0)[6U][tx_0];
        float _S116 = (*kernelContext_4->tileB_0)[6U][_S65];
        bv_6[1U] = (*kernelContext_4->tileB_0)[6U][_S65];
        float _S117 = (*kernelContext_4->tileB_0)[6U][_S67];
        bv_6[2U] = (*kernelContext_4->tileB_0)[6U][_S67];
        float _S118 = (*kernelContext_4->tileB_0)[6U][_S69];
        bv_6[3U] = (*kernelContext_4->tileB_0)[6U][_S69];
        acc_1[0U][0U] = fma(_S111, _S115, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S111, _S116, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S111, _S117, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S111, _S118, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S112, _S115, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S112, _S116, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S112, _S117, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S112, _S118, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S113, _S115, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S113, _S116, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S113, _S117, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S113, _S118, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S114, _S115, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S114, _S116, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S114, _S117, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S114, _S118, acc_1[3U][3U]);
        thread array<float, int(4)> av_7;
        float _S119 = (*kernelContext_4->tileA_0)[7U][ty_0];
        av_7[0U] = (*kernelContext_4->tileA_0)[7U][ty_0];
        float _S120 = (*kernelContext_4->tileA_0)[7U][_S58];
        av_7[1U] = (*kernelContext_4->tileA_0)[7U][_S58];
        float _S121 = (*kernelContext_4->tileA_0)[7U][_S60];
        av_7[2U] = (*kernelContext_4->tileA_0)[7U][_S60];
        float _S122 = (*kernelContext_4->tileA_0)[7U][_S62];
        av_7[3U] = (*kernelContext_4->tileA_0)[7U][_S62];
        thread array<float, int(4)> bv_7;
        float _S123 = (*kernelContext_4->tileB_0)[7U][tx_0];
        bv_7[0U] = (*kernelContext_4->tileB_0)[7U][tx_0];
        float _S124 = (*kernelContext_4->tileB_0)[7U][_S65];
        bv_7[1U] = (*kernelContext_4->tileB_0)[7U][_S65];
        float _S125 = (*kernelContext_4->tileB_0)[7U][_S67];
        bv_7[2U] = (*kernelContext_4->tileB_0)[7U][_S67];
        float _S126 = (*kernelContext_4->tileB_0)[7U][_S69];
        bv_7[3U] = (*kernelContext_4->tileB_0)[7U][_S69];
        acc_1[0U][0U] = fma(_S119, _S123, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S119, _S124, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S119, _S125, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S119, _S126, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S120, _S123, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S120, _S124, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S120, _S125, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S120, _S126, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S121, _S123, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S121, _S124, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S121, _S125, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S121, _S126, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S122, _S123, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S122, _S124, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S122, _S125, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S122, _S126, acc_1[3U][3U]);
        thread array<float, int(4)> av_8;
        float _S127 = (*kernelContext_4->tileA_0)[8U][ty_0];
        av_8[0U] = (*kernelContext_4->tileA_0)[8U][ty_0];
        float _S128 = (*kernelContext_4->tileA_0)[8U][_S58];
        av_8[1U] = (*kernelContext_4->tileA_0)[8U][_S58];
        float _S129 = (*kernelContext_4->tileA_0)[8U][_S60];
        av_8[2U] = (*kernelContext_4->tileA_0)[8U][_S60];
        float _S130 = (*kernelContext_4->tileA_0)[8U][_S62];
        av_8[3U] = (*kernelContext_4->tileA_0)[8U][_S62];
        thread array<float, int(4)> bv_8;
        float _S131 = (*kernelContext_4->tileB_0)[8U][tx_0];
        bv_8[0U] = (*kernelContext_4->tileB_0)[8U][tx_0];
        float _S132 = (*kernelContext_4->tileB_0)[8U][_S65];
        bv_8[1U] = (*kernelContext_4->tileB_0)[8U][_S65];
        float _S133 = (*kernelContext_4->tileB_0)[8U][_S67];
        bv_8[2U] = (*kernelContext_4->tileB_0)[8U][_S67];
        float _S134 = (*kernelContext_4->tileB_0)[8U][_S69];
        bv_8[3U] = (*kernelContext_4->tileB_0)[8U][_S69];
        acc_1[0U][0U] = fma(_S127, _S131, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S127, _S132, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S127, _S133, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S127, _S134, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S128, _S131, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S128, _S132, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S128, _S133, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S128, _S134, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S129, _S131, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S129, _S132, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S129, _S133, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S129, _S134, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S130, _S131, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S130, _S132, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S130, _S133, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S130, _S134, acc_1[3U][3U]);
        thread array<float, int(4)> av_9;
        float _S135 = (*kernelContext_4->tileA_0)[9U][ty_0];
        av_9[0U] = (*kernelContext_4->tileA_0)[9U][ty_0];
        float _S136 = (*kernelContext_4->tileA_0)[9U][_S58];
        av_9[1U] = (*kernelContext_4->tileA_0)[9U][_S58];
        float _S137 = (*kernelContext_4->tileA_0)[9U][_S60];
        av_9[2U] = (*kernelContext_4->tileA_0)[9U][_S60];
        float _S138 = (*kernelContext_4->tileA_0)[9U][_S62];
        av_9[3U] = (*kernelContext_4->tileA_0)[9U][_S62];
        thread array<float, int(4)> bv_9;
        float _S139 = (*kernelContext_4->tileB_0)[9U][tx_0];
        bv_9[0U] = (*kernelContext_4->tileB_0)[9U][tx_0];
        float _S140 = (*kernelContext_4->tileB_0)[9U][_S65];
        bv_9[1U] = (*kernelContext_4->tileB_0)[9U][_S65];
        float _S141 = (*kernelContext_4->tileB_0)[9U][_S67];
        bv_9[2U] = (*kernelContext_4->tileB_0)[9U][_S67];
        float _S142 = (*kernelContext_4->tileB_0)[9U][_S69];
        bv_9[3U] = (*kernelContext_4->tileB_0)[9U][_S69];
        acc_1[0U][0U] = fma(_S135, _S139, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S135, _S140, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S135, _S141, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S135, _S142, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S136, _S139, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S136, _S140, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S136, _S141, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S136, _S142, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S137, _S139, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S137, _S140, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S137, _S141, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S137, _S142, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S138, _S139, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S138, _S140, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S138, _S141, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S138, _S142, acc_1[3U][3U]);
        thread array<float, int(4)> av_10;
        float _S143 = (*kernelContext_4->tileA_0)[10U][ty_0];
        av_10[0U] = (*kernelContext_4->tileA_0)[10U][ty_0];
        float _S144 = (*kernelContext_4->tileA_0)[10U][_S58];
        av_10[1U] = (*kernelContext_4->tileA_0)[10U][_S58];
        float _S145 = (*kernelContext_4->tileA_0)[10U][_S60];
        av_10[2U] = (*kernelContext_4->tileA_0)[10U][_S60];
        float _S146 = (*kernelContext_4->tileA_0)[10U][_S62];
        av_10[3U] = (*kernelContext_4->tileA_0)[10U][_S62];
        thread array<float, int(4)> bv_10;
        float _S147 = (*kernelContext_4->tileB_0)[10U][tx_0];
        bv_10[0U] = (*kernelContext_4->tileB_0)[10U][tx_0];
        float _S148 = (*kernelContext_4->tileB_0)[10U][_S65];
        bv_10[1U] = (*kernelContext_4->tileB_0)[10U][_S65];
        float _S149 = (*kernelContext_4->tileB_0)[10U][_S67];
        bv_10[2U] = (*kernelContext_4->tileB_0)[10U][_S67];
        float _S150 = (*kernelContext_4->tileB_0)[10U][_S69];
        bv_10[3U] = (*kernelContext_4->tileB_0)[10U][_S69];
        acc_1[0U][0U] = fma(_S143, _S147, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S143, _S148, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S143, _S149, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S143, _S150, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S144, _S147, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S144, _S148, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S144, _S149, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S144, _S150, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S145, _S147, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S145, _S148, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S145, _S149, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S145, _S150, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S146, _S147, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S146, _S148, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S146, _S149, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S146, _S150, acc_1[3U][3U]);
        thread array<float, int(4)> av_11;
        float _S151 = (*kernelContext_4->tileA_0)[11U][ty_0];
        av_11[0U] = (*kernelContext_4->tileA_0)[11U][ty_0];
        float _S152 = (*kernelContext_4->tileA_0)[11U][_S58];
        av_11[1U] = (*kernelContext_4->tileA_0)[11U][_S58];
        float _S153 = (*kernelContext_4->tileA_0)[11U][_S60];
        av_11[2U] = (*kernelContext_4->tileA_0)[11U][_S60];
        float _S154 = (*kernelContext_4->tileA_0)[11U][_S62];
        av_11[3U] = (*kernelContext_4->tileA_0)[11U][_S62];
        thread array<float, int(4)> bv_11;
        float _S155 = (*kernelContext_4->tileB_0)[11U][tx_0];
        bv_11[0U] = (*kernelContext_4->tileB_0)[11U][tx_0];
        float _S156 = (*kernelContext_4->tileB_0)[11U][_S65];
        bv_11[1U] = (*kernelContext_4->tileB_0)[11U][_S65];
        float _S157 = (*kernelContext_4->tileB_0)[11U][_S67];
        bv_11[2U] = (*kernelContext_4->tileB_0)[11U][_S67];
        float _S158 = (*kernelContext_4->tileB_0)[11U][_S69];
        bv_11[3U] = (*kernelContext_4->tileB_0)[11U][_S69];
        acc_1[0U][0U] = fma(_S151, _S155, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S151, _S156, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S151, _S157, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S151, _S158, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S152, _S155, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S152, _S156, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S152, _S157, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S152, _S158, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S153, _S155, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S153, _S156, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S153, _S157, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S153, _S158, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S154, _S155, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S154, _S156, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S154, _S157, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S154, _S158, acc_1[3U][3U]);
        thread array<float, int(4)> av_12;
        float _S159 = (*kernelContext_4->tileA_0)[12U][ty_0];
        av_12[0U] = (*kernelContext_4->tileA_0)[12U][ty_0];
        float _S160 = (*kernelContext_4->tileA_0)[12U][_S58];
        av_12[1U] = (*kernelContext_4->tileA_0)[12U][_S58];
        float _S161 = (*kernelContext_4->tileA_0)[12U][_S60];
        av_12[2U] = (*kernelContext_4->tileA_0)[12U][_S60];
        float _S162 = (*kernelContext_4->tileA_0)[12U][_S62];
        av_12[3U] = (*kernelContext_4->tileA_0)[12U][_S62];
        thread array<float, int(4)> bv_12;
        float _S163 = (*kernelContext_4->tileB_0)[12U][tx_0];
        bv_12[0U] = (*kernelContext_4->tileB_0)[12U][tx_0];
        float _S164 = (*kernelContext_4->tileB_0)[12U][_S65];
        bv_12[1U] = (*kernelContext_4->tileB_0)[12U][_S65];
        float _S165 = (*kernelContext_4->tileB_0)[12U][_S67];
        bv_12[2U] = (*kernelContext_4->tileB_0)[12U][_S67];
        float _S166 = (*kernelContext_4->tileB_0)[12U][_S69];
        bv_12[3U] = (*kernelContext_4->tileB_0)[12U][_S69];
        acc_1[0U][0U] = fma(_S159, _S163, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S159, _S164, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S159, _S165, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S159, _S166, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S160, _S163, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S160, _S164, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S160, _S165, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S160, _S166, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S161, _S163, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S161, _S164, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S161, _S165, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S161, _S166, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S162, _S163, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S162, _S164, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S162, _S165, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S162, _S166, acc_1[3U][3U]);
        thread array<float, int(4)> av_13;
        float _S167 = (*kernelContext_4->tileA_0)[13U][ty_0];
        av_13[0U] = (*kernelContext_4->tileA_0)[13U][ty_0];
        float _S168 = (*kernelContext_4->tileA_0)[13U][_S58];
        av_13[1U] = (*kernelContext_4->tileA_0)[13U][_S58];
        float _S169 = (*kernelContext_4->tileA_0)[13U][_S60];
        av_13[2U] = (*kernelContext_4->tileA_0)[13U][_S60];
        float _S170 = (*kernelContext_4->tileA_0)[13U][_S62];
        av_13[3U] = (*kernelContext_4->tileA_0)[13U][_S62];
        thread array<float, int(4)> bv_13;
        float _S171 = (*kernelContext_4->tileB_0)[13U][tx_0];
        bv_13[0U] = (*kernelContext_4->tileB_0)[13U][tx_0];
        float _S172 = (*kernelContext_4->tileB_0)[13U][_S65];
        bv_13[1U] = (*kernelContext_4->tileB_0)[13U][_S65];
        float _S173 = (*kernelContext_4->tileB_0)[13U][_S67];
        bv_13[2U] = (*kernelContext_4->tileB_0)[13U][_S67];
        float _S174 = (*kernelContext_4->tileB_0)[13U][_S69];
        bv_13[3U] = (*kernelContext_4->tileB_0)[13U][_S69];
        acc_1[0U][0U] = fma(_S167, _S171, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S167, _S172, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S167, _S173, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S167, _S174, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S168, _S171, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S168, _S172, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S168, _S173, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S168, _S174, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S169, _S171, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S169, _S172, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S169, _S173, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S169, _S174, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S170, _S171, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S170, _S172, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S170, _S173, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S170, _S174, acc_1[3U][3U]);
        thread array<float, int(4)> av_14;
        float _S175 = (*kernelContext_4->tileA_0)[14U][ty_0];
        av_14[0U] = (*kernelContext_4->tileA_0)[14U][ty_0];
        float _S176 = (*kernelContext_4->tileA_0)[14U][_S58];
        av_14[1U] = (*kernelContext_4->tileA_0)[14U][_S58];
        float _S177 = (*kernelContext_4->tileA_0)[14U][_S60];
        av_14[2U] = (*kernelContext_4->tileA_0)[14U][_S60];
        float _S178 = (*kernelContext_4->tileA_0)[14U][_S62];
        av_14[3U] = (*kernelContext_4->tileA_0)[14U][_S62];
        thread array<float, int(4)> bv_14;
        float _S179 = (*kernelContext_4->tileB_0)[14U][tx_0];
        bv_14[0U] = (*kernelContext_4->tileB_0)[14U][tx_0];
        float _S180 = (*kernelContext_4->tileB_0)[14U][_S65];
        bv_14[1U] = (*kernelContext_4->tileB_0)[14U][_S65];
        float _S181 = (*kernelContext_4->tileB_0)[14U][_S67];
        bv_14[2U] = (*kernelContext_4->tileB_0)[14U][_S67];
        float _S182 = (*kernelContext_4->tileB_0)[14U][_S69];
        bv_14[3U] = (*kernelContext_4->tileB_0)[14U][_S69];
        acc_1[0U][0U] = fma(_S175, _S179, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S175, _S180, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S175, _S181, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S175, _S182, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S176, _S179, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S176, _S180, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S176, _S181, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S176, _S182, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S177, _S179, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S177, _S180, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S177, _S181, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S177, _S182, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S178, _S179, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S178, _S180, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S178, _S181, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S178, _S182, acc_1[3U][3U]);
        thread array<float, int(4)> av_15;
        float _S183 = (*kernelContext_4->tileA_0)[15U][ty_0];
        av_15[0U] = (*kernelContext_4->tileA_0)[15U][ty_0];
        float _S184 = (*kernelContext_4->tileA_0)[15U][_S58];
        av_15[1U] = (*kernelContext_4->tileA_0)[15U][_S58];
        float _S185 = (*kernelContext_4->tileA_0)[15U][_S60];
        av_15[2U] = (*kernelContext_4->tileA_0)[15U][_S60];
        float _S186 = (*kernelContext_4->tileA_0)[15U][_S62];
        av_15[3U] = (*kernelContext_4->tileA_0)[15U][_S62];
        thread array<float, int(4)> bv_15;
        float _S187 = (*kernelContext_4->tileB_0)[15U][tx_0];
        bv_15[0U] = (*kernelContext_4->tileB_0)[15U][tx_0];
        float _S188 = (*kernelContext_4->tileB_0)[15U][_S65];
        bv_15[1U] = (*kernelContext_4->tileB_0)[15U][_S65];
        float _S189 = (*kernelContext_4->tileB_0)[15U][_S67];
        bv_15[2U] = (*kernelContext_4->tileB_0)[15U][_S67];
        float _S190 = (*kernelContext_4->tileB_0)[15U][_S69];
        bv_15[3U] = (*kernelContext_4->tileB_0)[15U][_S69];
        acc_1[0U][0U] = fma(_S183, _S187, acc_1[0U][0U]);
        acc_1[0U][1U] = fma(_S183, _S188, acc_1[0U][1U]);
        acc_1[0U][2U] = fma(_S183, _S189, acc_1[0U][2U]);
        acc_1[0U][3U] = fma(_S183, _S190, acc_1[0U][3U]);
        acc_1[1U][0U] = fma(_S184, _S187, acc_1[1U][0U]);
        acc_1[1U][1U] = fma(_S184, _S188, acc_1[1U][1U]);
        acc_1[1U][2U] = fma(_S184, _S189, acc_1[1U][2U]);
        acc_1[1U][3U] = fma(_S184, _S190, acc_1[1U][3U]);
        acc_1[2U][0U] = fma(_S185, _S187, acc_1[2U][0U]);
        acc_1[2U][1U] = fma(_S185, _S188, acc_1[2U][1U]);
        acc_1[2U][2U] = fma(_S185, _S189, acc_1[2U][2U]);
        acc_1[2U][3U] = fma(_S185, _S190, acc_1[2U][3U]);
        acc_1[3U][0U] = fma(_S186, _S187, acc_1[3U][0U]);
        acc_1[3U][1U] = fma(_S186, _S188, acc_1[3U][1U]);
        acc_1[3U][2U] = fma(_S186, _S189, acc_1[3U][2U]);
        acc_1[3U][3U] = fma(_S186, _S190, acc_1[3U][3U]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k0_0 = k0_0 + 16U;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                uint _S191 = _S26 + ty_0;
                _S15 = _S191;
                _S16 = pr_0->m_2;
                if(_S191 >= (pr_0->m_2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S192 = _S27 + tx_0;
                            _S17 = _S192;
                            _S18 = pr_0->n_2;
                            if(_S192 >= (pr_0->n_2))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S191, _S192, acc_1[0U][0U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_0 = _S17 + 16U;
                            if(col_0 >= _S18)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S191, col_0, acc_1[0U][1U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_1 = _S17 + 32U;
                            if(col_1 >= _S18)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S191, col_1, acc_1[0U][2U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_2 = _S17 + 48U;
                            if(col_2 >= _S18)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, _S191, col_2, acc_1[0U][3U], kernelContext_4);
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
                uint r_2 = _S15 + 16U;
                if(r_2 >= _S16)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S193 = _S27 + tx_0;
                            _S19 = _S193;
                            _S20 = pr_0->n_2;
                            if(_S193 >= (pr_0->n_2))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, _S193, acc_1[1U][0U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_3 = _S19 + 16U;
                            if(col_3 >= _S20)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, col_3, acc_1[1U][1U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_4 = _S19 + 32U;
                            if(col_4 >= _S20)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, col_4, acc_1[1U][2U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_5 = _S19 + 48U;
                            if(col_5 >= _S20)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_2, col_5, acc_1[1U][3U], kernelContext_4);
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
                uint r_3 = _S15 + 32U;
                if(r_3 >= _S16)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S194 = _S27 + tx_0;
                            _S21 = _S194;
                            _S22 = pr_0->n_2;
                            if(_S194 >= (pr_0->n_2))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, _S194, acc_1[2U][0U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_6 = _S21 + 16U;
                            if(col_6 >= _S22)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, col_6, acc_1[2U][1U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_7 = _S21 + 32U;
                            if(col_7 >= _S22)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, col_7, acc_1[2U][2U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_8 = _S21 + 48U;
                            if(col_8 >= _S22)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_3, col_8, acc_1[2U][3U], kernelContext_4);
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
                uint r_4 = _S15 + 48U;
                if(r_4 >= _S16)
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S195 = _S27 + tx_0;
                            _S23 = _S195;
                            _S24 = pr_0->n_2;
                            if(_S195 >= (pr_0->n_2))
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, _S195, acc_1[3U][0U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_9 = _S23 + 16U;
                            if(col_9 >= _S24)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, col_9, acc_1[3U][1U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_10 = _S23 + 32U;
                            if(col_10 >= _S24)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, col_10, acc_1[3U][2U], kernelContext_4);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            uint col_11 = _S23 + 48U;
                            if(col_11 >= _S24)
                            {
                                break;
                            }
                            gemmStore_0(pr_0, r_4, col_11, acc_1[3U][3U], kernelContext_4);
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

[[kernel]] void gemm_grouped_bf16(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(8)]], ProblemRecord_0 device* problems_1 [[buffer(6)]], uint device* errors_2 [[buffer(7)]], uint device* a_2 [[buffer(0)]], uint device* b_2 [[buffer(1)]], float device* c_1 [[buffer(2)]], float device* epBias_1 [[buffer(3)]], float device* epGate_1 [[buffer(4)]], float device* epResidual_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->p_0 = p_1;
    (&kernelContext_5)->problems_0 = problems_1;
    (&kernelContext_5)->errors_1 = errors_2;
    (&kernelContext_5)->a_0 = a_2;
    (&kernelContext_5)->b_0 = b_2;
    (&kernelContext_5)->c_0 = c_1;
    (&kernelContext_5)->epBias_0 = epBias_1;
    (&kernelContext_5)->epGate_0 = epGate_1;
    (&kernelContext_5)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(64)>, int(16)> tileA_1;
    (&kernelContext_5)->tileA_0 = &tileA_1;
    threadgroup array<array<float, int(64)>, int(16)> tileB_1;
    (&kernelContext_5)->tileB_0 = &tileB_1;
    uint z_0 = group_0.z;
    if(z_0 >= (p_1->count_0))
    {
        return;
    }
    ProblemRecord_0 device* _S196 = (&kernelContext_5)->problems_0+z_0;
    ProblemRecord_0 r_5 = *_S196;
    uint _S197 = group_0.x;
    bool ok_0;
    if((_S197 * 64U) >= ((*_S196).n_1))
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
    bool transA_1 = ((&kernelContext_5)->p_0->transA_0) != 0U;
    bool transB_1 = ((&kernelContext_5)->p_0->transB_0) != 0U;
    uint _S198;
    if(transA_1)
    {
        _S198 = r_5.k_1;
    }
    else
    {
        _S198 = r_5.m_1;
    }
    uint _S199;
    if(transA_1)
    {
        _S199 = r_5.m_1;
    }
    else
    {
        _S199 = r_5.k_1;
    }
    bool _S200 = fits_0(r_5.offsetA_0, _S198, _S199, r_5.lda_1, (&kernelContext_5)->p_0->aLength_0);
    if(_S200)
    {
        if(transB_1)
        {
            _S198 = r_5.n_1;
        }
        else
        {
            _S198 = r_5.k_1;
        }
        if(transB_1)
        {
            _S199 = r_5.k_1;
        }
        else
        {
            _S199 = r_5.n_1;
        }
        bool _S201 = fits_0(r_5.offsetB_0, _S198, _S199, r_5.ldb_1, (&kernelContext_5)->p_0->bLength_0);
        ok_0 = _S201;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        bool _S202 = fits_0(r_5.offsetC_0, r_5.m_1, r_5.n_1, r_5.ldc_1, (&kernelContext_5)->p_0->cLength_0);
        ok_0 = _S202;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        ok_0 = (r_5.m_1) <= ((&kernelContext_5)->p_0->m_0);
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        ok_0 = (r_5.n_1) <= ((&kernelContext_5)->p_0->n_0);
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        if(((&kernelContext_5)->p_0->bias_0) == 0U)
        {
            ok_0 = true;
        }
        else
        {
            if(((&kernelContext_5)->p_0->bias_0) == 1U)
            {
                _S198 = r_5.n_1;
            }
            else
            {
                _S198 = r_5.m_1;
            }
            if(((&kernelContext_5)->p_0->bias_0) == 1U)
            {
                _S199 = r_5.n_1;
            }
            else
            {
                _S199 = r_5.m_1;
            }
            bool _S203 = fits_0(r_5.offsetBias_0, 1U, _S198, _S199, (&kernelContext_5)->p_0->biasLength_0);
            ok_0 = _S203;
        }
    }
    else
    {
        ok_0 = false;
    }
    if(!ok_0)
    {
        if(_S197 == 0U)
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
            dg_report_error_0((&kernelContext_5)->errors_1, 1U, z_0);
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
    uint2 _S204 = group_0.xy;
    thread GemmProblem_0 _S205 = pr_1;
    gemmTile_0(&_S205, _S204, local_1, &kernelContext_5);
    return;
}

