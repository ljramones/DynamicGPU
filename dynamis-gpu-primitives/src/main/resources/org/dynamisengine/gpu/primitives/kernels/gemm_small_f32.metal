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
    float device* a_0;
    float device* b_0;
    float device* c_0;
    float device* epBias_0;
    float device* epGate_0;
    float device* epResidual_0;
    array<array<float, int(16)>, int(16)> threadgroup* smallA_0;
    array<array<float, int(16)>, int(16)> threadgroup* smallB_0;
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
    return *(kernelContext_1->a_0+e_0);
}

float loadB_0(uint e_1, KernelContext_0 thread* kernelContext_2)
{
    return *(kernelContext_2->b_0+e_1);
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
        _S5 = r_0 + *(kernelContext_3->epBias_0+(biasBase_0 + j_0));
    }
    else
    {
        if((kernelContext_3->p_0->bias_0) == 2U)
        {
            _S5 = r_0 + *(kernelContext_3->epBias_0+(biasBase_0 + i_0));
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
        _S5 = _S5 * *(kernelContext_3->epGate_0+idx_0);
    }
    if((kernelContext_3->p_0->residual_0) != 0U)
    {
        _S5 = _S5 + *(kernelContext_3->epResidual_0+idx_0);
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
        r_1 = kernelContext_4->p_0->alpha_0 * acc_0 + kernelContext_4->p_0->beta_0 * *(kernelContext_4->c_0+idx_1);
    }
    float device* _S6 = kernelContext_4->c_0+idx_1;
    float _S7 = gemmEpilogue_0(r_1, i_1, j_1, idx_1, q_1->baseBias_0, kernelContext_4);
    *_S6 = _S7;
    return;
}

[[kernel]] void gemm_small_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], float device* a_2 [[buffer(0)]], float device* b_2 [[buffer(1)]], float device* c_1 [[buffer(2)]], float device* epBias_1 [[buffer(3)]], float device* epGate_1 [[buffer(4)]], float device* epResidual_1 [[buffer(5)]])
{
    bool _S8;
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->p_0 = p_1;
    (&kernelContext_5)->a_0 = a_2;
    (&kernelContext_5)->b_0 = b_2;
    (&kernelContext_5)->c_0 = c_1;
    (&kernelContext_5)->epBias_0 = epBias_1;
    (&kernelContext_5)->epGate_0 = epGate_1;
    (&kernelContext_5)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(16)>, int(16)> smallA_1;
    (&kernelContext_5)->smallA_0 = &smallA_1;
    threadgroup array<array<float, int(16)>, int(16)> smallB_1;
    (&kernelContext_5)->smallB_0 = &smallB_1;
    GemmProblem_0 _S9 = gemmBatchProblem_0(group_0.z, &kernelContext_5);
    uint tx_0 = local_0.x;
    uint ty_0 = local_0.y;
    uint row0_0 = group_0.y * 16U;
    uint col0_0 = group_0.x * 16U;
    uint k0_0 = 0U;
    float acc_1 = 0.0;
    for(;;)
    {
        if(k0_0 < (_S9.k_0))
        {
        }
        else
        {
            break;
        }
        uint gr_0 = row0_0 + ty_0;
        uint gk_0 = k0_0 + tx_0;
        uint ai_0;
        if(((&kernelContext_5)->p_0->transA_0) != 0U)
        {
            ai_0 = gk_0 * _S9.lda_0 + gr_0;
        }
        else
        {
            ai_0 = gr_0 * _S9.lda_0 + gk_0;
        }
        if(gr_0 < (_S9.m_0))
        {
            _S8 = gk_0 < (_S9.k_0);
        }
        else
        {
            _S8 = false;
        }
        float _S10;
        if(_S8)
        {
            float _S11 = loadA_0(_S9.baseA_0 + ai_0, &kernelContext_5);
            _S10 = _S11;
        }
        else
        {
            _S10 = 0.0;
        }
        (*(&kernelContext_5)->smallA_0)[tx_0][ty_0] = _S10;
        uint hk_0 = k0_0 + ty_0;
        uint gc_0 = col0_0 + tx_0;
        uint bi_0;
        if(((&kernelContext_5)->p_0->transB_0) != 0U)
        {
            bi_0 = gc_0 * _S9.ldb_0 + hk_0;
        }
        else
        {
            bi_0 = hk_0 * _S9.ldb_0 + gc_0;
        }
        bool _S12;
        if(hk_0 < (_S9.k_0))
        {
            _S12 = gc_0 < (_S9.n_0);
        }
        else
        {
            _S12 = false;
        }
        float _S13;
        if(_S12)
        {
            float _S14 = loadB_0(_S9.baseB_0 + bi_0, &kernelContext_5);
            _S13 = _S14;
        }
        else
        {
            _S13 = 0.0;
        }
        (*(&kernelContext_5)->smallB_0)[ty_0][tx_0] = _S13;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc_2 = fma((*(&kernelContext_5)->smallA_0)[15U][ty_0], (*(&kernelContext_5)->smallB_0)[15U][tx_0], fma((*(&kernelContext_5)->smallA_0)[14U][ty_0], (*(&kernelContext_5)->smallB_0)[14U][tx_0], fma((*(&kernelContext_5)->smallA_0)[13U][ty_0], (*(&kernelContext_5)->smallB_0)[13U][tx_0], fma((*(&kernelContext_5)->smallA_0)[12U][ty_0], (*(&kernelContext_5)->smallB_0)[12U][tx_0], fma((*(&kernelContext_5)->smallA_0)[11U][ty_0], (*(&kernelContext_5)->smallB_0)[11U][tx_0], fma((*(&kernelContext_5)->smallA_0)[10U][ty_0], (*(&kernelContext_5)->smallB_0)[10U][tx_0], fma((*(&kernelContext_5)->smallA_0)[9U][ty_0], (*(&kernelContext_5)->smallB_0)[9U][tx_0], fma((*(&kernelContext_5)->smallA_0)[8U][ty_0], (*(&kernelContext_5)->smallB_0)[8U][tx_0], fma((*(&kernelContext_5)->smallA_0)[7U][ty_0], (*(&kernelContext_5)->smallB_0)[7U][tx_0], fma((*(&kernelContext_5)->smallA_0)[6U][ty_0], (*(&kernelContext_5)->smallB_0)[6U][tx_0], fma((*(&kernelContext_5)->smallA_0)[5U][ty_0], (*(&kernelContext_5)->smallB_0)[5U][tx_0], fma((*(&kernelContext_5)->smallA_0)[4U][ty_0], (*(&kernelContext_5)->smallB_0)[4U][tx_0], fma((*(&kernelContext_5)->smallA_0)[3U][ty_0], (*(&kernelContext_5)->smallB_0)[3U][tx_0], fma((*(&kernelContext_5)->smallA_0)[2U][ty_0], (*(&kernelContext_5)->smallB_0)[2U][tx_0], fma((*(&kernelContext_5)->smallA_0)[1U][ty_0], (*(&kernelContext_5)->smallB_0)[1U][tx_0], fma((*(&kernelContext_5)->smallA_0)[0U][ty_0], (*(&kernelContext_5)->smallB_0)[0U][tx_0], acc_1))))))))))))))));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k0_0 = k0_0 + 16U;
        acc_1 = acc_2;
    }
    uint row_0 = row0_0 + ty_0;
    uint col_0 = col0_0 + tx_0;
    if(row_0 < (_S9.m_0))
    {
        _S8 = col_0 < (_S9.n_0);
    }
    else
    {
        _S8 = false;
    }
    if(_S8)
    {
        thread GemmProblem_0 _S15 = _S9;
        gemmStore_0(&_S15, row_0, col_0, acc_1, &kernelContext_5);
    }
    return;
}

