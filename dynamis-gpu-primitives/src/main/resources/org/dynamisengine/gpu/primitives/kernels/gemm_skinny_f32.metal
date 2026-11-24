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
    uint device* b_0;
    uint device* a_0;
    float device* c_0;
    float device* epBias_0;
    float device* epGate_0;
    float device* epResidual_0;
    array<array<float, int(256)>, int(4)> threadgroup* red_0;
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

void load8B_0(uint e_0, array<float, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_1)
{
    (*v_0)[0U] = (as_type<float>((*(kernelContext_1->b_0+e_0))));
    (*v_0)[1U] = (as_type<float>((*(kernelContext_1->b_0+(e_0 + 1U)))));
    (*v_0)[2U] = (as_type<float>((*(kernelContext_1->b_0+(e_0 + 2U)))));
    (*v_0)[3U] = (as_type<float>((*(kernelContext_1->b_0+(e_0 + 3U)))));
    (*v_0)[4U] = (as_type<float>((*(kernelContext_1->b_0+(e_0 + 4U)))));
    (*v_0)[5U] = (as_type<float>((*(kernelContext_1->b_0+(e_0 + 5U)))));
    (*v_0)[6U] = (as_type<float>((*(kernelContext_1->b_0+(e_0 + 6U)))));
    (*v_0)[7U] = (as_type<float>((*(kernelContext_1->b_0+(e_0 + 7U)))));
    return;
}

void load8A_0(uint e_1, array<float, int(8)> thread* v_1, KernelContext_0 thread* kernelContext_2)
{
    (*v_1)[0U] = (as_type<float>((*(kernelContext_2->a_0+e_1))));
    (*v_1)[1U] = (as_type<float>((*(kernelContext_2->a_0+(e_1 + 1U)))));
    (*v_1)[2U] = (as_type<float>((*(kernelContext_2->a_0+(e_1 + 2U)))));
    (*v_1)[3U] = (as_type<float>((*(kernelContext_2->a_0+(e_1 + 3U)))));
    (*v_1)[4U] = (as_type<float>((*(kernelContext_2->a_0+(e_1 + 4U)))));
    (*v_1)[5U] = (as_type<float>((*(kernelContext_2->a_0+(e_1 + 5U)))));
    (*v_1)[6U] = (as_type<float>((*(kernelContext_2->a_0+(e_1 + 6U)))));
    (*v_1)[7U] = (as_type<float>((*(kernelContext_2->a_0+(e_1 + 7U)))));
    return;
}

float elemB_0(uint e_2, KernelContext_0 thread* kernelContext_3)
{
    return (as_type<float>((*(kernelContext_3->b_0+e_2))));
}

float elemA_0(uint e_3, KernelContext_0 thread* kernelContext_4)
{
    return (as_type<float>((*(kernelContext_4->a_0+e_3))));
}

void load4B_0(uint e_4, array<float, int(4)> thread* v_2, KernelContext_0 thread* kernelContext_5)
{
    (*v_2)[0U] = (as_type<float>((*(kernelContext_5->b_0+e_4))));
    (*v_2)[1U] = (as_type<float>((*(kernelContext_5->b_0+(e_4 + 1U)))));
    (*v_2)[2U] = (as_type<float>((*(kernelContext_5->b_0+(e_4 + 2U)))));
    (*v_2)[3U] = (as_type<float>((*(kernelContext_5->b_0+(e_4 + 3U)))));
    return;
}

void load2A_0(uint e_5, float thread* v0_0, float thread* v1_0, KernelContext_0 thread* kernelContext_6)
{
    *v0_0 = (as_type<float>((*(kernelContext_6->a_0+e_5))));
    *v1_0 = (as_type<float>((*(kernelContext_6->a_0+(e_5 + 1U)))));
    return;
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

float gemmEpilogue_0(float r_0, uint i_0, uint j_0, uint idx_0, uint biasBase_0, KernelContext_0 thread* kernelContext_7)
{
    float _S5;
    if((kernelContext_7->p_0->bias_0) == 1U)
    {
        _S5 = r_0 + *(kernelContext_7->epBias_0+(biasBase_0 + j_0));
    }
    else
    {
        if((kernelContext_7->p_0->bias_0) == 2U)
        {
            _S5 = r_0 + *(kernelContext_7->epBias_0+(biasBase_0 + i_0));
        }
        else
        {
            _S5 = r_0;
        }
    }
    if((kernelContext_7->p_0->act_0) == 1U)
    {
        _S5 = dg_max_0(_S5, 0.0);
    }
    else
    {
        if((kernelContext_7->p_0->act_0) == 2U)
        {
            _S5 = 0.5 * _S5 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S5 + 0.04471499845385551 * _S5 * _S5 * _S5))) + 1.0)));
        }
        else
        {
            if((kernelContext_7->p_0->act_0) == 3U)
            {
                _S5 = _S5 / (1.0 + exp(- _S5));
            }
        }
    }
    if((kernelContext_7->p_0->gated_0) != 0U)
    {
        _S5 = _S5 * *(kernelContext_7->epGate_0+idx_0);
    }
    if((kernelContext_7->p_0->residual_0) != 0U)
    {
        _S5 = _S5 + *(kernelContext_7->epResidual_0+idx_0);
    }
    return _S5;
}

void gemmStore_0(const GemmProblem_0 thread* q_1, uint i_1, uint j_1, float acc_0, KernelContext_0 thread* kernelContext_8)
{
    uint idx_1 = q_1->baseC_0 + i_1 * q_1->ldc_0 + j_1;
    float r_1;
    if((kernelContext_8->p_0->beta_0) == 0.0)
    {
        r_1 = kernelContext_8->p_0->alpha_0 * acc_0;
    }
    else
    {
        r_1 = kernelContext_8->p_0->alpha_0 * acc_0 + kernelContext_8->p_0->beta_0 * *(kernelContext_8->c_0+idx_1);
    }
    float device* _S6 = kernelContext_8->c_0+idx_1;
    float _S7 = gemmEpilogue_0(r_1, i_1, j_1, idx_1, q_1->baseBias_0, kernelContext_8);
    *_S6 = _S7;
    return;
}

[[kernel]] void gemm_skinny_f32(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], uint device* b_2 [[buffer(1)]], uint device* a_2 [[buffer(0)]], float device* c_1 [[buffer(2)]], float device* epBias_1 [[buffer(3)]], float device* epGate_1 [[buffer(4)]], float device* epResidual_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_9;
    (&kernelContext_9)->p_0 = p_1;
    (&kernelContext_9)->b_0 = b_2;
    (&kernelContext_9)->a_0 = a_2;
    (&kernelContext_9)->c_0 = c_1;
    (&kernelContext_9)->epBias_0 = epBias_1;
    (&kernelContext_9)->epGate_0 = epGate_1;
    (&kernelContext_9)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(256)>, int(4)> red_1;
    (&kernelContext_9)->red_0 = &red_1;
    uint lid_0 = local_0.x;
    GemmProblem_0 _S8 = gemmBatchProblem_0(group_0.z, &kernelContext_9);
    uint _S9 = min((&kernelContext_9)->p_0->m_1, 16U);
    uint k_2 = (&kernelContext_9)->p_0->k_1;
    thread array<array<float, int(4)>, int(16)> acc_1;
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
    acc_1[4U][0U] = 0.0;
    acc_1[4U][1U] = 0.0;
    acc_1[4U][2U] = 0.0;
    acc_1[4U][3U] = 0.0;
    acc_1[5U][0U] = 0.0;
    acc_1[5U][1U] = 0.0;
    acc_1[5U][2U] = 0.0;
    acc_1[5U][3U] = 0.0;
    acc_1[6U][0U] = 0.0;
    acc_1[6U][1U] = 0.0;
    acc_1[6U][2U] = 0.0;
    acc_1[6U][3U] = 0.0;
    acc_1[7U][0U] = 0.0;
    acc_1[7U][1U] = 0.0;
    acc_1[7U][2U] = 0.0;
    acc_1[7U][3U] = 0.0;
    acc_1[8U][0U] = 0.0;
    acc_1[8U][1U] = 0.0;
    acc_1[8U][2U] = 0.0;
    acc_1[8U][3U] = 0.0;
    acc_1[9U][0U] = 0.0;
    acc_1[9U][1U] = 0.0;
    acc_1[9U][2U] = 0.0;
    acc_1[9U][3U] = 0.0;
    acc_1[10U][0U] = 0.0;
    acc_1[10U][1U] = 0.0;
    acc_1[10U][2U] = 0.0;
    acc_1[10U][3U] = 0.0;
    acc_1[11U][0U] = 0.0;
    acc_1[11U][1U] = 0.0;
    acc_1[11U][2U] = 0.0;
    acc_1[11U][3U] = 0.0;
    acc_1[12U][0U] = 0.0;
    acc_1[12U][1U] = 0.0;
    acc_1[12U][2U] = 0.0;
    acc_1[12U][3U] = 0.0;
    acc_1[13U][0U] = 0.0;
    acc_1[13U][1U] = 0.0;
    acc_1[13U][2U] = 0.0;
    acc_1[13U][3U] = 0.0;
    acc_1[14U][0U] = 0.0;
    acc_1[14U][1U] = 0.0;
    acc_1[14U][2U] = 0.0;
    acc_1[14U][3U] = 0.0;
    acc_1[15U][0U] = 0.0;
    acc_1[15U][1U] = 0.0;
    acc_1[15U][2U] = 0.0;
    acc_1[15U][3U] = 0.0;
    bool transB_1 = ((&kernelContext_9)->p_0->transB_0) != 0U;
    uint _S10;
    if(transB_1)
    {
        _S10 = 32U;
    }
    else
    {
        _S10 = 16U;
    }
    uint lane_0;
    if(transB_1)
    {
        lane_0 = lid_0 % 32U;
    }
    else
    {
        lane_0 = lid_0 / 16U;
    }
    uint _S11;
    if(transB_1)
    {
        _S11 = 2U;
    }
    else
    {
        _S11 = 4U;
    }
    uint col0_0;
    if(transB_1)
    {
        col0_0 = group_0.x * 16U + lid_0 / 32U * 2U;
    }
    else
    {
        col0_0 = group_0.x * 64U + lid_0 % 16U * 4U;
    }
    bool _S12;
    if(col0_0 < ((&kernelContext_9)->p_0->n_1))
    {
        _S12 = transB_1;
    }
    else
    {
        _S12 = false;
    }
    uint ch_0;
    uint i_2;
    int _S13;
    int _S14;
    int _S15;
    int _S16;
    int _S17;
    int _S18;
    int _S19;
    int _S20;
    int _S21;
    int _S22;
    int _S23;
    int _S24;
    int _S25;
    int _S26;
    int _S27;
    int _S28;
    int _S29;
    int _S30;
    int _S31;
    int _S32;
    int _S33;
    int _S34;
    int _S35;
    int _S36;
    int _S37;
    int _S38;
    int _S39;
    int _S40;
    int _S41;
    int _S42;
    int _S43;
    int _S44;
    if(_S12)
    {
        uint chunks_0 = k_2 / 8U;
        ch_0 = lane_0;
        for(;;)
        {
            if(ch_0 < chunks_0)
            {
            }
            else
            {
                break;
            }
            uint _S45 = ch_0 * 8U;
            thread array<array<float, int(8)>, int(2)> bv_0;
            load8B_0(_S8.baseB_0 + min(col0_0, (&kernelContext_9)->p_0->n_1 - 1U) * (&kernelContext_9)->p_0->ldb_1 + _S45, &bv_0[0U], &kernelContext_9);
            load8B_0(_S8.baseB_0 + min(col0_0 + 1U, (&kernelContext_9)->p_0->n_1 - 1U) * (&kernelContext_9)->p_0->ldb_1 + _S45, &bv_0[1U], &kernelContext_9);
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        if(0U >= _S9)
                        {
                            _S13 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_0;
                        load8A_0(_S8.baseA_0 + _S45, &av_0, &kernelContext_9);
                        acc_1[0U][0U] = fma(av_0[0U], bv_0[0U][0U], acc_1[0U][0U]);
                        acc_1[0U][1U] = fma(av_0[0U], bv_0[1U][0U], acc_1[0U][1U]);
                        acc_1[0U][0U] = fma(av_0[1U], bv_0[0U][1U], acc_1[0U][0U]);
                        acc_1[0U][1U] = fma(av_0[1U], bv_0[1U][1U], acc_1[0U][1U]);
                        acc_1[0U][0U] = fma(av_0[2U], bv_0[0U][2U], acc_1[0U][0U]);
                        acc_1[0U][1U] = fma(av_0[2U], bv_0[1U][2U], acc_1[0U][1U]);
                        acc_1[0U][0U] = fma(av_0[3U], bv_0[0U][3U], acc_1[0U][0U]);
                        acc_1[0U][1U] = fma(av_0[3U], bv_0[1U][3U], acc_1[0U][1U]);
                        acc_1[0U][0U] = fma(av_0[4U], bv_0[0U][4U], acc_1[0U][0U]);
                        acc_1[0U][1U] = fma(av_0[4U], bv_0[1U][4U], acc_1[0U][1U]);
                        acc_1[0U][0U] = fma(av_0[5U], bv_0[0U][5U], acc_1[0U][0U]);
                        acc_1[0U][1U] = fma(av_0[5U], bv_0[1U][5U], acc_1[0U][1U]);
                        acc_1[0U][0U] = fma(av_0[6U], bv_0[0U][6U], acc_1[0U][0U]);
                        acc_1[0U][1U] = fma(av_0[6U], bv_0[1U][6U], acc_1[0U][1U]);
                        acc_1[0U][0U] = fma(av_0[7U], bv_0[0U][7U], acc_1[0U][0U]);
                        acc_1[0U][1U] = fma(av_0[7U], bv_0[1U][7U], acc_1[0U][1U]);
                        _S13 = int(3);
                        break;
                    }
                    if(_S13 != int(3))
                    {
                        _S14 = _S13;
                        break;
                    }
                    _S14 = int(2);
                    break;
                }
                if(_S14 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(1U >= _S9)
                        {
                            _S15 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_1;
                        load8A_0(_S8.baseA_0 + (&kernelContext_9)->p_0->lda_1 + _S45, &av_1, &kernelContext_9);
                        acc_1[1U][0U] = fma(av_1[0U], bv_0[0U][0U], acc_1[1U][0U]);
                        acc_1[1U][1U] = fma(av_1[0U], bv_0[1U][0U], acc_1[1U][1U]);
                        acc_1[1U][0U] = fma(av_1[1U], bv_0[0U][1U], acc_1[1U][0U]);
                        acc_1[1U][1U] = fma(av_1[1U], bv_0[1U][1U], acc_1[1U][1U]);
                        acc_1[1U][0U] = fma(av_1[2U], bv_0[0U][2U], acc_1[1U][0U]);
                        acc_1[1U][1U] = fma(av_1[2U], bv_0[1U][2U], acc_1[1U][1U]);
                        acc_1[1U][0U] = fma(av_1[3U], bv_0[0U][3U], acc_1[1U][0U]);
                        acc_1[1U][1U] = fma(av_1[3U], bv_0[1U][3U], acc_1[1U][1U]);
                        acc_1[1U][0U] = fma(av_1[4U], bv_0[0U][4U], acc_1[1U][0U]);
                        acc_1[1U][1U] = fma(av_1[4U], bv_0[1U][4U], acc_1[1U][1U]);
                        acc_1[1U][0U] = fma(av_1[5U], bv_0[0U][5U], acc_1[1U][0U]);
                        acc_1[1U][1U] = fma(av_1[5U], bv_0[1U][5U], acc_1[1U][1U]);
                        acc_1[1U][0U] = fma(av_1[6U], bv_0[0U][6U], acc_1[1U][0U]);
                        acc_1[1U][1U] = fma(av_1[6U], bv_0[1U][6U], acc_1[1U][1U]);
                        acc_1[1U][0U] = fma(av_1[7U], bv_0[0U][7U], acc_1[1U][0U]);
                        acc_1[1U][1U] = fma(av_1[7U], bv_0[1U][7U], acc_1[1U][1U]);
                        _S15 = int(3);
                        break;
                    }
                    if(_S15 != int(3))
                    {
                        _S16 = _S15;
                        break;
                    }
                    _S16 = int(2);
                    break;
                }
                if(_S16 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(2U >= _S9)
                        {
                            _S17 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_2;
                        load8A_0(_S8.baseA_0 + 2U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_2, &kernelContext_9);
                        acc_1[2U][0U] = fma(av_2[0U], bv_0[0U][0U], acc_1[2U][0U]);
                        acc_1[2U][1U] = fma(av_2[0U], bv_0[1U][0U], acc_1[2U][1U]);
                        acc_1[2U][0U] = fma(av_2[1U], bv_0[0U][1U], acc_1[2U][0U]);
                        acc_1[2U][1U] = fma(av_2[1U], bv_0[1U][1U], acc_1[2U][1U]);
                        acc_1[2U][0U] = fma(av_2[2U], bv_0[0U][2U], acc_1[2U][0U]);
                        acc_1[2U][1U] = fma(av_2[2U], bv_0[1U][2U], acc_1[2U][1U]);
                        acc_1[2U][0U] = fma(av_2[3U], bv_0[0U][3U], acc_1[2U][0U]);
                        acc_1[2U][1U] = fma(av_2[3U], bv_0[1U][3U], acc_1[2U][1U]);
                        acc_1[2U][0U] = fma(av_2[4U], bv_0[0U][4U], acc_1[2U][0U]);
                        acc_1[2U][1U] = fma(av_2[4U], bv_0[1U][4U], acc_1[2U][1U]);
                        acc_1[2U][0U] = fma(av_2[5U], bv_0[0U][5U], acc_1[2U][0U]);
                        acc_1[2U][1U] = fma(av_2[5U], bv_0[1U][5U], acc_1[2U][1U]);
                        acc_1[2U][0U] = fma(av_2[6U], bv_0[0U][6U], acc_1[2U][0U]);
                        acc_1[2U][1U] = fma(av_2[6U], bv_0[1U][6U], acc_1[2U][1U]);
                        acc_1[2U][0U] = fma(av_2[7U], bv_0[0U][7U], acc_1[2U][0U]);
                        acc_1[2U][1U] = fma(av_2[7U], bv_0[1U][7U], acc_1[2U][1U]);
                        _S17 = int(3);
                        break;
                    }
                    if(_S17 != int(3))
                    {
                        _S18 = _S17;
                        break;
                    }
                    _S18 = int(2);
                    break;
                }
                if(_S18 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(3U >= _S9)
                        {
                            _S19 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_3;
                        load8A_0(_S8.baseA_0 + 3U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_3, &kernelContext_9);
                        acc_1[3U][0U] = fma(av_3[0U], bv_0[0U][0U], acc_1[3U][0U]);
                        acc_1[3U][1U] = fma(av_3[0U], bv_0[1U][0U], acc_1[3U][1U]);
                        acc_1[3U][0U] = fma(av_3[1U], bv_0[0U][1U], acc_1[3U][0U]);
                        acc_1[3U][1U] = fma(av_3[1U], bv_0[1U][1U], acc_1[3U][1U]);
                        acc_1[3U][0U] = fma(av_3[2U], bv_0[0U][2U], acc_1[3U][0U]);
                        acc_1[3U][1U] = fma(av_3[2U], bv_0[1U][2U], acc_1[3U][1U]);
                        acc_1[3U][0U] = fma(av_3[3U], bv_0[0U][3U], acc_1[3U][0U]);
                        acc_1[3U][1U] = fma(av_3[3U], bv_0[1U][3U], acc_1[3U][1U]);
                        acc_1[3U][0U] = fma(av_3[4U], bv_0[0U][4U], acc_1[3U][0U]);
                        acc_1[3U][1U] = fma(av_3[4U], bv_0[1U][4U], acc_1[3U][1U]);
                        acc_1[3U][0U] = fma(av_3[5U], bv_0[0U][5U], acc_1[3U][0U]);
                        acc_1[3U][1U] = fma(av_3[5U], bv_0[1U][5U], acc_1[3U][1U]);
                        acc_1[3U][0U] = fma(av_3[6U], bv_0[0U][6U], acc_1[3U][0U]);
                        acc_1[3U][1U] = fma(av_3[6U], bv_0[1U][6U], acc_1[3U][1U]);
                        acc_1[3U][0U] = fma(av_3[7U], bv_0[0U][7U], acc_1[3U][0U]);
                        acc_1[3U][1U] = fma(av_3[7U], bv_0[1U][7U], acc_1[3U][1U]);
                        _S19 = int(3);
                        break;
                    }
                    if(_S19 != int(3))
                    {
                        _S20 = _S19;
                        break;
                    }
                    _S20 = int(2);
                    break;
                }
                if(_S20 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(4U >= _S9)
                        {
                            _S21 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_4;
                        load8A_0(_S8.baseA_0 + 4U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_4, &kernelContext_9);
                        acc_1[4U][0U] = fma(av_4[0U], bv_0[0U][0U], acc_1[4U][0U]);
                        acc_1[4U][1U] = fma(av_4[0U], bv_0[1U][0U], acc_1[4U][1U]);
                        acc_1[4U][0U] = fma(av_4[1U], bv_0[0U][1U], acc_1[4U][0U]);
                        acc_1[4U][1U] = fma(av_4[1U], bv_0[1U][1U], acc_1[4U][1U]);
                        acc_1[4U][0U] = fma(av_4[2U], bv_0[0U][2U], acc_1[4U][0U]);
                        acc_1[4U][1U] = fma(av_4[2U], bv_0[1U][2U], acc_1[4U][1U]);
                        acc_1[4U][0U] = fma(av_4[3U], bv_0[0U][3U], acc_1[4U][0U]);
                        acc_1[4U][1U] = fma(av_4[3U], bv_0[1U][3U], acc_1[4U][1U]);
                        acc_1[4U][0U] = fma(av_4[4U], bv_0[0U][4U], acc_1[4U][0U]);
                        acc_1[4U][1U] = fma(av_4[4U], bv_0[1U][4U], acc_1[4U][1U]);
                        acc_1[4U][0U] = fma(av_4[5U], bv_0[0U][5U], acc_1[4U][0U]);
                        acc_1[4U][1U] = fma(av_4[5U], bv_0[1U][5U], acc_1[4U][1U]);
                        acc_1[4U][0U] = fma(av_4[6U], bv_0[0U][6U], acc_1[4U][0U]);
                        acc_1[4U][1U] = fma(av_4[6U], bv_0[1U][6U], acc_1[4U][1U]);
                        acc_1[4U][0U] = fma(av_4[7U], bv_0[0U][7U], acc_1[4U][0U]);
                        acc_1[4U][1U] = fma(av_4[7U], bv_0[1U][7U], acc_1[4U][1U]);
                        _S21 = int(3);
                        break;
                    }
                    if(_S21 != int(3))
                    {
                        _S22 = _S21;
                        break;
                    }
                    _S22 = int(2);
                    break;
                }
                if(_S22 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(5U >= _S9)
                        {
                            _S23 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_5;
                        load8A_0(_S8.baseA_0 + 5U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_5, &kernelContext_9);
                        acc_1[5U][0U] = fma(av_5[0U], bv_0[0U][0U], acc_1[5U][0U]);
                        acc_1[5U][1U] = fma(av_5[0U], bv_0[1U][0U], acc_1[5U][1U]);
                        acc_1[5U][0U] = fma(av_5[1U], bv_0[0U][1U], acc_1[5U][0U]);
                        acc_1[5U][1U] = fma(av_5[1U], bv_0[1U][1U], acc_1[5U][1U]);
                        acc_1[5U][0U] = fma(av_5[2U], bv_0[0U][2U], acc_1[5U][0U]);
                        acc_1[5U][1U] = fma(av_5[2U], bv_0[1U][2U], acc_1[5U][1U]);
                        acc_1[5U][0U] = fma(av_5[3U], bv_0[0U][3U], acc_1[5U][0U]);
                        acc_1[5U][1U] = fma(av_5[3U], bv_0[1U][3U], acc_1[5U][1U]);
                        acc_1[5U][0U] = fma(av_5[4U], bv_0[0U][4U], acc_1[5U][0U]);
                        acc_1[5U][1U] = fma(av_5[4U], bv_0[1U][4U], acc_1[5U][1U]);
                        acc_1[5U][0U] = fma(av_5[5U], bv_0[0U][5U], acc_1[5U][0U]);
                        acc_1[5U][1U] = fma(av_5[5U], bv_0[1U][5U], acc_1[5U][1U]);
                        acc_1[5U][0U] = fma(av_5[6U], bv_0[0U][6U], acc_1[5U][0U]);
                        acc_1[5U][1U] = fma(av_5[6U], bv_0[1U][6U], acc_1[5U][1U]);
                        acc_1[5U][0U] = fma(av_5[7U], bv_0[0U][7U], acc_1[5U][0U]);
                        acc_1[5U][1U] = fma(av_5[7U], bv_0[1U][7U], acc_1[5U][1U]);
                        _S23 = int(3);
                        break;
                    }
                    if(_S23 != int(3))
                    {
                        _S24 = _S23;
                        break;
                    }
                    _S24 = int(2);
                    break;
                }
                if(_S24 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(6U >= _S9)
                        {
                            _S25 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_6;
                        load8A_0(_S8.baseA_0 + 6U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_6, &kernelContext_9);
                        acc_1[6U][0U] = fma(av_6[0U], bv_0[0U][0U], acc_1[6U][0U]);
                        acc_1[6U][1U] = fma(av_6[0U], bv_0[1U][0U], acc_1[6U][1U]);
                        acc_1[6U][0U] = fma(av_6[1U], bv_0[0U][1U], acc_1[6U][0U]);
                        acc_1[6U][1U] = fma(av_6[1U], bv_0[1U][1U], acc_1[6U][1U]);
                        acc_1[6U][0U] = fma(av_6[2U], bv_0[0U][2U], acc_1[6U][0U]);
                        acc_1[6U][1U] = fma(av_6[2U], bv_0[1U][2U], acc_1[6U][1U]);
                        acc_1[6U][0U] = fma(av_6[3U], bv_0[0U][3U], acc_1[6U][0U]);
                        acc_1[6U][1U] = fma(av_6[3U], bv_0[1U][3U], acc_1[6U][1U]);
                        acc_1[6U][0U] = fma(av_6[4U], bv_0[0U][4U], acc_1[6U][0U]);
                        acc_1[6U][1U] = fma(av_6[4U], bv_0[1U][4U], acc_1[6U][1U]);
                        acc_1[6U][0U] = fma(av_6[5U], bv_0[0U][5U], acc_1[6U][0U]);
                        acc_1[6U][1U] = fma(av_6[5U], bv_0[1U][5U], acc_1[6U][1U]);
                        acc_1[6U][0U] = fma(av_6[6U], bv_0[0U][6U], acc_1[6U][0U]);
                        acc_1[6U][1U] = fma(av_6[6U], bv_0[1U][6U], acc_1[6U][1U]);
                        acc_1[6U][0U] = fma(av_6[7U], bv_0[0U][7U], acc_1[6U][0U]);
                        acc_1[6U][1U] = fma(av_6[7U], bv_0[1U][7U], acc_1[6U][1U]);
                        _S25 = int(3);
                        break;
                    }
                    if(_S25 != int(3))
                    {
                        _S26 = _S25;
                        break;
                    }
                    _S26 = int(2);
                    break;
                }
                if(_S26 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(7U >= _S9)
                        {
                            _S27 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_7;
                        load8A_0(_S8.baseA_0 + 7U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_7, &kernelContext_9);
                        acc_1[7U][0U] = fma(av_7[0U], bv_0[0U][0U], acc_1[7U][0U]);
                        acc_1[7U][1U] = fma(av_7[0U], bv_0[1U][0U], acc_1[7U][1U]);
                        acc_1[7U][0U] = fma(av_7[1U], bv_0[0U][1U], acc_1[7U][0U]);
                        acc_1[7U][1U] = fma(av_7[1U], bv_0[1U][1U], acc_1[7U][1U]);
                        acc_1[7U][0U] = fma(av_7[2U], bv_0[0U][2U], acc_1[7U][0U]);
                        acc_1[7U][1U] = fma(av_7[2U], bv_0[1U][2U], acc_1[7U][1U]);
                        acc_1[7U][0U] = fma(av_7[3U], bv_0[0U][3U], acc_1[7U][0U]);
                        acc_1[7U][1U] = fma(av_7[3U], bv_0[1U][3U], acc_1[7U][1U]);
                        acc_1[7U][0U] = fma(av_7[4U], bv_0[0U][4U], acc_1[7U][0U]);
                        acc_1[7U][1U] = fma(av_7[4U], bv_0[1U][4U], acc_1[7U][1U]);
                        acc_1[7U][0U] = fma(av_7[5U], bv_0[0U][5U], acc_1[7U][0U]);
                        acc_1[7U][1U] = fma(av_7[5U], bv_0[1U][5U], acc_1[7U][1U]);
                        acc_1[7U][0U] = fma(av_7[6U], bv_0[0U][6U], acc_1[7U][0U]);
                        acc_1[7U][1U] = fma(av_7[6U], bv_0[1U][6U], acc_1[7U][1U]);
                        acc_1[7U][0U] = fma(av_7[7U], bv_0[0U][7U], acc_1[7U][0U]);
                        acc_1[7U][1U] = fma(av_7[7U], bv_0[1U][7U], acc_1[7U][1U]);
                        _S27 = int(3);
                        break;
                    }
                    if(_S27 != int(3))
                    {
                        _S28 = _S27;
                        break;
                    }
                    _S28 = int(2);
                    break;
                }
                if(_S28 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(8U >= _S9)
                        {
                            _S29 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_8;
                        load8A_0(_S8.baseA_0 + 8U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_8, &kernelContext_9);
                        acc_1[8U][0U] = fma(av_8[0U], bv_0[0U][0U], acc_1[8U][0U]);
                        acc_1[8U][1U] = fma(av_8[0U], bv_0[1U][0U], acc_1[8U][1U]);
                        acc_1[8U][0U] = fma(av_8[1U], bv_0[0U][1U], acc_1[8U][0U]);
                        acc_1[8U][1U] = fma(av_8[1U], bv_0[1U][1U], acc_1[8U][1U]);
                        acc_1[8U][0U] = fma(av_8[2U], bv_0[0U][2U], acc_1[8U][0U]);
                        acc_1[8U][1U] = fma(av_8[2U], bv_0[1U][2U], acc_1[8U][1U]);
                        acc_1[8U][0U] = fma(av_8[3U], bv_0[0U][3U], acc_1[8U][0U]);
                        acc_1[8U][1U] = fma(av_8[3U], bv_0[1U][3U], acc_1[8U][1U]);
                        acc_1[8U][0U] = fma(av_8[4U], bv_0[0U][4U], acc_1[8U][0U]);
                        acc_1[8U][1U] = fma(av_8[4U], bv_0[1U][4U], acc_1[8U][1U]);
                        acc_1[8U][0U] = fma(av_8[5U], bv_0[0U][5U], acc_1[8U][0U]);
                        acc_1[8U][1U] = fma(av_8[5U], bv_0[1U][5U], acc_1[8U][1U]);
                        acc_1[8U][0U] = fma(av_8[6U], bv_0[0U][6U], acc_1[8U][0U]);
                        acc_1[8U][1U] = fma(av_8[6U], bv_0[1U][6U], acc_1[8U][1U]);
                        acc_1[8U][0U] = fma(av_8[7U], bv_0[0U][7U], acc_1[8U][0U]);
                        acc_1[8U][1U] = fma(av_8[7U], bv_0[1U][7U], acc_1[8U][1U]);
                        _S29 = int(3);
                        break;
                    }
                    if(_S29 != int(3))
                    {
                        _S30 = _S29;
                        break;
                    }
                    _S30 = int(2);
                    break;
                }
                if(_S30 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(9U >= _S9)
                        {
                            _S31 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_9;
                        load8A_0(_S8.baseA_0 + 9U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_9, &kernelContext_9);
                        acc_1[9U][0U] = fma(av_9[0U], bv_0[0U][0U], acc_1[9U][0U]);
                        acc_1[9U][1U] = fma(av_9[0U], bv_0[1U][0U], acc_1[9U][1U]);
                        acc_1[9U][0U] = fma(av_9[1U], bv_0[0U][1U], acc_1[9U][0U]);
                        acc_1[9U][1U] = fma(av_9[1U], bv_0[1U][1U], acc_1[9U][1U]);
                        acc_1[9U][0U] = fma(av_9[2U], bv_0[0U][2U], acc_1[9U][0U]);
                        acc_1[9U][1U] = fma(av_9[2U], bv_0[1U][2U], acc_1[9U][1U]);
                        acc_1[9U][0U] = fma(av_9[3U], bv_0[0U][3U], acc_1[9U][0U]);
                        acc_1[9U][1U] = fma(av_9[3U], bv_0[1U][3U], acc_1[9U][1U]);
                        acc_1[9U][0U] = fma(av_9[4U], bv_0[0U][4U], acc_1[9U][0U]);
                        acc_1[9U][1U] = fma(av_9[4U], bv_0[1U][4U], acc_1[9U][1U]);
                        acc_1[9U][0U] = fma(av_9[5U], bv_0[0U][5U], acc_1[9U][0U]);
                        acc_1[9U][1U] = fma(av_9[5U], bv_0[1U][5U], acc_1[9U][1U]);
                        acc_1[9U][0U] = fma(av_9[6U], bv_0[0U][6U], acc_1[9U][0U]);
                        acc_1[9U][1U] = fma(av_9[6U], bv_0[1U][6U], acc_1[9U][1U]);
                        acc_1[9U][0U] = fma(av_9[7U], bv_0[0U][7U], acc_1[9U][0U]);
                        acc_1[9U][1U] = fma(av_9[7U], bv_0[1U][7U], acc_1[9U][1U]);
                        _S31 = int(3);
                        break;
                    }
                    if(_S31 != int(3))
                    {
                        _S32 = _S31;
                        break;
                    }
                    _S32 = int(2);
                    break;
                }
                if(_S32 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(10U >= _S9)
                        {
                            _S33 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_10;
                        load8A_0(_S8.baseA_0 + 10U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_10, &kernelContext_9);
                        acc_1[10U][0U] = fma(av_10[0U], bv_0[0U][0U], acc_1[10U][0U]);
                        acc_1[10U][1U] = fma(av_10[0U], bv_0[1U][0U], acc_1[10U][1U]);
                        acc_1[10U][0U] = fma(av_10[1U], bv_0[0U][1U], acc_1[10U][0U]);
                        acc_1[10U][1U] = fma(av_10[1U], bv_0[1U][1U], acc_1[10U][1U]);
                        acc_1[10U][0U] = fma(av_10[2U], bv_0[0U][2U], acc_1[10U][0U]);
                        acc_1[10U][1U] = fma(av_10[2U], bv_0[1U][2U], acc_1[10U][1U]);
                        acc_1[10U][0U] = fma(av_10[3U], bv_0[0U][3U], acc_1[10U][0U]);
                        acc_1[10U][1U] = fma(av_10[3U], bv_0[1U][3U], acc_1[10U][1U]);
                        acc_1[10U][0U] = fma(av_10[4U], bv_0[0U][4U], acc_1[10U][0U]);
                        acc_1[10U][1U] = fma(av_10[4U], bv_0[1U][4U], acc_1[10U][1U]);
                        acc_1[10U][0U] = fma(av_10[5U], bv_0[0U][5U], acc_1[10U][0U]);
                        acc_1[10U][1U] = fma(av_10[5U], bv_0[1U][5U], acc_1[10U][1U]);
                        acc_1[10U][0U] = fma(av_10[6U], bv_0[0U][6U], acc_1[10U][0U]);
                        acc_1[10U][1U] = fma(av_10[6U], bv_0[1U][6U], acc_1[10U][1U]);
                        acc_1[10U][0U] = fma(av_10[7U], bv_0[0U][7U], acc_1[10U][0U]);
                        acc_1[10U][1U] = fma(av_10[7U], bv_0[1U][7U], acc_1[10U][1U]);
                        _S33 = int(3);
                        break;
                    }
                    if(_S33 != int(3))
                    {
                        _S34 = _S33;
                        break;
                    }
                    _S34 = int(2);
                    break;
                }
                if(_S34 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(11U >= _S9)
                        {
                            _S35 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_11;
                        load8A_0(_S8.baseA_0 + 11U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_11, &kernelContext_9);
                        acc_1[11U][0U] = fma(av_11[0U], bv_0[0U][0U], acc_1[11U][0U]);
                        acc_1[11U][1U] = fma(av_11[0U], bv_0[1U][0U], acc_1[11U][1U]);
                        acc_1[11U][0U] = fma(av_11[1U], bv_0[0U][1U], acc_1[11U][0U]);
                        acc_1[11U][1U] = fma(av_11[1U], bv_0[1U][1U], acc_1[11U][1U]);
                        acc_1[11U][0U] = fma(av_11[2U], bv_0[0U][2U], acc_1[11U][0U]);
                        acc_1[11U][1U] = fma(av_11[2U], bv_0[1U][2U], acc_1[11U][1U]);
                        acc_1[11U][0U] = fma(av_11[3U], bv_0[0U][3U], acc_1[11U][0U]);
                        acc_1[11U][1U] = fma(av_11[3U], bv_0[1U][3U], acc_1[11U][1U]);
                        acc_1[11U][0U] = fma(av_11[4U], bv_0[0U][4U], acc_1[11U][0U]);
                        acc_1[11U][1U] = fma(av_11[4U], bv_0[1U][4U], acc_1[11U][1U]);
                        acc_1[11U][0U] = fma(av_11[5U], bv_0[0U][5U], acc_1[11U][0U]);
                        acc_1[11U][1U] = fma(av_11[5U], bv_0[1U][5U], acc_1[11U][1U]);
                        acc_1[11U][0U] = fma(av_11[6U], bv_0[0U][6U], acc_1[11U][0U]);
                        acc_1[11U][1U] = fma(av_11[6U], bv_0[1U][6U], acc_1[11U][1U]);
                        acc_1[11U][0U] = fma(av_11[7U], bv_0[0U][7U], acc_1[11U][0U]);
                        acc_1[11U][1U] = fma(av_11[7U], bv_0[1U][7U], acc_1[11U][1U]);
                        _S35 = int(3);
                        break;
                    }
                    if(_S35 != int(3))
                    {
                        _S36 = _S35;
                        break;
                    }
                    _S36 = int(2);
                    break;
                }
                if(_S36 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(12U >= _S9)
                        {
                            _S37 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_12;
                        load8A_0(_S8.baseA_0 + 12U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_12, &kernelContext_9);
                        acc_1[12U][0U] = fma(av_12[0U], bv_0[0U][0U], acc_1[12U][0U]);
                        acc_1[12U][1U] = fma(av_12[0U], bv_0[1U][0U], acc_1[12U][1U]);
                        acc_1[12U][0U] = fma(av_12[1U], bv_0[0U][1U], acc_1[12U][0U]);
                        acc_1[12U][1U] = fma(av_12[1U], bv_0[1U][1U], acc_1[12U][1U]);
                        acc_1[12U][0U] = fma(av_12[2U], bv_0[0U][2U], acc_1[12U][0U]);
                        acc_1[12U][1U] = fma(av_12[2U], bv_0[1U][2U], acc_1[12U][1U]);
                        acc_1[12U][0U] = fma(av_12[3U], bv_0[0U][3U], acc_1[12U][0U]);
                        acc_1[12U][1U] = fma(av_12[3U], bv_0[1U][3U], acc_1[12U][1U]);
                        acc_1[12U][0U] = fma(av_12[4U], bv_0[0U][4U], acc_1[12U][0U]);
                        acc_1[12U][1U] = fma(av_12[4U], bv_0[1U][4U], acc_1[12U][1U]);
                        acc_1[12U][0U] = fma(av_12[5U], bv_0[0U][5U], acc_1[12U][0U]);
                        acc_1[12U][1U] = fma(av_12[5U], bv_0[1U][5U], acc_1[12U][1U]);
                        acc_1[12U][0U] = fma(av_12[6U], bv_0[0U][6U], acc_1[12U][0U]);
                        acc_1[12U][1U] = fma(av_12[6U], bv_0[1U][6U], acc_1[12U][1U]);
                        acc_1[12U][0U] = fma(av_12[7U], bv_0[0U][7U], acc_1[12U][0U]);
                        acc_1[12U][1U] = fma(av_12[7U], bv_0[1U][7U], acc_1[12U][1U]);
                        _S37 = int(3);
                        break;
                    }
                    if(_S37 != int(3))
                    {
                        _S38 = _S37;
                        break;
                    }
                    _S38 = int(2);
                    break;
                }
                if(_S38 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(13U >= _S9)
                        {
                            _S39 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_13;
                        load8A_0(_S8.baseA_0 + 13U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_13, &kernelContext_9);
                        acc_1[13U][0U] = fma(av_13[0U], bv_0[0U][0U], acc_1[13U][0U]);
                        acc_1[13U][1U] = fma(av_13[0U], bv_0[1U][0U], acc_1[13U][1U]);
                        acc_1[13U][0U] = fma(av_13[1U], bv_0[0U][1U], acc_1[13U][0U]);
                        acc_1[13U][1U] = fma(av_13[1U], bv_0[1U][1U], acc_1[13U][1U]);
                        acc_1[13U][0U] = fma(av_13[2U], bv_0[0U][2U], acc_1[13U][0U]);
                        acc_1[13U][1U] = fma(av_13[2U], bv_0[1U][2U], acc_1[13U][1U]);
                        acc_1[13U][0U] = fma(av_13[3U], bv_0[0U][3U], acc_1[13U][0U]);
                        acc_1[13U][1U] = fma(av_13[3U], bv_0[1U][3U], acc_1[13U][1U]);
                        acc_1[13U][0U] = fma(av_13[4U], bv_0[0U][4U], acc_1[13U][0U]);
                        acc_1[13U][1U] = fma(av_13[4U], bv_0[1U][4U], acc_1[13U][1U]);
                        acc_1[13U][0U] = fma(av_13[5U], bv_0[0U][5U], acc_1[13U][0U]);
                        acc_1[13U][1U] = fma(av_13[5U], bv_0[1U][5U], acc_1[13U][1U]);
                        acc_1[13U][0U] = fma(av_13[6U], bv_0[0U][6U], acc_1[13U][0U]);
                        acc_1[13U][1U] = fma(av_13[6U], bv_0[1U][6U], acc_1[13U][1U]);
                        acc_1[13U][0U] = fma(av_13[7U], bv_0[0U][7U], acc_1[13U][0U]);
                        acc_1[13U][1U] = fma(av_13[7U], bv_0[1U][7U], acc_1[13U][1U]);
                        _S39 = int(3);
                        break;
                    }
                    if(_S39 != int(3))
                    {
                        _S40 = _S39;
                        break;
                    }
                    _S40 = int(2);
                    break;
                }
                if(_S40 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(14U >= _S9)
                        {
                            _S41 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_14;
                        load8A_0(_S8.baseA_0 + 14U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_14, &kernelContext_9);
                        acc_1[14U][0U] = fma(av_14[0U], bv_0[0U][0U], acc_1[14U][0U]);
                        acc_1[14U][1U] = fma(av_14[0U], bv_0[1U][0U], acc_1[14U][1U]);
                        acc_1[14U][0U] = fma(av_14[1U], bv_0[0U][1U], acc_1[14U][0U]);
                        acc_1[14U][1U] = fma(av_14[1U], bv_0[1U][1U], acc_1[14U][1U]);
                        acc_1[14U][0U] = fma(av_14[2U], bv_0[0U][2U], acc_1[14U][0U]);
                        acc_1[14U][1U] = fma(av_14[2U], bv_0[1U][2U], acc_1[14U][1U]);
                        acc_1[14U][0U] = fma(av_14[3U], bv_0[0U][3U], acc_1[14U][0U]);
                        acc_1[14U][1U] = fma(av_14[3U], bv_0[1U][3U], acc_1[14U][1U]);
                        acc_1[14U][0U] = fma(av_14[4U], bv_0[0U][4U], acc_1[14U][0U]);
                        acc_1[14U][1U] = fma(av_14[4U], bv_0[1U][4U], acc_1[14U][1U]);
                        acc_1[14U][0U] = fma(av_14[5U], bv_0[0U][5U], acc_1[14U][0U]);
                        acc_1[14U][1U] = fma(av_14[5U], bv_0[1U][5U], acc_1[14U][1U]);
                        acc_1[14U][0U] = fma(av_14[6U], bv_0[0U][6U], acc_1[14U][0U]);
                        acc_1[14U][1U] = fma(av_14[6U], bv_0[1U][6U], acc_1[14U][1U]);
                        acc_1[14U][0U] = fma(av_14[7U], bv_0[0U][7U], acc_1[14U][0U]);
                        acc_1[14U][1U] = fma(av_14[7U], bv_0[1U][7U], acc_1[14U][1U]);
                        _S41 = int(3);
                        break;
                    }
                    if(_S41 != int(3))
                    {
                        _S42 = _S41;
                        break;
                    }
                    _S42 = int(2);
                    break;
                }
                if(_S42 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(15U >= _S9)
                        {
                            _S43 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_15;
                        load8A_0(_S8.baseA_0 + 15U * (&kernelContext_9)->p_0->lda_1 + _S45, &av_15, &kernelContext_9);
                        acc_1[15U][0U] = fma(av_15[0U], bv_0[0U][0U], acc_1[15U][0U]);
                        acc_1[15U][1U] = fma(av_15[0U], bv_0[1U][0U], acc_1[15U][1U]);
                        acc_1[15U][0U] = fma(av_15[1U], bv_0[0U][1U], acc_1[15U][0U]);
                        acc_1[15U][1U] = fma(av_15[1U], bv_0[1U][1U], acc_1[15U][1U]);
                        acc_1[15U][0U] = fma(av_15[2U], bv_0[0U][2U], acc_1[15U][0U]);
                        acc_1[15U][1U] = fma(av_15[2U], bv_0[1U][2U], acc_1[15U][1U]);
                        acc_1[15U][0U] = fma(av_15[3U], bv_0[0U][3U], acc_1[15U][0U]);
                        acc_1[15U][1U] = fma(av_15[3U], bv_0[1U][3U], acc_1[15U][1U]);
                        acc_1[15U][0U] = fma(av_15[4U], bv_0[0U][4U], acc_1[15U][0U]);
                        acc_1[15U][1U] = fma(av_15[4U], bv_0[1U][4U], acc_1[15U][1U]);
                        acc_1[15U][0U] = fma(av_15[5U], bv_0[0U][5U], acc_1[15U][0U]);
                        acc_1[15U][1U] = fma(av_15[5U], bv_0[1U][5U], acc_1[15U][1U]);
                        acc_1[15U][0U] = fma(av_15[6U], bv_0[0U][6U], acc_1[15U][0U]);
                        acc_1[15U][1U] = fma(av_15[6U], bv_0[1U][6U], acc_1[15U][1U]);
                        acc_1[15U][0U] = fma(av_15[7U], bv_0[0U][7U], acc_1[15U][0U]);
                        acc_1[15U][1U] = fma(av_15[7U], bv_0[1U][7U], acc_1[15U][1U]);
                        _S43 = int(3);
                        break;
                    }
                    if(_S43 != int(3))
                    {
                        _S44 = _S43;
                        break;
                    }
                    _S44 = int(2);
                    break;
                }
                if(_S44 != int(2))
                {
                    break;
                }
                break;
            }
            ch_0 = ch_0 + 32U;
        }
        if((k_2 % 8U) != 0U)
        {
            _S12 = lane_0 == (chunks_0 % 32U);
        }
        else
        {
            _S12 = false;
        }
        if(_S12)
        {
            i_2 = chunks_0 * 8U;
            for(;;)
            {
                if(i_2 < k_2)
                {
                }
                else
                {
                    break;
                }
                thread array<float, int(2)> bv_1;
                float _S46 = elemB_0(_S8.baseB_0 + min(col0_0, (&kernelContext_9)->p_0->n_1 - 1U) * (&kernelContext_9)->p_0->ldb_1 + i_2, &kernelContext_9);
                bv_1[0U] = _S46;
                float _S47 = elemB_0(_S8.baseB_0 + min(col0_0 + 1U, (&kernelContext_9)->p_0->n_1 - 1U) * (&kernelContext_9)->p_0->ldb_1 + i_2, &kernelContext_9);
                bv_1[1U] = _S47;
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            if(0U >= _S9)
                            {
                                _S13 = int(1);
                                break;
                            }
                            float _S48 = elemA_0(_S8.baseA_0 + i_2, &kernelContext_9);
                            acc_1[0U][0U] = fma(_S48, bv_1[0U], acc_1[0U][0U]);
                            acc_1[0U][1U] = fma(_S48, bv_1[1U], acc_1[0U][1U]);
                            _S13 = int(3);
                            break;
                        }
                        if(_S13 != int(3))
                        {
                            _S14 = _S13;
                            break;
                        }
                        _S14 = int(2);
                        break;
                    }
                    if(_S14 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(1U >= _S9)
                            {
                                _S15 = int(1);
                                break;
                            }
                            float _S49 = elemA_0(_S8.baseA_0 + (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[1U][0U] = fma(_S49, bv_1[0U], acc_1[1U][0U]);
                            acc_1[1U][1U] = fma(_S49, bv_1[1U], acc_1[1U][1U]);
                            _S15 = int(3);
                            break;
                        }
                        if(_S15 != int(3))
                        {
                            _S16 = _S15;
                            break;
                        }
                        _S16 = int(2);
                        break;
                    }
                    if(_S16 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(2U >= _S9)
                            {
                                _S17 = int(1);
                                break;
                            }
                            float _S50 = elemA_0(_S8.baseA_0 + 2U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[2U][0U] = fma(_S50, bv_1[0U], acc_1[2U][0U]);
                            acc_1[2U][1U] = fma(_S50, bv_1[1U], acc_1[2U][1U]);
                            _S17 = int(3);
                            break;
                        }
                        if(_S17 != int(3))
                        {
                            _S18 = _S17;
                            break;
                        }
                        _S18 = int(2);
                        break;
                    }
                    if(_S18 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(3U >= _S9)
                            {
                                _S19 = int(1);
                                break;
                            }
                            float _S51 = elemA_0(_S8.baseA_0 + 3U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[3U][0U] = fma(_S51, bv_1[0U], acc_1[3U][0U]);
                            acc_1[3U][1U] = fma(_S51, bv_1[1U], acc_1[3U][1U]);
                            _S19 = int(3);
                            break;
                        }
                        if(_S19 != int(3))
                        {
                            _S20 = _S19;
                            break;
                        }
                        _S20 = int(2);
                        break;
                    }
                    if(_S20 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(4U >= _S9)
                            {
                                _S21 = int(1);
                                break;
                            }
                            float _S52 = elemA_0(_S8.baseA_0 + 4U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[4U][0U] = fma(_S52, bv_1[0U], acc_1[4U][0U]);
                            acc_1[4U][1U] = fma(_S52, bv_1[1U], acc_1[4U][1U]);
                            _S21 = int(3);
                            break;
                        }
                        if(_S21 != int(3))
                        {
                            _S22 = _S21;
                            break;
                        }
                        _S22 = int(2);
                        break;
                    }
                    if(_S22 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(5U >= _S9)
                            {
                                _S23 = int(1);
                                break;
                            }
                            float _S53 = elemA_0(_S8.baseA_0 + 5U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[5U][0U] = fma(_S53, bv_1[0U], acc_1[5U][0U]);
                            acc_1[5U][1U] = fma(_S53, bv_1[1U], acc_1[5U][1U]);
                            _S23 = int(3);
                            break;
                        }
                        if(_S23 != int(3))
                        {
                            _S24 = _S23;
                            break;
                        }
                        _S24 = int(2);
                        break;
                    }
                    if(_S24 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(6U >= _S9)
                            {
                                _S25 = int(1);
                                break;
                            }
                            float _S54 = elemA_0(_S8.baseA_0 + 6U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[6U][0U] = fma(_S54, bv_1[0U], acc_1[6U][0U]);
                            acc_1[6U][1U] = fma(_S54, bv_1[1U], acc_1[6U][1U]);
                            _S25 = int(3);
                            break;
                        }
                        if(_S25 != int(3))
                        {
                            _S26 = _S25;
                            break;
                        }
                        _S26 = int(2);
                        break;
                    }
                    if(_S26 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(7U >= _S9)
                            {
                                _S27 = int(1);
                                break;
                            }
                            float _S55 = elemA_0(_S8.baseA_0 + 7U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[7U][0U] = fma(_S55, bv_1[0U], acc_1[7U][0U]);
                            acc_1[7U][1U] = fma(_S55, bv_1[1U], acc_1[7U][1U]);
                            _S27 = int(3);
                            break;
                        }
                        if(_S27 != int(3))
                        {
                            _S28 = _S27;
                            break;
                        }
                        _S28 = int(2);
                        break;
                    }
                    if(_S28 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(8U >= _S9)
                            {
                                _S29 = int(1);
                                break;
                            }
                            float _S56 = elemA_0(_S8.baseA_0 + 8U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[8U][0U] = fma(_S56, bv_1[0U], acc_1[8U][0U]);
                            acc_1[8U][1U] = fma(_S56, bv_1[1U], acc_1[8U][1U]);
                            _S29 = int(3);
                            break;
                        }
                        if(_S29 != int(3))
                        {
                            _S30 = _S29;
                            break;
                        }
                        _S30 = int(2);
                        break;
                    }
                    if(_S30 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(9U >= _S9)
                            {
                                _S31 = int(1);
                                break;
                            }
                            float _S57 = elemA_0(_S8.baseA_0 + 9U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[9U][0U] = fma(_S57, bv_1[0U], acc_1[9U][0U]);
                            acc_1[9U][1U] = fma(_S57, bv_1[1U], acc_1[9U][1U]);
                            _S31 = int(3);
                            break;
                        }
                        if(_S31 != int(3))
                        {
                            _S32 = _S31;
                            break;
                        }
                        _S32 = int(2);
                        break;
                    }
                    if(_S32 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(10U >= _S9)
                            {
                                _S33 = int(1);
                                break;
                            }
                            float _S58 = elemA_0(_S8.baseA_0 + 10U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[10U][0U] = fma(_S58, bv_1[0U], acc_1[10U][0U]);
                            acc_1[10U][1U] = fma(_S58, bv_1[1U], acc_1[10U][1U]);
                            _S33 = int(3);
                            break;
                        }
                        if(_S33 != int(3))
                        {
                            _S34 = _S33;
                            break;
                        }
                        _S34 = int(2);
                        break;
                    }
                    if(_S34 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(11U >= _S9)
                            {
                                _S35 = int(1);
                                break;
                            }
                            float _S59 = elemA_0(_S8.baseA_0 + 11U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[11U][0U] = fma(_S59, bv_1[0U], acc_1[11U][0U]);
                            acc_1[11U][1U] = fma(_S59, bv_1[1U], acc_1[11U][1U]);
                            _S35 = int(3);
                            break;
                        }
                        if(_S35 != int(3))
                        {
                            _S36 = _S35;
                            break;
                        }
                        _S36 = int(2);
                        break;
                    }
                    if(_S36 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(12U >= _S9)
                            {
                                _S37 = int(1);
                                break;
                            }
                            float _S60 = elemA_0(_S8.baseA_0 + 12U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[12U][0U] = fma(_S60, bv_1[0U], acc_1[12U][0U]);
                            acc_1[12U][1U] = fma(_S60, bv_1[1U], acc_1[12U][1U]);
                            _S37 = int(3);
                            break;
                        }
                        if(_S37 != int(3))
                        {
                            _S38 = _S37;
                            break;
                        }
                        _S38 = int(2);
                        break;
                    }
                    if(_S38 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(13U >= _S9)
                            {
                                _S39 = int(1);
                                break;
                            }
                            float _S61 = elemA_0(_S8.baseA_0 + 13U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[13U][0U] = fma(_S61, bv_1[0U], acc_1[13U][0U]);
                            acc_1[13U][1U] = fma(_S61, bv_1[1U], acc_1[13U][1U]);
                            _S39 = int(3);
                            break;
                        }
                        if(_S39 != int(3))
                        {
                            _S40 = _S39;
                            break;
                        }
                        _S40 = int(2);
                        break;
                    }
                    if(_S40 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(14U >= _S9)
                            {
                                _S41 = int(1);
                                break;
                            }
                            float _S62 = elemA_0(_S8.baseA_0 + 14U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[14U][0U] = fma(_S62, bv_1[0U], acc_1[14U][0U]);
                            acc_1[14U][1U] = fma(_S62, bv_1[1U], acc_1[14U][1U]);
                            _S41 = int(3);
                            break;
                        }
                        if(_S41 != int(3))
                        {
                            _S42 = _S41;
                            break;
                        }
                        _S42 = int(2);
                        break;
                    }
                    if(_S42 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(15U >= _S9)
                            {
                                _S43 = int(1);
                                break;
                            }
                            float _S63 = elemA_0(_S8.baseA_0 + 15U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[15U][0U] = fma(_S63, bv_1[0U], acc_1[15U][0U]);
                            acc_1[15U][1U] = fma(_S63, bv_1[1U], acc_1[15U][1U]);
                            _S43 = int(3);
                            break;
                        }
                        if(_S43 != int(3))
                        {
                            _S44 = _S43;
                            break;
                        }
                        _S44 = int(2);
                        break;
                    }
                    if(_S44 != int(2))
                    {
                        break;
                    }
                    break;
                }
                i_2 = i_2 + 1U;
            }
        }
    }
    else
    {
        if(col0_0 < ((&kernelContext_9)->p_0->n_1))
        {
            bool _S64 = (col0_0 + 4U) <= ((&kernelContext_9)->p_0->n_1);
            ch_0 = lane_0;
            for(;;)
            {
                uint _S65 = 2U * ch_0;
                if(_S65 < k_2)
                {
                }
                else
                {
                    break;
                }
                uint _S66 = _S65 + 1U;
                bool _S67 = _S66 < k_2;
                thread array<float, int(4)> b0_0;
                thread array<float, int(4)> b1_0;
                uint _S68 = k_2 - 1U;
                uint e_6 = _S8.baseB_0 + min(_S65, _S68) * (&kernelContext_9)->p_0->ldb_1 + col0_0;
                thread array<float, int(4)> v_3;
                if(_S64)
                {
                    load4B_0(e_6, &v_3, &kernelContext_9);
                }
                else
                {
                    float _S69 = elemB_0(e_6 + min(0U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_3[0U] = _S69;
                    float _S70 = elemB_0(e_6 + min(1U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_3[1U] = _S70;
                    float _S71 = elemB_0(e_6 + min(2U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_3[2U] = _S71;
                    float _S72 = elemB_0(e_6 + min(3U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_3[3U] = _S72;
                }
                b0_0[0U] = v_3[0U];
                b0_0[1U] = v_3[1U];
                b0_0[2U] = v_3[2U];
                b0_0[3U] = v_3[3U];
                uint e_7 = _S8.baseB_0 + min(_S66, _S68) * (&kernelContext_9)->p_0->ldb_1 + col0_0;
                thread array<float, int(4)> v_4;
                if(_S64)
                {
                    load4B_0(e_7, &v_4, &kernelContext_9);
                }
                else
                {
                    float _S73 = elemB_0(e_7 + min(0U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_4[0U] = _S73;
                    float _S74 = elemB_0(e_7 + min(1U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_4[1U] = _S74;
                    float _S75 = elemB_0(e_7 + min(2U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_4[2U] = _S75;
                    float _S76 = elemB_0(e_7 + min(3U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_4[3U] = _S76;
                }
                b1_0[0U] = v_4[0U];
                b1_0[1U] = v_4[1U];
                b1_0[2U] = v_4[2U];
                b1_0[3U] = v_4[3U];
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            if(0U >= _S9)
                            {
                                _S13 = int(1);
                                break;
                            }
                            uint e_8 = _S8.baseA_0 + _S65;
                            thread float a0_0;
                            thread float a1_0 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_8, &a0_0, &a1_0, &kernelContext_9);
                            }
                            else
                            {
                                float _S77 = elemA_0(e_8, &kernelContext_9);
                                a0_0 = _S77;
                            }
                            acc_1[0U][0U] = fma(a0_0, b0_0[0U], acc_1[0U][0U]);
                            if(_S67)
                            {
                                acc_1[0U][0U] = fma(a1_0, b1_0[0U], acc_1[0U][0U]);
                            }
                            acc_1[0U][1U] = fma(a0_0, b0_0[1U], acc_1[0U][1U]);
                            if(_S67)
                            {
                                acc_1[0U][1U] = fma(a1_0, b1_0[1U], acc_1[0U][1U]);
                            }
                            acc_1[0U][2U] = fma(a0_0, b0_0[2U], acc_1[0U][2U]);
                            if(_S67)
                            {
                                acc_1[0U][2U] = fma(a1_0, b1_0[2U], acc_1[0U][2U]);
                            }
                            acc_1[0U][3U] = fma(a0_0, b0_0[3U], acc_1[0U][3U]);
                            if(_S67)
                            {
                                acc_1[0U][3U] = fma(a1_0, b1_0[3U], acc_1[0U][3U]);
                            }
                            _S13 = int(3);
                            break;
                        }
                        if(_S13 != int(3))
                        {
                            _S14 = _S13;
                            break;
                        }
                        _S14 = int(2);
                        break;
                    }
                    if(_S14 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(1U >= _S9)
                            {
                                _S15 = int(1);
                                break;
                            }
                            uint e_9 = _S8.baseA_0 + (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_1;
                            thread float a1_1 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_9, &a0_1, &a1_1, &kernelContext_9);
                            }
                            else
                            {
                                float _S78 = elemA_0(e_9, &kernelContext_9);
                                a0_1 = _S78;
                            }
                            acc_1[1U][0U] = fma(a0_1, b0_0[0U], acc_1[1U][0U]);
                            if(_S67)
                            {
                                acc_1[1U][0U] = fma(a1_1, b1_0[0U], acc_1[1U][0U]);
                            }
                            acc_1[1U][1U] = fma(a0_1, b0_0[1U], acc_1[1U][1U]);
                            if(_S67)
                            {
                                acc_1[1U][1U] = fma(a1_1, b1_0[1U], acc_1[1U][1U]);
                            }
                            acc_1[1U][2U] = fma(a0_1, b0_0[2U], acc_1[1U][2U]);
                            if(_S67)
                            {
                                acc_1[1U][2U] = fma(a1_1, b1_0[2U], acc_1[1U][2U]);
                            }
                            acc_1[1U][3U] = fma(a0_1, b0_0[3U], acc_1[1U][3U]);
                            if(_S67)
                            {
                                acc_1[1U][3U] = fma(a1_1, b1_0[3U], acc_1[1U][3U]);
                            }
                            _S15 = int(3);
                            break;
                        }
                        if(_S15 != int(3))
                        {
                            _S16 = _S15;
                            break;
                        }
                        _S16 = int(2);
                        break;
                    }
                    if(_S16 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(2U >= _S9)
                            {
                                _S17 = int(1);
                                break;
                            }
                            uint e_10 = _S8.baseA_0 + 2U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_2;
                            thread float a1_2 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_10, &a0_2, &a1_2, &kernelContext_9);
                            }
                            else
                            {
                                float _S79 = elemA_0(e_10, &kernelContext_9);
                                a0_2 = _S79;
                            }
                            acc_1[2U][0U] = fma(a0_2, b0_0[0U], acc_1[2U][0U]);
                            if(_S67)
                            {
                                acc_1[2U][0U] = fma(a1_2, b1_0[0U], acc_1[2U][0U]);
                            }
                            acc_1[2U][1U] = fma(a0_2, b0_0[1U], acc_1[2U][1U]);
                            if(_S67)
                            {
                                acc_1[2U][1U] = fma(a1_2, b1_0[1U], acc_1[2U][1U]);
                            }
                            acc_1[2U][2U] = fma(a0_2, b0_0[2U], acc_1[2U][2U]);
                            if(_S67)
                            {
                                acc_1[2U][2U] = fma(a1_2, b1_0[2U], acc_1[2U][2U]);
                            }
                            acc_1[2U][3U] = fma(a0_2, b0_0[3U], acc_1[2U][3U]);
                            if(_S67)
                            {
                                acc_1[2U][3U] = fma(a1_2, b1_0[3U], acc_1[2U][3U]);
                            }
                            _S17 = int(3);
                            break;
                        }
                        if(_S17 != int(3))
                        {
                            _S18 = _S17;
                            break;
                        }
                        _S18 = int(2);
                        break;
                    }
                    if(_S18 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(3U >= _S9)
                            {
                                _S19 = int(1);
                                break;
                            }
                            uint e_11 = _S8.baseA_0 + 3U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_3;
                            thread float a1_3 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_11, &a0_3, &a1_3, &kernelContext_9);
                            }
                            else
                            {
                                float _S80 = elemA_0(e_11, &kernelContext_9);
                                a0_3 = _S80;
                            }
                            acc_1[3U][0U] = fma(a0_3, b0_0[0U], acc_1[3U][0U]);
                            if(_S67)
                            {
                                acc_1[3U][0U] = fma(a1_3, b1_0[0U], acc_1[3U][0U]);
                            }
                            acc_1[3U][1U] = fma(a0_3, b0_0[1U], acc_1[3U][1U]);
                            if(_S67)
                            {
                                acc_1[3U][1U] = fma(a1_3, b1_0[1U], acc_1[3U][1U]);
                            }
                            acc_1[3U][2U] = fma(a0_3, b0_0[2U], acc_1[3U][2U]);
                            if(_S67)
                            {
                                acc_1[3U][2U] = fma(a1_3, b1_0[2U], acc_1[3U][2U]);
                            }
                            acc_1[3U][3U] = fma(a0_3, b0_0[3U], acc_1[3U][3U]);
                            if(_S67)
                            {
                                acc_1[3U][3U] = fma(a1_3, b1_0[3U], acc_1[3U][3U]);
                            }
                            _S19 = int(3);
                            break;
                        }
                        if(_S19 != int(3))
                        {
                            _S20 = _S19;
                            break;
                        }
                        _S20 = int(2);
                        break;
                    }
                    if(_S20 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(4U >= _S9)
                            {
                                _S21 = int(1);
                                break;
                            }
                            uint e_12 = _S8.baseA_0 + 4U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_4;
                            thread float a1_4 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_12, &a0_4, &a1_4, &kernelContext_9);
                            }
                            else
                            {
                                float _S81 = elemA_0(e_12, &kernelContext_9);
                                a0_4 = _S81;
                            }
                            acc_1[4U][0U] = fma(a0_4, b0_0[0U], acc_1[4U][0U]);
                            if(_S67)
                            {
                                acc_1[4U][0U] = fma(a1_4, b1_0[0U], acc_1[4U][0U]);
                            }
                            acc_1[4U][1U] = fma(a0_4, b0_0[1U], acc_1[4U][1U]);
                            if(_S67)
                            {
                                acc_1[4U][1U] = fma(a1_4, b1_0[1U], acc_1[4U][1U]);
                            }
                            acc_1[4U][2U] = fma(a0_4, b0_0[2U], acc_1[4U][2U]);
                            if(_S67)
                            {
                                acc_1[4U][2U] = fma(a1_4, b1_0[2U], acc_1[4U][2U]);
                            }
                            acc_1[4U][3U] = fma(a0_4, b0_0[3U], acc_1[4U][3U]);
                            if(_S67)
                            {
                                acc_1[4U][3U] = fma(a1_4, b1_0[3U], acc_1[4U][3U]);
                            }
                            _S21 = int(3);
                            break;
                        }
                        if(_S21 != int(3))
                        {
                            _S22 = _S21;
                            break;
                        }
                        _S22 = int(2);
                        break;
                    }
                    if(_S22 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(5U >= _S9)
                            {
                                _S23 = int(1);
                                break;
                            }
                            uint e_13 = _S8.baseA_0 + 5U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_5;
                            thread float a1_5 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_13, &a0_5, &a1_5, &kernelContext_9);
                            }
                            else
                            {
                                float _S82 = elemA_0(e_13, &kernelContext_9);
                                a0_5 = _S82;
                            }
                            acc_1[5U][0U] = fma(a0_5, b0_0[0U], acc_1[5U][0U]);
                            if(_S67)
                            {
                                acc_1[5U][0U] = fma(a1_5, b1_0[0U], acc_1[5U][0U]);
                            }
                            acc_1[5U][1U] = fma(a0_5, b0_0[1U], acc_1[5U][1U]);
                            if(_S67)
                            {
                                acc_1[5U][1U] = fma(a1_5, b1_0[1U], acc_1[5U][1U]);
                            }
                            acc_1[5U][2U] = fma(a0_5, b0_0[2U], acc_1[5U][2U]);
                            if(_S67)
                            {
                                acc_1[5U][2U] = fma(a1_5, b1_0[2U], acc_1[5U][2U]);
                            }
                            acc_1[5U][3U] = fma(a0_5, b0_0[3U], acc_1[5U][3U]);
                            if(_S67)
                            {
                                acc_1[5U][3U] = fma(a1_5, b1_0[3U], acc_1[5U][3U]);
                            }
                            _S23 = int(3);
                            break;
                        }
                        if(_S23 != int(3))
                        {
                            _S24 = _S23;
                            break;
                        }
                        _S24 = int(2);
                        break;
                    }
                    if(_S24 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(6U >= _S9)
                            {
                                _S25 = int(1);
                                break;
                            }
                            uint e_14 = _S8.baseA_0 + 6U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_6;
                            thread float a1_6 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_14, &a0_6, &a1_6, &kernelContext_9);
                            }
                            else
                            {
                                float _S83 = elemA_0(e_14, &kernelContext_9);
                                a0_6 = _S83;
                            }
                            acc_1[6U][0U] = fma(a0_6, b0_0[0U], acc_1[6U][0U]);
                            if(_S67)
                            {
                                acc_1[6U][0U] = fma(a1_6, b1_0[0U], acc_1[6U][0U]);
                            }
                            acc_1[6U][1U] = fma(a0_6, b0_0[1U], acc_1[6U][1U]);
                            if(_S67)
                            {
                                acc_1[6U][1U] = fma(a1_6, b1_0[1U], acc_1[6U][1U]);
                            }
                            acc_1[6U][2U] = fma(a0_6, b0_0[2U], acc_1[6U][2U]);
                            if(_S67)
                            {
                                acc_1[6U][2U] = fma(a1_6, b1_0[2U], acc_1[6U][2U]);
                            }
                            acc_1[6U][3U] = fma(a0_6, b0_0[3U], acc_1[6U][3U]);
                            if(_S67)
                            {
                                acc_1[6U][3U] = fma(a1_6, b1_0[3U], acc_1[6U][3U]);
                            }
                            _S25 = int(3);
                            break;
                        }
                        if(_S25 != int(3))
                        {
                            _S26 = _S25;
                            break;
                        }
                        _S26 = int(2);
                        break;
                    }
                    if(_S26 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(7U >= _S9)
                            {
                                _S27 = int(1);
                                break;
                            }
                            uint e_15 = _S8.baseA_0 + 7U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_7;
                            thread float a1_7 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_15, &a0_7, &a1_7, &kernelContext_9);
                            }
                            else
                            {
                                float _S84 = elemA_0(e_15, &kernelContext_9);
                                a0_7 = _S84;
                            }
                            acc_1[7U][0U] = fma(a0_7, b0_0[0U], acc_1[7U][0U]);
                            if(_S67)
                            {
                                acc_1[7U][0U] = fma(a1_7, b1_0[0U], acc_1[7U][0U]);
                            }
                            acc_1[7U][1U] = fma(a0_7, b0_0[1U], acc_1[7U][1U]);
                            if(_S67)
                            {
                                acc_1[7U][1U] = fma(a1_7, b1_0[1U], acc_1[7U][1U]);
                            }
                            acc_1[7U][2U] = fma(a0_7, b0_0[2U], acc_1[7U][2U]);
                            if(_S67)
                            {
                                acc_1[7U][2U] = fma(a1_7, b1_0[2U], acc_1[7U][2U]);
                            }
                            acc_1[7U][3U] = fma(a0_7, b0_0[3U], acc_1[7U][3U]);
                            if(_S67)
                            {
                                acc_1[7U][3U] = fma(a1_7, b1_0[3U], acc_1[7U][3U]);
                            }
                            _S27 = int(3);
                            break;
                        }
                        if(_S27 != int(3))
                        {
                            _S28 = _S27;
                            break;
                        }
                        _S28 = int(2);
                        break;
                    }
                    if(_S28 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(8U >= _S9)
                            {
                                _S29 = int(1);
                                break;
                            }
                            uint e_16 = _S8.baseA_0 + 8U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_8;
                            thread float a1_8 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_16, &a0_8, &a1_8, &kernelContext_9);
                            }
                            else
                            {
                                float _S85 = elemA_0(e_16, &kernelContext_9);
                                a0_8 = _S85;
                            }
                            acc_1[8U][0U] = fma(a0_8, b0_0[0U], acc_1[8U][0U]);
                            if(_S67)
                            {
                                acc_1[8U][0U] = fma(a1_8, b1_0[0U], acc_1[8U][0U]);
                            }
                            acc_1[8U][1U] = fma(a0_8, b0_0[1U], acc_1[8U][1U]);
                            if(_S67)
                            {
                                acc_1[8U][1U] = fma(a1_8, b1_0[1U], acc_1[8U][1U]);
                            }
                            acc_1[8U][2U] = fma(a0_8, b0_0[2U], acc_1[8U][2U]);
                            if(_S67)
                            {
                                acc_1[8U][2U] = fma(a1_8, b1_0[2U], acc_1[8U][2U]);
                            }
                            acc_1[8U][3U] = fma(a0_8, b0_0[3U], acc_1[8U][3U]);
                            if(_S67)
                            {
                                acc_1[8U][3U] = fma(a1_8, b1_0[3U], acc_1[8U][3U]);
                            }
                            _S29 = int(3);
                            break;
                        }
                        if(_S29 != int(3))
                        {
                            _S30 = _S29;
                            break;
                        }
                        _S30 = int(2);
                        break;
                    }
                    if(_S30 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(9U >= _S9)
                            {
                                _S31 = int(1);
                                break;
                            }
                            uint e_17 = _S8.baseA_0 + 9U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_9;
                            thread float a1_9 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_17, &a0_9, &a1_9, &kernelContext_9);
                            }
                            else
                            {
                                float _S86 = elemA_0(e_17, &kernelContext_9);
                                a0_9 = _S86;
                            }
                            acc_1[9U][0U] = fma(a0_9, b0_0[0U], acc_1[9U][0U]);
                            if(_S67)
                            {
                                acc_1[9U][0U] = fma(a1_9, b1_0[0U], acc_1[9U][0U]);
                            }
                            acc_1[9U][1U] = fma(a0_9, b0_0[1U], acc_1[9U][1U]);
                            if(_S67)
                            {
                                acc_1[9U][1U] = fma(a1_9, b1_0[1U], acc_1[9U][1U]);
                            }
                            acc_1[9U][2U] = fma(a0_9, b0_0[2U], acc_1[9U][2U]);
                            if(_S67)
                            {
                                acc_1[9U][2U] = fma(a1_9, b1_0[2U], acc_1[9U][2U]);
                            }
                            acc_1[9U][3U] = fma(a0_9, b0_0[3U], acc_1[9U][3U]);
                            if(_S67)
                            {
                                acc_1[9U][3U] = fma(a1_9, b1_0[3U], acc_1[9U][3U]);
                            }
                            _S31 = int(3);
                            break;
                        }
                        if(_S31 != int(3))
                        {
                            _S32 = _S31;
                            break;
                        }
                        _S32 = int(2);
                        break;
                    }
                    if(_S32 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(10U >= _S9)
                            {
                                _S33 = int(1);
                                break;
                            }
                            uint e_18 = _S8.baseA_0 + 10U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_10;
                            thread float a1_10 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_18, &a0_10, &a1_10, &kernelContext_9);
                            }
                            else
                            {
                                float _S87 = elemA_0(e_18, &kernelContext_9);
                                a0_10 = _S87;
                            }
                            acc_1[10U][0U] = fma(a0_10, b0_0[0U], acc_1[10U][0U]);
                            if(_S67)
                            {
                                acc_1[10U][0U] = fma(a1_10, b1_0[0U], acc_1[10U][0U]);
                            }
                            acc_1[10U][1U] = fma(a0_10, b0_0[1U], acc_1[10U][1U]);
                            if(_S67)
                            {
                                acc_1[10U][1U] = fma(a1_10, b1_0[1U], acc_1[10U][1U]);
                            }
                            acc_1[10U][2U] = fma(a0_10, b0_0[2U], acc_1[10U][2U]);
                            if(_S67)
                            {
                                acc_1[10U][2U] = fma(a1_10, b1_0[2U], acc_1[10U][2U]);
                            }
                            acc_1[10U][3U] = fma(a0_10, b0_0[3U], acc_1[10U][3U]);
                            if(_S67)
                            {
                                acc_1[10U][3U] = fma(a1_10, b1_0[3U], acc_1[10U][3U]);
                            }
                            _S33 = int(3);
                            break;
                        }
                        if(_S33 != int(3))
                        {
                            _S34 = _S33;
                            break;
                        }
                        _S34 = int(2);
                        break;
                    }
                    if(_S34 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(11U >= _S9)
                            {
                                _S35 = int(1);
                                break;
                            }
                            uint e_19 = _S8.baseA_0 + 11U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_11;
                            thread float a1_11 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_19, &a0_11, &a1_11, &kernelContext_9);
                            }
                            else
                            {
                                float _S88 = elemA_0(e_19, &kernelContext_9);
                                a0_11 = _S88;
                            }
                            acc_1[11U][0U] = fma(a0_11, b0_0[0U], acc_1[11U][0U]);
                            if(_S67)
                            {
                                acc_1[11U][0U] = fma(a1_11, b1_0[0U], acc_1[11U][0U]);
                            }
                            acc_1[11U][1U] = fma(a0_11, b0_0[1U], acc_1[11U][1U]);
                            if(_S67)
                            {
                                acc_1[11U][1U] = fma(a1_11, b1_0[1U], acc_1[11U][1U]);
                            }
                            acc_1[11U][2U] = fma(a0_11, b0_0[2U], acc_1[11U][2U]);
                            if(_S67)
                            {
                                acc_1[11U][2U] = fma(a1_11, b1_0[2U], acc_1[11U][2U]);
                            }
                            acc_1[11U][3U] = fma(a0_11, b0_0[3U], acc_1[11U][3U]);
                            if(_S67)
                            {
                                acc_1[11U][3U] = fma(a1_11, b1_0[3U], acc_1[11U][3U]);
                            }
                            _S35 = int(3);
                            break;
                        }
                        if(_S35 != int(3))
                        {
                            _S36 = _S35;
                            break;
                        }
                        _S36 = int(2);
                        break;
                    }
                    if(_S36 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(12U >= _S9)
                            {
                                _S37 = int(1);
                                break;
                            }
                            uint e_20 = _S8.baseA_0 + 12U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_12;
                            thread float a1_12 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_20, &a0_12, &a1_12, &kernelContext_9);
                            }
                            else
                            {
                                float _S89 = elemA_0(e_20, &kernelContext_9);
                                a0_12 = _S89;
                            }
                            acc_1[12U][0U] = fma(a0_12, b0_0[0U], acc_1[12U][0U]);
                            if(_S67)
                            {
                                acc_1[12U][0U] = fma(a1_12, b1_0[0U], acc_1[12U][0U]);
                            }
                            acc_1[12U][1U] = fma(a0_12, b0_0[1U], acc_1[12U][1U]);
                            if(_S67)
                            {
                                acc_1[12U][1U] = fma(a1_12, b1_0[1U], acc_1[12U][1U]);
                            }
                            acc_1[12U][2U] = fma(a0_12, b0_0[2U], acc_1[12U][2U]);
                            if(_S67)
                            {
                                acc_1[12U][2U] = fma(a1_12, b1_0[2U], acc_1[12U][2U]);
                            }
                            acc_1[12U][3U] = fma(a0_12, b0_0[3U], acc_1[12U][3U]);
                            if(_S67)
                            {
                                acc_1[12U][3U] = fma(a1_12, b1_0[3U], acc_1[12U][3U]);
                            }
                            _S37 = int(3);
                            break;
                        }
                        if(_S37 != int(3))
                        {
                            _S38 = _S37;
                            break;
                        }
                        _S38 = int(2);
                        break;
                    }
                    if(_S38 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(13U >= _S9)
                            {
                                _S39 = int(1);
                                break;
                            }
                            uint e_21 = _S8.baseA_0 + 13U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_13;
                            thread float a1_13 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_21, &a0_13, &a1_13, &kernelContext_9);
                            }
                            else
                            {
                                float _S90 = elemA_0(e_21, &kernelContext_9);
                                a0_13 = _S90;
                            }
                            acc_1[13U][0U] = fma(a0_13, b0_0[0U], acc_1[13U][0U]);
                            if(_S67)
                            {
                                acc_1[13U][0U] = fma(a1_13, b1_0[0U], acc_1[13U][0U]);
                            }
                            acc_1[13U][1U] = fma(a0_13, b0_0[1U], acc_1[13U][1U]);
                            if(_S67)
                            {
                                acc_1[13U][1U] = fma(a1_13, b1_0[1U], acc_1[13U][1U]);
                            }
                            acc_1[13U][2U] = fma(a0_13, b0_0[2U], acc_1[13U][2U]);
                            if(_S67)
                            {
                                acc_1[13U][2U] = fma(a1_13, b1_0[2U], acc_1[13U][2U]);
                            }
                            acc_1[13U][3U] = fma(a0_13, b0_0[3U], acc_1[13U][3U]);
                            if(_S67)
                            {
                                acc_1[13U][3U] = fma(a1_13, b1_0[3U], acc_1[13U][3U]);
                            }
                            _S39 = int(3);
                            break;
                        }
                        if(_S39 != int(3))
                        {
                            _S40 = _S39;
                            break;
                        }
                        _S40 = int(2);
                        break;
                    }
                    if(_S40 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(14U >= _S9)
                            {
                                _S41 = int(1);
                                break;
                            }
                            uint e_22 = _S8.baseA_0 + 14U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_14;
                            thread float a1_14 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_22, &a0_14, &a1_14, &kernelContext_9);
                            }
                            else
                            {
                                float _S91 = elemA_0(e_22, &kernelContext_9);
                                a0_14 = _S91;
                            }
                            acc_1[14U][0U] = fma(a0_14, b0_0[0U], acc_1[14U][0U]);
                            if(_S67)
                            {
                                acc_1[14U][0U] = fma(a1_14, b1_0[0U], acc_1[14U][0U]);
                            }
                            acc_1[14U][1U] = fma(a0_14, b0_0[1U], acc_1[14U][1U]);
                            if(_S67)
                            {
                                acc_1[14U][1U] = fma(a1_14, b1_0[1U], acc_1[14U][1U]);
                            }
                            acc_1[14U][2U] = fma(a0_14, b0_0[2U], acc_1[14U][2U]);
                            if(_S67)
                            {
                                acc_1[14U][2U] = fma(a1_14, b1_0[2U], acc_1[14U][2U]);
                            }
                            acc_1[14U][3U] = fma(a0_14, b0_0[3U], acc_1[14U][3U]);
                            if(_S67)
                            {
                                acc_1[14U][3U] = fma(a1_14, b1_0[3U], acc_1[14U][3U]);
                            }
                            _S41 = int(3);
                            break;
                        }
                        if(_S41 != int(3))
                        {
                            _S42 = _S41;
                            break;
                        }
                        _S42 = int(2);
                        break;
                    }
                    if(_S42 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(15U >= _S9)
                            {
                                _S43 = int(1);
                                break;
                            }
                            uint e_23 = _S8.baseA_0 + 15U * (&kernelContext_9)->p_0->lda_1 + _S65;
                            thread float a0_15;
                            thread float a1_15 = 0.0;
                            if(_S67)
                            {
                                load2A_0(e_23, &a0_15, &a1_15, &kernelContext_9);
                            }
                            else
                            {
                                float _S92 = elemA_0(e_23, &kernelContext_9);
                                a0_15 = _S92;
                            }
                            acc_1[15U][0U] = fma(a0_15, b0_0[0U], acc_1[15U][0U]);
                            if(_S67)
                            {
                                acc_1[15U][0U] = fma(a1_15, b1_0[0U], acc_1[15U][0U]);
                            }
                            acc_1[15U][1U] = fma(a0_15, b0_0[1U], acc_1[15U][1U]);
                            if(_S67)
                            {
                                acc_1[15U][1U] = fma(a1_15, b1_0[1U], acc_1[15U][1U]);
                            }
                            acc_1[15U][2U] = fma(a0_15, b0_0[2U], acc_1[15U][2U]);
                            if(_S67)
                            {
                                acc_1[15U][2U] = fma(a1_15, b1_0[2U], acc_1[15U][2U]);
                            }
                            acc_1[15U][3U] = fma(a0_15, b0_0[3U], acc_1[15U][3U]);
                            if(_S67)
                            {
                                acc_1[15U][3U] = fma(a1_15, b1_0[3U], acc_1[15U][3U]);
                            }
                            _S43 = int(3);
                            break;
                        }
                        if(_S43 != int(3))
                        {
                            _S44 = _S43;
                            break;
                        }
                        _S44 = int(2);
                        break;
                    }
                    if(_S44 != int(2))
                    {
                        break;
                    }
                    break;
                }
                ch_0 = ch_0 + 16U;
            }
        }
    }
    if(transB_1)
    {
        ch_0 = 1U;
    }
    else
    {
        ch_0 = 16U;
    }
    i_2 = 0U;
    for(;;)
    {
        if(i_2 < _S9)
        {
        }
        else
        {
            break;
        }
        (*(&kernelContext_9)->red_0)[0U][lid_0] = acc_1[i_2][0U];
        (*(&kernelContext_9)->red_0)[1U][lid_0] = acc_1[i_2][1U];
        (*(&kernelContext_9)->red_0)[2U][lid_0] = acc_1[i_2][2U];
        (*(&kernelContext_9)->red_0)[3U][lid_0] = acc_1[i_2][3U];
        uint w_0 = _S10 / 2U;
        for(;;)
        {
            if(w_0 > 0U)
            {
            }
            else
            {
                break;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane_0 < w_0)
            {
                if(0U < _S11)
                {
                    (*(&kernelContext_9)->red_0)[0U][lid_0] = (*(&kernelContext_9)->red_0)[0U][lid_0] + (*(&kernelContext_9)->red_0)[0U][lid_0 + w_0 * ch_0];
                }
                if(1U < _S11)
                {
                    (*(&kernelContext_9)->red_0)[1U][lid_0] = (*(&kernelContext_9)->red_0)[1U][lid_0] + (*(&kernelContext_9)->red_0)[1U][lid_0 + w_0 * ch_0];
                }
                if(2U < _S11)
                {
                    (*(&kernelContext_9)->red_0)[2U][lid_0] = (*(&kernelContext_9)->red_0)[2U][lid_0] + (*(&kernelContext_9)->red_0)[2U][lid_0 + w_0 * ch_0];
                }
                if(3U < _S11)
                {
                    (*(&kernelContext_9)->red_0)[3U][lid_0] = (*(&kernelContext_9)->red_0)[3U][lid_0] + (*(&kernelContext_9)->red_0)[3U][lid_0 + w_0 * ch_0];
                }
            }
            w_0 = w_0 / 2U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane_0 == 0U)
        {
            uint cc_0 = 0U;
            for(;;)
            {
                if(cc_0 < _S11)
                {
                }
                else
                {
                    break;
                }
                uint _S93 = col0_0 + cc_0;
                if(_S93 < ((&kernelContext_9)->p_0->n_1))
                {
                    float _S94 = (*(&kernelContext_9)->red_0)[cc_0][lid_0];
                    thread GemmProblem_0 _S95 = _S8;
                    gemmStore_0(&_S95, i_2, _S93, _S94, &kernelContext_9);
                }
                cc_0 = cc_0 + 1U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_2 = i_2 + 1U;
    }
    return;
}

