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
    uint4 device* b_0;
    uint4 device* a_0;
    half device* c_0;
    half device* epBias_0;
    half device* epGate_0;
    half device* epResidual_0;
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

float wordLo_0(uint w_0)
{
    return (as_type<half>((ushort)((w_0 & 65535U))));
}

float wordHi_0(uint w_1)
{
    return (as_type<half>((ushort)((w_1 >> 16U))));
}

void load8B_0(uint e_0, array<float, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_1)
{
    uint4 device* _S1 = kernelContext_1->b_0+(e_0 >> 3U);
    uint4 _S2 = *_S1;
    (*v_0)[0U] = wordLo_0((*_S1)[0U]);
    (*v_0)[1U] = wordHi_0(_S2[0U]);
    (*v_0)[2U] = wordLo_0(_S2[1U]);
    (*v_0)[3U] = wordHi_0(_S2[1U]);
    (*v_0)[4U] = wordLo_0(_S2[2U]);
    (*v_0)[5U] = wordHi_0(_S2[2U]);
    (*v_0)[6U] = wordLo_0(_S2[3U]);
    (*v_0)[7U] = wordHi_0(_S2[3U]);
    return;
}

void load8A_0(uint e_1, array<float, int(8)> thread* v_1, KernelContext_0 thread* kernelContext_2)
{
    uint4 device* _S3 = kernelContext_2->a_0+(e_1 >> 3U);
    uint4 _S4 = *_S3;
    (*v_1)[0U] = wordLo_0((*_S3)[0U]);
    (*v_1)[1U] = wordHi_0(_S4[0U]);
    (*v_1)[2U] = wordLo_0(_S4[1U]);
    (*v_1)[3U] = wordHi_0(_S4[1U]);
    (*v_1)[4U] = wordLo_0(_S4[2U]);
    (*v_1)[5U] = wordHi_0(_S4[2U]);
    (*v_1)[6U] = wordLo_0(_S4[3U]);
    (*v_1)[7U] = wordHi_0(_S4[3U]);
    return;
}

bool dg_is_nan_0(float x_0)
{
    uint _S5 = (as_type<uint>((x_0)));
    bool _S6;
    if((_S5 & 2139095040U) == 2139095040U)
    {
        _S6 = (_S5 & 8388607U) != 0U;
    }
    else
    {
        _S6 = false;
    }
    return _S6;
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
    bool _S7;
    if(a_1 == 0.0)
    {
        _S7 = b_1 == 0.0;
    }
    else
    {
        _S7 = false;
    }
    if(_S7)
    {
        return (as_type<float>(((as_type<uint>((a_1))) & (as_type<uint>((b_1))))));
    }
    float _S8;
    if(a_1 > b_1)
    {
        _S8 = a_1;
    }
    else
    {
        _S8 = b_1;
    }
    return _S8;
}

float gemmEpilogue_0(float r_0, uint i_0, uint j_0, uint idx_0, uint biasBase_0, KernelContext_0 thread* kernelContext_3)
{
    float _S9;
    if((kernelContext_3->p_0->bias_0) == 1U)
    {
        _S9 = r_0 + float(*(kernelContext_3->epBias_0+(biasBase_0 + j_0)));
    }
    else
    {
        if((kernelContext_3->p_0->bias_0) == 2U)
        {
            _S9 = r_0 + float(*(kernelContext_3->epBias_0+(biasBase_0 + i_0)));
        }
        else
        {
            _S9 = r_0;
        }
    }
    if((kernelContext_3->p_0->act_0) == 1U)
    {
        _S9 = dg_max_0(_S9, 0.0);
    }
    else
    {
        if((kernelContext_3->p_0->act_0) == 2U)
        {
            _S9 = 0.5 * _S9 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S9 + 0.04471499845385551 * _S9 * _S9 * _S9))) + 1.0)));
        }
        else
        {
            if((kernelContext_3->p_0->act_0) == 3U)
            {
                _S9 = _S9 / (1.0 + exp(- _S9));
            }
        }
    }
    if((kernelContext_3->p_0->gated_0) != 0U)
    {
        _S9 = _S9 * float(*(kernelContext_3->epGate_0+idx_0));
    }
    if((kernelContext_3->p_0->residual_0) != 0U)
    {
        _S9 = _S9 + float(*(kernelContext_3->epResidual_0+idx_0));
    }
    return _S9;
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
    half device* _S10 = kernelContext_4->c_0+idx_1;
    float _S11 = gemmEpilogue_0(r_1, i_1, j_1, idx_1, q_1->baseBias_0, kernelContext_4);
    *_S10 = half(_S11);
    return;
}

[[kernel]] void gemm_skinny_v_h(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], uint4 device* b_2 [[buffer(1)]], uint4 device* a_2 [[buffer(0)]], half device* c_1 [[buffer(2)]], half device* epBias_1 [[buffer(3)]], half device* epGate_1 [[buffer(4)]], half device* epResidual_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->p_0 = p_1;
    (&kernelContext_5)->b_0 = b_2;
    (&kernelContext_5)->a_0 = a_2;
    (&kernelContext_5)->c_0 = c_1;
    (&kernelContext_5)->epBias_0 = epBias_1;
    (&kernelContext_5)->epGate_0 = epGate_1;
    (&kernelContext_5)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(256)>, int(4)> red_1;
    (&kernelContext_5)->red_0 = &red_1;
    uint lid_0 = local_0.x;
    GemmProblem_0 _S12 = gemmBatchProblem_0(group_0.z, &kernelContext_5);
    uint _S13 = min((&kernelContext_5)->p_0->m_1, 16U);
    uint k_2 = (&kernelContext_5)->p_0->k_1;
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
    bool transB_1 = ((&kernelContext_5)->p_0->transB_0) != 0U;
    uint _S14;
    if(transB_1)
    {
        _S14 = 32U;
    }
    else
    {
        _S14 = 16U;
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
    uint _S15;
    if(transB_1)
    {
        _S15 = 2U;
    }
    else
    {
        _S15 = 4U;
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
    bool _S16;
    if(col0_0 < ((&kernelContext_5)->p_0->n_1))
    {
        _S16 = transB_1;
    }
    else
    {
        _S16 = false;
    }
    uint ch_0;
    if(_S16)
    {
        uint _S17 = k_2 / 8U;
        ch_0 = lane_0;
        for(;;)
        {
            if(ch_0 < _S17)
            {
            }
            else
            {
                break;
            }
            uint _S18 = ch_0 * 8U;
            thread array<array<float, int(8)>, int(2)> bv_0;
            load8B_0(_S12.baseB_0 + min(col0_0, (&kernelContext_5)->p_0->n_1 - 1U) * (&kernelContext_5)->p_0->ldb_1 + _S18, &bv_0[0U], &kernelContext_5);
            load8B_0(_S12.baseB_0 + min(col0_0 + 1U, (&kernelContext_5)->p_0->n_1 - 1U) * (&kernelContext_5)->p_0->ldb_1 + _S18, &bv_0[1U], &kernelContext_5);
            for(;;)
            {
                int _S19;
                for(;;)
                {
                    int _S20;
                    for(;;)
                    {
                        if(0U >= _S13)
                        {
                            _S20 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_0;
                        load8A_0(_S12.baseA_0 + _S18, &av_0, &kernelContext_5);
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
                        _S20 = int(3);
                        break;
                    }
                    if(_S20 != int(3))
                    {
                        _S19 = _S20;
                        break;
                    }
                    _S19 = int(2);
                    break;
                }
                if(_S19 != int(2))
                {
                    break;
                }
                int _S21;
                for(;;)
                {
                    int _S22;
                    for(;;)
                    {
                        if(1U >= _S13)
                        {
                            _S22 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_1;
                        load8A_0(_S12.baseA_0 + (&kernelContext_5)->p_0->lda_1 + _S18, &av_1, &kernelContext_5);
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
                        _S22 = int(3);
                        break;
                    }
                    if(_S22 != int(3))
                    {
                        _S21 = _S22;
                        break;
                    }
                    _S21 = int(2);
                    break;
                }
                if(_S21 != int(2))
                {
                    break;
                }
                int _S23;
                for(;;)
                {
                    int _S24;
                    for(;;)
                    {
                        if(2U >= _S13)
                        {
                            _S24 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_2;
                        load8A_0(_S12.baseA_0 + 2U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_2, &kernelContext_5);
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
                        _S24 = int(3);
                        break;
                    }
                    if(_S24 != int(3))
                    {
                        _S23 = _S24;
                        break;
                    }
                    _S23 = int(2);
                    break;
                }
                if(_S23 != int(2))
                {
                    break;
                }
                int _S25;
                for(;;)
                {
                    int _S26;
                    for(;;)
                    {
                        if(3U >= _S13)
                        {
                            _S26 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_3;
                        load8A_0(_S12.baseA_0 + 3U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_3, &kernelContext_5);
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
                        _S26 = int(3);
                        break;
                    }
                    if(_S26 != int(3))
                    {
                        _S25 = _S26;
                        break;
                    }
                    _S25 = int(2);
                    break;
                }
                if(_S25 != int(2))
                {
                    break;
                }
                int _S27;
                for(;;)
                {
                    int _S28;
                    for(;;)
                    {
                        if(4U >= _S13)
                        {
                            _S28 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_4;
                        load8A_0(_S12.baseA_0 + 4U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_4, &kernelContext_5);
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
                        _S28 = int(3);
                        break;
                    }
                    if(_S28 != int(3))
                    {
                        _S27 = _S28;
                        break;
                    }
                    _S27 = int(2);
                    break;
                }
                if(_S27 != int(2))
                {
                    break;
                }
                int _S29;
                for(;;)
                {
                    int _S30;
                    for(;;)
                    {
                        if(5U >= _S13)
                        {
                            _S30 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_5;
                        load8A_0(_S12.baseA_0 + 5U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_5, &kernelContext_5);
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
                        _S30 = int(3);
                        break;
                    }
                    if(_S30 != int(3))
                    {
                        _S29 = _S30;
                        break;
                    }
                    _S29 = int(2);
                    break;
                }
                if(_S29 != int(2))
                {
                    break;
                }
                int _S31;
                for(;;)
                {
                    int _S32;
                    for(;;)
                    {
                        if(6U >= _S13)
                        {
                            _S32 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_6;
                        load8A_0(_S12.baseA_0 + 6U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_6, &kernelContext_5);
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
                        _S32 = int(3);
                        break;
                    }
                    if(_S32 != int(3))
                    {
                        _S31 = _S32;
                        break;
                    }
                    _S31 = int(2);
                    break;
                }
                if(_S31 != int(2))
                {
                    break;
                }
                int _S33;
                for(;;)
                {
                    int _S34;
                    for(;;)
                    {
                        if(7U >= _S13)
                        {
                            _S34 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_7;
                        load8A_0(_S12.baseA_0 + 7U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_7, &kernelContext_5);
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
                        _S34 = int(3);
                        break;
                    }
                    if(_S34 != int(3))
                    {
                        _S33 = _S34;
                        break;
                    }
                    _S33 = int(2);
                    break;
                }
                if(_S33 != int(2))
                {
                    break;
                }
                int _S35;
                for(;;)
                {
                    int _S36;
                    for(;;)
                    {
                        if(8U >= _S13)
                        {
                            _S36 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_8;
                        load8A_0(_S12.baseA_0 + 8U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_8, &kernelContext_5);
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
                        _S36 = int(3);
                        break;
                    }
                    if(_S36 != int(3))
                    {
                        _S35 = _S36;
                        break;
                    }
                    _S35 = int(2);
                    break;
                }
                if(_S35 != int(2))
                {
                    break;
                }
                int _S37;
                for(;;)
                {
                    int _S38;
                    for(;;)
                    {
                        if(9U >= _S13)
                        {
                            _S38 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_9;
                        load8A_0(_S12.baseA_0 + 9U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_9, &kernelContext_5);
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
                        _S38 = int(3);
                        break;
                    }
                    if(_S38 != int(3))
                    {
                        _S37 = _S38;
                        break;
                    }
                    _S37 = int(2);
                    break;
                }
                if(_S37 != int(2))
                {
                    break;
                }
                int _S39;
                for(;;)
                {
                    int _S40;
                    for(;;)
                    {
                        if(10U >= _S13)
                        {
                            _S40 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_10;
                        load8A_0(_S12.baseA_0 + 10U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_10, &kernelContext_5);
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
                        _S40 = int(3);
                        break;
                    }
                    if(_S40 != int(3))
                    {
                        _S39 = _S40;
                        break;
                    }
                    _S39 = int(2);
                    break;
                }
                if(_S39 != int(2))
                {
                    break;
                }
                int _S41;
                for(;;)
                {
                    int _S42;
                    for(;;)
                    {
                        if(11U >= _S13)
                        {
                            _S42 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_11;
                        load8A_0(_S12.baseA_0 + 11U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_11, &kernelContext_5);
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
                        _S42 = int(3);
                        break;
                    }
                    if(_S42 != int(3))
                    {
                        _S41 = _S42;
                        break;
                    }
                    _S41 = int(2);
                    break;
                }
                if(_S41 != int(2))
                {
                    break;
                }
                int _S43;
                for(;;)
                {
                    int _S44;
                    for(;;)
                    {
                        if(12U >= _S13)
                        {
                            _S44 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_12;
                        load8A_0(_S12.baseA_0 + 12U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_12, &kernelContext_5);
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
                        _S44 = int(3);
                        break;
                    }
                    if(_S44 != int(3))
                    {
                        _S43 = _S44;
                        break;
                    }
                    _S43 = int(2);
                    break;
                }
                if(_S43 != int(2))
                {
                    break;
                }
                int _S45;
                for(;;)
                {
                    int _S46;
                    for(;;)
                    {
                        if(13U >= _S13)
                        {
                            _S46 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_13;
                        load8A_0(_S12.baseA_0 + 13U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_13, &kernelContext_5);
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
                        _S46 = int(3);
                        break;
                    }
                    if(_S46 != int(3))
                    {
                        _S45 = _S46;
                        break;
                    }
                    _S45 = int(2);
                    break;
                }
                if(_S45 != int(2))
                {
                    break;
                }
                int _S47;
                for(;;)
                {
                    int _S48;
                    for(;;)
                    {
                        if(14U >= _S13)
                        {
                            _S48 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_14;
                        load8A_0(_S12.baseA_0 + 14U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_14, &kernelContext_5);
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
                        _S48 = int(3);
                        break;
                    }
                    if(_S48 != int(3))
                    {
                        _S47 = _S48;
                        break;
                    }
                    _S47 = int(2);
                    break;
                }
                if(_S47 != int(2))
                {
                    break;
                }
                int _S49;
                for(;;)
                {
                    int _S50;
                    for(;;)
                    {
                        if(15U >= _S13)
                        {
                            _S50 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_15;
                        load8A_0(_S12.baseA_0 + 15U * (&kernelContext_5)->p_0->lda_1 + _S18, &av_15, &kernelContext_5);
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
                        _S50 = int(3);
                        break;
                    }
                    if(_S50 != int(3))
                    {
                        _S49 = _S50;
                        break;
                    }
                    _S49 = int(2);
                    break;
                }
                if(_S49 != int(2))
                {
                    break;
                }
                break;
            }
            ch_0 = ch_0 + 32U;
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
    uint i_2 = 0U;
    for(;;)
    {
        if(i_2 < _S13)
        {
        }
        else
        {
            break;
        }
        (*(&kernelContext_5)->red_0)[0U][lid_0] = acc_1[i_2][0U];
        (*(&kernelContext_5)->red_0)[1U][lid_0] = acc_1[i_2][1U];
        (*(&kernelContext_5)->red_0)[2U][lid_0] = acc_1[i_2][2U];
        (*(&kernelContext_5)->red_0)[3U][lid_0] = acc_1[i_2][3U];
        uint w_2 = _S14 / 2U;
        for(;;)
        {
            if(w_2 > 0U)
            {
            }
            else
            {
                break;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane_0 < w_2)
            {
                if(0U < _S15)
                {
                    (*(&kernelContext_5)->red_0)[0U][lid_0] = (*(&kernelContext_5)->red_0)[0U][lid_0] + (*(&kernelContext_5)->red_0)[0U][lid_0 + w_2 * ch_0];
                }
                if(1U < _S15)
                {
                    (*(&kernelContext_5)->red_0)[1U][lid_0] = (*(&kernelContext_5)->red_0)[1U][lid_0] + (*(&kernelContext_5)->red_0)[1U][lid_0 + w_2 * ch_0];
                }
                if(2U < _S15)
                {
                    (*(&kernelContext_5)->red_0)[2U][lid_0] = (*(&kernelContext_5)->red_0)[2U][lid_0] + (*(&kernelContext_5)->red_0)[2U][lid_0 + w_2 * ch_0];
                }
                if(3U < _S15)
                {
                    (*(&kernelContext_5)->red_0)[3U][lid_0] = (*(&kernelContext_5)->red_0)[3U][lid_0] + (*(&kernelContext_5)->red_0)[3U][lid_0 + w_2 * ch_0];
                }
            }
            w_2 = w_2 / 2U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane_0 == 0U)
        {
            uint cc_0 = 0U;
            for(;;)
            {
                if(cc_0 < _S15)
                {
                }
                else
                {
                    break;
                }
                uint _S51 = col0_0 + cc_0;
                if(_S51 < ((&kernelContext_5)->p_0->n_1))
                {
                    float _S52 = (*(&kernelContext_5)->red_0)[cc_0][lid_0];
                    thread GemmProblem_0 _S53 = _S12;
                    gemmStore_0(&_S53, i_2, _S51, _S52, &kernelContext_5);
                }
                cc_0 = cc_0 + 1U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_2 = i_2 + 1U;
    }
    return;
}

