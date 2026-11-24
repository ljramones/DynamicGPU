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
    return (as_type<float>((w_0 << 16U)));
}

float wordHi_0(uint w_1)
{
    return (as_type<float>((w_1 & 4294901760U)));
}

float elemB_0(uint e_0, KernelContext_0 thread* kernelContext_1)
{
    uint w_2 = *(kernelContext_1->b_0+(e_0 >> 1U));
    float _S1;
    if((e_0 & 1U) != 0U)
    {
        _S1 = wordHi_0(w_2);
    }
    else
    {
        _S1 = wordLo_0(w_2);
    }
    return _S1;
}

void load8B_0(uint e_1, array<float, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_2)
{
    if((e_1 & 1U) == 0U)
    {
        uint _S2 = e_1 >> 1U;
        uint device* _S3 = kernelContext_2->b_0+_S2;
        uint w_3 = *_S3;
        (*v_0)[0U] = wordLo_0(*_S3);
        (*v_0)[1U] = wordHi_0(w_3);
        uint device* _S4 = kernelContext_2->b_0+(_S2 + 1U);
        uint w_4 = *_S4;
        (*v_0)[2U] = wordLo_0(*_S4);
        (*v_0)[3U] = wordHi_0(w_4);
        uint device* _S5 = kernelContext_2->b_0+(_S2 + 2U);
        uint w_5 = *_S5;
        (*v_0)[4U] = wordLo_0(*_S5);
        (*v_0)[5U] = wordHi_0(w_5);
        uint device* _S6 = kernelContext_2->b_0+(_S2 + 3U);
        uint w_6 = *_S6;
        (*v_0)[6U] = wordLo_0(*_S6);
        (*v_0)[7U] = wordHi_0(w_6);
    }
    else
    {
        float _S7 = elemB_0(e_1, kernelContext_2);
        (*v_0)[0U] = _S7;
        float _S8 = elemB_0(e_1 + 1U, kernelContext_2);
        (*v_0)[1U] = _S8;
        float _S9 = elemB_0(e_1 + 2U, kernelContext_2);
        (*v_0)[2U] = _S9;
        float _S10 = elemB_0(e_1 + 3U, kernelContext_2);
        (*v_0)[3U] = _S10;
        float _S11 = elemB_0(e_1 + 4U, kernelContext_2);
        (*v_0)[4U] = _S11;
        float _S12 = elemB_0(e_1 + 5U, kernelContext_2);
        (*v_0)[5U] = _S12;
        float _S13 = elemB_0(e_1 + 6U, kernelContext_2);
        (*v_0)[6U] = _S13;
        float _S14 = elemB_0(e_1 + 7U, kernelContext_2);
        (*v_0)[7U] = _S14;
    }
    return;
}

float elemA_0(uint e_2, KernelContext_0 thread* kernelContext_3)
{
    uint w_7 = *(kernelContext_3->a_0+(e_2 >> 1U));
    float _S15;
    if((e_2 & 1U) != 0U)
    {
        _S15 = wordHi_0(w_7);
    }
    else
    {
        _S15 = wordLo_0(w_7);
    }
    return _S15;
}

void load8A_0(uint e_3, array<float, int(8)> thread* v_1, KernelContext_0 thread* kernelContext_4)
{
    if((e_3 & 1U) == 0U)
    {
        uint _S16 = e_3 >> 1U;
        uint device* _S17 = kernelContext_4->a_0+_S16;
        uint w_8 = *_S17;
        (*v_1)[0U] = wordLo_0(*_S17);
        (*v_1)[1U] = wordHi_0(w_8);
        uint device* _S18 = kernelContext_4->a_0+(_S16 + 1U);
        uint w_9 = *_S18;
        (*v_1)[2U] = wordLo_0(*_S18);
        (*v_1)[3U] = wordHi_0(w_9);
        uint device* _S19 = kernelContext_4->a_0+(_S16 + 2U);
        uint w_10 = *_S19;
        (*v_1)[4U] = wordLo_0(*_S19);
        (*v_1)[5U] = wordHi_0(w_10);
        uint device* _S20 = kernelContext_4->a_0+(_S16 + 3U);
        uint w_11 = *_S20;
        (*v_1)[6U] = wordLo_0(*_S20);
        (*v_1)[7U] = wordHi_0(w_11);
    }
    else
    {
        float _S21 = elemA_0(e_3, kernelContext_4);
        (*v_1)[0U] = _S21;
        float _S22 = elemA_0(e_3 + 1U, kernelContext_4);
        (*v_1)[1U] = _S22;
        float _S23 = elemA_0(e_3 + 2U, kernelContext_4);
        (*v_1)[2U] = _S23;
        float _S24 = elemA_0(e_3 + 3U, kernelContext_4);
        (*v_1)[3U] = _S24;
        float _S25 = elemA_0(e_3 + 4U, kernelContext_4);
        (*v_1)[4U] = _S25;
        float _S26 = elemA_0(e_3 + 5U, kernelContext_4);
        (*v_1)[5U] = _S26;
        float _S27 = elemA_0(e_3 + 6U, kernelContext_4);
        (*v_1)[6U] = _S27;
        float _S28 = elemA_0(e_3 + 7U, kernelContext_4);
        (*v_1)[7U] = _S28;
    }
    return;
}

void load4B_0(uint e_4, array<float, int(4)> thread* v_2, KernelContext_0 thread* kernelContext_5)
{
    if((e_4 & 1U) == 0U)
    {
        uint _S29 = e_4 >> 1U;
        uint device* _S30 = kernelContext_5->b_0+_S29;
        uint w0_0 = *_S30;
        uint w1_0 = *(kernelContext_5->b_0+(_S29 + 1U));
        (*v_2)[int(0)] = wordLo_0(*_S30);
        (*v_2)[int(1)] = wordHi_0(w0_0);
        (*v_2)[int(2)] = wordLo_0(w1_0);
        (*v_2)[int(3)] = wordHi_0(w1_0);
    }
    else
    {
        float _S31 = elemB_0(e_4, kernelContext_5);
        (*v_2)[0U] = _S31;
        float _S32 = elemB_0(e_4 + 1U, kernelContext_5);
        (*v_2)[1U] = _S32;
        float _S33 = elemB_0(e_4 + 2U, kernelContext_5);
        (*v_2)[2U] = _S33;
        float _S34 = elemB_0(e_4 + 3U, kernelContext_5);
        (*v_2)[3U] = _S34;
    }
    return;
}

void load2A_0(uint e_5, float thread* v0_0, float thread* v1_0, KernelContext_0 thread* kernelContext_6)
{
    if((e_5 & 1U) == 0U)
    {
        uint device* _S35 = kernelContext_6->a_0+(e_5 >> 1U);
        uint w_12 = *_S35;
        *v0_0 = wordLo_0(*_S35);
        *v1_0 = wordHi_0(w_12);
    }
    else
    {
        float _S36 = elemA_0(e_5, kernelContext_6);
        *v0_0 = _S36;
        float _S37 = elemA_0(e_5 + 1U, kernelContext_6);
        *v1_0 = _S37;
    }
    return;
}

bool dg_is_nan_0(float x_0)
{
    uint _S38 = (as_type<uint>((x_0)));
    bool _S39;
    if((_S38 & 2139095040U) == 2139095040U)
    {
        _S39 = (_S38 & 8388607U) != 0U;
    }
    else
    {
        _S39 = false;
    }
    return _S39;
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
    bool _S40;
    if(a_1 == 0.0)
    {
        _S40 = b_1 == 0.0;
    }
    else
    {
        _S40 = false;
    }
    if(_S40)
    {
        return (as_type<float>(((as_type<uint>((a_1))) & (as_type<uint>((b_1))))));
    }
    float _S41;
    if(a_1 > b_1)
    {
        _S41 = a_1;
    }
    else
    {
        _S41 = b_1;
    }
    return _S41;
}

float gemmEpilogue_0(float r_0, uint i_0, uint j_0, uint idx_0, uint biasBase_0, KernelContext_0 thread* kernelContext_7)
{
    float _S42;
    if((kernelContext_7->p_0->bias_0) == 1U)
    {
        _S42 = r_0 + float(*(kernelContext_7->epBias_0+(biasBase_0 + j_0)));
    }
    else
    {
        if((kernelContext_7->p_0->bias_0) == 2U)
        {
            _S42 = r_0 + float(*(kernelContext_7->epBias_0+(biasBase_0 + i_0)));
        }
        else
        {
            _S42 = r_0;
        }
    }
    if((kernelContext_7->p_0->act_0) == 1U)
    {
        _S42 = dg_max_0(_S42, 0.0);
    }
    else
    {
        if((kernelContext_7->p_0->act_0) == 2U)
        {
            _S42 = 0.5 * _S42 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S42 + 0.04471499845385551 * _S42 * _S42 * _S42))) + 1.0)));
        }
        else
        {
            if((kernelContext_7->p_0->act_0) == 3U)
            {
                _S42 = _S42 / (1.0 + exp(- _S42));
            }
        }
    }
    if((kernelContext_7->p_0->gated_0) != 0U)
    {
        _S42 = _S42 * float(*(kernelContext_7->epGate_0+idx_0));
    }
    if((kernelContext_7->p_0->residual_0) != 0U)
    {
        _S42 = _S42 + float(*(kernelContext_7->epResidual_0+idx_0));
    }
    return _S42;
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
        r_1 = kernelContext_8->p_0->alpha_0 * acc_0 + kernelContext_8->p_0->beta_0 * float(*(kernelContext_8->c_0+idx_1));
    }
    half device* _S43 = kernelContext_8->c_0+idx_1;
    float _S44 = gemmEpilogue_0(r_1, i_1, j_1, idx_1, q_1->baseBias_0, kernelContext_8);
    *_S43 = half(_S44);
    return;
}

[[kernel]] void gemm_skinny_bf16_h(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], uint device* b_2 [[buffer(1)]], uint device* a_2 [[buffer(0)]], half device* c_1 [[buffer(2)]], half device* epBias_1 [[buffer(3)]], half device* epGate_1 [[buffer(4)]], half device* epResidual_1 [[buffer(5)]])
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
    GemmProblem_0 _S45 = gemmBatchProblem_0(group_0.z, &kernelContext_9);
    uint _S46 = min((&kernelContext_9)->p_0->m_1, 16U);
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
    uint _S47;
    if(transB_1)
    {
        _S47 = 32U;
    }
    else
    {
        _S47 = 16U;
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
    uint _S48;
    if(transB_1)
    {
        _S48 = 2U;
    }
    else
    {
        _S48 = 4U;
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
    bool _S49;
    if(col0_0 < ((&kernelContext_9)->p_0->n_1))
    {
        _S49 = transB_1;
    }
    else
    {
        _S49 = false;
    }
    uint ch_0;
    uint i_2;
    int _S50;
    int _S51;
    int _S52;
    int _S53;
    int _S54;
    int _S55;
    int _S56;
    int _S57;
    int _S58;
    int _S59;
    int _S60;
    int _S61;
    int _S62;
    int _S63;
    int _S64;
    int _S65;
    int _S66;
    int _S67;
    int _S68;
    int _S69;
    int _S70;
    int _S71;
    int _S72;
    int _S73;
    int _S74;
    int _S75;
    int _S76;
    int _S77;
    int _S78;
    int _S79;
    int _S80;
    int _S81;
    if(_S49)
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
            uint _S82 = ch_0 * 8U;
            thread array<array<float, int(8)>, int(2)> bv_0;
            load8B_0(_S45.baseB_0 + min(col0_0, (&kernelContext_9)->p_0->n_1 - 1U) * (&kernelContext_9)->p_0->ldb_1 + _S82, &bv_0[0U], &kernelContext_9);
            load8B_0(_S45.baseB_0 + min(col0_0 + 1U, (&kernelContext_9)->p_0->n_1 - 1U) * (&kernelContext_9)->p_0->ldb_1 + _S82, &bv_0[1U], &kernelContext_9);
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        if(0U >= _S46)
                        {
                            _S50 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_0;
                        load8A_0(_S45.baseA_0 + _S82, &av_0, &kernelContext_9);
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
                        _S50 = int(3);
                        break;
                    }
                    if(_S50 != int(3))
                    {
                        _S51 = _S50;
                        break;
                    }
                    _S51 = int(2);
                    break;
                }
                if(_S51 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(1U >= _S46)
                        {
                            _S52 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_1;
                        load8A_0(_S45.baseA_0 + (&kernelContext_9)->p_0->lda_1 + _S82, &av_1, &kernelContext_9);
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
                        _S52 = int(3);
                        break;
                    }
                    if(_S52 != int(3))
                    {
                        _S53 = _S52;
                        break;
                    }
                    _S53 = int(2);
                    break;
                }
                if(_S53 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(2U >= _S46)
                        {
                            _S54 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_2;
                        load8A_0(_S45.baseA_0 + 2U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_2, &kernelContext_9);
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
                        _S54 = int(3);
                        break;
                    }
                    if(_S54 != int(3))
                    {
                        _S55 = _S54;
                        break;
                    }
                    _S55 = int(2);
                    break;
                }
                if(_S55 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(3U >= _S46)
                        {
                            _S56 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_3;
                        load8A_0(_S45.baseA_0 + 3U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_3, &kernelContext_9);
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
                        _S56 = int(3);
                        break;
                    }
                    if(_S56 != int(3))
                    {
                        _S57 = _S56;
                        break;
                    }
                    _S57 = int(2);
                    break;
                }
                if(_S57 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(4U >= _S46)
                        {
                            _S58 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_4;
                        load8A_0(_S45.baseA_0 + 4U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_4, &kernelContext_9);
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
                        _S58 = int(3);
                        break;
                    }
                    if(_S58 != int(3))
                    {
                        _S59 = _S58;
                        break;
                    }
                    _S59 = int(2);
                    break;
                }
                if(_S59 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(5U >= _S46)
                        {
                            _S60 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_5;
                        load8A_0(_S45.baseA_0 + 5U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_5, &kernelContext_9);
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
                        _S60 = int(3);
                        break;
                    }
                    if(_S60 != int(3))
                    {
                        _S61 = _S60;
                        break;
                    }
                    _S61 = int(2);
                    break;
                }
                if(_S61 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(6U >= _S46)
                        {
                            _S62 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_6;
                        load8A_0(_S45.baseA_0 + 6U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_6, &kernelContext_9);
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
                        _S62 = int(3);
                        break;
                    }
                    if(_S62 != int(3))
                    {
                        _S63 = _S62;
                        break;
                    }
                    _S63 = int(2);
                    break;
                }
                if(_S63 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(7U >= _S46)
                        {
                            _S64 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_7;
                        load8A_0(_S45.baseA_0 + 7U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_7, &kernelContext_9);
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
                        _S64 = int(3);
                        break;
                    }
                    if(_S64 != int(3))
                    {
                        _S65 = _S64;
                        break;
                    }
                    _S65 = int(2);
                    break;
                }
                if(_S65 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(8U >= _S46)
                        {
                            _S66 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_8;
                        load8A_0(_S45.baseA_0 + 8U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_8, &kernelContext_9);
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
                        _S66 = int(3);
                        break;
                    }
                    if(_S66 != int(3))
                    {
                        _S67 = _S66;
                        break;
                    }
                    _S67 = int(2);
                    break;
                }
                if(_S67 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(9U >= _S46)
                        {
                            _S68 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_9;
                        load8A_0(_S45.baseA_0 + 9U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_9, &kernelContext_9);
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
                        _S68 = int(3);
                        break;
                    }
                    if(_S68 != int(3))
                    {
                        _S69 = _S68;
                        break;
                    }
                    _S69 = int(2);
                    break;
                }
                if(_S69 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(10U >= _S46)
                        {
                            _S70 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_10;
                        load8A_0(_S45.baseA_0 + 10U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_10, &kernelContext_9);
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
                        _S70 = int(3);
                        break;
                    }
                    if(_S70 != int(3))
                    {
                        _S71 = _S70;
                        break;
                    }
                    _S71 = int(2);
                    break;
                }
                if(_S71 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(11U >= _S46)
                        {
                            _S72 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_11;
                        load8A_0(_S45.baseA_0 + 11U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_11, &kernelContext_9);
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
                        _S72 = int(3);
                        break;
                    }
                    if(_S72 != int(3))
                    {
                        _S73 = _S72;
                        break;
                    }
                    _S73 = int(2);
                    break;
                }
                if(_S73 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(12U >= _S46)
                        {
                            _S74 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_12;
                        load8A_0(_S45.baseA_0 + 12U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_12, &kernelContext_9);
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
                        _S74 = int(3);
                        break;
                    }
                    if(_S74 != int(3))
                    {
                        _S75 = _S74;
                        break;
                    }
                    _S75 = int(2);
                    break;
                }
                if(_S75 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(13U >= _S46)
                        {
                            _S76 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_13;
                        load8A_0(_S45.baseA_0 + 13U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_13, &kernelContext_9);
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
                        _S76 = int(3);
                        break;
                    }
                    if(_S76 != int(3))
                    {
                        _S77 = _S76;
                        break;
                    }
                    _S77 = int(2);
                    break;
                }
                if(_S77 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(14U >= _S46)
                        {
                            _S78 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_14;
                        load8A_0(_S45.baseA_0 + 14U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_14, &kernelContext_9);
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
                        _S78 = int(3);
                        break;
                    }
                    if(_S78 != int(3))
                    {
                        _S79 = _S78;
                        break;
                    }
                    _S79 = int(2);
                    break;
                }
                if(_S79 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(15U >= _S46)
                        {
                            _S80 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_15;
                        load8A_0(_S45.baseA_0 + 15U * (&kernelContext_9)->p_0->lda_1 + _S82, &av_15, &kernelContext_9);
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
                        _S80 = int(3);
                        break;
                    }
                    if(_S80 != int(3))
                    {
                        _S81 = _S80;
                        break;
                    }
                    _S81 = int(2);
                    break;
                }
                if(_S81 != int(2))
                {
                    break;
                }
                break;
            }
            ch_0 = ch_0 + 32U;
        }
        if((k_2 % 8U) != 0U)
        {
            _S49 = lane_0 == (chunks_0 % 32U);
        }
        else
        {
            _S49 = false;
        }
        if(_S49)
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
                float _S83 = elemB_0(_S45.baseB_0 + min(col0_0, (&kernelContext_9)->p_0->n_1 - 1U) * (&kernelContext_9)->p_0->ldb_1 + i_2, &kernelContext_9);
                bv_1[0U] = _S83;
                float _S84 = elemB_0(_S45.baseB_0 + min(col0_0 + 1U, (&kernelContext_9)->p_0->n_1 - 1U) * (&kernelContext_9)->p_0->ldb_1 + i_2, &kernelContext_9);
                bv_1[1U] = _S84;
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            if(0U >= _S46)
                            {
                                _S50 = int(1);
                                break;
                            }
                            float _S85 = elemA_0(_S45.baseA_0 + i_2, &kernelContext_9);
                            acc_1[0U][0U] = fma(_S85, bv_1[0U], acc_1[0U][0U]);
                            acc_1[0U][1U] = fma(_S85, bv_1[1U], acc_1[0U][1U]);
                            _S50 = int(3);
                            break;
                        }
                        if(_S50 != int(3))
                        {
                            _S51 = _S50;
                            break;
                        }
                        _S51 = int(2);
                        break;
                    }
                    if(_S51 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(1U >= _S46)
                            {
                                _S52 = int(1);
                                break;
                            }
                            float _S86 = elemA_0(_S45.baseA_0 + (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[1U][0U] = fma(_S86, bv_1[0U], acc_1[1U][0U]);
                            acc_1[1U][1U] = fma(_S86, bv_1[1U], acc_1[1U][1U]);
                            _S52 = int(3);
                            break;
                        }
                        if(_S52 != int(3))
                        {
                            _S53 = _S52;
                            break;
                        }
                        _S53 = int(2);
                        break;
                    }
                    if(_S53 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(2U >= _S46)
                            {
                                _S54 = int(1);
                                break;
                            }
                            float _S87 = elemA_0(_S45.baseA_0 + 2U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[2U][0U] = fma(_S87, bv_1[0U], acc_1[2U][0U]);
                            acc_1[2U][1U] = fma(_S87, bv_1[1U], acc_1[2U][1U]);
                            _S54 = int(3);
                            break;
                        }
                        if(_S54 != int(3))
                        {
                            _S55 = _S54;
                            break;
                        }
                        _S55 = int(2);
                        break;
                    }
                    if(_S55 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(3U >= _S46)
                            {
                                _S56 = int(1);
                                break;
                            }
                            float _S88 = elemA_0(_S45.baseA_0 + 3U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[3U][0U] = fma(_S88, bv_1[0U], acc_1[3U][0U]);
                            acc_1[3U][1U] = fma(_S88, bv_1[1U], acc_1[3U][1U]);
                            _S56 = int(3);
                            break;
                        }
                        if(_S56 != int(3))
                        {
                            _S57 = _S56;
                            break;
                        }
                        _S57 = int(2);
                        break;
                    }
                    if(_S57 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(4U >= _S46)
                            {
                                _S58 = int(1);
                                break;
                            }
                            float _S89 = elemA_0(_S45.baseA_0 + 4U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[4U][0U] = fma(_S89, bv_1[0U], acc_1[4U][0U]);
                            acc_1[4U][1U] = fma(_S89, bv_1[1U], acc_1[4U][1U]);
                            _S58 = int(3);
                            break;
                        }
                        if(_S58 != int(3))
                        {
                            _S59 = _S58;
                            break;
                        }
                        _S59 = int(2);
                        break;
                    }
                    if(_S59 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(5U >= _S46)
                            {
                                _S60 = int(1);
                                break;
                            }
                            float _S90 = elemA_0(_S45.baseA_0 + 5U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[5U][0U] = fma(_S90, bv_1[0U], acc_1[5U][0U]);
                            acc_1[5U][1U] = fma(_S90, bv_1[1U], acc_1[5U][1U]);
                            _S60 = int(3);
                            break;
                        }
                        if(_S60 != int(3))
                        {
                            _S61 = _S60;
                            break;
                        }
                        _S61 = int(2);
                        break;
                    }
                    if(_S61 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(6U >= _S46)
                            {
                                _S62 = int(1);
                                break;
                            }
                            float _S91 = elemA_0(_S45.baseA_0 + 6U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[6U][0U] = fma(_S91, bv_1[0U], acc_1[6U][0U]);
                            acc_1[6U][1U] = fma(_S91, bv_1[1U], acc_1[6U][1U]);
                            _S62 = int(3);
                            break;
                        }
                        if(_S62 != int(3))
                        {
                            _S63 = _S62;
                            break;
                        }
                        _S63 = int(2);
                        break;
                    }
                    if(_S63 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(7U >= _S46)
                            {
                                _S64 = int(1);
                                break;
                            }
                            float _S92 = elemA_0(_S45.baseA_0 + 7U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[7U][0U] = fma(_S92, bv_1[0U], acc_1[7U][0U]);
                            acc_1[7U][1U] = fma(_S92, bv_1[1U], acc_1[7U][1U]);
                            _S64 = int(3);
                            break;
                        }
                        if(_S64 != int(3))
                        {
                            _S65 = _S64;
                            break;
                        }
                        _S65 = int(2);
                        break;
                    }
                    if(_S65 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(8U >= _S46)
                            {
                                _S66 = int(1);
                                break;
                            }
                            float _S93 = elemA_0(_S45.baseA_0 + 8U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[8U][0U] = fma(_S93, bv_1[0U], acc_1[8U][0U]);
                            acc_1[8U][1U] = fma(_S93, bv_1[1U], acc_1[8U][1U]);
                            _S66 = int(3);
                            break;
                        }
                        if(_S66 != int(3))
                        {
                            _S67 = _S66;
                            break;
                        }
                        _S67 = int(2);
                        break;
                    }
                    if(_S67 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(9U >= _S46)
                            {
                                _S68 = int(1);
                                break;
                            }
                            float _S94 = elemA_0(_S45.baseA_0 + 9U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[9U][0U] = fma(_S94, bv_1[0U], acc_1[9U][0U]);
                            acc_1[9U][1U] = fma(_S94, bv_1[1U], acc_1[9U][1U]);
                            _S68 = int(3);
                            break;
                        }
                        if(_S68 != int(3))
                        {
                            _S69 = _S68;
                            break;
                        }
                        _S69 = int(2);
                        break;
                    }
                    if(_S69 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(10U >= _S46)
                            {
                                _S70 = int(1);
                                break;
                            }
                            float _S95 = elemA_0(_S45.baseA_0 + 10U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[10U][0U] = fma(_S95, bv_1[0U], acc_1[10U][0U]);
                            acc_1[10U][1U] = fma(_S95, bv_1[1U], acc_1[10U][1U]);
                            _S70 = int(3);
                            break;
                        }
                        if(_S70 != int(3))
                        {
                            _S71 = _S70;
                            break;
                        }
                        _S71 = int(2);
                        break;
                    }
                    if(_S71 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(11U >= _S46)
                            {
                                _S72 = int(1);
                                break;
                            }
                            float _S96 = elemA_0(_S45.baseA_0 + 11U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[11U][0U] = fma(_S96, bv_1[0U], acc_1[11U][0U]);
                            acc_1[11U][1U] = fma(_S96, bv_1[1U], acc_1[11U][1U]);
                            _S72 = int(3);
                            break;
                        }
                        if(_S72 != int(3))
                        {
                            _S73 = _S72;
                            break;
                        }
                        _S73 = int(2);
                        break;
                    }
                    if(_S73 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(12U >= _S46)
                            {
                                _S74 = int(1);
                                break;
                            }
                            float _S97 = elemA_0(_S45.baseA_0 + 12U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[12U][0U] = fma(_S97, bv_1[0U], acc_1[12U][0U]);
                            acc_1[12U][1U] = fma(_S97, bv_1[1U], acc_1[12U][1U]);
                            _S74 = int(3);
                            break;
                        }
                        if(_S74 != int(3))
                        {
                            _S75 = _S74;
                            break;
                        }
                        _S75 = int(2);
                        break;
                    }
                    if(_S75 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(13U >= _S46)
                            {
                                _S76 = int(1);
                                break;
                            }
                            float _S98 = elemA_0(_S45.baseA_0 + 13U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[13U][0U] = fma(_S98, bv_1[0U], acc_1[13U][0U]);
                            acc_1[13U][1U] = fma(_S98, bv_1[1U], acc_1[13U][1U]);
                            _S76 = int(3);
                            break;
                        }
                        if(_S76 != int(3))
                        {
                            _S77 = _S76;
                            break;
                        }
                        _S77 = int(2);
                        break;
                    }
                    if(_S77 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(14U >= _S46)
                            {
                                _S78 = int(1);
                                break;
                            }
                            float _S99 = elemA_0(_S45.baseA_0 + 14U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[14U][0U] = fma(_S99, bv_1[0U], acc_1[14U][0U]);
                            acc_1[14U][1U] = fma(_S99, bv_1[1U], acc_1[14U][1U]);
                            _S78 = int(3);
                            break;
                        }
                        if(_S78 != int(3))
                        {
                            _S79 = _S78;
                            break;
                        }
                        _S79 = int(2);
                        break;
                    }
                    if(_S79 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(15U >= _S46)
                            {
                                _S80 = int(1);
                                break;
                            }
                            float _S100 = elemA_0(_S45.baseA_0 + 15U * (&kernelContext_9)->p_0->lda_1 + i_2, &kernelContext_9);
                            acc_1[15U][0U] = fma(_S100, bv_1[0U], acc_1[15U][0U]);
                            acc_1[15U][1U] = fma(_S100, bv_1[1U], acc_1[15U][1U]);
                            _S80 = int(3);
                            break;
                        }
                        if(_S80 != int(3))
                        {
                            _S81 = _S80;
                            break;
                        }
                        _S81 = int(2);
                        break;
                    }
                    if(_S81 != int(2))
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
            bool _S101 = (col0_0 + 4U) <= ((&kernelContext_9)->p_0->n_1);
            ch_0 = lane_0;
            for(;;)
            {
                uint _S102 = 2U * ch_0;
                if(_S102 < k_2)
                {
                }
                else
                {
                    break;
                }
                uint _S103 = _S102 + 1U;
                bool _S104 = _S103 < k_2;
                thread array<float, int(4)> b0_0;
                thread array<float, int(4)> b1_0;
                uint _S105 = k_2 - 1U;
                uint e_6 = _S45.baseB_0 + min(_S102, _S105) * (&kernelContext_9)->p_0->ldb_1 + col0_0;
                thread array<float, int(4)> v_3;
                if(_S101)
                {
                    load4B_0(e_6, &v_3, &kernelContext_9);
                }
                else
                {
                    float _S106 = elemB_0(e_6 + min(0U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_3[0U] = _S106;
                    float _S107 = elemB_0(e_6 + min(1U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_3[1U] = _S107;
                    float _S108 = elemB_0(e_6 + min(2U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_3[2U] = _S108;
                    float _S109 = elemB_0(e_6 + min(3U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_3[3U] = _S109;
                }
                b0_0[0U] = v_3[0U];
                b0_0[1U] = v_3[1U];
                b0_0[2U] = v_3[2U];
                b0_0[3U] = v_3[3U];
                uint e_7 = _S45.baseB_0 + min(_S103, _S105) * (&kernelContext_9)->p_0->ldb_1 + col0_0;
                thread array<float, int(4)> v_4;
                if(_S101)
                {
                    load4B_0(e_7, &v_4, &kernelContext_9);
                }
                else
                {
                    float _S110 = elemB_0(e_7 + min(0U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_4[0U] = _S110;
                    float _S111 = elemB_0(e_7 + min(1U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_4[1U] = _S111;
                    float _S112 = elemB_0(e_7 + min(2U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_4[2U] = _S112;
                    float _S113 = elemB_0(e_7 + min(3U, (&kernelContext_9)->p_0->n_1 - 1U - col0_0), &kernelContext_9);
                    v_4[3U] = _S113;
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
                            if(0U >= _S46)
                            {
                                _S50 = int(1);
                                break;
                            }
                            uint e_8 = _S45.baseA_0 + _S102;
                            thread float a0_0;
                            thread float a1_0 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_8, &a0_0, &a1_0, &kernelContext_9);
                            }
                            else
                            {
                                float _S114 = elemA_0(e_8, &kernelContext_9);
                                a0_0 = _S114;
                            }
                            acc_1[0U][0U] = fma(a0_0, b0_0[0U], acc_1[0U][0U]);
                            if(_S104)
                            {
                                acc_1[0U][0U] = fma(a1_0, b1_0[0U], acc_1[0U][0U]);
                            }
                            acc_1[0U][1U] = fma(a0_0, b0_0[1U], acc_1[0U][1U]);
                            if(_S104)
                            {
                                acc_1[0U][1U] = fma(a1_0, b1_0[1U], acc_1[0U][1U]);
                            }
                            acc_1[0U][2U] = fma(a0_0, b0_0[2U], acc_1[0U][2U]);
                            if(_S104)
                            {
                                acc_1[0U][2U] = fma(a1_0, b1_0[2U], acc_1[0U][2U]);
                            }
                            acc_1[0U][3U] = fma(a0_0, b0_0[3U], acc_1[0U][3U]);
                            if(_S104)
                            {
                                acc_1[0U][3U] = fma(a1_0, b1_0[3U], acc_1[0U][3U]);
                            }
                            _S50 = int(3);
                            break;
                        }
                        if(_S50 != int(3))
                        {
                            _S51 = _S50;
                            break;
                        }
                        _S51 = int(2);
                        break;
                    }
                    if(_S51 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(1U >= _S46)
                            {
                                _S52 = int(1);
                                break;
                            }
                            uint e_9 = _S45.baseA_0 + (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_1;
                            thread float a1_1 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_9, &a0_1, &a1_1, &kernelContext_9);
                            }
                            else
                            {
                                float _S115 = elemA_0(e_9, &kernelContext_9);
                                a0_1 = _S115;
                            }
                            acc_1[1U][0U] = fma(a0_1, b0_0[0U], acc_1[1U][0U]);
                            if(_S104)
                            {
                                acc_1[1U][0U] = fma(a1_1, b1_0[0U], acc_1[1U][0U]);
                            }
                            acc_1[1U][1U] = fma(a0_1, b0_0[1U], acc_1[1U][1U]);
                            if(_S104)
                            {
                                acc_1[1U][1U] = fma(a1_1, b1_0[1U], acc_1[1U][1U]);
                            }
                            acc_1[1U][2U] = fma(a0_1, b0_0[2U], acc_1[1U][2U]);
                            if(_S104)
                            {
                                acc_1[1U][2U] = fma(a1_1, b1_0[2U], acc_1[1U][2U]);
                            }
                            acc_1[1U][3U] = fma(a0_1, b0_0[3U], acc_1[1U][3U]);
                            if(_S104)
                            {
                                acc_1[1U][3U] = fma(a1_1, b1_0[3U], acc_1[1U][3U]);
                            }
                            _S52 = int(3);
                            break;
                        }
                        if(_S52 != int(3))
                        {
                            _S53 = _S52;
                            break;
                        }
                        _S53 = int(2);
                        break;
                    }
                    if(_S53 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(2U >= _S46)
                            {
                                _S54 = int(1);
                                break;
                            }
                            uint e_10 = _S45.baseA_0 + 2U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_2;
                            thread float a1_2 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_10, &a0_2, &a1_2, &kernelContext_9);
                            }
                            else
                            {
                                float _S116 = elemA_0(e_10, &kernelContext_9);
                                a0_2 = _S116;
                            }
                            acc_1[2U][0U] = fma(a0_2, b0_0[0U], acc_1[2U][0U]);
                            if(_S104)
                            {
                                acc_1[2U][0U] = fma(a1_2, b1_0[0U], acc_1[2U][0U]);
                            }
                            acc_1[2U][1U] = fma(a0_2, b0_0[1U], acc_1[2U][1U]);
                            if(_S104)
                            {
                                acc_1[2U][1U] = fma(a1_2, b1_0[1U], acc_1[2U][1U]);
                            }
                            acc_1[2U][2U] = fma(a0_2, b0_0[2U], acc_1[2U][2U]);
                            if(_S104)
                            {
                                acc_1[2U][2U] = fma(a1_2, b1_0[2U], acc_1[2U][2U]);
                            }
                            acc_1[2U][3U] = fma(a0_2, b0_0[3U], acc_1[2U][3U]);
                            if(_S104)
                            {
                                acc_1[2U][3U] = fma(a1_2, b1_0[3U], acc_1[2U][3U]);
                            }
                            _S54 = int(3);
                            break;
                        }
                        if(_S54 != int(3))
                        {
                            _S55 = _S54;
                            break;
                        }
                        _S55 = int(2);
                        break;
                    }
                    if(_S55 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(3U >= _S46)
                            {
                                _S56 = int(1);
                                break;
                            }
                            uint e_11 = _S45.baseA_0 + 3U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_3;
                            thread float a1_3 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_11, &a0_3, &a1_3, &kernelContext_9);
                            }
                            else
                            {
                                float _S117 = elemA_0(e_11, &kernelContext_9);
                                a0_3 = _S117;
                            }
                            acc_1[3U][0U] = fma(a0_3, b0_0[0U], acc_1[3U][0U]);
                            if(_S104)
                            {
                                acc_1[3U][0U] = fma(a1_3, b1_0[0U], acc_1[3U][0U]);
                            }
                            acc_1[3U][1U] = fma(a0_3, b0_0[1U], acc_1[3U][1U]);
                            if(_S104)
                            {
                                acc_1[3U][1U] = fma(a1_3, b1_0[1U], acc_1[3U][1U]);
                            }
                            acc_1[3U][2U] = fma(a0_3, b0_0[2U], acc_1[3U][2U]);
                            if(_S104)
                            {
                                acc_1[3U][2U] = fma(a1_3, b1_0[2U], acc_1[3U][2U]);
                            }
                            acc_1[3U][3U] = fma(a0_3, b0_0[3U], acc_1[3U][3U]);
                            if(_S104)
                            {
                                acc_1[3U][3U] = fma(a1_3, b1_0[3U], acc_1[3U][3U]);
                            }
                            _S56 = int(3);
                            break;
                        }
                        if(_S56 != int(3))
                        {
                            _S57 = _S56;
                            break;
                        }
                        _S57 = int(2);
                        break;
                    }
                    if(_S57 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(4U >= _S46)
                            {
                                _S58 = int(1);
                                break;
                            }
                            uint e_12 = _S45.baseA_0 + 4U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_4;
                            thread float a1_4 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_12, &a0_4, &a1_4, &kernelContext_9);
                            }
                            else
                            {
                                float _S118 = elemA_0(e_12, &kernelContext_9);
                                a0_4 = _S118;
                            }
                            acc_1[4U][0U] = fma(a0_4, b0_0[0U], acc_1[4U][0U]);
                            if(_S104)
                            {
                                acc_1[4U][0U] = fma(a1_4, b1_0[0U], acc_1[4U][0U]);
                            }
                            acc_1[4U][1U] = fma(a0_4, b0_0[1U], acc_1[4U][1U]);
                            if(_S104)
                            {
                                acc_1[4U][1U] = fma(a1_4, b1_0[1U], acc_1[4U][1U]);
                            }
                            acc_1[4U][2U] = fma(a0_4, b0_0[2U], acc_1[4U][2U]);
                            if(_S104)
                            {
                                acc_1[4U][2U] = fma(a1_4, b1_0[2U], acc_1[4U][2U]);
                            }
                            acc_1[4U][3U] = fma(a0_4, b0_0[3U], acc_1[4U][3U]);
                            if(_S104)
                            {
                                acc_1[4U][3U] = fma(a1_4, b1_0[3U], acc_1[4U][3U]);
                            }
                            _S58 = int(3);
                            break;
                        }
                        if(_S58 != int(3))
                        {
                            _S59 = _S58;
                            break;
                        }
                        _S59 = int(2);
                        break;
                    }
                    if(_S59 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(5U >= _S46)
                            {
                                _S60 = int(1);
                                break;
                            }
                            uint e_13 = _S45.baseA_0 + 5U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_5;
                            thread float a1_5 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_13, &a0_5, &a1_5, &kernelContext_9);
                            }
                            else
                            {
                                float _S119 = elemA_0(e_13, &kernelContext_9);
                                a0_5 = _S119;
                            }
                            acc_1[5U][0U] = fma(a0_5, b0_0[0U], acc_1[5U][0U]);
                            if(_S104)
                            {
                                acc_1[5U][0U] = fma(a1_5, b1_0[0U], acc_1[5U][0U]);
                            }
                            acc_1[5U][1U] = fma(a0_5, b0_0[1U], acc_1[5U][1U]);
                            if(_S104)
                            {
                                acc_1[5U][1U] = fma(a1_5, b1_0[1U], acc_1[5U][1U]);
                            }
                            acc_1[5U][2U] = fma(a0_5, b0_0[2U], acc_1[5U][2U]);
                            if(_S104)
                            {
                                acc_1[5U][2U] = fma(a1_5, b1_0[2U], acc_1[5U][2U]);
                            }
                            acc_1[5U][3U] = fma(a0_5, b0_0[3U], acc_1[5U][3U]);
                            if(_S104)
                            {
                                acc_1[5U][3U] = fma(a1_5, b1_0[3U], acc_1[5U][3U]);
                            }
                            _S60 = int(3);
                            break;
                        }
                        if(_S60 != int(3))
                        {
                            _S61 = _S60;
                            break;
                        }
                        _S61 = int(2);
                        break;
                    }
                    if(_S61 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(6U >= _S46)
                            {
                                _S62 = int(1);
                                break;
                            }
                            uint e_14 = _S45.baseA_0 + 6U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_6;
                            thread float a1_6 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_14, &a0_6, &a1_6, &kernelContext_9);
                            }
                            else
                            {
                                float _S120 = elemA_0(e_14, &kernelContext_9);
                                a0_6 = _S120;
                            }
                            acc_1[6U][0U] = fma(a0_6, b0_0[0U], acc_1[6U][0U]);
                            if(_S104)
                            {
                                acc_1[6U][0U] = fma(a1_6, b1_0[0U], acc_1[6U][0U]);
                            }
                            acc_1[6U][1U] = fma(a0_6, b0_0[1U], acc_1[6U][1U]);
                            if(_S104)
                            {
                                acc_1[6U][1U] = fma(a1_6, b1_0[1U], acc_1[6U][1U]);
                            }
                            acc_1[6U][2U] = fma(a0_6, b0_0[2U], acc_1[6U][2U]);
                            if(_S104)
                            {
                                acc_1[6U][2U] = fma(a1_6, b1_0[2U], acc_1[6U][2U]);
                            }
                            acc_1[6U][3U] = fma(a0_6, b0_0[3U], acc_1[6U][3U]);
                            if(_S104)
                            {
                                acc_1[6U][3U] = fma(a1_6, b1_0[3U], acc_1[6U][3U]);
                            }
                            _S62 = int(3);
                            break;
                        }
                        if(_S62 != int(3))
                        {
                            _S63 = _S62;
                            break;
                        }
                        _S63 = int(2);
                        break;
                    }
                    if(_S63 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(7U >= _S46)
                            {
                                _S64 = int(1);
                                break;
                            }
                            uint e_15 = _S45.baseA_0 + 7U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_7;
                            thread float a1_7 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_15, &a0_7, &a1_7, &kernelContext_9);
                            }
                            else
                            {
                                float _S121 = elemA_0(e_15, &kernelContext_9);
                                a0_7 = _S121;
                            }
                            acc_1[7U][0U] = fma(a0_7, b0_0[0U], acc_1[7U][0U]);
                            if(_S104)
                            {
                                acc_1[7U][0U] = fma(a1_7, b1_0[0U], acc_1[7U][0U]);
                            }
                            acc_1[7U][1U] = fma(a0_7, b0_0[1U], acc_1[7U][1U]);
                            if(_S104)
                            {
                                acc_1[7U][1U] = fma(a1_7, b1_0[1U], acc_1[7U][1U]);
                            }
                            acc_1[7U][2U] = fma(a0_7, b0_0[2U], acc_1[7U][2U]);
                            if(_S104)
                            {
                                acc_1[7U][2U] = fma(a1_7, b1_0[2U], acc_1[7U][2U]);
                            }
                            acc_1[7U][3U] = fma(a0_7, b0_0[3U], acc_1[7U][3U]);
                            if(_S104)
                            {
                                acc_1[7U][3U] = fma(a1_7, b1_0[3U], acc_1[7U][3U]);
                            }
                            _S64 = int(3);
                            break;
                        }
                        if(_S64 != int(3))
                        {
                            _S65 = _S64;
                            break;
                        }
                        _S65 = int(2);
                        break;
                    }
                    if(_S65 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(8U >= _S46)
                            {
                                _S66 = int(1);
                                break;
                            }
                            uint e_16 = _S45.baseA_0 + 8U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_8;
                            thread float a1_8 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_16, &a0_8, &a1_8, &kernelContext_9);
                            }
                            else
                            {
                                float _S122 = elemA_0(e_16, &kernelContext_9);
                                a0_8 = _S122;
                            }
                            acc_1[8U][0U] = fma(a0_8, b0_0[0U], acc_1[8U][0U]);
                            if(_S104)
                            {
                                acc_1[8U][0U] = fma(a1_8, b1_0[0U], acc_1[8U][0U]);
                            }
                            acc_1[8U][1U] = fma(a0_8, b0_0[1U], acc_1[8U][1U]);
                            if(_S104)
                            {
                                acc_1[8U][1U] = fma(a1_8, b1_0[1U], acc_1[8U][1U]);
                            }
                            acc_1[8U][2U] = fma(a0_8, b0_0[2U], acc_1[8U][2U]);
                            if(_S104)
                            {
                                acc_1[8U][2U] = fma(a1_8, b1_0[2U], acc_1[8U][2U]);
                            }
                            acc_1[8U][3U] = fma(a0_8, b0_0[3U], acc_1[8U][3U]);
                            if(_S104)
                            {
                                acc_1[8U][3U] = fma(a1_8, b1_0[3U], acc_1[8U][3U]);
                            }
                            _S66 = int(3);
                            break;
                        }
                        if(_S66 != int(3))
                        {
                            _S67 = _S66;
                            break;
                        }
                        _S67 = int(2);
                        break;
                    }
                    if(_S67 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(9U >= _S46)
                            {
                                _S68 = int(1);
                                break;
                            }
                            uint e_17 = _S45.baseA_0 + 9U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_9;
                            thread float a1_9 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_17, &a0_9, &a1_9, &kernelContext_9);
                            }
                            else
                            {
                                float _S123 = elemA_0(e_17, &kernelContext_9);
                                a0_9 = _S123;
                            }
                            acc_1[9U][0U] = fma(a0_9, b0_0[0U], acc_1[9U][0U]);
                            if(_S104)
                            {
                                acc_1[9U][0U] = fma(a1_9, b1_0[0U], acc_1[9U][0U]);
                            }
                            acc_1[9U][1U] = fma(a0_9, b0_0[1U], acc_1[9U][1U]);
                            if(_S104)
                            {
                                acc_1[9U][1U] = fma(a1_9, b1_0[1U], acc_1[9U][1U]);
                            }
                            acc_1[9U][2U] = fma(a0_9, b0_0[2U], acc_1[9U][2U]);
                            if(_S104)
                            {
                                acc_1[9U][2U] = fma(a1_9, b1_0[2U], acc_1[9U][2U]);
                            }
                            acc_1[9U][3U] = fma(a0_9, b0_0[3U], acc_1[9U][3U]);
                            if(_S104)
                            {
                                acc_1[9U][3U] = fma(a1_9, b1_0[3U], acc_1[9U][3U]);
                            }
                            _S68 = int(3);
                            break;
                        }
                        if(_S68 != int(3))
                        {
                            _S69 = _S68;
                            break;
                        }
                        _S69 = int(2);
                        break;
                    }
                    if(_S69 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(10U >= _S46)
                            {
                                _S70 = int(1);
                                break;
                            }
                            uint e_18 = _S45.baseA_0 + 10U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_10;
                            thread float a1_10 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_18, &a0_10, &a1_10, &kernelContext_9);
                            }
                            else
                            {
                                float _S124 = elemA_0(e_18, &kernelContext_9);
                                a0_10 = _S124;
                            }
                            acc_1[10U][0U] = fma(a0_10, b0_0[0U], acc_1[10U][0U]);
                            if(_S104)
                            {
                                acc_1[10U][0U] = fma(a1_10, b1_0[0U], acc_1[10U][0U]);
                            }
                            acc_1[10U][1U] = fma(a0_10, b0_0[1U], acc_1[10U][1U]);
                            if(_S104)
                            {
                                acc_1[10U][1U] = fma(a1_10, b1_0[1U], acc_1[10U][1U]);
                            }
                            acc_1[10U][2U] = fma(a0_10, b0_0[2U], acc_1[10U][2U]);
                            if(_S104)
                            {
                                acc_1[10U][2U] = fma(a1_10, b1_0[2U], acc_1[10U][2U]);
                            }
                            acc_1[10U][3U] = fma(a0_10, b0_0[3U], acc_1[10U][3U]);
                            if(_S104)
                            {
                                acc_1[10U][3U] = fma(a1_10, b1_0[3U], acc_1[10U][3U]);
                            }
                            _S70 = int(3);
                            break;
                        }
                        if(_S70 != int(3))
                        {
                            _S71 = _S70;
                            break;
                        }
                        _S71 = int(2);
                        break;
                    }
                    if(_S71 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(11U >= _S46)
                            {
                                _S72 = int(1);
                                break;
                            }
                            uint e_19 = _S45.baseA_0 + 11U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_11;
                            thread float a1_11 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_19, &a0_11, &a1_11, &kernelContext_9);
                            }
                            else
                            {
                                float _S125 = elemA_0(e_19, &kernelContext_9);
                                a0_11 = _S125;
                            }
                            acc_1[11U][0U] = fma(a0_11, b0_0[0U], acc_1[11U][0U]);
                            if(_S104)
                            {
                                acc_1[11U][0U] = fma(a1_11, b1_0[0U], acc_1[11U][0U]);
                            }
                            acc_1[11U][1U] = fma(a0_11, b0_0[1U], acc_1[11U][1U]);
                            if(_S104)
                            {
                                acc_1[11U][1U] = fma(a1_11, b1_0[1U], acc_1[11U][1U]);
                            }
                            acc_1[11U][2U] = fma(a0_11, b0_0[2U], acc_1[11U][2U]);
                            if(_S104)
                            {
                                acc_1[11U][2U] = fma(a1_11, b1_0[2U], acc_1[11U][2U]);
                            }
                            acc_1[11U][3U] = fma(a0_11, b0_0[3U], acc_1[11U][3U]);
                            if(_S104)
                            {
                                acc_1[11U][3U] = fma(a1_11, b1_0[3U], acc_1[11U][3U]);
                            }
                            _S72 = int(3);
                            break;
                        }
                        if(_S72 != int(3))
                        {
                            _S73 = _S72;
                            break;
                        }
                        _S73 = int(2);
                        break;
                    }
                    if(_S73 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(12U >= _S46)
                            {
                                _S74 = int(1);
                                break;
                            }
                            uint e_20 = _S45.baseA_0 + 12U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_12;
                            thread float a1_12 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_20, &a0_12, &a1_12, &kernelContext_9);
                            }
                            else
                            {
                                float _S126 = elemA_0(e_20, &kernelContext_9);
                                a0_12 = _S126;
                            }
                            acc_1[12U][0U] = fma(a0_12, b0_0[0U], acc_1[12U][0U]);
                            if(_S104)
                            {
                                acc_1[12U][0U] = fma(a1_12, b1_0[0U], acc_1[12U][0U]);
                            }
                            acc_1[12U][1U] = fma(a0_12, b0_0[1U], acc_1[12U][1U]);
                            if(_S104)
                            {
                                acc_1[12U][1U] = fma(a1_12, b1_0[1U], acc_1[12U][1U]);
                            }
                            acc_1[12U][2U] = fma(a0_12, b0_0[2U], acc_1[12U][2U]);
                            if(_S104)
                            {
                                acc_1[12U][2U] = fma(a1_12, b1_0[2U], acc_1[12U][2U]);
                            }
                            acc_1[12U][3U] = fma(a0_12, b0_0[3U], acc_1[12U][3U]);
                            if(_S104)
                            {
                                acc_1[12U][3U] = fma(a1_12, b1_0[3U], acc_1[12U][3U]);
                            }
                            _S74 = int(3);
                            break;
                        }
                        if(_S74 != int(3))
                        {
                            _S75 = _S74;
                            break;
                        }
                        _S75 = int(2);
                        break;
                    }
                    if(_S75 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(13U >= _S46)
                            {
                                _S76 = int(1);
                                break;
                            }
                            uint e_21 = _S45.baseA_0 + 13U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_13;
                            thread float a1_13 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_21, &a0_13, &a1_13, &kernelContext_9);
                            }
                            else
                            {
                                float _S127 = elemA_0(e_21, &kernelContext_9);
                                a0_13 = _S127;
                            }
                            acc_1[13U][0U] = fma(a0_13, b0_0[0U], acc_1[13U][0U]);
                            if(_S104)
                            {
                                acc_1[13U][0U] = fma(a1_13, b1_0[0U], acc_1[13U][0U]);
                            }
                            acc_1[13U][1U] = fma(a0_13, b0_0[1U], acc_1[13U][1U]);
                            if(_S104)
                            {
                                acc_1[13U][1U] = fma(a1_13, b1_0[1U], acc_1[13U][1U]);
                            }
                            acc_1[13U][2U] = fma(a0_13, b0_0[2U], acc_1[13U][2U]);
                            if(_S104)
                            {
                                acc_1[13U][2U] = fma(a1_13, b1_0[2U], acc_1[13U][2U]);
                            }
                            acc_1[13U][3U] = fma(a0_13, b0_0[3U], acc_1[13U][3U]);
                            if(_S104)
                            {
                                acc_1[13U][3U] = fma(a1_13, b1_0[3U], acc_1[13U][3U]);
                            }
                            _S76 = int(3);
                            break;
                        }
                        if(_S76 != int(3))
                        {
                            _S77 = _S76;
                            break;
                        }
                        _S77 = int(2);
                        break;
                    }
                    if(_S77 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(14U >= _S46)
                            {
                                _S78 = int(1);
                                break;
                            }
                            uint e_22 = _S45.baseA_0 + 14U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_14;
                            thread float a1_14 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_22, &a0_14, &a1_14, &kernelContext_9);
                            }
                            else
                            {
                                float _S128 = elemA_0(e_22, &kernelContext_9);
                                a0_14 = _S128;
                            }
                            acc_1[14U][0U] = fma(a0_14, b0_0[0U], acc_1[14U][0U]);
                            if(_S104)
                            {
                                acc_1[14U][0U] = fma(a1_14, b1_0[0U], acc_1[14U][0U]);
                            }
                            acc_1[14U][1U] = fma(a0_14, b0_0[1U], acc_1[14U][1U]);
                            if(_S104)
                            {
                                acc_1[14U][1U] = fma(a1_14, b1_0[1U], acc_1[14U][1U]);
                            }
                            acc_1[14U][2U] = fma(a0_14, b0_0[2U], acc_1[14U][2U]);
                            if(_S104)
                            {
                                acc_1[14U][2U] = fma(a1_14, b1_0[2U], acc_1[14U][2U]);
                            }
                            acc_1[14U][3U] = fma(a0_14, b0_0[3U], acc_1[14U][3U]);
                            if(_S104)
                            {
                                acc_1[14U][3U] = fma(a1_14, b1_0[3U], acc_1[14U][3U]);
                            }
                            _S78 = int(3);
                            break;
                        }
                        if(_S78 != int(3))
                        {
                            _S79 = _S78;
                            break;
                        }
                        _S79 = int(2);
                        break;
                    }
                    if(_S79 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(15U >= _S46)
                            {
                                _S80 = int(1);
                                break;
                            }
                            uint e_23 = _S45.baseA_0 + 15U * (&kernelContext_9)->p_0->lda_1 + _S102;
                            thread float a0_15;
                            thread float a1_15 = 0.0;
                            if(_S104)
                            {
                                load2A_0(e_23, &a0_15, &a1_15, &kernelContext_9);
                            }
                            else
                            {
                                float _S129 = elemA_0(e_23, &kernelContext_9);
                                a0_15 = _S129;
                            }
                            acc_1[15U][0U] = fma(a0_15, b0_0[0U], acc_1[15U][0U]);
                            if(_S104)
                            {
                                acc_1[15U][0U] = fma(a1_15, b1_0[0U], acc_1[15U][0U]);
                            }
                            acc_1[15U][1U] = fma(a0_15, b0_0[1U], acc_1[15U][1U]);
                            if(_S104)
                            {
                                acc_1[15U][1U] = fma(a1_15, b1_0[1U], acc_1[15U][1U]);
                            }
                            acc_1[15U][2U] = fma(a0_15, b0_0[2U], acc_1[15U][2U]);
                            if(_S104)
                            {
                                acc_1[15U][2U] = fma(a1_15, b1_0[2U], acc_1[15U][2U]);
                            }
                            acc_1[15U][3U] = fma(a0_15, b0_0[3U], acc_1[15U][3U]);
                            if(_S104)
                            {
                                acc_1[15U][3U] = fma(a1_15, b1_0[3U], acc_1[15U][3U]);
                            }
                            _S80 = int(3);
                            break;
                        }
                        if(_S80 != int(3))
                        {
                            _S81 = _S80;
                            break;
                        }
                        _S81 = int(2);
                        break;
                    }
                    if(_S81 != int(2))
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
        if(i_2 < _S46)
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
        uint w_13 = _S47 / 2U;
        for(;;)
        {
            if(w_13 > 0U)
            {
            }
            else
            {
                break;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane_0 < w_13)
            {
                if(0U < _S48)
                {
                    (*(&kernelContext_9)->red_0)[0U][lid_0] = (*(&kernelContext_9)->red_0)[0U][lid_0] + (*(&kernelContext_9)->red_0)[0U][lid_0 + w_13 * ch_0];
                }
                if(1U < _S48)
                {
                    (*(&kernelContext_9)->red_0)[1U][lid_0] = (*(&kernelContext_9)->red_0)[1U][lid_0] + (*(&kernelContext_9)->red_0)[1U][lid_0 + w_13 * ch_0];
                }
                if(2U < _S48)
                {
                    (*(&kernelContext_9)->red_0)[2U][lid_0] = (*(&kernelContext_9)->red_0)[2U][lid_0] + (*(&kernelContext_9)->red_0)[2U][lid_0 + w_13 * ch_0];
                }
                if(3U < _S48)
                {
                    (*(&kernelContext_9)->red_0)[3U][lid_0] = (*(&kernelContext_9)->red_0)[3U][lid_0] + (*(&kernelContext_9)->red_0)[3U][lid_0 + w_13 * ch_0];
                }
            }
            w_13 = w_13 / 2U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane_0 == 0U)
        {
            uint cc_0 = 0U;
            for(;;)
            {
                if(cc_0 < _S48)
                {
                }
                else
                {
                    break;
                }
                uint _S130 = col0_0 + cc_0;
                if(_S130 < ((&kernelContext_9)->p_0->n_1))
                {
                    float _S131 = (*(&kernelContext_9)->red_0)[cc_0][lid_0];
                    thread GemmProblem_0 _S132 = _S45;
                    gemmStore_0(&_S132, i_2, _S130, _S131, &kernelContext_9);
                }
                cc_0 = cc_0 + 1U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_2 = i_2 + 1U;
    }
    return;
}

