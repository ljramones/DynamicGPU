#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
float lowHalf_0(uint w_0)
{
    return (as_type<half>((ushort)((w_0 & 65535U))));
}

float highHalf_0(uint w_1)
{
    return (as_type<half>((ushort)((w_1 >> 16U))));
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
    uint device* b_0;
    uint device* a_0;
    float device* c_0;
    float device* epBias_0;
    float device* epGate_0;
    float device* epResidual_0;
    array<array<float, int(256)>, int(4)> threadgroup* red_0;
};

float halfB_0(uint e_0, KernelContext_0 thread* kernelContext_0)
{
    uint w_2 = *(kernelContext_0->b_0+(e_0 >> 1U));
    float _S1;
    if((e_0 & 1U) != 0U)
    {
        _S1 = highHalf_0(w_2);
    }
    else
    {
        _S1 = lowHalf_0(w_2);
    }
    return _S1;
}

void load8B_0(uint e_1, array<float, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_1)
{
    if((e_1 & 1U) == 0U)
    {
        uint _S2 = e_1 >> 1U;
        uint device* _S3 = kernelContext_1->b_0+_S2;
        uint w_3 = *_S3;
        (*v_0)[0U] = lowHalf_0(*_S3);
        (*v_0)[1U] = highHalf_0(w_3);
        uint device* _S4 = kernelContext_1->b_0+(_S2 + 1U);
        uint w_4 = *_S4;
        (*v_0)[2U] = lowHalf_0(*_S4);
        (*v_0)[3U] = highHalf_0(w_4);
        uint device* _S5 = kernelContext_1->b_0+(_S2 + 2U);
        uint w_5 = *_S5;
        (*v_0)[4U] = lowHalf_0(*_S5);
        (*v_0)[5U] = highHalf_0(w_5);
        uint device* _S6 = kernelContext_1->b_0+(_S2 + 3U);
        uint w_6 = *_S6;
        (*v_0)[6U] = lowHalf_0(*_S6);
        (*v_0)[7U] = highHalf_0(w_6);
    }
    else
    {
        float _S7 = halfB_0(e_1, kernelContext_1);
        (*v_0)[0U] = _S7;
        float _S8 = halfB_0(e_1 + 1U, kernelContext_1);
        (*v_0)[1U] = _S8;
        float _S9 = halfB_0(e_1 + 2U, kernelContext_1);
        (*v_0)[2U] = _S9;
        float _S10 = halfB_0(e_1 + 3U, kernelContext_1);
        (*v_0)[3U] = _S10;
        float _S11 = halfB_0(e_1 + 4U, kernelContext_1);
        (*v_0)[4U] = _S11;
        float _S12 = halfB_0(e_1 + 5U, kernelContext_1);
        (*v_0)[5U] = _S12;
        float _S13 = halfB_0(e_1 + 6U, kernelContext_1);
        (*v_0)[6U] = _S13;
        float _S14 = halfB_0(e_1 + 7U, kernelContext_1);
        (*v_0)[7U] = _S14;
    }
    return;
}

float halfA_0(uint e_2, KernelContext_0 thread* kernelContext_2)
{
    uint w_7 = *(kernelContext_2->a_0+(e_2 >> 1U));
    float _S15;
    if((e_2 & 1U) != 0U)
    {
        _S15 = highHalf_0(w_7);
    }
    else
    {
        _S15 = lowHalf_0(w_7);
    }
    return _S15;
}

void load8A_0(uint e_3, array<float, int(8)> thread* v_1, KernelContext_0 thread* kernelContext_3)
{
    if((e_3 & 1U) == 0U)
    {
        uint _S16 = e_3 >> 1U;
        uint device* _S17 = kernelContext_3->a_0+_S16;
        uint w_8 = *_S17;
        (*v_1)[0U] = lowHalf_0(*_S17);
        (*v_1)[1U] = highHalf_0(w_8);
        uint device* _S18 = kernelContext_3->a_0+(_S16 + 1U);
        uint w_9 = *_S18;
        (*v_1)[2U] = lowHalf_0(*_S18);
        (*v_1)[3U] = highHalf_0(w_9);
        uint device* _S19 = kernelContext_3->a_0+(_S16 + 2U);
        uint w_10 = *_S19;
        (*v_1)[4U] = lowHalf_0(*_S19);
        (*v_1)[5U] = highHalf_0(w_10);
        uint device* _S20 = kernelContext_3->a_0+(_S16 + 3U);
        uint w_11 = *_S20;
        (*v_1)[6U] = lowHalf_0(*_S20);
        (*v_1)[7U] = highHalf_0(w_11);
    }
    else
    {
        float _S21 = halfA_0(e_3, kernelContext_3);
        (*v_1)[0U] = _S21;
        float _S22 = halfA_0(e_3 + 1U, kernelContext_3);
        (*v_1)[1U] = _S22;
        float _S23 = halfA_0(e_3 + 2U, kernelContext_3);
        (*v_1)[2U] = _S23;
        float _S24 = halfA_0(e_3 + 3U, kernelContext_3);
        (*v_1)[3U] = _S24;
        float _S25 = halfA_0(e_3 + 4U, kernelContext_3);
        (*v_1)[4U] = _S25;
        float _S26 = halfA_0(e_3 + 5U, kernelContext_3);
        (*v_1)[5U] = _S26;
        float _S27 = halfA_0(e_3 + 6U, kernelContext_3);
        (*v_1)[6U] = _S27;
        float _S28 = halfA_0(e_3 + 7U, kernelContext_3);
        (*v_1)[7U] = _S28;
    }
    return;
}

bool dg_is_nan_0(float x_0)
{
    uint _S29 = (as_type<uint>((x_0)));
    bool _S30;
    if((_S29 & 2139095040U) == 2139095040U)
    {
        _S30 = (_S29 & 8388607U) != 0U;
    }
    else
    {
        _S30 = false;
    }
    return _S30;
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
    bool _S31;
    if(a_1 == 0.0)
    {
        _S31 = b_1 == 0.0;
    }
    else
    {
        _S31 = false;
    }
    if(_S31)
    {
        return (as_type<float>(((as_type<uint>((a_1))) & (as_type<uint>((b_1))))));
    }
    float _S32;
    if(a_1 > b_1)
    {
        _S32 = a_1;
    }
    else
    {
        _S32 = b_1;
    }
    return _S32;
}

float gemmEpilogue_0(float r_0, uint i_0, uint j_0, uint idx_0, KernelContext_0 thread* kernelContext_4)
{
    float _S33;
    if((kernelContext_4->p_0->bias_0) == 1U)
    {
        _S33 = r_0 + *(kernelContext_4->epBias_0+j_0);
    }
    else
    {
        if((kernelContext_4->p_0->bias_0) == 2U)
        {
            _S33 = r_0 + *(kernelContext_4->epBias_0+i_0);
        }
        else
        {
            _S33 = r_0;
        }
    }
    if((kernelContext_4->p_0->act_0) == 1U)
    {
        _S33 = dg_max_0(_S33, 0.0);
    }
    else
    {
        if((kernelContext_4->p_0->act_0) == 2U)
        {
            _S33 = 0.5 * _S33 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S33 + 0.04471499845385551 * _S33 * _S33 * _S33))) + 1.0)));
        }
        else
        {
            if((kernelContext_4->p_0->act_0) == 3U)
            {
                _S33 = _S33 / (1.0 + exp(- _S33));
            }
        }
    }
    if((kernelContext_4->p_0->gated_0) != 0U)
    {
        _S33 = _S33 * *(kernelContext_4->epGate_0+idx_0);
    }
    if((kernelContext_4->p_0->residual_0) != 0U)
    {
        _S33 = _S33 + *(kernelContext_4->epResidual_0+idx_0);
    }
    return _S33;
}

void gemmStore_0(uint i_1, uint j_1, uint idx_1, float acc_0, KernelContext_0 thread* kernelContext_5)
{
    float r_1;
    if((kernelContext_5->p_0->beta_0) == 0.0)
    {
        r_1 = kernelContext_5->p_0->alpha_0 * acc_0;
    }
    else
    {
        r_1 = kernelContext_5->p_0->alpha_0 * acc_0 + kernelContext_5->p_0->beta_0 * *(kernelContext_5->c_0+idx_1);
    }
    float device* _S34 = kernelContext_5->c_0+idx_1;
    float _S35 = gemmEpilogue_0(r_1, i_1, j_1, idx_1, kernelContext_5);
    *_S34 = _S35;
    return;
}

[[kernel]] void gemm_skinny(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], uint device* b_2 [[buffer(1)]], uint device* a_2 [[buffer(0)]], float device* c_1 [[buffer(2)]], float device* epBias_1 [[buffer(3)]], float device* epGate_1 [[buffer(4)]], float device* epResidual_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->p_0 = p_1;
    (&kernelContext_6)->b_0 = b_2;
    (&kernelContext_6)->a_0 = a_2;
    (&kernelContext_6)->c_0 = c_1;
    (&kernelContext_6)->epBias_0 = epBias_1;
    (&kernelContext_6)->epGate_0 = epGate_1;
    (&kernelContext_6)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(256)>, int(4)> red_1;
    (&kernelContext_6)->red_0 = &red_1;
    uint lid_0 = local_0.x;
    uint _S36 = group_0.z;
    uint _S37 = _S36 * p_1->strideA_0;
    uint _S38 = _S36 * p_1->strideB_0;
    uint _S39 = _S36 * p_1->strideC_0;
    uint _S40 = min(p_1->m_0, 16U);
    uint k_1 = p_1->k_0;
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
    bool transB_1 = (p_1->transB_0) != 0U;
    uint _S41;
    if(transB_1)
    {
        _S41 = 32U;
    }
    else
    {
        _S41 = 16U;
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
    uint _S42;
    if(transB_1)
    {
        _S42 = 2U;
    }
    else
    {
        _S42 = 4U;
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
    bool _S43;
    if(col0_0 < ((&kernelContext_6)->p_0->n_0))
    {
        _S43 = transB_1;
    }
    else
    {
        _S43 = false;
    }
    uint ch_0;
    uint i_2;
    int _S44;
    int _S45;
    int _S46;
    int _S47;
    int _S48;
    int _S49;
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
    if(_S43)
    {
        uint chunks_0 = k_1 / 8U;
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
            uint _S76 = ch_0 * 8U;
            thread array<array<float, int(8)>, int(2)> bv_0;
            load8B_0(_S38 + min(col0_0, (&kernelContext_6)->p_0->n_0 - 1U) * (&kernelContext_6)->p_0->ldb_0 + _S76, &bv_0[0U], &kernelContext_6);
            load8B_0(_S38 + min(col0_0 + 1U, (&kernelContext_6)->p_0->n_0 - 1U) * (&kernelContext_6)->p_0->ldb_0 + _S76, &bv_0[1U], &kernelContext_6);
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        if(0U >= _S40)
                        {
                            _S44 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_0;
                        load8A_0(_S37 + _S76, &av_0, &kernelContext_6);
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
                        _S44 = int(3);
                        break;
                    }
                    if(_S44 != int(3))
                    {
                        _S45 = _S44;
                        break;
                    }
                    _S45 = int(2);
                    break;
                }
                if(_S45 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(1U >= _S40)
                        {
                            _S46 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_1;
                        load8A_0(_S37 + (&kernelContext_6)->p_0->lda_0 + _S76, &av_1, &kernelContext_6);
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
                        _S46 = int(3);
                        break;
                    }
                    if(_S46 != int(3))
                    {
                        _S47 = _S46;
                        break;
                    }
                    _S47 = int(2);
                    break;
                }
                if(_S47 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(2U >= _S40)
                        {
                            _S48 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_2;
                        load8A_0(_S37 + 2U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_2, &kernelContext_6);
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
                        _S48 = int(3);
                        break;
                    }
                    if(_S48 != int(3))
                    {
                        _S49 = _S48;
                        break;
                    }
                    _S49 = int(2);
                    break;
                }
                if(_S49 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(3U >= _S40)
                        {
                            _S50 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_3;
                        load8A_0(_S37 + 3U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_3, &kernelContext_6);
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
                        if(4U >= _S40)
                        {
                            _S52 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_4;
                        load8A_0(_S37 + 4U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_4, &kernelContext_6);
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
                        if(5U >= _S40)
                        {
                            _S54 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_5;
                        load8A_0(_S37 + 5U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_5, &kernelContext_6);
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
                        if(6U >= _S40)
                        {
                            _S56 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_6;
                        load8A_0(_S37 + 6U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_6, &kernelContext_6);
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
                        if(7U >= _S40)
                        {
                            _S58 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_7;
                        load8A_0(_S37 + 7U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_7, &kernelContext_6);
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
                        if(8U >= _S40)
                        {
                            _S60 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_8;
                        load8A_0(_S37 + 8U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_8, &kernelContext_6);
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
                        if(9U >= _S40)
                        {
                            _S62 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_9;
                        load8A_0(_S37 + 9U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_9, &kernelContext_6);
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
                        if(10U >= _S40)
                        {
                            _S64 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_10;
                        load8A_0(_S37 + 10U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_10, &kernelContext_6);
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
                        if(11U >= _S40)
                        {
                            _S66 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_11;
                        load8A_0(_S37 + 11U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_11, &kernelContext_6);
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
                        if(12U >= _S40)
                        {
                            _S68 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_12;
                        load8A_0(_S37 + 12U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_12, &kernelContext_6);
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
                        if(13U >= _S40)
                        {
                            _S70 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_13;
                        load8A_0(_S37 + 13U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_13, &kernelContext_6);
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
                        if(14U >= _S40)
                        {
                            _S72 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_14;
                        load8A_0(_S37 + 14U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_14, &kernelContext_6);
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
                        if(15U >= _S40)
                        {
                            _S74 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_15;
                        load8A_0(_S37 + 15U * (&kernelContext_6)->p_0->lda_0 + _S76, &av_15, &kernelContext_6);
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
                break;
            }
            ch_0 = ch_0 + 32U;
        }
        if((k_1 % 8U) != 0U)
        {
            _S43 = lane_0 == (chunks_0 % 32U);
        }
        else
        {
            _S43 = false;
        }
        if(_S43)
        {
            i_2 = chunks_0 * 8U;
            for(;;)
            {
                if(i_2 < k_1)
                {
                }
                else
                {
                    break;
                }
                thread array<float, int(2)> bv_1;
                float _S77 = halfB_0(_S38 + min(col0_0, (&kernelContext_6)->p_0->n_0 - 1U) * (&kernelContext_6)->p_0->ldb_0 + i_2, &kernelContext_6);
                bv_1[0U] = _S77;
                float _S78 = halfB_0(_S38 + min(col0_0 + 1U, (&kernelContext_6)->p_0->n_0 - 1U) * (&kernelContext_6)->p_0->ldb_0 + i_2, &kernelContext_6);
                bv_1[1U] = _S78;
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            if(0U >= _S40)
                            {
                                _S44 = int(1);
                                break;
                            }
                            float _S79 = halfA_0(_S37 + i_2, &kernelContext_6);
                            acc_1[0U][0U] = fma(_S79, bv_1[0U], acc_1[0U][0U]);
                            acc_1[0U][1U] = fma(_S79, bv_1[1U], acc_1[0U][1U]);
                            _S44 = int(3);
                            break;
                        }
                        if(_S44 != int(3))
                        {
                            _S45 = _S44;
                            break;
                        }
                        _S45 = int(2);
                        break;
                    }
                    if(_S45 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(1U >= _S40)
                            {
                                _S46 = int(1);
                                break;
                            }
                            float _S80 = halfA_0(_S37 + (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[1U][0U] = fma(_S80, bv_1[0U], acc_1[1U][0U]);
                            acc_1[1U][1U] = fma(_S80, bv_1[1U], acc_1[1U][1U]);
                            _S46 = int(3);
                            break;
                        }
                        if(_S46 != int(3))
                        {
                            _S47 = _S46;
                            break;
                        }
                        _S47 = int(2);
                        break;
                    }
                    if(_S47 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(2U >= _S40)
                            {
                                _S48 = int(1);
                                break;
                            }
                            float _S81 = halfA_0(_S37 + 2U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[2U][0U] = fma(_S81, bv_1[0U], acc_1[2U][0U]);
                            acc_1[2U][1U] = fma(_S81, bv_1[1U], acc_1[2U][1U]);
                            _S48 = int(3);
                            break;
                        }
                        if(_S48 != int(3))
                        {
                            _S49 = _S48;
                            break;
                        }
                        _S49 = int(2);
                        break;
                    }
                    if(_S49 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(3U >= _S40)
                            {
                                _S50 = int(1);
                                break;
                            }
                            float _S82 = halfA_0(_S37 + 3U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[3U][0U] = fma(_S82, bv_1[0U], acc_1[3U][0U]);
                            acc_1[3U][1U] = fma(_S82, bv_1[1U], acc_1[3U][1U]);
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
                            if(4U >= _S40)
                            {
                                _S52 = int(1);
                                break;
                            }
                            float _S83 = halfA_0(_S37 + 4U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[4U][0U] = fma(_S83, bv_1[0U], acc_1[4U][0U]);
                            acc_1[4U][1U] = fma(_S83, bv_1[1U], acc_1[4U][1U]);
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
                            if(5U >= _S40)
                            {
                                _S54 = int(1);
                                break;
                            }
                            float _S84 = halfA_0(_S37 + 5U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[5U][0U] = fma(_S84, bv_1[0U], acc_1[5U][0U]);
                            acc_1[5U][1U] = fma(_S84, bv_1[1U], acc_1[5U][1U]);
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
                            if(6U >= _S40)
                            {
                                _S56 = int(1);
                                break;
                            }
                            float _S85 = halfA_0(_S37 + 6U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[6U][0U] = fma(_S85, bv_1[0U], acc_1[6U][0U]);
                            acc_1[6U][1U] = fma(_S85, bv_1[1U], acc_1[6U][1U]);
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
                            if(7U >= _S40)
                            {
                                _S58 = int(1);
                                break;
                            }
                            float _S86 = halfA_0(_S37 + 7U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[7U][0U] = fma(_S86, bv_1[0U], acc_1[7U][0U]);
                            acc_1[7U][1U] = fma(_S86, bv_1[1U], acc_1[7U][1U]);
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
                            if(8U >= _S40)
                            {
                                _S60 = int(1);
                                break;
                            }
                            float _S87 = halfA_0(_S37 + 8U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[8U][0U] = fma(_S87, bv_1[0U], acc_1[8U][0U]);
                            acc_1[8U][1U] = fma(_S87, bv_1[1U], acc_1[8U][1U]);
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
                            if(9U >= _S40)
                            {
                                _S62 = int(1);
                                break;
                            }
                            float _S88 = halfA_0(_S37 + 9U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[9U][0U] = fma(_S88, bv_1[0U], acc_1[9U][0U]);
                            acc_1[9U][1U] = fma(_S88, bv_1[1U], acc_1[9U][1U]);
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
                            if(10U >= _S40)
                            {
                                _S64 = int(1);
                                break;
                            }
                            float _S89 = halfA_0(_S37 + 10U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[10U][0U] = fma(_S89, bv_1[0U], acc_1[10U][0U]);
                            acc_1[10U][1U] = fma(_S89, bv_1[1U], acc_1[10U][1U]);
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
                            if(11U >= _S40)
                            {
                                _S66 = int(1);
                                break;
                            }
                            float _S90 = halfA_0(_S37 + 11U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[11U][0U] = fma(_S90, bv_1[0U], acc_1[11U][0U]);
                            acc_1[11U][1U] = fma(_S90, bv_1[1U], acc_1[11U][1U]);
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
                            if(12U >= _S40)
                            {
                                _S68 = int(1);
                                break;
                            }
                            float _S91 = halfA_0(_S37 + 12U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[12U][0U] = fma(_S91, bv_1[0U], acc_1[12U][0U]);
                            acc_1[12U][1U] = fma(_S91, bv_1[1U], acc_1[12U][1U]);
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
                            if(13U >= _S40)
                            {
                                _S70 = int(1);
                                break;
                            }
                            float _S92 = halfA_0(_S37 + 13U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[13U][0U] = fma(_S92, bv_1[0U], acc_1[13U][0U]);
                            acc_1[13U][1U] = fma(_S92, bv_1[1U], acc_1[13U][1U]);
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
                            if(14U >= _S40)
                            {
                                _S72 = int(1);
                                break;
                            }
                            float _S93 = halfA_0(_S37 + 14U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[14U][0U] = fma(_S93, bv_1[0U], acc_1[14U][0U]);
                            acc_1[14U][1U] = fma(_S93, bv_1[1U], acc_1[14U][1U]);
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
                            if(15U >= _S40)
                            {
                                _S74 = int(1);
                                break;
                            }
                            float _S94 = halfA_0(_S37 + 15U * (&kernelContext_6)->p_0->lda_0 + i_2, &kernelContext_6);
                            acc_1[15U][0U] = fma(_S94, bv_1[0U], acc_1[15U][0U]);
                            acc_1[15U][1U] = fma(_S94, bv_1[1U], acc_1[15U][1U]);
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
                    break;
                }
                i_2 = i_2 + 1U;
            }
        }
    }
    else
    {
        if(col0_0 < ((&kernelContext_6)->p_0->n_0))
        {
            bool _S95 = ((((&kernelContext_6)->p_0->lda_0) | (p_1->strideA_0)) & 1U) == 0U;
            if(((((&kernelContext_6)->p_0->ldb_0) | (p_1->strideB_0)) & 1U) == 0U)
            {
                _S43 = (col0_0 + 4U) <= ((&kernelContext_6)->p_0->n_0);
            }
            else
            {
                _S43 = false;
            }
            ch_0 = lane_0;
            for(;;)
            {
                uint _S96 = 2U * ch_0;
                if(_S96 < k_1)
                {
                }
                else
                {
                    break;
                }
                uint _S97 = _S96 + 1U;
                bool _S98 = _S97 < k_1;
                thread array<float, int(4)> b0_0;
                thread array<float, int(4)> b1_0;
                uint _S99 = k_1 - 1U;
                uint e_4 = _S38 + min(_S96, _S99) * (&kernelContext_6)->p_0->ldb_0 + col0_0;
                thread array<float, int(4)> v_2;
                if(_S43)
                {
                    uint _S100 = e_4 >> 1U;
                    uint device* _S101 = (&kernelContext_6)->b_0+_S100;
                    uint device* _S102 = (&kernelContext_6)->b_0+(_S100 + 1U);
                    v_2[int(0)] = lowHalf_0(*_S101);
                    v_2[int(1)] = highHalf_0(*_S101);
                    v_2[int(2)] = lowHalf_0(*_S102);
                    v_2[int(3)] = highHalf_0(*_S102);
                }
                else
                {
                    float _S103 = halfB_0(e_4 + min(0U, (&kernelContext_6)->p_0->n_0 - 1U - col0_0), &kernelContext_6);
                    v_2[0U] = _S103;
                    float _S104 = halfB_0(e_4 + min(1U, (&kernelContext_6)->p_0->n_0 - 1U - col0_0), &kernelContext_6);
                    v_2[1U] = _S104;
                    float _S105 = halfB_0(e_4 + min(2U, (&kernelContext_6)->p_0->n_0 - 1U - col0_0), &kernelContext_6);
                    v_2[2U] = _S105;
                    float _S106 = halfB_0(e_4 + min(3U, (&kernelContext_6)->p_0->n_0 - 1U - col0_0), &kernelContext_6);
                    v_2[3U] = _S106;
                }
                b0_0[0U] = v_2[0U];
                b0_0[1U] = v_2[1U];
                b0_0[2U] = v_2[2U];
                b0_0[3U] = v_2[3U];
                uint e_5 = _S38 + min(_S97, _S99) * (&kernelContext_6)->p_0->ldb_0 + col0_0;
                thread array<float, int(4)> v_3;
                if(_S43)
                {
                    uint _S107 = e_5 >> 1U;
                    uint device* _S108 = (&kernelContext_6)->b_0+_S107;
                    uint device* _S109 = (&kernelContext_6)->b_0+(_S107 + 1U);
                    v_3[int(0)] = lowHalf_0(*_S108);
                    v_3[int(1)] = highHalf_0(*_S108);
                    v_3[int(2)] = lowHalf_0(*_S109);
                    v_3[int(3)] = highHalf_0(*_S109);
                }
                else
                {
                    float _S110 = halfB_0(e_5 + min(0U, (&kernelContext_6)->p_0->n_0 - 1U - col0_0), &kernelContext_6);
                    v_3[0U] = _S110;
                    float _S111 = halfB_0(e_5 + min(1U, (&kernelContext_6)->p_0->n_0 - 1U - col0_0), &kernelContext_6);
                    v_3[1U] = _S111;
                    float _S112 = halfB_0(e_5 + min(2U, (&kernelContext_6)->p_0->n_0 - 1U - col0_0), &kernelContext_6);
                    v_3[2U] = _S112;
                    float _S113 = halfB_0(e_5 + min(3U, (&kernelContext_6)->p_0->n_0 - 1U - col0_0), &kernelContext_6);
                    v_3[3U] = _S113;
                }
                b1_0[0U] = v_3[0U];
                b1_0[1U] = v_3[1U];
                b1_0[2U] = v_3[2U];
                b1_0[3U] = v_3[3U];
                for(;;)
                {
                    float a1_0;
                    float a0_0;
                    bool _S114;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(0U >= _S40)
                            {
                                _S44 = int(1);
                                break;
                            }
                            uint e_6 = _S37 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S115 = (&kernelContext_6)->a_0+(e_6 >> 1U);
                                float _S116 = highHalf_0(*_S115);
                                a0_0 = lowHalf_0(*_S115);
                                a1_0 = _S116;
                            }
                            else
                            {
                                float _S117 = halfA_0(e_6, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S118 = halfA_0(e_6 + 1U, &kernelContext_6);
                                    a0_0 = _S118;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S119 = a0_0;
                                a0_0 = _S117;
                                a1_0 = _S119;
                            }
                            acc_1[0U][0U] = fma(a0_0, b0_0[0U], acc_1[0U][0U]);
                            if(_S98)
                            {
                                acc_1[0U][0U] = fma(a1_0, b1_0[0U], acc_1[0U][0U]);
                            }
                            acc_1[0U][1U] = fma(a0_0, b0_0[1U], acc_1[0U][1U]);
                            if(_S98)
                            {
                                acc_1[0U][1U] = fma(a1_0, b1_0[1U], acc_1[0U][1U]);
                            }
                            acc_1[0U][2U] = fma(a0_0, b0_0[2U], acc_1[0U][2U]);
                            if(_S98)
                            {
                                acc_1[0U][2U] = fma(a1_0, b1_0[2U], acc_1[0U][2U]);
                            }
                            acc_1[0U][3U] = fma(a0_0, b0_0[3U], acc_1[0U][3U]);
                            if(_S98)
                            {
                                acc_1[0U][3U] = fma(a1_0, b1_0[3U], acc_1[0U][3U]);
                            }
                            _S44 = int(3);
                            break;
                        }
                        if(_S44 != int(3))
                        {
                            _S45 = _S44;
                            break;
                        }
                        _S45 = int(2);
                        break;
                    }
                    if(_S45 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(1U >= _S40)
                            {
                                _S46 = int(1);
                                break;
                            }
                            uint e_7 = _S37 + (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S120 = (&kernelContext_6)->a_0+(e_7 >> 1U);
                                float _S121 = highHalf_0(*_S120);
                                a0_0 = lowHalf_0(*_S120);
                                a1_0 = _S121;
                            }
                            else
                            {
                                float _S122 = halfA_0(e_7, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S123 = halfA_0(e_7 + 1U, &kernelContext_6);
                                    a0_0 = _S123;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S124 = a0_0;
                                a0_0 = _S122;
                                a1_0 = _S124;
                            }
                            acc_1[1U][0U] = fma(a0_0, b0_0[0U], acc_1[1U][0U]);
                            if(_S98)
                            {
                                acc_1[1U][0U] = fma(a1_0, b1_0[0U], acc_1[1U][0U]);
                            }
                            acc_1[1U][1U] = fma(a0_0, b0_0[1U], acc_1[1U][1U]);
                            if(_S98)
                            {
                                acc_1[1U][1U] = fma(a1_0, b1_0[1U], acc_1[1U][1U]);
                            }
                            acc_1[1U][2U] = fma(a0_0, b0_0[2U], acc_1[1U][2U]);
                            if(_S98)
                            {
                                acc_1[1U][2U] = fma(a1_0, b1_0[2U], acc_1[1U][2U]);
                            }
                            acc_1[1U][3U] = fma(a0_0, b0_0[3U], acc_1[1U][3U]);
                            if(_S98)
                            {
                                acc_1[1U][3U] = fma(a1_0, b1_0[3U], acc_1[1U][3U]);
                            }
                            _S46 = int(3);
                            break;
                        }
                        if(_S46 != int(3))
                        {
                            _S47 = _S46;
                            break;
                        }
                        _S47 = int(2);
                        break;
                    }
                    if(_S47 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(2U >= _S40)
                            {
                                _S48 = int(1);
                                break;
                            }
                            uint e_8 = _S37 + 2U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S125 = (&kernelContext_6)->a_0+(e_8 >> 1U);
                                float _S126 = highHalf_0(*_S125);
                                a0_0 = lowHalf_0(*_S125);
                                a1_0 = _S126;
                            }
                            else
                            {
                                float _S127 = halfA_0(e_8, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S128 = halfA_0(e_8 + 1U, &kernelContext_6);
                                    a0_0 = _S128;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S129 = a0_0;
                                a0_0 = _S127;
                                a1_0 = _S129;
                            }
                            acc_1[2U][0U] = fma(a0_0, b0_0[0U], acc_1[2U][0U]);
                            if(_S98)
                            {
                                acc_1[2U][0U] = fma(a1_0, b1_0[0U], acc_1[2U][0U]);
                            }
                            acc_1[2U][1U] = fma(a0_0, b0_0[1U], acc_1[2U][1U]);
                            if(_S98)
                            {
                                acc_1[2U][1U] = fma(a1_0, b1_0[1U], acc_1[2U][1U]);
                            }
                            acc_1[2U][2U] = fma(a0_0, b0_0[2U], acc_1[2U][2U]);
                            if(_S98)
                            {
                                acc_1[2U][2U] = fma(a1_0, b1_0[2U], acc_1[2U][2U]);
                            }
                            acc_1[2U][3U] = fma(a0_0, b0_0[3U], acc_1[2U][3U]);
                            if(_S98)
                            {
                                acc_1[2U][3U] = fma(a1_0, b1_0[3U], acc_1[2U][3U]);
                            }
                            _S48 = int(3);
                            break;
                        }
                        if(_S48 != int(3))
                        {
                            _S49 = _S48;
                            break;
                        }
                        _S49 = int(2);
                        break;
                    }
                    if(_S49 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(3U >= _S40)
                            {
                                _S50 = int(1);
                                break;
                            }
                            uint e_9 = _S37 + 3U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S130 = (&kernelContext_6)->a_0+(e_9 >> 1U);
                                float _S131 = highHalf_0(*_S130);
                                a0_0 = lowHalf_0(*_S130);
                                a1_0 = _S131;
                            }
                            else
                            {
                                float _S132 = halfA_0(e_9, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S133 = halfA_0(e_9 + 1U, &kernelContext_6);
                                    a0_0 = _S133;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S134 = a0_0;
                                a0_0 = _S132;
                                a1_0 = _S134;
                            }
                            acc_1[3U][0U] = fma(a0_0, b0_0[0U], acc_1[3U][0U]);
                            if(_S98)
                            {
                                acc_1[3U][0U] = fma(a1_0, b1_0[0U], acc_1[3U][0U]);
                            }
                            acc_1[3U][1U] = fma(a0_0, b0_0[1U], acc_1[3U][1U]);
                            if(_S98)
                            {
                                acc_1[3U][1U] = fma(a1_0, b1_0[1U], acc_1[3U][1U]);
                            }
                            acc_1[3U][2U] = fma(a0_0, b0_0[2U], acc_1[3U][2U]);
                            if(_S98)
                            {
                                acc_1[3U][2U] = fma(a1_0, b1_0[2U], acc_1[3U][2U]);
                            }
                            acc_1[3U][3U] = fma(a0_0, b0_0[3U], acc_1[3U][3U]);
                            if(_S98)
                            {
                                acc_1[3U][3U] = fma(a1_0, b1_0[3U], acc_1[3U][3U]);
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
                            if(4U >= _S40)
                            {
                                _S52 = int(1);
                                break;
                            }
                            uint e_10 = _S37 + 4U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S135 = (&kernelContext_6)->a_0+(e_10 >> 1U);
                                float _S136 = highHalf_0(*_S135);
                                a0_0 = lowHalf_0(*_S135);
                                a1_0 = _S136;
                            }
                            else
                            {
                                float _S137 = halfA_0(e_10, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S138 = halfA_0(e_10 + 1U, &kernelContext_6);
                                    a0_0 = _S138;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S139 = a0_0;
                                a0_0 = _S137;
                                a1_0 = _S139;
                            }
                            acc_1[4U][0U] = fma(a0_0, b0_0[0U], acc_1[4U][0U]);
                            if(_S98)
                            {
                                acc_1[4U][0U] = fma(a1_0, b1_0[0U], acc_1[4U][0U]);
                            }
                            acc_1[4U][1U] = fma(a0_0, b0_0[1U], acc_1[4U][1U]);
                            if(_S98)
                            {
                                acc_1[4U][1U] = fma(a1_0, b1_0[1U], acc_1[4U][1U]);
                            }
                            acc_1[4U][2U] = fma(a0_0, b0_0[2U], acc_1[4U][2U]);
                            if(_S98)
                            {
                                acc_1[4U][2U] = fma(a1_0, b1_0[2U], acc_1[4U][2U]);
                            }
                            acc_1[4U][3U] = fma(a0_0, b0_0[3U], acc_1[4U][3U]);
                            if(_S98)
                            {
                                acc_1[4U][3U] = fma(a1_0, b1_0[3U], acc_1[4U][3U]);
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
                            if(5U >= _S40)
                            {
                                _S54 = int(1);
                                break;
                            }
                            uint e_11 = _S37 + 5U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S140 = (&kernelContext_6)->a_0+(e_11 >> 1U);
                                float _S141 = highHalf_0(*_S140);
                                a0_0 = lowHalf_0(*_S140);
                                a1_0 = _S141;
                            }
                            else
                            {
                                float _S142 = halfA_0(e_11, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S143 = halfA_0(e_11 + 1U, &kernelContext_6);
                                    a0_0 = _S143;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S144 = a0_0;
                                a0_0 = _S142;
                                a1_0 = _S144;
                            }
                            acc_1[5U][0U] = fma(a0_0, b0_0[0U], acc_1[5U][0U]);
                            if(_S98)
                            {
                                acc_1[5U][0U] = fma(a1_0, b1_0[0U], acc_1[5U][0U]);
                            }
                            acc_1[5U][1U] = fma(a0_0, b0_0[1U], acc_1[5U][1U]);
                            if(_S98)
                            {
                                acc_1[5U][1U] = fma(a1_0, b1_0[1U], acc_1[5U][1U]);
                            }
                            acc_1[5U][2U] = fma(a0_0, b0_0[2U], acc_1[5U][2U]);
                            if(_S98)
                            {
                                acc_1[5U][2U] = fma(a1_0, b1_0[2U], acc_1[5U][2U]);
                            }
                            acc_1[5U][3U] = fma(a0_0, b0_0[3U], acc_1[5U][3U]);
                            if(_S98)
                            {
                                acc_1[5U][3U] = fma(a1_0, b1_0[3U], acc_1[5U][3U]);
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
                            if(6U >= _S40)
                            {
                                _S56 = int(1);
                                break;
                            }
                            uint e_12 = _S37 + 6U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S145 = (&kernelContext_6)->a_0+(e_12 >> 1U);
                                float _S146 = highHalf_0(*_S145);
                                a0_0 = lowHalf_0(*_S145);
                                a1_0 = _S146;
                            }
                            else
                            {
                                float _S147 = halfA_0(e_12, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S148 = halfA_0(e_12 + 1U, &kernelContext_6);
                                    a0_0 = _S148;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S149 = a0_0;
                                a0_0 = _S147;
                                a1_0 = _S149;
                            }
                            acc_1[6U][0U] = fma(a0_0, b0_0[0U], acc_1[6U][0U]);
                            if(_S98)
                            {
                                acc_1[6U][0U] = fma(a1_0, b1_0[0U], acc_1[6U][0U]);
                            }
                            acc_1[6U][1U] = fma(a0_0, b0_0[1U], acc_1[6U][1U]);
                            if(_S98)
                            {
                                acc_1[6U][1U] = fma(a1_0, b1_0[1U], acc_1[6U][1U]);
                            }
                            acc_1[6U][2U] = fma(a0_0, b0_0[2U], acc_1[6U][2U]);
                            if(_S98)
                            {
                                acc_1[6U][2U] = fma(a1_0, b1_0[2U], acc_1[6U][2U]);
                            }
                            acc_1[6U][3U] = fma(a0_0, b0_0[3U], acc_1[6U][3U]);
                            if(_S98)
                            {
                                acc_1[6U][3U] = fma(a1_0, b1_0[3U], acc_1[6U][3U]);
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
                            if(7U >= _S40)
                            {
                                _S58 = int(1);
                                break;
                            }
                            uint e_13 = _S37 + 7U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S150 = (&kernelContext_6)->a_0+(e_13 >> 1U);
                                float _S151 = highHalf_0(*_S150);
                                a0_0 = lowHalf_0(*_S150);
                                a1_0 = _S151;
                            }
                            else
                            {
                                float _S152 = halfA_0(e_13, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S153 = halfA_0(e_13 + 1U, &kernelContext_6);
                                    a0_0 = _S153;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S154 = a0_0;
                                a0_0 = _S152;
                                a1_0 = _S154;
                            }
                            acc_1[7U][0U] = fma(a0_0, b0_0[0U], acc_1[7U][0U]);
                            if(_S98)
                            {
                                acc_1[7U][0U] = fma(a1_0, b1_0[0U], acc_1[7U][0U]);
                            }
                            acc_1[7U][1U] = fma(a0_0, b0_0[1U], acc_1[7U][1U]);
                            if(_S98)
                            {
                                acc_1[7U][1U] = fma(a1_0, b1_0[1U], acc_1[7U][1U]);
                            }
                            acc_1[7U][2U] = fma(a0_0, b0_0[2U], acc_1[7U][2U]);
                            if(_S98)
                            {
                                acc_1[7U][2U] = fma(a1_0, b1_0[2U], acc_1[7U][2U]);
                            }
                            acc_1[7U][3U] = fma(a0_0, b0_0[3U], acc_1[7U][3U]);
                            if(_S98)
                            {
                                acc_1[7U][3U] = fma(a1_0, b1_0[3U], acc_1[7U][3U]);
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
                            if(8U >= _S40)
                            {
                                _S60 = int(1);
                                break;
                            }
                            uint e_14 = _S37 + 8U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S155 = (&kernelContext_6)->a_0+(e_14 >> 1U);
                                float _S156 = highHalf_0(*_S155);
                                a0_0 = lowHalf_0(*_S155);
                                a1_0 = _S156;
                            }
                            else
                            {
                                float _S157 = halfA_0(e_14, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S158 = halfA_0(e_14 + 1U, &kernelContext_6);
                                    a0_0 = _S158;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S159 = a0_0;
                                a0_0 = _S157;
                                a1_0 = _S159;
                            }
                            acc_1[8U][0U] = fma(a0_0, b0_0[0U], acc_1[8U][0U]);
                            if(_S98)
                            {
                                acc_1[8U][0U] = fma(a1_0, b1_0[0U], acc_1[8U][0U]);
                            }
                            acc_1[8U][1U] = fma(a0_0, b0_0[1U], acc_1[8U][1U]);
                            if(_S98)
                            {
                                acc_1[8U][1U] = fma(a1_0, b1_0[1U], acc_1[8U][1U]);
                            }
                            acc_1[8U][2U] = fma(a0_0, b0_0[2U], acc_1[8U][2U]);
                            if(_S98)
                            {
                                acc_1[8U][2U] = fma(a1_0, b1_0[2U], acc_1[8U][2U]);
                            }
                            acc_1[8U][3U] = fma(a0_0, b0_0[3U], acc_1[8U][3U]);
                            if(_S98)
                            {
                                acc_1[8U][3U] = fma(a1_0, b1_0[3U], acc_1[8U][3U]);
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
                            if(9U >= _S40)
                            {
                                _S62 = int(1);
                                break;
                            }
                            uint e_15 = _S37 + 9U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S160 = (&kernelContext_6)->a_0+(e_15 >> 1U);
                                float _S161 = highHalf_0(*_S160);
                                a0_0 = lowHalf_0(*_S160);
                                a1_0 = _S161;
                            }
                            else
                            {
                                float _S162 = halfA_0(e_15, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S163 = halfA_0(e_15 + 1U, &kernelContext_6);
                                    a0_0 = _S163;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S164 = a0_0;
                                a0_0 = _S162;
                                a1_0 = _S164;
                            }
                            acc_1[9U][0U] = fma(a0_0, b0_0[0U], acc_1[9U][0U]);
                            if(_S98)
                            {
                                acc_1[9U][0U] = fma(a1_0, b1_0[0U], acc_1[9U][0U]);
                            }
                            acc_1[9U][1U] = fma(a0_0, b0_0[1U], acc_1[9U][1U]);
                            if(_S98)
                            {
                                acc_1[9U][1U] = fma(a1_0, b1_0[1U], acc_1[9U][1U]);
                            }
                            acc_1[9U][2U] = fma(a0_0, b0_0[2U], acc_1[9U][2U]);
                            if(_S98)
                            {
                                acc_1[9U][2U] = fma(a1_0, b1_0[2U], acc_1[9U][2U]);
                            }
                            acc_1[9U][3U] = fma(a0_0, b0_0[3U], acc_1[9U][3U]);
                            if(_S98)
                            {
                                acc_1[9U][3U] = fma(a1_0, b1_0[3U], acc_1[9U][3U]);
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
                            if(10U >= _S40)
                            {
                                _S64 = int(1);
                                break;
                            }
                            uint e_16 = _S37 + 10U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S165 = (&kernelContext_6)->a_0+(e_16 >> 1U);
                                float _S166 = highHalf_0(*_S165);
                                a0_0 = lowHalf_0(*_S165);
                                a1_0 = _S166;
                            }
                            else
                            {
                                float _S167 = halfA_0(e_16, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S168 = halfA_0(e_16 + 1U, &kernelContext_6);
                                    a0_0 = _S168;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S169 = a0_0;
                                a0_0 = _S167;
                                a1_0 = _S169;
                            }
                            acc_1[10U][0U] = fma(a0_0, b0_0[0U], acc_1[10U][0U]);
                            if(_S98)
                            {
                                acc_1[10U][0U] = fma(a1_0, b1_0[0U], acc_1[10U][0U]);
                            }
                            acc_1[10U][1U] = fma(a0_0, b0_0[1U], acc_1[10U][1U]);
                            if(_S98)
                            {
                                acc_1[10U][1U] = fma(a1_0, b1_0[1U], acc_1[10U][1U]);
                            }
                            acc_1[10U][2U] = fma(a0_0, b0_0[2U], acc_1[10U][2U]);
                            if(_S98)
                            {
                                acc_1[10U][2U] = fma(a1_0, b1_0[2U], acc_1[10U][2U]);
                            }
                            acc_1[10U][3U] = fma(a0_0, b0_0[3U], acc_1[10U][3U]);
                            if(_S98)
                            {
                                acc_1[10U][3U] = fma(a1_0, b1_0[3U], acc_1[10U][3U]);
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
                            if(11U >= _S40)
                            {
                                _S66 = int(1);
                                break;
                            }
                            uint e_17 = _S37 + 11U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S170 = (&kernelContext_6)->a_0+(e_17 >> 1U);
                                float _S171 = highHalf_0(*_S170);
                                a0_0 = lowHalf_0(*_S170);
                                a1_0 = _S171;
                            }
                            else
                            {
                                float _S172 = halfA_0(e_17, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S173 = halfA_0(e_17 + 1U, &kernelContext_6);
                                    a0_0 = _S173;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S174 = a0_0;
                                a0_0 = _S172;
                                a1_0 = _S174;
                            }
                            acc_1[11U][0U] = fma(a0_0, b0_0[0U], acc_1[11U][0U]);
                            if(_S98)
                            {
                                acc_1[11U][0U] = fma(a1_0, b1_0[0U], acc_1[11U][0U]);
                            }
                            acc_1[11U][1U] = fma(a0_0, b0_0[1U], acc_1[11U][1U]);
                            if(_S98)
                            {
                                acc_1[11U][1U] = fma(a1_0, b1_0[1U], acc_1[11U][1U]);
                            }
                            acc_1[11U][2U] = fma(a0_0, b0_0[2U], acc_1[11U][2U]);
                            if(_S98)
                            {
                                acc_1[11U][2U] = fma(a1_0, b1_0[2U], acc_1[11U][2U]);
                            }
                            acc_1[11U][3U] = fma(a0_0, b0_0[3U], acc_1[11U][3U]);
                            if(_S98)
                            {
                                acc_1[11U][3U] = fma(a1_0, b1_0[3U], acc_1[11U][3U]);
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
                            if(12U >= _S40)
                            {
                                _S68 = int(1);
                                break;
                            }
                            uint e_18 = _S37 + 12U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S175 = (&kernelContext_6)->a_0+(e_18 >> 1U);
                                float _S176 = highHalf_0(*_S175);
                                a0_0 = lowHalf_0(*_S175);
                                a1_0 = _S176;
                            }
                            else
                            {
                                float _S177 = halfA_0(e_18, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S178 = halfA_0(e_18 + 1U, &kernelContext_6);
                                    a0_0 = _S178;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S179 = a0_0;
                                a0_0 = _S177;
                                a1_0 = _S179;
                            }
                            acc_1[12U][0U] = fma(a0_0, b0_0[0U], acc_1[12U][0U]);
                            if(_S98)
                            {
                                acc_1[12U][0U] = fma(a1_0, b1_0[0U], acc_1[12U][0U]);
                            }
                            acc_1[12U][1U] = fma(a0_0, b0_0[1U], acc_1[12U][1U]);
                            if(_S98)
                            {
                                acc_1[12U][1U] = fma(a1_0, b1_0[1U], acc_1[12U][1U]);
                            }
                            acc_1[12U][2U] = fma(a0_0, b0_0[2U], acc_1[12U][2U]);
                            if(_S98)
                            {
                                acc_1[12U][2U] = fma(a1_0, b1_0[2U], acc_1[12U][2U]);
                            }
                            acc_1[12U][3U] = fma(a0_0, b0_0[3U], acc_1[12U][3U]);
                            if(_S98)
                            {
                                acc_1[12U][3U] = fma(a1_0, b1_0[3U], acc_1[12U][3U]);
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
                            if(13U >= _S40)
                            {
                                _S70 = int(1);
                                break;
                            }
                            uint e_19 = _S37 + 13U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S180 = (&kernelContext_6)->a_0+(e_19 >> 1U);
                                float _S181 = highHalf_0(*_S180);
                                a0_0 = lowHalf_0(*_S180);
                                a1_0 = _S181;
                            }
                            else
                            {
                                float _S182 = halfA_0(e_19, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S183 = halfA_0(e_19 + 1U, &kernelContext_6);
                                    a0_0 = _S183;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S184 = a0_0;
                                a0_0 = _S182;
                                a1_0 = _S184;
                            }
                            acc_1[13U][0U] = fma(a0_0, b0_0[0U], acc_1[13U][0U]);
                            if(_S98)
                            {
                                acc_1[13U][0U] = fma(a1_0, b1_0[0U], acc_1[13U][0U]);
                            }
                            acc_1[13U][1U] = fma(a0_0, b0_0[1U], acc_1[13U][1U]);
                            if(_S98)
                            {
                                acc_1[13U][1U] = fma(a1_0, b1_0[1U], acc_1[13U][1U]);
                            }
                            acc_1[13U][2U] = fma(a0_0, b0_0[2U], acc_1[13U][2U]);
                            if(_S98)
                            {
                                acc_1[13U][2U] = fma(a1_0, b1_0[2U], acc_1[13U][2U]);
                            }
                            acc_1[13U][3U] = fma(a0_0, b0_0[3U], acc_1[13U][3U]);
                            if(_S98)
                            {
                                acc_1[13U][3U] = fma(a1_0, b1_0[3U], acc_1[13U][3U]);
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
                            if(14U >= _S40)
                            {
                                _S72 = int(1);
                                break;
                            }
                            uint e_20 = _S37 + 14U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S185 = (&kernelContext_6)->a_0+(e_20 >> 1U);
                                float _S186 = highHalf_0(*_S185);
                                a0_0 = lowHalf_0(*_S185);
                                a1_0 = _S186;
                            }
                            else
                            {
                                float _S187 = halfA_0(e_20, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S188 = halfA_0(e_20 + 1U, &kernelContext_6);
                                    a0_0 = _S188;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S189 = a0_0;
                                a0_0 = _S187;
                                a1_0 = _S189;
                            }
                            acc_1[14U][0U] = fma(a0_0, b0_0[0U], acc_1[14U][0U]);
                            if(_S98)
                            {
                                acc_1[14U][0U] = fma(a1_0, b1_0[0U], acc_1[14U][0U]);
                            }
                            acc_1[14U][1U] = fma(a0_0, b0_0[1U], acc_1[14U][1U]);
                            if(_S98)
                            {
                                acc_1[14U][1U] = fma(a1_0, b1_0[1U], acc_1[14U][1U]);
                            }
                            acc_1[14U][2U] = fma(a0_0, b0_0[2U], acc_1[14U][2U]);
                            if(_S98)
                            {
                                acc_1[14U][2U] = fma(a1_0, b1_0[2U], acc_1[14U][2U]);
                            }
                            acc_1[14U][3U] = fma(a0_0, b0_0[3U], acc_1[14U][3U]);
                            if(_S98)
                            {
                                acc_1[14U][3U] = fma(a1_0, b1_0[3U], acc_1[14U][3U]);
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
                            if(15U >= _S40)
                            {
                                _S74 = int(1);
                                break;
                            }
                            uint e_21 = _S37 + 15U * (&kernelContext_6)->p_0->lda_0 + _S96;
                            if(_S95)
                            {
                                _S114 = _S98;
                            }
                            else
                            {
                                _S114 = false;
                            }
                            if(_S114)
                            {
                                uint device* _S190 = (&kernelContext_6)->a_0+(e_21 >> 1U);
                                float _S191 = highHalf_0(*_S190);
                                a0_0 = lowHalf_0(*_S190);
                                a1_0 = _S191;
                            }
                            else
                            {
                                float _S192 = halfA_0(e_21, &kernelContext_6);
                                if(_S98)
                                {
                                    float _S193 = halfA_0(e_21 + 1U, &kernelContext_6);
                                    a0_0 = _S193;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S194 = a0_0;
                                a0_0 = _S192;
                                a1_0 = _S194;
                            }
                            acc_1[15U][0U] = fma(a0_0, b0_0[0U], acc_1[15U][0U]);
                            if(_S98)
                            {
                                acc_1[15U][0U] = fma(a1_0, b1_0[0U], acc_1[15U][0U]);
                            }
                            acc_1[15U][1U] = fma(a0_0, b0_0[1U], acc_1[15U][1U]);
                            if(_S98)
                            {
                                acc_1[15U][1U] = fma(a1_0, b1_0[1U], acc_1[15U][1U]);
                            }
                            acc_1[15U][2U] = fma(a0_0, b0_0[2U], acc_1[15U][2U]);
                            if(_S98)
                            {
                                acc_1[15U][2U] = fma(a1_0, b1_0[2U], acc_1[15U][2U]);
                            }
                            acc_1[15U][3U] = fma(a0_0, b0_0[3U], acc_1[15U][3U]);
                            if(_S98)
                            {
                                acc_1[15U][3U] = fma(a1_0, b1_0[3U], acc_1[15U][3U]);
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
        if(i_2 < _S40)
        {
        }
        else
        {
            break;
        }
        (*(&kernelContext_6)->red_0)[0U][lid_0] = acc_1[i_2][0U];
        (*(&kernelContext_6)->red_0)[1U][lid_0] = acc_1[i_2][1U];
        (*(&kernelContext_6)->red_0)[2U][lid_0] = acc_1[i_2][2U];
        (*(&kernelContext_6)->red_0)[3U][lid_0] = acc_1[i_2][3U];
        uint w_12 = _S41 / 2U;
        for(;;)
        {
            if(w_12 > 0U)
            {
            }
            else
            {
                break;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane_0 < w_12)
            {
                if(0U < _S42)
                {
                    (*(&kernelContext_6)->red_0)[0U][lid_0] = (*(&kernelContext_6)->red_0)[0U][lid_0] + (*(&kernelContext_6)->red_0)[0U][lid_0 + w_12 * ch_0];
                }
                if(1U < _S42)
                {
                    (*(&kernelContext_6)->red_0)[1U][lid_0] = (*(&kernelContext_6)->red_0)[1U][lid_0] + (*(&kernelContext_6)->red_0)[1U][lid_0 + w_12 * ch_0];
                }
                if(2U < _S42)
                {
                    (*(&kernelContext_6)->red_0)[2U][lid_0] = (*(&kernelContext_6)->red_0)[2U][lid_0] + (*(&kernelContext_6)->red_0)[2U][lid_0 + w_12 * ch_0];
                }
                if(3U < _S42)
                {
                    (*(&kernelContext_6)->red_0)[3U][lid_0] = (*(&kernelContext_6)->red_0)[3U][lid_0] + (*(&kernelContext_6)->red_0)[3U][lid_0 + w_12 * ch_0];
                }
            }
            w_12 = w_12 / 2U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane_0 == 0U)
        {
            uint cc_0 = 0U;
            for(;;)
            {
                if(cc_0 < _S42)
                {
                }
                else
                {
                    break;
                }
                uint _S195 = col0_0 + cc_0;
                if(_S195 < ((&kernelContext_6)->p_0->n_0))
                {
                    gemmStore_0(i_2, _S195, _S39 + i_2 * (&kernelContext_6)->p_0->ldc_0 + col0_0 + cc_0, (*(&kernelContext_6)->red_0)[cc_0][lid_0], &kernelContext_6);
                }
                cc_0 = cc_0 + 1U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_2 = i_2 + 1U;
    }
    return;
}

