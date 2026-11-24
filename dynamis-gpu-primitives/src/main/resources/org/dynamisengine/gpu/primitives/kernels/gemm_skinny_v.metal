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
    uint4 device* b_0;
    uint4 device* a_0;
    float device* c_0;
    float device* epBias_0;
    float device* epGate_0;
    float device* epResidual_0;
    array<array<float, int(256)>, int(4)> threadgroup* red_0;
};

void load8B_0(uint e_0, array<float, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_0)
{
    uint4 device* _S1 = kernelContext_0->b_0+(e_0 >> 3U);
    uint4 _S2 = *_S1;
    (*v_0)[0U] = lowHalf_0((*_S1)[0U]);
    (*v_0)[1U] = highHalf_0(_S2[0U]);
    (*v_0)[2U] = lowHalf_0(_S2[1U]);
    (*v_0)[3U] = highHalf_0(_S2[1U]);
    (*v_0)[4U] = lowHalf_0(_S2[2U]);
    (*v_0)[5U] = highHalf_0(_S2[2U]);
    (*v_0)[6U] = lowHalf_0(_S2[3U]);
    (*v_0)[7U] = highHalf_0(_S2[3U]);
    return;
}

void load8A_0(uint e_1, array<float, int(8)> thread* v_1, KernelContext_0 thread* kernelContext_1)
{
    uint4 device* _S3 = kernelContext_1->a_0+(e_1 >> 3U);
    uint4 _S4 = *_S3;
    (*v_1)[0U] = lowHalf_0((*_S3)[0U]);
    (*v_1)[1U] = highHalf_0(_S4[0U]);
    (*v_1)[2U] = lowHalf_0(_S4[1U]);
    (*v_1)[3U] = highHalf_0(_S4[1U]);
    (*v_1)[4U] = lowHalf_0(_S4[2U]);
    (*v_1)[5U] = highHalf_0(_S4[2U]);
    (*v_1)[6U] = lowHalf_0(_S4[3U]);
    (*v_1)[7U] = highHalf_0(_S4[3U]);
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

float gemmEpilogue_0(float r_0, uint i_0, uint j_0, uint idx_0, KernelContext_0 thread* kernelContext_2)
{
    float _S9;
    if((kernelContext_2->p_0->bias_0) == 1U)
    {
        _S9 = r_0 + *(kernelContext_2->epBias_0+j_0);
    }
    else
    {
        if((kernelContext_2->p_0->bias_0) == 2U)
        {
            _S9 = r_0 + *(kernelContext_2->epBias_0+i_0);
        }
        else
        {
            _S9 = r_0;
        }
    }
    if((kernelContext_2->p_0->act_0) == 1U)
    {
        _S9 = dg_max_0(_S9, 0.0);
    }
    else
    {
        if((kernelContext_2->p_0->act_0) == 2U)
        {
            _S9 = 0.5 * _S9 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S9 + 0.04471499845385551 * _S9 * _S9 * _S9))) + 1.0)));
        }
        else
        {
            if((kernelContext_2->p_0->act_0) == 3U)
            {
                _S9 = _S9 / (1.0 + exp(- _S9));
            }
        }
    }
    if((kernelContext_2->p_0->gated_0) != 0U)
    {
        _S9 = _S9 * *(kernelContext_2->epGate_0+idx_0);
    }
    if((kernelContext_2->p_0->residual_0) != 0U)
    {
        _S9 = _S9 + *(kernelContext_2->epResidual_0+idx_0);
    }
    return _S9;
}

void gemmStore_0(uint i_1, uint j_1, uint idx_1, float acc_0, KernelContext_0 thread* kernelContext_3)
{
    float r_1;
    if((kernelContext_3->p_0->beta_0) == 0.0)
    {
        r_1 = kernelContext_3->p_0->alpha_0 * acc_0;
    }
    else
    {
        r_1 = kernelContext_3->p_0->alpha_0 * acc_0 + kernelContext_3->p_0->beta_0 * *(kernelContext_3->c_0+idx_1);
    }
    float device* _S10 = kernelContext_3->c_0+idx_1;
    float _S11 = gemmEpilogue_0(r_1, i_1, j_1, idx_1, kernelContext_3);
    *_S10 = _S11;
    return;
}

[[kernel]] void gemm_skinny_v(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], uint4 device* b_2 [[buffer(1)]], uint4 device* a_2 [[buffer(0)]], float device* c_1 [[buffer(2)]], float device* epBias_1 [[buffer(3)]], float device* epGate_1 [[buffer(4)]], float device* epResidual_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->p_0 = p_1;
    (&kernelContext_4)->b_0 = b_2;
    (&kernelContext_4)->a_0 = a_2;
    (&kernelContext_4)->c_0 = c_1;
    (&kernelContext_4)->epBias_0 = epBias_1;
    (&kernelContext_4)->epGate_0 = epGate_1;
    (&kernelContext_4)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(256)>, int(4)> red_1;
    (&kernelContext_4)->red_0 = &red_1;
    uint lid_0 = local_0.x;
    uint _S12 = group_0.z;
    uint _S13 = _S12 * p_1->strideA_0;
    uint _S14 = _S12 * p_1->strideB_0;
    uint _S15 = _S12 * p_1->strideC_0;
    uint _S16 = min(p_1->m_0, 16U);
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
    uint _S17;
    if(transB_1)
    {
        _S17 = 32U;
    }
    else
    {
        _S17 = 16U;
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
    uint _S18;
    if(transB_1)
    {
        _S18 = 2U;
    }
    else
    {
        _S18 = 4U;
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
    bool _S19;
    if(col0_0 < ((&kernelContext_4)->p_0->n_0))
    {
        _S19 = transB_1;
    }
    else
    {
        _S19 = false;
    }
    uint ch_0;
    if(_S19)
    {
        uint _S20 = k_1 / 8U;
        ch_0 = lane_0;
        for(;;)
        {
            if(ch_0 < _S20)
            {
            }
            else
            {
                break;
            }
            uint _S21 = ch_0 * 8U;
            thread array<array<float, int(8)>, int(2)> bv_0;
            load8B_0(_S14 + min(col0_0, (&kernelContext_4)->p_0->n_0 - 1U) * (&kernelContext_4)->p_0->ldb_0 + _S21, &bv_0[0U], &kernelContext_4);
            load8B_0(_S14 + min(col0_0 + 1U, (&kernelContext_4)->p_0->n_0 - 1U) * (&kernelContext_4)->p_0->ldb_0 + _S21, &bv_0[1U], &kernelContext_4);
            for(;;)
            {
                int _S22;
                for(;;)
                {
                    int _S23;
                    for(;;)
                    {
                        if(0U >= _S16)
                        {
                            _S23 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_0;
                        load8A_0(_S13 + _S21, &av_0, &kernelContext_4);
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
                        _S23 = int(3);
                        break;
                    }
                    if(_S23 != int(3))
                    {
                        _S22 = _S23;
                        break;
                    }
                    _S22 = int(2);
                    break;
                }
                if(_S22 != int(2))
                {
                    break;
                }
                int _S24;
                for(;;)
                {
                    int _S25;
                    for(;;)
                    {
                        if(1U >= _S16)
                        {
                            _S25 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_1;
                        load8A_0(_S13 + (&kernelContext_4)->p_0->lda_0 + _S21, &av_1, &kernelContext_4);
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
                        _S25 = int(3);
                        break;
                    }
                    if(_S25 != int(3))
                    {
                        _S24 = _S25;
                        break;
                    }
                    _S24 = int(2);
                    break;
                }
                if(_S24 != int(2))
                {
                    break;
                }
                int _S26;
                for(;;)
                {
                    int _S27;
                    for(;;)
                    {
                        if(2U >= _S16)
                        {
                            _S27 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_2;
                        load8A_0(_S13 + 2U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_2, &kernelContext_4);
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
                        _S27 = int(3);
                        break;
                    }
                    if(_S27 != int(3))
                    {
                        _S26 = _S27;
                        break;
                    }
                    _S26 = int(2);
                    break;
                }
                if(_S26 != int(2))
                {
                    break;
                }
                int _S28;
                for(;;)
                {
                    int _S29;
                    for(;;)
                    {
                        if(3U >= _S16)
                        {
                            _S29 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_3;
                        load8A_0(_S13 + 3U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_3, &kernelContext_4);
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
                        _S29 = int(3);
                        break;
                    }
                    if(_S29 != int(3))
                    {
                        _S28 = _S29;
                        break;
                    }
                    _S28 = int(2);
                    break;
                }
                if(_S28 != int(2))
                {
                    break;
                }
                int _S30;
                for(;;)
                {
                    int _S31;
                    for(;;)
                    {
                        if(4U >= _S16)
                        {
                            _S31 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_4;
                        load8A_0(_S13 + 4U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_4, &kernelContext_4);
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
                        _S31 = int(3);
                        break;
                    }
                    if(_S31 != int(3))
                    {
                        _S30 = _S31;
                        break;
                    }
                    _S30 = int(2);
                    break;
                }
                if(_S30 != int(2))
                {
                    break;
                }
                int _S32;
                for(;;)
                {
                    int _S33;
                    for(;;)
                    {
                        if(5U >= _S16)
                        {
                            _S33 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_5;
                        load8A_0(_S13 + 5U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_5, &kernelContext_4);
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
                        _S33 = int(3);
                        break;
                    }
                    if(_S33 != int(3))
                    {
                        _S32 = _S33;
                        break;
                    }
                    _S32 = int(2);
                    break;
                }
                if(_S32 != int(2))
                {
                    break;
                }
                int _S34;
                for(;;)
                {
                    int _S35;
                    for(;;)
                    {
                        if(6U >= _S16)
                        {
                            _S35 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_6;
                        load8A_0(_S13 + 6U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_6, &kernelContext_4);
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
                        _S35 = int(3);
                        break;
                    }
                    if(_S35 != int(3))
                    {
                        _S34 = _S35;
                        break;
                    }
                    _S34 = int(2);
                    break;
                }
                if(_S34 != int(2))
                {
                    break;
                }
                int _S36;
                for(;;)
                {
                    int _S37;
                    for(;;)
                    {
                        if(7U >= _S16)
                        {
                            _S37 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_7;
                        load8A_0(_S13 + 7U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_7, &kernelContext_4);
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
                        _S37 = int(3);
                        break;
                    }
                    if(_S37 != int(3))
                    {
                        _S36 = _S37;
                        break;
                    }
                    _S36 = int(2);
                    break;
                }
                if(_S36 != int(2))
                {
                    break;
                }
                int _S38;
                for(;;)
                {
                    int _S39;
                    for(;;)
                    {
                        if(8U >= _S16)
                        {
                            _S39 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_8;
                        load8A_0(_S13 + 8U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_8, &kernelContext_4);
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
                        _S39 = int(3);
                        break;
                    }
                    if(_S39 != int(3))
                    {
                        _S38 = _S39;
                        break;
                    }
                    _S38 = int(2);
                    break;
                }
                if(_S38 != int(2))
                {
                    break;
                }
                int _S40;
                for(;;)
                {
                    int _S41;
                    for(;;)
                    {
                        if(9U >= _S16)
                        {
                            _S41 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_9;
                        load8A_0(_S13 + 9U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_9, &kernelContext_4);
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
                        _S41 = int(3);
                        break;
                    }
                    if(_S41 != int(3))
                    {
                        _S40 = _S41;
                        break;
                    }
                    _S40 = int(2);
                    break;
                }
                if(_S40 != int(2))
                {
                    break;
                }
                int _S42;
                for(;;)
                {
                    int _S43;
                    for(;;)
                    {
                        if(10U >= _S16)
                        {
                            _S43 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_10;
                        load8A_0(_S13 + 10U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_10, &kernelContext_4);
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
                        _S43 = int(3);
                        break;
                    }
                    if(_S43 != int(3))
                    {
                        _S42 = _S43;
                        break;
                    }
                    _S42 = int(2);
                    break;
                }
                if(_S42 != int(2))
                {
                    break;
                }
                int _S44;
                for(;;)
                {
                    int _S45;
                    for(;;)
                    {
                        if(11U >= _S16)
                        {
                            _S45 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_11;
                        load8A_0(_S13 + 11U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_11, &kernelContext_4);
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
                        _S45 = int(3);
                        break;
                    }
                    if(_S45 != int(3))
                    {
                        _S44 = _S45;
                        break;
                    }
                    _S44 = int(2);
                    break;
                }
                if(_S44 != int(2))
                {
                    break;
                }
                int _S46;
                for(;;)
                {
                    int _S47;
                    for(;;)
                    {
                        if(12U >= _S16)
                        {
                            _S47 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_12;
                        load8A_0(_S13 + 12U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_12, &kernelContext_4);
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
                        _S47 = int(3);
                        break;
                    }
                    if(_S47 != int(3))
                    {
                        _S46 = _S47;
                        break;
                    }
                    _S46 = int(2);
                    break;
                }
                if(_S46 != int(2))
                {
                    break;
                }
                int _S48;
                for(;;)
                {
                    int _S49;
                    for(;;)
                    {
                        if(13U >= _S16)
                        {
                            _S49 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_13;
                        load8A_0(_S13 + 13U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_13, &kernelContext_4);
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
                        _S49 = int(3);
                        break;
                    }
                    if(_S49 != int(3))
                    {
                        _S48 = _S49;
                        break;
                    }
                    _S48 = int(2);
                    break;
                }
                if(_S48 != int(2))
                {
                    break;
                }
                int _S50;
                for(;;)
                {
                    int _S51;
                    for(;;)
                    {
                        if(14U >= _S16)
                        {
                            _S51 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_14;
                        load8A_0(_S13 + 14U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_14, &kernelContext_4);
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
                        _S51 = int(3);
                        break;
                    }
                    if(_S51 != int(3))
                    {
                        _S50 = _S51;
                        break;
                    }
                    _S50 = int(2);
                    break;
                }
                if(_S50 != int(2))
                {
                    break;
                }
                int _S52;
                for(;;)
                {
                    int _S53;
                    for(;;)
                    {
                        if(15U >= _S16)
                        {
                            _S53 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_15;
                        load8A_0(_S13 + 15U * (&kernelContext_4)->p_0->lda_0 + _S21, &av_15, &kernelContext_4);
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
                        _S53 = int(3);
                        break;
                    }
                    if(_S53 != int(3))
                    {
                        _S52 = _S53;
                        break;
                    }
                    _S52 = int(2);
                    break;
                }
                if(_S52 != int(2))
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
        if(i_2 < _S16)
        {
        }
        else
        {
            break;
        }
        (*(&kernelContext_4)->red_0)[0U][lid_0] = acc_1[i_2][0U];
        (*(&kernelContext_4)->red_0)[1U][lid_0] = acc_1[i_2][1U];
        (*(&kernelContext_4)->red_0)[2U][lid_0] = acc_1[i_2][2U];
        (*(&kernelContext_4)->red_0)[3U][lid_0] = acc_1[i_2][3U];
        uint w_2 = _S17 / 2U;
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
                if(0U < _S18)
                {
                    (*(&kernelContext_4)->red_0)[0U][lid_0] = (*(&kernelContext_4)->red_0)[0U][lid_0] + (*(&kernelContext_4)->red_0)[0U][lid_0 + w_2 * ch_0];
                }
                if(1U < _S18)
                {
                    (*(&kernelContext_4)->red_0)[1U][lid_0] = (*(&kernelContext_4)->red_0)[1U][lid_0] + (*(&kernelContext_4)->red_0)[1U][lid_0 + w_2 * ch_0];
                }
                if(2U < _S18)
                {
                    (*(&kernelContext_4)->red_0)[2U][lid_0] = (*(&kernelContext_4)->red_0)[2U][lid_0] + (*(&kernelContext_4)->red_0)[2U][lid_0 + w_2 * ch_0];
                }
                if(3U < _S18)
                {
                    (*(&kernelContext_4)->red_0)[3U][lid_0] = (*(&kernelContext_4)->red_0)[3U][lid_0] + (*(&kernelContext_4)->red_0)[3U][lid_0 + w_2 * ch_0];
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
                if(cc_0 < _S18)
                {
                }
                else
                {
                    break;
                }
                uint _S54 = col0_0 + cc_0;
                if(_S54 < ((&kernelContext_4)->p_0->n_0))
                {
                    gemmStore_0(i_2, _S54, _S15 + i_2 * (&kernelContext_4)->p_0->ldc_0 + col0_0 + cc_0, (*(&kernelContext_4)->red_0)[cc_0][lid_0], &kernelContext_4);
                }
                cc_0 = cc_0 + 1U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_2 = i_2 + 1U;
    }
    return;
}

