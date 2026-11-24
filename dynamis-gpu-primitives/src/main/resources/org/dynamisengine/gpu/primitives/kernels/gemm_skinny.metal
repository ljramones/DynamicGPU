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
};

struct KernelContext_0
{
    GemmParams_0 constant* p_0;
    uint device* b_0;
    uint device* a_0;
    float device* c_0;
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

void store_0(uint idx_0, float acc_0, KernelContext_0 thread* kernelContext_4)
{
    float v_2;
    if((kernelContext_4->p_0->beta_0) == 0.0)
    {
        v_2 = kernelContext_4->p_0->alpha_0 * acc_0;
    }
    else
    {
        v_2 = kernelContext_4->p_0->alpha_0 * acc_0 + kernelContext_4->p_0->beta_0 * *(kernelContext_4->c_0+idx_0);
    }
    *(kernelContext_4->c_0+idx_0) = v_2;
    return;
}

[[kernel]] void gemm_skinny(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(3)]], uint device* b_1 [[buffer(1)]], uint device* a_1 [[buffer(0)]], float device* c_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->p_0 = p_1;
    (&kernelContext_5)->b_0 = b_1;
    (&kernelContext_5)->a_0 = a_1;
    (&kernelContext_5)->c_0 = c_1;
    threadgroup array<array<float, int(256)>, int(4)> red_1;
    (&kernelContext_5)->red_0 = &red_1;
    uint lid_0 = local_0.x;
    uint _S29 = group_0.z;
    uint _S30 = _S29 * p_1->strideA_0;
    uint _S31 = _S29 * p_1->strideB_0;
    uint _S32 = _S29 * p_1->strideC_0;
    uint _S33 = min(p_1->m_0, 16U);
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
    uint _S34;
    if(transB_1)
    {
        _S34 = 32U;
    }
    else
    {
        _S34 = 16U;
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
    uint _S35;
    if(transB_1)
    {
        _S35 = 2U;
    }
    else
    {
        _S35 = 4U;
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
    bool _S36;
    if(col0_0 < ((&kernelContext_5)->p_0->n_0))
    {
        _S36 = transB_1;
    }
    else
    {
        _S36 = false;
    }
    uint ch_0;
    uint i_0;
    int _S37;
    int _S38;
    int _S39;
    int _S40;
    int _S41;
    int _S42;
    int _S43;
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
    if(_S36)
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
            uint _S69 = ch_0 * 8U;
            thread array<array<float, int(8)>, int(2)> bv_0;
            load8B_0(_S31 + min(col0_0, (&kernelContext_5)->p_0->n_0 - 1U) * (&kernelContext_5)->p_0->ldb_0 + _S69, &bv_0[0U], &kernelContext_5);
            load8B_0(_S31 + min(col0_0 + 1U, (&kernelContext_5)->p_0->n_0 - 1U) * (&kernelContext_5)->p_0->ldb_0 + _S69, &bv_0[1U], &kernelContext_5);
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        if(0U >= _S33)
                        {
                            _S37 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_0;
                        load8A_0(_S30 + _S69, &av_0, &kernelContext_5);
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
                        if(1U >= _S33)
                        {
                            _S39 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_1;
                        load8A_0(_S30 + (&kernelContext_5)->p_0->lda_0 + _S69, &av_1, &kernelContext_5);
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
                        if(2U >= _S33)
                        {
                            _S41 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_2;
                        load8A_0(_S30 + 2U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_2, &kernelContext_5);
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
                        if(3U >= _S33)
                        {
                            _S43 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_3;
                        load8A_0(_S30 + 3U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_3, &kernelContext_5);
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
                for(;;)
                {
                    for(;;)
                    {
                        if(4U >= _S33)
                        {
                            _S45 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_4;
                        load8A_0(_S30 + 4U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_4, &kernelContext_5);
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
                        _S45 = int(3);
                        break;
                    }
                    if(_S45 != int(3))
                    {
                        _S46 = _S45;
                        break;
                    }
                    _S46 = int(2);
                    break;
                }
                if(_S46 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(5U >= _S33)
                        {
                            _S47 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_5;
                        load8A_0(_S30 + 5U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_5, &kernelContext_5);
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
                        _S47 = int(3);
                        break;
                    }
                    if(_S47 != int(3))
                    {
                        _S48 = _S47;
                        break;
                    }
                    _S48 = int(2);
                    break;
                }
                if(_S48 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(6U >= _S33)
                        {
                            _S49 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_6;
                        load8A_0(_S30 + 6U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_6, &kernelContext_5);
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
                        _S49 = int(3);
                        break;
                    }
                    if(_S49 != int(3))
                    {
                        _S50 = _S49;
                        break;
                    }
                    _S50 = int(2);
                    break;
                }
                if(_S50 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(7U >= _S33)
                        {
                            _S51 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_7;
                        load8A_0(_S30 + 7U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_7, &kernelContext_5);
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
                        _S51 = int(3);
                        break;
                    }
                    if(_S51 != int(3))
                    {
                        _S52 = _S51;
                        break;
                    }
                    _S52 = int(2);
                    break;
                }
                if(_S52 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(8U >= _S33)
                        {
                            _S53 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_8;
                        load8A_0(_S30 + 8U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_8, &kernelContext_5);
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
                        _S53 = int(3);
                        break;
                    }
                    if(_S53 != int(3))
                    {
                        _S54 = _S53;
                        break;
                    }
                    _S54 = int(2);
                    break;
                }
                if(_S54 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(9U >= _S33)
                        {
                            _S55 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_9;
                        load8A_0(_S30 + 9U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_9, &kernelContext_5);
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
                        _S55 = int(3);
                        break;
                    }
                    if(_S55 != int(3))
                    {
                        _S56 = _S55;
                        break;
                    }
                    _S56 = int(2);
                    break;
                }
                if(_S56 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(10U >= _S33)
                        {
                            _S57 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_10;
                        load8A_0(_S30 + 10U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_10, &kernelContext_5);
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
                        _S57 = int(3);
                        break;
                    }
                    if(_S57 != int(3))
                    {
                        _S58 = _S57;
                        break;
                    }
                    _S58 = int(2);
                    break;
                }
                if(_S58 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(11U >= _S33)
                        {
                            _S59 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_11;
                        load8A_0(_S30 + 11U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_11, &kernelContext_5);
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
                        _S59 = int(3);
                        break;
                    }
                    if(_S59 != int(3))
                    {
                        _S60 = _S59;
                        break;
                    }
                    _S60 = int(2);
                    break;
                }
                if(_S60 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(12U >= _S33)
                        {
                            _S61 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_12;
                        load8A_0(_S30 + 12U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_12, &kernelContext_5);
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
                        _S61 = int(3);
                        break;
                    }
                    if(_S61 != int(3))
                    {
                        _S62 = _S61;
                        break;
                    }
                    _S62 = int(2);
                    break;
                }
                if(_S62 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(13U >= _S33)
                        {
                            _S63 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_13;
                        load8A_0(_S30 + 13U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_13, &kernelContext_5);
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
                        _S63 = int(3);
                        break;
                    }
                    if(_S63 != int(3))
                    {
                        _S64 = _S63;
                        break;
                    }
                    _S64 = int(2);
                    break;
                }
                if(_S64 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(14U >= _S33)
                        {
                            _S65 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_14;
                        load8A_0(_S30 + 14U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_14, &kernelContext_5);
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
                        _S65 = int(3);
                        break;
                    }
                    if(_S65 != int(3))
                    {
                        _S66 = _S65;
                        break;
                    }
                    _S66 = int(2);
                    break;
                }
                if(_S66 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(15U >= _S33)
                        {
                            _S67 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_15;
                        load8A_0(_S30 + 15U * (&kernelContext_5)->p_0->lda_0 + _S69, &av_15, &kernelContext_5);
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
                        _S67 = int(3);
                        break;
                    }
                    if(_S67 != int(3))
                    {
                        _S68 = _S67;
                        break;
                    }
                    _S68 = int(2);
                    break;
                }
                if(_S68 != int(2))
                {
                    break;
                }
                break;
            }
            ch_0 = ch_0 + 32U;
        }
        if((k_1 % 8U) != 0U)
        {
            _S36 = lane_0 == (chunks_0 % 32U);
        }
        else
        {
            _S36 = false;
        }
        if(_S36)
        {
            i_0 = chunks_0 * 8U;
            for(;;)
            {
                if(i_0 < k_1)
                {
                }
                else
                {
                    break;
                }
                thread array<float, int(2)> bv_1;
                float _S70 = halfB_0(_S31 + min(col0_0, (&kernelContext_5)->p_0->n_0 - 1U) * (&kernelContext_5)->p_0->ldb_0 + i_0, &kernelContext_5);
                bv_1[0U] = _S70;
                float _S71 = halfB_0(_S31 + min(col0_0 + 1U, (&kernelContext_5)->p_0->n_0 - 1U) * (&kernelContext_5)->p_0->ldb_0 + i_0, &kernelContext_5);
                bv_1[1U] = _S71;
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            if(0U >= _S33)
                            {
                                _S37 = int(1);
                                break;
                            }
                            float _S72 = halfA_0(_S30 + i_0, &kernelContext_5);
                            acc_1[0U][0U] = fma(_S72, bv_1[0U], acc_1[0U][0U]);
                            acc_1[0U][1U] = fma(_S72, bv_1[1U], acc_1[0U][1U]);
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
                            if(1U >= _S33)
                            {
                                _S39 = int(1);
                                break;
                            }
                            float _S73 = halfA_0(_S30 + (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[1U][0U] = fma(_S73, bv_1[0U], acc_1[1U][0U]);
                            acc_1[1U][1U] = fma(_S73, bv_1[1U], acc_1[1U][1U]);
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
                            if(2U >= _S33)
                            {
                                _S41 = int(1);
                                break;
                            }
                            float _S74 = halfA_0(_S30 + 2U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[2U][0U] = fma(_S74, bv_1[0U], acc_1[2U][0U]);
                            acc_1[2U][1U] = fma(_S74, bv_1[1U], acc_1[2U][1U]);
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
                            if(3U >= _S33)
                            {
                                _S43 = int(1);
                                break;
                            }
                            float _S75 = halfA_0(_S30 + 3U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[3U][0U] = fma(_S75, bv_1[0U], acc_1[3U][0U]);
                            acc_1[3U][1U] = fma(_S75, bv_1[1U], acc_1[3U][1U]);
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
                    for(;;)
                    {
                        for(;;)
                        {
                            if(4U >= _S33)
                            {
                                _S45 = int(1);
                                break;
                            }
                            float _S76 = halfA_0(_S30 + 4U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[4U][0U] = fma(_S76, bv_1[0U], acc_1[4U][0U]);
                            acc_1[4U][1U] = fma(_S76, bv_1[1U], acc_1[4U][1U]);
                            _S45 = int(3);
                            break;
                        }
                        if(_S45 != int(3))
                        {
                            _S46 = _S45;
                            break;
                        }
                        _S46 = int(2);
                        break;
                    }
                    if(_S46 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(5U >= _S33)
                            {
                                _S47 = int(1);
                                break;
                            }
                            float _S77 = halfA_0(_S30 + 5U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[5U][0U] = fma(_S77, bv_1[0U], acc_1[5U][0U]);
                            acc_1[5U][1U] = fma(_S77, bv_1[1U], acc_1[5U][1U]);
                            _S47 = int(3);
                            break;
                        }
                        if(_S47 != int(3))
                        {
                            _S48 = _S47;
                            break;
                        }
                        _S48 = int(2);
                        break;
                    }
                    if(_S48 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(6U >= _S33)
                            {
                                _S49 = int(1);
                                break;
                            }
                            float _S78 = halfA_0(_S30 + 6U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[6U][0U] = fma(_S78, bv_1[0U], acc_1[6U][0U]);
                            acc_1[6U][1U] = fma(_S78, bv_1[1U], acc_1[6U][1U]);
                            _S49 = int(3);
                            break;
                        }
                        if(_S49 != int(3))
                        {
                            _S50 = _S49;
                            break;
                        }
                        _S50 = int(2);
                        break;
                    }
                    if(_S50 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(7U >= _S33)
                            {
                                _S51 = int(1);
                                break;
                            }
                            float _S79 = halfA_0(_S30 + 7U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[7U][0U] = fma(_S79, bv_1[0U], acc_1[7U][0U]);
                            acc_1[7U][1U] = fma(_S79, bv_1[1U], acc_1[7U][1U]);
                            _S51 = int(3);
                            break;
                        }
                        if(_S51 != int(3))
                        {
                            _S52 = _S51;
                            break;
                        }
                        _S52 = int(2);
                        break;
                    }
                    if(_S52 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(8U >= _S33)
                            {
                                _S53 = int(1);
                                break;
                            }
                            float _S80 = halfA_0(_S30 + 8U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[8U][0U] = fma(_S80, bv_1[0U], acc_1[8U][0U]);
                            acc_1[8U][1U] = fma(_S80, bv_1[1U], acc_1[8U][1U]);
                            _S53 = int(3);
                            break;
                        }
                        if(_S53 != int(3))
                        {
                            _S54 = _S53;
                            break;
                        }
                        _S54 = int(2);
                        break;
                    }
                    if(_S54 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(9U >= _S33)
                            {
                                _S55 = int(1);
                                break;
                            }
                            float _S81 = halfA_0(_S30 + 9U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[9U][0U] = fma(_S81, bv_1[0U], acc_1[9U][0U]);
                            acc_1[9U][1U] = fma(_S81, bv_1[1U], acc_1[9U][1U]);
                            _S55 = int(3);
                            break;
                        }
                        if(_S55 != int(3))
                        {
                            _S56 = _S55;
                            break;
                        }
                        _S56 = int(2);
                        break;
                    }
                    if(_S56 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(10U >= _S33)
                            {
                                _S57 = int(1);
                                break;
                            }
                            float _S82 = halfA_0(_S30 + 10U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[10U][0U] = fma(_S82, bv_1[0U], acc_1[10U][0U]);
                            acc_1[10U][1U] = fma(_S82, bv_1[1U], acc_1[10U][1U]);
                            _S57 = int(3);
                            break;
                        }
                        if(_S57 != int(3))
                        {
                            _S58 = _S57;
                            break;
                        }
                        _S58 = int(2);
                        break;
                    }
                    if(_S58 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(11U >= _S33)
                            {
                                _S59 = int(1);
                                break;
                            }
                            float _S83 = halfA_0(_S30 + 11U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[11U][0U] = fma(_S83, bv_1[0U], acc_1[11U][0U]);
                            acc_1[11U][1U] = fma(_S83, bv_1[1U], acc_1[11U][1U]);
                            _S59 = int(3);
                            break;
                        }
                        if(_S59 != int(3))
                        {
                            _S60 = _S59;
                            break;
                        }
                        _S60 = int(2);
                        break;
                    }
                    if(_S60 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(12U >= _S33)
                            {
                                _S61 = int(1);
                                break;
                            }
                            float _S84 = halfA_0(_S30 + 12U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[12U][0U] = fma(_S84, bv_1[0U], acc_1[12U][0U]);
                            acc_1[12U][1U] = fma(_S84, bv_1[1U], acc_1[12U][1U]);
                            _S61 = int(3);
                            break;
                        }
                        if(_S61 != int(3))
                        {
                            _S62 = _S61;
                            break;
                        }
                        _S62 = int(2);
                        break;
                    }
                    if(_S62 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(13U >= _S33)
                            {
                                _S63 = int(1);
                                break;
                            }
                            float _S85 = halfA_0(_S30 + 13U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[13U][0U] = fma(_S85, bv_1[0U], acc_1[13U][0U]);
                            acc_1[13U][1U] = fma(_S85, bv_1[1U], acc_1[13U][1U]);
                            _S63 = int(3);
                            break;
                        }
                        if(_S63 != int(3))
                        {
                            _S64 = _S63;
                            break;
                        }
                        _S64 = int(2);
                        break;
                    }
                    if(_S64 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(14U >= _S33)
                            {
                                _S65 = int(1);
                                break;
                            }
                            float _S86 = halfA_0(_S30 + 14U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[14U][0U] = fma(_S86, bv_1[0U], acc_1[14U][0U]);
                            acc_1[14U][1U] = fma(_S86, bv_1[1U], acc_1[14U][1U]);
                            _S65 = int(3);
                            break;
                        }
                        if(_S65 != int(3))
                        {
                            _S66 = _S65;
                            break;
                        }
                        _S66 = int(2);
                        break;
                    }
                    if(_S66 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(15U >= _S33)
                            {
                                _S67 = int(1);
                                break;
                            }
                            float _S87 = halfA_0(_S30 + 15U * (&kernelContext_5)->p_0->lda_0 + i_0, &kernelContext_5);
                            acc_1[15U][0U] = fma(_S87, bv_1[0U], acc_1[15U][0U]);
                            acc_1[15U][1U] = fma(_S87, bv_1[1U], acc_1[15U][1U]);
                            _S67 = int(3);
                            break;
                        }
                        if(_S67 != int(3))
                        {
                            _S68 = _S67;
                            break;
                        }
                        _S68 = int(2);
                        break;
                    }
                    if(_S68 != int(2))
                    {
                        break;
                    }
                    break;
                }
                i_0 = i_0 + 1U;
            }
        }
    }
    else
    {
        if(col0_0 < ((&kernelContext_5)->p_0->n_0))
        {
            bool _S88 = ((((&kernelContext_5)->p_0->lda_0) | (p_1->strideA_0)) & 1U) == 0U;
            if(((((&kernelContext_5)->p_0->ldb_0) | (p_1->strideB_0)) & 1U) == 0U)
            {
                _S36 = (col0_0 + 4U) <= ((&kernelContext_5)->p_0->n_0);
            }
            else
            {
                _S36 = false;
            }
            ch_0 = lane_0;
            for(;;)
            {
                uint _S89 = 2U * ch_0;
                if(_S89 < k_1)
                {
                }
                else
                {
                    break;
                }
                uint _S90 = _S89 + 1U;
                bool _S91 = _S90 < k_1;
                thread array<float, int(4)> b0_0;
                thread array<float, int(4)> b1_0;
                uint _S92 = k_1 - 1U;
                uint e_4 = _S31 + min(_S89, _S92) * (&kernelContext_5)->p_0->ldb_0 + col0_0;
                thread array<float, int(4)> v_3;
                if(_S36)
                {
                    uint _S93 = e_4 >> 1U;
                    uint device* _S94 = (&kernelContext_5)->b_0+_S93;
                    uint device* _S95 = (&kernelContext_5)->b_0+(_S93 + 1U);
                    v_3[int(0)] = lowHalf_0(*_S94);
                    v_3[int(1)] = highHalf_0(*_S94);
                    v_3[int(2)] = lowHalf_0(*_S95);
                    v_3[int(3)] = highHalf_0(*_S95);
                }
                else
                {
                    float _S96 = halfB_0(e_4 + min(0U, (&kernelContext_5)->p_0->n_0 - 1U - col0_0), &kernelContext_5);
                    v_3[0U] = _S96;
                    float _S97 = halfB_0(e_4 + min(1U, (&kernelContext_5)->p_0->n_0 - 1U - col0_0), &kernelContext_5);
                    v_3[1U] = _S97;
                    float _S98 = halfB_0(e_4 + min(2U, (&kernelContext_5)->p_0->n_0 - 1U - col0_0), &kernelContext_5);
                    v_3[2U] = _S98;
                    float _S99 = halfB_0(e_4 + min(3U, (&kernelContext_5)->p_0->n_0 - 1U - col0_0), &kernelContext_5);
                    v_3[3U] = _S99;
                }
                b0_0[0U] = v_3[0U];
                b0_0[1U] = v_3[1U];
                b0_0[2U] = v_3[2U];
                b0_0[3U] = v_3[3U];
                uint e_5 = _S31 + min(_S90, _S92) * (&kernelContext_5)->p_0->ldb_0 + col0_0;
                thread array<float, int(4)> v_4;
                if(_S36)
                {
                    uint _S100 = e_5 >> 1U;
                    uint device* _S101 = (&kernelContext_5)->b_0+_S100;
                    uint device* _S102 = (&kernelContext_5)->b_0+(_S100 + 1U);
                    v_4[int(0)] = lowHalf_0(*_S101);
                    v_4[int(1)] = highHalf_0(*_S101);
                    v_4[int(2)] = lowHalf_0(*_S102);
                    v_4[int(3)] = highHalf_0(*_S102);
                }
                else
                {
                    float _S103 = halfB_0(e_5 + min(0U, (&kernelContext_5)->p_0->n_0 - 1U - col0_0), &kernelContext_5);
                    v_4[0U] = _S103;
                    float _S104 = halfB_0(e_5 + min(1U, (&kernelContext_5)->p_0->n_0 - 1U - col0_0), &kernelContext_5);
                    v_4[1U] = _S104;
                    float _S105 = halfB_0(e_5 + min(2U, (&kernelContext_5)->p_0->n_0 - 1U - col0_0), &kernelContext_5);
                    v_4[2U] = _S105;
                    float _S106 = halfB_0(e_5 + min(3U, (&kernelContext_5)->p_0->n_0 - 1U - col0_0), &kernelContext_5);
                    v_4[3U] = _S106;
                }
                b1_0[0U] = v_4[0U];
                b1_0[1U] = v_4[1U];
                b1_0[2U] = v_4[2U];
                b1_0[3U] = v_4[3U];
                for(;;)
                {
                    float a1_0;
                    float a0_0;
                    bool _S107;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(0U >= _S33)
                            {
                                _S37 = int(1);
                                break;
                            }
                            uint e_6 = _S30 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S108 = (&kernelContext_5)->a_0+(e_6 >> 1U);
                                float _S109 = highHalf_0(*_S108);
                                a0_0 = lowHalf_0(*_S108);
                                a1_0 = _S109;
                            }
                            else
                            {
                                float _S110 = halfA_0(e_6, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S111 = halfA_0(e_6 + 1U, &kernelContext_5);
                                    a0_0 = _S111;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S112 = a0_0;
                                a0_0 = _S110;
                                a1_0 = _S112;
                            }
                            acc_1[0U][0U] = fma(a0_0, b0_0[0U], acc_1[0U][0U]);
                            if(_S91)
                            {
                                acc_1[0U][0U] = fma(a1_0, b1_0[0U], acc_1[0U][0U]);
                            }
                            acc_1[0U][1U] = fma(a0_0, b0_0[1U], acc_1[0U][1U]);
                            if(_S91)
                            {
                                acc_1[0U][1U] = fma(a1_0, b1_0[1U], acc_1[0U][1U]);
                            }
                            acc_1[0U][2U] = fma(a0_0, b0_0[2U], acc_1[0U][2U]);
                            if(_S91)
                            {
                                acc_1[0U][2U] = fma(a1_0, b1_0[2U], acc_1[0U][2U]);
                            }
                            acc_1[0U][3U] = fma(a0_0, b0_0[3U], acc_1[0U][3U]);
                            if(_S91)
                            {
                                acc_1[0U][3U] = fma(a1_0, b1_0[3U], acc_1[0U][3U]);
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
                            if(1U >= _S33)
                            {
                                _S39 = int(1);
                                break;
                            }
                            uint e_7 = _S30 + (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S113 = (&kernelContext_5)->a_0+(e_7 >> 1U);
                                float _S114 = highHalf_0(*_S113);
                                a0_0 = lowHalf_0(*_S113);
                                a1_0 = _S114;
                            }
                            else
                            {
                                float _S115 = halfA_0(e_7, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S116 = halfA_0(e_7 + 1U, &kernelContext_5);
                                    a0_0 = _S116;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S117 = a0_0;
                                a0_0 = _S115;
                                a1_0 = _S117;
                            }
                            acc_1[1U][0U] = fma(a0_0, b0_0[0U], acc_1[1U][0U]);
                            if(_S91)
                            {
                                acc_1[1U][0U] = fma(a1_0, b1_0[0U], acc_1[1U][0U]);
                            }
                            acc_1[1U][1U] = fma(a0_0, b0_0[1U], acc_1[1U][1U]);
                            if(_S91)
                            {
                                acc_1[1U][1U] = fma(a1_0, b1_0[1U], acc_1[1U][1U]);
                            }
                            acc_1[1U][2U] = fma(a0_0, b0_0[2U], acc_1[1U][2U]);
                            if(_S91)
                            {
                                acc_1[1U][2U] = fma(a1_0, b1_0[2U], acc_1[1U][2U]);
                            }
                            acc_1[1U][3U] = fma(a0_0, b0_0[3U], acc_1[1U][3U]);
                            if(_S91)
                            {
                                acc_1[1U][3U] = fma(a1_0, b1_0[3U], acc_1[1U][3U]);
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
                            if(2U >= _S33)
                            {
                                _S41 = int(1);
                                break;
                            }
                            uint e_8 = _S30 + 2U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S118 = (&kernelContext_5)->a_0+(e_8 >> 1U);
                                float _S119 = highHalf_0(*_S118);
                                a0_0 = lowHalf_0(*_S118);
                                a1_0 = _S119;
                            }
                            else
                            {
                                float _S120 = halfA_0(e_8, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S121 = halfA_0(e_8 + 1U, &kernelContext_5);
                                    a0_0 = _S121;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S122 = a0_0;
                                a0_0 = _S120;
                                a1_0 = _S122;
                            }
                            acc_1[2U][0U] = fma(a0_0, b0_0[0U], acc_1[2U][0U]);
                            if(_S91)
                            {
                                acc_1[2U][0U] = fma(a1_0, b1_0[0U], acc_1[2U][0U]);
                            }
                            acc_1[2U][1U] = fma(a0_0, b0_0[1U], acc_1[2U][1U]);
                            if(_S91)
                            {
                                acc_1[2U][1U] = fma(a1_0, b1_0[1U], acc_1[2U][1U]);
                            }
                            acc_1[2U][2U] = fma(a0_0, b0_0[2U], acc_1[2U][2U]);
                            if(_S91)
                            {
                                acc_1[2U][2U] = fma(a1_0, b1_0[2U], acc_1[2U][2U]);
                            }
                            acc_1[2U][3U] = fma(a0_0, b0_0[3U], acc_1[2U][3U]);
                            if(_S91)
                            {
                                acc_1[2U][3U] = fma(a1_0, b1_0[3U], acc_1[2U][3U]);
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
                            if(3U >= _S33)
                            {
                                _S43 = int(1);
                                break;
                            }
                            uint e_9 = _S30 + 3U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S123 = (&kernelContext_5)->a_0+(e_9 >> 1U);
                                float _S124 = highHalf_0(*_S123);
                                a0_0 = lowHalf_0(*_S123);
                                a1_0 = _S124;
                            }
                            else
                            {
                                float _S125 = halfA_0(e_9, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S126 = halfA_0(e_9 + 1U, &kernelContext_5);
                                    a0_0 = _S126;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S127 = a0_0;
                                a0_0 = _S125;
                                a1_0 = _S127;
                            }
                            acc_1[3U][0U] = fma(a0_0, b0_0[0U], acc_1[3U][0U]);
                            if(_S91)
                            {
                                acc_1[3U][0U] = fma(a1_0, b1_0[0U], acc_1[3U][0U]);
                            }
                            acc_1[3U][1U] = fma(a0_0, b0_0[1U], acc_1[3U][1U]);
                            if(_S91)
                            {
                                acc_1[3U][1U] = fma(a1_0, b1_0[1U], acc_1[3U][1U]);
                            }
                            acc_1[3U][2U] = fma(a0_0, b0_0[2U], acc_1[3U][2U]);
                            if(_S91)
                            {
                                acc_1[3U][2U] = fma(a1_0, b1_0[2U], acc_1[3U][2U]);
                            }
                            acc_1[3U][3U] = fma(a0_0, b0_0[3U], acc_1[3U][3U]);
                            if(_S91)
                            {
                                acc_1[3U][3U] = fma(a1_0, b1_0[3U], acc_1[3U][3U]);
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
                    for(;;)
                    {
                        for(;;)
                        {
                            if(4U >= _S33)
                            {
                                _S45 = int(1);
                                break;
                            }
                            uint e_10 = _S30 + 4U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S128 = (&kernelContext_5)->a_0+(e_10 >> 1U);
                                float _S129 = highHalf_0(*_S128);
                                a0_0 = lowHalf_0(*_S128);
                                a1_0 = _S129;
                            }
                            else
                            {
                                float _S130 = halfA_0(e_10, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S131 = halfA_0(e_10 + 1U, &kernelContext_5);
                                    a0_0 = _S131;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S132 = a0_0;
                                a0_0 = _S130;
                                a1_0 = _S132;
                            }
                            acc_1[4U][0U] = fma(a0_0, b0_0[0U], acc_1[4U][0U]);
                            if(_S91)
                            {
                                acc_1[4U][0U] = fma(a1_0, b1_0[0U], acc_1[4U][0U]);
                            }
                            acc_1[4U][1U] = fma(a0_0, b0_0[1U], acc_1[4U][1U]);
                            if(_S91)
                            {
                                acc_1[4U][1U] = fma(a1_0, b1_0[1U], acc_1[4U][1U]);
                            }
                            acc_1[4U][2U] = fma(a0_0, b0_0[2U], acc_1[4U][2U]);
                            if(_S91)
                            {
                                acc_1[4U][2U] = fma(a1_0, b1_0[2U], acc_1[4U][2U]);
                            }
                            acc_1[4U][3U] = fma(a0_0, b0_0[3U], acc_1[4U][3U]);
                            if(_S91)
                            {
                                acc_1[4U][3U] = fma(a1_0, b1_0[3U], acc_1[4U][3U]);
                            }
                            _S45 = int(3);
                            break;
                        }
                        if(_S45 != int(3))
                        {
                            _S46 = _S45;
                            break;
                        }
                        _S46 = int(2);
                        break;
                    }
                    if(_S46 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(5U >= _S33)
                            {
                                _S47 = int(1);
                                break;
                            }
                            uint e_11 = _S30 + 5U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S133 = (&kernelContext_5)->a_0+(e_11 >> 1U);
                                float _S134 = highHalf_0(*_S133);
                                a0_0 = lowHalf_0(*_S133);
                                a1_0 = _S134;
                            }
                            else
                            {
                                float _S135 = halfA_0(e_11, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S136 = halfA_0(e_11 + 1U, &kernelContext_5);
                                    a0_0 = _S136;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S137 = a0_0;
                                a0_0 = _S135;
                                a1_0 = _S137;
                            }
                            acc_1[5U][0U] = fma(a0_0, b0_0[0U], acc_1[5U][0U]);
                            if(_S91)
                            {
                                acc_1[5U][0U] = fma(a1_0, b1_0[0U], acc_1[5U][0U]);
                            }
                            acc_1[5U][1U] = fma(a0_0, b0_0[1U], acc_1[5U][1U]);
                            if(_S91)
                            {
                                acc_1[5U][1U] = fma(a1_0, b1_0[1U], acc_1[5U][1U]);
                            }
                            acc_1[5U][2U] = fma(a0_0, b0_0[2U], acc_1[5U][2U]);
                            if(_S91)
                            {
                                acc_1[5U][2U] = fma(a1_0, b1_0[2U], acc_1[5U][2U]);
                            }
                            acc_1[5U][3U] = fma(a0_0, b0_0[3U], acc_1[5U][3U]);
                            if(_S91)
                            {
                                acc_1[5U][3U] = fma(a1_0, b1_0[3U], acc_1[5U][3U]);
                            }
                            _S47 = int(3);
                            break;
                        }
                        if(_S47 != int(3))
                        {
                            _S48 = _S47;
                            break;
                        }
                        _S48 = int(2);
                        break;
                    }
                    if(_S48 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(6U >= _S33)
                            {
                                _S49 = int(1);
                                break;
                            }
                            uint e_12 = _S30 + 6U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S138 = (&kernelContext_5)->a_0+(e_12 >> 1U);
                                float _S139 = highHalf_0(*_S138);
                                a0_0 = lowHalf_0(*_S138);
                                a1_0 = _S139;
                            }
                            else
                            {
                                float _S140 = halfA_0(e_12, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S141 = halfA_0(e_12 + 1U, &kernelContext_5);
                                    a0_0 = _S141;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S142 = a0_0;
                                a0_0 = _S140;
                                a1_0 = _S142;
                            }
                            acc_1[6U][0U] = fma(a0_0, b0_0[0U], acc_1[6U][0U]);
                            if(_S91)
                            {
                                acc_1[6U][0U] = fma(a1_0, b1_0[0U], acc_1[6U][0U]);
                            }
                            acc_1[6U][1U] = fma(a0_0, b0_0[1U], acc_1[6U][1U]);
                            if(_S91)
                            {
                                acc_1[6U][1U] = fma(a1_0, b1_0[1U], acc_1[6U][1U]);
                            }
                            acc_1[6U][2U] = fma(a0_0, b0_0[2U], acc_1[6U][2U]);
                            if(_S91)
                            {
                                acc_1[6U][2U] = fma(a1_0, b1_0[2U], acc_1[6U][2U]);
                            }
                            acc_1[6U][3U] = fma(a0_0, b0_0[3U], acc_1[6U][3U]);
                            if(_S91)
                            {
                                acc_1[6U][3U] = fma(a1_0, b1_0[3U], acc_1[6U][3U]);
                            }
                            _S49 = int(3);
                            break;
                        }
                        if(_S49 != int(3))
                        {
                            _S50 = _S49;
                            break;
                        }
                        _S50 = int(2);
                        break;
                    }
                    if(_S50 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(7U >= _S33)
                            {
                                _S51 = int(1);
                                break;
                            }
                            uint e_13 = _S30 + 7U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S143 = (&kernelContext_5)->a_0+(e_13 >> 1U);
                                float _S144 = highHalf_0(*_S143);
                                a0_0 = lowHalf_0(*_S143);
                                a1_0 = _S144;
                            }
                            else
                            {
                                float _S145 = halfA_0(e_13, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S146 = halfA_0(e_13 + 1U, &kernelContext_5);
                                    a0_0 = _S146;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S147 = a0_0;
                                a0_0 = _S145;
                                a1_0 = _S147;
                            }
                            acc_1[7U][0U] = fma(a0_0, b0_0[0U], acc_1[7U][0U]);
                            if(_S91)
                            {
                                acc_1[7U][0U] = fma(a1_0, b1_0[0U], acc_1[7U][0U]);
                            }
                            acc_1[7U][1U] = fma(a0_0, b0_0[1U], acc_1[7U][1U]);
                            if(_S91)
                            {
                                acc_1[7U][1U] = fma(a1_0, b1_0[1U], acc_1[7U][1U]);
                            }
                            acc_1[7U][2U] = fma(a0_0, b0_0[2U], acc_1[7U][2U]);
                            if(_S91)
                            {
                                acc_1[7U][2U] = fma(a1_0, b1_0[2U], acc_1[7U][2U]);
                            }
                            acc_1[7U][3U] = fma(a0_0, b0_0[3U], acc_1[7U][3U]);
                            if(_S91)
                            {
                                acc_1[7U][3U] = fma(a1_0, b1_0[3U], acc_1[7U][3U]);
                            }
                            _S51 = int(3);
                            break;
                        }
                        if(_S51 != int(3))
                        {
                            _S52 = _S51;
                            break;
                        }
                        _S52 = int(2);
                        break;
                    }
                    if(_S52 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(8U >= _S33)
                            {
                                _S53 = int(1);
                                break;
                            }
                            uint e_14 = _S30 + 8U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S148 = (&kernelContext_5)->a_0+(e_14 >> 1U);
                                float _S149 = highHalf_0(*_S148);
                                a0_0 = lowHalf_0(*_S148);
                                a1_0 = _S149;
                            }
                            else
                            {
                                float _S150 = halfA_0(e_14, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S151 = halfA_0(e_14 + 1U, &kernelContext_5);
                                    a0_0 = _S151;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S152 = a0_0;
                                a0_0 = _S150;
                                a1_0 = _S152;
                            }
                            acc_1[8U][0U] = fma(a0_0, b0_0[0U], acc_1[8U][0U]);
                            if(_S91)
                            {
                                acc_1[8U][0U] = fma(a1_0, b1_0[0U], acc_1[8U][0U]);
                            }
                            acc_1[8U][1U] = fma(a0_0, b0_0[1U], acc_1[8U][1U]);
                            if(_S91)
                            {
                                acc_1[8U][1U] = fma(a1_0, b1_0[1U], acc_1[8U][1U]);
                            }
                            acc_1[8U][2U] = fma(a0_0, b0_0[2U], acc_1[8U][2U]);
                            if(_S91)
                            {
                                acc_1[8U][2U] = fma(a1_0, b1_0[2U], acc_1[8U][2U]);
                            }
                            acc_1[8U][3U] = fma(a0_0, b0_0[3U], acc_1[8U][3U]);
                            if(_S91)
                            {
                                acc_1[8U][3U] = fma(a1_0, b1_0[3U], acc_1[8U][3U]);
                            }
                            _S53 = int(3);
                            break;
                        }
                        if(_S53 != int(3))
                        {
                            _S54 = _S53;
                            break;
                        }
                        _S54 = int(2);
                        break;
                    }
                    if(_S54 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(9U >= _S33)
                            {
                                _S55 = int(1);
                                break;
                            }
                            uint e_15 = _S30 + 9U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S153 = (&kernelContext_5)->a_0+(e_15 >> 1U);
                                float _S154 = highHalf_0(*_S153);
                                a0_0 = lowHalf_0(*_S153);
                                a1_0 = _S154;
                            }
                            else
                            {
                                float _S155 = halfA_0(e_15, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S156 = halfA_0(e_15 + 1U, &kernelContext_5);
                                    a0_0 = _S156;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S157 = a0_0;
                                a0_0 = _S155;
                                a1_0 = _S157;
                            }
                            acc_1[9U][0U] = fma(a0_0, b0_0[0U], acc_1[9U][0U]);
                            if(_S91)
                            {
                                acc_1[9U][0U] = fma(a1_0, b1_0[0U], acc_1[9U][0U]);
                            }
                            acc_1[9U][1U] = fma(a0_0, b0_0[1U], acc_1[9U][1U]);
                            if(_S91)
                            {
                                acc_1[9U][1U] = fma(a1_0, b1_0[1U], acc_1[9U][1U]);
                            }
                            acc_1[9U][2U] = fma(a0_0, b0_0[2U], acc_1[9U][2U]);
                            if(_S91)
                            {
                                acc_1[9U][2U] = fma(a1_0, b1_0[2U], acc_1[9U][2U]);
                            }
                            acc_1[9U][3U] = fma(a0_0, b0_0[3U], acc_1[9U][3U]);
                            if(_S91)
                            {
                                acc_1[9U][3U] = fma(a1_0, b1_0[3U], acc_1[9U][3U]);
                            }
                            _S55 = int(3);
                            break;
                        }
                        if(_S55 != int(3))
                        {
                            _S56 = _S55;
                            break;
                        }
                        _S56 = int(2);
                        break;
                    }
                    if(_S56 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(10U >= _S33)
                            {
                                _S57 = int(1);
                                break;
                            }
                            uint e_16 = _S30 + 10U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S158 = (&kernelContext_5)->a_0+(e_16 >> 1U);
                                float _S159 = highHalf_0(*_S158);
                                a0_0 = lowHalf_0(*_S158);
                                a1_0 = _S159;
                            }
                            else
                            {
                                float _S160 = halfA_0(e_16, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S161 = halfA_0(e_16 + 1U, &kernelContext_5);
                                    a0_0 = _S161;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S162 = a0_0;
                                a0_0 = _S160;
                                a1_0 = _S162;
                            }
                            acc_1[10U][0U] = fma(a0_0, b0_0[0U], acc_1[10U][0U]);
                            if(_S91)
                            {
                                acc_1[10U][0U] = fma(a1_0, b1_0[0U], acc_1[10U][0U]);
                            }
                            acc_1[10U][1U] = fma(a0_0, b0_0[1U], acc_1[10U][1U]);
                            if(_S91)
                            {
                                acc_1[10U][1U] = fma(a1_0, b1_0[1U], acc_1[10U][1U]);
                            }
                            acc_1[10U][2U] = fma(a0_0, b0_0[2U], acc_1[10U][2U]);
                            if(_S91)
                            {
                                acc_1[10U][2U] = fma(a1_0, b1_0[2U], acc_1[10U][2U]);
                            }
                            acc_1[10U][3U] = fma(a0_0, b0_0[3U], acc_1[10U][3U]);
                            if(_S91)
                            {
                                acc_1[10U][3U] = fma(a1_0, b1_0[3U], acc_1[10U][3U]);
                            }
                            _S57 = int(3);
                            break;
                        }
                        if(_S57 != int(3))
                        {
                            _S58 = _S57;
                            break;
                        }
                        _S58 = int(2);
                        break;
                    }
                    if(_S58 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(11U >= _S33)
                            {
                                _S59 = int(1);
                                break;
                            }
                            uint e_17 = _S30 + 11U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S163 = (&kernelContext_5)->a_0+(e_17 >> 1U);
                                float _S164 = highHalf_0(*_S163);
                                a0_0 = lowHalf_0(*_S163);
                                a1_0 = _S164;
                            }
                            else
                            {
                                float _S165 = halfA_0(e_17, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S166 = halfA_0(e_17 + 1U, &kernelContext_5);
                                    a0_0 = _S166;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S167 = a0_0;
                                a0_0 = _S165;
                                a1_0 = _S167;
                            }
                            acc_1[11U][0U] = fma(a0_0, b0_0[0U], acc_1[11U][0U]);
                            if(_S91)
                            {
                                acc_1[11U][0U] = fma(a1_0, b1_0[0U], acc_1[11U][0U]);
                            }
                            acc_1[11U][1U] = fma(a0_0, b0_0[1U], acc_1[11U][1U]);
                            if(_S91)
                            {
                                acc_1[11U][1U] = fma(a1_0, b1_0[1U], acc_1[11U][1U]);
                            }
                            acc_1[11U][2U] = fma(a0_0, b0_0[2U], acc_1[11U][2U]);
                            if(_S91)
                            {
                                acc_1[11U][2U] = fma(a1_0, b1_0[2U], acc_1[11U][2U]);
                            }
                            acc_1[11U][3U] = fma(a0_0, b0_0[3U], acc_1[11U][3U]);
                            if(_S91)
                            {
                                acc_1[11U][3U] = fma(a1_0, b1_0[3U], acc_1[11U][3U]);
                            }
                            _S59 = int(3);
                            break;
                        }
                        if(_S59 != int(3))
                        {
                            _S60 = _S59;
                            break;
                        }
                        _S60 = int(2);
                        break;
                    }
                    if(_S60 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(12U >= _S33)
                            {
                                _S61 = int(1);
                                break;
                            }
                            uint e_18 = _S30 + 12U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S168 = (&kernelContext_5)->a_0+(e_18 >> 1U);
                                float _S169 = highHalf_0(*_S168);
                                a0_0 = lowHalf_0(*_S168);
                                a1_0 = _S169;
                            }
                            else
                            {
                                float _S170 = halfA_0(e_18, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S171 = halfA_0(e_18 + 1U, &kernelContext_5);
                                    a0_0 = _S171;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S172 = a0_0;
                                a0_0 = _S170;
                                a1_0 = _S172;
                            }
                            acc_1[12U][0U] = fma(a0_0, b0_0[0U], acc_1[12U][0U]);
                            if(_S91)
                            {
                                acc_1[12U][0U] = fma(a1_0, b1_0[0U], acc_1[12U][0U]);
                            }
                            acc_1[12U][1U] = fma(a0_0, b0_0[1U], acc_1[12U][1U]);
                            if(_S91)
                            {
                                acc_1[12U][1U] = fma(a1_0, b1_0[1U], acc_1[12U][1U]);
                            }
                            acc_1[12U][2U] = fma(a0_0, b0_0[2U], acc_1[12U][2U]);
                            if(_S91)
                            {
                                acc_1[12U][2U] = fma(a1_0, b1_0[2U], acc_1[12U][2U]);
                            }
                            acc_1[12U][3U] = fma(a0_0, b0_0[3U], acc_1[12U][3U]);
                            if(_S91)
                            {
                                acc_1[12U][3U] = fma(a1_0, b1_0[3U], acc_1[12U][3U]);
                            }
                            _S61 = int(3);
                            break;
                        }
                        if(_S61 != int(3))
                        {
                            _S62 = _S61;
                            break;
                        }
                        _S62 = int(2);
                        break;
                    }
                    if(_S62 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(13U >= _S33)
                            {
                                _S63 = int(1);
                                break;
                            }
                            uint e_19 = _S30 + 13U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S173 = (&kernelContext_5)->a_0+(e_19 >> 1U);
                                float _S174 = highHalf_0(*_S173);
                                a0_0 = lowHalf_0(*_S173);
                                a1_0 = _S174;
                            }
                            else
                            {
                                float _S175 = halfA_0(e_19, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S176 = halfA_0(e_19 + 1U, &kernelContext_5);
                                    a0_0 = _S176;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S177 = a0_0;
                                a0_0 = _S175;
                                a1_0 = _S177;
                            }
                            acc_1[13U][0U] = fma(a0_0, b0_0[0U], acc_1[13U][0U]);
                            if(_S91)
                            {
                                acc_1[13U][0U] = fma(a1_0, b1_0[0U], acc_1[13U][0U]);
                            }
                            acc_1[13U][1U] = fma(a0_0, b0_0[1U], acc_1[13U][1U]);
                            if(_S91)
                            {
                                acc_1[13U][1U] = fma(a1_0, b1_0[1U], acc_1[13U][1U]);
                            }
                            acc_1[13U][2U] = fma(a0_0, b0_0[2U], acc_1[13U][2U]);
                            if(_S91)
                            {
                                acc_1[13U][2U] = fma(a1_0, b1_0[2U], acc_1[13U][2U]);
                            }
                            acc_1[13U][3U] = fma(a0_0, b0_0[3U], acc_1[13U][3U]);
                            if(_S91)
                            {
                                acc_1[13U][3U] = fma(a1_0, b1_0[3U], acc_1[13U][3U]);
                            }
                            _S63 = int(3);
                            break;
                        }
                        if(_S63 != int(3))
                        {
                            _S64 = _S63;
                            break;
                        }
                        _S64 = int(2);
                        break;
                    }
                    if(_S64 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(14U >= _S33)
                            {
                                _S65 = int(1);
                                break;
                            }
                            uint e_20 = _S30 + 14U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S178 = (&kernelContext_5)->a_0+(e_20 >> 1U);
                                float _S179 = highHalf_0(*_S178);
                                a0_0 = lowHalf_0(*_S178);
                                a1_0 = _S179;
                            }
                            else
                            {
                                float _S180 = halfA_0(e_20, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S181 = halfA_0(e_20 + 1U, &kernelContext_5);
                                    a0_0 = _S181;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S182 = a0_0;
                                a0_0 = _S180;
                                a1_0 = _S182;
                            }
                            acc_1[14U][0U] = fma(a0_0, b0_0[0U], acc_1[14U][0U]);
                            if(_S91)
                            {
                                acc_1[14U][0U] = fma(a1_0, b1_0[0U], acc_1[14U][0U]);
                            }
                            acc_1[14U][1U] = fma(a0_0, b0_0[1U], acc_1[14U][1U]);
                            if(_S91)
                            {
                                acc_1[14U][1U] = fma(a1_0, b1_0[1U], acc_1[14U][1U]);
                            }
                            acc_1[14U][2U] = fma(a0_0, b0_0[2U], acc_1[14U][2U]);
                            if(_S91)
                            {
                                acc_1[14U][2U] = fma(a1_0, b1_0[2U], acc_1[14U][2U]);
                            }
                            acc_1[14U][3U] = fma(a0_0, b0_0[3U], acc_1[14U][3U]);
                            if(_S91)
                            {
                                acc_1[14U][3U] = fma(a1_0, b1_0[3U], acc_1[14U][3U]);
                            }
                            _S65 = int(3);
                            break;
                        }
                        if(_S65 != int(3))
                        {
                            _S66 = _S65;
                            break;
                        }
                        _S66 = int(2);
                        break;
                    }
                    if(_S66 != int(2))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(15U >= _S33)
                            {
                                _S67 = int(1);
                                break;
                            }
                            uint e_21 = _S30 + 15U * (&kernelContext_5)->p_0->lda_0 + _S89;
                            if(_S88)
                            {
                                _S107 = _S91;
                            }
                            else
                            {
                                _S107 = false;
                            }
                            if(_S107)
                            {
                                uint device* _S183 = (&kernelContext_5)->a_0+(e_21 >> 1U);
                                float _S184 = highHalf_0(*_S183);
                                a0_0 = lowHalf_0(*_S183);
                                a1_0 = _S184;
                            }
                            else
                            {
                                float _S185 = halfA_0(e_21, &kernelContext_5);
                                if(_S91)
                                {
                                    float _S186 = halfA_0(e_21 + 1U, &kernelContext_5);
                                    a0_0 = _S186;
                                }
                                else
                                {
                                    a0_0 = 0.0;
                                }
                                float _S187 = a0_0;
                                a0_0 = _S185;
                                a1_0 = _S187;
                            }
                            acc_1[15U][0U] = fma(a0_0, b0_0[0U], acc_1[15U][0U]);
                            if(_S91)
                            {
                                acc_1[15U][0U] = fma(a1_0, b1_0[0U], acc_1[15U][0U]);
                            }
                            acc_1[15U][1U] = fma(a0_0, b0_0[1U], acc_1[15U][1U]);
                            if(_S91)
                            {
                                acc_1[15U][1U] = fma(a1_0, b1_0[1U], acc_1[15U][1U]);
                            }
                            acc_1[15U][2U] = fma(a0_0, b0_0[2U], acc_1[15U][2U]);
                            if(_S91)
                            {
                                acc_1[15U][2U] = fma(a1_0, b1_0[2U], acc_1[15U][2U]);
                            }
                            acc_1[15U][3U] = fma(a0_0, b0_0[3U], acc_1[15U][3U]);
                            if(_S91)
                            {
                                acc_1[15U][3U] = fma(a1_0, b1_0[3U], acc_1[15U][3U]);
                            }
                            _S67 = int(3);
                            break;
                        }
                        if(_S67 != int(3))
                        {
                            _S68 = _S67;
                            break;
                        }
                        _S68 = int(2);
                        break;
                    }
                    if(_S68 != int(2))
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
    i_0 = 0U;
    for(;;)
    {
        if(i_0 < _S33)
        {
        }
        else
        {
            break;
        }
        (*(&kernelContext_5)->red_0)[0U][lid_0] = acc_1[i_0][0U];
        (*(&kernelContext_5)->red_0)[1U][lid_0] = acc_1[i_0][1U];
        (*(&kernelContext_5)->red_0)[2U][lid_0] = acc_1[i_0][2U];
        (*(&kernelContext_5)->red_0)[3U][lid_0] = acc_1[i_0][3U];
        uint w_12 = _S34 / 2U;
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
                if(0U < _S35)
                {
                    (*(&kernelContext_5)->red_0)[0U][lid_0] = (*(&kernelContext_5)->red_0)[0U][lid_0] + (*(&kernelContext_5)->red_0)[0U][lid_0 + w_12 * ch_0];
                }
                if(1U < _S35)
                {
                    (*(&kernelContext_5)->red_0)[1U][lid_0] = (*(&kernelContext_5)->red_0)[1U][lid_0] + (*(&kernelContext_5)->red_0)[1U][lid_0 + w_12 * ch_0];
                }
                if(2U < _S35)
                {
                    (*(&kernelContext_5)->red_0)[2U][lid_0] = (*(&kernelContext_5)->red_0)[2U][lid_0] + (*(&kernelContext_5)->red_0)[2U][lid_0 + w_12 * ch_0];
                }
                if(3U < _S35)
                {
                    (*(&kernelContext_5)->red_0)[3U][lid_0] = (*(&kernelContext_5)->red_0)[3U][lid_0] + (*(&kernelContext_5)->red_0)[3U][lid_0 + w_12 * ch_0];
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
                if(cc_0 < _S35)
                {
                }
                else
                {
                    break;
                }
                if((col0_0 + cc_0) < ((&kernelContext_5)->p_0->n_0))
                {
                    store_0(_S32 + i_0 * (&kernelContext_5)->p_0->ldc_0 + col0_0 + cc_0, (*(&kernelContext_5)->red_0)[cc_0][lid_0], &kernelContext_5);
                }
                cc_0 = cc_0 + 1U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_0 = i_0 + 1U;
    }
    return;
}

