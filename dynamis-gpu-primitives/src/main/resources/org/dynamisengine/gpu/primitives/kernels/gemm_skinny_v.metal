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
    uint4 device* b_0;
    uint4 device* a_0;
    float device* c_0;
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

void store_0(uint idx_0, float acc_0, KernelContext_0 thread* kernelContext_2)
{
    float v_2;
    if((kernelContext_2->p_0->beta_0) == 0.0)
    {
        v_2 = kernelContext_2->p_0->alpha_0 * acc_0;
    }
    else
    {
        v_2 = kernelContext_2->p_0->alpha_0 * acc_0 + kernelContext_2->p_0->beta_0 * *(kernelContext_2->c_0+idx_0);
    }
    *(kernelContext_2->c_0+idx_0) = v_2;
    return;
}

[[kernel]] void gemm_skinny_v(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(3)]], uint4 device* b_1 [[buffer(1)]], uint4 device* a_1 [[buffer(0)]], float device* c_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->p_0 = p_1;
    (&kernelContext_3)->b_0 = b_1;
    (&kernelContext_3)->a_0 = a_1;
    (&kernelContext_3)->c_0 = c_1;
    threadgroup array<array<float, int(256)>, int(4)> red_1;
    (&kernelContext_3)->red_0 = &red_1;
    uint lid_0 = local_0.x;
    uint _S5 = group_0.z;
    uint _S6 = _S5 * p_1->strideA_0;
    uint _S7 = _S5 * p_1->strideB_0;
    uint _S8 = _S5 * p_1->strideC_0;
    uint _S9 = min(p_1->m_0, 16U);
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
    if(col0_0 < ((&kernelContext_3)->p_0->n_0))
    {
        _S12 = transB_1;
    }
    else
    {
        _S12 = false;
    }
    uint ch_0;
    if(_S12)
    {
        uint _S13 = k_1 / 8U;
        ch_0 = lane_0;
        for(;;)
        {
            if(ch_0 < _S13)
            {
            }
            else
            {
                break;
            }
            uint _S14 = ch_0 * 8U;
            thread array<array<float, int(8)>, int(2)> bv_0;
            load8B_0(_S7 + min(col0_0, (&kernelContext_3)->p_0->n_0 - 1U) * (&kernelContext_3)->p_0->ldb_0 + _S14, &bv_0[0U], &kernelContext_3);
            load8B_0(_S7 + min(col0_0 + 1U, (&kernelContext_3)->p_0->n_0 - 1U) * (&kernelContext_3)->p_0->ldb_0 + _S14, &bv_0[1U], &kernelContext_3);
            for(;;)
            {
                int _S15;
                for(;;)
                {
                    int _S16;
                    for(;;)
                    {
                        if(0U >= _S9)
                        {
                            _S16 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_0;
                        load8A_0(_S6 + _S14, &av_0, &kernelContext_3);
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
                        _S16 = int(3);
                        break;
                    }
                    if(_S16 != int(3))
                    {
                        _S15 = _S16;
                        break;
                    }
                    _S15 = int(2);
                    break;
                }
                if(_S15 != int(2))
                {
                    break;
                }
                int _S17;
                for(;;)
                {
                    int _S18;
                    for(;;)
                    {
                        if(1U >= _S9)
                        {
                            _S18 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_1;
                        load8A_0(_S6 + (&kernelContext_3)->p_0->lda_0 + _S14, &av_1, &kernelContext_3);
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
                        _S18 = int(3);
                        break;
                    }
                    if(_S18 != int(3))
                    {
                        _S17 = _S18;
                        break;
                    }
                    _S17 = int(2);
                    break;
                }
                if(_S17 != int(2))
                {
                    break;
                }
                int _S19;
                for(;;)
                {
                    int _S20;
                    for(;;)
                    {
                        if(2U >= _S9)
                        {
                            _S20 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_2;
                        load8A_0(_S6 + 2U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_2, &kernelContext_3);
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
                        if(3U >= _S9)
                        {
                            _S22 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_3;
                        load8A_0(_S6 + 3U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_3, &kernelContext_3);
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
                        if(4U >= _S9)
                        {
                            _S24 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_4;
                        load8A_0(_S6 + 4U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_4, &kernelContext_3);
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
                        if(5U >= _S9)
                        {
                            _S26 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_5;
                        load8A_0(_S6 + 5U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_5, &kernelContext_3);
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
                        if(6U >= _S9)
                        {
                            _S28 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_6;
                        load8A_0(_S6 + 6U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_6, &kernelContext_3);
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
                        if(7U >= _S9)
                        {
                            _S30 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_7;
                        load8A_0(_S6 + 7U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_7, &kernelContext_3);
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
                        if(8U >= _S9)
                        {
                            _S32 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_8;
                        load8A_0(_S6 + 8U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_8, &kernelContext_3);
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
                        if(9U >= _S9)
                        {
                            _S34 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_9;
                        load8A_0(_S6 + 9U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_9, &kernelContext_3);
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
                        if(10U >= _S9)
                        {
                            _S36 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_10;
                        load8A_0(_S6 + 10U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_10, &kernelContext_3);
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
                        if(11U >= _S9)
                        {
                            _S38 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_11;
                        load8A_0(_S6 + 11U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_11, &kernelContext_3);
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
                        if(12U >= _S9)
                        {
                            _S40 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_12;
                        load8A_0(_S6 + 12U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_12, &kernelContext_3);
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
                        if(13U >= _S9)
                        {
                            _S42 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_13;
                        load8A_0(_S6 + 13U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_13, &kernelContext_3);
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
                        if(14U >= _S9)
                        {
                            _S44 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_14;
                        load8A_0(_S6 + 14U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_14, &kernelContext_3);
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
                        if(15U >= _S9)
                        {
                            _S46 = int(1);
                            break;
                        }
                        thread array<float, int(8)> av_15;
                        load8A_0(_S6 + 15U * (&kernelContext_3)->p_0->lda_0 + _S14, &av_15, &kernelContext_3);
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
    uint i_0 = 0U;
    for(;;)
    {
        if(i_0 < _S9)
        {
        }
        else
        {
            break;
        }
        (*(&kernelContext_3)->red_0)[0U][lid_0] = acc_1[i_0][0U];
        (*(&kernelContext_3)->red_0)[1U][lid_0] = acc_1[i_0][1U];
        (*(&kernelContext_3)->red_0)[2U][lid_0] = acc_1[i_0][2U];
        (*(&kernelContext_3)->red_0)[3U][lid_0] = acc_1[i_0][3U];
        uint w_2 = _S10 / 2U;
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
                if(0U < _S11)
                {
                    (*(&kernelContext_3)->red_0)[0U][lid_0] = (*(&kernelContext_3)->red_0)[0U][lid_0] + (*(&kernelContext_3)->red_0)[0U][lid_0 + w_2 * ch_0];
                }
                if(1U < _S11)
                {
                    (*(&kernelContext_3)->red_0)[1U][lid_0] = (*(&kernelContext_3)->red_0)[1U][lid_0] + (*(&kernelContext_3)->red_0)[1U][lid_0 + w_2 * ch_0];
                }
                if(2U < _S11)
                {
                    (*(&kernelContext_3)->red_0)[2U][lid_0] = (*(&kernelContext_3)->red_0)[2U][lid_0] + (*(&kernelContext_3)->red_0)[2U][lid_0 + w_2 * ch_0];
                }
                if(3U < _S11)
                {
                    (*(&kernelContext_3)->red_0)[3U][lid_0] = (*(&kernelContext_3)->red_0)[3U][lid_0] + (*(&kernelContext_3)->red_0)[3U][lid_0 + w_2 * ch_0];
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
                if(cc_0 < _S11)
                {
                }
                else
                {
                    break;
                }
                if((col0_0 + cc_0) < ((&kernelContext_3)->p_0->n_0))
                {
                    store_0(_S8 + i_0 * (&kernelContext_3)->p_0->ldc_0 + col0_0 + cc_0, (*(&kernelContext_3)->red_0)[cc_0][lid_0], &kernelContext_3);
                }
                cc_0 = cc_0 + 1U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_0 = i_0 + 1U;
    }
    return;
}

