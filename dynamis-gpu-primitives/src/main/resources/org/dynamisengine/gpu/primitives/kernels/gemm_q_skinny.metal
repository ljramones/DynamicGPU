#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant uint fc_QFORMAT_0 [[function_constant(1)]];
constant uint QFORMAT_0 = is_function_constant_defined(fc_QFORMAT_0) ? fc_QFORMAT_0 : 0U;
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
    uint qparams_0;
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
    array<array<float, int(256)>, int(2)> threadgroup* red_0;
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

constant uint fc_QROWS_0 [[function_constant(3)]];
constant uint QROWS_0 = is_function_constant_defined(fc_QROWS_0) ? fc_QROWS_0 : 16U;
constant uint fc_QBLOCK_0 [[function_constant(2)]];
constant uint QBLOCK_0 = is_function_constant_defined(fc_QBLOCK_0) ? fc_QBLOCK_0 : 32U;
uint qParamHalf_0(uint format_0, uint b_1, uint t_0)
{
    uint _S1;
    if(format_0 == 2U)
    {
        _S1 = 2U * b_1 + t_0;
    }
    else
    {
        _S1 = b_1;
    }
    return _S1;
}

uint wordB_0(uint w_0, KernelContext_0 thread* kernelContext_1)
{
    return *(kernelContext_1->b_0+w_0);
}

float qF16ToF32_0(uint h_0)
{
    return (as_type<half>((ushort)((h_0 & 65535U))));
}

float blockParameter_0(uint r_0, uint bl_0, uint t_1, KernelContext_0 thread* kernelContext_2)
{
    uint h_1 = qParamHalf_0(QFORMAT_0, r_0 * (kernelContext_2->p_0->k_1 / QBLOCK_0) + bl_0, t_1);
    uint _S2 = wordB_0(kernelContext_2->p_0->qparams_0 + (h_1 >> 1U), kernelContext_2);
    return qF16ToF32_0(_S2 >> (16U * (h_1 & 1U)));
}

int qCode_0(uint format_1, uint word_0, uint col_0)
{
    int _S3;
    if(format_1 == 0U)
    {
        uint b_2 = (word_0 >> (8U * (col_0 & 3U))) & 255U;
        int _S4 = int(b_2);
        if(b_2 >= 128U)
        {
            _S3 = int(256);
        }
        else
        {
            _S3 = int(0);
        }
        return _S4 - _S3;
    }
    int u_0 = int((word_0 >> (4U * (col_0 & 7U))) & 15U);
    if(format_1 == 1U)
    {
        _S3 = u_0 - int(8);
    }
    else
    {
        _S3 = u_0;
    }
    return _S3;
}

void load8Codes_0(uint r_1, uint c0_0, array<float, int(8)> thread* q_1, KernelContext_0 thread* kernelContext_3)
{
    if(QFORMAT_0 == 0U)
    {
        uint w_1 = (r_1 * kernelContext_3->p_0->k_1 + c0_0) >> 2U;
        uint _S5 = wordB_0(w_1, kernelContext_3);
        uint _S6 = wordB_0(w_1 + 1U, kernelContext_3);
        (*q_1)[0U] = float(qCode_0(QFORMAT_0, _S5, 0U));
        (*q_1)[4U] = float(qCode_0(QFORMAT_0, _S6, 0U));
        (*q_1)[1U] = float(qCode_0(QFORMAT_0, _S5, 1U));
        (*q_1)[5U] = float(qCode_0(QFORMAT_0, _S6, 1U));
        (*q_1)[2U] = float(qCode_0(QFORMAT_0, _S5, 2U));
        (*q_1)[6U] = float(qCode_0(QFORMAT_0, _S6, 2U));
        (*q_1)[3U] = float(qCode_0(QFORMAT_0, _S5, 3U));
        (*q_1)[7U] = float(qCode_0(QFORMAT_0, _S6, 3U));
    }
    else
    {
        uint _S7 = wordB_0((r_1 * kernelContext_3->p_0->k_1 + c0_0) >> 3U, kernelContext_3);
        (*q_1)[0U] = float(qCode_0(QFORMAT_0, _S7, 0U));
        (*q_1)[1U] = float(qCode_0(QFORMAT_0, _S7, 1U));
        (*q_1)[2U] = float(qCode_0(QFORMAT_0, _S7, 2U));
        (*q_1)[3U] = float(qCode_0(QFORMAT_0, _S7, 3U));
        (*q_1)[4U] = float(qCode_0(QFORMAT_0, _S7, 4U));
        (*q_1)[5U] = float(qCode_0(QFORMAT_0, _S7, 5U));
        (*q_1)[6U] = float(qCode_0(QFORMAT_0, _S7, 6U));
        (*q_1)[7U] = float(qCode_0(QFORMAT_0, _S7, 7U));
    }
    return;
}

float aLo_0(uint w_2)
{
    return (as_type<half>((ushort)((w_2 & 65535U))));
}

float aHi_0(uint w_3)
{
    return (as_type<half>((ushort)((w_3 >> 16U))));
}

float elemA_0(uint e_0, KernelContext_0 thread* kernelContext_4)
{
    uint w_4 = *(kernelContext_4->a_0+(e_0 >> 1U));
    float _S8;
    if((e_0 & 1U) != 0U)
    {
        _S8 = aHi_0(w_4);
    }
    else
    {
        _S8 = aLo_0(w_4);
    }
    return _S8;
}

void load8A_0(uint e_1, array<float, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_5)
{
    if((e_1 & 1U) == 0U)
    {
        uint _S9 = e_1 >> 1U;
        uint device* _S10 = kernelContext_5->a_0+_S9;
        uint w_5 = *_S10;
        (*v_0)[0U] = aLo_0(*_S10);
        (*v_0)[1U] = aHi_0(w_5);
        uint device* _S11 = kernelContext_5->a_0+(_S9 + 1U);
        uint w_6 = *_S11;
        (*v_0)[2U] = aLo_0(*_S11);
        (*v_0)[3U] = aHi_0(w_6);
        uint device* _S12 = kernelContext_5->a_0+(_S9 + 2U);
        uint w_7 = *_S12;
        (*v_0)[4U] = aLo_0(*_S12);
        (*v_0)[5U] = aHi_0(w_7);
        uint device* _S13 = kernelContext_5->a_0+(_S9 + 3U);
        uint w_8 = *_S13;
        (*v_0)[6U] = aLo_0(*_S13);
        (*v_0)[7U] = aHi_0(w_8);
    }
    else
    {
        float _S14 = elemA_0(e_1, kernelContext_5);
        (*v_0)[0U] = _S14;
        float _S15 = elemA_0(e_1 + 1U, kernelContext_5);
        (*v_0)[1U] = _S15;
        float _S16 = elemA_0(e_1 + 2U, kernelContext_5);
        (*v_0)[2U] = _S16;
        float _S17 = elemA_0(e_1 + 3U, kernelContext_5);
        (*v_0)[3U] = _S17;
        float _S18 = elemA_0(e_1 + 4U, kernelContext_5);
        (*v_0)[4U] = _S18;
        float _S19 = elemA_0(e_1 + 5U, kernelContext_5);
        (*v_0)[5U] = _S19;
        float _S20 = elemA_0(e_1 + 6U, kernelContext_5);
        (*v_0)[6U] = _S20;
        float _S21 = elemA_0(e_1 + 7U, kernelContext_5);
        (*v_0)[7U] = _S21;
    }
    return;
}

float blockTerm_0(float s_0, float sumA_0, float d_0, float m_2)
{
    float t_2 = d_0 * s_0;
    float t_3;
    if(QFORMAT_0 == 2U)
    {
        t_3 = t_2 + m_2 * sumA_0;
    }
    else
    {
        t_3 = t_2;
    }
    return t_3;
}

bool dg_is_nan_0(float x_0)
{
    uint _S22 = (as_type<uint>((x_0)));
    bool _S23;
    if((_S22 & 2139095040U) == 2139095040U)
    {
        _S23 = (_S22 & 8388607U) != 0U;
    }
    else
    {
        _S23 = false;
    }
    return _S23;
}

float dg_max_0(float a_1, float b_3)
{
    if(dg_is_nan_0(a_1))
    {
        return a_1;
    }
    if(dg_is_nan_0(b_3))
    {
        return b_3;
    }
    bool _S24;
    if(a_1 == 0.0)
    {
        _S24 = b_3 == 0.0;
    }
    else
    {
        _S24 = false;
    }
    if(_S24)
    {
        return (as_type<float>(((as_type<uint>((a_1))) & (as_type<uint>((b_3))))));
    }
    float _S25;
    if(a_1 > b_3)
    {
        _S25 = a_1;
    }
    else
    {
        _S25 = b_3;
    }
    return _S25;
}

float gemmEpilogue_0(float r_2, uint i_0, uint j_0, uint idx_0, uint biasBase_0, KernelContext_0 thread* kernelContext_6)
{
    float _S26;
    if((kernelContext_6->p_0->bias_0) == 1U)
    {
        _S26 = r_2 + *(kernelContext_6->epBias_0+(biasBase_0 + j_0));
    }
    else
    {
        if((kernelContext_6->p_0->bias_0) == 2U)
        {
            _S26 = r_2 + *(kernelContext_6->epBias_0+(biasBase_0 + i_0));
        }
        else
        {
            _S26 = r_2;
        }
    }
    if((kernelContext_6->p_0->act_0) == 1U)
    {
        _S26 = dg_max_0(_S26, 0.0);
    }
    else
    {
        if((kernelContext_6->p_0->act_0) == 2U)
        {
            _S26 = 0.5 * _S26 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S26 + 0.04471499845385551 * _S26 * _S26 * _S26))) + 1.0)));
        }
        else
        {
            if((kernelContext_6->p_0->act_0) == 3U)
            {
                _S26 = _S26 / (1.0 + exp(- _S26));
            }
        }
    }
    if((kernelContext_6->p_0->gated_0) != 0U)
    {
        _S26 = _S26 * *(kernelContext_6->epGate_0+idx_0);
    }
    if((kernelContext_6->p_0->residual_0) != 0U)
    {
        _S26 = _S26 + *(kernelContext_6->epResidual_0+idx_0);
    }
    return _S26;
}

void gemmStore_0(const GemmProblem_0 thread* q_2, uint i_1, uint j_1, float acc_0, KernelContext_0 thread* kernelContext_7)
{
    uint idx_1 = q_2->baseC_0 + i_1 * q_2->ldc_0 + j_1;
    float r_3;
    if((kernelContext_7->p_0->beta_0) == 0.0)
    {
        r_3 = kernelContext_7->p_0->alpha_0 * acc_0;
    }
    else
    {
        r_3 = kernelContext_7->p_0->alpha_0 * acc_0 + kernelContext_7->p_0->beta_0 * *(kernelContext_7->c_0+idx_1);
    }
    float device* _S27 = kernelContext_7->c_0+idx_1;
    float _S28 = gemmEpilogue_0(r_3, i_1, j_1, idx_1, q_2->baseBias_0, kernelContext_7);
    *_S27 = _S28;
    return;
}

[[kernel]] void gemm_q_skinny(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], uint device* b_4 [[buffer(1)]], uint device* a_2 [[buffer(0)]], float device* c_1 [[buffer(2)]], float device* epBias_1 [[buffer(3)]], float device* epGate_1 [[buffer(4)]], float device* epResidual_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_8;
    (&kernelContext_8)->p_0 = p_1;
    (&kernelContext_8)->b_0 = b_4;
    (&kernelContext_8)->a_0 = a_2;
    (&kernelContext_8)->c_0 = c_1;
    (&kernelContext_8)->epBias_0 = epBias_1;
    (&kernelContext_8)->epGate_0 = epGate_1;
    (&kernelContext_8)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(256)>, int(2)> red_1;
    (&kernelContext_8)->red_0 = &red_1;
    uint lid_0 = local_0.x;
    GemmProblem_0 _S29 = gemmBatchProblem_0(group_0.z, &kernelContext_8);
    uint _S30 = min((&kernelContext_8)->p_0->m_1, QROWS_0);
    uint lane_0 = lid_0 % 32U;
    uint col0_0 = group_0.x * 16U + lid_0 / 32U * 2U;
    thread array<array<float, int(2)>, int(16)> acc_1;
    acc_1[0U][0U] = 0.0;
    acc_1[0U][1U] = 0.0;
    acc_1[1U][0U] = 0.0;
    acc_1[1U][1U] = 0.0;
    acc_1[2U][0U] = 0.0;
    acc_1[2U][1U] = 0.0;
    acc_1[3U][0U] = 0.0;
    acc_1[3U][1U] = 0.0;
    acc_1[4U][0U] = 0.0;
    acc_1[4U][1U] = 0.0;
    acc_1[5U][0U] = 0.0;
    acc_1[5U][1U] = 0.0;
    acc_1[6U][0U] = 0.0;
    acc_1[6U][1U] = 0.0;
    acc_1[7U][0U] = 0.0;
    acc_1[7U][1U] = 0.0;
    acc_1[8U][0U] = 0.0;
    acc_1[8U][1U] = 0.0;
    acc_1[9U][0U] = 0.0;
    acc_1[9U][1U] = 0.0;
    acc_1[10U][0U] = 0.0;
    acc_1[10U][1U] = 0.0;
    acc_1[11U][0U] = 0.0;
    acc_1[11U][1U] = 0.0;
    acc_1[12U][0U] = 0.0;
    acc_1[12U][1U] = 0.0;
    acc_1[13U][0U] = 0.0;
    acc_1[13U][1U] = 0.0;
    acc_1[14U][0U] = 0.0;
    acc_1[14U][1U] = 0.0;
    acc_1[15U][0U] = 0.0;
    acc_1[15U][1U] = 0.0;
    uint i_2;
    uint w_9;
    if(col0_0 < ((&kernelContext_8)->p_0->n_1))
    {
        uint _S31 = (&kernelContext_8)->p_0->k_1 / QBLOCK_0;
        thread array<uint, int(2)> rows_0;
        rows_0[0U] = min(col0_0, (&kernelContext_8)->p_0->n_1 - 1U);
        rows_0[1U] = min(col0_0 + 1U, (&kernelContext_8)->p_0->n_1 - 1U);
        i_2 = lane_0;
        for(;;)
        {
            if(i_2 < _S31)
            {
            }
            else
            {
                break;
            }
            thread array<float, int(2)> d_1;
            thread array<float, int(2)> mn_0;
            float _S32 = blockParameter_0(rows_0[0U], i_2, 0U, &kernelContext_8);
            d_1[0U] = _S32;
            float _S33;
            if(QFORMAT_0 == 2U)
            {
                float _S34 = blockParameter_0(rows_0[0U], i_2, 1U, &kernelContext_8);
                _S33 = _S34;
            }
            else
            {
                _S33 = 0.0;
            }
            mn_0[0U] = _S33;
            float _S35 = blockParameter_0(rows_0[1U], i_2, 0U, &kernelContext_8);
            d_1[1U] = _S35;
            float _S36;
            if(QFORMAT_0 == 2U)
            {
                float _S37 = blockParameter_0(rows_0[1U], i_2, 1U, &kernelContext_8);
                _S36 = _S37;
            }
            else
            {
                _S36 = 0.0;
            }
            mn_0[1U] = _S36;
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
            int _S69;
            thread array<float, int(16)> sumA_1;
            sumA_1[0U] = 0.0;
            thread array<array<float, int(2)>, int(16)> s_1;
            s_1[0U][0U] = 0.0;
            s_1[0U][1U] = 0.0;
            sumA_1[1U] = 0.0;
            s_1[1U][0U] = 0.0;
            s_1[1U][1U] = 0.0;
            sumA_1[2U] = 0.0;
            s_1[2U][0U] = 0.0;
            s_1[2U][1U] = 0.0;
            sumA_1[3U] = 0.0;
            s_1[3U][0U] = 0.0;
            s_1[3U][1U] = 0.0;
            sumA_1[4U] = 0.0;
            s_1[4U][0U] = 0.0;
            s_1[4U][1U] = 0.0;
            sumA_1[5U] = 0.0;
            s_1[5U][0U] = 0.0;
            s_1[5U][1U] = 0.0;
            sumA_1[6U] = 0.0;
            s_1[6U][0U] = 0.0;
            s_1[6U][1U] = 0.0;
            sumA_1[7U] = 0.0;
            s_1[7U][0U] = 0.0;
            s_1[7U][1U] = 0.0;
            sumA_1[8U] = 0.0;
            s_1[8U][0U] = 0.0;
            s_1[8U][1U] = 0.0;
            sumA_1[9U] = 0.0;
            s_1[9U][0U] = 0.0;
            s_1[9U][1U] = 0.0;
            sumA_1[10U] = 0.0;
            s_1[10U][0U] = 0.0;
            s_1[10U][1U] = 0.0;
            sumA_1[11U] = 0.0;
            s_1[11U][0U] = 0.0;
            s_1[11U][1U] = 0.0;
            sumA_1[12U] = 0.0;
            s_1[12U][0U] = 0.0;
            s_1[12U][1U] = 0.0;
            sumA_1[13U] = 0.0;
            s_1[13U][0U] = 0.0;
            s_1[13U][1U] = 0.0;
            sumA_1[14U] = 0.0;
            s_1[14U][0U] = 0.0;
            s_1[14U][1U] = 0.0;
            sumA_1[15U] = 0.0;
            s_1[15U][0U] = 0.0;
            s_1[15U][1U] = 0.0;
            w_9 = 0U;
            for(;;)
            {
                if(w_9 < QBLOCK_0)
                {
                }
                else
                {
                    break;
                }
                uint _S70 = i_2 * QBLOCK_0 + w_9;
                thread array<array<float, int(8)>, int(2)> q_3;
                load8Codes_0(rows_0[0U], _S70, &q_3[0U], &kernelContext_8);
                load8Codes_0(rows_0[1U], _S70, &q_3[1U], &kernelContext_8);
                for(;;)
                {
                    uint _S71;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(0U >= _S30)
                            {
                                _S69 = int(2);
                                _S71 = 0U;
                                break;
                            }
                            thread array<float, int(8)> av_0;
                            load8A_0(_S29.baseA_0 + _S70, &av_0, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[0U] = sumA_1[0U] + av_0[0U];
                            }
                            s_1[0U][0U] = fma(av_0[0U], q_3[0U][0U], s_1[0U][0U]);
                            s_1[0U][1U] = fma(av_0[0U], q_3[1U][0U], s_1[0U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[0U] = sumA_1[0U] + av_0[1U];
                            }
                            s_1[0U][0U] = fma(av_0[1U], q_3[0U][1U], s_1[0U][0U]);
                            s_1[0U][1U] = fma(av_0[1U], q_3[1U][1U], s_1[0U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[0U] = sumA_1[0U] + av_0[2U];
                            }
                            s_1[0U][0U] = fma(av_0[2U], q_3[0U][2U], s_1[0U][0U]);
                            s_1[0U][1U] = fma(av_0[2U], q_3[1U][2U], s_1[0U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[0U] = sumA_1[0U] + av_0[3U];
                            }
                            s_1[0U][0U] = fma(av_0[3U], q_3[0U][3U], s_1[0U][0U]);
                            s_1[0U][1U] = fma(av_0[3U], q_3[1U][3U], s_1[0U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[0U] = sumA_1[0U] + av_0[4U];
                            }
                            s_1[0U][0U] = fma(av_0[4U], q_3[0U][4U], s_1[0U][0U]);
                            s_1[0U][1U] = fma(av_0[4U], q_3[1U][4U], s_1[0U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[0U] = sumA_1[0U] + av_0[5U];
                            }
                            s_1[0U][0U] = fma(av_0[5U], q_3[0U][5U], s_1[0U][0U]);
                            s_1[0U][1U] = fma(av_0[5U], q_3[1U][5U], s_1[0U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[0U] = sumA_1[0U] + av_0[6U];
                            }
                            s_1[0U][0U] = fma(av_0[6U], q_3[0U][6U], s_1[0U][0U]);
                            s_1[0U][1U] = fma(av_0[6U], q_3[1U][6U], s_1[0U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[0U] = sumA_1[0U] + av_0[7U];
                            }
                            s_1[0U][0U] = fma(av_0[7U], q_3[0U][7U], s_1[0U][0U]);
                            s_1[0U][1U] = fma(av_0[7U], q_3[1U][7U], s_1[0U][1U]);
                            _S69 = int(4);
                            _S71 = _S29.baseA_0;
                            break;
                        }
                        if(_S69 != int(4))
                        {
                            _S68 = _S69;
                            break;
                        }
                        _S68 = int(3);
                        break;
                    }
                    if(_S68 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(1U >= _S30)
                            {
                                _S67 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_1;
                            load8A_0(_S71 + (&kernelContext_8)->p_0->lda_1 + _S70, &av_1, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[1U] = sumA_1[1U] + av_1[0U];
                            }
                            s_1[1U][0U] = fma(av_1[0U], q_3[0U][0U], s_1[1U][0U]);
                            s_1[1U][1U] = fma(av_1[0U], q_3[1U][0U], s_1[1U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[1U] = sumA_1[1U] + av_1[1U];
                            }
                            s_1[1U][0U] = fma(av_1[1U], q_3[0U][1U], s_1[1U][0U]);
                            s_1[1U][1U] = fma(av_1[1U], q_3[1U][1U], s_1[1U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[1U] = sumA_1[1U] + av_1[2U];
                            }
                            s_1[1U][0U] = fma(av_1[2U], q_3[0U][2U], s_1[1U][0U]);
                            s_1[1U][1U] = fma(av_1[2U], q_3[1U][2U], s_1[1U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[1U] = sumA_1[1U] + av_1[3U];
                            }
                            s_1[1U][0U] = fma(av_1[3U], q_3[0U][3U], s_1[1U][0U]);
                            s_1[1U][1U] = fma(av_1[3U], q_3[1U][3U], s_1[1U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[1U] = sumA_1[1U] + av_1[4U];
                            }
                            s_1[1U][0U] = fma(av_1[4U], q_3[0U][4U], s_1[1U][0U]);
                            s_1[1U][1U] = fma(av_1[4U], q_3[1U][4U], s_1[1U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[1U] = sumA_1[1U] + av_1[5U];
                            }
                            s_1[1U][0U] = fma(av_1[5U], q_3[0U][5U], s_1[1U][0U]);
                            s_1[1U][1U] = fma(av_1[5U], q_3[1U][5U], s_1[1U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[1U] = sumA_1[1U] + av_1[6U];
                            }
                            s_1[1U][0U] = fma(av_1[6U], q_3[0U][6U], s_1[1U][0U]);
                            s_1[1U][1U] = fma(av_1[6U], q_3[1U][6U], s_1[1U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[1U] = sumA_1[1U] + av_1[7U];
                            }
                            s_1[1U][0U] = fma(av_1[7U], q_3[0U][7U], s_1[1U][0U]);
                            s_1[1U][1U] = fma(av_1[7U], q_3[1U][7U], s_1[1U][1U]);
                            _S67 = int(4);
                            break;
                        }
                        if(_S67 != int(4))
                        {
                            _S66 = _S67;
                            break;
                        }
                        _S66 = int(3);
                        break;
                    }
                    if(_S66 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(2U >= _S30)
                            {
                                _S65 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_2;
                            load8A_0(_S71 + 2U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_2, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[2U] = sumA_1[2U] + av_2[0U];
                            }
                            s_1[2U][0U] = fma(av_2[0U], q_3[0U][0U], s_1[2U][0U]);
                            s_1[2U][1U] = fma(av_2[0U], q_3[1U][0U], s_1[2U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[2U] = sumA_1[2U] + av_2[1U];
                            }
                            s_1[2U][0U] = fma(av_2[1U], q_3[0U][1U], s_1[2U][0U]);
                            s_1[2U][1U] = fma(av_2[1U], q_3[1U][1U], s_1[2U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[2U] = sumA_1[2U] + av_2[2U];
                            }
                            s_1[2U][0U] = fma(av_2[2U], q_3[0U][2U], s_1[2U][0U]);
                            s_1[2U][1U] = fma(av_2[2U], q_3[1U][2U], s_1[2U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[2U] = sumA_1[2U] + av_2[3U];
                            }
                            s_1[2U][0U] = fma(av_2[3U], q_3[0U][3U], s_1[2U][0U]);
                            s_1[2U][1U] = fma(av_2[3U], q_3[1U][3U], s_1[2U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[2U] = sumA_1[2U] + av_2[4U];
                            }
                            s_1[2U][0U] = fma(av_2[4U], q_3[0U][4U], s_1[2U][0U]);
                            s_1[2U][1U] = fma(av_2[4U], q_3[1U][4U], s_1[2U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[2U] = sumA_1[2U] + av_2[5U];
                            }
                            s_1[2U][0U] = fma(av_2[5U], q_3[0U][5U], s_1[2U][0U]);
                            s_1[2U][1U] = fma(av_2[5U], q_3[1U][5U], s_1[2U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[2U] = sumA_1[2U] + av_2[6U];
                            }
                            s_1[2U][0U] = fma(av_2[6U], q_3[0U][6U], s_1[2U][0U]);
                            s_1[2U][1U] = fma(av_2[6U], q_3[1U][6U], s_1[2U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[2U] = sumA_1[2U] + av_2[7U];
                            }
                            s_1[2U][0U] = fma(av_2[7U], q_3[0U][7U], s_1[2U][0U]);
                            s_1[2U][1U] = fma(av_2[7U], q_3[1U][7U], s_1[2U][1U]);
                            _S65 = int(4);
                            break;
                        }
                        if(_S65 != int(4))
                        {
                            _S64 = _S65;
                            break;
                        }
                        _S64 = int(3);
                        break;
                    }
                    if(_S64 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(3U >= _S30)
                            {
                                _S63 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_3;
                            load8A_0(_S71 + 3U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_3, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[3U] = sumA_1[3U] + av_3[0U];
                            }
                            s_1[3U][0U] = fma(av_3[0U], q_3[0U][0U], s_1[3U][0U]);
                            s_1[3U][1U] = fma(av_3[0U], q_3[1U][0U], s_1[3U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[3U] = sumA_1[3U] + av_3[1U];
                            }
                            s_1[3U][0U] = fma(av_3[1U], q_3[0U][1U], s_1[3U][0U]);
                            s_1[3U][1U] = fma(av_3[1U], q_3[1U][1U], s_1[3U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[3U] = sumA_1[3U] + av_3[2U];
                            }
                            s_1[3U][0U] = fma(av_3[2U], q_3[0U][2U], s_1[3U][0U]);
                            s_1[3U][1U] = fma(av_3[2U], q_3[1U][2U], s_1[3U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[3U] = sumA_1[3U] + av_3[3U];
                            }
                            s_1[3U][0U] = fma(av_3[3U], q_3[0U][3U], s_1[3U][0U]);
                            s_1[3U][1U] = fma(av_3[3U], q_3[1U][3U], s_1[3U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[3U] = sumA_1[3U] + av_3[4U];
                            }
                            s_1[3U][0U] = fma(av_3[4U], q_3[0U][4U], s_1[3U][0U]);
                            s_1[3U][1U] = fma(av_3[4U], q_3[1U][4U], s_1[3U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[3U] = sumA_1[3U] + av_3[5U];
                            }
                            s_1[3U][0U] = fma(av_3[5U], q_3[0U][5U], s_1[3U][0U]);
                            s_1[3U][1U] = fma(av_3[5U], q_3[1U][5U], s_1[3U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[3U] = sumA_1[3U] + av_3[6U];
                            }
                            s_1[3U][0U] = fma(av_3[6U], q_3[0U][6U], s_1[3U][0U]);
                            s_1[3U][1U] = fma(av_3[6U], q_3[1U][6U], s_1[3U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[3U] = sumA_1[3U] + av_3[7U];
                            }
                            s_1[3U][0U] = fma(av_3[7U], q_3[0U][7U], s_1[3U][0U]);
                            s_1[3U][1U] = fma(av_3[7U], q_3[1U][7U], s_1[3U][1U]);
                            _S63 = int(4);
                            break;
                        }
                        if(_S63 != int(4))
                        {
                            _S62 = _S63;
                            break;
                        }
                        _S62 = int(3);
                        break;
                    }
                    if(_S62 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(4U >= _S30)
                            {
                                _S61 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_4;
                            load8A_0(_S71 + 4U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_4, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[4U] = sumA_1[4U] + av_4[0U];
                            }
                            s_1[4U][0U] = fma(av_4[0U], q_3[0U][0U], s_1[4U][0U]);
                            s_1[4U][1U] = fma(av_4[0U], q_3[1U][0U], s_1[4U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[4U] = sumA_1[4U] + av_4[1U];
                            }
                            s_1[4U][0U] = fma(av_4[1U], q_3[0U][1U], s_1[4U][0U]);
                            s_1[4U][1U] = fma(av_4[1U], q_3[1U][1U], s_1[4U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[4U] = sumA_1[4U] + av_4[2U];
                            }
                            s_1[4U][0U] = fma(av_4[2U], q_3[0U][2U], s_1[4U][0U]);
                            s_1[4U][1U] = fma(av_4[2U], q_3[1U][2U], s_1[4U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[4U] = sumA_1[4U] + av_4[3U];
                            }
                            s_1[4U][0U] = fma(av_4[3U], q_3[0U][3U], s_1[4U][0U]);
                            s_1[4U][1U] = fma(av_4[3U], q_3[1U][3U], s_1[4U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[4U] = sumA_1[4U] + av_4[4U];
                            }
                            s_1[4U][0U] = fma(av_4[4U], q_3[0U][4U], s_1[4U][0U]);
                            s_1[4U][1U] = fma(av_4[4U], q_3[1U][4U], s_1[4U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[4U] = sumA_1[4U] + av_4[5U];
                            }
                            s_1[4U][0U] = fma(av_4[5U], q_3[0U][5U], s_1[4U][0U]);
                            s_1[4U][1U] = fma(av_4[5U], q_3[1U][5U], s_1[4U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[4U] = sumA_1[4U] + av_4[6U];
                            }
                            s_1[4U][0U] = fma(av_4[6U], q_3[0U][6U], s_1[4U][0U]);
                            s_1[4U][1U] = fma(av_4[6U], q_3[1U][6U], s_1[4U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[4U] = sumA_1[4U] + av_4[7U];
                            }
                            s_1[4U][0U] = fma(av_4[7U], q_3[0U][7U], s_1[4U][0U]);
                            s_1[4U][1U] = fma(av_4[7U], q_3[1U][7U], s_1[4U][1U]);
                            _S61 = int(4);
                            break;
                        }
                        if(_S61 != int(4))
                        {
                            _S60 = _S61;
                            break;
                        }
                        _S60 = int(3);
                        break;
                    }
                    if(_S60 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(5U >= _S30)
                            {
                                _S59 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_5;
                            load8A_0(_S71 + 5U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_5, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[5U] = sumA_1[5U] + av_5[0U];
                            }
                            s_1[5U][0U] = fma(av_5[0U], q_3[0U][0U], s_1[5U][0U]);
                            s_1[5U][1U] = fma(av_5[0U], q_3[1U][0U], s_1[5U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[5U] = sumA_1[5U] + av_5[1U];
                            }
                            s_1[5U][0U] = fma(av_5[1U], q_3[0U][1U], s_1[5U][0U]);
                            s_1[5U][1U] = fma(av_5[1U], q_3[1U][1U], s_1[5U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[5U] = sumA_1[5U] + av_5[2U];
                            }
                            s_1[5U][0U] = fma(av_5[2U], q_3[0U][2U], s_1[5U][0U]);
                            s_1[5U][1U] = fma(av_5[2U], q_3[1U][2U], s_1[5U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[5U] = sumA_1[5U] + av_5[3U];
                            }
                            s_1[5U][0U] = fma(av_5[3U], q_3[0U][3U], s_1[5U][0U]);
                            s_1[5U][1U] = fma(av_5[3U], q_3[1U][3U], s_1[5U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[5U] = sumA_1[5U] + av_5[4U];
                            }
                            s_1[5U][0U] = fma(av_5[4U], q_3[0U][4U], s_1[5U][0U]);
                            s_1[5U][1U] = fma(av_5[4U], q_3[1U][4U], s_1[5U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[5U] = sumA_1[5U] + av_5[5U];
                            }
                            s_1[5U][0U] = fma(av_5[5U], q_3[0U][5U], s_1[5U][0U]);
                            s_1[5U][1U] = fma(av_5[5U], q_3[1U][5U], s_1[5U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[5U] = sumA_1[5U] + av_5[6U];
                            }
                            s_1[5U][0U] = fma(av_5[6U], q_3[0U][6U], s_1[5U][0U]);
                            s_1[5U][1U] = fma(av_5[6U], q_3[1U][6U], s_1[5U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[5U] = sumA_1[5U] + av_5[7U];
                            }
                            s_1[5U][0U] = fma(av_5[7U], q_3[0U][7U], s_1[5U][0U]);
                            s_1[5U][1U] = fma(av_5[7U], q_3[1U][7U], s_1[5U][1U]);
                            _S59 = int(4);
                            break;
                        }
                        if(_S59 != int(4))
                        {
                            _S58 = _S59;
                            break;
                        }
                        _S58 = int(3);
                        break;
                    }
                    if(_S58 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(6U >= _S30)
                            {
                                _S57 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_6;
                            load8A_0(_S71 + 6U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_6, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[6U] = sumA_1[6U] + av_6[0U];
                            }
                            s_1[6U][0U] = fma(av_6[0U], q_3[0U][0U], s_1[6U][0U]);
                            s_1[6U][1U] = fma(av_6[0U], q_3[1U][0U], s_1[6U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[6U] = sumA_1[6U] + av_6[1U];
                            }
                            s_1[6U][0U] = fma(av_6[1U], q_3[0U][1U], s_1[6U][0U]);
                            s_1[6U][1U] = fma(av_6[1U], q_3[1U][1U], s_1[6U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[6U] = sumA_1[6U] + av_6[2U];
                            }
                            s_1[6U][0U] = fma(av_6[2U], q_3[0U][2U], s_1[6U][0U]);
                            s_1[6U][1U] = fma(av_6[2U], q_3[1U][2U], s_1[6U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[6U] = sumA_1[6U] + av_6[3U];
                            }
                            s_1[6U][0U] = fma(av_6[3U], q_3[0U][3U], s_1[6U][0U]);
                            s_1[6U][1U] = fma(av_6[3U], q_3[1U][3U], s_1[6U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[6U] = sumA_1[6U] + av_6[4U];
                            }
                            s_1[6U][0U] = fma(av_6[4U], q_3[0U][4U], s_1[6U][0U]);
                            s_1[6U][1U] = fma(av_6[4U], q_3[1U][4U], s_1[6U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[6U] = sumA_1[6U] + av_6[5U];
                            }
                            s_1[6U][0U] = fma(av_6[5U], q_3[0U][5U], s_1[6U][0U]);
                            s_1[6U][1U] = fma(av_6[5U], q_3[1U][5U], s_1[6U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[6U] = sumA_1[6U] + av_6[6U];
                            }
                            s_1[6U][0U] = fma(av_6[6U], q_3[0U][6U], s_1[6U][0U]);
                            s_1[6U][1U] = fma(av_6[6U], q_3[1U][6U], s_1[6U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[6U] = sumA_1[6U] + av_6[7U];
                            }
                            s_1[6U][0U] = fma(av_6[7U], q_3[0U][7U], s_1[6U][0U]);
                            s_1[6U][1U] = fma(av_6[7U], q_3[1U][7U], s_1[6U][1U]);
                            _S57 = int(4);
                            break;
                        }
                        if(_S57 != int(4))
                        {
                            _S56 = _S57;
                            break;
                        }
                        _S56 = int(3);
                        break;
                    }
                    if(_S56 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(7U >= _S30)
                            {
                                _S55 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_7;
                            load8A_0(_S71 + 7U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_7, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[7U] = sumA_1[7U] + av_7[0U];
                            }
                            s_1[7U][0U] = fma(av_7[0U], q_3[0U][0U], s_1[7U][0U]);
                            s_1[7U][1U] = fma(av_7[0U], q_3[1U][0U], s_1[7U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[7U] = sumA_1[7U] + av_7[1U];
                            }
                            s_1[7U][0U] = fma(av_7[1U], q_3[0U][1U], s_1[7U][0U]);
                            s_1[7U][1U] = fma(av_7[1U], q_3[1U][1U], s_1[7U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[7U] = sumA_1[7U] + av_7[2U];
                            }
                            s_1[7U][0U] = fma(av_7[2U], q_3[0U][2U], s_1[7U][0U]);
                            s_1[7U][1U] = fma(av_7[2U], q_3[1U][2U], s_1[7U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[7U] = sumA_1[7U] + av_7[3U];
                            }
                            s_1[7U][0U] = fma(av_7[3U], q_3[0U][3U], s_1[7U][0U]);
                            s_1[7U][1U] = fma(av_7[3U], q_3[1U][3U], s_1[7U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[7U] = sumA_1[7U] + av_7[4U];
                            }
                            s_1[7U][0U] = fma(av_7[4U], q_3[0U][4U], s_1[7U][0U]);
                            s_1[7U][1U] = fma(av_7[4U], q_3[1U][4U], s_1[7U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[7U] = sumA_1[7U] + av_7[5U];
                            }
                            s_1[7U][0U] = fma(av_7[5U], q_3[0U][5U], s_1[7U][0U]);
                            s_1[7U][1U] = fma(av_7[5U], q_3[1U][5U], s_1[7U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[7U] = sumA_1[7U] + av_7[6U];
                            }
                            s_1[7U][0U] = fma(av_7[6U], q_3[0U][6U], s_1[7U][0U]);
                            s_1[7U][1U] = fma(av_7[6U], q_3[1U][6U], s_1[7U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[7U] = sumA_1[7U] + av_7[7U];
                            }
                            s_1[7U][0U] = fma(av_7[7U], q_3[0U][7U], s_1[7U][0U]);
                            s_1[7U][1U] = fma(av_7[7U], q_3[1U][7U], s_1[7U][1U]);
                            _S55 = int(4);
                            break;
                        }
                        if(_S55 != int(4))
                        {
                            _S54 = _S55;
                            break;
                        }
                        _S54 = int(3);
                        break;
                    }
                    if(_S54 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(8U >= _S30)
                            {
                                _S53 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_8;
                            load8A_0(_S71 + 8U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_8, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[8U] = sumA_1[8U] + av_8[0U];
                            }
                            s_1[8U][0U] = fma(av_8[0U], q_3[0U][0U], s_1[8U][0U]);
                            s_1[8U][1U] = fma(av_8[0U], q_3[1U][0U], s_1[8U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[8U] = sumA_1[8U] + av_8[1U];
                            }
                            s_1[8U][0U] = fma(av_8[1U], q_3[0U][1U], s_1[8U][0U]);
                            s_1[8U][1U] = fma(av_8[1U], q_3[1U][1U], s_1[8U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[8U] = sumA_1[8U] + av_8[2U];
                            }
                            s_1[8U][0U] = fma(av_8[2U], q_3[0U][2U], s_1[8U][0U]);
                            s_1[8U][1U] = fma(av_8[2U], q_3[1U][2U], s_1[8U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[8U] = sumA_1[8U] + av_8[3U];
                            }
                            s_1[8U][0U] = fma(av_8[3U], q_3[0U][3U], s_1[8U][0U]);
                            s_1[8U][1U] = fma(av_8[3U], q_3[1U][3U], s_1[8U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[8U] = sumA_1[8U] + av_8[4U];
                            }
                            s_1[8U][0U] = fma(av_8[4U], q_3[0U][4U], s_1[8U][0U]);
                            s_1[8U][1U] = fma(av_8[4U], q_3[1U][4U], s_1[8U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[8U] = sumA_1[8U] + av_8[5U];
                            }
                            s_1[8U][0U] = fma(av_8[5U], q_3[0U][5U], s_1[8U][0U]);
                            s_1[8U][1U] = fma(av_8[5U], q_3[1U][5U], s_1[8U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[8U] = sumA_1[8U] + av_8[6U];
                            }
                            s_1[8U][0U] = fma(av_8[6U], q_3[0U][6U], s_1[8U][0U]);
                            s_1[8U][1U] = fma(av_8[6U], q_3[1U][6U], s_1[8U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[8U] = sumA_1[8U] + av_8[7U];
                            }
                            s_1[8U][0U] = fma(av_8[7U], q_3[0U][7U], s_1[8U][0U]);
                            s_1[8U][1U] = fma(av_8[7U], q_3[1U][7U], s_1[8U][1U]);
                            _S53 = int(4);
                            break;
                        }
                        if(_S53 != int(4))
                        {
                            _S52 = _S53;
                            break;
                        }
                        _S52 = int(3);
                        break;
                    }
                    if(_S52 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(9U >= _S30)
                            {
                                _S51 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_9;
                            load8A_0(_S71 + 9U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_9, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[9U] = sumA_1[9U] + av_9[0U];
                            }
                            s_1[9U][0U] = fma(av_9[0U], q_3[0U][0U], s_1[9U][0U]);
                            s_1[9U][1U] = fma(av_9[0U], q_3[1U][0U], s_1[9U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[9U] = sumA_1[9U] + av_9[1U];
                            }
                            s_1[9U][0U] = fma(av_9[1U], q_3[0U][1U], s_1[9U][0U]);
                            s_1[9U][1U] = fma(av_9[1U], q_3[1U][1U], s_1[9U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[9U] = sumA_1[9U] + av_9[2U];
                            }
                            s_1[9U][0U] = fma(av_9[2U], q_3[0U][2U], s_1[9U][0U]);
                            s_1[9U][1U] = fma(av_9[2U], q_3[1U][2U], s_1[9U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[9U] = sumA_1[9U] + av_9[3U];
                            }
                            s_1[9U][0U] = fma(av_9[3U], q_3[0U][3U], s_1[9U][0U]);
                            s_1[9U][1U] = fma(av_9[3U], q_3[1U][3U], s_1[9U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[9U] = sumA_1[9U] + av_9[4U];
                            }
                            s_1[9U][0U] = fma(av_9[4U], q_3[0U][4U], s_1[9U][0U]);
                            s_1[9U][1U] = fma(av_9[4U], q_3[1U][4U], s_1[9U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[9U] = sumA_1[9U] + av_9[5U];
                            }
                            s_1[9U][0U] = fma(av_9[5U], q_3[0U][5U], s_1[9U][0U]);
                            s_1[9U][1U] = fma(av_9[5U], q_3[1U][5U], s_1[9U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[9U] = sumA_1[9U] + av_9[6U];
                            }
                            s_1[9U][0U] = fma(av_9[6U], q_3[0U][6U], s_1[9U][0U]);
                            s_1[9U][1U] = fma(av_9[6U], q_3[1U][6U], s_1[9U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[9U] = sumA_1[9U] + av_9[7U];
                            }
                            s_1[9U][0U] = fma(av_9[7U], q_3[0U][7U], s_1[9U][0U]);
                            s_1[9U][1U] = fma(av_9[7U], q_3[1U][7U], s_1[9U][1U]);
                            _S51 = int(4);
                            break;
                        }
                        if(_S51 != int(4))
                        {
                            _S50 = _S51;
                            break;
                        }
                        _S50 = int(3);
                        break;
                    }
                    if(_S50 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(10U >= _S30)
                            {
                                _S49 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_10;
                            load8A_0(_S71 + 10U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_10, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[10U] = sumA_1[10U] + av_10[0U];
                            }
                            s_1[10U][0U] = fma(av_10[0U], q_3[0U][0U], s_1[10U][0U]);
                            s_1[10U][1U] = fma(av_10[0U], q_3[1U][0U], s_1[10U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[10U] = sumA_1[10U] + av_10[1U];
                            }
                            s_1[10U][0U] = fma(av_10[1U], q_3[0U][1U], s_1[10U][0U]);
                            s_1[10U][1U] = fma(av_10[1U], q_3[1U][1U], s_1[10U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[10U] = sumA_1[10U] + av_10[2U];
                            }
                            s_1[10U][0U] = fma(av_10[2U], q_3[0U][2U], s_1[10U][0U]);
                            s_1[10U][1U] = fma(av_10[2U], q_3[1U][2U], s_1[10U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[10U] = sumA_1[10U] + av_10[3U];
                            }
                            s_1[10U][0U] = fma(av_10[3U], q_3[0U][3U], s_1[10U][0U]);
                            s_1[10U][1U] = fma(av_10[3U], q_3[1U][3U], s_1[10U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[10U] = sumA_1[10U] + av_10[4U];
                            }
                            s_1[10U][0U] = fma(av_10[4U], q_3[0U][4U], s_1[10U][0U]);
                            s_1[10U][1U] = fma(av_10[4U], q_3[1U][4U], s_1[10U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[10U] = sumA_1[10U] + av_10[5U];
                            }
                            s_1[10U][0U] = fma(av_10[5U], q_3[0U][5U], s_1[10U][0U]);
                            s_1[10U][1U] = fma(av_10[5U], q_3[1U][5U], s_1[10U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[10U] = sumA_1[10U] + av_10[6U];
                            }
                            s_1[10U][0U] = fma(av_10[6U], q_3[0U][6U], s_1[10U][0U]);
                            s_1[10U][1U] = fma(av_10[6U], q_3[1U][6U], s_1[10U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[10U] = sumA_1[10U] + av_10[7U];
                            }
                            s_1[10U][0U] = fma(av_10[7U], q_3[0U][7U], s_1[10U][0U]);
                            s_1[10U][1U] = fma(av_10[7U], q_3[1U][7U], s_1[10U][1U]);
                            _S49 = int(4);
                            break;
                        }
                        if(_S49 != int(4))
                        {
                            _S48 = _S49;
                            break;
                        }
                        _S48 = int(3);
                        break;
                    }
                    if(_S48 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(11U >= _S30)
                            {
                                _S47 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_11;
                            load8A_0(_S71 + 11U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_11, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[11U] = sumA_1[11U] + av_11[0U];
                            }
                            s_1[11U][0U] = fma(av_11[0U], q_3[0U][0U], s_1[11U][0U]);
                            s_1[11U][1U] = fma(av_11[0U], q_3[1U][0U], s_1[11U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[11U] = sumA_1[11U] + av_11[1U];
                            }
                            s_1[11U][0U] = fma(av_11[1U], q_3[0U][1U], s_1[11U][0U]);
                            s_1[11U][1U] = fma(av_11[1U], q_3[1U][1U], s_1[11U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[11U] = sumA_1[11U] + av_11[2U];
                            }
                            s_1[11U][0U] = fma(av_11[2U], q_3[0U][2U], s_1[11U][0U]);
                            s_1[11U][1U] = fma(av_11[2U], q_3[1U][2U], s_1[11U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[11U] = sumA_1[11U] + av_11[3U];
                            }
                            s_1[11U][0U] = fma(av_11[3U], q_3[0U][3U], s_1[11U][0U]);
                            s_1[11U][1U] = fma(av_11[3U], q_3[1U][3U], s_1[11U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[11U] = sumA_1[11U] + av_11[4U];
                            }
                            s_1[11U][0U] = fma(av_11[4U], q_3[0U][4U], s_1[11U][0U]);
                            s_1[11U][1U] = fma(av_11[4U], q_3[1U][4U], s_1[11U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[11U] = sumA_1[11U] + av_11[5U];
                            }
                            s_1[11U][0U] = fma(av_11[5U], q_3[0U][5U], s_1[11U][0U]);
                            s_1[11U][1U] = fma(av_11[5U], q_3[1U][5U], s_1[11U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[11U] = sumA_1[11U] + av_11[6U];
                            }
                            s_1[11U][0U] = fma(av_11[6U], q_3[0U][6U], s_1[11U][0U]);
                            s_1[11U][1U] = fma(av_11[6U], q_3[1U][6U], s_1[11U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[11U] = sumA_1[11U] + av_11[7U];
                            }
                            s_1[11U][0U] = fma(av_11[7U], q_3[0U][7U], s_1[11U][0U]);
                            s_1[11U][1U] = fma(av_11[7U], q_3[1U][7U], s_1[11U][1U]);
                            _S47 = int(4);
                            break;
                        }
                        if(_S47 != int(4))
                        {
                            _S46 = _S47;
                            break;
                        }
                        _S46 = int(3);
                        break;
                    }
                    if(_S46 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(12U >= _S30)
                            {
                                _S45 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_12;
                            load8A_0(_S71 + 12U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_12, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[12U] = sumA_1[12U] + av_12[0U];
                            }
                            s_1[12U][0U] = fma(av_12[0U], q_3[0U][0U], s_1[12U][0U]);
                            s_1[12U][1U] = fma(av_12[0U], q_3[1U][0U], s_1[12U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[12U] = sumA_1[12U] + av_12[1U];
                            }
                            s_1[12U][0U] = fma(av_12[1U], q_3[0U][1U], s_1[12U][0U]);
                            s_1[12U][1U] = fma(av_12[1U], q_3[1U][1U], s_1[12U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[12U] = sumA_1[12U] + av_12[2U];
                            }
                            s_1[12U][0U] = fma(av_12[2U], q_3[0U][2U], s_1[12U][0U]);
                            s_1[12U][1U] = fma(av_12[2U], q_3[1U][2U], s_1[12U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[12U] = sumA_1[12U] + av_12[3U];
                            }
                            s_1[12U][0U] = fma(av_12[3U], q_3[0U][3U], s_1[12U][0U]);
                            s_1[12U][1U] = fma(av_12[3U], q_3[1U][3U], s_1[12U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[12U] = sumA_1[12U] + av_12[4U];
                            }
                            s_1[12U][0U] = fma(av_12[4U], q_3[0U][4U], s_1[12U][0U]);
                            s_1[12U][1U] = fma(av_12[4U], q_3[1U][4U], s_1[12U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[12U] = sumA_1[12U] + av_12[5U];
                            }
                            s_1[12U][0U] = fma(av_12[5U], q_3[0U][5U], s_1[12U][0U]);
                            s_1[12U][1U] = fma(av_12[5U], q_3[1U][5U], s_1[12U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[12U] = sumA_1[12U] + av_12[6U];
                            }
                            s_1[12U][0U] = fma(av_12[6U], q_3[0U][6U], s_1[12U][0U]);
                            s_1[12U][1U] = fma(av_12[6U], q_3[1U][6U], s_1[12U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[12U] = sumA_1[12U] + av_12[7U];
                            }
                            s_1[12U][0U] = fma(av_12[7U], q_3[0U][7U], s_1[12U][0U]);
                            s_1[12U][1U] = fma(av_12[7U], q_3[1U][7U], s_1[12U][1U]);
                            _S45 = int(4);
                            break;
                        }
                        if(_S45 != int(4))
                        {
                            _S44 = _S45;
                            break;
                        }
                        _S44 = int(3);
                        break;
                    }
                    if(_S44 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(13U >= _S30)
                            {
                                _S43 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_13;
                            load8A_0(_S71 + 13U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_13, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[13U] = sumA_1[13U] + av_13[0U];
                            }
                            s_1[13U][0U] = fma(av_13[0U], q_3[0U][0U], s_1[13U][0U]);
                            s_1[13U][1U] = fma(av_13[0U], q_3[1U][0U], s_1[13U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[13U] = sumA_1[13U] + av_13[1U];
                            }
                            s_1[13U][0U] = fma(av_13[1U], q_3[0U][1U], s_1[13U][0U]);
                            s_1[13U][1U] = fma(av_13[1U], q_3[1U][1U], s_1[13U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[13U] = sumA_1[13U] + av_13[2U];
                            }
                            s_1[13U][0U] = fma(av_13[2U], q_3[0U][2U], s_1[13U][0U]);
                            s_1[13U][1U] = fma(av_13[2U], q_3[1U][2U], s_1[13U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[13U] = sumA_1[13U] + av_13[3U];
                            }
                            s_1[13U][0U] = fma(av_13[3U], q_3[0U][3U], s_1[13U][0U]);
                            s_1[13U][1U] = fma(av_13[3U], q_3[1U][3U], s_1[13U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[13U] = sumA_1[13U] + av_13[4U];
                            }
                            s_1[13U][0U] = fma(av_13[4U], q_3[0U][4U], s_1[13U][0U]);
                            s_1[13U][1U] = fma(av_13[4U], q_3[1U][4U], s_1[13U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[13U] = sumA_1[13U] + av_13[5U];
                            }
                            s_1[13U][0U] = fma(av_13[5U], q_3[0U][5U], s_1[13U][0U]);
                            s_1[13U][1U] = fma(av_13[5U], q_3[1U][5U], s_1[13U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[13U] = sumA_1[13U] + av_13[6U];
                            }
                            s_1[13U][0U] = fma(av_13[6U], q_3[0U][6U], s_1[13U][0U]);
                            s_1[13U][1U] = fma(av_13[6U], q_3[1U][6U], s_1[13U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[13U] = sumA_1[13U] + av_13[7U];
                            }
                            s_1[13U][0U] = fma(av_13[7U], q_3[0U][7U], s_1[13U][0U]);
                            s_1[13U][1U] = fma(av_13[7U], q_3[1U][7U], s_1[13U][1U]);
                            _S43 = int(4);
                            break;
                        }
                        if(_S43 != int(4))
                        {
                            _S42 = _S43;
                            break;
                        }
                        _S42 = int(3);
                        break;
                    }
                    if(_S42 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(14U >= _S30)
                            {
                                _S41 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_14;
                            load8A_0(_S71 + 14U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_14, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[14U] = sumA_1[14U] + av_14[0U];
                            }
                            s_1[14U][0U] = fma(av_14[0U], q_3[0U][0U], s_1[14U][0U]);
                            s_1[14U][1U] = fma(av_14[0U], q_3[1U][0U], s_1[14U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[14U] = sumA_1[14U] + av_14[1U];
                            }
                            s_1[14U][0U] = fma(av_14[1U], q_3[0U][1U], s_1[14U][0U]);
                            s_1[14U][1U] = fma(av_14[1U], q_3[1U][1U], s_1[14U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[14U] = sumA_1[14U] + av_14[2U];
                            }
                            s_1[14U][0U] = fma(av_14[2U], q_3[0U][2U], s_1[14U][0U]);
                            s_1[14U][1U] = fma(av_14[2U], q_3[1U][2U], s_1[14U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[14U] = sumA_1[14U] + av_14[3U];
                            }
                            s_1[14U][0U] = fma(av_14[3U], q_3[0U][3U], s_1[14U][0U]);
                            s_1[14U][1U] = fma(av_14[3U], q_3[1U][3U], s_1[14U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[14U] = sumA_1[14U] + av_14[4U];
                            }
                            s_1[14U][0U] = fma(av_14[4U], q_3[0U][4U], s_1[14U][0U]);
                            s_1[14U][1U] = fma(av_14[4U], q_3[1U][4U], s_1[14U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[14U] = sumA_1[14U] + av_14[5U];
                            }
                            s_1[14U][0U] = fma(av_14[5U], q_3[0U][5U], s_1[14U][0U]);
                            s_1[14U][1U] = fma(av_14[5U], q_3[1U][5U], s_1[14U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[14U] = sumA_1[14U] + av_14[6U];
                            }
                            s_1[14U][0U] = fma(av_14[6U], q_3[0U][6U], s_1[14U][0U]);
                            s_1[14U][1U] = fma(av_14[6U], q_3[1U][6U], s_1[14U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[14U] = sumA_1[14U] + av_14[7U];
                            }
                            s_1[14U][0U] = fma(av_14[7U], q_3[0U][7U], s_1[14U][0U]);
                            s_1[14U][1U] = fma(av_14[7U], q_3[1U][7U], s_1[14U][1U]);
                            _S41 = int(4);
                            break;
                        }
                        if(_S41 != int(4))
                        {
                            _S40 = _S41;
                            break;
                        }
                        _S40 = int(3);
                        break;
                    }
                    if(_S40 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(15U >= _S30)
                            {
                                _S39 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_15;
                            load8A_0(_S71 + 15U * (&kernelContext_8)->p_0->lda_1 + _S70, &av_15, &kernelContext_8);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[15U] = sumA_1[15U] + av_15[0U];
                            }
                            s_1[15U][0U] = fma(av_15[0U], q_3[0U][0U], s_1[15U][0U]);
                            s_1[15U][1U] = fma(av_15[0U], q_3[1U][0U], s_1[15U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[15U] = sumA_1[15U] + av_15[1U];
                            }
                            s_1[15U][0U] = fma(av_15[1U], q_3[0U][1U], s_1[15U][0U]);
                            s_1[15U][1U] = fma(av_15[1U], q_3[1U][1U], s_1[15U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[15U] = sumA_1[15U] + av_15[2U];
                            }
                            s_1[15U][0U] = fma(av_15[2U], q_3[0U][2U], s_1[15U][0U]);
                            s_1[15U][1U] = fma(av_15[2U], q_3[1U][2U], s_1[15U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[15U] = sumA_1[15U] + av_15[3U];
                            }
                            s_1[15U][0U] = fma(av_15[3U], q_3[0U][3U], s_1[15U][0U]);
                            s_1[15U][1U] = fma(av_15[3U], q_3[1U][3U], s_1[15U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[15U] = sumA_1[15U] + av_15[4U];
                            }
                            s_1[15U][0U] = fma(av_15[4U], q_3[0U][4U], s_1[15U][0U]);
                            s_1[15U][1U] = fma(av_15[4U], q_3[1U][4U], s_1[15U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[15U] = sumA_1[15U] + av_15[5U];
                            }
                            s_1[15U][0U] = fma(av_15[5U], q_3[0U][5U], s_1[15U][0U]);
                            s_1[15U][1U] = fma(av_15[5U], q_3[1U][5U], s_1[15U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[15U] = sumA_1[15U] + av_15[6U];
                            }
                            s_1[15U][0U] = fma(av_15[6U], q_3[0U][6U], s_1[15U][0U]);
                            s_1[15U][1U] = fma(av_15[6U], q_3[1U][6U], s_1[15U][1U]);
                            if(QFORMAT_0 == 2U)
                            {
                                sumA_1[15U] = sumA_1[15U] + av_15[7U];
                            }
                            s_1[15U][0U] = fma(av_15[7U], q_3[0U][7U], s_1[15U][0U]);
                            s_1[15U][1U] = fma(av_15[7U], q_3[1U][7U], s_1[15U][1U]);
                            _S39 = int(4);
                            break;
                        }
                        if(_S39 != int(4))
                        {
                            _S38 = _S39;
                            break;
                        }
                        _S38 = int(3);
                        break;
                    }
                    if(_S38 != int(3))
                    {
                        break;
                    }
                    break;
                }
                w_9 = w_9 + 8U;
            }
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        if(0U >= _S30)
                        {
                            _S69 = int(1);
                            break;
                        }
                        acc_1[0U][0U] = acc_1[0U][0U] + blockTerm_0(s_1[0U][0U], sumA_1[0U], d_1[0U], mn_0[0U]);
                        acc_1[0U][1U] = acc_1[0U][1U] + blockTerm_0(s_1[0U][1U], sumA_1[0U], d_1[1U], mn_0[1U]);
                        _S69 = int(3);
                        break;
                    }
                    if(_S69 != int(3))
                    {
                        _S68 = _S69;
                        break;
                    }
                    _S68 = int(2);
                    break;
                }
                if(_S68 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(1U >= _S30)
                        {
                            _S67 = int(1);
                            break;
                        }
                        acc_1[1U][0U] = acc_1[1U][0U] + blockTerm_0(s_1[1U][0U], sumA_1[1U], d_1[0U], mn_0[0U]);
                        acc_1[1U][1U] = acc_1[1U][1U] + blockTerm_0(s_1[1U][1U], sumA_1[1U], d_1[1U], mn_0[1U]);
                        _S67 = int(3);
                        break;
                    }
                    if(_S67 != int(3))
                    {
                        _S66 = _S67;
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
                        if(2U >= _S30)
                        {
                            _S65 = int(1);
                            break;
                        }
                        acc_1[2U][0U] = acc_1[2U][0U] + blockTerm_0(s_1[2U][0U], sumA_1[2U], d_1[0U], mn_0[0U]);
                        acc_1[2U][1U] = acc_1[2U][1U] + blockTerm_0(s_1[2U][1U], sumA_1[2U], d_1[1U], mn_0[1U]);
                        _S65 = int(3);
                        break;
                    }
                    if(_S65 != int(3))
                    {
                        _S64 = _S65;
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
                        if(3U >= _S30)
                        {
                            _S63 = int(1);
                            break;
                        }
                        acc_1[3U][0U] = acc_1[3U][0U] + blockTerm_0(s_1[3U][0U], sumA_1[3U], d_1[0U], mn_0[0U]);
                        acc_1[3U][1U] = acc_1[3U][1U] + blockTerm_0(s_1[3U][1U], sumA_1[3U], d_1[1U], mn_0[1U]);
                        _S63 = int(3);
                        break;
                    }
                    if(_S63 != int(3))
                    {
                        _S62 = _S63;
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
                        if(4U >= _S30)
                        {
                            _S61 = int(1);
                            break;
                        }
                        acc_1[4U][0U] = acc_1[4U][0U] + blockTerm_0(s_1[4U][0U], sumA_1[4U], d_1[0U], mn_0[0U]);
                        acc_1[4U][1U] = acc_1[4U][1U] + blockTerm_0(s_1[4U][1U], sumA_1[4U], d_1[1U], mn_0[1U]);
                        _S61 = int(3);
                        break;
                    }
                    if(_S61 != int(3))
                    {
                        _S60 = _S61;
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
                        if(5U >= _S30)
                        {
                            _S59 = int(1);
                            break;
                        }
                        acc_1[5U][0U] = acc_1[5U][0U] + blockTerm_0(s_1[5U][0U], sumA_1[5U], d_1[0U], mn_0[0U]);
                        acc_1[5U][1U] = acc_1[5U][1U] + blockTerm_0(s_1[5U][1U], sumA_1[5U], d_1[1U], mn_0[1U]);
                        _S59 = int(3);
                        break;
                    }
                    if(_S59 != int(3))
                    {
                        _S58 = _S59;
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
                        if(6U >= _S30)
                        {
                            _S57 = int(1);
                            break;
                        }
                        acc_1[6U][0U] = acc_1[6U][0U] + blockTerm_0(s_1[6U][0U], sumA_1[6U], d_1[0U], mn_0[0U]);
                        acc_1[6U][1U] = acc_1[6U][1U] + blockTerm_0(s_1[6U][1U], sumA_1[6U], d_1[1U], mn_0[1U]);
                        _S57 = int(3);
                        break;
                    }
                    if(_S57 != int(3))
                    {
                        _S56 = _S57;
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
                        if(7U >= _S30)
                        {
                            _S55 = int(1);
                            break;
                        }
                        acc_1[7U][0U] = acc_1[7U][0U] + blockTerm_0(s_1[7U][0U], sumA_1[7U], d_1[0U], mn_0[0U]);
                        acc_1[7U][1U] = acc_1[7U][1U] + blockTerm_0(s_1[7U][1U], sumA_1[7U], d_1[1U], mn_0[1U]);
                        _S55 = int(3);
                        break;
                    }
                    if(_S55 != int(3))
                    {
                        _S54 = _S55;
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
                        if(8U >= _S30)
                        {
                            _S53 = int(1);
                            break;
                        }
                        acc_1[8U][0U] = acc_1[8U][0U] + blockTerm_0(s_1[8U][0U], sumA_1[8U], d_1[0U], mn_0[0U]);
                        acc_1[8U][1U] = acc_1[8U][1U] + blockTerm_0(s_1[8U][1U], sumA_1[8U], d_1[1U], mn_0[1U]);
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
                for(;;)
                {
                    for(;;)
                    {
                        if(9U >= _S30)
                        {
                            _S51 = int(1);
                            break;
                        }
                        acc_1[9U][0U] = acc_1[9U][0U] + blockTerm_0(s_1[9U][0U], sumA_1[9U], d_1[0U], mn_0[0U]);
                        acc_1[9U][1U] = acc_1[9U][1U] + blockTerm_0(s_1[9U][1U], sumA_1[9U], d_1[1U], mn_0[1U]);
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
                for(;;)
                {
                    for(;;)
                    {
                        if(10U >= _S30)
                        {
                            _S49 = int(1);
                            break;
                        }
                        acc_1[10U][0U] = acc_1[10U][0U] + blockTerm_0(s_1[10U][0U], sumA_1[10U], d_1[0U], mn_0[0U]);
                        acc_1[10U][1U] = acc_1[10U][1U] + blockTerm_0(s_1[10U][1U], sumA_1[10U], d_1[1U], mn_0[1U]);
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
                for(;;)
                {
                    for(;;)
                    {
                        if(11U >= _S30)
                        {
                            _S47 = int(1);
                            break;
                        }
                        acc_1[11U][0U] = acc_1[11U][0U] + blockTerm_0(s_1[11U][0U], sumA_1[11U], d_1[0U], mn_0[0U]);
                        acc_1[11U][1U] = acc_1[11U][1U] + blockTerm_0(s_1[11U][1U], sumA_1[11U], d_1[1U], mn_0[1U]);
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
                for(;;)
                {
                    for(;;)
                    {
                        if(12U >= _S30)
                        {
                            _S45 = int(1);
                            break;
                        }
                        acc_1[12U][0U] = acc_1[12U][0U] + blockTerm_0(s_1[12U][0U], sumA_1[12U], d_1[0U], mn_0[0U]);
                        acc_1[12U][1U] = acc_1[12U][1U] + blockTerm_0(s_1[12U][1U], sumA_1[12U], d_1[1U], mn_0[1U]);
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
                for(;;)
                {
                    for(;;)
                    {
                        if(13U >= _S30)
                        {
                            _S43 = int(1);
                            break;
                        }
                        acc_1[13U][0U] = acc_1[13U][0U] + blockTerm_0(s_1[13U][0U], sumA_1[13U], d_1[0U], mn_0[0U]);
                        acc_1[13U][1U] = acc_1[13U][1U] + blockTerm_0(s_1[13U][1U], sumA_1[13U], d_1[1U], mn_0[1U]);
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
                for(;;)
                {
                    for(;;)
                    {
                        if(14U >= _S30)
                        {
                            _S41 = int(1);
                            break;
                        }
                        acc_1[14U][0U] = acc_1[14U][0U] + blockTerm_0(s_1[14U][0U], sumA_1[14U], d_1[0U], mn_0[0U]);
                        acc_1[14U][1U] = acc_1[14U][1U] + blockTerm_0(s_1[14U][1U], sumA_1[14U], d_1[1U], mn_0[1U]);
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
                for(;;)
                {
                    for(;;)
                    {
                        if(15U >= _S30)
                        {
                            _S39 = int(1);
                            break;
                        }
                        acc_1[15U][0U] = acc_1[15U][0U] + blockTerm_0(s_1[15U][0U], sumA_1[15U], d_1[0U], mn_0[0U]);
                        acc_1[15U][1U] = acc_1[15U][1U] + blockTerm_0(s_1[15U][1U], sumA_1[15U], d_1[1U], mn_0[1U]);
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
                break;
            }
            i_2 = i_2 + 32U;
        }
    }
    i_2 = 0U;
    for(;;)
    {
        if(i_2 < _S30)
        {
        }
        else
        {
            break;
        }
        (*(&kernelContext_8)->red_0)[0U][lid_0] = acc_1[i_2][0U];
        (*(&kernelContext_8)->red_0)[1U][lid_0] = acc_1[i_2][1U];
        w_9 = 16U;
        for(;;)
        {
            if(w_9 > 0U)
            {
            }
            else
            {
                break;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane_0 < w_9)
            {
                uint _S72 = lid_0 + w_9;
                (*(&kernelContext_8)->red_0)[0U][lid_0] = (*(&kernelContext_8)->red_0)[0U][lid_0] + (*(&kernelContext_8)->red_0)[0U][_S72];
                (*(&kernelContext_8)->red_0)[1U][lid_0] = (*(&kernelContext_8)->red_0)[1U][lid_0] + (*(&kernelContext_8)->red_0)[1U][_S72];
            }
            w_9 = w_9 / 2U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane_0 == 0U)
        {
            if(col0_0 < ((&kernelContext_8)->p_0->n_1))
            {
                float _S73 = (*(&kernelContext_8)->red_0)[0U][lid_0];
                thread GemmProblem_0 _S74 = _S29;
                gemmStore_0(&_S74, i_2, col0_0, _S73, &kernelContext_8);
            }
            uint _S75 = col0_0 + 1U;
            if(_S75 < ((&kernelContext_8)->p_0->n_1))
            {
                float _S76 = (*(&kernelContext_8)->red_0)[1U][lid_0];
                thread GemmProblem_0 _S77 = _S29;
                gemmStore_0(&_S77, i_2, _S75, _S76, &kernelContext_8);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_2 = i_2 + 1U;
    }
    return;
}

