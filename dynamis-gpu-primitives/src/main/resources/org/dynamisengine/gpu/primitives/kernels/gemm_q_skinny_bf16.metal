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
    uint qhigh_0;
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
uint qWord_0(uint w_0, KernelContext_0 thread* kernelContext_1)
{
    return *(kernelContext_1->b_0+w_0);
}

float qF16ToF32_0(uint h_0)
{
    return (as_type<half>((ushort)((h_0 & 65535U))));
}

float2 kScale6Pair_0(uint qparams_1, uint r_0, uint col_0, uint cols_0, KernelContext_0 thread* kernelContext_2)
{
    uint base_0 = qparams_1 + (r_0 * (cols_0 / 256U) + col_0 / 256U) * 5U;
    uint _S1 = qWord_0(base_0, kernelContext_2);
    float d_0 = qF16ToF32_0(_S1);
    uint j_0 = col_0 % 256U / 16U;
    uint _S2 = qWord_0(base_0 + 1U + j_0 / 4U, kernelContext_2);
    uint w_1 = _S2 >> (8U * (j_0 % 4U));
    int s0_0 = int(w_1 & 255U);
    int s1_0 = int((w_1 >> 8U) & 255U);
    int s0_1;
    if(s0_0 >= int(128))
    {
        s0_1 = s0_0 - int(256);
    }
    else
    {
        s0_1 = s0_0;
    }
    int s1_1;
    if(s1_0 >= int(128))
    {
        s1_1 = s1_0 - int(256);
    }
    else
    {
        s1_1 = s1_0;
    }
    return float2(d_0 * float(s0_1), d_0 * float(s1_1));
}

uint qByte_0(uint at_0, KernelContext_0 thread* kernelContext_3)
{
    uint _S3 = qWord_0(at_0 >> 2U, kernelContext_3);
    return (_S3 >> (8U * (at_0 & 3U))) & 255U;
}

float2 kScale_0(uint format_0, uint qparams_2, uint r_1, uint col_1, uint cols_1, KernelContext_0 thread* kernelContext_4)
{
    uint sb_0 = r_1 * (cols_1 / 256U) + col_1 / 256U;
    bool _S4 = format_0 == 5U;
    uint sc_0;
    if(_S4)
    {
        sc_0 = 5U;
    }
    else
    {
        sc_0 = 4U;
    }
    uint base_1 = qparams_2 + sb_0 * sc_0;
    uint _S5 = qWord_0(base_1, kernelContext_4);
    float d_1 = qF16ToF32_0(_S5);
    uint at_1 = 4U * (base_1 + 1U);
    if(_S4)
    {
        uint _S6 = qByte_0(at_1 + col_1 % 256U / 16U, kernelContext_4);
        int sc_1 = int(_S6);
        int sc_2;
        if(sc_1 >= int(128))
        {
            sc_2 = sc_1 - int(256);
        }
        else
        {
            sc_2 = sc_1;
        }
        return float2(d_1 * float(sc_2), 0.0);
    }
    uint j_1 = col_1 % 256U / 32U;
    uint _S7 = at_1 + j_1;
    uint _S8 = qByte_0(_S7, kernelContext_4);
    uint _S9 = qByte_0(_S7 + 4U, kernelContext_4);
    uint _S10 = qByte_0(at_1 + (j_1 ^ 4U), kernelContext_4);
    bool low_0 = j_1 < 4U;
    if(low_0)
    {
        sc_0 = _S8 & 63U;
    }
    else
    {
        sc_0 = (_S9 & 15U) | ((_S10 >> 6U) << 4U);
    }
    uint mn_0;
    if(low_0)
    {
        mn_0 = _S9 & 63U;
    }
    else
    {
        mn_0 = (_S9 >> 4U) | ((_S8 >> 6U) << 4U);
    }
    return float2(d_1 * float(sc_0), qF16ToF32_0(_S5 >> 16U) * float(mn_0));
}

uint qParamHalf_0(uint format_1, uint b_1, uint t_0)
{
    uint _S11;
    if(format_1 == 2U)
    {
        _S11 = 2U * b_1 + t_0;
    }
    else
    {
        _S11 = b_1;
    }
    return _S11;
}

uint wordB_0(uint w_2, KernelContext_0 thread* kernelContext_5)
{
    return *(kernelContext_5->b_0+w_2);
}

float blockParameter_0(uint r_2, uint bl_0, uint t_1, KernelContext_0 thread* kernelContext_6)
{
    uint h_1 = qParamHalf_0(QFORMAT_0, r_2 * (kernelContext_6->p_0->k_1 / QBLOCK_0) + bl_0, t_1);
    uint _S12 = wordB_0(kernelContext_6->p_0->qparams_0 + (h_1 >> 1U), kernelContext_6);
    return qF16ToF32_0(_S12 >> (16U * (h_1 & 1U)));
}

float2 unitScale_0(uint r_3, uint u_0, KernelContext_0 thread* kernelContext_7)
{
    uint _S13;
    if(QFORMAT_0 == 5U)
    {
        uint _S14 = kernelContext_7->p_0->qparams_0;
        if(QFORMAT_0 >= 3U)
        {
            _S13 = 32U;
        }
        else
        {
            _S13 = QBLOCK_0;
        }
        float2 _S15 = kScale6Pair_0(_S14, r_3, u_0 * _S13, kernelContext_7->p_0->k_1, kernelContext_7);
        return _S15;
    }
    if(QFORMAT_0 >= 3U)
    {
        uint _S16 = kernelContext_7->p_0->qparams_0;
        if(QFORMAT_0 >= 3U)
        {
            _S13 = 32U;
        }
        else
        {
            _S13 = QBLOCK_0;
        }
        float2 _S17 = kScale_0(QFORMAT_0, _S16, r_3, u_0 * _S13, kernelContext_7->p_0->k_1, kernelContext_7);
        return _S17;
    }
    float _S18 = blockParameter_0(r_3, u_0, 0U, kernelContext_7);
    float _S19;
    if(QFORMAT_0 == 2U)
    {
        float _S20 = blockParameter_0(r_3, u_0, 1U, kernelContext_7);
        _S19 = _S20;
    }
    else
    {
        _S19 = 0.0;
    }
    return float2(_S18, _S19);
}

uint kHigh8_0(uint format_2, uint qhigh_1, uint e_0, KernelContext_0 thread* kernelContext_8)
{
    if(format_2 == 4U)
    {
        uint _S21 = qWord_0(qhigh_1 + (e_0 >> 5U), kernelContext_8);
        return _S21 >> (e_0 & 31U);
    }
    uint _S22 = qWord_0(qhigh_1 + (e_0 >> 4U), kernelContext_8);
    return _S22 >> (2U * (e_0 & 15U));
}

int kCode_0(uint format_3, uint u_1, uint h_2)
{
    if(format_3 == 3U)
    {
        return int(u_1);
    }
    if(format_3 == 4U)
    {
        return int(u_1 | ((h_2 & 1U) << 4U));
    }
    return int(u_1 | ((h_2 & 3U) << 4U)) - int(32);
}

int qCode_0(uint format_4, uint word_0, uint col_2)
{
    int _S23;
    if(format_4 == 0U)
    {
        uint b_2 = (word_0 >> (8U * (col_2 & 3U))) & 255U;
        int _S24 = int(b_2);
        if(b_2 >= 128U)
        {
            _S23 = int(256);
        }
        else
        {
            _S23 = int(0);
        }
        return _S24 - _S23;
    }
    int u_2 = int((word_0 >> (4U * (col_2 & 7U))) & 15U);
    if(format_4 == 1U)
    {
        _S23 = u_2 - int(8);
    }
    else
    {
        _S23 = u_2;
    }
    return _S23;
}

void load8Codes_0(uint r_4, uint c0_0, array<float, int(8)> thread* q_1, KernelContext_0 thread* kernelContext_9)
{
    if(QFORMAT_0 >= 3U)
    {
        uint _S25 = wordB_0((r_4 * kernelContext_9->p_0->k_1 + c0_0) >> 3U, kernelContext_9);
        uint _S26;
        if(QFORMAT_0 == 3U)
        {
            _S26 = 0U;
        }
        else
        {
            uint _S27 = kHigh8_0(QFORMAT_0, kernelContext_9->p_0->qhigh_0, r_4 * kernelContext_9->p_0->k_1 + c0_0, kernelContext_9);
            _S26 = _S27;
        }
        uint _S28;
        if(QFORMAT_0 == 4U)
        {
            _S28 = 1U;
        }
        else
        {
            _S28 = 2U;
        }
        (*q_1)[0U] = float(kCode_0(QFORMAT_0, (_S25 >> 0U) & 15U, _S26 >> 0U));
        (*q_1)[1U] = float(kCode_0(QFORMAT_0, (_S25 >> 4U) & 15U, _S26 >> _S28));
        (*q_1)[2U] = float(kCode_0(QFORMAT_0, (_S25 >> 8U) & 15U, _S26 >> (_S28 * 2U)));
        (*q_1)[3U] = float(kCode_0(QFORMAT_0, (_S25 >> 12U) & 15U, _S26 >> (_S28 * 3U)));
        (*q_1)[4U] = float(kCode_0(QFORMAT_0, (_S25 >> 16U) & 15U, _S26 >> (_S28 * 4U)));
        (*q_1)[5U] = float(kCode_0(QFORMAT_0, (_S25 >> 20U) & 15U, _S26 >> (_S28 * 5U)));
        (*q_1)[6U] = float(kCode_0(QFORMAT_0, (_S25 >> 24U) & 15U, _S26 >> (_S28 * 6U)));
        (*q_1)[7U] = float(kCode_0(QFORMAT_0, (_S25 >> 28U) & 15U, _S26 >> (_S28 * 7U)));
    }
    else
    {
        if(QFORMAT_0 == 0U)
        {
            uint w_3 = (r_4 * kernelContext_9->p_0->k_1 + c0_0) >> 2U;
            uint _S29 = wordB_0(w_3, kernelContext_9);
            uint _S30 = wordB_0(w_3 + 1U, kernelContext_9);
            (*q_1)[0U] = float(qCode_0(QFORMAT_0, _S29, 0U));
            (*q_1)[4U] = float(qCode_0(QFORMAT_0, _S30, 0U));
            (*q_1)[1U] = float(qCode_0(QFORMAT_0, _S29, 1U));
            (*q_1)[5U] = float(qCode_0(QFORMAT_0, _S30, 1U));
            (*q_1)[2U] = float(qCode_0(QFORMAT_0, _S29, 2U));
            (*q_1)[6U] = float(qCode_0(QFORMAT_0, _S30, 2U));
            (*q_1)[3U] = float(qCode_0(QFORMAT_0, _S29, 3U));
            (*q_1)[7U] = float(qCode_0(QFORMAT_0, _S30, 3U));
        }
        else
        {
            uint _S31 = wordB_0((r_4 * kernelContext_9->p_0->k_1 + c0_0) >> 3U, kernelContext_9);
            (*q_1)[0U] = float(qCode_0(QFORMAT_0, _S31, 0U));
            (*q_1)[1U] = float(qCode_0(QFORMAT_0, _S31, 1U));
            (*q_1)[2U] = float(qCode_0(QFORMAT_0, _S31, 2U));
            (*q_1)[3U] = float(qCode_0(QFORMAT_0, _S31, 3U));
            (*q_1)[4U] = float(qCode_0(QFORMAT_0, _S31, 4U));
            (*q_1)[5U] = float(qCode_0(QFORMAT_0, _S31, 5U));
            (*q_1)[6U] = float(qCode_0(QFORMAT_0, _S31, 6U));
            (*q_1)[7U] = float(qCode_0(QFORMAT_0, _S31, 7U));
        }
    }
    return;
}

float aLo_0(uint w_4)
{
    return (as_type<float>((w_4 << 16U)));
}

float aHi_0(uint w_5)
{
    return (as_type<float>((w_5 & 4294901760U)));
}

float elemA_0(uint e_1, KernelContext_0 thread* kernelContext_10)
{
    uint w_6 = *(kernelContext_10->a_0+(e_1 >> 1U));
    float _S32;
    if((e_1 & 1U) != 0U)
    {
        _S32 = aHi_0(w_6);
    }
    else
    {
        _S32 = aLo_0(w_6);
    }
    return _S32;
}

void load8A_0(uint e_2, array<float, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_11)
{
    if((e_2 & 1U) == 0U)
    {
        uint _S33 = e_2 >> 1U;
        uint device* _S34 = kernelContext_11->a_0+_S33;
        uint w_7 = *_S34;
        (*v_0)[0U] = aLo_0(*_S34);
        (*v_0)[1U] = aHi_0(w_7);
        uint device* _S35 = kernelContext_11->a_0+(_S33 + 1U);
        uint w_8 = *_S35;
        (*v_0)[2U] = aLo_0(*_S35);
        (*v_0)[3U] = aHi_0(w_8);
        uint device* _S36 = kernelContext_11->a_0+(_S33 + 2U);
        uint w_9 = *_S36;
        (*v_0)[4U] = aLo_0(*_S36);
        (*v_0)[5U] = aHi_0(w_9);
        uint device* _S37 = kernelContext_11->a_0+(_S33 + 3U);
        uint w_10 = *_S37;
        (*v_0)[6U] = aLo_0(*_S37);
        (*v_0)[7U] = aHi_0(w_10);
    }
    else
    {
        float _S38 = elemA_0(e_2, kernelContext_11);
        (*v_0)[0U] = _S38;
        float _S39 = elemA_0(e_2 + 1U, kernelContext_11);
        (*v_0)[1U] = _S39;
        float _S40 = elemA_0(e_2 + 2U, kernelContext_11);
        (*v_0)[2U] = _S40;
        float _S41 = elemA_0(e_2 + 3U, kernelContext_11);
        (*v_0)[3U] = _S41;
        float _S42 = elemA_0(e_2 + 4U, kernelContext_11);
        (*v_0)[4U] = _S42;
        float _S43 = elemA_0(e_2 + 5U, kernelContext_11);
        (*v_0)[5U] = _S43;
        float _S44 = elemA_0(e_2 + 6U, kernelContext_11);
        (*v_0)[6U] = _S44;
        float _S45 = elemA_0(e_2 + 7U, kernelContext_11);
        (*v_0)[7U] = _S45;
    }
    return;
}

float blockTerm_0(float s_0, float sumA_0, float d_2, float m_2)
{
    float t_2 = d_2 * s_0;
    float t_3;
    if(QFORMAT_0 == 2U)
    {
        t_3 = t_2 + m_2 * sumA_0;
    }
    else
    {
        bool _S46;
        if(QFORMAT_0 == 3U)
        {
            _S46 = true;
        }
        else
        {
            _S46 = QFORMAT_0 == 4U;
        }
        if(_S46)
        {
            t_3 = t_2 - m_2 * sumA_0;
        }
        else
        {
            t_3 = t_2;
        }
    }
    return t_3;
}

float addTerms_0(float acc_0, float s_1, float s2_0, float sumA_1, float2 sc_3)
{
    if(QFORMAT_0 == 5U)
    {
        return acc_0 + blockTerm_0(s_1, 0.0, sc_3.x, 0.0) + blockTerm_0(s2_0, 0.0, sc_3.y, 0.0);
    }
    return acc_0 + blockTerm_0(s_1, sumA_1, sc_3.x, sc_3.y);
}

bool dg_is_nan_0(float x_0)
{
    uint _S47 = (as_type<uint>((x_0)));
    bool _S48;
    if((_S47 & 2139095040U) == 2139095040U)
    {
        _S48 = (_S47 & 8388607U) != 0U;
    }
    else
    {
        _S48 = false;
    }
    return _S48;
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
    bool _S49;
    if(a_1 == 0.0)
    {
        _S49 = b_3 == 0.0;
    }
    else
    {
        _S49 = false;
    }
    if(_S49)
    {
        return (as_type<float>(((as_type<uint>((a_1))) & (as_type<uint>((b_3))))));
    }
    float _S50;
    if(a_1 > b_3)
    {
        _S50 = a_1;
    }
    else
    {
        _S50 = b_3;
    }
    return _S50;
}

float gemmEpilogue_0(float r_5, uint i_0, uint j_2, uint idx_0, uint biasBase_0, KernelContext_0 thread* kernelContext_12)
{
    float _S51;
    if((kernelContext_12->p_0->bias_0) == 1U)
    {
        _S51 = r_5 + *(kernelContext_12->epBias_0+(biasBase_0 + j_2));
    }
    else
    {
        if((kernelContext_12->p_0->bias_0) == 2U)
        {
            _S51 = r_5 + *(kernelContext_12->epBias_0+(biasBase_0 + i_0));
        }
        else
        {
            _S51 = r_5;
        }
    }
    if((kernelContext_12->p_0->act_0) == 1U)
    {
        _S51 = dg_max_0(_S51, 0.0);
    }
    else
    {
        if((kernelContext_12->p_0->act_0) == 2U)
        {
            _S51 = 0.5 * _S51 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S51 + 0.04471499845385551 * _S51 * _S51 * _S51))) + 1.0)));
        }
        else
        {
            if((kernelContext_12->p_0->act_0) == 3U)
            {
                _S51 = _S51 / (1.0 + exp(- _S51));
            }
        }
    }
    if((kernelContext_12->p_0->gated_0) != 0U)
    {
        _S51 = _S51 * *(kernelContext_12->epGate_0+idx_0);
    }
    if((kernelContext_12->p_0->residual_0) != 0U)
    {
        _S51 = _S51 + *(kernelContext_12->epResidual_0+idx_0);
    }
    return _S51;
}

void gemmStore_0(const GemmProblem_0 thread* q_2, uint i_1, uint j_3, float acc_1, KernelContext_0 thread* kernelContext_13)
{
    uint idx_1 = q_2->baseC_0 + i_1 * q_2->ldc_0 + j_3;
    float r_6;
    if((kernelContext_13->p_0->beta_0) == 0.0)
    {
        r_6 = kernelContext_13->p_0->alpha_0 * acc_1;
    }
    else
    {
        r_6 = kernelContext_13->p_0->alpha_0 * acc_1 + kernelContext_13->p_0->beta_0 * *(kernelContext_13->c_0+idx_1);
    }
    float device* _S52 = kernelContext_13->c_0+idx_1;
    float _S53 = gemmEpilogue_0(r_6, i_1, j_3, idx_1, q_2->baseBias_0, kernelContext_13);
    *_S52 = _S53;
    return;
}

[[kernel]] void gemm_q_skinny_bf16(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], uint device* b_4 [[buffer(1)]], uint device* a_2 [[buffer(0)]], float device* c_1 [[buffer(2)]], float device* epBias_1 [[buffer(3)]], float device* epGate_1 [[buffer(4)]], float device* epResidual_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_14;
    (&kernelContext_14)->p_0 = p_1;
    (&kernelContext_14)->b_0 = b_4;
    (&kernelContext_14)->a_0 = a_2;
    (&kernelContext_14)->c_0 = c_1;
    (&kernelContext_14)->epBias_0 = epBias_1;
    (&kernelContext_14)->epGate_0 = epGate_1;
    (&kernelContext_14)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(256)>, int(2)> red_1;
    (&kernelContext_14)->red_0 = &red_1;
    uint lid_0 = local_0.x;
    GemmProblem_0 _S54 = gemmBatchProblem_0(group_0.z, &kernelContext_14);
    uint _S55 = min((&kernelContext_14)->p_0->m_1, QROWS_0);
    uint lane_0 = lid_0 % 32U;
    uint col0_0 = group_0.x * 16U + lid_0 / 32U * 2U;
    thread array<array<float, int(2)>, int(16)> acc_2;
    acc_2[0U][0U] = 0.0;
    acc_2[0U][1U] = 0.0;
    acc_2[1U][0U] = 0.0;
    acc_2[1U][1U] = 0.0;
    acc_2[2U][0U] = 0.0;
    acc_2[2U][1U] = 0.0;
    acc_2[3U][0U] = 0.0;
    acc_2[3U][1U] = 0.0;
    acc_2[4U][0U] = 0.0;
    acc_2[4U][1U] = 0.0;
    acc_2[5U][0U] = 0.0;
    acc_2[5U][1U] = 0.0;
    acc_2[6U][0U] = 0.0;
    acc_2[6U][1U] = 0.0;
    acc_2[7U][0U] = 0.0;
    acc_2[7U][1U] = 0.0;
    acc_2[8U][0U] = 0.0;
    acc_2[8U][1U] = 0.0;
    acc_2[9U][0U] = 0.0;
    acc_2[9U][1U] = 0.0;
    acc_2[10U][0U] = 0.0;
    acc_2[10U][1U] = 0.0;
    acc_2[11U][0U] = 0.0;
    acc_2[11U][1U] = 0.0;
    acc_2[12U][0U] = 0.0;
    acc_2[12U][1U] = 0.0;
    acc_2[13U][0U] = 0.0;
    acc_2[13U][1U] = 0.0;
    acc_2[14U][0U] = 0.0;
    acc_2[14U][1U] = 0.0;
    acc_2[15U][0U] = 0.0;
    acc_2[15U][1U] = 0.0;
    uint i_2;
    uint w_11;
    if(col0_0 < ((&kernelContext_14)->p_0->n_1))
    {
        uint _S56 = (&kernelContext_14)->p_0->k_1;
        bool _S57 = QFORMAT_0 >= 3U;
        if(_S57)
        {
            i_2 = 32U;
        }
        else
        {
            i_2 = QBLOCK_0;
        }
        uint _S58 = _S56 / i_2;
        thread array<uint, int(2)> rows_0;
        rows_0[0U] = min(col0_0, (&kernelContext_14)->p_0->n_1 - 1U);
        rows_0[1U] = min(col0_0 + 1U, (&kernelContext_14)->p_0->n_1 - 1U);
        i_2 = lane_0;
        for(;;)
        {
            if(i_2 < _S58)
            {
            }
            else
            {
                break;
            }
            thread array<float2, int(2)> sc_4;
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
            int _S82;
            int _S83;
            int _S84;
            int _S85;
            int _S86;
            int _S87;
            int _S88;
            int _S89;
            int _S90;
            float2 _S91 = unitScale_0(rows_0[0U], i_2, &kernelContext_14);
            sc_4[0U] = _S91;
            float2 _S92 = unitScale_0(rows_0[1U], i_2, &kernelContext_14);
            sc_4[1U] = _S92;
            thread array<float, int(16)> sumA_2;
            sumA_2[0U] = 0.0;
            thread array<array<float, int(2)>, int(16)> s_2;
            s_2[0U][0U] = 0.0;
            thread array<array<float, int(2)>, int(16)> s2_1;
            s2_1[0U][0U] = 0.0;
            s_2[0U][1U] = 0.0;
            s2_1[0U][1U] = 0.0;
            sumA_2[1U] = 0.0;
            s_2[1U][0U] = 0.0;
            s2_1[1U][0U] = 0.0;
            s_2[1U][1U] = 0.0;
            s2_1[1U][1U] = 0.0;
            sumA_2[2U] = 0.0;
            s_2[2U][0U] = 0.0;
            s2_1[2U][0U] = 0.0;
            s_2[2U][1U] = 0.0;
            s2_1[2U][1U] = 0.0;
            sumA_2[3U] = 0.0;
            s_2[3U][0U] = 0.0;
            s2_1[3U][0U] = 0.0;
            s_2[3U][1U] = 0.0;
            s2_1[3U][1U] = 0.0;
            sumA_2[4U] = 0.0;
            s_2[4U][0U] = 0.0;
            s2_1[4U][0U] = 0.0;
            s_2[4U][1U] = 0.0;
            s2_1[4U][1U] = 0.0;
            sumA_2[5U] = 0.0;
            s_2[5U][0U] = 0.0;
            s2_1[5U][0U] = 0.0;
            s_2[5U][1U] = 0.0;
            s2_1[5U][1U] = 0.0;
            sumA_2[6U] = 0.0;
            s_2[6U][0U] = 0.0;
            s2_1[6U][0U] = 0.0;
            s_2[6U][1U] = 0.0;
            s2_1[6U][1U] = 0.0;
            sumA_2[7U] = 0.0;
            s_2[7U][0U] = 0.0;
            s2_1[7U][0U] = 0.0;
            s_2[7U][1U] = 0.0;
            s2_1[7U][1U] = 0.0;
            sumA_2[8U] = 0.0;
            s_2[8U][0U] = 0.0;
            s2_1[8U][0U] = 0.0;
            s_2[8U][1U] = 0.0;
            s2_1[8U][1U] = 0.0;
            sumA_2[9U] = 0.0;
            s_2[9U][0U] = 0.0;
            s2_1[9U][0U] = 0.0;
            s_2[9U][1U] = 0.0;
            s2_1[9U][1U] = 0.0;
            sumA_2[10U] = 0.0;
            s_2[10U][0U] = 0.0;
            s2_1[10U][0U] = 0.0;
            s_2[10U][1U] = 0.0;
            s2_1[10U][1U] = 0.0;
            sumA_2[11U] = 0.0;
            s_2[11U][0U] = 0.0;
            s2_1[11U][0U] = 0.0;
            s_2[11U][1U] = 0.0;
            s2_1[11U][1U] = 0.0;
            sumA_2[12U] = 0.0;
            s_2[12U][0U] = 0.0;
            s2_1[12U][0U] = 0.0;
            s_2[12U][1U] = 0.0;
            s2_1[12U][1U] = 0.0;
            sumA_2[13U] = 0.0;
            s_2[13U][0U] = 0.0;
            s2_1[13U][0U] = 0.0;
            s_2[13U][1U] = 0.0;
            s2_1[13U][1U] = 0.0;
            sumA_2[14U] = 0.0;
            s_2[14U][0U] = 0.0;
            s2_1[14U][0U] = 0.0;
            s_2[14U][1U] = 0.0;
            s2_1[14U][1U] = 0.0;
            sumA_2[15U] = 0.0;
            s_2[15U][0U] = 0.0;
            s2_1[15U][0U] = 0.0;
            s_2[15U][1U] = 0.0;
            s2_1[15U][1U] = 0.0;
            w_11 = 0U;
            for(;;)
            {
                uint _S93;
                if(_S57)
                {
                    _S93 = 32U;
                }
                else
                {
                    _S93 = QBLOCK_0;
                }
                if(w_11 < _S93)
                {
                }
                else
                {
                    break;
                }
                uint _S94;
                if(_S57)
                {
                    _S94 = 32U;
                }
                else
                {
                    _S94 = QBLOCK_0;
                }
                uint _S95 = i_2 * _S94 + w_11;
                bool _S96;
                if(QFORMAT_0 == 5U)
                {
                    _S96 = w_11 >= 16U;
                }
                else
                {
                    _S96 = false;
                }
                thread array<array<float, int(8)>, int(2)> q_3;
                load8Codes_0(rows_0[0U], _S95, &q_3[0U], &kernelContext_14);
                load8Codes_0(rows_0[1U], _S95, &q_3[1U], &kernelContext_14);
                for(;;)
                {
                    bool _S97;
                    bool _S98;
                    bool _S99;
                    bool _S100;
                    bool _S101;
                    bool _S102;
                    bool _S103;
                    bool _S104;
                    bool _S105;
                    bool _S106;
                    bool _S107;
                    bool _S108;
                    bool _S109;
                    bool _S110;
                    bool _S111;
                    bool _S112;
                    uint _S113;
                    for(;;)
                    {
                        for(;;)
                        {
                            if(0U >= _S55)
                            {
                                _S90 = int(2);
                                _S113 = 0U;
                                break;
                            }
                            thread array<float, int(8)> av_0;
                            load8A_0(_S54.baseA_0 + _S95, &av_0, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[0U] = sumA_2[0U] + av_0[0U];
                            }
                            if(_S96)
                            {
                                s2_1[0U][0U] = fma(av_0[0U], q_3[0U][0U], s2_1[0U][0U]);
                            }
                            else
                            {
                                s_2[0U][0U] = fma(av_0[0U], q_3[0U][0U], s_2[0U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[0U][1U] = fma(av_0[0U], q_3[1U][0U], s2_1[0U][1U]);
                            }
                            else
                            {
                                s_2[0U][1U] = fma(av_0[0U], q_3[1U][0U], s_2[0U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[0U] = sumA_2[0U] + av_0[1U];
                            }
                            if(_S96)
                            {
                                s2_1[0U][0U] = fma(av_0[1U], q_3[0U][1U], s2_1[0U][0U]);
                            }
                            else
                            {
                                s_2[0U][0U] = fma(av_0[1U], q_3[0U][1U], s_2[0U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[0U][1U] = fma(av_0[1U], q_3[1U][1U], s2_1[0U][1U]);
                            }
                            else
                            {
                                s_2[0U][1U] = fma(av_0[1U], q_3[1U][1U], s_2[0U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[0U] = sumA_2[0U] + av_0[2U];
                            }
                            if(_S96)
                            {
                                s2_1[0U][0U] = fma(av_0[2U], q_3[0U][2U], s2_1[0U][0U]);
                            }
                            else
                            {
                                s_2[0U][0U] = fma(av_0[2U], q_3[0U][2U], s_2[0U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[0U][1U] = fma(av_0[2U], q_3[1U][2U], s2_1[0U][1U]);
                            }
                            else
                            {
                                s_2[0U][1U] = fma(av_0[2U], q_3[1U][2U], s_2[0U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[0U] = sumA_2[0U] + av_0[3U];
                            }
                            if(_S96)
                            {
                                s2_1[0U][0U] = fma(av_0[3U], q_3[0U][3U], s2_1[0U][0U]);
                            }
                            else
                            {
                                s_2[0U][0U] = fma(av_0[3U], q_3[0U][3U], s_2[0U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[0U][1U] = fma(av_0[3U], q_3[1U][3U], s2_1[0U][1U]);
                            }
                            else
                            {
                                s_2[0U][1U] = fma(av_0[3U], q_3[1U][3U], s_2[0U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[0U] = sumA_2[0U] + av_0[4U];
                            }
                            if(_S96)
                            {
                                s2_1[0U][0U] = fma(av_0[4U], q_3[0U][4U], s2_1[0U][0U]);
                            }
                            else
                            {
                                s_2[0U][0U] = fma(av_0[4U], q_3[0U][4U], s_2[0U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[0U][1U] = fma(av_0[4U], q_3[1U][4U], s2_1[0U][1U]);
                            }
                            else
                            {
                                s_2[0U][1U] = fma(av_0[4U], q_3[1U][4U], s_2[0U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[0U] = sumA_2[0U] + av_0[5U];
                            }
                            if(_S96)
                            {
                                s2_1[0U][0U] = fma(av_0[5U], q_3[0U][5U], s2_1[0U][0U]);
                            }
                            else
                            {
                                s_2[0U][0U] = fma(av_0[5U], q_3[0U][5U], s_2[0U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[0U][1U] = fma(av_0[5U], q_3[1U][5U], s2_1[0U][1U]);
                            }
                            else
                            {
                                s_2[0U][1U] = fma(av_0[5U], q_3[1U][5U], s_2[0U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[0U] = sumA_2[0U] + av_0[6U];
                            }
                            if(_S96)
                            {
                                s2_1[0U][0U] = fma(av_0[6U], q_3[0U][6U], s2_1[0U][0U]);
                            }
                            else
                            {
                                s_2[0U][0U] = fma(av_0[6U], q_3[0U][6U], s_2[0U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[0U][1U] = fma(av_0[6U], q_3[1U][6U], s2_1[0U][1U]);
                            }
                            else
                            {
                                s_2[0U][1U] = fma(av_0[6U], q_3[1U][6U], s_2[0U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[0U] = sumA_2[0U] + av_0[7U];
                            }
                            if(_S96)
                            {
                                s2_1[0U][0U] = fma(av_0[7U], q_3[0U][7U], s2_1[0U][0U]);
                            }
                            else
                            {
                                s_2[0U][0U] = fma(av_0[7U], q_3[0U][7U], s_2[0U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[0U][1U] = fma(av_0[7U], q_3[1U][7U], s2_1[0U][1U]);
                            }
                            else
                            {
                                s_2[0U][1U] = fma(av_0[7U], q_3[1U][7U], s_2[0U][1U]);
                            }
                            _S90 = int(4);
                            _S113 = _S54.baseA_0;
                            break;
                        }
                        if(_S90 != int(4))
                        {
                            _S89 = _S90;
                            break;
                        }
                        _S89 = int(3);
                        break;
                    }
                    if(_S89 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(1U >= _S55)
                            {
                                _S88 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_1;
                            load8A_0(_S113 + (&kernelContext_14)->p_0->lda_1 + _S95, &av_1, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[1U] = sumA_2[1U] + av_1[0U];
                            }
                            if(_S96)
                            {
                                s2_1[1U][0U] = fma(av_1[0U], q_3[0U][0U], s2_1[1U][0U]);
                            }
                            else
                            {
                                s_2[1U][0U] = fma(av_1[0U], q_3[0U][0U], s_2[1U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[1U][1U] = fma(av_1[0U], q_3[1U][0U], s2_1[1U][1U]);
                            }
                            else
                            {
                                s_2[1U][1U] = fma(av_1[0U], q_3[1U][0U], s_2[1U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[1U] = sumA_2[1U] + av_1[1U];
                            }
                            if(_S96)
                            {
                                s2_1[1U][0U] = fma(av_1[1U], q_3[0U][1U], s2_1[1U][0U]);
                            }
                            else
                            {
                                s_2[1U][0U] = fma(av_1[1U], q_3[0U][1U], s_2[1U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[1U][1U] = fma(av_1[1U], q_3[1U][1U], s2_1[1U][1U]);
                            }
                            else
                            {
                                s_2[1U][1U] = fma(av_1[1U], q_3[1U][1U], s_2[1U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[1U] = sumA_2[1U] + av_1[2U];
                            }
                            if(_S96)
                            {
                                s2_1[1U][0U] = fma(av_1[2U], q_3[0U][2U], s2_1[1U][0U]);
                            }
                            else
                            {
                                s_2[1U][0U] = fma(av_1[2U], q_3[0U][2U], s_2[1U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[1U][1U] = fma(av_1[2U], q_3[1U][2U], s2_1[1U][1U]);
                            }
                            else
                            {
                                s_2[1U][1U] = fma(av_1[2U], q_3[1U][2U], s_2[1U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[1U] = sumA_2[1U] + av_1[3U];
                            }
                            if(_S96)
                            {
                                s2_1[1U][0U] = fma(av_1[3U], q_3[0U][3U], s2_1[1U][0U]);
                            }
                            else
                            {
                                s_2[1U][0U] = fma(av_1[3U], q_3[0U][3U], s_2[1U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[1U][1U] = fma(av_1[3U], q_3[1U][3U], s2_1[1U][1U]);
                            }
                            else
                            {
                                s_2[1U][1U] = fma(av_1[3U], q_3[1U][3U], s_2[1U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[1U] = sumA_2[1U] + av_1[4U];
                            }
                            if(_S96)
                            {
                                s2_1[1U][0U] = fma(av_1[4U], q_3[0U][4U], s2_1[1U][0U]);
                            }
                            else
                            {
                                s_2[1U][0U] = fma(av_1[4U], q_3[0U][4U], s_2[1U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[1U][1U] = fma(av_1[4U], q_3[1U][4U], s2_1[1U][1U]);
                            }
                            else
                            {
                                s_2[1U][1U] = fma(av_1[4U], q_3[1U][4U], s_2[1U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[1U] = sumA_2[1U] + av_1[5U];
                            }
                            if(_S96)
                            {
                                s2_1[1U][0U] = fma(av_1[5U], q_3[0U][5U], s2_1[1U][0U]);
                            }
                            else
                            {
                                s_2[1U][0U] = fma(av_1[5U], q_3[0U][5U], s_2[1U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[1U][1U] = fma(av_1[5U], q_3[1U][5U], s2_1[1U][1U]);
                            }
                            else
                            {
                                s_2[1U][1U] = fma(av_1[5U], q_3[1U][5U], s_2[1U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[1U] = sumA_2[1U] + av_1[6U];
                            }
                            if(_S96)
                            {
                                s2_1[1U][0U] = fma(av_1[6U], q_3[0U][6U], s2_1[1U][0U]);
                            }
                            else
                            {
                                s_2[1U][0U] = fma(av_1[6U], q_3[0U][6U], s_2[1U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[1U][1U] = fma(av_1[6U], q_3[1U][6U], s2_1[1U][1U]);
                            }
                            else
                            {
                                s_2[1U][1U] = fma(av_1[6U], q_3[1U][6U], s_2[1U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[1U] = sumA_2[1U] + av_1[7U];
                            }
                            if(_S96)
                            {
                                s2_1[1U][0U] = fma(av_1[7U], q_3[0U][7U], s2_1[1U][0U]);
                            }
                            else
                            {
                                s_2[1U][0U] = fma(av_1[7U], q_3[0U][7U], s_2[1U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[1U][1U] = fma(av_1[7U], q_3[1U][7U], s2_1[1U][1U]);
                            }
                            else
                            {
                                s_2[1U][1U] = fma(av_1[7U], q_3[1U][7U], s_2[1U][1U]);
                            }
                            _S88 = int(4);
                            break;
                        }
                        if(_S88 != int(4))
                        {
                            _S87 = _S88;
                            break;
                        }
                        _S87 = int(3);
                        break;
                    }
                    if(_S87 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(2U >= _S55)
                            {
                                _S86 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_2;
                            load8A_0(_S113 + 2U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_2, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[2U] = sumA_2[2U] + av_2[0U];
                            }
                            if(_S96)
                            {
                                s2_1[2U][0U] = fma(av_2[0U], q_3[0U][0U], s2_1[2U][0U]);
                            }
                            else
                            {
                                s_2[2U][0U] = fma(av_2[0U], q_3[0U][0U], s_2[2U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[2U][1U] = fma(av_2[0U], q_3[1U][0U], s2_1[2U][1U]);
                            }
                            else
                            {
                                s_2[2U][1U] = fma(av_2[0U], q_3[1U][0U], s_2[2U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[2U] = sumA_2[2U] + av_2[1U];
                            }
                            if(_S96)
                            {
                                s2_1[2U][0U] = fma(av_2[1U], q_3[0U][1U], s2_1[2U][0U]);
                            }
                            else
                            {
                                s_2[2U][0U] = fma(av_2[1U], q_3[0U][1U], s_2[2U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[2U][1U] = fma(av_2[1U], q_3[1U][1U], s2_1[2U][1U]);
                            }
                            else
                            {
                                s_2[2U][1U] = fma(av_2[1U], q_3[1U][1U], s_2[2U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[2U] = sumA_2[2U] + av_2[2U];
                            }
                            if(_S96)
                            {
                                s2_1[2U][0U] = fma(av_2[2U], q_3[0U][2U], s2_1[2U][0U]);
                            }
                            else
                            {
                                s_2[2U][0U] = fma(av_2[2U], q_3[0U][2U], s_2[2U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[2U][1U] = fma(av_2[2U], q_3[1U][2U], s2_1[2U][1U]);
                            }
                            else
                            {
                                s_2[2U][1U] = fma(av_2[2U], q_3[1U][2U], s_2[2U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[2U] = sumA_2[2U] + av_2[3U];
                            }
                            if(_S96)
                            {
                                s2_1[2U][0U] = fma(av_2[3U], q_3[0U][3U], s2_1[2U][0U]);
                            }
                            else
                            {
                                s_2[2U][0U] = fma(av_2[3U], q_3[0U][3U], s_2[2U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[2U][1U] = fma(av_2[3U], q_3[1U][3U], s2_1[2U][1U]);
                            }
                            else
                            {
                                s_2[2U][1U] = fma(av_2[3U], q_3[1U][3U], s_2[2U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[2U] = sumA_2[2U] + av_2[4U];
                            }
                            if(_S96)
                            {
                                s2_1[2U][0U] = fma(av_2[4U], q_3[0U][4U], s2_1[2U][0U]);
                            }
                            else
                            {
                                s_2[2U][0U] = fma(av_2[4U], q_3[0U][4U], s_2[2U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[2U][1U] = fma(av_2[4U], q_3[1U][4U], s2_1[2U][1U]);
                            }
                            else
                            {
                                s_2[2U][1U] = fma(av_2[4U], q_3[1U][4U], s_2[2U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[2U] = sumA_2[2U] + av_2[5U];
                            }
                            if(_S96)
                            {
                                s2_1[2U][0U] = fma(av_2[5U], q_3[0U][5U], s2_1[2U][0U]);
                            }
                            else
                            {
                                s_2[2U][0U] = fma(av_2[5U], q_3[0U][5U], s_2[2U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[2U][1U] = fma(av_2[5U], q_3[1U][5U], s2_1[2U][1U]);
                            }
                            else
                            {
                                s_2[2U][1U] = fma(av_2[5U], q_3[1U][5U], s_2[2U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[2U] = sumA_2[2U] + av_2[6U];
                            }
                            if(_S96)
                            {
                                s2_1[2U][0U] = fma(av_2[6U], q_3[0U][6U], s2_1[2U][0U]);
                            }
                            else
                            {
                                s_2[2U][0U] = fma(av_2[6U], q_3[0U][6U], s_2[2U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[2U][1U] = fma(av_2[6U], q_3[1U][6U], s2_1[2U][1U]);
                            }
                            else
                            {
                                s_2[2U][1U] = fma(av_2[6U], q_3[1U][6U], s_2[2U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[2U] = sumA_2[2U] + av_2[7U];
                            }
                            if(_S96)
                            {
                                s2_1[2U][0U] = fma(av_2[7U], q_3[0U][7U], s2_1[2U][0U]);
                            }
                            else
                            {
                                s_2[2U][0U] = fma(av_2[7U], q_3[0U][7U], s_2[2U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[2U][1U] = fma(av_2[7U], q_3[1U][7U], s2_1[2U][1U]);
                            }
                            else
                            {
                                s_2[2U][1U] = fma(av_2[7U], q_3[1U][7U], s_2[2U][1U]);
                            }
                            _S86 = int(4);
                            break;
                        }
                        if(_S86 != int(4))
                        {
                            _S85 = _S86;
                            break;
                        }
                        _S85 = int(3);
                        break;
                    }
                    if(_S85 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(3U >= _S55)
                            {
                                _S84 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_3;
                            load8A_0(_S113 + 3U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_3, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[3U] = sumA_2[3U] + av_3[0U];
                            }
                            if(_S96)
                            {
                                s2_1[3U][0U] = fma(av_3[0U], q_3[0U][0U], s2_1[3U][0U]);
                            }
                            else
                            {
                                s_2[3U][0U] = fma(av_3[0U], q_3[0U][0U], s_2[3U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[3U][1U] = fma(av_3[0U], q_3[1U][0U], s2_1[3U][1U]);
                            }
                            else
                            {
                                s_2[3U][1U] = fma(av_3[0U], q_3[1U][0U], s_2[3U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[3U] = sumA_2[3U] + av_3[1U];
                            }
                            if(_S96)
                            {
                                s2_1[3U][0U] = fma(av_3[1U], q_3[0U][1U], s2_1[3U][0U]);
                            }
                            else
                            {
                                s_2[3U][0U] = fma(av_3[1U], q_3[0U][1U], s_2[3U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[3U][1U] = fma(av_3[1U], q_3[1U][1U], s2_1[3U][1U]);
                            }
                            else
                            {
                                s_2[3U][1U] = fma(av_3[1U], q_3[1U][1U], s_2[3U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[3U] = sumA_2[3U] + av_3[2U];
                            }
                            if(_S96)
                            {
                                s2_1[3U][0U] = fma(av_3[2U], q_3[0U][2U], s2_1[3U][0U]);
                            }
                            else
                            {
                                s_2[3U][0U] = fma(av_3[2U], q_3[0U][2U], s_2[3U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[3U][1U] = fma(av_3[2U], q_3[1U][2U], s2_1[3U][1U]);
                            }
                            else
                            {
                                s_2[3U][1U] = fma(av_3[2U], q_3[1U][2U], s_2[3U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[3U] = sumA_2[3U] + av_3[3U];
                            }
                            if(_S96)
                            {
                                s2_1[3U][0U] = fma(av_3[3U], q_3[0U][3U], s2_1[3U][0U]);
                            }
                            else
                            {
                                s_2[3U][0U] = fma(av_3[3U], q_3[0U][3U], s_2[3U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[3U][1U] = fma(av_3[3U], q_3[1U][3U], s2_1[3U][1U]);
                            }
                            else
                            {
                                s_2[3U][1U] = fma(av_3[3U], q_3[1U][3U], s_2[3U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[3U] = sumA_2[3U] + av_3[4U];
                            }
                            if(_S96)
                            {
                                s2_1[3U][0U] = fma(av_3[4U], q_3[0U][4U], s2_1[3U][0U]);
                            }
                            else
                            {
                                s_2[3U][0U] = fma(av_3[4U], q_3[0U][4U], s_2[3U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[3U][1U] = fma(av_3[4U], q_3[1U][4U], s2_1[3U][1U]);
                            }
                            else
                            {
                                s_2[3U][1U] = fma(av_3[4U], q_3[1U][4U], s_2[3U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[3U] = sumA_2[3U] + av_3[5U];
                            }
                            if(_S96)
                            {
                                s2_1[3U][0U] = fma(av_3[5U], q_3[0U][5U], s2_1[3U][0U]);
                            }
                            else
                            {
                                s_2[3U][0U] = fma(av_3[5U], q_3[0U][5U], s_2[3U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[3U][1U] = fma(av_3[5U], q_3[1U][5U], s2_1[3U][1U]);
                            }
                            else
                            {
                                s_2[3U][1U] = fma(av_3[5U], q_3[1U][5U], s_2[3U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[3U] = sumA_2[3U] + av_3[6U];
                            }
                            if(_S96)
                            {
                                s2_1[3U][0U] = fma(av_3[6U], q_3[0U][6U], s2_1[3U][0U]);
                            }
                            else
                            {
                                s_2[3U][0U] = fma(av_3[6U], q_3[0U][6U], s_2[3U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[3U][1U] = fma(av_3[6U], q_3[1U][6U], s2_1[3U][1U]);
                            }
                            else
                            {
                                s_2[3U][1U] = fma(av_3[6U], q_3[1U][6U], s_2[3U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[3U] = sumA_2[3U] + av_3[7U];
                            }
                            if(_S96)
                            {
                                s2_1[3U][0U] = fma(av_3[7U], q_3[0U][7U], s2_1[3U][0U]);
                            }
                            else
                            {
                                s_2[3U][0U] = fma(av_3[7U], q_3[0U][7U], s_2[3U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[3U][1U] = fma(av_3[7U], q_3[1U][7U], s2_1[3U][1U]);
                            }
                            else
                            {
                                s_2[3U][1U] = fma(av_3[7U], q_3[1U][7U], s_2[3U][1U]);
                            }
                            _S84 = int(4);
                            break;
                        }
                        if(_S84 != int(4))
                        {
                            _S83 = _S84;
                            break;
                        }
                        _S83 = int(3);
                        break;
                    }
                    if(_S83 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(4U >= _S55)
                            {
                                _S82 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_4;
                            load8A_0(_S113 + 4U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_4, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[4U] = sumA_2[4U] + av_4[0U];
                            }
                            if(_S96)
                            {
                                s2_1[4U][0U] = fma(av_4[0U], q_3[0U][0U], s2_1[4U][0U]);
                            }
                            else
                            {
                                s_2[4U][0U] = fma(av_4[0U], q_3[0U][0U], s_2[4U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[4U][1U] = fma(av_4[0U], q_3[1U][0U], s2_1[4U][1U]);
                            }
                            else
                            {
                                s_2[4U][1U] = fma(av_4[0U], q_3[1U][0U], s_2[4U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[4U] = sumA_2[4U] + av_4[1U];
                            }
                            if(_S96)
                            {
                                s2_1[4U][0U] = fma(av_4[1U], q_3[0U][1U], s2_1[4U][0U]);
                            }
                            else
                            {
                                s_2[4U][0U] = fma(av_4[1U], q_3[0U][1U], s_2[4U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[4U][1U] = fma(av_4[1U], q_3[1U][1U], s2_1[4U][1U]);
                            }
                            else
                            {
                                s_2[4U][1U] = fma(av_4[1U], q_3[1U][1U], s_2[4U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[4U] = sumA_2[4U] + av_4[2U];
                            }
                            if(_S96)
                            {
                                s2_1[4U][0U] = fma(av_4[2U], q_3[0U][2U], s2_1[4U][0U]);
                            }
                            else
                            {
                                s_2[4U][0U] = fma(av_4[2U], q_3[0U][2U], s_2[4U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[4U][1U] = fma(av_4[2U], q_3[1U][2U], s2_1[4U][1U]);
                            }
                            else
                            {
                                s_2[4U][1U] = fma(av_4[2U], q_3[1U][2U], s_2[4U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[4U] = sumA_2[4U] + av_4[3U];
                            }
                            if(_S96)
                            {
                                s2_1[4U][0U] = fma(av_4[3U], q_3[0U][3U], s2_1[4U][0U]);
                            }
                            else
                            {
                                s_2[4U][0U] = fma(av_4[3U], q_3[0U][3U], s_2[4U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[4U][1U] = fma(av_4[3U], q_3[1U][3U], s2_1[4U][1U]);
                            }
                            else
                            {
                                s_2[4U][1U] = fma(av_4[3U], q_3[1U][3U], s_2[4U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[4U] = sumA_2[4U] + av_4[4U];
                            }
                            if(_S96)
                            {
                                s2_1[4U][0U] = fma(av_4[4U], q_3[0U][4U], s2_1[4U][0U]);
                            }
                            else
                            {
                                s_2[4U][0U] = fma(av_4[4U], q_3[0U][4U], s_2[4U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[4U][1U] = fma(av_4[4U], q_3[1U][4U], s2_1[4U][1U]);
                            }
                            else
                            {
                                s_2[4U][1U] = fma(av_4[4U], q_3[1U][4U], s_2[4U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[4U] = sumA_2[4U] + av_4[5U];
                            }
                            if(_S96)
                            {
                                s2_1[4U][0U] = fma(av_4[5U], q_3[0U][5U], s2_1[4U][0U]);
                            }
                            else
                            {
                                s_2[4U][0U] = fma(av_4[5U], q_3[0U][5U], s_2[4U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[4U][1U] = fma(av_4[5U], q_3[1U][5U], s2_1[4U][1U]);
                            }
                            else
                            {
                                s_2[4U][1U] = fma(av_4[5U], q_3[1U][5U], s_2[4U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[4U] = sumA_2[4U] + av_4[6U];
                            }
                            if(_S96)
                            {
                                s2_1[4U][0U] = fma(av_4[6U], q_3[0U][6U], s2_1[4U][0U]);
                            }
                            else
                            {
                                s_2[4U][0U] = fma(av_4[6U], q_3[0U][6U], s_2[4U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[4U][1U] = fma(av_4[6U], q_3[1U][6U], s2_1[4U][1U]);
                            }
                            else
                            {
                                s_2[4U][1U] = fma(av_4[6U], q_3[1U][6U], s_2[4U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[4U] = sumA_2[4U] + av_4[7U];
                            }
                            if(_S96)
                            {
                                s2_1[4U][0U] = fma(av_4[7U], q_3[0U][7U], s2_1[4U][0U]);
                            }
                            else
                            {
                                s_2[4U][0U] = fma(av_4[7U], q_3[0U][7U], s_2[4U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[4U][1U] = fma(av_4[7U], q_3[1U][7U], s2_1[4U][1U]);
                            }
                            else
                            {
                                s_2[4U][1U] = fma(av_4[7U], q_3[1U][7U], s_2[4U][1U]);
                            }
                            _S82 = int(4);
                            break;
                        }
                        if(_S82 != int(4))
                        {
                            _S81 = _S82;
                            break;
                        }
                        _S81 = int(3);
                        break;
                    }
                    if(_S81 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(5U >= _S55)
                            {
                                _S80 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_5;
                            load8A_0(_S113 + 5U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_5, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[5U] = sumA_2[5U] + av_5[0U];
                            }
                            if(_S96)
                            {
                                s2_1[5U][0U] = fma(av_5[0U], q_3[0U][0U], s2_1[5U][0U]);
                            }
                            else
                            {
                                s_2[5U][0U] = fma(av_5[0U], q_3[0U][0U], s_2[5U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[5U][1U] = fma(av_5[0U], q_3[1U][0U], s2_1[5U][1U]);
                            }
                            else
                            {
                                s_2[5U][1U] = fma(av_5[0U], q_3[1U][0U], s_2[5U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[5U] = sumA_2[5U] + av_5[1U];
                            }
                            if(_S96)
                            {
                                s2_1[5U][0U] = fma(av_5[1U], q_3[0U][1U], s2_1[5U][0U]);
                            }
                            else
                            {
                                s_2[5U][0U] = fma(av_5[1U], q_3[0U][1U], s_2[5U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[5U][1U] = fma(av_5[1U], q_3[1U][1U], s2_1[5U][1U]);
                            }
                            else
                            {
                                s_2[5U][1U] = fma(av_5[1U], q_3[1U][1U], s_2[5U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[5U] = sumA_2[5U] + av_5[2U];
                            }
                            if(_S96)
                            {
                                s2_1[5U][0U] = fma(av_5[2U], q_3[0U][2U], s2_1[5U][0U]);
                            }
                            else
                            {
                                s_2[5U][0U] = fma(av_5[2U], q_3[0U][2U], s_2[5U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[5U][1U] = fma(av_5[2U], q_3[1U][2U], s2_1[5U][1U]);
                            }
                            else
                            {
                                s_2[5U][1U] = fma(av_5[2U], q_3[1U][2U], s_2[5U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[5U] = sumA_2[5U] + av_5[3U];
                            }
                            if(_S96)
                            {
                                s2_1[5U][0U] = fma(av_5[3U], q_3[0U][3U], s2_1[5U][0U]);
                            }
                            else
                            {
                                s_2[5U][0U] = fma(av_5[3U], q_3[0U][3U], s_2[5U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[5U][1U] = fma(av_5[3U], q_3[1U][3U], s2_1[5U][1U]);
                            }
                            else
                            {
                                s_2[5U][1U] = fma(av_5[3U], q_3[1U][3U], s_2[5U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[5U] = sumA_2[5U] + av_5[4U];
                            }
                            if(_S96)
                            {
                                s2_1[5U][0U] = fma(av_5[4U], q_3[0U][4U], s2_1[5U][0U]);
                            }
                            else
                            {
                                s_2[5U][0U] = fma(av_5[4U], q_3[0U][4U], s_2[5U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[5U][1U] = fma(av_5[4U], q_3[1U][4U], s2_1[5U][1U]);
                            }
                            else
                            {
                                s_2[5U][1U] = fma(av_5[4U], q_3[1U][4U], s_2[5U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[5U] = sumA_2[5U] + av_5[5U];
                            }
                            if(_S96)
                            {
                                s2_1[5U][0U] = fma(av_5[5U], q_3[0U][5U], s2_1[5U][0U]);
                            }
                            else
                            {
                                s_2[5U][0U] = fma(av_5[5U], q_3[0U][5U], s_2[5U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[5U][1U] = fma(av_5[5U], q_3[1U][5U], s2_1[5U][1U]);
                            }
                            else
                            {
                                s_2[5U][1U] = fma(av_5[5U], q_3[1U][5U], s_2[5U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[5U] = sumA_2[5U] + av_5[6U];
                            }
                            if(_S96)
                            {
                                s2_1[5U][0U] = fma(av_5[6U], q_3[0U][6U], s2_1[5U][0U]);
                            }
                            else
                            {
                                s_2[5U][0U] = fma(av_5[6U], q_3[0U][6U], s_2[5U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[5U][1U] = fma(av_5[6U], q_3[1U][6U], s2_1[5U][1U]);
                            }
                            else
                            {
                                s_2[5U][1U] = fma(av_5[6U], q_3[1U][6U], s_2[5U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[5U] = sumA_2[5U] + av_5[7U];
                            }
                            if(_S96)
                            {
                                s2_1[5U][0U] = fma(av_5[7U], q_3[0U][7U], s2_1[5U][0U]);
                            }
                            else
                            {
                                s_2[5U][0U] = fma(av_5[7U], q_3[0U][7U], s_2[5U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[5U][1U] = fma(av_5[7U], q_3[1U][7U], s2_1[5U][1U]);
                            }
                            else
                            {
                                s_2[5U][1U] = fma(av_5[7U], q_3[1U][7U], s_2[5U][1U]);
                            }
                            _S80 = int(4);
                            break;
                        }
                        if(_S80 != int(4))
                        {
                            _S79 = _S80;
                            break;
                        }
                        _S79 = int(3);
                        break;
                    }
                    if(_S79 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(6U >= _S55)
                            {
                                _S78 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_6;
                            load8A_0(_S113 + 6U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_6, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[6U] = sumA_2[6U] + av_6[0U];
                            }
                            if(_S96)
                            {
                                s2_1[6U][0U] = fma(av_6[0U], q_3[0U][0U], s2_1[6U][0U]);
                            }
                            else
                            {
                                s_2[6U][0U] = fma(av_6[0U], q_3[0U][0U], s_2[6U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[6U][1U] = fma(av_6[0U], q_3[1U][0U], s2_1[6U][1U]);
                            }
                            else
                            {
                                s_2[6U][1U] = fma(av_6[0U], q_3[1U][0U], s_2[6U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[6U] = sumA_2[6U] + av_6[1U];
                            }
                            if(_S96)
                            {
                                s2_1[6U][0U] = fma(av_6[1U], q_3[0U][1U], s2_1[6U][0U]);
                            }
                            else
                            {
                                s_2[6U][0U] = fma(av_6[1U], q_3[0U][1U], s_2[6U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[6U][1U] = fma(av_6[1U], q_3[1U][1U], s2_1[6U][1U]);
                            }
                            else
                            {
                                s_2[6U][1U] = fma(av_6[1U], q_3[1U][1U], s_2[6U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[6U] = sumA_2[6U] + av_6[2U];
                            }
                            if(_S96)
                            {
                                s2_1[6U][0U] = fma(av_6[2U], q_3[0U][2U], s2_1[6U][0U]);
                            }
                            else
                            {
                                s_2[6U][0U] = fma(av_6[2U], q_3[0U][2U], s_2[6U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[6U][1U] = fma(av_6[2U], q_3[1U][2U], s2_1[6U][1U]);
                            }
                            else
                            {
                                s_2[6U][1U] = fma(av_6[2U], q_3[1U][2U], s_2[6U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[6U] = sumA_2[6U] + av_6[3U];
                            }
                            if(_S96)
                            {
                                s2_1[6U][0U] = fma(av_6[3U], q_3[0U][3U], s2_1[6U][0U]);
                            }
                            else
                            {
                                s_2[6U][0U] = fma(av_6[3U], q_3[0U][3U], s_2[6U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[6U][1U] = fma(av_6[3U], q_3[1U][3U], s2_1[6U][1U]);
                            }
                            else
                            {
                                s_2[6U][1U] = fma(av_6[3U], q_3[1U][3U], s_2[6U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[6U] = sumA_2[6U] + av_6[4U];
                            }
                            if(_S96)
                            {
                                s2_1[6U][0U] = fma(av_6[4U], q_3[0U][4U], s2_1[6U][0U]);
                            }
                            else
                            {
                                s_2[6U][0U] = fma(av_6[4U], q_3[0U][4U], s_2[6U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[6U][1U] = fma(av_6[4U], q_3[1U][4U], s2_1[6U][1U]);
                            }
                            else
                            {
                                s_2[6U][1U] = fma(av_6[4U], q_3[1U][4U], s_2[6U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[6U] = sumA_2[6U] + av_6[5U];
                            }
                            if(_S96)
                            {
                                s2_1[6U][0U] = fma(av_6[5U], q_3[0U][5U], s2_1[6U][0U]);
                            }
                            else
                            {
                                s_2[6U][0U] = fma(av_6[5U], q_3[0U][5U], s_2[6U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[6U][1U] = fma(av_6[5U], q_3[1U][5U], s2_1[6U][1U]);
                            }
                            else
                            {
                                s_2[6U][1U] = fma(av_6[5U], q_3[1U][5U], s_2[6U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[6U] = sumA_2[6U] + av_6[6U];
                            }
                            if(_S96)
                            {
                                s2_1[6U][0U] = fma(av_6[6U], q_3[0U][6U], s2_1[6U][0U]);
                            }
                            else
                            {
                                s_2[6U][0U] = fma(av_6[6U], q_3[0U][6U], s_2[6U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[6U][1U] = fma(av_6[6U], q_3[1U][6U], s2_1[6U][1U]);
                            }
                            else
                            {
                                s_2[6U][1U] = fma(av_6[6U], q_3[1U][6U], s_2[6U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[6U] = sumA_2[6U] + av_6[7U];
                            }
                            if(_S96)
                            {
                                s2_1[6U][0U] = fma(av_6[7U], q_3[0U][7U], s2_1[6U][0U]);
                            }
                            else
                            {
                                s_2[6U][0U] = fma(av_6[7U], q_3[0U][7U], s_2[6U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[6U][1U] = fma(av_6[7U], q_3[1U][7U], s2_1[6U][1U]);
                            }
                            else
                            {
                                s_2[6U][1U] = fma(av_6[7U], q_3[1U][7U], s_2[6U][1U]);
                            }
                            _S78 = int(4);
                            break;
                        }
                        if(_S78 != int(4))
                        {
                            _S77 = _S78;
                            break;
                        }
                        _S77 = int(3);
                        break;
                    }
                    if(_S77 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(7U >= _S55)
                            {
                                _S76 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_7;
                            load8A_0(_S113 + 7U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_7, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[7U] = sumA_2[7U] + av_7[0U];
                            }
                            if(_S96)
                            {
                                s2_1[7U][0U] = fma(av_7[0U], q_3[0U][0U], s2_1[7U][0U]);
                            }
                            else
                            {
                                s_2[7U][0U] = fma(av_7[0U], q_3[0U][0U], s_2[7U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[7U][1U] = fma(av_7[0U], q_3[1U][0U], s2_1[7U][1U]);
                            }
                            else
                            {
                                s_2[7U][1U] = fma(av_7[0U], q_3[1U][0U], s_2[7U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[7U] = sumA_2[7U] + av_7[1U];
                            }
                            if(_S96)
                            {
                                s2_1[7U][0U] = fma(av_7[1U], q_3[0U][1U], s2_1[7U][0U]);
                            }
                            else
                            {
                                s_2[7U][0U] = fma(av_7[1U], q_3[0U][1U], s_2[7U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[7U][1U] = fma(av_7[1U], q_3[1U][1U], s2_1[7U][1U]);
                            }
                            else
                            {
                                s_2[7U][1U] = fma(av_7[1U], q_3[1U][1U], s_2[7U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[7U] = sumA_2[7U] + av_7[2U];
                            }
                            if(_S96)
                            {
                                s2_1[7U][0U] = fma(av_7[2U], q_3[0U][2U], s2_1[7U][0U]);
                            }
                            else
                            {
                                s_2[7U][0U] = fma(av_7[2U], q_3[0U][2U], s_2[7U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[7U][1U] = fma(av_7[2U], q_3[1U][2U], s2_1[7U][1U]);
                            }
                            else
                            {
                                s_2[7U][1U] = fma(av_7[2U], q_3[1U][2U], s_2[7U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[7U] = sumA_2[7U] + av_7[3U];
                            }
                            if(_S96)
                            {
                                s2_1[7U][0U] = fma(av_7[3U], q_3[0U][3U], s2_1[7U][0U]);
                            }
                            else
                            {
                                s_2[7U][0U] = fma(av_7[3U], q_3[0U][3U], s_2[7U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[7U][1U] = fma(av_7[3U], q_3[1U][3U], s2_1[7U][1U]);
                            }
                            else
                            {
                                s_2[7U][1U] = fma(av_7[3U], q_3[1U][3U], s_2[7U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[7U] = sumA_2[7U] + av_7[4U];
                            }
                            if(_S96)
                            {
                                s2_1[7U][0U] = fma(av_7[4U], q_3[0U][4U], s2_1[7U][0U]);
                            }
                            else
                            {
                                s_2[7U][0U] = fma(av_7[4U], q_3[0U][4U], s_2[7U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[7U][1U] = fma(av_7[4U], q_3[1U][4U], s2_1[7U][1U]);
                            }
                            else
                            {
                                s_2[7U][1U] = fma(av_7[4U], q_3[1U][4U], s_2[7U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[7U] = sumA_2[7U] + av_7[5U];
                            }
                            if(_S96)
                            {
                                s2_1[7U][0U] = fma(av_7[5U], q_3[0U][5U], s2_1[7U][0U]);
                            }
                            else
                            {
                                s_2[7U][0U] = fma(av_7[5U], q_3[0U][5U], s_2[7U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[7U][1U] = fma(av_7[5U], q_3[1U][5U], s2_1[7U][1U]);
                            }
                            else
                            {
                                s_2[7U][1U] = fma(av_7[5U], q_3[1U][5U], s_2[7U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[7U] = sumA_2[7U] + av_7[6U];
                            }
                            if(_S96)
                            {
                                s2_1[7U][0U] = fma(av_7[6U], q_3[0U][6U], s2_1[7U][0U]);
                            }
                            else
                            {
                                s_2[7U][0U] = fma(av_7[6U], q_3[0U][6U], s_2[7U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[7U][1U] = fma(av_7[6U], q_3[1U][6U], s2_1[7U][1U]);
                            }
                            else
                            {
                                s_2[7U][1U] = fma(av_7[6U], q_3[1U][6U], s_2[7U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[7U] = sumA_2[7U] + av_7[7U];
                            }
                            if(_S96)
                            {
                                s2_1[7U][0U] = fma(av_7[7U], q_3[0U][7U], s2_1[7U][0U]);
                            }
                            else
                            {
                                s_2[7U][0U] = fma(av_7[7U], q_3[0U][7U], s_2[7U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[7U][1U] = fma(av_7[7U], q_3[1U][7U], s2_1[7U][1U]);
                            }
                            else
                            {
                                s_2[7U][1U] = fma(av_7[7U], q_3[1U][7U], s_2[7U][1U]);
                            }
                            _S76 = int(4);
                            break;
                        }
                        if(_S76 != int(4))
                        {
                            _S75 = _S76;
                            break;
                        }
                        _S75 = int(3);
                        break;
                    }
                    if(_S75 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(8U >= _S55)
                            {
                                _S74 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_8;
                            load8A_0(_S113 + 8U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_8, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[8U] = sumA_2[8U] + av_8[0U];
                            }
                            if(_S96)
                            {
                                s2_1[8U][0U] = fma(av_8[0U], q_3[0U][0U], s2_1[8U][0U]);
                            }
                            else
                            {
                                s_2[8U][0U] = fma(av_8[0U], q_3[0U][0U], s_2[8U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[8U][1U] = fma(av_8[0U], q_3[1U][0U], s2_1[8U][1U]);
                            }
                            else
                            {
                                s_2[8U][1U] = fma(av_8[0U], q_3[1U][0U], s_2[8U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[8U] = sumA_2[8U] + av_8[1U];
                            }
                            if(_S96)
                            {
                                s2_1[8U][0U] = fma(av_8[1U], q_3[0U][1U], s2_1[8U][0U]);
                            }
                            else
                            {
                                s_2[8U][0U] = fma(av_8[1U], q_3[0U][1U], s_2[8U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[8U][1U] = fma(av_8[1U], q_3[1U][1U], s2_1[8U][1U]);
                            }
                            else
                            {
                                s_2[8U][1U] = fma(av_8[1U], q_3[1U][1U], s_2[8U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[8U] = sumA_2[8U] + av_8[2U];
                            }
                            if(_S96)
                            {
                                s2_1[8U][0U] = fma(av_8[2U], q_3[0U][2U], s2_1[8U][0U]);
                            }
                            else
                            {
                                s_2[8U][0U] = fma(av_8[2U], q_3[0U][2U], s_2[8U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[8U][1U] = fma(av_8[2U], q_3[1U][2U], s2_1[8U][1U]);
                            }
                            else
                            {
                                s_2[8U][1U] = fma(av_8[2U], q_3[1U][2U], s_2[8U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[8U] = sumA_2[8U] + av_8[3U];
                            }
                            if(_S96)
                            {
                                s2_1[8U][0U] = fma(av_8[3U], q_3[0U][3U], s2_1[8U][0U]);
                            }
                            else
                            {
                                s_2[8U][0U] = fma(av_8[3U], q_3[0U][3U], s_2[8U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[8U][1U] = fma(av_8[3U], q_3[1U][3U], s2_1[8U][1U]);
                            }
                            else
                            {
                                s_2[8U][1U] = fma(av_8[3U], q_3[1U][3U], s_2[8U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[8U] = sumA_2[8U] + av_8[4U];
                            }
                            if(_S96)
                            {
                                s2_1[8U][0U] = fma(av_8[4U], q_3[0U][4U], s2_1[8U][0U]);
                            }
                            else
                            {
                                s_2[8U][0U] = fma(av_8[4U], q_3[0U][4U], s_2[8U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[8U][1U] = fma(av_8[4U], q_3[1U][4U], s2_1[8U][1U]);
                            }
                            else
                            {
                                s_2[8U][1U] = fma(av_8[4U], q_3[1U][4U], s_2[8U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[8U] = sumA_2[8U] + av_8[5U];
                            }
                            if(_S96)
                            {
                                s2_1[8U][0U] = fma(av_8[5U], q_3[0U][5U], s2_1[8U][0U]);
                            }
                            else
                            {
                                s_2[8U][0U] = fma(av_8[5U], q_3[0U][5U], s_2[8U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[8U][1U] = fma(av_8[5U], q_3[1U][5U], s2_1[8U][1U]);
                            }
                            else
                            {
                                s_2[8U][1U] = fma(av_8[5U], q_3[1U][5U], s_2[8U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[8U] = sumA_2[8U] + av_8[6U];
                            }
                            if(_S96)
                            {
                                s2_1[8U][0U] = fma(av_8[6U], q_3[0U][6U], s2_1[8U][0U]);
                            }
                            else
                            {
                                s_2[8U][0U] = fma(av_8[6U], q_3[0U][6U], s_2[8U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[8U][1U] = fma(av_8[6U], q_3[1U][6U], s2_1[8U][1U]);
                            }
                            else
                            {
                                s_2[8U][1U] = fma(av_8[6U], q_3[1U][6U], s_2[8U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[8U] = sumA_2[8U] + av_8[7U];
                            }
                            if(_S96)
                            {
                                s2_1[8U][0U] = fma(av_8[7U], q_3[0U][7U], s2_1[8U][0U]);
                            }
                            else
                            {
                                s_2[8U][0U] = fma(av_8[7U], q_3[0U][7U], s_2[8U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[8U][1U] = fma(av_8[7U], q_3[1U][7U], s2_1[8U][1U]);
                            }
                            else
                            {
                                s_2[8U][1U] = fma(av_8[7U], q_3[1U][7U], s_2[8U][1U]);
                            }
                            _S74 = int(4);
                            break;
                        }
                        if(_S74 != int(4))
                        {
                            _S73 = _S74;
                            break;
                        }
                        _S73 = int(3);
                        break;
                    }
                    if(_S73 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(9U >= _S55)
                            {
                                _S72 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_9;
                            load8A_0(_S113 + 9U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_9, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[9U] = sumA_2[9U] + av_9[0U];
                            }
                            if(_S96)
                            {
                                s2_1[9U][0U] = fma(av_9[0U], q_3[0U][0U], s2_1[9U][0U]);
                            }
                            else
                            {
                                s_2[9U][0U] = fma(av_9[0U], q_3[0U][0U], s_2[9U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[9U][1U] = fma(av_9[0U], q_3[1U][0U], s2_1[9U][1U]);
                            }
                            else
                            {
                                s_2[9U][1U] = fma(av_9[0U], q_3[1U][0U], s_2[9U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[9U] = sumA_2[9U] + av_9[1U];
                            }
                            if(_S96)
                            {
                                s2_1[9U][0U] = fma(av_9[1U], q_3[0U][1U], s2_1[9U][0U]);
                            }
                            else
                            {
                                s_2[9U][0U] = fma(av_9[1U], q_3[0U][1U], s_2[9U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[9U][1U] = fma(av_9[1U], q_3[1U][1U], s2_1[9U][1U]);
                            }
                            else
                            {
                                s_2[9U][1U] = fma(av_9[1U], q_3[1U][1U], s_2[9U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[9U] = sumA_2[9U] + av_9[2U];
                            }
                            if(_S96)
                            {
                                s2_1[9U][0U] = fma(av_9[2U], q_3[0U][2U], s2_1[9U][0U]);
                            }
                            else
                            {
                                s_2[9U][0U] = fma(av_9[2U], q_3[0U][2U], s_2[9U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[9U][1U] = fma(av_9[2U], q_3[1U][2U], s2_1[9U][1U]);
                            }
                            else
                            {
                                s_2[9U][1U] = fma(av_9[2U], q_3[1U][2U], s_2[9U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[9U] = sumA_2[9U] + av_9[3U];
                            }
                            if(_S96)
                            {
                                s2_1[9U][0U] = fma(av_9[3U], q_3[0U][3U], s2_1[9U][0U]);
                            }
                            else
                            {
                                s_2[9U][0U] = fma(av_9[3U], q_3[0U][3U], s_2[9U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[9U][1U] = fma(av_9[3U], q_3[1U][3U], s2_1[9U][1U]);
                            }
                            else
                            {
                                s_2[9U][1U] = fma(av_9[3U], q_3[1U][3U], s_2[9U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[9U] = sumA_2[9U] + av_9[4U];
                            }
                            if(_S96)
                            {
                                s2_1[9U][0U] = fma(av_9[4U], q_3[0U][4U], s2_1[9U][0U]);
                            }
                            else
                            {
                                s_2[9U][0U] = fma(av_9[4U], q_3[0U][4U], s_2[9U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[9U][1U] = fma(av_9[4U], q_3[1U][4U], s2_1[9U][1U]);
                            }
                            else
                            {
                                s_2[9U][1U] = fma(av_9[4U], q_3[1U][4U], s_2[9U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[9U] = sumA_2[9U] + av_9[5U];
                            }
                            if(_S96)
                            {
                                s2_1[9U][0U] = fma(av_9[5U], q_3[0U][5U], s2_1[9U][0U]);
                            }
                            else
                            {
                                s_2[9U][0U] = fma(av_9[5U], q_3[0U][5U], s_2[9U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[9U][1U] = fma(av_9[5U], q_3[1U][5U], s2_1[9U][1U]);
                            }
                            else
                            {
                                s_2[9U][1U] = fma(av_9[5U], q_3[1U][5U], s_2[9U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[9U] = sumA_2[9U] + av_9[6U];
                            }
                            if(_S96)
                            {
                                s2_1[9U][0U] = fma(av_9[6U], q_3[0U][6U], s2_1[9U][0U]);
                            }
                            else
                            {
                                s_2[9U][0U] = fma(av_9[6U], q_3[0U][6U], s_2[9U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[9U][1U] = fma(av_9[6U], q_3[1U][6U], s2_1[9U][1U]);
                            }
                            else
                            {
                                s_2[9U][1U] = fma(av_9[6U], q_3[1U][6U], s_2[9U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[9U] = sumA_2[9U] + av_9[7U];
                            }
                            if(_S96)
                            {
                                s2_1[9U][0U] = fma(av_9[7U], q_3[0U][7U], s2_1[9U][0U]);
                            }
                            else
                            {
                                s_2[9U][0U] = fma(av_9[7U], q_3[0U][7U], s_2[9U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[9U][1U] = fma(av_9[7U], q_3[1U][7U], s2_1[9U][1U]);
                            }
                            else
                            {
                                s_2[9U][1U] = fma(av_9[7U], q_3[1U][7U], s_2[9U][1U]);
                            }
                            _S72 = int(4);
                            break;
                        }
                        if(_S72 != int(4))
                        {
                            _S71 = _S72;
                            break;
                        }
                        _S71 = int(3);
                        break;
                    }
                    if(_S71 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(10U >= _S55)
                            {
                                _S70 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_10;
                            load8A_0(_S113 + 10U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_10, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[10U] = sumA_2[10U] + av_10[0U];
                            }
                            if(_S96)
                            {
                                s2_1[10U][0U] = fma(av_10[0U], q_3[0U][0U], s2_1[10U][0U]);
                            }
                            else
                            {
                                s_2[10U][0U] = fma(av_10[0U], q_3[0U][0U], s_2[10U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[10U][1U] = fma(av_10[0U], q_3[1U][0U], s2_1[10U][1U]);
                            }
                            else
                            {
                                s_2[10U][1U] = fma(av_10[0U], q_3[1U][0U], s_2[10U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[10U] = sumA_2[10U] + av_10[1U];
                            }
                            if(_S96)
                            {
                                s2_1[10U][0U] = fma(av_10[1U], q_3[0U][1U], s2_1[10U][0U]);
                            }
                            else
                            {
                                s_2[10U][0U] = fma(av_10[1U], q_3[0U][1U], s_2[10U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[10U][1U] = fma(av_10[1U], q_3[1U][1U], s2_1[10U][1U]);
                            }
                            else
                            {
                                s_2[10U][1U] = fma(av_10[1U], q_3[1U][1U], s_2[10U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[10U] = sumA_2[10U] + av_10[2U];
                            }
                            if(_S96)
                            {
                                s2_1[10U][0U] = fma(av_10[2U], q_3[0U][2U], s2_1[10U][0U]);
                            }
                            else
                            {
                                s_2[10U][0U] = fma(av_10[2U], q_3[0U][2U], s_2[10U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[10U][1U] = fma(av_10[2U], q_3[1U][2U], s2_1[10U][1U]);
                            }
                            else
                            {
                                s_2[10U][1U] = fma(av_10[2U], q_3[1U][2U], s_2[10U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[10U] = sumA_2[10U] + av_10[3U];
                            }
                            if(_S96)
                            {
                                s2_1[10U][0U] = fma(av_10[3U], q_3[0U][3U], s2_1[10U][0U]);
                            }
                            else
                            {
                                s_2[10U][0U] = fma(av_10[3U], q_3[0U][3U], s_2[10U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[10U][1U] = fma(av_10[3U], q_3[1U][3U], s2_1[10U][1U]);
                            }
                            else
                            {
                                s_2[10U][1U] = fma(av_10[3U], q_3[1U][3U], s_2[10U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[10U] = sumA_2[10U] + av_10[4U];
                            }
                            if(_S96)
                            {
                                s2_1[10U][0U] = fma(av_10[4U], q_3[0U][4U], s2_1[10U][0U]);
                            }
                            else
                            {
                                s_2[10U][0U] = fma(av_10[4U], q_3[0U][4U], s_2[10U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[10U][1U] = fma(av_10[4U], q_3[1U][4U], s2_1[10U][1U]);
                            }
                            else
                            {
                                s_2[10U][1U] = fma(av_10[4U], q_3[1U][4U], s_2[10U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[10U] = sumA_2[10U] + av_10[5U];
                            }
                            if(_S96)
                            {
                                s2_1[10U][0U] = fma(av_10[5U], q_3[0U][5U], s2_1[10U][0U]);
                            }
                            else
                            {
                                s_2[10U][0U] = fma(av_10[5U], q_3[0U][5U], s_2[10U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[10U][1U] = fma(av_10[5U], q_3[1U][5U], s2_1[10U][1U]);
                            }
                            else
                            {
                                s_2[10U][1U] = fma(av_10[5U], q_3[1U][5U], s_2[10U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[10U] = sumA_2[10U] + av_10[6U];
                            }
                            if(_S96)
                            {
                                s2_1[10U][0U] = fma(av_10[6U], q_3[0U][6U], s2_1[10U][0U]);
                            }
                            else
                            {
                                s_2[10U][0U] = fma(av_10[6U], q_3[0U][6U], s_2[10U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[10U][1U] = fma(av_10[6U], q_3[1U][6U], s2_1[10U][1U]);
                            }
                            else
                            {
                                s_2[10U][1U] = fma(av_10[6U], q_3[1U][6U], s_2[10U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[10U] = sumA_2[10U] + av_10[7U];
                            }
                            if(_S96)
                            {
                                s2_1[10U][0U] = fma(av_10[7U], q_3[0U][7U], s2_1[10U][0U]);
                            }
                            else
                            {
                                s_2[10U][0U] = fma(av_10[7U], q_3[0U][7U], s_2[10U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[10U][1U] = fma(av_10[7U], q_3[1U][7U], s2_1[10U][1U]);
                            }
                            else
                            {
                                s_2[10U][1U] = fma(av_10[7U], q_3[1U][7U], s_2[10U][1U]);
                            }
                            _S70 = int(4);
                            break;
                        }
                        if(_S70 != int(4))
                        {
                            _S69 = _S70;
                            break;
                        }
                        _S69 = int(3);
                        break;
                    }
                    if(_S69 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(11U >= _S55)
                            {
                                _S68 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_11;
                            load8A_0(_S113 + 11U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_11, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[11U] = sumA_2[11U] + av_11[0U];
                            }
                            if(_S96)
                            {
                                s2_1[11U][0U] = fma(av_11[0U], q_3[0U][0U], s2_1[11U][0U]);
                            }
                            else
                            {
                                s_2[11U][0U] = fma(av_11[0U], q_3[0U][0U], s_2[11U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[11U][1U] = fma(av_11[0U], q_3[1U][0U], s2_1[11U][1U]);
                            }
                            else
                            {
                                s_2[11U][1U] = fma(av_11[0U], q_3[1U][0U], s_2[11U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[11U] = sumA_2[11U] + av_11[1U];
                            }
                            if(_S96)
                            {
                                s2_1[11U][0U] = fma(av_11[1U], q_3[0U][1U], s2_1[11U][0U]);
                            }
                            else
                            {
                                s_2[11U][0U] = fma(av_11[1U], q_3[0U][1U], s_2[11U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[11U][1U] = fma(av_11[1U], q_3[1U][1U], s2_1[11U][1U]);
                            }
                            else
                            {
                                s_2[11U][1U] = fma(av_11[1U], q_3[1U][1U], s_2[11U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[11U] = sumA_2[11U] + av_11[2U];
                            }
                            if(_S96)
                            {
                                s2_1[11U][0U] = fma(av_11[2U], q_3[0U][2U], s2_1[11U][0U]);
                            }
                            else
                            {
                                s_2[11U][0U] = fma(av_11[2U], q_3[0U][2U], s_2[11U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[11U][1U] = fma(av_11[2U], q_3[1U][2U], s2_1[11U][1U]);
                            }
                            else
                            {
                                s_2[11U][1U] = fma(av_11[2U], q_3[1U][2U], s_2[11U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[11U] = sumA_2[11U] + av_11[3U];
                            }
                            if(_S96)
                            {
                                s2_1[11U][0U] = fma(av_11[3U], q_3[0U][3U], s2_1[11U][0U]);
                            }
                            else
                            {
                                s_2[11U][0U] = fma(av_11[3U], q_3[0U][3U], s_2[11U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[11U][1U] = fma(av_11[3U], q_3[1U][3U], s2_1[11U][1U]);
                            }
                            else
                            {
                                s_2[11U][1U] = fma(av_11[3U], q_3[1U][3U], s_2[11U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[11U] = sumA_2[11U] + av_11[4U];
                            }
                            if(_S96)
                            {
                                s2_1[11U][0U] = fma(av_11[4U], q_3[0U][4U], s2_1[11U][0U]);
                            }
                            else
                            {
                                s_2[11U][0U] = fma(av_11[4U], q_3[0U][4U], s_2[11U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[11U][1U] = fma(av_11[4U], q_3[1U][4U], s2_1[11U][1U]);
                            }
                            else
                            {
                                s_2[11U][1U] = fma(av_11[4U], q_3[1U][4U], s_2[11U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[11U] = sumA_2[11U] + av_11[5U];
                            }
                            if(_S96)
                            {
                                s2_1[11U][0U] = fma(av_11[5U], q_3[0U][5U], s2_1[11U][0U]);
                            }
                            else
                            {
                                s_2[11U][0U] = fma(av_11[5U], q_3[0U][5U], s_2[11U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[11U][1U] = fma(av_11[5U], q_3[1U][5U], s2_1[11U][1U]);
                            }
                            else
                            {
                                s_2[11U][1U] = fma(av_11[5U], q_3[1U][5U], s_2[11U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[11U] = sumA_2[11U] + av_11[6U];
                            }
                            if(_S96)
                            {
                                s2_1[11U][0U] = fma(av_11[6U], q_3[0U][6U], s2_1[11U][0U]);
                            }
                            else
                            {
                                s_2[11U][0U] = fma(av_11[6U], q_3[0U][6U], s_2[11U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[11U][1U] = fma(av_11[6U], q_3[1U][6U], s2_1[11U][1U]);
                            }
                            else
                            {
                                s_2[11U][1U] = fma(av_11[6U], q_3[1U][6U], s_2[11U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[11U] = sumA_2[11U] + av_11[7U];
                            }
                            if(_S96)
                            {
                                s2_1[11U][0U] = fma(av_11[7U], q_3[0U][7U], s2_1[11U][0U]);
                            }
                            else
                            {
                                s_2[11U][0U] = fma(av_11[7U], q_3[0U][7U], s_2[11U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[11U][1U] = fma(av_11[7U], q_3[1U][7U], s2_1[11U][1U]);
                            }
                            else
                            {
                                s_2[11U][1U] = fma(av_11[7U], q_3[1U][7U], s_2[11U][1U]);
                            }
                            _S68 = int(4);
                            break;
                        }
                        if(_S68 != int(4))
                        {
                            _S67 = _S68;
                            break;
                        }
                        _S67 = int(3);
                        break;
                    }
                    if(_S67 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(12U >= _S55)
                            {
                                _S66 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_12;
                            load8A_0(_S113 + 12U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_12, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[12U] = sumA_2[12U] + av_12[0U];
                            }
                            if(_S96)
                            {
                                s2_1[12U][0U] = fma(av_12[0U], q_3[0U][0U], s2_1[12U][0U]);
                            }
                            else
                            {
                                s_2[12U][0U] = fma(av_12[0U], q_3[0U][0U], s_2[12U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[12U][1U] = fma(av_12[0U], q_3[1U][0U], s2_1[12U][1U]);
                            }
                            else
                            {
                                s_2[12U][1U] = fma(av_12[0U], q_3[1U][0U], s_2[12U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[12U] = sumA_2[12U] + av_12[1U];
                            }
                            if(_S96)
                            {
                                s2_1[12U][0U] = fma(av_12[1U], q_3[0U][1U], s2_1[12U][0U]);
                            }
                            else
                            {
                                s_2[12U][0U] = fma(av_12[1U], q_3[0U][1U], s_2[12U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[12U][1U] = fma(av_12[1U], q_3[1U][1U], s2_1[12U][1U]);
                            }
                            else
                            {
                                s_2[12U][1U] = fma(av_12[1U], q_3[1U][1U], s_2[12U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[12U] = sumA_2[12U] + av_12[2U];
                            }
                            if(_S96)
                            {
                                s2_1[12U][0U] = fma(av_12[2U], q_3[0U][2U], s2_1[12U][0U]);
                            }
                            else
                            {
                                s_2[12U][0U] = fma(av_12[2U], q_3[0U][2U], s_2[12U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[12U][1U] = fma(av_12[2U], q_3[1U][2U], s2_1[12U][1U]);
                            }
                            else
                            {
                                s_2[12U][1U] = fma(av_12[2U], q_3[1U][2U], s_2[12U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[12U] = sumA_2[12U] + av_12[3U];
                            }
                            if(_S96)
                            {
                                s2_1[12U][0U] = fma(av_12[3U], q_3[0U][3U], s2_1[12U][0U]);
                            }
                            else
                            {
                                s_2[12U][0U] = fma(av_12[3U], q_3[0U][3U], s_2[12U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[12U][1U] = fma(av_12[3U], q_3[1U][3U], s2_1[12U][1U]);
                            }
                            else
                            {
                                s_2[12U][1U] = fma(av_12[3U], q_3[1U][3U], s_2[12U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[12U] = sumA_2[12U] + av_12[4U];
                            }
                            if(_S96)
                            {
                                s2_1[12U][0U] = fma(av_12[4U], q_3[0U][4U], s2_1[12U][0U]);
                            }
                            else
                            {
                                s_2[12U][0U] = fma(av_12[4U], q_3[0U][4U], s_2[12U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[12U][1U] = fma(av_12[4U], q_3[1U][4U], s2_1[12U][1U]);
                            }
                            else
                            {
                                s_2[12U][1U] = fma(av_12[4U], q_3[1U][4U], s_2[12U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[12U] = sumA_2[12U] + av_12[5U];
                            }
                            if(_S96)
                            {
                                s2_1[12U][0U] = fma(av_12[5U], q_3[0U][5U], s2_1[12U][0U]);
                            }
                            else
                            {
                                s_2[12U][0U] = fma(av_12[5U], q_3[0U][5U], s_2[12U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[12U][1U] = fma(av_12[5U], q_3[1U][5U], s2_1[12U][1U]);
                            }
                            else
                            {
                                s_2[12U][1U] = fma(av_12[5U], q_3[1U][5U], s_2[12U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[12U] = sumA_2[12U] + av_12[6U];
                            }
                            if(_S96)
                            {
                                s2_1[12U][0U] = fma(av_12[6U], q_3[0U][6U], s2_1[12U][0U]);
                            }
                            else
                            {
                                s_2[12U][0U] = fma(av_12[6U], q_3[0U][6U], s_2[12U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[12U][1U] = fma(av_12[6U], q_3[1U][6U], s2_1[12U][1U]);
                            }
                            else
                            {
                                s_2[12U][1U] = fma(av_12[6U], q_3[1U][6U], s_2[12U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[12U] = sumA_2[12U] + av_12[7U];
                            }
                            if(_S96)
                            {
                                s2_1[12U][0U] = fma(av_12[7U], q_3[0U][7U], s2_1[12U][0U]);
                            }
                            else
                            {
                                s_2[12U][0U] = fma(av_12[7U], q_3[0U][7U], s_2[12U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[12U][1U] = fma(av_12[7U], q_3[1U][7U], s2_1[12U][1U]);
                            }
                            else
                            {
                                s_2[12U][1U] = fma(av_12[7U], q_3[1U][7U], s_2[12U][1U]);
                            }
                            _S66 = int(4);
                            break;
                        }
                        if(_S66 != int(4))
                        {
                            _S65 = _S66;
                            break;
                        }
                        _S65 = int(3);
                        break;
                    }
                    if(_S65 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(13U >= _S55)
                            {
                                _S64 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_13;
                            load8A_0(_S113 + 13U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_13, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[13U] = sumA_2[13U] + av_13[0U];
                            }
                            if(_S96)
                            {
                                s2_1[13U][0U] = fma(av_13[0U], q_3[0U][0U], s2_1[13U][0U]);
                            }
                            else
                            {
                                s_2[13U][0U] = fma(av_13[0U], q_3[0U][0U], s_2[13U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[13U][1U] = fma(av_13[0U], q_3[1U][0U], s2_1[13U][1U]);
                            }
                            else
                            {
                                s_2[13U][1U] = fma(av_13[0U], q_3[1U][0U], s_2[13U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[13U] = sumA_2[13U] + av_13[1U];
                            }
                            if(_S96)
                            {
                                s2_1[13U][0U] = fma(av_13[1U], q_3[0U][1U], s2_1[13U][0U]);
                            }
                            else
                            {
                                s_2[13U][0U] = fma(av_13[1U], q_3[0U][1U], s_2[13U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[13U][1U] = fma(av_13[1U], q_3[1U][1U], s2_1[13U][1U]);
                            }
                            else
                            {
                                s_2[13U][1U] = fma(av_13[1U], q_3[1U][1U], s_2[13U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[13U] = sumA_2[13U] + av_13[2U];
                            }
                            if(_S96)
                            {
                                s2_1[13U][0U] = fma(av_13[2U], q_3[0U][2U], s2_1[13U][0U]);
                            }
                            else
                            {
                                s_2[13U][0U] = fma(av_13[2U], q_3[0U][2U], s_2[13U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[13U][1U] = fma(av_13[2U], q_3[1U][2U], s2_1[13U][1U]);
                            }
                            else
                            {
                                s_2[13U][1U] = fma(av_13[2U], q_3[1U][2U], s_2[13U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[13U] = sumA_2[13U] + av_13[3U];
                            }
                            if(_S96)
                            {
                                s2_1[13U][0U] = fma(av_13[3U], q_3[0U][3U], s2_1[13U][0U]);
                            }
                            else
                            {
                                s_2[13U][0U] = fma(av_13[3U], q_3[0U][3U], s_2[13U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[13U][1U] = fma(av_13[3U], q_3[1U][3U], s2_1[13U][1U]);
                            }
                            else
                            {
                                s_2[13U][1U] = fma(av_13[3U], q_3[1U][3U], s_2[13U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[13U] = sumA_2[13U] + av_13[4U];
                            }
                            if(_S96)
                            {
                                s2_1[13U][0U] = fma(av_13[4U], q_3[0U][4U], s2_1[13U][0U]);
                            }
                            else
                            {
                                s_2[13U][0U] = fma(av_13[4U], q_3[0U][4U], s_2[13U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[13U][1U] = fma(av_13[4U], q_3[1U][4U], s2_1[13U][1U]);
                            }
                            else
                            {
                                s_2[13U][1U] = fma(av_13[4U], q_3[1U][4U], s_2[13U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[13U] = sumA_2[13U] + av_13[5U];
                            }
                            if(_S96)
                            {
                                s2_1[13U][0U] = fma(av_13[5U], q_3[0U][5U], s2_1[13U][0U]);
                            }
                            else
                            {
                                s_2[13U][0U] = fma(av_13[5U], q_3[0U][5U], s_2[13U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[13U][1U] = fma(av_13[5U], q_3[1U][5U], s2_1[13U][1U]);
                            }
                            else
                            {
                                s_2[13U][1U] = fma(av_13[5U], q_3[1U][5U], s_2[13U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[13U] = sumA_2[13U] + av_13[6U];
                            }
                            if(_S96)
                            {
                                s2_1[13U][0U] = fma(av_13[6U], q_3[0U][6U], s2_1[13U][0U]);
                            }
                            else
                            {
                                s_2[13U][0U] = fma(av_13[6U], q_3[0U][6U], s_2[13U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[13U][1U] = fma(av_13[6U], q_3[1U][6U], s2_1[13U][1U]);
                            }
                            else
                            {
                                s_2[13U][1U] = fma(av_13[6U], q_3[1U][6U], s_2[13U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[13U] = sumA_2[13U] + av_13[7U];
                            }
                            if(_S96)
                            {
                                s2_1[13U][0U] = fma(av_13[7U], q_3[0U][7U], s2_1[13U][0U]);
                            }
                            else
                            {
                                s_2[13U][0U] = fma(av_13[7U], q_3[0U][7U], s_2[13U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[13U][1U] = fma(av_13[7U], q_3[1U][7U], s2_1[13U][1U]);
                            }
                            else
                            {
                                s_2[13U][1U] = fma(av_13[7U], q_3[1U][7U], s_2[13U][1U]);
                            }
                            _S64 = int(4);
                            break;
                        }
                        if(_S64 != int(4))
                        {
                            _S63 = _S64;
                            break;
                        }
                        _S63 = int(3);
                        break;
                    }
                    if(_S63 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(14U >= _S55)
                            {
                                _S62 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_14;
                            load8A_0(_S113 + 14U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_14, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[14U] = sumA_2[14U] + av_14[0U];
                            }
                            if(_S96)
                            {
                                s2_1[14U][0U] = fma(av_14[0U], q_3[0U][0U], s2_1[14U][0U]);
                            }
                            else
                            {
                                s_2[14U][0U] = fma(av_14[0U], q_3[0U][0U], s_2[14U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[14U][1U] = fma(av_14[0U], q_3[1U][0U], s2_1[14U][1U]);
                            }
                            else
                            {
                                s_2[14U][1U] = fma(av_14[0U], q_3[1U][0U], s_2[14U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[14U] = sumA_2[14U] + av_14[1U];
                            }
                            if(_S96)
                            {
                                s2_1[14U][0U] = fma(av_14[1U], q_3[0U][1U], s2_1[14U][0U]);
                            }
                            else
                            {
                                s_2[14U][0U] = fma(av_14[1U], q_3[0U][1U], s_2[14U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[14U][1U] = fma(av_14[1U], q_3[1U][1U], s2_1[14U][1U]);
                            }
                            else
                            {
                                s_2[14U][1U] = fma(av_14[1U], q_3[1U][1U], s_2[14U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[14U] = sumA_2[14U] + av_14[2U];
                            }
                            if(_S96)
                            {
                                s2_1[14U][0U] = fma(av_14[2U], q_3[0U][2U], s2_1[14U][0U]);
                            }
                            else
                            {
                                s_2[14U][0U] = fma(av_14[2U], q_3[0U][2U], s_2[14U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[14U][1U] = fma(av_14[2U], q_3[1U][2U], s2_1[14U][1U]);
                            }
                            else
                            {
                                s_2[14U][1U] = fma(av_14[2U], q_3[1U][2U], s_2[14U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[14U] = sumA_2[14U] + av_14[3U];
                            }
                            if(_S96)
                            {
                                s2_1[14U][0U] = fma(av_14[3U], q_3[0U][3U], s2_1[14U][0U]);
                            }
                            else
                            {
                                s_2[14U][0U] = fma(av_14[3U], q_3[0U][3U], s_2[14U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[14U][1U] = fma(av_14[3U], q_3[1U][3U], s2_1[14U][1U]);
                            }
                            else
                            {
                                s_2[14U][1U] = fma(av_14[3U], q_3[1U][3U], s_2[14U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[14U] = sumA_2[14U] + av_14[4U];
                            }
                            if(_S96)
                            {
                                s2_1[14U][0U] = fma(av_14[4U], q_3[0U][4U], s2_1[14U][0U]);
                            }
                            else
                            {
                                s_2[14U][0U] = fma(av_14[4U], q_3[0U][4U], s_2[14U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[14U][1U] = fma(av_14[4U], q_3[1U][4U], s2_1[14U][1U]);
                            }
                            else
                            {
                                s_2[14U][1U] = fma(av_14[4U], q_3[1U][4U], s_2[14U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[14U] = sumA_2[14U] + av_14[5U];
                            }
                            if(_S96)
                            {
                                s2_1[14U][0U] = fma(av_14[5U], q_3[0U][5U], s2_1[14U][0U]);
                            }
                            else
                            {
                                s_2[14U][0U] = fma(av_14[5U], q_3[0U][5U], s_2[14U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[14U][1U] = fma(av_14[5U], q_3[1U][5U], s2_1[14U][1U]);
                            }
                            else
                            {
                                s_2[14U][1U] = fma(av_14[5U], q_3[1U][5U], s_2[14U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[14U] = sumA_2[14U] + av_14[6U];
                            }
                            if(_S96)
                            {
                                s2_1[14U][0U] = fma(av_14[6U], q_3[0U][6U], s2_1[14U][0U]);
                            }
                            else
                            {
                                s_2[14U][0U] = fma(av_14[6U], q_3[0U][6U], s_2[14U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[14U][1U] = fma(av_14[6U], q_3[1U][6U], s2_1[14U][1U]);
                            }
                            else
                            {
                                s_2[14U][1U] = fma(av_14[6U], q_3[1U][6U], s_2[14U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[14U] = sumA_2[14U] + av_14[7U];
                            }
                            if(_S96)
                            {
                                s2_1[14U][0U] = fma(av_14[7U], q_3[0U][7U], s2_1[14U][0U]);
                            }
                            else
                            {
                                s_2[14U][0U] = fma(av_14[7U], q_3[0U][7U], s_2[14U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[14U][1U] = fma(av_14[7U], q_3[1U][7U], s2_1[14U][1U]);
                            }
                            else
                            {
                                s_2[14U][1U] = fma(av_14[7U], q_3[1U][7U], s_2[14U][1U]);
                            }
                            _S62 = int(4);
                            break;
                        }
                        if(_S62 != int(4))
                        {
                            _S61 = _S62;
                            break;
                        }
                        _S61 = int(3);
                        break;
                    }
                    if(_S61 != int(3))
                    {
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(15U >= _S55)
                            {
                                _S60 = int(2);
                                break;
                            }
                            thread array<float, int(8)> av_15;
                            load8A_0(_S113 + 15U * (&kernelContext_14)->p_0->lda_1 + _S95, &av_15, &kernelContext_14);
                            if(QFORMAT_0 == 2U)
                            {
                                _S112 = true;
                            }
                            else
                            {
                                _S112 = QFORMAT_0 == 3U;
                            }
                            if(_S112)
                            {
                                _S111 = true;
                            }
                            else
                            {
                                _S111 = QFORMAT_0 == 4U;
                            }
                            if(_S111)
                            {
                                sumA_2[15U] = sumA_2[15U] + av_15[0U];
                            }
                            if(_S96)
                            {
                                s2_1[15U][0U] = fma(av_15[0U], q_3[0U][0U], s2_1[15U][0U]);
                            }
                            else
                            {
                                s_2[15U][0U] = fma(av_15[0U], q_3[0U][0U], s_2[15U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[15U][1U] = fma(av_15[0U], q_3[1U][0U], s2_1[15U][1U]);
                            }
                            else
                            {
                                s_2[15U][1U] = fma(av_15[0U], q_3[1U][0U], s_2[15U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S110 = true;
                            }
                            else
                            {
                                _S110 = QFORMAT_0 == 3U;
                            }
                            if(_S110)
                            {
                                _S109 = true;
                            }
                            else
                            {
                                _S109 = QFORMAT_0 == 4U;
                            }
                            if(_S109)
                            {
                                sumA_2[15U] = sumA_2[15U] + av_15[1U];
                            }
                            if(_S96)
                            {
                                s2_1[15U][0U] = fma(av_15[1U], q_3[0U][1U], s2_1[15U][0U]);
                            }
                            else
                            {
                                s_2[15U][0U] = fma(av_15[1U], q_3[0U][1U], s_2[15U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[15U][1U] = fma(av_15[1U], q_3[1U][1U], s2_1[15U][1U]);
                            }
                            else
                            {
                                s_2[15U][1U] = fma(av_15[1U], q_3[1U][1U], s_2[15U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S108 = true;
                            }
                            else
                            {
                                _S108 = QFORMAT_0 == 3U;
                            }
                            if(_S108)
                            {
                                _S107 = true;
                            }
                            else
                            {
                                _S107 = QFORMAT_0 == 4U;
                            }
                            if(_S107)
                            {
                                sumA_2[15U] = sumA_2[15U] + av_15[2U];
                            }
                            if(_S96)
                            {
                                s2_1[15U][0U] = fma(av_15[2U], q_3[0U][2U], s2_1[15U][0U]);
                            }
                            else
                            {
                                s_2[15U][0U] = fma(av_15[2U], q_3[0U][2U], s_2[15U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[15U][1U] = fma(av_15[2U], q_3[1U][2U], s2_1[15U][1U]);
                            }
                            else
                            {
                                s_2[15U][1U] = fma(av_15[2U], q_3[1U][2U], s_2[15U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S106 = true;
                            }
                            else
                            {
                                _S106 = QFORMAT_0 == 3U;
                            }
                            if(_S106)
                            {
                                _S105 = true;
                            }
                            else
                            {
                                _S105 = QFORMAT_0 == 4U;
                            }
                            if(_S105)
                            {
                                sumA_2[15U] = sumA_2[15U] + av_15[3U];
                            }
                            if(_S96)
                            {
                                s2_1[15U][0U] = fma(av_15[3U], q_3[0U][3U], s2_1[15U][0U]);
                            }
                            else
                            {
                                s_2[15U][0U] = fma(av_15[3U], q_3[0U][3U], s_2[15U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[15U][1U] = fma(av_15[3U], q_3[1U][3U], s2_1[15U][1U]);
                            }
                            else
                            {
                                s_2[15U][1U] = fma(av_15[3U], q_3[1U][3U], s_2[15U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S104 = true;
                            }
                            else
                            {
                                _S104 = QFORMAT_0 == 3U;
                            }
                            if(_S104)
                            {
                                _S103 = true;
                            }
                            else
                            {
                                _S103 = QFORMAT_0 == 4U;
                            }
                            if(_S103)
                            {
                                sumA_2[15U] = sumA_2[15U] + av_15[4U];
                            }
                            if(_S96)
                            {
                                s2_1[15U][0U] = fma(av_15[4U], q_3[0U][4U], s2_1[15U][0U]);
                            }
                            else
                            {
                                s_2[15U][0U] = fma(av_15[4U], q_3[0U][4U], s_2[15U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[15U][1U] = fma(av_15[4U], q_3[1U][4U], s2_1[15U][1U]);
                            }
                            else
                            {
                                s_2[15U][1U] = fma(av_15[4U], q_3[1U][4U], s_2[15U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S102 = true;
                            }
                            else
                            {
                                _S102 = QFORMAT_0 == 3U;
                            }
                            if(_S102)
                            {
                                _S101 = true;
                            }
                            else
                            {
                                _S101 = QFORMAT_0 == 4U;
                            }
                            if(_S101)
                            {
                                sumA_2[15U] = sumA_2[15U] + av_15[5U];
                            }
                            if(_S96)
                            {
                                s2_1[15U][0U] = fma(av_15[5U], q_3[0U][5U], s2_1[15U][0U]);
                            }
                            else
                            {
                                s_2[15U][0U] = fma(av_15[5U], q_3[0U][5U], s_2[15U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[15U][1U] = fma(av_15[5U], q_3[1U][5U], s2_1[15U][1U]);
                            }
                            else
                            {
                                s_2[15U][1U] = fma(av_15[5U], q_3[1U][5U], s_2[15U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S100 = true;
                            }
                            else
                            {
                                _S100 = QFORMAT_0 == 3U;
                            }
                            if(_S100)
                            {
                                _S99 = true;
                            }
                            else
                            {
                                _S99 = QFORMAT_0 == 4U;
                            }
                            if(_S99)
                            {
                                sumA_2[15U] = sumA_2[15U] + av_15[6U];
                            }
                            if(_S96)
                            {
                                s2_1[15U][0U] = fma(av_15[6U], q_3[0U][6U], s2_1[15U][0U]);
                            }
                            else
                            {
                                s_2[15U][0U] = fma(av_15[6U], q_3[0U][6U], s_2[15U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[15U][1U] = fma(av_15[6U], q_3[1U][6U], s2_1[15U][1U]);
                            }
                            else
                            {
                                s_2[15U][1U] = fma(av_15[6U], q_3[1U][6U], s_2[15U][1U]);
                            }
                            if(QFORMAT_0 == 2U)
                            {
                                _S98 = true;
                            }
                            else
                            {
                                _S98 = QFORMAT_0 == 3U;
                            }
                            if(_S98)
                            {
                                _S97 = true;
                            }
                            else
                            {
                                _S97 = QFORMAT_0 == 4U;
                            }
                            if(_S97)
                            {
                                sumA_2[15U] = sumA_2[15U] + av_15[7U];
                            }
                            if(_S96)
                            {
                                s2_1[15U][0U] = fma(av_15[7U], q_3[0U][7U], s2_1[15U][0U]);
                            }
                            else
                            {
                                s_2[15U][0U] = fma(av_15[7U], q_3[0U][7U], s_2[15U][0U]);
                            }
                            if(_S96)
                            {
                                s2_1[15U][1U] = fma(av_15[7U], q_3[1U][7U], s2_1[15U][1U]);
                            }
                            else
                            {
                                s_2[15U][1U] = fma(av_15[7U], q_3[1U][7U], s_2[15U][1U]);
                            }
                            _S60 = int(4);
                            break;
                        }
                        if(_S60 != int(4))
                        {
                            _S59 = _S60;
                            break;
                        }
                        _S59 = int(3);
                        break;
                    }
                    if(_S59 != int(3))
                    {
                        break;
                    }
                    break;
                }
                w_11 = w_11 + 8U;
            }
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        if(0U >= _S55)
                        {
                            _S90 = int(1);
                            break;
                        }
                        acc_2[0U][0U] = addTerms_0(acc_2[0U][0U], s_2[0U][0U], s2_1[0U][0U], sumA_2[0U], sc_4[0U]);
                        acc_2[0U][1U] = addTerms_0(acc_2[0U][1U], s_2[0U][1U], s2_1[0U][1U], sumA_2[0U], sc_4[1U]);
                        _S90 = int(3);
                        break;
                    }
                    if(_S90 != int(3))
                    {
                        _S89 = _S90;
                        break;
                    }
                    _S89 = int(2);
                    break;
                }
                if(_S89 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(1U >= _S55)
                        {
                            _S88 = int(1);
                            break;
                        }
                        acc_2[1U][0U] = addTerms_0(acc_2[1U][0U], s_2[1U][0U], s2_1[1U][0U], sumA_2[1U], sc_4[0U]);
                        acc_2[1U][1U] = addTerms_0(acc_2[1U][1U], s_2[1U][1U], s2_1[1U][1U], sumA_2[1U], sc_4[1U]);
                        _S88 = int(3);
                        break;
                    }
                    if(_S88 != int(3))
                    {
                        _S87 = _S88;
                        break;
                    }
                    _S87 = int(2);
                    break;
                }
                if(_S87 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(2U >= _S55)
                        {
                            _S86 = int(1);
                            break;
                        }
                        acc_2[2U][0U] = addTerms_0(acc_2[2U][0U], s_2[2U][0U], s2_1[2U][0U], sumA_2[2U], sc_4[0U]);
                        acc_2[2U][1U] = addTerms_0(acc_2[2U][1U], s_2[2U][1U], s2_1[2U][1U], sumA_2[2U], sc_4[1U]);
                        _S86 = int(3);
                        break;
                    }
                    if(_S86 != int(3))
                    {
                        _S85 = _S86;
                        break;
                    }
                    _S85 = int(2);
                    break;
                }
                if(_S85 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(3U >= _S55)
                        {
                            _S84 = int(1);
                            break;
                        }
                        acc_2[3U][0U] = addTerms_0(acc_2[3U][0U], s_2[3U][0U], s2_1[3U][0U], sumA_2[3U], sc_4[0U]);
                        acc_2[3U][1U] = addTerms_0(acc_2[3U][1U], s_2[3U][1U], s2_1[3U][1U], sumA_2[3U], sc_4[1U]);
                        _S84 = int(3);
                        break;
                    }
                    if(_S84 != int(3))
                    {
                        _S83 = _S84;
                        break;
                    }
                    _S83 = int(2);
                    break;
                }
                if(_S83 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(4U >= _S55)
                        {
                            _S82 = int(1);
                            break;
                        }
                        acc_2[4U][0U] = addTerms_0(acc_2[4U][0U], s_2[4U][0U], s2_1[4U][0U], sumA_2[4U], sc_4[0U]);
                        acc_2[4U][1U] = addTerms_0(acc_2[4U][1U], s_2[4U][1U], s2_1[4U][1U], sumA_2[4U], sc_4[1U]);
                        _S82 = int(3);
                        break;
                    }
                    if(_S82 != int(3))
                    {
                        _S81 = _S82;
                        break;
                    }
                    _S81 = int(2);
                    break;
                }
                if(_S81 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(5U >= _S55)
                        {
                            _S80 = int(1);
                            break;
                        }
                        acc_2[5U][0U] = addTerms_0(acc_2[5U][0U], s_2[5U][0U], s2_1[5U][0U], sumA_2[5U], sc_4[0U]);
                        acc_2[5U][1U] = addTerms_0(acc_2[5U][1U], s_2[5U][1U], s2_1[5U][1U], sumA_2[5U], sc_4[1U]);
                        _S80 = int(3);
                        break;
                    }
                    if(_S80 != int(3))
                    {
                        _S79 = _S80;
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
                        if(6U >= _S55)
                        {
                            _S78 = int(1);
                            break;
                        }
                        acc_2[6U][0U] = addTerms_0(acc_2[6U][0U], s_2[6U][0U], s2_1[6U][0U], sumA_2[6U], sc_4[0U]);
                        acc_2[6U][1U] = addTerms_0(acc_2[6U][1U], s_2[6U][1U], s2_1[6U][1U], sumA_2[6U], sc_4[1U]);
                        _S78 = int(3);
                        break;
                    }
                    if(_S78 != int(3))
                    {
                        _S77 = _S78;
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
                        if(7U >= _S55)
                        {
                            _S76 = int(1);
                            break;
                        }
                        acc_2[7U][0U] = addTerms_0(acc_2[7U][0U], s_2[7U][0U], s2_1[7U][0U], sumA_2[7U], sc_4[0U]);
                        acc_2[7U][1U] = addTerms_0(acc_2[7U][1U], s_2[7U][1U], s2_1[7U][1U], sumA_2[7U], sc_4[1U]);
                        _S76 = int(3);
                        break;
                    }
                    if(_S76 != int(3))
                    {
                        _S75 = _S76;
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
                        if(8U >= _S55)
                        {
                            _S74 = int(1);
                            break;
                        }
                        acc_2[8U][0U] = addTerms_0(acc_2[8U][0U], s_2[8U][0U], s2_1[8U][0U], sumA_2[8U], sc_4[0U]);
                        acc_2[8U][1U] = addTerms_0(acc_2[8U][1U], s_2[8U][1U], s2_1[8U][1U], sumA_2[8U], sc_4[1U]);
                        _S74 = int(3);
                        break;
                    }
                    if(_S74 != int(3))
                    {
                        _S73 = _S74;
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
                        if(9U >= _S55)
                        {
                            _S72 = int(1);
                            break;
                        }
                        acc_2[9U][0U] = addTerms_0(acc_2[9U][0U], s_2[9U][0U], s2_1[9U][0U], sumA_2[9U], sc_4[0U]);
                        acc_2[9U][1U] = addTerms_0(acc_2[9U][1U], s_2[9U][1U], s2_1[9U][1U], sumA_2[9U], sc_4[1U]);
                        _S72 = int(3);
                        break;
                    }
                    if(_S72 != int(3))
                    {
                        _S71 = _S72;
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
                        if(10U >= _S55)
                        {
                            _S70 = int(1);
                            break;
                        }
                        acc_2[10U][0U] = addTerms_0(acc_2[10U][0U], s_2[10U][0U], s2_1[10U][0U], sumA_2[10U], sc_4[0U]);
                        acc_2[10U][1U] = addTerms_0(acc_2[10U][1U], s_2[10U][1U], s2_1[10U][1U], sumA_2[10U], sc_4[1U]);
                        _S70 = int(3);
                        break;
                    }
                    if(_S70 != int(3))
                    {
                        _S69 = _S70;
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
                        if(11U >= _S55)
                        {
                            _S68 = int(1);
                            break;
                        }
                        acc_2[11U][0U] = addTerms_0(acc_2[11U][0U], s_2[11U][0U], s2_1[11U][0U], sumA_2[11U], sc_4[0U]);
                        acc_2[11U][1U] = addTerms_0(acc_2[11U][1U], s_2[11U][1U], s2_1[11U][1U], sumA_2[11U], sc_4[1U]);
                        _S68 = int(3);
                        break;
                    }
                    if(_S68 != int(3))
                    {
                        _S67 = _S68;
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
                        if(12U >= _S55)
                        {
                            _S66 = int(1);
                            break;
                        }
                        acc_2[12U][0U] = addTerms_0(acc_2[12U][0U], s_2[12U][0U], s2_1[12U][0U], sumA_2[12U], sc_4[0U]);
                        acc_2[12U][1U] = addTerms_0(acc_2[12U][1U], s_2[12U][1U], s2_1[12U][1U], sumA_2[12U], sc_4[1U]);
                        _S66 = int(3);
                        break;
                    }
                    if(_S66 != int(3))
                    {
                        _S65 = _S66;
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
                        if(13U >= _S55)
                        {
                            _S64 = int(1);
                            break;
                        }
                        acc_2[13U][0U] = addTerms_0(acc_2[13U][0U], s_2[13U][0U], s2_1[13U][0U], sumA_2[13U], sc_4[0U]);
                        acc_2[13U][1U] = addTerms_0(acc_2[13U][1U], s_2[13U][1U], s2_1[13U][1U], sumA_2[13U], sc_4[1U]);
                        _S64 = int(3);
                        break;
                    }
                    if(_S64 != int(3))
                    {
                        _S63 = _S64;
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
                        if(14U >= _S55)
                        {
                            _S62 = int(1);
                            break;
                        }
                        acc_2[14U][0U] = addTerms_0(acc_2[14U][0U], s_2[14U][0U], s2_1[14U][0U], sumA_2[14U], sc_4[0U]);
                        acc_2[14U][1U] = addTerms_0(acc_2[14U][1U], s_2[14U][1U], s2_1[14U][1U], sumA_2[14U], sc_4[1U]);
                        _S62 = int(3);
                        break;
                    }
                    if(_S62 != int(3))
                    {
                        _S61 = _S62;
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
                        if(15U >= _S55)
                        {
                            _S60 = int(1);
                            break;
                        }
                        acc_2[15U][0U] = addTerms_0(acc_2[15U][0U], s_2[15U][0U], s2_1[15U][0U], sumA_2[15U], sc_4[0U]);
                        acc_2[15U][1U] = addTerms_0(acc_2[15U][1U], s_2[15U][1U], s2_1[15U][1U], sumA_2[15U], sc_4[1U]);
                        _S60 = int(3);
                        break;
                    }
                    if(_S60 != int(3))
                    {
                        _S59 = _S60;
                        break;
                    }
                    _S59 = int(2);
                    break;
                }
                if(_S59 != int(2))
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
        if(i_2 < _S55)
        {
        }
        else
        {
            break;
        }
        (*(&kernelContext_14)->red_0)[0U][lid_0] = acc_2[i_2][0U];
        (*(&kernelContext_14)->red_0)[1U][lid_0] = acc_2[i_2][1U];
        w_11 = 16U;
        for(;;)
        {
            if(w_11 > 0U)
            {
            }
            else
            {
                break;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane_0 < w_11)
            {
                uint _S114 = lid_0 + w_11;
                (*(&kernelContext_14)->red_0)[0U][lid_0] = (*(&kernelContext_14)->red_0)[0U][lid_0] + (*(&kernelContext_14)->red_0)[0U][_S114];
                (*(&kernelContext_14)->red_0)[1U][lid_0] = (*(&kernelContext_14)->red_0)[1U][lid_0] + (*(&kernelContext_14)->red_0)[1U][_S114];
            }
            w_11 = w_11 / 2U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane_0 == 0U)
        {
            if(col0_0 < ((&kernelContext_14)->p_0->n_1))
            {
                float _S115 = (*(&kernelContext_14)->red_0)[0U][lid_0];
                thread GemmProblem_0 _S116 = _S54;
                gemmStore_0(&_S116, i_2, col0_0, _S115, &kernelContext_14);
            }
            uint _S117 = col0_0 + 1U;
            if(_S117 < ((&kernelContext_14)->p_0->n_1))
            {
                float _S118 = (*(&kernelContext_14)->red_0)[1U][lid_0];
                thread GemmProblem_0 _S119 = _S54;
                gemmStore_0(&_S119, i_2, _S117, _S118, &kernelContext_14);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_2 = i_2 + 1U;
    }
    return;
}

