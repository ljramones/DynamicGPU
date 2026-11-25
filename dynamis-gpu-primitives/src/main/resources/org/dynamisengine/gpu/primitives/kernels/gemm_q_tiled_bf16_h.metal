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
    uint device* a_0;
    uint device* b_0;
    half device* c_0;
    half device* epBias_0;
    half device* epGate_0;
    half device* epResidual_0;
    array<array<uint, int(32)>, int(32)> threadgroup* qTileA_0;
    array<array<float, int(64)>, int(32)> threadgroup* qTileB_0;
    array<array<float2, int(64)>, int(2)> threadgroup* qScale_0;
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

uint rawA_0(uint e_0, KernelContext_0 thread* kernelContext_1)
{
    return ((*(kernelContext_1->a_0+(e_0 >> 1U))) >> (16U * (e_0 & 1U))) & 65535U;
}

uint wordB_0(uint w_0, KernelContext_0 thread* kernelContext_2)
{
    return *(kernelContext_2->b_0+w_0);
}

uint qWord_0(uint w_1, KernelContext_0 thread* kernelContext_3)
{
    return *(kernelContext_3->b_0+w_1);
}

uint kHigh8_0(uint format_0, uint qhigh_1, uint e_1, KernelContext_0 thread* kernelContext_4)
{
    if(format_0 == 4U)
    {
        uint _S1 = qWord_0(qhigh_1 + (e_1 >> 5U), kernelContext_4);
        return _S1 >> (e_1 & 31U);
    }
    uint _S2 = qWord_0(qhigh_1 + (e_1 >> 4U), kernelContext_4);
    return _S2 >> (2U * (e_1 & 15U));
}

int kCode_0(uint format_1, uint u_0, uint h_0)
{
    if(format_1 == 3U)
    {
        return int(u_0);
    }
    if(format_1 == 4U)
    {
        return int(u_0 | ((h_0 & 1U) << 4U));
    }
    return int(u_0 | ((h_0 & 3U) << 4U)) - int(32);
}

int qCode_0(uint format_2, uint word_0, uint col_0)
{
    int _S3;
    if(format_2 == 0U)
    {
        uint b_1 = (word_0 >> (8U * (col_0 & 3U))) & 255U;
        int _S4 = int(b_1);
        if(b_1 >= 128U)
        {
            _S3 = int(256);
        }
        else
        {
            _S3 = int(0);
        }
        return _S4 - _S3;
    }
    int u_1 = int((word_0 >> (4U * (col_0 & 7U))) & 15U);
    if(format_2 == 1U)
    {
        _S3 = u_1 - int(8);
    }
    else
    {
        _S3 = u_1;
    }
    return _S3;
}

void load8Codes_0(uint r_0, uint c0_0, array<float, int(8)> thread* q_1, KernelContext_0 thread* kernelContext_5)
{
    if(QFORMAT_0 >= 3U)
    {
        uint _S5 = wordB_0((r_0 * kernelContext_5->p_0->k_1 + c0_0) >> 3U, kernelContext_5);
        uint _S6;
        if(QFORMAT_0 == 3U)
        {
            _S6 = 0U;
        }
        else
        {
            uint _S7 = kHigh8_0(QFORMAT_0, kernelContext_5->p_0->qhigh_0, r_0 * kernelContext_5->p_0->k_1 + c0_0, kernelContext_5);
            _S6 = _S7;
        }
        uint _S8;
        if(QFORMAT_0 == 4U)
        {
            _S8 = 1U;
        }
        else
        {
            _S8 = 2U;
        }
        (*q_1)[0U] = float(kCode_0(QFORMAT_0, (_S5 >> 0U) & 15U, _S6 >> 0U));
        (*q_1)[1U] = float(kCode_0(QFORMAT_0, (_S5 >> 4U) & 15U, _S6 >> _S8));
        (*q_1)[2U] = float(kCode_0(QFORMAT_0, (_S5 >> 8U) & 15U, _S6 >> (_S8 * 2U)));
        (*q_1)[3U] = float(kCode_0(QFORMAT_0, (_S5 >> 12U) & 15U, _S6 >> (_S8 * 3U)));
        (*q_1)[4U] = float(kCode_0(QFORMAT_0, (_S5 >> 16U) & 15U, _S6 >> (_S8 * 4U)));
        (*q_1)[5U] = float(kCode_0(QFORMAT_0, (_S5 >> 20U) & 15U, _S6 >> (_S8 * 5U)));
        (*q_1)[6U] = float(kCode_0(QFORMAT_0, (_S5 >> 24U) & 15U, _S6 >> (_S8 * 6U)));
        (*q_1)[7U] = float(kCode_0(QFORMAT_0, (_S5 >> 28U) & 15U, _S6 >> (_S8 * 7U)));
    }
    else
    {
        if(QFORMAT_0 == 0U)
        {
            uint w_2 = (r_0 * kernelContext_5->p_0->k_1 + c0_0) >> 2U;
            uint _S9 = wordB_0(w_2, kernelContext_5);
            uint _S10 = wordB_0(w_2 + 1U, kernelContext_5);
            (*q_1)[0U] = float(qCode_0(QFORMAT_0, _S9, 0U));
            (*q_1)[4U] = float(qCode_0(QFORMAT_0, _S10, 0U));
            (*q_1)[1U] = float(qCode_0(QFORMAT_0, _S9, 1U));
            (*q_1)[5U] = float(qCode_0(QFORMAT_0, _S10, 1U));
            (*q_1)[2U] = float(qCode_0(QFORMAT_0, _S9, 2U));
            (*q_1)[6U] = float(qCode_0(QFORMAT_0, _S10, 2U));
            (*q_1)[3U] = float(qCode_0(QFORMAT_0, _S9, 3U));
            (*q_1)[7U] = float(qCode_0(QFORMAT_0, _S10, 3U));
        }
        else
        {
            uint _S11 = wordB_0((r_0 * kernelContext_5->p_0->k_1 + c0_0) >> 3U, kernelContext_5);
            (*q_1)[0U] = float(qCode_0(QFORMAT_0, _S11, 0U));
            (*q_1)[1U] = float(qCode_0(QFORMAT_0, _S11, 1U));
            (*q_1)[2U] = float(qCode_0(QFORMAT_0, _S11, 2U));
            (*q_1)[3U] = float(qCode_0(QFORMAT_0, _S11, 3U));
            (*q_1)[4U] = float(qCode_0(QFORMAT_0, _S11, 4U));
            (*q_1)[5U] = float(qCode_0(QFORMAT_0, _S11, 5U));
            (*q_1)[6U] = float(qCode_0(QFORMAT_0, _S11, 6U));
            (*q_1)[7U] = float(qCode_0(QFORMAT_0, _S11, 7U));
        }
    }
    return;
}

constant uint fc_QBLOCK_0 [[function_constant(2)]];
constant uint QBLOCK_0 = is_function_constant_defined(fc_QBLOCK_0) ? fc_QBLOCK_0 : 32U;
float qF16ToF32_0(uint h_1)
{
    return (as_type<half>((ushort)((h_1 & 65535U))));
}

float2 kScale6Pair_0(uint qparams_1, uint r_1, uint col_1, uint cols_0, KernelContext_0 thread* kernelContext_6)
{
    uint base_0 = qparams_1 + (r_1 * (cols_0 / 256U) + col_1 / 256U) * 5U;
    uint _S12 = qWord_0(base_0, kernelContext_6);
    float d_0 = qF16ToF32_0(_S12);
    uint j_0 = col_1 % 256U / 16U;
    uint _S13 = qWord_0(base_0 + 1U + j_0 / 4U, kernelContext_6);
    uint w_3 = _S13 >> (8U * (j_0 % 4U));
    int s0_0 = int(w_3 & 255U);
    int s1_0 = int((w_3 >> 8U) & 255U);
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

uint qByte_0(uint at_0, KernelContext_0 thread* kernelContext_7)
{
    uint _S14 = qWord_0(at_0 >> 2U, kernelContext_7);
    return (_S14 >> (8U * (at_0 & 3U))) & 255U;
}

float2 kScale_0(uint format_3, uint qparams_2, uint r_2, uint col_2, uint cols_1, KernelContext_0 thread* kernelContext_8)
{
    uint sb_0 = r_2 * (cols_1 / 256U) + col_2 / 256U;
    bool _S15 = format_3 == 5U;
    uint sc_0;
    if(_S15)
    {
        sc_0 = 5U;
    }
    else
    {
        sc_0 = 4U;
    }
    uint base_1 = qparams_2 + sb_0 * sc_0;
    uint _S16 = qWord_0(base_1, kernelContext_8);
    float d_1 = qF16ToF32_0(_S16);
    uint at_1 = 4U * (base_1 + 1U);
    if(_S15)
    {
        uint _S17 = qByte_0(at_1 + col_2 % 256U / 16U, kernelContext_8);
        int sc_1 = int(_S17);
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
    uint j_1 = col_2 % 256U / 32U;
    uint _S18 = at_1 + j_1;
    uint _S19 = qByte_0(_S18, kernelContext_8);
    uint _S20 = qByte_0(_S18 + 4U, kernelContext_8);
    uint _S21 = qByte_0(at_1 + (j_1 ^ 4U), kernelContext_8);
    bool low_0 = j_1 < 4U;
    if(low_0)
    {
        sc_0 = _S19 & 63U;
    }
    else
    {
        sc_0 = (_S20 & 15U) | ((_S21 >> 6U) << 4U);
    }
    uint mn_0;
    if(low_0)
    {
        mn_0 = _S20 & 63U;
    }
    else
    {
        mn_0 = (_S20 >> 4U) | ((_S19 >> 6U) << 4U);
    }
    return float2(d_1 * float(sc_0), qF16ToF32_0(_S16 >> 16U) * float(mn_0));
}

uint qParamHalf_0(uint format_4, uint b_2, uint t_0)
{
    uint _S22;
    if(format_4 == 2U)
    {
        _S22 = 2U * b_2 + t_0;
    }
    else
    {
        _S22 = b_2;
    }
    return _S22;
}

float blockParameter_0(uint r_3, uint bl_0, uint t_1, KernelContext_0 thread* kernelContext_9)
{
    uint h_2 = qParamHalf_0(QFORMAT_0, r_3 * (kernelContext_9->p_0->k_1 / QBLOCK_0) + bl_0, t_1);
    uint _S23 = wordB_0(kernelContext_9->p_0->qparams_0 + (h_2 >> 1U), kernelContext_9);
    return qF16ToF32_0(_S23 >> (16U * (h_2 & 1U)));
}

float2 unitScale_0(uint r_4, uint u_2, KernelContext_0 thread* kernelContext_10)
{
    uint _S24;
    if(QFORMAT_0 == 5U)
    {
        uint _S25 = kernelContext_10->p_0->qparams_0;
        if(QFORMAT_0 >= 3U)
        {
            _S24 = 32U;
        }
        else
        {
            _S24 = QBLOCK_0;
        }
        float2 _S26 = kScale6Pair_0(_S25, r_4, u_2 * _S24, kernelContext_10->p_0->k_1, kernelContext_10);
        return _S26;
    }
    if(QFORMAT_0 >= 3U)
    {
        uint _S27 = kernelContext_10->p_0->qparams_0;
        if(QFORMAT_0 >= 3U)
        {
            _S24 = 32U;
        }
        else
        {
            _S24 = QBLOCK_0;
        }
        float2 _S28 = kScale_0(QFORMAT_0, _S27, r_4, u_2 * _S24, kernelContext_10->p_0->k_1, kernelContext_10);
        return _S28;
    }
    float _S29 = blockParameter_0(r_4, u_2, 0U, kernelContext_10);
    float _S30;
    if(QFORMAT_0 == 2U)
    {
        float _S31 = blockParameter_0(r_4, u_2, 1U, kernelContext_10);
        _S30 = _S31;
    }
    else
    {
        _S30 = 0.0;
    }
    return float2(_S29, _S30);
}

float aLo_0(uint w_4)
{
    return (as_type<float>((w_4 << 16U)));
}

float aHi_0(uint w_5)
{
    return (as_type<float>((w_5 & 4294901760U)));
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
        bool _S32;
        if(QFORMAT_0 == 3U)
        {
            _S32 = true;
        }
        else
        {
            _S32 = QFORMAT_0 == 4U;
        }
        if(_S32)
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
    uint _S33 = (as_type<uint>((x_0)));
    bool _S34;
    if((_S33 & 2139095040U) == 2139095040U)
    {
        _S34 = (_S33 & 8388607U) != 0U;
    }
    else
    {
        _S34 = false;
    }
    return _S34;
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
    bool _S35;
    if(a_1 == 0.0)
    {
        _S35 = b_3 == 0.0;
    }
    else
    {
        _S35 = false;
    }
    if(_S35)
    {
        return (as_type<float>(((as_type<uint>((a_1))) & (as_type<uint>((b_3))))));
    }
    float _S36;
    if(a_1 > b_3)
    {
        _S36 = a_1;
    }
    else
    {
        _S36 = b_3;
    }
    return _S36;
}

float gemmEpilogue_0(float r_5, uint i_0, uint j_2, uint idx_0, uint biasBase_0, KernelContext_0 thread* kernelContext_11)
{
    float _S37;
    if((kernelContext_11->p_0->bias_0) == 1U)
    {
        _S37 = r_5 + float(*(kernelContext_11->epBias_0+(biasBase_0 + j_2)));
    }
    else
    {
        if((kernelContext_11->p_0->bias_0) == 2U)
        {
            _S37 = r_5 + float(*(kernelContext_11->epBias_0+(biasBase_0 + i_0)));
        }
        else
        {
            _S37 = r_5;
        }
    }
    if((kernelContext_11->p_0->act_0) == 1U)
    {
        _S37 = dg_max_0(_S37, 0.0);
    }
    else
    {
        if((kernelContext_11->p_0->act_0) == 2U)
        {
            _S37 = 0.5 * _S37 * (1.0 + (1.0 - 2.0 / (exp(2.0 * (0.79788458347320557 * (_S37 + 0.04471499845385551 * _S37 * _S37 * _S37))) + 1.0)));
        }
        else
        {
            if((kernelContext_11->p_0->act_0) == 3U)
            {
                _S37 = _S37 / (1.0 + exp(- _S37));
            }
        }
    }
    if((kernelContext_11->p_0->gated_0) != 0U)
    {
        _S37 = _S37 * float(*(kernelContext_11->epGate_0+idx_0));
    }
    if((kernelContext_11->p_0->residual_0) != 0U)
    {
        _S37 = _S37 + float(*(kernelContext_11->epResidual_0+idx_0));
    }
    return _S37;
}

void gemmStore_0(const GemmProblem_0 thread* q_2, uint i_1, uint j_3, float acc_1, KernelContext_0 thread* kernelContext_12)
{
    uint idx_1 = q_2->baseC_0 + i_1 * q_2->ldc_0 + j_3;
    float r_6;
    if((kernelContext_12->p_0->beta_0) == 0.0)
    {
        r_6 = kernelContext_12->p_0->alpha_0 * acc_1;
    }
    else
    {
        r_6 = kernelContext_12->p_0->alpha_0 * acc_1 + kernelContext_12->p_0->beta_0 * float(*(kernelContext_12->c_0+idx_1));
    }
    half device* _S38 = kernelContext_12->c_0+idx_1;
    float _S39 = gemmEpilogue_0(r_6, i_1, j_3, idx_1, q_2->baseBias_0, kernelContext_12);
    *_S38 = half(_S39);
    return;
}

[[kernel]] void gemm_q_tiled_bf16_h(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], uint device* a_2 [[buffer(0)]], uint device* b_4 [[buffer(1)]], half device* c_1 [[buffer(2)]], half device* epBias_1 [[buffer(3)]], half device* epGate_1 [[buffer(4)]], half device* epResidual_1 [[buffer(5)]])
{
    uint _S40;
    thread KernelContext_0 kernelContext_13;
    (&kernelContext_13)->p_0 = p_1;
    (&kernelContext_13)->a_0 = a_2;
    (&kernelContext_13)->b_0 = b_4;
    (&kernelContext_13)->c_0 = c_1;
    (&kernelContext_13)->epBias_0 = epBias_1;
    (&kernelContext_13)->epGate_0 = epGate_1;
    (&kernelContext_13)->epResidual_0 = epResidual_1;
    threadgroup array<array<uint, int(32)>, int(32)> qTileA_1;
    (&kernelContext_13)->qTileA_0 = &qTileA_1;
    threadgroup array<array<float, int(64)>, int(32)> qTileB_1;
    (&kernelContext_13)->qTileB_0 = &qTileB_1;
    threadgroup array<array<float2, int(64)>, int(2)> qScale_1;
    (&kernelContext_13)->qScale_0 = &qScale_1;
    GemmProblem_0 _S41 = gemmBatchProblem_0(group_0.z, &kernelContext_13);
    uint tx_0 = local_0.x;
    uint ty_0 = local_0.y;
    uint _S42 = ty_0 * 16U + tx_0;
    uint _S43 = group_0.y * 64U;
    uint _S44 = group_0.x * 64U;
    thread array<float, int(4)> sumA_2;
    sumA_2[0U] = 0.0;
    thread array<array<float, int(4)>, int(4)> acc_2;
    acc_2[0U][0U] = 0.0;
    thread array<array<float, int(4)>, int(4)> s_2;
    s_2[0U][0U] = 0.0;
    thread array<array<float, int(4)>, int(4)> s2_1;
    s2_1[0U][0U] = 0.0;
    acc_2[0U][1U] = 0.0;
    s_2[0U][1U] = 0.0;
    s2_1[0U][1U] = 0.0;
    acc_2[0U][2U] = 0.0;
    s_2[0U][2U] = 0.0;
    s2_1[0U][2U] = 0.0;
    acc_2[0U][3U] = 0.0;
    s_2[0U][3U] = 0.0;
    s2_1[0U][3U] = 0.0;
    sumA_2[1U] = 0.0;
    acc_2[1U][0U] = 0.0;
    s_2[1U][0U] = 0.0;
    s2_1[1U][0U] = 0.0;
    acc_2[1U][1U] = 0.0;
    s_2[1U][1U] = 0.0;
    s2_1[1U][1U] = 0.0;
    acc_2[1U][2U] = 0.0;
    s_2[1U][2U] = 0.0;
    s2_1[1U][2U] = 0.0;
    acc_2[1U][3U] = 0.0;
    s_2[1U][3U] = 0.0;
    s2_1[1U][3U] = 0.0;
    sumA_2[2U] = 0.0;
    acc_2[2U][0U] = 0.0;
    s_2[2U][0U] = 0.0;
    s2_1[2U][0U] = 0.0;
    acc_2[2U][1U] = 0.0;
    s_2[2U][1U] = 0.0;
    s2_1[2U][1U] = 0.0;
    acc_2[2U][2U] = 0.0;
    s_2[2U][2U] = 0.0;
    s2_1[2U][2U] = 0.0;
    acc_2[2U][3U] = 0.0;
    s_2[2U][3U] = 0.0;
    s2_1[2U][3U] = 0.0;
    sumA_2[3U] = 0.0;
    acc_2[3U][0U] = 0.0;
    s_2[3U][0U] = 0.0;
    s2_1[3U][0U] = 0.0;
    acc_2[3U][1U] = 0.0;
    s_2[3U][1U] = 0.0;
    s2_1[3U][1U] = 0.0;
    acc_2[3U][2U] = 0.0;
    s_2[3U][2U] = 0.0;
    s2_1[3U][2U] = 0.0;
    acc_2[3U][3U] = 0.0;
    s_2[3U][3U] = 0.0;
    s2_1[3U][3U] = 0.0;
    uint k0_0 = 0U;
    for(;;)
    {
        if(k0_0 < ((&kernelContext_13)->p_0->k_1))
        {
        }
        else
        {
            break;
        }
        uint ar_0 = _S42 / 8U;
        uint _S45 = _S42 % 8U * 4U;
        uint g0_0 = _S43 + ar_0;
        uint _S46 = g0_0 + 32U;
        uint lo_0;
        if(g0_0 < ((&kernelContext_13)->p_0->m_1))
        {
            uint _S47 = rawA_0(_S41.baseA_0 + g0_0 * (&kernelContext_13)->p_0->lda_1 + k0_0 + _S45, &kernelContext_13);
            lo_0 = _S47;
        }
        else
        {
            lo_0 = 0U;
        }
        uint hi_0;
        if(_S46 < ((&kernelContext_13)->p_0->m_1))
        {
            uint _S48 = rawA_0(_S41.baseA_0 + _S46 * (&kernelContext_13)->p_0->lda_1 + k0_0 + _S45, &kernelContext_13);
            hi_0 = _S48;
        }
        else
        {
            hi_0 = 0U;
        }
        (*(&kernelContext_13)->qTileA_0)[_S45][ar_0] = lo_0 | (hi_0 << 16U);
        uint lo_1;
        if(g0_0 < ((&kernelContext_13)->p_0->m_1))
        {
            uint _S49 = rawA_0(_S41.baseA_0 + g0_0 * (&kernelContext_13)->p_0->lda_1 + k0_0 + _S45 + 1U, &kernelContext_13);
            lo_1 = _S49;
        }
        else
        {
            lo_1 = 0U;
        }
        uint hi_1;
        if(_S46 < ((&kernelContext_13)->p_0->m_1))
        {
            uint _S50 = rawA_0(_S41.baseA_0 + _S46 * (&kernelContext_13)->p_0->lda_1 + k0_0 + _S45 + 1U, &kernelContext_13);
            hi_1 = _S50;
        }
        else
        {
            hi_1 = 0U;
        }
        (*(&kernelContext_13)->qTileA_0)[_S45 + 1U][ar_0] = lo_1 | (hi_1 << 16U);
        uint lo_2;
        if(g0_0 < ((&kernelContext_13)->p_0->m_1))
        {
            uint _S51 = rawA_0(_S41.baseA_0 + g0_0 * (&kernelContext_13)->p_0->lda_1 + k0_0 + _S45 + 2U, &kernelContext_13);
            lo_2 = _S51;
        }
        else
        {
            lo_2 = 0U;
        }
        uint hi_2;
        if(_S46 < ((&kernelContext_13)->p_0->m_1))
        {
            uint _S52 = rawA_0(_S41.baseA_0 + _S46 * (&kernelContext_13)->p_0->lda_1 + k0_0 + _S45 + 2U, &kernelContext_13);
            hi_2 = _S52;
        }
        else
        {
            hi_2 = 0U;
        }
        (*(&kernelContext_13)->qTileA_0)[_S45 + 2U][ar_0] = lo_2 | (hi_2 << 16U);
        uint lo_3;
        if(g0_0 < ((&kernelContext_13)->p_0->m_1))
        {
            uint _S53 = rawA_0(_S41.baseA_0 + g0_0 * (&kernelContext_13)->p_0->lda_1 + k0_0 + _S45 + 3U, &kernelContext_13);
            lo_3 = _S53;
        }
        else
        {
            lo_3 = 0U;
        }
        uint hi_3;
        if(_S46 < ((&kernelContext_13)->p_0->m_1))
        {
            uint _S54 = rawA_0(_S41.baseA_0 + _S46 * (&kernelContext_13)->p_0->lda_1 + k0_0 + _S45 + 3U, &kernelContext_13);
            hi_3 = _S54;
        }
        else
        {
            hi_3 = 0U;
        }
        (*(&kernelContext_13)->qTileA_0)[_S45 + 3U][ar_0] = lo_3 | (hi_3 << 16U);
        uint bk_0 = _S42 % 4U * 8U;
        thread array<float, int(8)> q_3;
        load8Codes_0(min(_S44 + _S42 / 4U, (&kernelContext_13)->p_0->n_1 - 1U), k0_0 + bk_0, &q_3, &kernelContext_13);
        (*(&kernelContext_13)->qTileB_0)[bk_0][_S42 / 4U] = q_3[0U];
        (*(&kernelContext_13)->qTileB_0)[bk_0 + 1U][_S42 / 4U] = q_3[1U];
        (*(&kernelContext_13)->qTileB_0)[bk_0 + 2U][_S42 / 4U] = q_3[2U];
        (*(&kernelContext_13)->qTileB_0)[bk_0 + 3U][_S42 / 4U] = q_3[3U];
        (*(&kernelContext_13)->qTileB_0)[bk_0 + 4U][_S42 / 4U] = q_3[4U];
        (*(&kernelContext_13)->qTileB_0)[bk_0 + 5U][_S42 / 4U] = q_3[5U];
        (*(&kernelContext_13)->qTileB_0)[bk_0 + 6U][_S42 / 4U] = q_3[6U];
        (*(&kernelContext_13)->qTileB_0)[bk_0 + 7U][_S42 / 4U] = q_3[7U];
        uint _S55 = k0_0 + 32U;
        bool _S56 = QFORMAT_0 >= 3U;
        uint _S57;
        if(_S56)
        {
            _S57 = 32U;
        }
        else
        {
            _S57 = QBLOCK_0;
        }
        uint _S58 = _S55 % _S57;
        bool _S59;
        if(_S58 == 0U)
        {
            _S59 = _S42 < 64U;
        }
        else
        {
            _S59 = false;
        }
        uint _S60;
        if(_S59)
        {
            uint _S61 = min(_S44 + _S42, (&kernelContext_13)->p_0->n_1 - 1U);
            if(_S56)
            {
                _S60 = 32U;
            }
            else
            {
                _S60 = QBLOCK_0;
            }
            uint u_3 = k0_0 / _S60;
            float2 _S62 = unitScale_0(_S61, u_3, &kernelContext_13);
            (*(&kernelContext_13)->qScale_0)[u_3 & 1U][_S42] = _S62;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint w0_0 = (*(&kernelContext_13)->qTileA_0)[0U][ty_0];
        uint _S63 = ty_0 + 16U;
        uint w1_0 = (*(&kernelContext_13)->qTileA_0)[0U][_S63];
        thread array<float, int(4)> av_0;
        av_0[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[0U][ty_0]);
        av_0[int(1)] = aLo_0(w1_0);
        av_0[int(2)] = aHi_0(w0_0);
        av_0[int(3)] = aHi_0(w1_0);
        thread array<float, int(4)> bv_0;
        bv_0[0U] = (*(&kernelContext_13)->qTileB_0)[0U][tx_0];
        uint _S64 = tx_0 + 16U;
        bv_0[1U] = (*(&kernelContext_13)->qTileB_0)[0U][_S64];
        uint _S65 = tx_0 + 32U;
        bv_0[2U] = (*(&kernelContext_13)->qTileB_0)[0U][_S65];
        uint _S66 = tx_0 + 48U;
        bv_0[3U] = (*(&kernelContext_13)->qTileB_0)[0U][_S66];
        bool _S67;
        if(QFORMAT_0 == 2U)
        {
            _S67 = true;
        }
        else
        {
            _S67 = QFORMAT_0 == 3U;
        }
        bool _S68;
        if(_S67)
        {
            _S68 = true;
        }
        else
        {
            _S68 = QFORMAT_0 == 4U;
        }
        if(_S68)
        {
            sumA_2[0U] = sumA_2[0U] + av_0[0U];
        }
        s_2[0U][0U] = fma(av_0[0U], bv_0[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_0[0U], bv_0[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_0[0U], bv_0[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_0[0U], bv_0[3U], s_2[0U][3U]);
        bool _S69;
        if(QFORMAT_0 == 2U)
        {
            _S69 = true;
        }
        else
        {
            _S69 = QFORMAT_0 == 3U;
        }
        bool _S70;
        if(_S69)
        {
            _S70 = true;
        }
        else
        {
            _S70 = QFORMAT_0 == 4U;
        }
        if(_S70)
        {
            sumA_2[1U] = sumA_2[1U] + av_0[1U];
        }
        s_2[1U][0U] = fma(av_0[1U], bv_0[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_0[1U], bv_0[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_0[1U], bv_0[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_0[1U], bv_0[3U], s_2[1U][3U]);
        bool _S71;
        if(QFORMAT_0 == 2U)
        {
            _S71 = true;
        }
        else
        {
            _S71 = QFORMAT_0 == 3U;
        }
        bool _S72;
        if(_S71)
        {
            _S72 = true;
        }
        else
        {
            _S72 = QFORMAT_0 == 4U;
        }
        if(_S72)
        {
            sumA_2[2U] = sumA_2[2U] + av_0[2U];
        }
        s_2[2U][0U] = fma(av_0[2U], bv_0[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_0[2U], bv_0[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_0[2U], bv_0[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_0[2U], bv_0[3U], s_2[2U][3U]);
        bool _S73;
        if(QFORMAT_0 == 2U)
        {
            _S73 = true;
        }
        else
        {
            _S73 = QFORMAT_0 == 3U;
        }
        bool _S74;
        if(_S73)
        {
            _S74 = true;
        }
        else
        {
            _S74 = QFORMAT_0 == 4U;
        }
        if(_S74)
        {
            sumA_2[3U] = sumA_2[3U] + av_0[3U];
        }
        s_2[3U][0U] = fma(av_0[3U], bv_0[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_0[3U], bv_0[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_0[3U], bv_0[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_0[3U], bv_0[3U], s_2[3U][3U]);
        uint w0_1 = (*(&kernelContext_13)->qTileA_0)[1U][ty_0];
        uint w1_1 = (*(&kernelContext_13)->qTileA_0)[1U][_S63];
        thread array<float, int(4)> av_1;
        av_1[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[1U][ty_0]);
        av_1[int(1)] = aLo_0(w1_1);
        av_1[int(2)] = aHi_0(w0_1);
        av_1[int(3)] = aHi_0(w1_1);
        thread array<float, int(4)> bv_1;
        bv_1[0U] = (*(&kernelContext_13)->qTileB_0)[1U][tx_0];
        bv_1[1U] = (*(&kernelContext_13)->qTileB_0)[1U][_S64];
        bv_1[2U] = (*(&kernelContext_13)->qTileB_0)[1U][_S65];
        bv_1[3U] = (*(&kernelContext_13)->qTileB_0)[1U][_S66];
        bool _S75;
        if(QFORMAT_0 == 2U)
        {
            _S75 = true;
        }
        else
        {
            _S75 = QFORMAT_0 == 3U;
        }
        bool _S76;
        if(_S75)
        {
            _S76 = true;
        }
        else
        {
            _S76 = QFORMAT_0 == 4U;
        }
        if(_S76)
        {
            sumA_2[0U] = sumA_2[0U] + av_1[0U];
        }
        s_2[0U][0U] = fma(av_1[0U], bv_1[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_1[0U], bv_1[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_1[0U], bv_1[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_1[0U], bv_1[3U], s_2[0U][3U]);
        bool _S77;
        if(QFORMAT_0 == 2U)
        {
            _S77 = true;
        }
        else
        {
            _S77 = QFORMAT_0 == 3U;
        }
        bool _S78;
        if(_S77)
        {
            _S78 = true;
        }
        else
        {
            _S78 = QFORMAT_0 == 4U;
        }
        if(_S78)
        {
            sumA_2[1U] = sumA_2[1U] + av_1[1U];
        }
        s_2[1U][0U] = fma(av_1[1U], bv_1[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_1[1U], bv_1[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_1[1U], bv_1[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_1[1U], bv_1[3U], s_2[1U][3U]);
        bool _S79;
        if(QFORMAT_0 == 2U)
        {
            _S79 = true;
        }
        else
        {
            _S79 = QFORMAT_0 == 3U;
        }
        bool _S80;
        if(_S79)
        {
            _S80 = true;
        }
        else
        {
            _S80 = QFORMAT_0 == 4U;
        }
        if(_S80)
        {
            sumA_2[2U] = sumA_2[2U] + av_1[2U];
        }
        s_2[2U][0U] = fma(av_1[2U], bv_1[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_1[2U], bv_1[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_1[2U], bv_1[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_1[2U], bv_1[3U], s_2[2U][3U]);
        bool _S81;
        if(QFORMAT_0 == 2U)
        {
            _S81 = true;
        }
        else
        {
            _S81 = QFORMAT_0 == 3U;
        }
        bool _S82;
        if(_S81)
        {
            _S82 = true;
        }
        else
        {
            _S82 = QFORMAT_0 == 4U;
        }
        if(_S82)
        {
            sumA_2[3U] = sumA_2[3U] + av_1[3U];
        }
        s_2[3U][0U] = fma(av_1[3U], bv_1[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_1[3U], bv_1[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_1[3U], bv_1[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_1[3U], bv_1[3U], s_2[3U][3U]);
        uint w0_2 = (*(&kernelContext_13)->qTileA_0)[2U][ty_0];
        uint w1_2 = (*(&kernelContext_13)->qTileA_0)[2U][_S63];
        thread array<float, int(4)> av_2;
        av_2[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[2U][ty_0]);
        av_2[int(1)] = aLo_0(w1_2);
        av_2[int(2)] = aHi_0(w0_2);
        av_2[int(3)] = aHi_0(w1_2);
        thread array<float, int(4)> bv_2;
        bv_2[0U] = (*(&kernelContext_13)->qTileB_0)[2U][tx_0];
        bv_2[1U] = (*(&kernelContext_13)->qTileB_0)[2U][_S64];
        bv_2[2U] = (*(&kernelContext_13)->qTileB_0)[2U][_S65];
        bv_2[3U] = (*(&kernelContext_13)->qTileB_0)[2U][_S66];
        bool _S83;
        if(QFORMAT_0 == 2U)
        {
            _S83 = true;
        }
        else
        {
            _S83 = QFORMAT_0 == 3U;
        }
        bool _S84;
        if(_S83)
        {
            _S84 = true;
        }
        else
        {
            _S84 = QFORMAT_0 == 4U;
        }
        if(_S84)
        {
            sumA_2[0U] = sumA_2[0U] + av_2[0U];
        }
        s_2[0U][0U] = fma(av_2[0U], bv_2[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_2[0U], bv_2[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_2[0U], bv_2[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_2[0U], bv_2[3U], s_2[0U][3U]);
        bool _S85;
        if(QFORMAT_0 == 2U)
        {
            _S85 = true;
        }
        else
        {
            _S85 = QFORMAT_0 == 3U;
        }
        bool _S86;
        if(_S85)
        {
            _S86 = true;
        }
        else
        {
            _S86 = QFORMAT_0 == 4U;
        }
        if(_S86)
        {
            sumA_2[1U] = sumA_2[1U] + av_2[1U];
        }
        s_2[1U][0U] = fma(av_2[1U], bv_2[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_2[1U], bv_2[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_2[1U], bv_2[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_2[1U], bv_2[3U], s_2[1U][3U]);
        bool _S87;
        if(QFORMAT_0 == 2U)
        {
            _S87 = true;
        }
        else
        {
            _S87 = QFORMAT_0 == 3U;
        }
        bool _S88;
        if(_S87)
        {
            _S88 = true;
        }
        else
        {
            _S88 = QFORMAT_0 == 4U;
        }
        if(_S88)
        {
            sumA_2[2U] = sumA_2[2U] + av_2[2U];
        }
        s_2[2U][0U] = fma(av_2[2U], bv_2[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_2[2U], bv_2[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_2[2U], bv_2[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_2[2U], bv_2[3U], s_2[2U][3U]);
        bool _S89;
        if(QFORMAT_0 == 2U)
        {
            _S89 = true;
        }
        else
        {
            _S89 = QFORMAT_0 == 3U;
        }
        bool _S90;
        if(_S89)
        {
            _S90 = true;
        }
        else
        {
            _S90 = QFORMAT_0 == 4U;
        }
        if(_S90)
        {
            sumA_2[3U] = sumA_2[3U] + av_2[3U];
        }
        s_2[3U][0U] = fma(av_2[3U], bv_2[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_2[3U], bv_2[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_2[3U], bv_2[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_2[3U], bv_2[3U], s_2[3U][3U]);
        uint w0_3 = (*(&kernelContext_13)->qTileA_0)[3U][ty_0];
        uint w1_3 = (*(&kernelContext_13)->qTileA_0)[3U][_S63];
        thread array<float, int(4)> av_3;
        av_3[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[3U][ty_0]);
        av_3[int(1)] = aLo_0(w1_3);
        av_3[int(2)] = aHi_0(w0_3);
        av_3[int(3)] = aHi_0(w1_3);
        thread array<float, int(4)> bv_3;
        bv_3[0U] = (*(&kernelContext_13)->qTileB_0)[3U][tx_0];
        bv_3[1U] = (*(&kernelContext_13)->qTileB_0)[3U][_S64];
        bv_3[2U] = (*(&kernelContext_13)->qTileB_0)[3U][_S65];
        bv_3[3U] = (*(&kernelContext_13)->qTileB_0)[3U][_S66];
        bool _S91;
        if(QFORMAT_0 == 2U)
        {
            _S91 = true;
        }
        else
        {
            _S91 = QFORMAT_0 == 3U;
        }
        bool _S92;
        if(_S91)
        {
            _S92 = true;
        }
        else
        {
            _S92 = QFORMAT_0 == 4U;
        }
        if(_S92)
        {
            sumA_2[0U] = sumA_2[0U] + av_3[0U];
        }
        s_2[0U][0U] = fma(av_3[0U], bv_3[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_3[0U], bv_3[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_3[0U], bv_3[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_3[0U], bv_3[3U], s_2[0U][3U]);
        bool _S93;
        if(QFORMAT_0 == 2U)
        {
            _S93 = true;
        }
        else
        {
            _S93 = QFORMAT_0 == 3U;
        }
        bool _S94;
        if(_S93)
        {
            _S94 = true;
        }
        else
        {
            _S94 = QFORMAT_0 == 4U;
        }
        if(_S94)
        {
            sumA_2[1U] = sumA_2[1U] + av_3[1U];
        }
        s_2[1U][0U] = fma(av_3[1U], bv_3[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_3[1U], bv_3[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_3[1U], bv_3[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_3[1U], bv_3[3U], s_2[1U][3U]);
        bool _S95;
        if(QFORMAT_0 == 2U)
        {
            _S95 = true;
        }
        else
        {
            _S95 = QFORMAT_0 == 3U;
        }
        bool _S96;
        if(_S95)
        {
            _S96 = true;
        }
        else
        {
            _S96 = QFORMAT_0 == 4U;
        }
        if(_S96)
        {
            sumA_2[2U] = sumA_2[2U] + av_3[2U];
        }
        s_2[2U][0U] = fma(av_3[2U], bv_3[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_3[2U], bv_3[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_3[2U], bv_3[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_3[2U], bv_3[3U], s_2[2U][3U]);
        bool _S97;
        if(QFORMAT_0 == 2U)
        {
            _S97 = true;
        }
        else
        {
            _S97 = QFORMAT_0 == 3U;
        }
        bool _S98;
        if(_S97)
        {
            _S98 = true;
        }
        else
        {
            _S98 = QFORMAT_0 == 4U;
        }
        if(_S98)
        {
            sumA_2[3U] = sumA_2[3U] + av_3[3U];
        }
        s_2[3U][0U] = fma(av_3[3U], bv_3[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_3[3U], bv_3[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_3[3U], bv_3[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_3[3U], bv_3[3U], s_2[3U][3U]);
        uint w0_4 = (*(&kernelContext_13)->qTileA_0)[4U][ty_0];
        uint w1_4 = (*(&kernelContext_13)->qTileA_0)[4U][_S63];
        thread array<float, int(4)> av_4;
        av_4[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[4U][ty_0]);
        av_4[int(1)] = aLo_0(w1_4);
        av_4[int(2)] = aHi_0(w0_4);
        av_4[int(3)] = aHi_0(w1_4);
        thread array<float, int(4)> bv_4;
        bv_4[0U] = (*(&kernelContext_13)->qTileB_0)[4U][tx_0];
        bv_4[1U] = (*(&kernelContext_13)->qTileB_0)[4U][_S64];
        bv_4[2U] = (*(&kernelContext_13)->qTileB_0)[4U][_S65];
        bv_4[3U] = (*(&kernelContext_13)->qTileB_0)[4U][_S66];
        bool _S99;
        if(QFORMAT_0 == 2U)
        {
            _S99 = true;
        }
        else
        {
            _S99 = QFORMAT_0 == 3U;
        }
        bool _S100;
        if(_S99)
        {
            _S100 = true;
        }
        else
        {
            _S100 = QFORMAT_0 == 4U;
        }
        if(_S100)
        {
            sumA_2[0U] = sumA_2[0U] + av_4[0U];
        }
        s_2[0U][0U] = fma(av_4[0U], bv_4[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_4[0U], bv_4[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_4[0U], bv_4[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_4[0U], bv_4[3U], s_2[0U][3U]);
        bool _S101;
        if(QFORMAT_0 == 2U)
        {
            _S101 = true;
        }
        else
        {
            _S101 = QFORMAT_0 == 3U;
        }
        bool _S102;
        if(_S101)
        {
            _S102 = true;
        }
        else
        {
            _S102 = QFORMAT_0 == 4U;
        }
        if(_S102)
        {
            sumA_2[1U] = sumA_2[1U] + av_4[1U];
        }
        s_2[1U][0U] = fma(av_4[1U], bv_4[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_4[1U], bv_4[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_4[1U], bv_4[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_4[1U], bv_4[3U], s_2[1U][3U]);
        bool _S103;
        if(QFORMAT_0 == 2U)
        {
            _S103 = true;
        }
        else
        {
            _S103 = QFORMAT_0 == 3U;
        }
        bool _S104;
        if(_S103)
        {
            _S104 = true;
        }
        else
        {
            _S104 = QFORMAT_0 == 4U;
        }
        if(_S104)
        {
            sumA_2[2U] = sumA_2[2U] + av_4[2U];
        }
        s_2[2U][0U] = fma(av_4[2U], bv_4[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_4[2U], bv_4[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_4[2U], bv_4[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_4[2U], bv_4[3U], s_2[2U][3U]);
        bool _S105;
        if(QFORMAT_0 == 2U)
        {
            _S105 = true;
        }
        else
        {
            _S105 = QFORMAT_0 == 3U;
        }
        bool _S106;
        if(_S105)
        {
            _S106 = true;
        }
        else
        {
            _S106 = QFORMAT_0 == 4U;
        }
        if(_S106)
        {
            sumA_2[3U] = sumA_2[3U] + av_4[3U];
        }
        s_2[3U][0U] = fma(av_4[3U], bv_4[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_4[3U], bv_4[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_4[3U], bv_4[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_4[3U], bv_4[3U], s_2[3U][3U]);
        uint w0_5 = (*(&kernelContext_13)->qTileA_0)[5U][ty_0];
        uint w1_5 = (*(&kernelContext_13)->qTileA_0)[5U][_S63];
        thread array<float, int(4)> av_5;
        av_5[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[5U][ty_0]);
        av_5[int(1)] = aLo_0(w1_5);
        av_5[int(2)] = aHi_0(w0_5);
        av_5[int(3)] = aHi_0(w1_5);
        thread array<float, int(4)> bv_5;
        bv_5[0U] = (*(&kernelContext_13)->qTileB_0)[5U][tx_0];
        bv_5[1U] = (*(&kernelContext_13)->qTileB_0)[5U][_S64];
        bv_5[2U] = (*(&kernelContext_13)->qTileB_0)[5U][_S65];
        bv_5[3U] = (*(&kernelContext_13)->qTileB_0)[5U][_S66];
        bool _S107;
        if(QFORMAT_0 == 2U)
        {
            _S107 = true;
        }
        else
        {
            _S107 = QFORMAT_0 == 3U;
        }
        bool _S108;
        if(_S107)
        {
            _S108 = true;
        }
        else
        {
            _S108 = QFORMAT_0 == 4U;
        }
        if(_S108)
        {
            sumA_2[0U] = sumA_2[0U] + av_5[0U];
        }
        s_2[0U][0U] = fma(av_5[0U], bv_5[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_5[0U], bv_5[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_5[0U], bv_5[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_5[0U], bv_5[3U], s_2[0U][3U]);
        bool _S109;
        if(QFORMAT_0 == 2U)
        {
            _S109 = true;
        }
        else
        {
            _S109 = QFORMAT_0 == 3U;
        }
        bool _S110;
        if(_S109)
        {
            _S110 = true;
        }
        else
        {
            _S110 = QFORMAT_0 == 4U;
        }
        if(_S110)
        {
            sumA_2[1U] = sumA_2[1U] + av_5[1U];
        }
        s_2[1U][0U] = fma(av_5[1U], bv_5[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_5[1U], bv_5[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_5[1U], bv_5[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_5[1U], bv_5[3U], s_2[1U][3U]);
        bool _S111;
        if(QFORMAT_0 == 2U)
        {
            _S111 = true;
        }
        else
        {
            _S111 = QFORMAT_0 == 3U;
        }
        bool _S112;
        if(_S111)
        {
            _S112 = true;
        }
        else
        {
            _S112 = QFORMAT_0 == 4U;
        }
        if(_S112)
        {
            sumA_2[2U] = sumA_2[2U] + av_5[2U];
        }
        s_2[2U][0U] = fma(av_5[2U], bv_5[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_5[2U], bv_5[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_5[2U], bv_5[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_5[2U], bv_5[3U], s_2[2U][3U]);
        bool _S113;
        if(QFORMAT_0 == 2U)
        {
            _S113 = true;
        }
        else
        {
            _S113 = QFORMAT_0 == 3U;
        }
        bool _S114;
        if(_S113)
        {
            _S114 = true;
        }
        else
        {
            _S114 = QFORMAT_0 == 4U;
        }
        if(_S114)
        {
            sumA_2[3U] = sumA_2[3U] + av_5[3U];
        }
        s_2[3U][0U] = fma(av_5[3U], bv_5[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_5[3U], bv_5[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_5[3U], bv_5[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_5[3U], bv_5[3U], s_2[3U][3U]);
        uint w0_6 = (*(&kernelContext_13)->qTileA_0)[6U][ty_0];
        uint w1_6 = (*(&kernelContext_13)->qTileA_0)[6U][_S63];
        thread array<float, int(4)> av_6;
        av_6[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[6U][ty_0]);
        av_6[int(1)] = aLo_0(w1_6);
        av_6[int(2)] = aHi_0(w0_6);
        av_6[int(3)] = aHi_0(w1_6);
        thread array<float, int(4)> bv_6;
        bv_6[0U] = (*(&kernelContext_13)->qTileB_0)[6U][tx_0];
        bv_6[1U] = (*(&kernelContext_13)->qTileB_0)[6U][_S64];
        bv_6[2U] = (*(&kernelContext_13)->qTileB_0)[6U][_S65];
        bv_6[3U] = (*(&kernelContext_13)->qTileB_0)[6U][_S66];
        bool _S115;
        if(QFORMAT_0 == 2U)
        {
            _S115 = true;
        }
        else
        {
            _S115 = QFORMAT_0 == 3U;
        }
        bool _S116;
        if(_S115)
        {
            _S116 = true;
        }
        else
        {
            _S116 = QFORMAT_0 == 4U;
        }
        if(_S116)
        {
            sumA_2[0U] = sumA_2[0U] + av_6[0U];
        }
        s_2[0U][0U] = fma(av_6[0U], bv_6[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_6[0U], bv_6[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_6[0U], bv_6[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_6[0U], bv_6[3U], s_2[0U][3U]);
        bool _S117;
        if(QFORMAT_0 == 2U)
        {
            _S117 = true;
        }
        else
        {
            _S117 = QFORMAT_0 == 3U;
        }
        bool _S118;
        if(_S117)
        {
            _S118 = true;
        }
        else
        {
            _S118 = QFORMAT_0 == 4U;
        }
        if(_S118)
        {
            sumA_2[1U] = sumA_2[1U] + av_6[1U];
        }
        s_2[1U][0U] = fma(av_6[1U], bv_6[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_6[1U], bv_6[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_6[1U], bv_6[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_6[1U], bv_6[3U], s_2[1U][3U]);
        bool _S119;
        if(QFORMAT_0 == 2U)
        {
            _S119 = true;
        }
        else
        {
            _S119 = QFORMAT_0 == 3U;
        }
        bool _S120;
        if(_S119)
        {
            _S120 = true;
        }
        else
        {
            _S120 = QFORMAT_0 == 4U;
        }
        if(_S120)
        {
            sumA_2[2U] = sumA_2[2U] + av_6[2U];
        }
        s_2[2U][0U] = fma(av_6[2U], bv_6[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_6[2U], bv_6[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_6[2U], bv_6[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_6[2U], bv_6[3U], s_2[2U][3U]);
        bool _S121;
        if(QFORMAT_0 == 2U)
        {
            _S121 = true;
        }
        else
        {
            _S121 = QFORMAT_0 == 3U;
        }
        bool _S122;
        if(_S121)
        {
            _S122 = true;
        }
        else
        {
            _S122 = QFORMAT_0 == 4U;
        }
        if(_S122)
        {
            sumA_2[3U] = sumA_2[3U] + av_6[3U];
        }
        s_2[3U][0U] = fma(av_6[3U], bv_6[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_6[3U], bv_6[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_6[3U], bv_6[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_6[3U], bv_6[3U], s_2[3U][3U]);
        uint w0_7 = (*(&kernelContext_13)->qTileA_0)[7U][ty_0];
        uint w1_7 = (*(&kernelContext_13)->qTileA_0)[7U][_S63];
        thread array<float, int(4)> av_7;
        av_7[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[7U][ty_0]);
        av_7[int(1)] = aLo_0(w1_7);
        av_7[int(2)] = aHi_0(w0_7);
        av_7[int(3)] = aHi_0(w1_7);
        thread array<float, int(4)> bv_7;
        bv_7[0U] = (*(&kernelContext_13)->qTileB_0)[7U][tx_0];
        bv_7[1U] = (*(&kernelContext_13)->qTileB_0)[7U][_S64];
        bv_7[2U] = (*(&kernelContext_13)->qTileB_0)[7U][_S65];
        bv_7[3U] = (*(&kernelContext_13)->qTileB_0)[7U][_S66];
        bool _S123;
        if(QFORMAT_0 == 2U)
        {
            _S123 = true;
        }
        else
        {
            _S123 = QFORMAT_0 == 3U;
        }
        bool _S124;
        if(_S123)
        {
            _S124 = true;
        }
        else
        {
            _S124 = QFORMAT_0 == 4U;
        }
        if(_S124)
        {
            sumA_2[0U] = sumA_2[0U] + av_7[0U];
        }
        s_2[0U][0U] = fma(av_7[0U], bv_7[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_7[0U], bv_7[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_7[0U], bv_7[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_7[0U], bv_7[3U], s_2[0U][3U]);
        bool _S125;
        if(QFORMAT_0 == 2U)
        {
            _S125 = true;
        }
        else
        {
            _S125 = QFORMAT_0 == 3U;
        }
        bool _S126;
        if(_S125)
        {
            _S126 = true;
        }
        else
        {
            _S126 = QFORMAT_0 == 4U;
        }
        if(_S126)
        {
            sumA_2[1U] = sumA_2[1U] + av_7[1U];
        }
        s_2[1U][0U] = fma(av_7[1U], bv_7[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_7[1U], bv_7[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_7[1U], bv_7[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_7[1U], bv_7[3U], s_2[1U][3U]);
        bool _S127;
        if(QFORMAT_0 == 2U)
        {
            _S127 = true;
        }
        else
        {
            _S127 = QFORMAT_0 == 3U;
        }
        bool _S128;
        if(_S127)
        {
            _S128 = true;
        }
        else
        {
            _S128 = QFORMAT_0 == 4U;
        }
        if(_S128)
        {
            sumA_2[2U] = sumA_2[2U] + av_7[2U];
        }
        s_2[2U][0U] = fma(av_7[2U], bv_7[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_7[2U], bv_7[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_7[2U], bv_7[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_7[2U], bv_7[3U], s_2[2U][3U]);
        bool _S129;
        if(QFORMAT_0 == 2U)
        {
            _S129 = true;
        }
        else
        {
            _S129 = QFORMAT_0 == 3U;
        }
        bool _S130;
        if(_S129)
        {
            _S130 = true;
        }
        else
        {
            _S130 = QFORMAT_0 == 4U;
        }
        if(_S130)
        {
            sumA_2[3U] = sumA_2[3U] + av_7[3U];
        }
        s_2[3U][0U] = fma(av_7[3U], bv_7[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_7[3U], bv_7[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_7[3U], bv_7[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_7[3U], bv_7[3U], s_2[3U][3U]);
        uint w0_8 = (*(&kernelContext_13)->qTileA_0)[8U][ty_0];
        uint w1_8 = (*(&kernelContext_13)->qTileA_0)[8U][_S63];
        thread array<float, int(4)> av_8;
        av_8[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[8U][ty_0]);
        av_8[int(1)] = aLo_0(w1_8);
        av_8[int(2)] = aHi_0(w0_8);
        av_8[int(3)] = aHi_0(w1_8);
        thread array<float, int(4)> bv_8;
        bv_8[0U] = (*(&kernelContext_13)->qTileB_0)[8U][tx_0];
        bv_8[1U] = (*(&kernelContext_13)->qTileB_0)[8U][_S64];
        bv_8[2U] = (*(&kernelContext_13)->qTileB_0)[8U][_S65];
        bv_8[3U] = (*(&kernelContext_13)->qTileB_0)[8U][_S66];
        bool _S131;
        if(QFORMAT_0 == 2U)
        {
            _S131 = true;
        }
        else
        {
            _S131 = QFORMAT_0 == 3U;
        }
        bool _S132;
        if(_S131)
        {
            _S132 = true;
        }
        else
        {
            _S132 = QFORMAT_0 == 4U;
        }
        if(_S132)
        {
            sumA_2[0U] = sumA_2[0U] + av_8[0U];
        }
        s_2[0U][0U] = fma(av_8[0U], bv_8[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_8[0U], bv_8[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_8[0U], bv_8[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_8[0U], bv_8[3U], s_2[0U][3U]);
        bool _S133;
        if(QFORMAT_0 == 2U)
        {
            _S133 = true;
        }
        else
        {
            _S133 = QFORMAT_0 == 3U;
        }
        bool _S134;
        if(_S133)
        {
            _S134 = true;
        }
        else
        {
            _S134 = QFORMAT_0 == 4U;
        }
        if(_S134)
        {
            sumA_2[1U] = sumA_2[1U] + av_8[1U];
        }
        s_2[1U][0U] = fma(av_8[1U], bv_8[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_8[1U], bv_8[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_8[1U], bv_8[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_8[1U], bv_8[3U], s_2[1U][3U]);
        bool _S135;
        if(QFORMAT_0 == 2U)
        {
            _S135 = true;
        }
        else
        {
            _S135 = QFORMAT_0 == 3U;
        }
        bool _S136;
        if(_S135)
        {
            _S136 = true;
        }
        else
        {
            _S136 = QFORMAT_0 == 4U;
        }
        if(_S136)
        {
            sumA_2[2U] = sumA_2[2U] + av_8[2U];
        }
        s_2[2U][0U] = fma(av_8[2U], bv_8[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_8[2U], bv_8[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_8[2U], bv_8[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_8[2U], bv_8[3U], s_2[2U][3U]);
        bool _S137;
        if(QFORMAT_0 == 2U)
        {
            _S137 = true;
        }
        else
        {
            _S137 = QFORMAT_0 == 3U;
        }
        bool _S138;
        if(_S137)
        {
            _S138 = true;
        }
        else
        {
            _S138 = QFORMAT_0 == 4U;
        }
        if(_S138)
        {
            sumA_2[3U] = sumA_2[3U] + av_8[3U];
        }
        s_2[3U][0U] = fma(av_8[3U], bv_8[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_8[3U], bv_8[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_8[3U], bv_8[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_8[3U], bv_8[3U], s_2[3U][3U]);
        uint w0_9 = (*(&kernelContext_13)->qTileA_0)[9U][ty_0];
        uint w1_9 = (*(&kernelContext_13)->qTileA_0)[9U][_S63];
        thread array<float, int(4)> av_9;
        av_9[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[9U][ty_0]);
        av_9[int(1)] = aLo_0(w1_9);
        av_9[int(2)] = aHi_0(w0_9);
        av_9[int(3)] = aHi_0(w1_9);
        thread array<float, int(4)> bv_9;
        bv_9[0U] = (*(&kernelContext_13)->qTileB_0)[9U][tx_0];
        bv_9[1U] = (*(&kernelContext_13)->qTileB_0)[9U][_S64];
        bv_9[2U] = (*(&kernelContext_13)->qTileB_0)[9U][_S65];
        bv_9[3U] = (*(&kernelContext_13)->qTileB_0)[9U][_S66];
        bool _S139;
        if(QFORMAT_0 == 2U)
        {
            _S139 = true;
        }
        else
        {
            _S139 = QFORMAT_0 == 3U;
        }
        bool _S140;
        if(_S139)
        {
            _S140 = true;
        }
        else
        {
            _S140 = QFORMAT_0 == 4U;
        }
        if(_S140)
        {
            sumA_2[0U] = sumA_2[0U] + av_9[0U];
        }
        s_2[0U][0U] = fma(av_9[0U], bv_9[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_9[0U], bv_9[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_9[0U], bv_9[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_9[0U], bv_9[3U], s_2[0U][3U]);
        bool _S141;
        if(QFORMAT_0 == 2U)
        {
            _S141 = true;
        }
        else
        {
            _S141 = QFORMAT_0 == 3U;
        }
        bool _S142;
        if(_S141)
        {
            _S142 = true;
        }
        else
        {
            _S142 = QFORMAT_0 == 4U;
        }
        if(_S142)
        {
            sumA_2[1U] = sumA_2[1U] + av_9[1U];
        }
        s_2[1U][0U] = fma(av_9[1U], bv_9[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_9[1U], bv_9[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_9[1U], bv_9[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_9[1U], bv_9[3U], s_2[1U][3U]);
        bool _S143;
        if(QFORMAT_0 == 2U)
        {
            _S143 = true;
        }
        else
        {
            _S143 = QFORMAT_0 == 3U;
        }
        bool _S144;
        if(_S143)
        {
            _S144 = true;
        }
        else
        {
            _S144 = QFORMAT_0 == 4U;
        }
        if(_S144)
        {
            sumA_2[2U] = sumA_2[2U] + av_9[2U];
        }
        s_2[2U][0U] = fma(av_9[2U], bv_9[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_9[2U], bv_9[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_9[2U], bv_9[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_9[2U], bv_9[3U], s_2[2U][3U]);
        bool _S145;
        if(QFORMAT_0 == 2U)
        {
            _S145 = true;
        }
        else
        {
            _S145 = QFORMAT_0 == 3U;
        }
        bool _S146;
        if(_S145)
        {
            _S146 = true;
        }
        else
        {
            _S146 = QFORMAT_0 == 4U;
        }
        if(_S146)
        {
            sumA_2[3U] = sumA_2[3U] + av_9[3U];
        }
        s_2[3U][0U] = fma(av_9[3U], bv_9[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_9[3U], bv_9[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_9[3U], bv_9[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_9[3U], bv_9[3U], s_2[3U][3U]);
        uint w0_10 = (*(&kernelContext_13)->qTileA_0)[10U][ty_0];
        uint w1_10 = (*(&kernelContext_13)->qTileA_0)[10U][_S63];
        thread array<float, int(4)> av_10;
        av_10[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[10U][ty_0]);
        av_10[int(1)] = aLo_0(w1_10);
        av_10[int(2)] = aHi_0(w0_10);
        av_10[int(3)] = aHi_0(w1_10);
        thread array<float, int(4)> bv_10;
        bv_10[0U] = (*(&kernelContext_13)->qTileB_0)[10U][tx_0];
        bv_10[1U] = (*(&kernelContext_13)->qTileB_0)[10U][_S64];
        bv_10[2U] = (*(&kernelContext_13)->qTileB_0)[10U][_S65];
        bv_10[3U] = (*(&kernelContext_13)->qTileB_0)[10U][_S66];
        bool _S147;
        if(QFORMAT_0 == 2U)
        {
            _S147 = true;
        }
        else
        {
            _S147 = QFORMAT_0 == 3U;
        }
        bool _S148;
        if(_S147)
        {
            _S148 = true;
        }
        else
        {
            _S148 = QFORMAT_0 == 4U;
        }
        if(_S148)
        {
            sumA_2[0U] = sumA_2[0U] + av_10[0U];
        }
        s_2[0U][0U] = fma(av_10[0U], bv_10[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_10[0U], bv_10[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_10[0U], bv_10[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_10[0U], bv_10[3U], s_2[0U][3U]);
        bool _S149;
        if(QFORMAT_0 == 2U)
        {
            _S149 = true;
        }
        else
        {
            _S149 = QFORMAT_0 == 3U;
        }
        bool _S150;
        if(_S149)
        {
            _S150 = true;
        }
        else
        {
            _S150 = QFORMAT_0 == 4U;
        }
        if(_S150)
        {
            sumA_2[1U] = sumA_2[1U] + av_10[1U];
        }
        s_2[1U][0U] = fma(av_10[1U], bv_10[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_10[1U], bv_10[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_10[1U], bv_10[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_10[1U], bv_10[3U], s_2[1U][3U]);
        bool _S151;
        if(QFORMAT_0 == 2U)
        {
            _S151 = true;
        }
        else
        {
            _S151 = QFORMAT_0 == 3U;
        }
        bool _S152;
        if(_S151)
        {
            _S152 = true;
        }
        else
        {
            _S152 = QFORMAT_0 == 4U;
        }
        if(_S152)
        {
            sumA_2[2U] = sumA_2[2U] + av_10[2U];
        }
        s_2[2U][0U] = fma(av_10[2U], bv_10[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_10[2U], bv_10[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_10[2U], bv_10[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_10[2U], bv_10[3U], s_2[2U][3U]);
        bool _S153;
        if(QFORMAT_0 == 2U)
        {
            _S153 = true;
        }
        else
        {
            _S153 = QFORMAT_0 == 3U;
        }
        bool _S154;
        if(_S153)
        {
            _S154 = true;
        }
        else
        {
            _S154 = QFORMAT_0 == 4U;
        }
        if(_S154)
        {
            sumA_2[3U] = sumA_2[3U] + av_10[3U];
        }
        s_2[3U][0U] = fma(av_10[3U], bv_10[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_10[3U], bv_10[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_10[3U], bv_10[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_10[3U], bv_10[3U], s_2[3U][3U]);
        uint w0_11 = (*(&kernelContext_13)->qTileA_0)[11U][ty_0];
        uint w1_11 = (*(&kernelContext_13)->qTileA_0)[11U][_S63];
        thread array<float, int(4)> av_11;
        av_11[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[11U][ty_0]);
        av_11[int(1)] = aLo_0(w1_11);
        av_11[int(2)] = aHi_0(w0_11);
        av_11[int(3)] = aHi_0(w1_11);
        thread array<float, int(4)> bv_11;
        bv_11[0U] = (*(&kernelContext_13)->qTileB_0)[11U][tx_0];
        bv_11[1U] = (*(&kernelContext_13)->qTileB_0)[11U][_S64];
        bv_11[2U] = (*(&kernelContext_13)->qTileB_0)[11U][_S65];
        bv_11[3U] = (*(&kernelContext_13)->qTileB_0)[11U][_S66];
        bool _S155;
        if(QFORMAT_0 == 2U)
        {
            _S155 = true;
        }
        else
        {
            _S155 = QFORMAT_0 == 3U;
        }
        bool _S156;
        if(_S155)
        {
            _S156 = true;
        }
        else
        {
            _S156 = QFORMAT_0 == 4U;
        }
        if(_S156)
        {
            sumA_2[0U] = sumA_2[0U] + av_11[0U];
        }
        s_2[0U][0U] = fma(av_11[0U], bv_11[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_11[0U], bv_11[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_11[0U], bv_11[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_11[0U], bv_11[3U], s_2[0U][3U]);
        bool _S157;
        if(QFORMAT_0 == 2U)
        {
            _S157 = true;
        }
        else
        {
            _S157 = QFORMAT_0 == 3U;
        }
        bool _S158;
        if(_S157)
        {
            _S158 = true;
        }
        else
        {
            _S158 = QFORMAT_0 == 4U;
        }
        if(_S158)
        {
            sumA_2[1U] = sumA_2[1U] + av_11[1U];
        }
        s_2[1U][0U] = fma(av_11[1U], bv_11[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_11[1U], bv_11[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_11[1U], bv_11[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_11[1U], bv_11[3U], s_2[1U][3U]);
        bool _S159;
        if(QFORMAT_0 == 2U)
        {
            _S159 = true;
        }
        else
        {
            _S159 = QFORMAT_0 == 3U;
        }
        bool _S160;
        if(_S159)
        {
            _S160 = true;
        }
        else
        {
            _S160 = QFORMAT_0 == 4U;
        }
        if(_S160)
        {
            sumA_2[2U] = sumA_2[2U] + av_11[2U];
        }
        s_2[2U][0U] = fma(av_11[2U], bv_11[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_11[2U], bv_11[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_11[2U], bv_11[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_11[2U], bv_11[3U], s_2[2U][3U]);
        bool _S161;
        if(QFORMAT_0 == 2U)
        {
            _S161 = true;
        }
        else
        {
            _S161 = QFORMAT_0 == 3U;
        }
        bool _S162;
        if(_S161)
        {
            _S162 = true;
        }
        else
        {
            _S162 = QFORMAT_0 == 4U;
        }
        if(_S162)
        {
            sumA_2[3U] = sumA_2[3U] + av_11[3U];
        }
        s_2[3U][0U] = fma(av_11[3U], bv_11[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_11[3U], bv_11[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_11[3U], bv_11[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_11[3U], bv_11[3U], s_2[3U][3U]);
        uint w0_12 = (*(&kernelContext_13)->qTileA_0)[12U][ty_0];
        uint w1_12 = (*(&kernelContext_13)->qTileA_0)[12U][_S63];
        thread array<float, int(4)> av_12;
        av_12[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[12U][ty_0]);
        av_12[int(1)] = aLo_0(w1_12);
        av_12[int(2)] = aHi_0(w0_12);
        av_12[int(3)] = aHi_0(w1_12);
        thread array<float, int(4)> bv_12;
        bv_12[0U] = (*(&kernelContext_13)->qTileB_0)[12U][tx_0];
        bv_12[1U] = (*(&kernelContext_13)->qTileB_0)[12U][_S64];
        bv_12[2U] = (*(&kernelContext_13)->qTileB_0)[12U][_S65];
        bv_12[3U] = (*(&kernelContext_13)->qTileB_0)[12U][_S66];
        bool _S163;
        if(QFORMAT_0 == 2U)
        {
            _S163 = true;
        }
        else
        {
            _S163 = QFORMAT_0 == 3U;
        }
        bool _S164;
        if(_S163)
        {
            _S164 = true;
        }
        else
        {
            _S164 = QFORMAT_0 == 4U;
        }
        if(_S164)
        {
            sumA_2[0U] = sumA_2[0U] + av_12[0U];
        }
        s_2[0U][0U] = fma(av_12[0U], bv_12[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_12[0U], bv_12[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_12[0U], bv_12[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_12[0U], bv_12[3U], s_2[0U][3U]);
        bool _S165;
        if(QFORMAT_0 == 2U)
        {
            _S165 = true;
        }
        else
        {
            _S165 = QFORMAT_0 == 3U;
        }
        bool _S166;
        if(_S165)
        {
            _S166 = true;
        }
        else
        {
            _S166 = QFORMAT_0 == 4U;
        }
        if(_S166)
        {
            sumA_2[1U] = sumA_2[1U] + av_12[1U];
        }
        s_2[1U][0U] = fma(av_12[1U], bv_12[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_12[1U], bv_12[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_12[1U], bv_12[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_12[1U], bv_12[3U], s_2[1U][3U]);
        bool _S167;
        if(QFORMAT_0 == 2U)
        {
            _S167 = true;
        }
        else
        {
            _S167 = QFORMAT_0 == 3U;
        }
        bool _S168;
        if(_S167)
        {
            _S168 = true;
        }
        else
        {
            _S168 = QFORMAT_0 == 4U;
        }
        if(_S168)
        {
            sumA_2[2U] = sumA_2[2U] + av_12[2U];
        }
        s_2[2U][0U] = fma(av_12[2U], bv_12[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_12[2U], bv_12[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_12[2U], bv_12[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_12[2U], bv_12[3U], s_2[2U][3U]);
        bool _S169;
        if(QFORMAT_0 == 2U)
        {
            _S169 = true;
        }
        else
        {
            _S169 = QFORMAT_0 == 3U;
        }
        bool _S170;
        if(_S169)
        {
            _S170 = true;
        }
        else
        {
            _S170 = QFORMAT_0 == 4U;
        }
        if(_S170)
        {
            sumA_2[3U] = sumA_2[3U] + av_12[3U];
        }
        s_2[3U][0U] = fma(av_12[3U], bv_12[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_12[3U], bv_12[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_12[3U], bv_12[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_12[3U], bv_12[3U], s_2[3U][3U]);
        uint w0_13 = (*(&kernelContext_13)->qTileA_0)[13U][ty_0];
        uint w1_13 = (*(&kernelContext_13)->qTileA_0)[13U][_S63];
        thread array<float, int(4)> av_13;
        av_13[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[13U][ty_0]);
        av_13[int(1)] = aLo_0(w1_13);
        av_13[int(2)] = aHi_0(w0_13);
        av_13[int(3)] = aHi_0(w1_13);
        thread array<float, int(4)> bv_13;
        bv_13[0U] = (*(&kernelContext_13)->qTileB_0)[13U][tx_0];
        bv_13[1U] = (*(&kernelContext_13)->qTileB_0)[13U][_S64];
        bv_13[2U] = (*(&kernelContext_13)->qTileB_0)[13U][_S65];
        bv_13[3U] = (*(&kernelContext_13)->qTileB_0)[13U][_S66];
        bool _S171;
        if(QFORMAT_0 == 2U)
        {
            _S171 = true;
        }
        else
        {
            _S171 = QFORMAT_0 == 3U;
        }
        bool _S172;
        if(_S171)
        {
            _S172 = true;
        }
        else
        {
            _S172 = QFORMAT_0 == 4U;
        }
        if(_S172)
        {
            sumA_2[0U] = sumA_2[0U] + av_13[0U];
        }
        s_2[0U][0U] = fma(av_13[0U], bv_13[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_13[0U], bv_13[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_13[0U], bv_13[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_13[0U], bv_13[3U], s_2[0U][3U]);
        bool _S173;
        if(QFORMAT_0 == 2U)
        {
            _S173 = true;
        }
        else
        {
            _S173 = QFORMAT_0 == 3U;
        }
        bool _S174;
        if(_S173)
        {
            _S174 = true;
        }
        else
        {
            _S174 = QFORMAT_0 == 4U;
        }
        if(_S174)
        {
            sumA_2[1U] = sumA_2[1U] + av_13[1U];
        }
        s_2[1U][0U] = fma(av_13[1U], bv_13[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_13[1U], bv_13[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_13[1U], bv_13[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_13[1U], bv_13[3U], s_2[1U][3U]);
        bool _S175;
        if(QFORMAT_0 == 2U)
        {
            _S175 = true;
        }
        else
        {
            _S175 = QFORMAT_0 == 3U;
        }
        bool _S176;
        if(_S175)
        {
            _S176 = true;
        }
        else
        {
            _S176 = QFORMAT_0 == 4U;
        }
        if(_S176)
        {
            sumA_2[2U] = sumA_2[2U] + av_13[2U];
        }
        s_2[2U][0U] = fma(av_13[2U], bv_13[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_13[2U], bv_13[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_13[2U], bv_13[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_13[2U], bv_13[3U], s_2[2U][3U]);
        bool _S177;
        if(QFORMAT_0 == 2U)
        {
            _S177 = true;
        }
        else
        {
            _S177 = QFORMAT_0 == 3U;
        }
        bool _S178;
        if(_S177)
        {
            _S178 = true;
        }
        else
        {
            _S178 = QFORMAT_0 == 4U;
        }
        if(_S178)
        {
            sumA_2[3U] = sumA_2[3U] + av_13[3U];
        }
        s_2[3U][0U] = fma(av_13[3U], bv_13[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_13[3U], bv_13[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_13[3U], bv_13[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_13[3U], bv_13[3U], s_2[3U][3U]);
        uint w0_14 = (*(&kernelContext_13)->qTileA_0)[14U][ty_0];
        uint w1_14 = (*(&kernelContext_13)->qTileA_0)[14U][_S63];
        thread array<float, int(4)> av_14;
        av_14[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[14U][ty_0]);
        av_14[int(1)] = aLo_0(w1_14);
        av_14[int(2)] = aHi_0(w0_14);
        av_14[int(3)] = aHi_0(w1_14);
        thread array<float, int(4)> bv_14;
        bv_14[0U] = (*(&kernelContext_13)->qTileB_0)[14U][tx_0];
        bv_14[1U] = (*(&kernelContext_13)->qTileB_0)[14U][_S64];
        bv_14[2U] = (*(&kernelContext_13)->qTileB_0)[14U][_S65];
        bv_14[3U] = (*(&kernelContext_13)->qTileB_0)[14U][_S66];
        bool _S179;
        if(QFORMAT_0 == 2U)
        {
            _S179 = true;
        }
        else
        {
            _S179 = QFORMAT_0 == 3U;
        }
        bool _S180;
        if(_S179)
        {
            _S180 = true;
        }
        else
        {
            _S180 = QFORMAT_0 == 4U;
        }
        if(_S180)
        {
            sumA_2[0U] = sumA_2[0U] + av_14[0U];
        }
        s_2[0U][0U] = fma(av_14[0U], bv_14[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_14[0U], bv_14[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_14[0U], bv_14[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_14[0U], bv_14[3U], s_2[0U][3U]);
        bool _S181;
        if(QFORMAT_0 == 2U)
        {
            _S181 = true;
        }
        else
        {
            _S181 = QFORMAT_0 == 3U;
        }
        bool _S182;
        if(_S181)
        {
            _S182 = true;
        }
        else
        {
            _S182 = QFORMAT_0 == 4U;
        }
        if(_S182)
        {
            sumA_2[1U] = sumA_2[1U] + av_14[1U];
        }
        s_2[1U][0U] = fma(av_14[1U], bv_14[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_14[1U], bv_14[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_14[1U], bv_14[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_14[1U], bv_14[3U], s_2[1U][3U]);
        bool _S183;
        if(QFORMAT_0 == 2U)
        {
            _S183 = true;
        }
        else
        {
            _S183 = QFORMAT_0 == 3U;
        }
        bool _S184;
        if(_S183)
        {
            _S184 = true;
        }
        else
        {
            _S184 = QFORMAT_0 == 4U;
        }
        if(_S184)
        {
            sumA_2[2U] = sumA_2[2U] + av_14[2U];
        }
        s_2[2U][0U] = fma(av_14[2U], bv_14[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_14[2U], bv_14[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_14[2U], bv_14[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_14[2U], bv_14[3U], s_2[2U][3U]);
        bool _S185;
        if(QFORMAT_0 == 2U)
        {
            _S185 = true;
        }
        else
        {
            _S185 = QFORMAT_0 == 3U;
        }
        bool _S186;
        if(_S185)
        {
            _S186 = true;
        }
        else
        {
            _S186 = QFORMAT_0 == 4U;
        }
        if(_S186)
        {
            sumA_2[3U] = sumA_2[3U] + av_14[3U];
        }
        s_2[3U][0U] = fma(av_14[3U], bv_14[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_14[3U], bv_14[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_14[3U], bv_14[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_14[3U], bv_14[3U], s_2[3U][3U]);
        uint w0_15 = (*(&kernelContext_13)->qTileA_0)[15U][ty_0];
        uint w1_15 = (*(&kernelContext_13)->qTileA_0)[15U][_S63];
        thread array<float, int(4)> av_15;
        av_15[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[15U][ty_0]);
        av_15[int(1)] = aLo_0(w1_15);
        av_15[int(2)] = aHi_0(w0_15);
        av_15[int(3)] = aHi_0(w1_15);
        thread array<float, int(4)> bv_15;
        bv_15[0U] = (*(&kernelContext_13)->qTileB_0)[15U][tx_0];
        bv_15[1U] = (*(&kernelContext_13)->qTileB_0)[15U][_S64];
        bv_15[2U] = (*(&kernelContext_13)->qTileB_0)[15U][_S65];
        bv_15[3U] = (*(&kernelContext_13)->qTileB_0)[15U][_S66];
        bool _S187;
        if(QFORMAT_0 == 2U)
        {
            _S187 = true;
        }
        else
        {
            _S187 = QFORMAT_0 == 3U;
        }
        bool _S188;
        if(_S187)
        {
            _S188 = true;
        }
        else
        {
            _S188 = QFORMAT_0 == 4U;
        }
        if(_S188)
        {
            sumA_2[0U] = sumA_2[0U] + av_15[0U];
        }
        s_2[0U][0U] = fma(av_15[0U], bv_15[0U], s_2[0U][0U]);
        s_2[0U][1U] = fma(av_15[0U], bv_15[1U], s_2[0U][1U]);
        s_2[0U][2U] = fma(av_15[0U], bv_15[2U], s_2[0U][2U]);
        s_2[0U][3U] = fma(av_15[0U], bv_15[3U], s_2[0U][3U]);
        bool _S189;
        if(QFORMAT_0 == 2U)
        {
            _S189 = true;
        }
        else
        {
            _S189 = QFORMAT_0 == 3U;
        }
        bool _S190;
        if(_S189)
        {
            _S190 = true;
        }
        else
        {
            _S190 = QFORMAT_0 == 4U;
        }
        if(_S190)
        {
            sumA_2[1U] = sumA_2[1U] + av_15[1U];
        }
        s_2[1U][0U] = fma(av_15[1U], bv_15[0U], s_2[1U][0U]);
        s_2[1U][1U] = fma(av_15[1U], bv_15[1U], s_2[1U][1U]);
        s_2[1U][2U] = fma(av_15[1U], bv_15[2U], s_2[1U][2U]);
        s_2[1U][3U] = fma(av_15[1U], bv_15[3U], s_2[1U][3U]);
        bool _S191;
        if(QFORMAT_0 == 2U)
        {
            _S191 = true;
        }
        else
        {
            _S191 = QFORMAT_0 == 3U;
        }
        bool _S192;
        if(_S191)
        {
            _S192 = true;
        }
        else
        {
            _S192 = QFORMAT_0 == 4U;
        }
        if(_S192)
        {
            sumA_2[2U] = sumA_2[2U] + av_15[2U];
        }
        s_2[2U][0U] = fma(av_15[2U], bv_15[0U], s_2[2U][0U]);
        s_2[2U][1U] = fma(av_15[2U], bv_15[1U], s_2[2U][1U]);
        s_2[2U][2U] = fma(av_15[2U], bv_15[2U], s_2[2U][2U]);
        s_2[2U][3U] = fma(av_15[2U], bv_15[3U], s_2[2U][3U]);
        bool _S193;
        if(QFORMAT_0 == 2U)
        {
            _S193 = true;
        }
        else
        {
            _S193 = QFORMAT_0 == 3U;
        }
        bool _S194;
        if(_S193)
        {
            _S194 = true;
        }
        else
        {
            _S194 = QFORMAT_0 == 4U;
        }
        if(_S194)
        {
            sumA_2[3U] = sumA_2[3U] + av_15[3U];
        }
        s_2[3U][0U] = fma(av_15[3U], bv_15[0U], s_2[3U][0U]);
        s_2[3U][1U] = fma(av_15[3U], bv_15[1U], s_2[3U][1U]);
        s_2[3U][2U] = fma(av_15[3U], bv_15[2U], s_2[3U][2U]);
        s_2[3U][3U] = fma(av_15[3U], bv_15[3U], s_2[3U][3U]);
        uint w0_16 = (*(&kernelContext_13)->qTileA_0)[16U][ty_0];
        uint w1_16 = (*(&kernelContext_13)->qTileA_0)[16U][_S63];
        thread array<float, int(4)> av_16;
        av_16[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[16U][ty_0]);
        av_16[int(1)] = aLo_0(w1_16);
        av_16[int(2)] = aHi_0(w0_16);
        av_16[int(3)] = aHi_0(w1_16);
        thread array<float, int(4)> bv_16;
        bv_16[0U] = (*(&kernelContext_13)->qTileB_0)[16U][tx_0];
        bv_16[1U] = (*(&kernelContext_13)->qTileB_0)[16U][_S64];
        bv_16[2U] = (*(&kernelContext_13)->qTileB_0)[16U][_S65];
        bv_16[3U] = (*(&kernelContext_13)->qTileB_0)[16U][_S66];
        bool _S195;
        if(QFORMAT_0 == 2U)
        {
            _S195 = true;
        }
        else
        {
            _S195 = QFORMAT_0 == 3U;
        }
        bool _S196;
        if(_S195)
        {
            _S196 = true;
        }
        else
        {
            _S196 = QFORMAT_0 == 4U;
        }
        if(_S196)
        {
            sumA_2[0U] = sumA_2[0U] + av_16[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_16[0U], bv_16[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_16[0U], bv_16[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_16[0U], bv_16[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_16[0U], bv_16[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_16[0U], bv_16[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_16[0U], bv_16[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_16[0U], bv_16[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_16[0U], bv_16[3U], s_2[0U][3U]);
        }
        bool _S197;
        if(QFORMAT_0 == 2U)
        {
            _S197 = true;
        }
        else
        {
            _S197 = QFORMAT_0 == 3U;
        }
        bool _S198;
        if(_S197)
        {
            _S198 = true;
        }
        else
        {
            _S198 = QFORMAT_0 == 4U;
        }
        if(_S198)
        {
            sumA_2[1U] = sumA_2[1U] + av_16[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_16[1U], bv_16[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_16[1U], bv_16[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_16[1U], bv_16[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_16[1U], bv_16[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_16[1U], bv_16[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_16[1U], bv_16[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_16[1U], bv_16[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_16[1U], bv_16[3U], s_2[1U][3U]);
        }
        bool _S199;
        if(QFORMAT_0 == 2U)
        {
            _S199 = true;
        }
        else
        {
            _S199 = QFORMAT_0 == 3U;
        }
        bool _S200;
        if(_S199)
        {
            _S200 = true;
        }
        else
        {
            _S200 = QFORMAT_0 == 4U;
        }
        if(_S200)
        {
            sumA_2[2U] = sumA_2[2U] + av_16[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_16[2U], bv_16[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_16[2U], bv_16[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_16[2U], bv_16[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_16[2U], bv_16[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_16[2U], bv_16[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_16[2U], bv_16[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_16[2U], bv_16[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_16[2U], bv_16[3U], s_2[2U][3U]);
        }
        bool _S201;
        if(QFORMAT_0 == 2U)
        {
            _S201 = true;
        }
        else
        {
            _S201 = QFORMAT_0 == 3U;
        }
        bool _S202;
        if(_S201)
        {
            _S202 = true;
        }
        else
        {
            _S202 = QFORMAT_0 == 4U;
        }
        if(_S202)
        {
            sumA_2[3U] = sumA_2[3U] + av_16[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_16[3U], bv_16[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_16[3U], bv_16[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_16[3U], bv_16[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_16[3U], bv_16[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_16[3U], bv_16[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_16[3U], bv_16[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_16[3U], bv_16[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_16[3U], bv_16[3U], s_2[3U][3U]);
        }
        uint w0_17 = (*(&kernelContext_13)->qTileA_0)[17U][ty_0];
        uint w1_17 = (*(&kernelContext_13)->qTileA_0)[17U][_S63];
        thread array<float, int(4)> av_17;
        av_17[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[17U][ty_0]);
        av_17[int(1)] = aLo_0(w1_17);
        av_17[int(2)] = aHi_0(w0_17);
        av_17[int(3)] = aHi_0(w1_17);
        thread array<float, int(4)> bv_17;
        bv_17[0U] = (*(&kernelContext_13)->qTileB_0)[17U][tx_0];
        bv_17[1U] = (*(&kernelContext_13)->qTileB_0)[17U][_S64];
        bv_17[2U] = (*(&kernelContext_13)->qTileB_0)[17U][_S65];
        bv_17[3U] = (*(&kernelContext_13)->qTileB_0)[17U][_S66];
        bool _S203;
        if(QFORMAT_0 == 2U)
        {
            _S203 = true;
        }
        else
        {
            _S203 = QFORMAT_0 == 3U;
        }
        bool _S204;
        if(_S203)
        {
            _S204 = true;
        }
        else
        {
            _S204 = QFORMAT_0 == 4U;
        }
        if(_S204)
        {
            sumA_2[0U] = sumA_2[0U] + av_17[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_17[0U], bv_17[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_17[0U], bv_17[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_17[0U], bv_17[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_17[0U], bv_17[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_17[0U], bv_17[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_17[0U], bv_17[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_17[0U], bv_17[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_17[0U], bv_17[3U], s_2[0U][3U]);
        }
        bool _S205;
        if(QFORMAT_0 == 2U)
        {
            _S205 = true;
        }
        else
        {
            _S205 = QFORMAT_0 == 3U;
        }
        bool _S206;
        if(_S205)
        {
            _S206 = true;
        }
        else
        {
            _S206 = QFORMAT_0 == 4U;
        }
        if(_S206)
        {
            sumA_2[1U] = sumA_2[1U] + av_17[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_17[1U], bv_17[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_17[1U], bv_17[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_17[1U], bv_17[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_17[1U], bv_17[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_17[1U], bv_17[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_17[1U], bv_17[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_17[1U], bv_17[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_17[1U], bv_17[3U], s_2[1U][3U]);
        }
        bool _S207;
        if(QFORMAT_0 == 2U)
        {
            _S207 = true;
        }
        else
        {
            _S207 = QFORMAT_0 == 3U;
        }
        bool _S208;
        if(_S207)
        {
            _S208 = true;
        }
        else
        {
            _S208 = QFORMAT_0 == 4U;
        }
        if(_S208)
        {
            sumA_2[2U] = sumA_2[2U] + av_17[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_17[2U], bv_17[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_17[2U], bv_17[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_17[2U], bv_17[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_17[2U], bv_17[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_17[2U], bv_17[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_17[2U], bv_17[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_17[2U], bv_17[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_17[2U], bv_17[3U], s_2[2U][3U]);
        }
        bool _S209;
        if(QFORMAT_0 == 2U)
        {
            _S209 = true;
        }
        else
        {
            _S209 = QFORMAT_0 == 3U;
        }
        bool _S210;
        if(_S209)
        {
            _S210 = true;
        }
        else
        {
            _S210 = QFORMAT_0 == 4U;
        }
        if(_S210)
        {
            sumA_2[3U] = sumA_2[3U] + av_17[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_17[3U], bv_17[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_17[3U], bv_17[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_17[3U], bv_17[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_17[3U], bv_17[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_17[3U], bv_17[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_17[3U], bv_17[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_17[3U], bv_17[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_17[3U], bv_17[3U], s_2[3U][3U]);
        }
        uint w0_18 = (*(&kernelContext_13)->qTileA_0)[18U][ty_0];
        uint w1_18 = (*(&kernelContext_13)->qTileA_0)[18U][_S63];
        thread array<float, int(4)> av_18;
        av_18[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[18U][ty_0]);
        av_18[int(1)] = aLo_0(w1_18);
        av_18[int(2)] = aHi_0(w0_18);
        av_18[int(3)] = aHi_0(w1_18);
        thread array<float, int(4)> bv_18;
        bv_18[0U] = (*(&kernelContext_13)->qTileB_0)[18U][tx_0];
        bv_18[1U] = (*(&kernelContext_13)->qTileB_0)[18U][_S64];
        bv_18[2U] = (*(&kernelContext_13)->qTileB_0)[18U][_S65];
        bv_18[3U] = (*(&kernelContext_13)->qTileB_0)[18U][_S66];
        bool _S211;
        if(QFORMAT_0 == 2U)
        {
            _S211 = true;
        }
        else
        {
            _S211 = QFORMAT_0 == 3U;
        }
        bool _S212;
        if(_S211)
        {
            _S212 = true;
        }
        else
        {
            _S212 = QFORMAT_0 == 4U;
        }
        if(_S212)
        {
            sumA_2[0U] = sumA_2[0U] + av_18[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_18[0U], bv_18[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_18[0U], bv_18[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_18[0U], bv_18[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_18[0U], bv_18[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_18[0U], bv_18[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_18[0U], bv_18[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_18[0U], bv_18[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_18[0U], bv_18[3U], s_2[0U][3U]);
        }
        bool _S213;
        if(QFORMAT_0 == 2U)
        {
            _S213 = true;
        }
        else
        {
            _S213 = QFORMAT_0 == 3U;
        }
        bool _S214;
        if(_S213)
        {
            _S214 = true;
        }
        else
        {
            _S214 = QFORMAT_0 == 4U;
        }
        if(_S214)
        {
            sumA_2[1U] = sumA_2[1U] + av_18[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_18[1U], bv_18[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_18[1U], bv_18[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_18[1U], bv_18[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_18[1U], bv_18[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_18[1U], bv_18[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_18[1U], bv_18[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_18[1U], bv_18[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_18[1U], bv_18[3U], s_2[1U][3U]);
        }
        bool _S215;
        if(QFORMAT_0 == 2U)
        {
            _S215 = true;
        }
        else
        {
            _S215 = QFORMAT_0 == 3U;
        }
        bool _S216;
        if(_S215)
        {
            _S216 = true;
        }
        else
        {
            _S216 = QFORMAT_0 == 4U;
        }
        if(_S216)
        {
            sumA_2[2U] = sumA_2[2U] + av_18[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_18[2U], bv_18[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_18[2U], bv_18[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_18[2U], bv_18[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_18[2U], bv_18[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_18[2U], bv_18[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_18[2U], bv_18[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_18[2U], bv_18[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_18[2U], bv_18[3U], s_2[2U][3U]);
        }
        bool _S217;
        if(QFORMAT_0 == 2U)
        {
            _S217 = true;
        }
        else
        {
            _S217 = QFORMAT_0 == 3U;
        }
        bool _S218;
        if(_S217)
        {
            _S218 = true;
        }
        else
        {
            _S218 = QFORMAT_0 == 4U;
        }
        if(_S218)
        {
            sumA_2[3U] = sumA_2[3U] + av_18[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_18[3U], bv_18[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_18[3U], bv_18[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_18[3U], bv_18[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_18[3U], bv_18[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_18[3U], bv_18[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_18[3U], bv_18[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_18[3U], bv_18[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_18[3U], bv_18[3U], s_2[3U][3U]);
        }
        uint w0_19 = (*(&kernelContext_13)->qTileA_0)[19U][ty_0];
        uint w1_19 = (*(&kernelContext_13)->qTileA_0)[19U][_S63];
        thread array<float, int(4)> av_19;
        av_19[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[19U][ty_0]);
        av_19[int(1)] = aLo_0(w1_19);
        av_19[int(2)] = aHi_0(w0_19);
        av_19[int(3)] = aHi_0(w1_19);
        thread array<float, int(4)> bv_19;
        bv_19[0U] = (*(&kernelContext_13)->qTileB_0)[19U][tx_0];
        bv_19[1U] = (*(&kernelContext_13)->qTileB_0)[19U][_S64];
        bv_19[2U] = (*(&kernelContext_13)->qTileB_0)[19U][_S65];
        bv_19[3U] = (*(&kernelContext_13)->qTileB_0)[19U][_S66];
        bool _S219;
        if(QFORMAT_0 == 2U)
        {
            _S219 = true;
        }
        else
        {
            _S219 = QFORMAT_0 == 3U;
        }
        bool _S220;
        if(_S219)
        {
            _S220 = true;
        }
        else
        {
            _S220 = QFORMAT_0 == 4U;
        }
        if(_S220)
        {
            sumA_2[0U] = sumA_2[0U] + av_19[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_19[0U], bv_19[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_19[0U], bv_19[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_19[0U], bv_19[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_19[0U], bv_19[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_19[0U], bv_19[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_19[0U], bv_19[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_19[0U], bv_19[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_19[0U], bv_19[3U], s_2[0U][3U]);
        }
        bool _S221;
        if(QFORMAT_0 == 2U)
        {
            _S221 = true;
        }
        else
        {
            _S221 = QFORMAT_0 == 3U;
        }
        bool _S222;
        if(_S221)
        {
            _S222 = true;
        }
        else
        {
            _S222 = QFORMAT_0 == 4U;
        }
        if(_S222)
        {
            sumA_2[1U] = sumA_2[1U] + av_19[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_19[1U], bv_19[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_19[1U], bv_19[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_19[1U], bv_19[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_19[1U], bv_19[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_19[1U], bv_19[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_19[1U], bv_19[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_19[1U], bv_19[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_19[1U], bv_19[3U], s_2[1U][3U]);
        }
        bool _S223;
        if(QFORMAT_0 == 2U)
        {
            _S223 = true;
        }
        else
        {
            _S223 = QFORMAT_0 == 3U;
        }
        bool _S224;
        if(_S223)
        {
            _S224 = true;
        }
        else
        {
            _S224 = QFORMAT_0 == 4U;
        }
        if(_S224)
        {
            sumA_2[2U] = sumA_2[2U] + av_19[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_19[2U], bv_19[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_19[2U], bv_19[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_19[2U], bv_19[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_19[2U], bv_19[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_19[2U], bv_19[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_19[2U], bv_19[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_19[2U], bv_19[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_19[2U], bv_19[3U], s_2[2U][3U]);
        }
        bool _S225;
        if(QFORMAT_0 == 2U)
        {
            _S225 = true;
        }
        else
        {
            _S225 = QFORMAT_0 == 3U;
        }
        bool _S226;
        if(_S225)
        {
            _S226 = true;
        }
        else
        {
            _S226 = QFORMAT_0 == 4U;
        }
        if(_S226)
        {
            sumA_2[3U] = sumA_2[3U] + av_19[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_19[3U], bv_19[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_19[3U], bv_19[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_19[3U], bv_19[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_19[3U], bv_19[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_19[3U], bv_19[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_19[3U], bv_19[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_19[3U], bv_19[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_19[3U], bv_19[3U], s_2[3U][3U]);
        }
        uint w0_20 = (*(&kernelContext_13)->qTileA_0)[20U][ty_0];
        uint w1_20 = (*(&kernelContext_13)->qTileA_0)[20U][_S63];
        thread array<float, int(4)> av_20;
        av_20[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[20U][ty_0]);
        av_20[int(1)] = aLo_0(w1_20);
        av_20[int(2)] = aHi_0(w0_20);
        av_20[int(3)] = aHi_0(w1_20);
        thread array<float, int(4)> bv_20;
        bv_20[0U] = (*(&kernelContext_13)->qTileB_0)[20U][tx_0];
        bv_20[1U] = (*(&kernelContext_13)->qTileB_0)[20U][_S64];
        bv_20[2U] = (*(&kernelContext_13)->qTileB_0)[20U][_S65];
        bv_20[3U] = (*(&kernelContext_13)->qTileB_0)[20U][_S66];
        bool _S227;
        if(QFORMAT_0 == 2U)
        {
            _S227 = true;
        }
        else
        {
            _S227 = QFORMAT_0 == 3U;
        }
        bool _S228;
        if(_S227)
        {
            _S228 = true;
        }
        else
        {
            _S228 = QFORMAT_0 == 4U;
        }
        if(_S228)
        {
            sumA_2[0U] = sumA_2[0U] + av_20[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_20[0U], bv_20[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_20[0U], bv_20[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_20[0U], bv_20[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_20[0U], bv_20[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_20[0U], bv_20[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_20[0U], bv_20[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_20[0U], bv_20[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_20[0U], bv_20[3U], s_2[0U][3U]);
        }
        bool _S229;
        if(QFORMAT_0 == 2U)
        {
            _S229 = true;
        }
        else
        {
            _S229 = QFORMAT_0 == 3U;
        }
        bool _S230;
        if(_S229)
        {
            _S230 = true;
        }
        else
        {
            _S230 = QFORMAT_0 == 4U;
        }
        if(_S230)
        {
            sumA_2[1U] = sumA_2[1U] + av_20[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_20[1U], bv_20[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_20[1U], bv_20[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_20[1U], bv_20[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_20[1U], bv_20[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_20[1U], bv_20[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_20[1U], bv_20[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_20[1U], bv_20[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_20[1U], bv_20[3U], s_2[1U][3U]);
        }
        bool _S231;
        if(QFORMAT_0 == 2U)
        {
            _S231 = true;
        }
        else
        {
            _S231 = QFORMAT_0 == 3U;
        }
        bool _S232;
        if(_S231)
        {
            _S232 = true;
        }
        else
        {
            _S232 = QFORMAT_0 == 4U;
        }
        if(_S232)
        {
            sumA_2[2U] = sumA_2[2U] + av_20[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_20[2U], bv_20[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_20[2U], bv_20[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_20[2U], bv_20[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_20[2U], bv_20[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_20[2U], bv_20[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_20[2U], bv_20[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_20[2U], bv_20[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_20[2U], bv_20[3U], s_2[2U][3U]);
        }
        bool _S233;
        if(QFORMAT_0 == 2U)
        {
            _S233 = true;
        }
        else
        {
            _S233 = QFORMAT_0 == 3U;
        }
        bool _S234;
        if(_S233)
        {
            _S234 = true;
        }
        else
        {
            _S234 = QFORMAT_0 == 4U;
        }
        if(_S234)
        {
            sumA_2[3U] = sumA_2[3U] + av_20[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_20[3U], bv_20[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_20[3U], bv_20[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_20[3U], bv_20[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_20[3U], bv_20[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_20[3U], bv_20[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_20[3U], bv_20[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_20[3U], bv_20[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_20[3U], bv_20[3U], s_2[3U][3U]);
        }
        uint w0_21 = (*(&kernelContext_13)->qTileA_0)[21U][ty_0];
        uint w1_21 = (*(&kernelContext_13)->qTileA_0)[21U][_S63];
        thread array<float, int(4)> av_21;
        av_21[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[21U][ty_0]);
        av_21[int(1)] = aLo_0(w1_21);
        av_21[int(2)] = aHi_0(w0_21);
        av_21[int(3)] = aHi_0(w1_21);
        thread array<float, int(4)> bv_21;
        bv_21[0U] = (*(&kernelContext_13)->qTileB_0)[21U][tx_0];
        bv_21[1U] = (*(&kernelContext_13)->qTileB_0)[21U][_S64];
        bv_21[2U] = (*(&kernelContext_13)->qTileB_0)[21U][_S65];
        bv_21[3U] = (*(&kernelContext_13)->qTileB_0)[21U][_S66];
        bool _S235;
        if(QFORMAT_0 == 2U)
        {
            _S235 = true;
        }
        else
        {
            _S235 = QFORMAT_0 == 3U;
        }
        bool _S236;
        if(_S235)
        {
            _S236 = true;
        }
        else
        {
            _S236 = QFORMAT_0 == 4U;
        }
        if(_S236)
        {
            sumA_2[0U] = sumA_2[0U] + av_21[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_21[0U], bv_21[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_21[0U], bv_21[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_21[0U], bv_21[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_21[0U], bv_21[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_21[0U], bv_21[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_21[0U], bv_21[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_21[0U], bv_21[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_21[0U], bv_21[3U], s_2[0U][3U]);
        }
        bool _S237;
        if(QFORMAT_0 == 2U)
        {
            _S237 = true;
        }
        else
        {
            _S237 = QFORMAT_0 == 3U;
        }
        bool _S238;
        if(_S237)
        {
            _S238 = true;
        }
        else
        {
            _S238 = QFORMAT_0 == 4U;
        }
        if(_S238)
        {
            sumA_2[1U] = sumA_2[1U] + av_21[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_21[1U], bv_21[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_21[1U], bv_21[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_21[1U], bv_21[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_21[1U], bv_21[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_21[1U], bv_21[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_21[1U], bv_21[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_21[1U], bv_21[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_21[1U], bv_21[3U], s_2[1U][3U]);
        }
        bool _S239;
        if(QFORMAT_0 == 2U)
        {
            _S239 = true;
        }
        else
        {
            _S239 = QFORMAT_0 == 3U;
        }
        bool _S240;
        if(_S239)
        {
            _S240 = true;
        }
        else
        {
            _S240 = QFORMAT_0 == 4U;
        }
        if(_S240)
        {
            sumA_2[2U] = sumA_2[2U] + av_21[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_21[2U], bv_21[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_21[2U], bv_21[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_21[2U], bv_21[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_21[2U], bv_21[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_21[2U], bv_21[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_21[2U], bv_21[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_21[2U], bv_21[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_21[2U], bv_21[3U], s_2[2U][3U]);
        }
        bool _S241;
        if(QFORMAT_0 == 2U)
        {
            _S241 = true;
        }
        else
        {
            _S241 = QFORMAT_0 == 3U;
        }
        bool _S242;
        if(_S241)
        {
            _S242 = true;
        }
        else
        {
            _S242 = QFORMAT_0 == 4U;
        }
        if(_S242)
        {
            sumA_2[3U] = sumA_2[3U] + av_21[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_21[3U], bv_21[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_21[3U], bv_21[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_21[3U], bv_21[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_21[3U], bv_21[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_21[3U], bv_21[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_21[3U], bv_21[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_21[3U], bv_21[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_21[3U], bv_21[3U], s_2[3U][3U]);
        }
        uint w0_22 = (*(&kernelContext_13)->qTileA_0)[22U][ty_0];
        uint w1_22 = (*(&kernelContext_13)->qTileA_0)[22U][_S63];
        thread array<float, int(4)> av_22;
        av_22[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[22U][ty_0]);
        av_22[int(1)] = aLo_0(w1_22);
        av_22[int(2)] = aHi_0(w0_22);
        av_22[int(3)] = aHi_0(w1_22);
        thread array<float, int(4)> bv_22;
        bv_22[0U] = (*(&kernelContext_13)->qTileB_0)[22U][tx_0];
        bv_22[1U] = (*(&kernelContext_13)->qTileB_0)[22U][_S64];
        bv_22[2U] = (*(&kernelContext_13)->qTileB_0)[22U][_S65];
        bv_22[3U] = (*(&kernelContext_13)->qTileB_0)[22U][_S66];
        bool _S243;
        if(QFORMAT_0 == 2U)
        {
            _S243 = true;
        }
        else
        {
            _S243 = QFORMAT_0 == 3U;
        }
        bool _S244;
        if(_S243)
        {
            _S244 = true;
        }
        else
        {
            _S244 = QFORMAT_0 == 4U;
        }
        if(_S244)
        {
            sumA_2[0U] = sumA_2[0U] + av_22[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_22[0U], bv_22[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_22[0U], bv_22[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_22[0U], bv_22[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_22[0U], bv_22[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_22[0U], bv_22[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_22[0U], bv_22[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_22[0U], bv_22[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_22[0U], bv_22[3U], s_2[0U][3U]);
        }
        bool _S245;
        if(QFORMAT_0 == 2U)
        {
            _S245 = true;
        }
        else
        {
            _S245 = QFORMAT_0 == 3U;
        }
        bool _S246;
        if(_S245)
        {
            _S246 = true;
        }
        else
        {
            _S246 = QFORMAT_0 == 4U;
        }
        if(_S246)
        {
            sumA_2[1U] = sumA_2[1U] + av_22[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_22[1U], bv_22[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_22[1U], bv_22[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_22[1U], bv_22[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_22[1U], bv_22[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_22[1U], bv_22[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_22[1U], bv_22[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_22[1U], bv_22[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_22[1U], bv_22[3U], s_2[1U][3U]);
        }
        bool _S247;
        if(QFORMAT_0 == 2U)
        {
            _S247 = true;
        }
        else
        {
            _S247 = QFORMAT_0 == 3U;
        }
        bool _S248;
        if(_S247)
        {
            _S248 = true;
        }
        else
        {
            _S248 = QFORMAT_0 == 4U;
        }
        if(_S248)
        {
            sumA_2[2U] = sumA_2[2U] + av_22[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_22[2U], bv_22[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_22[2U], bv_22[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_22[2U], bv_22[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_22[2U], bv_22[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_22[2U], bv_22[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_22[2U], bv_22[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_22[2U], bv_22[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_22[2U], bv_22[3U], s_2[2U][3U]);
        }
        bool _S249;
        if(QFORMAT_0 == 2U)
        {
            _S249 = true;
        }
        else
        {
            _S249 = QFORMAT_0 == 3U;
        }
        bool _S250;
        if(_S249)
        {
            _S250 = true;
        }
        else
        {
            _S250 = QFORMAT_0 == 4U;
        }
        if(_S250)
        {
            sumA_2[3U] = sumA_2[3U] + av_22[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_22[3U], bv_22[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_22[3U], bv_22[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_22[3U], bv_22[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_22[3U], bv_22[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_22[3U], bv_22[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_22[3U], bv_22[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_22[3U], bv_22[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_22[3U], bv_22[3U], s_2[3U][3U]);
        }
        uint w0_23 = (*(&kernelContext_13)->qTileA_0)[23U][ty_0];
        uint w1_23 = (*(&kernelContext_13)->qTileA_0)[23U][_S63];
        thread array<float, int(4)> av_23;
        av_23[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[23U][ty_0]);
        av_23[int(1)] = aLo_0(w1_23);
        av_23[int(2)] = aHi_0(w0_23);
        av_23[int(3)] = aHi_0(w1_23);
        thread array<float, int(4)> bv_23;
        bv_23[0U] = (*(&kernelContext_13)->qTileB_0)[23U][tx_0];
        bv_23[1U] = (*(&kernelContext_13)->qTileB_0)[23U][_S64];
        bv_23[2U] = (*(&kernelContext_13)->qTileB_0)[23U][_S65];
        bv_23[3U] = (*(&kernelContext_13)->qTileB_0)[23U][_S66];
        bool _S251;
        if(QFORMAT_0 == 2U)
        {
            _S251 = true;
        }
        else
        {
            _S251 = QFORMAT_0 == 3U;
        }
        bool _S252;
        if(_S251)
        {
            _S252 = true;
        }
        else
        {
            _S252 = QFORMAT_0 == 4U;
        }
        if(_S252)
        {
            sumA_2[0U] = sumA_2[0U] + av_23[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_23[0U], bv_23[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_23[0U], bv_23[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_23[0U], bv_23[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_23[0U], bv_23[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_23[0U], bv_23[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_23[0U], bv_23[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_23[0U], bv_23[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_23[0U], bv_23[3U], s_2[0U][3U]);
        }
        bool _S253;
        if(QFORMAT_0 == 2U)
        {
            _S253 = true;
        }
        else
        {
            _S253 = QFORMAT_0 == 3U;
        }
        bool _S254;
        if(_S253)
        {
            _S254 = true;
        }
        else
        {
            _S254 = QFORMAT_0 == 4U;
        }
        if(_S254)
        {
            sumA_2[1U] = sumA_2[1U] + av_23[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_23[1U], bv_23[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_23[1U], bv_23[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_23[1U], bv_23[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_23[1U], bv_23[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_23[1U], bv_23[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_23[1U], bv_23[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_23[1U], bv_23[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_23[1U], bv_23[3U], s_2[1U][3U]);
        }
        bool _S255;
        if(QFORMAT_0 == 2U)
        {
            _S255 = true;
        }
        else
        {
            _S255 = QFORMAT_0 == 3U;
        }
        bool _S256;
        if(_S255)
        {
            _S256 = true;
        }
        else
        {
            _S256 = QFORMAT_0 == 4U;
        }
        if(_S256)
        {
            sumA_2[2U] = sumA_2[2U] + av_23[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_23[2U], bv_23[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_23[2U], bv_23[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_23[2U], bv_23[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_23[2U], bv_23[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_23[2U], bv_23[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_23[2U], bv_23[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_23[2U], bv_23[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_23[2U], bv_23[3U], s_2[2U][3U]);
        }
        bool _S257;
        if(QFORMAT_0 == 2U)
        {
            _S257 = true;
        }
        else
        {
            _S257 = QFORMAT_0 == 3U;
        }
        bool _S258;
        if(_S257)
        {
            _S258 = true;
        }
        else
        {
            _S258 = QFORMAT_0 == 4U;
        }
        if(_S258)
        {
            sumA_2[3U] = sumA_2[3U] + av_23[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_23[3U], bv_23[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_23[3U], bv_23[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_23[3U], bv_23[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_23[3U], bv_23[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_23[3U], bv_23[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_23[3U], bv_23[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_23[3U], bv_23[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_23[3U], bv_23[3U], s_2[3U][3U]);
        }
        uint w0_24 = (*(&kernelContext_13)->qTileA_0)[24U][ty_0];
        uint w1_24 = (*(&kernelContext_13)->qTileA_0)[24U][_S63];
        thread array<float, int(4)> av_24;
        av_24[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[24U][ty_0]);
        av_24[int(1)] = aLo_0(w1_24);
        av_24[int(2)] = aHi_0(w0_24);
        av_24[int(3)] = aHi_0(w1_24);
        thread array<float, int(4)> bv_24;
        bv_24[0U] = (*(&kernelContext_13)->qTileB_0)[24U][tx_0];
        bv_24[1U] = (*(&kernelContext_13)->qTileB_0)[24U][_S64];
        bv_24[2U] = (*(&kernelContext_13)->qTileB_0)[24U][_S65];
        bv_24[3U] = (*(&kernelContext_13)->qTileB_0)[24U][_S66];
        bool _S259;
        if(QFORMAT_0 == 2U)
        {
            _S259 = true;
        }
        else
        {
            _S259 = QFORMAT_0 == 3U;
        }
        bool _S260;
        if(_S259)
        {
            _S260 = true;
        }
        else
        {
            _S260 = QFORMAT_0 == 4U;
        }
        if(_S260)
        {
            sumA_2[0U] = sumA_2[0U] + av_24[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_24[0U], bv_24[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_24[0U], bv_24[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_24[0U], bv_24[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_24[0U], bv_24[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_24[0U], bv_24[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_24[0U], bv_24[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_24[0U], bv_24[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_24[0U], bv_24[3U], s_2[0U][3U]);
        }
        bool _S261;
        if(QFORMAT_0 == 2U)
        {
            _S261 = true;
        }
        else
        {
            _S261 = QFORMAT_0 == 3U;
        }
        bool _S262;
        if(_S261)
        {
            _S262 = true;
        }
        else
        {
            _S262 = QFORMAT_0 == 4U;
        }
        if(_S262)
        {
            sumA_2[1U] = sumA_2[1U] + av_24[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_24[1U], bv_24[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_24[1U], bv_24[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_24[1U], bv_24[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_24[1U], bv_24[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_24[1U], bv_24[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_24[1U], bv_24[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_24[1U], bv_24[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_24[1U], bv_24[3U], s_2[1U][3U]);
        }
        bool _S263;
        if(QFORMAT_0 == 2U)
        {
            _S263 = true;
        }
        else
        {
            _S263 = QFORMAT_0 == 3U;
        }
        bool _S264;
        if(_S263)
        {
            _S264 = true;
        }
        else
        {
            _S264 = QFORMAT_0 == 4U;
        }
        if(_S264)
        {
            sumA_2[2U] = sumA_2[2U] + av_24[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_24[2U], bv_24[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_24[2U], bv_24[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_24[2U], bv_24[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_24[2U], bv_24[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_24[2U], bv_24[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_24[2U], bv_24[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_24[2U], bv_24[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_24[2U], bv_24[3U], s_2[2U][3U]);
        }
        bool _S265;
        if(QFORMAT_0 == 2U)
        {
            _S265 = true;
        }
        else
        {
            _S265 = QFORMAT_0 == 3U;
        }
        bool _S266;
        if(_S265)
        {
            _S266 = true;
        }
        else
        {
            _S266 = QFORMAT_0 == 4U;
        }
        if(_S266)
        {
            sumA_2[3U] = sumA_2[3U] + av_24[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_24[3U], bv_24[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_24[3U], bv_24[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_24[3U], bv_24[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_24[3U], bv_24[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_24[3U], bv_24[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_24[3U], bv_24[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_24[3U], bv_24[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_24[3U], bv_24[3U], s_2[3U][3U]);
        }
        uint w0_25 = (*(&kernelContext_13)->qTileA_0)[25U][ty_0];
        uint w1_25 = (*(&kernelContext_13)->qTileA_0)[25U][_S63];
        thread array<float, int(4)> av_25;
        av_25[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[25U][ty_0]);
        av_25[int(1)] = aLo_0(w1_25);
        av_25[int(2)] = aHi_0(w0_25);
        av_25[int(3)] = aHi_0(w1_25);
        thread array<float, int(4)> bv_25;
        bv_25[0U] = (*(&kernelContext_13)->qTileB_0)[25U][tx_0];
        bv_25[1U] = (*(&kernelContext_13)->qTileB_0)[25U][_S64];
        bv_25[2U] = (*(&kernelContext_13)->qTileB_0)[25U][_S65];
        bv_25[3U] = (*(&kernelContext_13)->qTileB_0)[25U][_S66];
        bool _S267;
        if(QFORMAT_0 == 2U)
        {
            _S267 = true;
        }
        else
        {
            _S267 = QFORMAT_0 == 3U;
        }
        bool _S268;
        if(_S267)
        {
            _S268 = true;
        }
        else
        {
            _S268 = QFORMAT_0 == 4U;
        }
        if(_S268)
        {
            sumA_2[0U] = sumA_2[0U] + av_25[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_25[0U], bv_25[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_25[0U], bv_25[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_25[0U], bv_25[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_25[0U], bv_25[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_25[0U], bv_25[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_25[0U], bv_25[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_25[0U], bv_25[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_25[0U], bv_25[3U], s_2[0U][3U]);
        }
        bool _S269;
        if(QFORMAT_0 == 2U)
        {
            _S269 = true;
        }
        else
        {
            _S269 = QFORMAT_0 == 3U;
        }
        bool _S270;
        if(_S269)
        {
            _S270 = true;
        }
        else
        {
            _S270 = QFORMAT_0 == 4U;
        }
        if(_S270)
        {
            sumA_2[1U] = sumA_2[1U] + av_25[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_25[1U], bv_25[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_25[1U], bv_25[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_25[1U], bv_25[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_25[1U], bv_25[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_25[1U], bv_25[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_25[1U], bv_25[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_25[1U], bv_25[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_25[1U], bv_25[3U], s_2[1U][3U]);
        }
        bool _S271;
        if(QFORMAT_0 == 2U)
        {
            _S271 = true;
        }
        else
        {
            _S271 = QFORMAT_0 == 3U;
        }
        bool _S272;
        if(_S271)
        {
            _S272 = true;
        }
        else
        {
            _S272 = QFORMAT_0 == 4U;
        }
        if(_S272)
        {
            sumA_2[2U] = sumA_2[2U] + av_25[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_25[2U], bv_25[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_25[2U], bv_25[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_25[2U], bv_25[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_25[2U], bv_25[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_25[2U], bv_25[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_25[2U], bv_25[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_25[2U], bv_25[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_25[2U], bv_25[3U], s_2[2U][3U]);
        }
        bool _S273;
        if(QFORMAT_0 == 2U)
        {
            _S273 = true;
        }
        else
        {
            _S273 = QFORMAT_0 == 3U;
        }
        bool _S274;
        if(_S273)
        {
            _S274 = true;
        }
        else
        {
            _S274 = QFORMAT_0 == 4U;
        }
        if(_S274)
        {
            sumA_2[3U] = sumA_2[3U] + av_25[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_25[3U], bv_25[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_25[3U], bv_25[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_25[3U], bv_25[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_25[3U], bv_25[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_25[3U], bv_25[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_25[3U], bv_25[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_25[3U], bv_25[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_25[3U], bv_25[3U], s_2[3U][3U]);
        }
        uint w0_26 = (*(&kernelContext_13)->qTileA_0)[26U][ty_0];
        uint w1_26 = (*(&kernelContext_13)->qTileA_0)[26U][_S63];
        thread array<float, int(4)> av_26;
        av_26[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[26U][ty_0]);
        av_26[int(1)] = aLo_0(w1_26);
        av_26[int(2)] = aHi_0(w0_26);
        av_26[int(3)] = aHi_0(w1_26);
        thread array<float, int(4)> bv_26;
        bv_26[0U] = (*(&kernelContext_13)->qTileB_0)[26U][tx_0];
        bv_26[1U] = (*(&kernelContext_13)->qTileB_0)[26U][_S64];
        bv_26[2U] = (*(&kernelContext_13)->qTileB_0)[26U][_S65];
        bv_26[3U] = (*(&kernelContext_13)->qTileB_0)[26U][_S66];
        bool _S275;
        if(QFORMAT_0 == 2U)
        {
            _S275 = true;
        }
        else
        {
            _S275 = QFORMAT_0 == 3U;
        }
        bool _S276;
        if(_S275)
        {
            _S276 = true;
        }
        else
        {
            _S276 = QFORMAT_0 == 4U;
        }
        if(_S276)
        {
            sumA_2[0U] = sumA_2[0U] + av_26[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_26[0U], bv_26[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_26[0U], bv_26[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_26[0U], bv_26[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_26[0U], bv_26[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_26[0U], bv_26[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_26[0U], bv_26[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_26[0U], bv_26[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_26[0U], bv_26[3U], s_2[0U][3U]);
        }
        bool _S277;
        if(QFORMAT_0 == 2U)
        {
            _S277 = true;
        }
        else
        {
            _S277 = QFORMAT_0 == 3U;
        }
        bool _S278;
        if(_S277)
        {
            _S278 = true;
        }
        else
        {
            _S278 = QFORMAT_0 == 4U;
        }
        if(_S278)
        {
            sumA_2[1U] = sumA_2[1U] + av_26[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_26[1U], bv_26[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_26[1U], bv_26[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_26[1U], bv_26[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_26[1U], bv_26[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_26[1U], bv_26[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_26[1U], bv_26[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_26[1U], bv_26[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_26[1U], bv_26[3U], s_2[1U][3U]);
        }
        bool _S279;
        if(QFORMAT_0 == 2U)
        {
            _S279 = true;
        }
        else
        {
            _S279 = QFORMAT_0 == 3U;
        }
        bool _S280;
        if(_S279)
        {
            _S280 = true;
        }
        else
        {
            _S280 = QFORMAT_0 == 4U;
        }
        if(_S280)
        {
            sumA_2[2U] = sumA_2[2U] + av_26[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_26[2U], bv_26[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_26[2U], bv_26[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_26[2U], bv_26[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_26[2U], bv_26[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_26[2U], bv_26[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_26[2U], bv_26[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_26[2U], bv_26[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_26[2U], bv_26[3U], s_2[2U][3U]);
        }
        bool _S281;
        if(QFORMAT_0 == 2U)
        {
            _S281 = true;
        }
        else
        {
            _S281 = QFORMAT_0 == 3U;
        }
        bool _S282;
        if(_S281)
        {
            _S282 = true;
        }
        else
        {
            _S282 = QFORMAT_0 == 4U;
        }
        if(_S282)
        {
            sumA_2[3U] = sumA_2[3U] + av_26[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_26[3U], bv_26[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_26[3U], bv_26[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_26[3U], bv_26[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_26[3U], bv_26[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_26[3U], bv_26[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_26[3U], bv_26[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_26[3U], bv_26[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_26[3U], bv_26[3U], s_2[3U][3U]);
        }
        uint w0_27 = (*(&kernelContext_13)->qTileA_0)[27U][ty_0];
        uint w1_27 = (*(&kernelContext_13)->qTileA_0)[27U][_S63];
        thread array<float, int(4)> av_27;
        av_27[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[27U][ty_0]);
        av_27[int(1)] = aLo_0(w1_27);
        av_27[int(2)] = aHi_0(w0_27);
        av_27[int(3)] = aHi_0(w1_27);
        thread array<float, int(4)> bv_27;
        bv_27[0U] = (*(&kernelContext_13)->qTileB_0)[27U][tx_0];
        bv_27[1U] = (*(&kernelContext_13)->qTileB_0)[27U][_S64];
        bv_27[2U] = (*(&kernelContext_13)->qTileB_0)[27U][_S65];
        bv_27[3U] = (*(&kernelContext_13)->qTileB_0)[27U][_S66];
        bool _S283;
        if(QFORMAT_0 == 2U)
        {
            _S283 = true;
        }
        else
        {
            _S283 = QFORMAT_0 == 3U;
        }
        bool _S284;
        if(_S283)
        {
            _S284 = true;
        }
        else
        {
            _S284 = QFORMAT_0 == 4U;
        }
        if(_S284)
        {
            sumA_2[0U] = sumA_2[0U] + av_27[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_27[0U], bv_27[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_27[0U], bv_27[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_27[0U], bv_27[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_27[0U], bv_27[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_27[0U], bv_27[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_27[0U], bv_27[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_27[0U], bv_27[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_27[0U], bv_27[3U], s_2[0U][3U]);
        }
        bool _S285;
        if(QFORMAT_0 == 2U)
        {
            _S285 = true;
        }
        else
        {
            _S285 = QFORMAT_0 == 3U;
        }
        bool _S286;
        if(_S285)
        {
            _S286 = true;
        }
        else
        {
            _S286 = QFORMAT_0 == 4U;
        }
        if(_S286)
        {
            sumA_2[1U] = sumA_2[1U] + av_27[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_27[1U], bv_27[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_27[1U], bv_27[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_27[1U], bv_27[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_27[1U], bv_27[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_27[1U], bv_27[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_27[1U], bv_27[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_27[1U], bv_27[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_27[1U], bv_27[3U], s_2[1U][3U]);
        }
        bool _S287;
        if(QFORMAT_0 == 2U)
        {
            _S287 = true;
        }
        else
        {
            _S287 = QFORMAT_0 == 3U;
        }
        bool _S288;
        if(_S287)
        {
            _S288 = true;
        }
        else
        {
            _S288 = QFORMAT_0 == 4U;
        }
        if(_S288)
        {
            sumA_2[2U] = sumA_2[2U] + av_27[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_27[2U], bv_27[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_27[2U], bv_27[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_27[2U], bv_27[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_27[2U], bv_27[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_27[2U], bv_27[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_27[2U], bv_27[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_27[2U], bv_27[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_27[2U], bv_27[3U], s_2[2U][3U]);
        }
        bool _S289;
        if(QFORMAT_0 == 2U)
        {
            _S289 = true;
        }
        else
        {
            _S289 = QFORMAT_0 == 3U;
        }
        bool _S290;
        if(_S289)
        {
            _S290 = true;
        }
        else
        {
            _S290 = QFORMAT_0 == 4U;
        }
        if(_S290)
        {
            sumA_2[3U] = sumA_2[3U] + av_27[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_27[3U], bv_27[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_27[3U], bv_27[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_27[3U], bv_27[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_27[3U], bv_27[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_27[3U], bv_27[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_27[3U], bv_27[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_27[3U], bv_27[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_27[3U], bv_27[3U], s_2[3U][3U]);
        }
        uint w0_28 = (*(&kernelContext_13)->qTileA_0)[28U][ty_0];
        uint w1_28 = (*(&kernelContext_13)->qTileA_0)[28U][_S63];
        thread array<float, int(4)> av_28;
        av_28[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[28U][ty_0]);
        av_28[int(1)] = aLo_0(w1_28);
        av_28[int(2)] = aHi_0(w0_28);
        av_28[int(3)] = aHi_0(w1_28);
        thread array<float, int(4)> bv_28;
        bv_28[0U] = (*(&kernelContext_13)->qTileB_0)[28U][tx_0];
        bv_28[1U] = (*(&kernelContext_13)->qTileB_0)[28U][_S64];
        bv_28[2U] = (*(&kernelContext_13)->qTileB_0)[28U][_S65];
        bv_28[3U] = (*(&kernelContext_13)->qTileB_0)[28U][_S66];
        bool _S291;
        if(QFORMAT_0 == 2U)
        {
            _S291 = true;
        }
        else
        {
            _S291 = QFORMAT_0 == 3U;
        }
        bool _S292;
        if(_S291)
        {
            _S292 = true;
        }
        else
        {
            _S292 = QFORMAT_0 == 4U;
        }
        if(_S292)
        {
            sumA_2[0U] = sumA_2[0U] + av_28[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_28[0U], bv_28[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_28[0U], bv_28[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_28[0U], bv_28[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_28[0U], bv_28[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_28[0U], bv_28[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_28[0U], bv_28[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_28[0U], bv_28[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_28[0U], bv_28[3U], s_2[0U][3U]);
        }
        bool _S293;
        if(QFORMAT_0 == 2U)
        {
            _S293 = true;
        }
        else
        {
            _S293 = QFORMAT_0 == 3U;
        }
        bool _S294;
        if(_S293)
        {
            _S294 = true;
        }
        else
        {
            _S294 = QFORMAT_0 == 4U;
        }
        if(_S294)
        {
            sumA_2[1U] = sumA_2[1U] + av_28[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_28[1U], bv_28[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_28[1U], bv_28[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_28[1U], bv_28[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_28[1U], bv_28[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_28[1U], bv_28[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_28[1U], bv_28[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_28[1U], bv_28[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_28[1U], bv_28[3U], s_2[1U][3U]);
        }
        bool _S295;
        if(QFORMAT_0 == 2U)
        {
            _S295 = true;
        }
        else
        {
            _S295 = QFORMAT_0 == 3U;
        }
        bool _S296;
        if(_S295)
        {
            _S296 = true;
        }
        else
        {
            _S296 = QFORMAT_0 == 4U;
        }
        if(_S296)
        {
            sumA_2[2U] = sumA_2[2U] + av_28[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_28[2U], bv_28[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_28[2U], bv_28[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_28[2U], bv_28[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_28[2U], bv_28[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_28[2U], bv_28[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_28[2U], bv_28[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_28[2U], bv_28[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_28[2U], bv_28[3U], s_2[2U][3U]);
        }
        bool _S297;
        if(QFORMAT_0 == 2U)
        {
            _S297 = true;
        }
        else
        {
            _S297 = QFORMAT_0 == 3U;
        }
        bool _S298;
        if(_S297)
        {
            _S298 = true;
        }
        else
        {
            _S298 = QFORMAT_0 == 4U;
        }
        if(_S298)
        {
            sumA_2[3U] = sumA_2[3U] + av_28[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_28[3U], bv_28[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_28[3U], bv_28[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_28[3U], bv_28[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_28[3U], bv_28[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_28[3U], bv_28[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_28[3U], bv_28[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_28[3U], bv_28[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_28[3U], bv_28[3U], s_2[3U][3U]);
        }
        uint w0_29 = (*(&kernelContext_13)->qTileA_0)[29U][ty_0];
        uint w1_29 = (*(&kernelContext_13)->qTileA_0)[29U][_S63];
        thread array<float, int(4)> av_29;
        av_29[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[29U][ty_0]);
        av_29[int(1)] = aLo_0(w1_29);
        av_29[int(2)] = aHi_0(w0_29);
        av_29[int(3)] = aHi_0(w1_29);
        thread array<float, int(4)> bv_29;
        bv_29[0U] = (*(&kernelContext_13)->qTileB_0)[29U][tx_0];
        bv_29[1U] = (*(&kernelContext_13)->qTileB_0)[29U][_S64];
        bv_29[2U] = (*(&kernelContext_13)->qTileB_0)[29U][_S65];
        bv_29[3U] = (*(&kernelContext_13)->qTileB_0)[29U][_S66];
        bool _S299;
        if(QFORMAT_0 == 2U)
        {
            _S299 = true;
        }
        else
        {
            _S299 = QFORMAT_0 == 3U;
        }
        bool _S300;
        if(_S299)
        {
            _S300 = true;
        }
        else
        {
            _S300 = QFORMAT_0 == 4U;
        }
        if(_S300)
        {
            sumA_2[0U] = sumA_2[0U] + av_29[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_29[0U], bv_29[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_29[0U], bv_29[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_29[0U], bv_29[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_29[0U], bv_29[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_29[0U], bv_29[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_29[0U], bv_29[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_29[0U], bv_29[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_29[0U], bv_29[3U], s_2[0U][3U]);
        }
        bool _S301;
        if(QFORMAT_0 == 2U)
        {
            _S301 = true;
        }
        else
        {
            _S301 = QFORMAT_0 == 3U;
        }
        bool _S302;
        if(_S301)
        {
            _S302 = true;
        }
        else
        {
            _S302 = QFORMAT_0 == 4U;
        }
        if(_S302)
        {
            sumA_2[1U] = sumA_2[1U] + av_29[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_29[1U], bv_29[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_29[1U], bv_29[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_29[1U], bv_29[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_29[1U], bv_29[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_29[1U], bv_29[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_29[1U], bv_29[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_29[1U], bv_29[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_29[1U], bv_29[3U], s_2[1U][3U]);
        }
        bool _S303;
        if(QFORMAT_0 == 2U)
        {
            _S303 = true;
        }
        else
        {
            _S303 = QFORMAT_0 == 3U;
        }
        bool _S304;
        if(_S303)
        {
            _S304 = true;
        }
        else
        {
            _S304 = QFORMAT_0 == 4U;
        }
        if(_S304)
        {
            sumA_2[2U] = sumA_2[2U] + av_29[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_29[2U], bv_29[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_29[2U], bv_29[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_29[2U], bv_29[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_29[2U], bv_29[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_29[2U], bv_29[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_29[2U], bv_29[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_29[2U], bv_29[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_29[2U], bv_29[3U], s_2[2U][3U]);
        }
        bool _S305;
        if(QFORMAT_0 == 2U)
        {
            _S305 = true;
        }
        else
        {
            _S305 = QFORMAT_0 == 3U;
        }
        bool _S306;
        if(_S305)
        {
            _S306 = true;
        }
        else
        {
            _S306 = QFORMAT_0 == 4U;
        }
        if(_S306)
        {
            sumA_2[3U] = sumA_2[3U] + av_29[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_29[3U], bv_29[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_29[3U], bv_29[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_29[3U], bv_29[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_29[3U], bv_29[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_29[3U], bv_29[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_29[3U], bv_29[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_29[3U], bv_29[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_29[3U], bv_29[3U], s_2[3U][3U]);
        }
        uint w0_30 = (*(&kernelContext_13)->qTileA_0)[30U][ty_0];
        uint w1_30 = (*(&kernelContext_13)->qTileA_0)[30U][_S63];
        thread array<float, int(4)> av_30;
        av_30[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[30U][ty_0]);
        av_30[int(1)] = aLo_0(w1_30);
        av_30[int(2)] = aHi_0(w0_30);
        av_30[int(3)] = aHi_0(w1_30);
        thread array<float, int(4)> bv_30;
        bv_30[0U] = (*(&kernelContext_13)->qTileB_0)[30U][tx_0];
        bv_30[1U] = (*(&kernelContext_13)->qTileB_0)[30U][_S64];
        bv_30[2U] = (*(&kernelContext_13)->qTileB_0)[30U][_S65];
        bv_30[3U] = (*(&kernelContext_13)->qTileB_0)[30U][_S66];
        bool _S307;
        if(QFORMAT_0 == 2U)
        {
            _S307 = true;
        }
        else
        {
            _S307 = QFORMAT_0 == 3U;
        }
        bool _S308;
        if(_S307)
        {
            _S308 = true;
        }
        else
        {
            _S308 = QFORMAT_0 == 4U;
        }
        if(_S308)
        {
            sumA_2[0U] = sumA_2[0U] + av_30[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_30[0U], bv_30[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_30[0U], bv_30[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_30[0U], bv_30[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_30[0U], bv_30[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_30[0U], bv_30[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_30[0U], bv_30[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_30[0U], bv_30[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_30[0U], bv_30[3U], s_2[0U][3U]);
        }
        bool _S309;
        if(QFORMAT_0 == 2U)
        {
            _S309 = true;
        }
        else
        {
            _S309 = QFORMAT_0 == 3U;
        }
        bool _S310;
        if(_S309)
        {
            _S310 = true;
        }
        else
        {
            _S310 = QFORMAT_0 == 4U;
        }
        if(_S310)
        {
            sumA_2[1U] = sumA_2[1U] + av_30[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_30[1U], bv_30[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_30[1U], bv_30[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_30[1U], bv_30[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_30[1U], bv_30[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_30[1U], bv_30[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_30[1U], bv_30[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_30[1U], bv_30[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_30[1U], bv_30[3U], s_2[1U][3U]);
        }
        bool _S311;
        if(QFORMAT_0 == 2U)
        {
            _S311 = true;
        }
        else
        {
            _S311 = QFORMAT_0 == 3U;
        }
        bool _S312;
        if(_S311)
        {
            _S312 = true;
        }
        else
        {
            _S312 = QFORMAT_0 == 4U;
        }
        if(_S312)
        {
            sumA_2[2U] = sumA_2[2U] + av_30[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_30[2U], bv_30[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_30[2U], bv_30[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_30[2U], bv_30[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_30[2U], bv_30[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_30[2U], bv_30[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_30[2U], bv_30[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_30[2U], bv_30[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_30[2U], bv_30[3U], s_2[2U][3U]);
        }
        bool _S313;
        if(QFORMAT_0 == 2U)
        {
            _S313 = true;
        }
        else
        {
            _S313 = QFORMAT_0 == 3U;
        }
        bool _S314;
        if(_S313)
        {
            _S314 = true;
        }
        else
        {
            _S314 = QFORMAT_0 == 4U;
        }
        if(_S314)
        {
            sumA_2[3U] = sumA_2[3U] + av_30[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_30[3U], bv_30[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_30[3U], bv_30[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_30[3U], bv_30[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_30[3U], bv_30[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_30[3U], bv_30[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_30[3U], bv_30[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_30[3U], bv_30[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_30[3U], bv_30[3U], s_2[3U][3U]);
        }
        uint w0_31 = (*(&kernelContext_13)->qTileA_0)[31U][ty_0];
        uint w1_31 = (*(&kernelContext_13)->qTileA_0)[31U][_S63];
        thread array<float, int(4)> av_31;
        av_31[int(0)] = aLo_0((*(&kernelContext_13)->qTileA_0)[31U][ty_0]);
        av_31[int(1)] = aLo_0(w1_31);
        av_31[int(2)] = aHi_0(w0_31);
        av_31[int(3)] = aHi_0(w1_31);
        thread array<float, int(4)> bv_31;
        bv_31[0U] = (*(&kernelContext_13)->qTileB_0)[31U][tx_0];
        bv_31[1U] = (*(&kernelContext_13)->qTileB_0)[31U][_S64];
        bv_31[2U] = (*(&kernelContext_13)->qTileB_0)[31U][_S65];
        bv_31[3U] = (*(&kernelContext_13)->qTileB_0)[31U][_S66];
        bool _S315;
        if(QFORMAT_0 == 2U)
        {
            _S315 = true;
        }
        else
        {
            _S315 = QFORMAT_0 == 3U;
        }
        bool _S316;
        if(_S315)
        {
            _S316 = true;
        }
        else
        {
            _S316 = QFORMAT_0 == 4U;
        }
        if(_S316)
        {
            sumA_2[0U] = sumA_2[0U] + av_31[0U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][0U] = fma(av_31[0U], bv_31[0U], s2_1[0U][0U]);
        }
        else
        {
            s_2[0U][0U] = fma(av_31[0U], bv_31[0U], s_2[0U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][1U] = fma(av_31[0U], bv_31[1U], s2_1[0U][1U]);
        }
        else
        {
            s_2[0U][1U] = fma(av_31[0U], bv_31[1U], s_2[0U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][2U] = fma(av_31[0U], bv_31[2U], s2_1[0U][2U]);
        }
        else
        {
            s_2[0U][2U] = fma(av_31[0U], bv_31[2U], s_2[0U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[0U][3U] = fma(av_31[0U], bv_31[3U], s2_1[0U][3U]);
        }
        else
        {
            s_2[0U][3U] = fma(av_31[0U], bv_31[3U], s_2[0U][3U]);
        }
        bool _S317;
        if(QFORMAT_0 == 2U)
        {
            _S317 = true;
        }
        else
        {
            _S317 = QFORMAT_0 == 3U;
        }
        bool _S318;
        if(_S317)
        {
            _S318 = true;
        }
        else
        {
            _S318 = QFORMAT_0 == 4U;
        }
        if(_S318)
        {
            sumA_2[1U] = sumA_2[1U] + av_31[1U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][0U] = fma(av_31[1U], bv_31[0U], s2_1[1U][0U]);
        }
        else
        {
            s_2[1U][0U] = fma(av_31[1U], bv_31[0U], s_2[1U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][1U] = fma(av_31[1U], bv_31[1U], s2_1[1U][1U]);
        }
        else
        {
            s_2[1U][1U] = fma(av_31[1U], bv_31[1U], s_2[1U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][2U] = fma(av_31[1U], bv_31[2U], s2_1[1U][2U]);
        }
        else
        {
            s_2[1U][2U] = fma(av_31[1U], bv_31[2U], s_2[1U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[1U][3U] = fma(av_31[1U], bv_31[3U], s2_1[1U][3U]);
        }
        else
        {
            s_2[1U][3U] = fma(av_31[1U], bv_31[3U], s_2[1U][3U]);
        }
        bool _S319;
        if(QFORMAT_0 == 2U)
        {
            _S319 = true;
        }
        else
        {
            _S319 = QFORMAT_0 == 3U;
        }
        bool _S320;
        if(_S319)
        {
            _S320 = true;
        }
        else
        {
            _S320 = QFORMAT_0 == 4U;
        }
        if(_S320)
        {
            sumA_2[2U] = sumA_2[2U] + av_31[2U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][0U] = fma(av_31[2U], bv_31[0U], s2_1[2U][0U]);
        }
        else
        {
            s_2[2U][0U] = fma(av_31[2U], bv_31[0U], s_2[2U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][1U] = fma(av_31[2U], bv_31[1U], s2_1[2U][1U]);
        }
        else
        {
            s_2[2U][1U] = fma(av_31[2U], bv_31[1U], s_2[2U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][2U] = fma(av_31[2U], bv_31[2U], s2_1[2U][2U]);
        }
        else
        {
            s_2[2U][2U] = fma(av_31[2U], bv_31[2U], s_2[2U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[2U][3U] = fma(av_31[2U], bv_31[3U], s2_1[2U][3U]);
        }
        else
        {
            s_2[2U][3U] = fma(av_31[2U], bv_31[3U], s_2[2U][3U]);
        }
        bool _S321;
        if(QFORMAT_0 == 2U)
        {
            _S321 = true;
        }
        else
        {
            _S321 = QFORMAT_0 == 3U;
        }
        bool _S322;
        if(_S321)
        {
            _S322 = true;
        }
        else
        {
            _S322 = QFORMAT_0 == 4U;
        }
        if(_S322)
        {
            sumA_2[3U] = sumA_2[3U] + av_31[3U];
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][0U] = fma(av_31[3U], bv_31[0U], s2_1[3U][0U]);
        }
        else
        {
            s_2[3U][0U] = fma(av_31[3U], bv_31[0U], s_2[3U][0U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][1U] = fma(av_31[3U], bv_31[1U], s2_1[3U][1U]);
        }
        else
        {
            s_2[3U][1U] = fma(av_31[3U], bv_31[1U], s_2[3U][1U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][2U] = fma(av_31[3U], bv_31[2U], s2_1[3U][2U]);
        }
        else
        {
            s_2[3U][2U] = fma(av_31[3U], bv_31[2U], s_2[3U][2U]);
        }
        if(QFORMAT_0 == 5U)
        {
            s2_1[3U][3U] = fma(av_31[3U], bv_31[3U], s2_1[3U][3U]);
        }
        else
        {
            s_2[3U][3U] = fma(av_31[3U], bv_31[3U], s_2[3U][3U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S56)
        {
            _S60 = 32U;
        }
        else
        {
            _S60 = QBLOCK_0;
        }
        uint _S323 = _S55 % _S60;
        if(_S323 == 0U)
        {
            uint _S324;
            if(QFORMAT_0 >= 3U)
            {
                _S324 = 32U;
            }
            else
            {
                _S324 = QBLOCK_0;
            }
            uint _S325 = k0_0 / _S324;
            float2 _S326 = (*(&kernelContext_13)->qScale_0)[_S325 & 1U][tx_0];
            acc_2[0U][0U] = addTerms_0(acc_2[0U][0U], s_2[0U][0U], s2_1[0U][0U], sumA_2[0U], _S326);
            s_2[0U][0U] = 0.0;
            s2_1[0U][0U] = 0.0;
            acc_2[1U][0U] = addTerms_0(acc_2[1U][0U], s_2[1U][0U], s2_1[1U][0U], sumA_2[1U], _S326);
            s_2[1U][0U] = 0.0;
            s2_1[1U][0U] = 0.0;
            acc_2[2U][0U] = addTerms_0(acc_2[2U][0U], s_2[2U][0U], s2_1[2U][0U], sumA_2[2U], _S326);
            s_2[2U][0U] = 0.0;
            s2_1[2U][0U] = 0.0;
            acc_2[3U][0U] = addTerms_0(acc_2[3U][0U], s_2[3U][0U], s2_1[3U][0U], sumA_2[3U], _S326);
            s_2[3U][0U] = 0.0;
            s2_1[3U][0U] = 0.0;
            uint _S327;
            if(QFORMAT_0 >= 3U)
            {
                _S327 = 32U;
            }
            else
            {
                _S327 = QBLOCK_0;
            }
            uint _S328 = k0_0 / _S327;
            float2 _S329 = (*(&kernelContext_13)->qScale_0)[_S328 & 1U][_S64];
            acc_2[0U][1U] = addTerms_0(acc_2[0U][1U], s_2[0U][1U], s2_1[0U][1U], sumA_2[0U], _S329);
            s_2[0U][1U] = 0.0;
            s2_1[0U][1U] = 0.0;
            acc_2[1U][1U] = addTerms_0(acc_2[1U][1U], s_2[1U][1U], s2_1[1U][1U], sumA_2[1U], _S329);
            s_2[1U][1U] = 0.0;
            s2_1[1U][1U] = 0.0;
            acc_2[2U][1U] = addTerms_0(acc_2[2U][1U], s_2[2U][1U], s2_1[2U][1U], sumA_2[2U], _S329);
            s_2[2U][1U] = 0.0;
            s2_1[2U][1U] = 0.0;
            acc_2[3U][1U] = addTerms_0(acc_2[3U][1U], s_2[3U][1U], s2_1[3U][1U], sumA_2[3U], _S329);
            s_2[3U][1U] = 0.0;
            s2_1[3U][1U] = 0.0;
            uint _S330;
            if(QFORMAT_0 >= 3U)
            {
                _S330 = 32U;
            }
            else
            {
                _S330 = QBLOCK_0;
            }
            uint _S331 = k0_0 / _S330;
            float2 _S332 = (*(&kernelContext_13)->qScale_0)[_S331 & 1U][_S65];
            acc_2[0U][2U] = addTerms_0(acc_2[0U][2U], s_2[0U][2U], s2_1[0U][2U], sumA_2[0U], _S332);
            s_2[0U][2U] = 0.0;
            s2_1[0U][2U] = 0.0;
            acc_2[1U][2U] = addTerms_0(acc_2[1U][2U], s_2[1U][2U], s2_1[1U][2U], sumA_2[1U], _S332);
            s_2[1U][2U] = 0.0;
            s2_1[1U][2U] = 0.0;
            acc_2[2U][2U] = addTerms_0(acc_2[2U][2U], s_2[2U][2U], s2_1[2U][2U], sumA_2[2U], _S332);
            s_2[2U][2U] = 0.0;
            s2_1[2U][2U] = 0.0;
            acc_2[3U][2U] = addTerms_0(acc_2[3U][2U], s_2[3U][2U], s2_1[3U][2U], sumA_2[3U], _S332);
            s_2[3U][2U] = 0.0;
            s2_1[3U][2U] = 0.0;
            uint _S333;
            if(QFORMAT_0 >= 3U)
            {
                _S333 = 32U;
            }
            else
            {
                _S333 = QBLOCK_0;
            }
            uint _S334 = k0_0 / _S333;
            float2 _S335 = (*(&kernelContext_13)->qScale_0)[_S334 & 1U][_S66];
            acc_2[0U][3U] = addTerms_0(acc_2[0U][3U], s_2[0U][3U], s2_1[0U][3U], sumA_2[0U], _S335);
            s_2[0U][3U] = 0.0;
            s2_1[0U][3U] = 0.0;
            acc_2[1U][3U] = addTerms_0(acc_2[1U][3U], s_2[1U][3U], s2_1[1U][3U], sumA_2[1U], _S335);
            s_2[1U][3U] = 0.0;
            s2_1[1U][3U] = 0.0;
            acc_2[2U][3U] = addTerms_0(acc_2[2U][3U], s_2[2U][3U], s2_1[2U][3U], sumA_2[2U], _S335);
            s_2[2U][3U] = 0.0;
            s2_1[2U][3U] = 0.0;
            acc_2[3U][3U] = addTerms_0(acc_2[3U][3U], s_2[3U][3U], s2_1[3U][3U], sumA_2[3U], _S335);
            s_2[3U][3U] = 0.0;
            s2_1[3U][3U] = 0.0;
            sumA_2[0U] = 0.0;
            sumA_2[1U] = 0.0;
            sumA_2[2U] = 0.0;
            sumA_2[3U] = 0.0;
        }
        k0_0 = _S55;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                uint _S336 = _S43 + ty_0;
                _S40 = _S336;
                if(_S336 >= ((&kernelContext_13)->p_0->m_1))
                {
                    break;
                }
                uint _S337 = _S44 + tx_0;
                if(_S337 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S338 = _S41;
                    gemmStore_0(&_S338, _S336, _S337, acc_2[0U][0U], &kernelContext_13);
                }
                uint col_3 = _S337 + 16U;
                if(col_3 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S339 = _S41;
                    gemmStore_0(&_S339, _S336, col_3, acc_2[0U][1U], &kernelContext_13);
                }
                uint col_4 = _S337 + 32U;
                if(col_4 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S340 = _S41;
                    gemmStore_0(&_S340, _S336, col_4, acc_2[0U][2U], &kernelContext_13);
                }
                uint col_5 = _S337 + 48U;
                if(col_5 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S341 = _S41;
                    gemmStore_0(&_S341, _S336, col_5, acc_2[0U][3U], &kernelContext_13);
                }
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint r_7 = _S40 + 16U;
                if(r_7 >= ((&kernelContext_13)->p_0->m_1))
                {
                    break;
                }
                uint _S342 = _S44 + tx_0;
                if(_S342 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S343 = _S41;
                    gemmStore_0(&_S343, r_7, _S342, acc_2[1U][0U], &kernelContext_13);
                }
                uint col_6 = _S342 + 16U;
                if(col_6 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S344 = _S41;
                    gemmStore_0(&_S344, r_7, col_6, acc_2[1U][1U], &kernelContext_13);
                }
                uint col_7 = _S342 + 32U;
                if(col_7 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S345 = _S41;
                    gemmStore_0(&_S345, r_7, col_7, acc_2[1U][2U], &kernelContext_13);
                }
                uint col_8 = _S342 + 48U;
                if(col_8 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S346 = _S41;
                    gemmStore_0(&_S346, r_7, col_8, acc_2[1U][3U], &kernelContext_13);
                }
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint r_8 = _S40 + 32U;
                if(r_8 >= ((&kernelContext_13)->p_0->m_1))
                {
                    break;
                }
                uint _S347 = _S44 + tx_0;
                if(_S347 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S348 = _S41;
                    gemmStore_0(&_S348, r_8, _S347, acc_2[2U][0U], &kernelContext_13);
                }
                uint col_9 = _S347 + 16U;
                if(col_9 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S349 = _S41;
                    gemmStore_0(&_S349, r_8, col_9, acc_2[2U][1U], &kernelContext_13);
                }
                uint col_10 = _S347 + 32U;
                if(col_10 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S350 = _S41;
                    gemmStore_0(&_S350, r_8, col_10, acc_2[2U][2U], &kernelContext_13);
                }
                uint col_11 = _S347 + 48U;
                if(col_11 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S351 = _S41;
                    gemmStore_0(&_S351, r_8, col_11, acc_2[2U][3U], &kernelContext_13);
                }
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint r_9 = _S40 + 48U;
                if(r_9 >= ((&kernelContext_13)->p_0->m_1))
                {
                    break;
                }
                uint _S352 = _S44 + tx_0;
                if(_S352 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S353 = _S41;
                    gemmStore_0(&_S353, r_9, _S352, acc_2[3U][0U], &kernelContext_13);
                }
                uint col_12 = _S352 + 16U;
                if(col_12 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S354 = _S41;
                    gemmStore_0(&_S354, r_9, col_12, acc_2[3U][1U], &kernelContext_13);
                }
                uint col_13 = _S352 + 32U;
                if(col_13 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S355 = _S41;
                    gemmStore_0(&_S355, r_9, col_13, acc_2[3U][2U], &kernelContext_13);
                }
                uint col_14 = _S352 + 48U;
                if(col_14 < ((&kernelContext_13)->p_0->n_1))
                {
                    thread GemmProblem_0 _S356 = _S41;
                    gemmStore_0(&_S356, r_9, col_14, acc_2[3U][3U], &kernelContext_13);
                }
                break;
            }
            break;
        }
        break;
    }
    return;
}

