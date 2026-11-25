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
    uint device* a_0;
    uint device* b_0;
    half device* c_0;
    half device* epBias_0;
    half device* epGate_0;
    half device* epResidual_0;
    array<array<float, int(64)>, int(32)> threadgroup* qTileA_0;
    array<array<float, int(64)>, int(32)> threadgroup* qTileB_0;
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

float aLo_0(uint w_0)
{
    return (as_type<float>((w_0 << 16U)));
}

float aHi_0(uint w_1)
{
    return (as_type<float>((w_1 & 4294901760U)));
}

float elemA_0(uint e_0, KernelContext_0 thread* kernelContext_1)
{
    uint w_2 = *(kernelContext_1->a_0+(e_0 >> 1U));
    float _S1;
    if((e_0 & 1U) != 0U)
    {
        _S1 = aHi_0(w_2);
    }
    else
    {
        _S1 = aLo_0(w_2);
    }
    return _S1;
}

void load8A_0(uint e_1, array<float, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_2)
{
    if((e_1 & 1U) == 0U)
    {
        uint _S2 = e_1 >> 1U;
        uint device* _S3 = kernelContext_2->a_0+_S2;
        uint w_3 = *_S3;
        (*v_0)[0U] = aLo_0(*_S3);
        (*v_0)[1U] = aHi_0(w_3);
        uint device* _S4 = kernelContext_2->a_0+(_S2 + 1U);
        uint w_4 = *_S4;
        (*v_0)[2U] = aLo_0(*_S4);
        (*v_0)[3U] = aHi_0(w_4);
        uint device* _S5 = kernelContext_2->a_0+(_S2 + 2U);
        uint w_5 = *_S5;
        (*v_0)[4U] = aLo_0(*_S5);
        (*v_0)[5U] = aHi_0(w_5);
        uint device* _S6 = kernelContext_2->a_0+(_S2 + 3U);
        uint w_6 = *_S6;
        (*v_0)[6U] = aLo_0(*_S6);
        (*v_0)[7U] = aHi_0(w_6);
    }
    else
    {
        float _S7 = elemA_0(e_1, kernelContext_2);
        (*v_0)[0U] = _S7;
        float _S8 = elemA_0(e_1 + 1U, kernelContext_2);
        (*v_0)[1U] = _S8;
        float _S9 = elemA_0(e_1 + 2U, kernelContext_2);
        (*v_0)[2U] = _S9;
        float _S10 = elemA_0(e_1 + 3U, kernelContext_2);
        (*v_0)[3U] = _S10;
        float _S11 = elemA_0(e_1 + 4U, kernelContext_2);
        (*v_0)[4U] = _S11;
        float _S12 = elemA_0(e_1 + 5U, kernelContext_2);
        (*v_0)[5U] = _S12;
        float _S13 = elemA_0(e_1 + 6U, kernelContext_2);
        (*v_0)[6U] = _S13;
        float _S14 = elemA_0(e_1 + 7U, kernelContext_2);
        (*v_0)[7U] = _S14;
    }
    return;
}

uint wordB_0(uint w_7, KernelContext_0 thread* kernelContext_3)
{
    return *(kernelContext_3->b_0+w_7);
}

int qCode_0(uint format_0, uint word_0, uint col_0)
{
    int _S15;
    if(format_0 == 0U)
    {
        uint b_1 = (word_0 >> (8U * (col_0 & 3U))) & 255U;
        int _S16 = int(b_1);
        if(b_1 >= 128U)
        {
            _S15 = int(256);
        }
        else
        {
            _S15 = int(0);
        }
        return _S16 - _S15;
    }
    int u_0 = int((word_0 >> (4U * (col_0 & 7U))) & 15U);
    if(format_0 == 1U)
    {
        _S15 = u_0 - int(8);
    }
    else
    {
        _S15 = u_0;
    }
    return _S15;
}

void load8Codes_0(uint r_0, uint c0_0, array<float, int(8)> thread* q_1, KernelContext_0 thread* kernelContext_4)
{
    if(QFORMAT_0 == 0U)
    {
        uint w_8 = (r_0 * kernelContext_4->p_0->k_1 + c0_0) >> 2U;
        uint _S17 = wordB_0(w_8, kernelContext_4);
        uint _S18 = wordB_0(w_8 + 1U, kernelContext_4);
        (*q_1)[0U] = float(qCode_0(QFORMAT_0, _S17, 0U));
        (*q_1)[4U] = float(qCode_0(QFORMAT_0, _S18, 0U));
        (*q_1)[1U] = float(qCode_0(QFORMAT_0, _S17, 1U));
        (*q_1)[5U] = float(qCode_0(QFORMAT_0, _S18, 1U));
        (*q_1)[2U] = float(qCode_0(QFORMAT_0, _S17, 2U));
        (*q_1)[6U] = float(qCode_0(QFORMAT_0, _S18, 2U));
        (*q_1)[3U] = float(qCode_0(QFORMAT_0, _S17, 3U));
        (*q_1)[7U] = float(qCode_0(QFORMAT_0, _S18, 3U));
    }
    else
    {
        uint _S19 = wordB_0((r_0 * kernelContext_4->p_0->k_1 + c0_0) >> 3U, kernelContext_4);
        (*q_1)[0U] = float(qCode_0(QFORMAT_0, _S19, 0U));
        (*q_1)[1U] = float(qCode_0(QFORMAT_0, _S19, 1U));
        (*q_1)[2U] = float(qCode_0(QFORMAT_0, _S19, 2U));
        (*q_1)[3U] = float(qCode_0(QFORMAT_0, _S19, 3U));
        (*q_1)[4U] = float(qCode_0(QFORMAT_0, _S19, 4U));
        (*q_1)[5U] = float(qCode_0(QFORMAT_0, _S19, 5U));
        (*q_1)[6U] = float(qCode_0(QFORMAT_0, _S19, 6U));
        (*q_1)[7U] = float(qCode_0(QFORMAT_0, _S19, 7U));
    }
    return;
}

constant uint fc_QBLOCK_0 [[function_constant(2)]];
constant uint QBLOCK_0 = is_function_constant_defined(fc_QBLOCK_0) ? fc_QBLOCK_0 : 32U;
uint qParamHalf_0(uint format_1, uint b_2, uint t_0)
{
    uint _S20;
    if(format_1 == 2U)
    {
        _S20 = 2U * b_2 + t_0;
    }
    else
    {
        _S20 = b_2;
    }
    return _S20;
}

float qF16ToF32_0(uint h_0)
{
    return (as_type<half>((ushort)((h_0 & 65535U))));
}

float blockParameter_0(uint r_1, uint bl_0, uint t_1, KernelContext_0 thread* kernelContext_5)
{
    uint h_1 = qParamHalf_0(QFORMAT_0, r_1 * (kernelContext_5->p_0->k_1 / QBLOCK_0) + bl_0, t_1);
    uint _S21 = wordB_0(kernelContext_5->p_0->qparams_0 + (h_1 >> 1U), kernelContext_5);
    return qF16ToF32_0(_S21 >> (16U * (h_1 & 1U)));
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
        _S26 = r_2 + float(*(kernelContext_6->epBias_0+(biasBase_0 + j_0)));
    }
    else
    {
        if((kernelContext_6->p_0->bias_0) == 2U)
        {
            _S26 = r_2 + float(*(kernelContext_6->epBias_0+(biasBase_0 + i_0)));
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
        _S26 = _S26 * float(*(kernelContext_6->epGate_0+idx_0));
    }
    if((kernelContext_6->p_0->residual_0) != 0U)
    {
        _S26 = _S26 + float(*(kernelContext_6->epResidual_0+idx_0));
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
        r_3 = kernelContext_7->p_0->alpha_0 * acc_0 + kernelContext_7->p_0->beta_0 * float(*(kernelContext_7->c_0+idx_1));
    }
    half device* _S27 = kernelContext_7->c_0+idx_1;
    float _S28 = gemmEpilogue_0(r_3, i_1, j_1, idx_1, q_2->baseBias_0, kernelContext_7);
    *_S27 = half(_S28);
    return;
}

[[kernel]] void gemm_q_tiled_bf16_h(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], GemmParams_0 constant* p_1 [[buffer(6)]], uint device* a_2 [[buffer(0)]], uint device* b_4 [[buffer(1)]], half device* c_1 [[buffer(2)]], half device* epBias_1 [[buffer(3)]], half device* epGate_1 [[buffer(4)]], half device* epResidual_1 [[buffer(5)]])
{
    uint _S29;
    thread KernelContext_0 kernelContext_8;
    (&kernelContext_8)->p_0 = p_1;
    (&kernelContext_8)->a_0 = a_2;
    (&kernelContext_8)->b_0 = b_4;
    (&kernelContext_8)->c_0 = c_1;
    (&kernelContext_8)->epBias_0 = epBias_1;
    (&kernelContext_8)->epGate_0 = epGate_1;
    (&kernelContext_8)->epResidual_0 = epResidual_1;
    threadgroup array<array<float, int(64)>, int(32)> qTileA_1;
    (&kernelContext_8)->qTileA_0 = &qTileA_1;
    threadgroup array<array<float, int(64)>, int(32)> qTileB_1;
    (&kernelContext_8)->qTileB_0 = &qTileB_1;
    GemmProblem_0 _S30 = gemmBatchProblem_0(group_0.z, &kernelContext_8);
    uint tx_0 = local_0.x;
    uint ty_0 = local_0.y;
    uint _S31 = ty_0 * 16U + tx_0;
    uint _S32 = group_0.y * 64U;
    uint _S33 = group_0.x * 64U;
    thread array<float, int(4)> sumA_1;
    sumA_1[0U] = 0.0;
    thread array<array<float, int(4)>, int(4)> acc_1;
    acc_1[0U][0U] = 0.0;
    thread array<array<float, int(4)>, int(4)> s_1;
    s_1[0U][0U] = 0.0;
    acc_1[0U][1U] = 0.0;
    s_1[0U][1U] = 0.0;
    acc_1[0U][2U] = 0.0;
    s_1[0U][2U] = 0.0;
    acc_1[0U][3U] = 0.0;
    s_1[0U][3U] = 0.0;
    sumA_1[1U] = 0.0;
    acc_1[1U][0U] = 0.0;
    s_1[1U][0U] = 0.0;
    acc_1[1U][1U] = 0.0;
    s_1[1U][1U] = 0.0;
    acc_1[1U][2U] = 0.0;
    s_1[1U][2U] = 0.0;
    acc_1[1U][3U] = 0.0;
    s_1[1U][3U] = 0.0;
    sumA_1[2U] = 0.0;
    acc_1[2U][0U] = 0.0;
    s_1[2U][0U] = 0.0;
    acc_1[2U][1U] = 0.0;
    s_1[2U][1U] = 0.0;
    acc_1[2U][2U] = 0.0;
    s_1[2U][2U] = 0.0;
    acc_1[2U][3U] = 0.0;
    s_1[2U][3U] = 0.0;
    sumA_1[3U] = 0.0;
    acc_1[3U][0U] = 0.0;
    s_1[3U][0U] = 0.0;
    acc_1[3U][1U] = 0.0;
    s_1[3U][1U] = 0.0;
    acc_1[3U][2U] = 0.0;
    s_1[3U][2U] = 0.0;
    acc_1[3U][3U] = 0.0;
    s_1[3U][3U] = 0.0;
    uint k0_0 = 0U;
    for(;;)
    {
        if(k0_0 < ((&kernelContext_8)->p_0->k_1))
        {
        }
        else
        {
            break;
        }
        uint ar_0 = _S31 / 4U;
        uint ak_0 = _S31 % 4U * 8U;
        uint gr_0 = _S32 + ar_0;
        thread array<float, int(8)> av_0;
        if(gr_0 < ((&kernelContext_8)->p_0->m_1))
        {
            load8A_0(_S30.baseA_0 + gr_0 * (&kernelContext_8)->p_0->lda_1 + k0_0 + ak_0, &av_0, &kernelContext_8);
        }
        else
        {
            av_0[0U] = 0.0;
            av_0[1U] = 0.0;
            av_0[2U] = 0.0;
            av_0[3U] = 0.0;
            av_0[4U] = 0.0;
            av_0[5U] = 0.0;
            av_0[6U] = 0.0;
            av_0[7U] = 0.0;
        }
        (*(&kernelContext_8)->qTileA_0)[ak_0][ar_0] = av_0[0U];
        uint _S34 = ak_0 + 1U;
        (*(&kernelContext_8)->qTileA_0)[_S34][ar_0] = av_0[1U];
        uint _S35 = ak_0 + 2U;
        (*(&kernelContext_8)->qTileA_0)[_S35][ar_0] = av_0[2U];
        uint _S36 = ak_0 + 3U;
        (*(&kernelContext_8)->qTileA_0)[_S36][ar_0] = av_0[3U];
        uint _S37 = ak_0 + 4U;
        (*(&kernelContext_8)->qTileA_0)[_S37][ar_0] = av_0[4U];
        uint _S38 = ak_0 + 5U;
        (*(&kernelContext_8)->qTileA_0)[_S38][ar_0] = av_0[5U];
        uint _S39 = ak_0 + 6U;
        (*(&kernelContext_8)->qTileA_0)[_S39][ar_0] = av_0[6U];
        uint _S40 = ak_0 + 7U;
        (*(&kernelContext_8)->qTileA_0)[_S40][ar_0] = av_0[7U];
        thread array<float, int(8)> q_3;
        load8Codes_0(min(_S33 + _S31 / 4U, (&kernelContext_8)->p_0->n_1 - 1U), k0_0 + ak_0, &q_3, &kernelContext_8);
        (*(&kernelContext_8)->qTileB_0)[ak_0][_S31 / 4U] = q_3[0U];
        (*(&kernelContext_8)->qTileB_0)[_S34][_S31 / 4U] = q_3[1U];
        (*(&kernelContext_8)->qTileB_0)[_S35][_S31 / 4U] = q_3[2U];
        (*(&kernelContext_8)->qTileB_0)[_S36][_S31 / 4U] = q_3[3U];
        (*(&kernelContext_8)->qTileB_0)[_S37][_S31 / 4U] = q_3[4U];
        (*(&kernelContext_8)->qTileB_0)[_S38][_S31 / 4U] = q_3[5U];
        (*(&kernelContext_8)->qTileB_0)[_S39][_S31 / 4U] = q_3[6U];
        (*(&kernelContext_8)->qTileB_0)[_S40][_S31 / 4U] = q_3[7U];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        thread array<float, int(4)> av_1;
        av_1[0U] = (*(&kernelContext_8)->qTileA_0)[0U][ty_0];
        uint _S41 = ty_0 + 16U;
        av_1[1U] = (*(&kernelContext_8)->qTileA_0)[0U][_S41];
        uint _S42 = ty_0 + 32U;
        av_1[2U] = (*(&kernelContext_8)->qTileA_0)[0U][_S42];
        uint _S43 = ty_0 + 48U;
        av_1[3U] = (*(&kernelContext_8)->qTileA_0)[0U][_S43];
        thread array<float, int(4)> bv_0;
        bv_0[0U] = (*(&kernelContext_8)->qTileB_0)[0U][tx_0];
        uint _S44 = tx_0 + 16U;
        bv_0[1U] = (*(&kernelContext_8)->qTileB_0)[0U][_S44];
        uint _S45 = tx_0 + 32U;
        bv_0[2U] = (*(&kernelContext_8)->qTileB_0)[0U][_S45];
        uint _S46 = tx_0 + 48U;
        bv_0[3U] = (*(&kernelContext_8)->qTileB_0)[0U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_1[0U];
        }
        s_1[0U][0U] = fma(av_1[0U], bv_0[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_1[0U], bv_0[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_1[0U], bv_0[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_1[0U], bv_0[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_1[1U];
        }
        s_1[1U][0U] = fma(av_1[1U], bv_0[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_1[1U], bv_0[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_1[1U], bv_0[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_1[1U], bv_0[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_1[2U];
        }
        s_1[2U][0U] = fma(av_1[2U], bv_0[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_1[2U], bv_0[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_1[2U], bv_0[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_1[2U], bv_0[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_1[3U];
        }
        s_1[3U][0U] = fma(av_1[3U], bv_0[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_1[3U], bv_0[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_1[3U], bv_0[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_1[3U], bv_0[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_2;
        av_2[0U] = (*(&kernelContext_8)->qTileA_0)[1U][ty_0];
        av_2[1U] = (*(&kernelContext_8)->qTileA_0)[1U][_S41];
        av_2[2U] = (*(&kernelContext_8)->qTileA_0)[1U][_S42];
        av_2[3U] = (*(&kernelContext_8)->qTileA_0)[1U][_S43];
        thread array<float, int(4)> bv_1;
        bv_1[0U] = (*(&kernelContext_8)->qTileB_0)[1U][tx_0];
        bv_1[1U] = (*(&kernelContext_8)->qTileB_0)[1U][_S44];
        bv_1[2U] = (*(&kernelContext_8)->qTileB_0)[1U][_S45];
        bv_1[3U] = (*(&kernelContext_8)->qTileB_0)[1U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_2[0U];
        }
        s_1[0U][0U] = fma(av_2[0U], bv_1[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_2[0U], bv_1[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_2[0U], bv_1[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_2[0U], bv_1[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_2[1U];
        }
        s_1[1U][0U] = fma(av_2[1U], bv_1[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_2[1U], bv_1[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_2[1U], bv_1[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_2[1U], bv_1[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_2[2U];
        }
        s_1[2U][0U] = fma(av_2[2U], bv_1[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_2[2U], bv_1[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_2[2U], bv_1[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_2[2U], bv_1[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_2[3U];
        }
        s_1[3U][0U] = fma(av_2[3U], bv_1[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_2[3U], bv_1[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_2[3U], bv_1[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_2[3U], bv_1[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_3;
        av_3[0U] = (*(&kernelContext_8)->qTileA_0)[2U][ty_0];
        av_3[1U] = (*(&kernelContext_8)->qTileA_0)[2U][_S41];
        av_3[2U] = (*(&kernelContext_8)->qTileA_0)[2U][_S42];
        av_3[3U] = (*(&kernelContext_8)->qTileA_0)[2U][_S43];
        thread array<float, int(4)> bv_2;
        bv_2[0U] = (*(&kernelContext_8)->qTileB_0)[2U][tx_0];
        bv_2[1U] = (*(&kernelContext_8)->qTileB_0)[2U][_S44];
        bv_2[2U] = (*(&kernelContext_8)->qTileB_0)[2U][_S45];
        bv_2[3U] = (*(&kernelContext_8)->qTileB_0)[2U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_3[0U];
        }
        s_1[0U][0U] = fma(av_3[0U], bv_2[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_3[0U], bv_2[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_3[0U], bv_2[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_3[0U], bv_2[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_3[1U];
        }
        s_1[1U][0U] = fma(av_3[1U], bv_2[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_3[1U], bv_2[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_3[1U], bv_2[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_3[1U], bv_2[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_3[2U];
        }
        s_1[2U][0U] = fma(av_3[2U], bv_2[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_3[2U], bv_2[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_3[2U], bv_2[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_3[2U], bv_2[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_3[3U];
        }
        s_1[3U][0U] = fma(av_3[3U], bv_2[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_3[3U], bv_2[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_3[3U], bv_2[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_3[3U], bv_2[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_4;
        av_4[0U] = (*(&kernelContext_8)->qTileA_0)[3U][ty_0];
        av_4[1U] = (*(&kernelContext_8)->qTileA_0)[3U][_S41];
        av_4[2U] = (*(&kernelContext_8)->qTileA_0)[3U][_S42];
        av_4[3U] = (*(&kernelContext_8)->qTileA_0)[3U][_S43];
        thread array<float, int(4)> bv_3;
        bv_3[0U] = (*(&kernelContext_8)->qTileB_0)[3U][tx_0];
        bv_3[1U] = (*(&kernelContext_8)->qTileB_0)[3U][_S44];
        bv_3[2U] = (*(&kernelContext_8)->qTileB_0)[3U][_S45];
        bv_3[3U] = (*(&kernelContext_8)->qTileB_0)[3U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_4[0U];
        }
        s_1[0U][0U] = fma(av_4[0U], bv_3[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_4[0U], bv_3[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_4[0U], bv_3[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_4[0U], bv_3[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_4[1U];
        }
        s_1[1U][0U] = fma(av_4[1U], bv_3[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_4[1U], bv_3[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_4[1U], bv_3[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_4[1U], bv_3[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_4[2U];
        }
        s_1[2U][0U] = fma(av_4[2U], bv_3[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_4[2U], bv_3[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_4[2U], bv_3[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_4[2U], bv_3[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_4[3U];
        }
        s_1[3U][0U] = fma(av_4[3U], bv_3[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_4[3U], bv_3[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_4[3U], bv_3[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_4[3U], bv_3[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_5;
        av_5[0U] = (*(&kernelContext_8)->qTileA_0)[4U][ty_0];
        av_5[1U] = (*(&kernelContext_8)->qTileA_0)[4U][_S41];
        av_5[2U] = (*(&kernelContext_8)->qTileA_0)[4U][_S42];
        av_5[3U] = (*(&kernelContext_8)->qTileA_0)[4U][_S43];
        thread array<float, int(4)> bv_4;
        bv_4[0U] = (*(&kernelContext_8)->qTileB_0)[4U][tx_0];
        bv_4[1U] = (*(&kernelContext_8)->qTileB_0)[4U][_S44];
        bv_4[2U] = (*(&kernelContext_8)->qTileB_0)[4U][_S45];
        bv_4[3U] = (*(&kernelContext_8)->qTileB_0)[4U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_5[0U];
        }
        s_1[0U][0U] = fma(av_5[0U], bv_4[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_5[0U], bv_4[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_5[0U], bv_4[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_5[0U], bv_4[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_5[1U];
        }
        s_1[1U][0U] = fma(av_5[1U], bv_4[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_5[1U], bv_4[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_5[1U], bv_4[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_5[1U], bv_4[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_5[2U];
        }
        s_1[2U][0U] = fma(av_5[2U], bv_4[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_5[2U], bv_4[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_5[2U], bv_4[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_5[2U], bv_4[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_5[3U];
        }
        s_1[3U][0U] = fma(av_5[3U], bv_4[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_5[3U], bv_4[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_5[3U], bv_4[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_5[3U], bv_4[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_6;
        av_6[0U] = (*(&kernelContext_8)->qTileA_0)[5U][ty_0];
        av_6[1U] = (*(&kernelContext_8)->qTileA_0)[5U][_S41];
        av_6[2U] = (*(&kernelContext_8)->qTileA_0)[5U][_S42];
        av_6[3U] = (*(&kernelContext_8)->qTileA_0)[5U][_S43];
        thread array<float, int(4)> bv_5;
        bv_5[0U] = (*(&kernelContext_8)->qTileB_0)[5U][tx_0];
        bv_5[1U] = (*(&kernelContext_8)->qTileB_0)[5U][_S44];
        bv_5[2U] = (*(&kernelContext_8)->qTileB_0)[5U][_S45];
        bv_5[3U] = (*(&kernelContext_8)->qTileB_0)[5U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_6[0U];
        }
        s_1[0U][0U] = fma(av_6[0U], bv_5[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_6[0U], bv_5[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_6[0U], bv_5[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_6[0U], bv_5[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_6[1U];
        }
        s_1[1U][0U] = fma(av_6[1U], bv_5[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_6[1U], bv_5[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_6[1U], bv_5[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_6[1U], bv_5[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_6[2U];
        }
        s_1[2U][0U] = fma(av_6[2U], bv_5[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_6[2U], bv_5[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_6[2U], bv_5[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_6[2U], bv_5[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_6[3U];
        }
        s_1[3U][0U] = fma(av_6[3U], bv_5[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_6[3U], bv_5[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_6[3U], bv_5[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_6[3U], bv_5[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_7;
        av_7[0U] = (*(&kernelContext_8)->qTileA_0)[6U][ty_0];
        av_7[1U] = (*(&kernelContext_8)->qTileA_0)[6U][_S41];
        av_7[2U] = (*(&kernelContext_8)->qTileA_0)[6U][_S42];
        av_7[3U] = (*(&kernelContext_8)->qTileA_0)[6U][_S43];
        thread array<float, int(4)> bv_6;
        bv_6[0U] = (*(&kernelContext_8)->qTileB_0)[6U][tx_0];
        bv_6[1U] = (*(&kernelContext_8)->qTileB_0)[6U][_S44];
        bv_6[2U] = (*(&kernelContext_8)->qTileB_0)[6U][_S45];
        bv_6[3U] = (*(&kernelContext_8)->qTileB_0)[6U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_7[0U];
        }
        s_1[0U][0U] = fma(av_7[0U], bv_6[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_7[0U], bv_6[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_7[0U], bv_6[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_7[0U], bv_6[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_7[1U];
        }
        s_1[1U][0U] = fma(av_7[1U], bv_6[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_7[1U], bv_6[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_7[1U], bv_6[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_7[1U], bv_6[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_7[2U];
        }
        s_1[2U][0U] = fma(av_7[2U], bv_6[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_7[2U], bv_6[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_7[2U], bv_6[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_7[2U], bv_6[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_7[3U];
        }
        s_1[3U][0U] = fma(av_7[3U], bv_6[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_7[3U], bv_6[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_7[3U], bv_6[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_7[3U], bv_6[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_8;
        av_8[0U] = (*(&kernelContext_8)->qTileA_0)[7U][ty_0];
        av_8[1U] = (*(&kernelContext_8)->qTileA_0)[7U][_S41];
        av_8[2U] = (*(&kernelContext_8)->qTileA_0)[7U][_S42];
        av_8[3U] = (*(&kernelContext_8)->qTileA_0)[7U][_S43];
        thread array<float, int(4)> bv_7;
        bv_7[0U] = (*(&kernelContext_8)->qTileB_0)[7U][tx_0];
        bv_7[1U] = (*(&kernelContext_8)->qTileB_0)[7U][_S44];
        bv_7[2U] = (*(&kernelContext_8)->qTileB_0)[7U][_S45];
        bv_7[3U] = (*(&kernelContext_8)->qTileB_0)[7U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_8[0U];
        }
        s_1[0U][0U] = fma(av_8[0U], bv_7[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_8[0U], bv_7[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_8[0U], bv_7[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_8[0U], bv_7[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_8[1U];
        }
        s_1[1U][0U] = fma(av_8[1U], bv_7[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_8[1U], bv_7[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_8[1U], bv_7[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_8[1U], bv_7[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_8[2U];
        }
        s_1[2U][0U] = fma(av_8[2U], bv_7[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_8[2U], bv_7[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_8[2U], bv_7[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_8[2U], bv_7[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_8[3U];
        }
        s_1[3U][0U] = fma(av_8[3U], bv_7[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_8[3U], bv_7[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_8[3U], bv_7[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_8[3U], bv_7[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_9;
        av_9[0U] = (*(&kernelContext_8)->qTileA_0)[8U][ty_0];
        av_9[1U] = (*(&kernelContext_8)->qTileA_0)[8U][_S41];
        av_9[2U] = (*(&kernelContext_8)->qTileA_0)[8U][_S42];
        av_9[3U] = (*(&kernelContext_8)->qTileA_0)[8U][_S43];
        thread array<float, int(4)> bv_8;
        bv_8[0U] = (*(&kernelContext_8)->qTileB_0)[8U][tx_0];
        bv_8[1U] = (*(&kernelContext_8)->qTileB_0)[8U][_S44];
        bv_8[2U] = (*(&kernelContext_8)->qTileB_0)[8U][_S45];
        bv_8[3U] = (*(&kernelContext_8)->qTileB_0)[8U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_9[0U];
        }
        s_1[0U][0U] = fma(av_9[0U], bv_8[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_9[0U], bv_8[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_9[0U], bv_8[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_9[0U], bv_8[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_9[1U];
        }
        s_1[1U][0U] = fma(av_9[1U], bv_8[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_9[1U], bv_8[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_9[1U], bv_8[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_9[1U], bv_8[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_9[2U];
        }
        s_1[2U][0U] = fma(av_9[2U], bv_8[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_9[2U], bv_8[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_9[2U], bv_8[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_9[2U], bv_8[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_9[3U];
        }
        s_1[3U][0U] = fma(av_9[3U], bv_8[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_9[3U], bv_8[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_9[3U], bv_8[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_9[3U], bv_8[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_10;
        av_10[0U] = (*(&kernelContext_8)->qTileA_0)[9U][ty_0];
        av_10[1U] = (*(&kernelContext_8)->qTileA_0)[9U][_S41];
        av_10[2U] = (*(&kernelContext_8)->qTileA_0)[9U][_S42];
        av_10[3U] = (*(&kernelContext_8)->qTileA_0)[9U][_S43];
        thread array<float, int(4)> bv_9;
        bv_9[0U] = (*(&kernelContext_8)->qTileB_0)[9U][tx_0];
        bv_9[1U] = (*(&kernelContext_8)->qTileB_0)[9U][_S44];
        bv_9[2U] = (*(&kernelContext_8)->qTileB_0)[9U][_S45];
        bv_9[3U] = (*(&kernelContext_8)->qTileB_0)[9U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_10[0U];
        }
        s_1[0U][0U] = fma(av_10[0U], bv_9[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_10[0U], bv_9[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_10[0U], bv_9[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_10[0U], bv_9[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_10[1U];
        }
        s_1[1U][0U] = fma(av_10[1U], bv_9[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_10[1U], bv_9[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_10[1U], bv_9[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_10[1U], bv_9[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_10[2U];
        }
        s_1[2U][0U] = fma(av_10[2U], bv_9[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_10[2U], bv_9[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_10[2U], bv_9[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_10[2U], bv_9[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_10[3U];
        }
        s_1[3U][0U] = fma(av_10[3U], bv_9[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_10[3U], bv_9[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_10[3U], bv_9[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_10[3U], bv_9[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_11;
        av_11[0U] = (*(&kernelContext_8)->qTileA_0)[10U][ty_0];
        av_11[1U] = (*(&kernelContext_8)->qTileA_0)[10U][_S41];
        av_11[2U] = (*(&kernelContext_8)->qTileA_0)[10U][_S42];
        av_11[3U] = (*(&kernelContext_8)->qTileA_0)[10U][_S43];
        thread array<float, int(4)> bv_10;
        bv_10[0U] = (*(&kernelContext_8)->qTileB_0)[10U][tx_0];
        bv_10[1U] = (*(&kernelContext_8)->qTileB_0)[10U][_S44];
        bv_10[2U] = (*(&kernelContext_8)->qTileB_0)[10U][_S45];
        bv_10[3U] = (*(&kernelContext_8)->qTileB_0)[10U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_11[0U];
        }
        s_1[0U][0U] = fma(av_11[0U], bv_10[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_11[0U], bv_10[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_11[0U], bv_10[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_11[0U], bv_10[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_11[1U];
        }
        s_1[1U][0U] = fma(av_11[1U], bv_10[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_11[1U], bv_10[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_11[1U], bv_10[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_11[1U], bv_10[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_11[2U];
        }
        s_1[2U][0U] = fma(av_11[2U], bv_10[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_11[2U], bv_10[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_11[2U], bv_10[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_11[2U], bv_10[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_11[3U];
        }
        s_1[3U][0U] = fma(av_11[3U], bv_10[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_11[3U], bv_10[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_11[3U], bv_10[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_11[3U], bv_10[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_12;
        av_12[0U] = (*(&kernelContext_8)->qTileA_0)[11U][ty_0];
        av_12[1U] = (*(&kernelContext_8)->qTileA_0)[11U][_S41];
        av_12[2U] = (*(&kernelContext_8)->qTileA_0)[11U][_S42];
        av_12[3U] = (*(&kernelContext_8)->qTileA_0)[11U][_S43];
        thread array<float, int(4)> bv_11;
        bv_11[0U] = (*(&kernelContext_8)->qTileB_0)[11U][tx_0];
        bv_11[1U] = (*(&kernelContext_8)->qTileB_0)[11U][_S44];
        bv_11[2U] = (*(&kernelContext_8)->qTileB_0)[11U][_S45];
        bv_11[3U] = (*(&kernelContext_8)->qTileB_0)[11U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_12[0U];
        }
        s_1[0U][0U] = fma(av_12[0U], bv_11[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_12[0U], bv_11[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_12[0U], bv_11[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_12[0U], bv_11[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_12[1U];
        }
        s_1[1U][0U] = fma(av_12[1U], bv_11[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_12[1U], bv_11[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_12[1U], bv_11[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_12[1U], bv_11[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_12[2U];
        }
        s_1[2U][0U] = fma(av_12[2U], bv_11[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_12[2U], bv_11[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_12[2U], bv_11[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_12[2U], bv_11[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_12[3U];
        }
        s_1[3U][0U] = fma(av_12[3U], bv_11[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_12[3U], bv_11[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_12[3U], bv_11[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_12[3U], bv_11[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_13;
        av_13[0U] = (*(&kernelContext_8)->qTileA_0)[12U][ty_0];
        av_13[1U] = (*(&kernelContext_8)->qTileA_0)[12U][_S41];
        av_13[2U] = (*(&kernelContext_8)->qTileA_0)[12U][_S42];
        av_13[3U] = (*(&kernelContext_8)->qTileA_0)[12U][_S43];
        thread array<float, int(4)> bv_12;
        bv_12[0U] = (*(&kernelContext_8)->qTileB_0)[12U][tx_0];
        bv_12[1U] = (*(&kernelContext_8)->qTileB_0)[12U][_S44];
        bv_12[2U] = (*(&kernelContext_8)->qTileB_0)[12U][_S45];
        bv_12[3U] = (*(&kernelContext_8)->qTileB_0)[12U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_13[0U];
        }
        s_1[0U][0U] = fma(av_13[0U], bv_12[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_13[0U], bv_12[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_13[0U], bv_12[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_13[0U], bv_12[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_13[1U];
        }
        s_1[1U][0U] = fma(av_13[1U], bv_12[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_13[1U], bv_12[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_13[1U], bv_12[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_13[1U], bv_12[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_13[2U];
        }
        s_1[2U][0U] = fma(av_13[2U], bv_12[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_13[2U], bv_12[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_13[2U], bv_12[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_13[2U], bv_12[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_13[3U];
        }
        s_1[3U][0U] = fma(av_13[3U], bv_12[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_13[3U], bv_12[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_13[3U], bv_12[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_13[3U], bv_12[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_14;
        av_14[0U] = (*(&kernelContext_8)->qTileA_0)[13U][ty_0];
        av_14[1U] = (*(&kernelContext_8)->qTileA_0)[13U][_S41];
        av_14[2U] = (*(&kernelContext_8)->qTileA_0)[13U][_S42];
        av_14[3U] = (*(&kernelContext_8)->qTileA_0)[13U][_S43];
        thread array<float, int(4)> bv_13;
        bv_13[0U] = (*(&kernelContext_8)->qTileB_0)[13U][tx_0];
        bv_13[1U] = (*(&kernelContext_8)->qTileB_0)[13U][_S44];
        bv_13[2U] = (*(&kernelContext_8)->qTileB_0)[13U][_S45];
        bv_13[3U] = (*(&kernelContext_8)->qTileB_0)[13U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_14[0U];
        }
        s_1[0U][0U] = fma(av_14[0U], bv_13[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_14[0U], bv_13[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_14[0U], bv_13[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_14[0U], bv_13[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_14[1U];
        }
        s_1[1U][0U] = fma(av_14[1U], bv_13[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_14[1U], bv_13[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_14[1U], bv_13[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_14[1U], bv_13[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_14[2U];
        }
        s_1[2U][0U] = fma(av_14[2U], bv_13[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_14[2U], bv_13[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_14[2U], bv_13[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_14[2U], bv_13[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_14[3U];
        }
        s_1[3U][0U] = fma(av_14[3U], bv_13[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_14[3U], bv_13[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_14[3U], bv_13[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_14[3U], bv_13[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_15;
        av_15[0U] = (*(&kernelContext_8)->qTileA_0)[14U][ty_0];
        av_15[1U] = (*(&kernelContext_8)->qTileA_0)[14U][_S41];
        av_15[2U] = (*(&kernelContext_8)->qTileA_0)[14U][_S42];
        av_15[3U] = (*(&kernelContext_8)->qTileA_0)[14U][_S43];
        thread array<float, int(4)> bv_14;
        bv_14[0U] = (*(&kernelContext_8)->qTileB_0)[14U][tx_0];
        bv_14[1U] = (*(&kernelContext_8)->qTileB_0)[14U][_S44];
        bv_14[2U] = (*(&kernelContext_8)->qTileB_0)[14U][_S45];
        bv_14[3U] = (*(&kernelContext_8)->qTileB_0)[14U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_15[0U];
        }
        s_1[0U][0U] = fma(av_15[0U], bv_14[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_15[0U], bv_14[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_15[0U], bv_14[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_15[0U], bv_14[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_15[1U];
        }
        s_1[1U][0U] = fma(av_15[1U], bv_14[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_15[1U], bv_14[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_15[1U], bv_14[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_15[1U], bv_14[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_15[2U];
        }
        s_1[2U][0U] = fma(av_15[2U], bv_14[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_15[2U], bv_14[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_15[2U], bv_14[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_15[2U], bv_14[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_15[3U];
        }
        s_1[3U][0U] = fma(av_15[3U], bv_14[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_15[3U], bv_14[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_15[3U], bv_14[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_15[3U], bv_14[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_16;
        av_16[0U] = (*(&kernelContext_8)->qTileA_0)[15U][ty_0];
        av_16[1U] = (*(&kernelContext_8)->qTileA_0)[15U][_S41];
        av_16[2U] = (*(&kernelContext_8)->qTileA_0)[15U][_S42];
        av_16[3U] = (*(&kernelContext_8)->qTileA_0)[15U][_S43];
        thread array<float, int(4)> bv_15;
        bv_15[0U] = (*(&kernelContext_8)->qTileB_0)[15U][tx_0];
        bv_15[1U] = (*(&kernelContext_8)->qTileB_0)[15U][_S44];
        bv_15[2U] = (*(&kernelContext_8)->qTileB_0)[15U][_S45];
        bv_15[3U] = (*(&kernelContext_8)->qTileB_0)[15U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_16[0U];
        }
        s_1[0U][0U] = fma(av_16[0U], bv_15[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_16[0U], bv_15[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_16[0U], bv_15[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_16[0U], bv_15[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_16[1U];
        }
        s_1[1U][0U] = fma(av_16[1U], bv_15[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_16[1U], bv_15[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_16[1U], bv_15[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_16[1U], bv_15[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_16[2U];
        }
        s_1[2U][0U] = fma(av_16[2U], bv_15[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_16[2U], bv_15[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_16[2U], bv_15[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_16[2U], bv_15[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_16[3U];
        }
        s_1[3U][0U] = fma(av_16[3U], bv_15[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_16[3U], bv_15[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_16[3U], bv_15[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_16[3U], bv_15[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_17;
        av_17[0U] = (*(&kernelContext_8)->qTileA_0)[16U][ty_0];
        av_17[1U] = (*(&kernelContext_8)->qTileA_0)[16U][_S41];
        av_17[2U] = (*(&kernelContext_8)->qTileA_0)[16U][_S42];
        av_17[3U] = (*(&kernelContext_8)->qTileA_0)[16U][_S43];
        thread array<float, int(4)> bv_16;
        bv_16[0U] = (*(&kernelContext_8)->qTileB_0)[16U][tx_0];
        bv_16[1U] = (*(&kernelContext_8)->qTileB_0)[16U][_S44];
        bv_16[2U] = (*(&kernelContext_8)->qTileB_0)[16U][_S45];
        bv_16[3U] = (*(&kernelContext_8)->qTileB_0)[16U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_17[0U];
        }
        s_1[0U][0U] = fma(av_17[0U], bv_16[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_17[0U], bv_16[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_17[0U], bv_16[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_17[0U], bv_16[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_17[1U];
        }
        s_1[1U][0U] = fma(av_17[1U], bv_16[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_17[1U], bv_16[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_17[1U], bv_16[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_17[1U], bv_16[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_17[2U];
        }
        s_1[2U][0U] = fma(av_17[2U], bv_16[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_17[2U], bv_16[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_17[2U], bv_16[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_17[2U], bv_16[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_17[3U];
        }
        s_1[3U][0U] = fma(av_17[3U], bv_16[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_17[3U], bv_16[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_17[3U], bv_16[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_17[3U], bv_16[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_18;
        av_18[0U] = (*(&kernelContext_8)->qTileA_0)[17U][ty_0];
        av_18[1U] = (*(&kernelContext_8)->qTileA_0)[17U][_S41];
        av_18[2U] = (*(&kernelContext_8)->qTileA_0)[17U][_S42];
        av_18[3U] = (*(&kernelContext_8)->qTileA_0)[17U][_S43];
        thread array<float, int(4)> bv_17;
        bv_17[0U] = (*(&kernelContext_8)->qTileB_0)[17U][tx_0];
        bv_17[1U] = (*(&kernelContext_8)->qTileB_0)[17U][_S44];
        bv_17[2U] = (*(&kernelContext_8)->qTileB_0)[17U][_S45];
        bv_17[3U] = (*(&kernelContext_8)->qTileB_0)[17U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_18[0U];
        }
        s_1[0U][0U] = fma(av_18[0U], bv_17[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_18[0U], bv_17[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_18[0U], bv_17[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_18[0U], bv_17[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_18[1U];
        }
        s_1[1U][0U] = fma(av_18[1U], bv_17[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_18[1U], bv_17[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_18[1U], bv_17[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_18[1U], bv_17[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_18[2U];
        }
        s_1[2U][0U] = fma(av_18[2U], bv_17[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_18[2U], bv_17[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_18[2U], bv_17[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_18[2U], bv_17[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_18[3U];
        }
        s_1[3U][0U] = fma(av_18[3U], bv_17[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_18[3U], bv_17[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_18[3U], bv_17[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_18[3U], bv_17[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_19;
        av_19[0U] = (*(&kernelContext_8)->qTileA_0)[18U][ty_0];
        av_19[1U] = (*(&kernelContext_8)->qTileA_0)[18U][_S41];
        av_19[2U] = (*(&kernelContext_8)->qTileA_0)[18U][_S42];
        av_19[3U] = (*(&kernelContext_8)->qTileA_0)[18U][_S43];
        thread array<float, int(4)> bv_18;
        bv_18[0U] = (*(&kernelContext_8)->qTileB_0)[18U][tx_0];
        bv_18[1U] = (*(&kernelContext_8)->qTileB_0)[18U][_S44];
        bv_18[2U] = (*(&kernelContext_8)->qTileB_0)[18U][_S45];
        bv_18[3U] = (*(&kernelContext_8)->qTileB_0)[18U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_19[0U];
        }
        s_1[0U][0U] = fma(av_19[0U], bv_18[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_19[0U], bv_18[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_19[0U], bv_18[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_19[0U], bv_18[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_19[1U];
        }
        s_1[1U][0U] = fma(av_19[1U], bv_18[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_19[1U], bv_18[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_19[1U], bv_18[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_19[1U], bv_18[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_19[2U];
        }
        s_1[2U][0U] = fma(av_19[2U], bv_18[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_19[2U], bv_18[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_19[2U], bv_18[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_19[2U], bv_18[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_19[3U];
        }
        s_1[3U][0U] = fma(av_19[3U], bv_18[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_19[3U], bv_18[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_19[3U], bv_18[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_19[3U], bv_18[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_20;
        av_20[0U] = (*(&kernelContext_8)->qTileA_0)[19U][ty_0];
        av_20[1U] = (*(&kernelContext_8)->qTileA_0)[19U][_S41];
        av_20[2U] = (*(&kernelContext_8)->qTileA_0)[19U][_S42];
        av_20[3U] = (*(&kernelContext_8)->qTileA_0)[19U][_S43];
        thread array<float, int(4)> bv_19;
        bv_19[0U] = (*(&kernelContext_8)->qTileB_0)[19U][tx_0];
        bv_19[1U] = (*(&kernelContext_8)->qTileB_0)[19U][_S44];
        bv_19[2U] = (*(&kernelContext_8)->qTileB_0)[19U][_S45];
        bv_19[3U] = (*(&kernelContext_8)->qTileB_0)[19U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_20[0U];
        }
        s_1[0U][0U] = fma(av_20[0U], bv_19[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_20[0U], bv_19[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_20[0U], bv_19[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_20[0U], bv_19[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_20[1U];
        }
        s_1[1U][0U] = fma(av_20[1U], bv_19[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_20[1U], bv_19[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_20[1U], bv_19[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_20[1U], bv_19[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_20[2U];
        }
        s_1[2U][0U] = fma(av_20[2U], bv_19[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_20[2U], bv_19[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_20[2U], bv_19[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_20[2U], bv_19[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_20[3U];
        }
        s_1[3U][0U] = fma(av_20[3U], bv_19[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_20[3U], bv_19[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_20[3U], bv_19[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_20[3U], bv_19[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_21;
        av_21[0U] = (*(&kernelContext_8)->qTileA_0)[20U][ty_0];
        av_21[1U] = (*(&kernelContext_8)->qTileA_0)[20U][_S41];
        av_21[2U] = (*(&kernelContext_8)->qTileA_0)[20U][_S42];
        av_21[3U] = (*(&kernelContext_8)->qTileA_0)[20U][_S43];
        thread array<float, int(4)> bv_20;
        bv_20[0U] = (*(&kernelContext_8)->qTileB_0)[20U][tx_0];
        bv_20[1U] = (*(&kernelContext_8)->qTileB_0)[20U][_S44];
        bv_20[2U] = (*(&kernelContext_8)->qTileB_0)[20U][_S45];
        bv_20[3U] = (*(&kernelContext_8)->qTileB_0)[20U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_21[0U];
        }
        s_1[0U][0U] = fma(av_21[0U], bv_20[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_21[0U], bv_20[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_21[0U], bv_20[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_21[0U], bv_20[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_21[1U];
        }
        s_1[1U][0U] = fma(av_21[1U], bv_20[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_21[1U], bv_20[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_21[1U], bv_20[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_21[1U], bv_20[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_21[2U];
        }
        s_1[2U][0U] = fma(av_21[2U], bv_20[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_21[2U], bv_20[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_21[2U], bv_20[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_21[2U], bv_20[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_21[3U];
        }
        s_1[3U][0U] = fma(av_21[3U], bv_20[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_21[3U], bv_20[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_21[3U], bv_20[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_21[3U], bv_20[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_22;
        av_22[0U] = (*(&kernelContext_8)->qTileA_0)[21U][ty_0];
        av_22[1U] = (*(&kernelContext_8)->qTileA_0)[21U][_S41];
        av_22[2U] = (*(&kernelContext_8)->qTileA_0)[21U][_S42];
        av_22[3U] = (*(&kernelContext_8)->qTileA_0)[21U][_S43];
        thread array<float, int(4)> bv_21;
        bv_21[0U] = (*(&kernelContext_8)->qTileB_0)[21U][tx_0];
        bv_21[1U] = (*(&kernelContext_8)->qTileB_0)[21U][_S44];
        bv_21[2U] = (*(&kernelContext_8)->qTileB_0)[21U][_S45];
        bv_21[3U] = (*(&kernelContext_8)->qTileB_0)[21U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_22[0U];
        }
        s_1[0U][0U] = fma(av_22[0U], bv_21[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_22[0U], bv_21[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_22[0U], bv_21[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_22[0U], bv_21[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_22[1U];
        }
        s_1[1U][0U] = fma(av_22[1U], bv_21[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_22[1U], bv_21[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_22[1U], bv_21[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_22[1U], bv_21[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_22[2U];
        }
        s_1[2U][0U] = fma(av_22[2U], bv_21[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_22[2U], bv_21[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_22[2U], bv_21[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_22[2U], bv_21[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_22[3U];
        }
        s_1[3U][0U] = fma(av_22[3U], bv_21[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_22[3U], bv_21[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_22[3U], bv_21[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_22[3U], bv_21[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_23;
        av_23[0U] = (*(&kernelContext_8)->qTileA_0)[22U][ty_0];
        av_23[1U] = (*(&kernelContext_8)->qTileA_0)[22U][_S41];
        av_23[2U] = (*(&kernelContext_8)->qTileA_0)[22U][_S42];
        av_23[3U] = (*(&kernelContext_8)->qTileA_0)[22U][_S43];
        thread array<float, int(4)> bv_22;
        bv_22[0U] = (*(&kernelContext_8)->qTileB_0)[22U][tx_0];
        bv_22[1U] = (*(&kernelContext_8)->qTileB_0)[22U][_S44];
        bv_22[2U] = (*(&kernelContext_8)->qTileB_0)[22U][_S45];
        bv_22[3U] = (*(&kernelContext_8)->qTileB_0)[22U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_23[0U];
        }
        s_1[0U][0U] = fma(av_23[0U], bv_22[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_23[0U], bv_22[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_23[0U], bv_22[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_23[0U], bv_22[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_23[1U];
        }
        s_1[1U][0U] = fma(av_23[1U], bv_22[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_23[1U], bv_22[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_23[1U], bv_22[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_23[1U], bv_22[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_23[2U];
        }
        s_1[2U][0U] = fma(av_23[2U], bv_22[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_23[2U], bv_22[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_23[2U], bv_22[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_23[2U], bv_22[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_23[3U];
        }
        s_1[3U][0U] = fma(av_23[3U], bv_22[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_23[3U], bv_22[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_23[3U], bv_22[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_23[3U], bv_22[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_24;
        av_24[0U] = (*(&kernelContext_8)->qTileA_0)[23U][ty_0];
        av_24[1U] = (*(&kernelContext_8)->qTileA_0)[23U][_S41];
        av_24[2U] = (*(&kernelContext_8)->qTileA_0)[23U][_S42];
        av_24[3U] = (*(&kernelContext_8)->qTileA_0)[23U][_S43];
        thread array<float, int(4)> bv_23;
        bv_23[0U] = (*(&kernelContext_8)->qTileB_0)[23U][tx_0];
        bv_23[1U] = (*(&kernelContext_8)->qTileB_0)[23U][_S44];
        bv_23[2U] = (*(&kernelContext_8)->qTileB_0)[23U][_S45];
        bv_23[3U] = (*(&kernelContext_8)->qTileB_0)[23U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_24[0U];
        }
        s_1[0U][0U] = fma(av_24[0U], bv_23[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_24[0U], bv_23[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_24[0U], bv_23[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_24[0U], bv_23[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_24[1U];
        }
        s_1[1U][0U] = fma(av_24[1U], bv_23[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_24[1U], bv_23[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_24[1U], bv_23[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_24[1U], bv_23[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_24[2U];
        }
        s_1[2U][0U] = fma(av_24[2U], bv_23[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_24[2U], bv_23[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_24[2U], bv_23[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_24[2U], bv_23[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_24[3U];
        }
        s_1[3U][0U] = fma(av_24[3U], bv_23[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_24[3U], bv_23[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_24[3U], bv_23[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_24[3U], bv_23[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_25;
        av_25[0U] = (*(&kernelContext_8)->qTileA_0)[24U][ty_0];
        av_25[1U] = (*(&kernelContext_8)->qTileA_0)[24U][_S41];
        av_25[2U] = (*(&kernelContext_8)->qTileA_0)[24U][_S42];
        av_25[3U] = (*(&kernelContext_8)->qTileA_0)[24U][_S43];
        thread array<float, int(4)> bv_24;
        bv_24[0U] = (*(&kernelContext_8)->qTileB_0)[24U][tx_0];
        bv_24[1U] = (*(&kernelContext_8)->qTileB_0)[24U][_S44];
        bv_24[2U] = (*(&kernelContext_8)->qTileB_0)[24U][_S45];
        bv_24[3U] = (*(&kernelContext_8)->qTileB_0)[24U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_25[0U];
        }
        s_1[0U][0U] = fma(av_25[0U], bv_24[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_25[0U], bv_24[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_25[0U], bv_24[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_25[0U], bv_24[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_25[1U];
        }
        s_1[1U][0U] = fma(av_25[1U], bv_24[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_25[1U], bv_24[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_25[1U], bv_24[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_25[1U], bv_24[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_25[2U];
        }
        s_1[2U][0U] = fma(av_25[2U], bv_24[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_25[2U], bv_24[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_25[2U], bv_24[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_25[2U], bv_24[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_25[3U];
        }
        s_1[3U][0U] = fma(av_25[3U], bv_24[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_25[3U], bv_24[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_25[3U], bv_24[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_25[3U], bv_24[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_26;
        av_26[0U] = (*(&kernelContext_8)->qTileA_0)[25U][ty_0];
        av_26[1U] = (*(&kernelContext_8)->qTileA_0)[25U][_S41];
        av_26[2U] = (*(&kernelContext_8)->qTileA_0)[25U][_S42];
        av_26[3U] = (*(&kernelContext_8)->qTileA_0)[25U][_S43];
        thread array<float, int(4)> bv_25;
        bv_25[0U] = (*(&kernelContext_8)->qTileB_0)[25U][tx_0];
        bv_25[1U] = (*(&kernelContext_8)->qTileB_0)[25U][_S44];
        bv_25[2U] = (*(&kernelContext_8)->qTileB_0)[25U][_S45];
        bv_25[3U] = (*(&kernelContext_8)->qTileB_0)[25U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_26[0U];
        }
        s_1[0U][0U] = fma(av_26[0U], bv_25[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_26[0U], bv_25[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_26[0U], bv_25[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_26[0U], bv_25[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_26[1U];
        }
        s_1[1U][0U] = fma(av_26[1U], bv_25[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_26[1U], bv_25[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_26[1U], bv_25[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_26[1U], bv_25[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_26[2U];
        }
        s_1[2U][0U] = fma(av_26[2U], bv_25[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_26[2U], bv_25[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_26[2U], bv_25[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_26[2U], bv_25[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_26[3U];
        }
        s_1[3U][0U] = fma(av_26[3U], bv_25[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_26[3U], bv_25[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_26[3U], bv_25[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_26[3U], bv_25[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_27;
        av_27[0U] = (*(&kernelContext_8)->qTileA_0)[26U][ty_0];
        av_27[1U] = (*(&kernelContext_8)->qTileA_0)[26U][_S41];
        av_27[2U] = (*(&kernelContext_8)->qTileA_0)[26U][_S42];
        av_27[3U] = (*(&kernelContext_8)->qTileA_0)[26U][_S43];
        thread array<float, int(4)> bv_26;
        bv_26[0U] = (*(&kernelContext_8)->qTileB_0)[26U][tx_0];
        bv_26[1U] = (*(&kernelContext_8)->qTileB_0)[26U][_S44];
        bv_26[2U] = (*(&kernelContext_8)->qTileB_0)[26U][_S45];
        bv_26[3U] = (*(&kernelContext_8)->qTileB_0)[26U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_27[0U];
        }
        s_1[0U][0U] = fma(av_27[0U], bv_26[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_27[0U], bv_26[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_27[0U], bv_26[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_27[0U], bv_26[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_27[1U];
        }
        s_1[1U][0U] = fma(av_27[1U], bv_26[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_27[1U], bv_26[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_27[1U], bv_26[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_27[1U], bv_26[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_27[2U];
        }
        s_1[2U][0U] = fma(av_27[2U], bv_26[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_27[2U], bv_26[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_27[2U], bv_26[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_27[2U], bv_26[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_27[3U];
        }
        s_1[3U][0U] = fma(av_27[3U], bv_26[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_27[3U], bv_26[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_27[3U], bv_26[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_27[3U], bv_26[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_28;
        av_28[0U] = (*(&kernelContext_8)->qTileA_0)[27U][ty_0];
        av_28[1U] = (*(&kernelContext_8)->qTileA_0)[27U][_S41];
        av_28[2U] = (*(&kernelContext_8)->qTileA_0)[27U][_S42];
        av_28[3U] = (*(&kernelContext_8)->qTileA_0)[27U][_S43];
        thread array<float, int(4)> bv_27;
        bv_27[0U] = (*(&kernelContext_8)->qTileB_0)[27U][tx_0];
        bv_27[1U] = (*(&kernelContext_8)->qTileB_0)[27U][_S44];
        bv_27[2U] = (*(&kernelContext_8)->qTileB_0)[27U][_S45];
        bv_27[3U] = (*(&kernelContext_8)->qTileB_0)[27U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_28[0U];
        }
        s_1[0U][0U] = fma(av_28[0U], bv_27[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_28[0U], bv_27[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_28[0U], bv_27[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_28[0U], bv_27[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_28[1U];
        }
        s_1[1U][0U] = fma(av_28[1U], bv_27[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_28[1U], bv_27[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_28[1U], bv_27[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_28[1U], bv_27[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_28[2U];
        }
        s_1[2U][0U] = fma(av_28[2U], bv_27[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_28[2U], bv_27[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_28[2U], bv_27[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_28[2U], bv_27[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_28[3U];
        }
        s_1[3U][0U] = fma(av_28[3U], bv_27[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_28[3U], bv_27[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_28[3U], bv_27[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_28[3U], bv_27[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_29;
        av_29[0U] = (*(&kernelContext_8)->qTileA_0)[28U][ty_0];
        av_29[1U] = (*(&kernelContext_8)->qTileA_0)[28U][_S41];
        av_29[2U] = (*(&kernelContext_8)->qTileA_0)[28U][_S42];
        av_29[3U] = (*(&kernelContext_8)->qTileA_0)[28U][_S43];
        thread array<float, int(4)> bv_28;
        bv_28[0U] = (*(&kernelContext_8)->qTileB_0)[28U][tx_0];
        bv_28[1U] = (*(&kernelContext_8)->qTileB_0)[28U][_S44];
        bv_28[2U] = (*(&kernelContext_8)->qTileB_0)[28U][_S45];
        bv_28[3U] = (*(&kernelContext_8)->qTileB_0)[28U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_29[0U];
        }
        s_1[0U][0U] = fma(av_29[0U], bv_28[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_29[0U], bv_28[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_29[0U], bv_28[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_29[0U], bv_28[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_29[1U];
        }
        s_1[1U][0U] = fma(av_29[1U], bv_28[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_29[1U], bv_28[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_29[1U], bv_28[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_29[1U], bv_28[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_29[2U];
        }
        s_1[2U][0U] = fma(av_29[2U], bv_28[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_29[2U], bv_28[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_29[2U], bv_28[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_29[2U], bv_28[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_29[3U];
        }
        s_1[3U][0U] = fma(av_29[3U], bv_28[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_29[3U], bv_28[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_29[3U], bv_28[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_29[3U], bv_28[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_30;
        av_30[0U] = (*(&kernelContext_8)->qTileA_0)[29U][ty_0];
        av_30[1U] = (*(&kernelContext_8)->qTileA_0)[29U][_S41];
        av_30[2U] = (*(&kernelContext_8)->qTileA_0)[29U][_S42];
        av_30[3U] = (*(&kernelContext_8)->qTileA_0)[29U][_S43];
        thread array<float, int(4)> bv_29;
        bv_29[0U] = (*(&kernelContext_8)->qTileB_0)[29U][tx_0];
        bv_29[1U] = (*(&kernelContext_8)->qTileB_0)[29U][_S44];
        bv_29[2U] = (*(&kernelContext_8)->qTileB_0)[29U][_S45];
        bv_29[3U] = (*(&kernelContext_8)->qTileB_0)[29U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_30[0U];
        }
        s_1[0U][0U] = fma(av_30[0U], bv_29[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_30[0U], bv_29[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_30[0U], bv_29[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_30[0U], bv_29[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_30[1U];
        }
        s_1[1U][0U] = fma(av_30[1U], bv_29[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_30[1U], bv_29[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_30[1U], bv_29[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_30[1U], bv_29[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_30[2U];
        }
        s_1[2U][0U] = fma(av_30[2U], bv_29[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_30[2U], bv_29[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_30[2U], bv_29[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_30[2U], bv_29[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_30[3U];
        }
        s_1[3U][0U] = fma(av_30[3U], bv_29[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_30[3U], bv_29[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_30[3U], bv_29[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_30[3U], bv_29[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_31;
        av_31[0U] = (*(&kernelContext_8)->qTileA_0)[30U][ty_0];
        av_31[1U] = (*(&kernelContext_8)->qTileA_0)[30U][_S41];
        av_31[2U] = (*(&kernelContext_8)->qTileA_0)[30U][_S42];
        av_31[3U] = (*(&kernelContext_8)->qTileA_0)[30U][_S43];
        thread array<float, int(4)> bv_30;
        bv_30[0U] = (*(&kernelContext_8)->qTileB_0)[30U][tx_0];
        bv_30[1U] = (*(&kernelContext_8)->qTileB_0)[30U][_S44];
        bv_30[2U] = (*(&kernelContext_8)->qTileB_0)[30U][_S45];
        bv_30[3U] = (*(&kernelContext_8)->qTileB_0)[30U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_31[0U];
        }
        s_1[0U][0U] = fma(av_31[0U], bv_30[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_31[0U], bv_30[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_31[0U], bv_30[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_31[0U], bv_30[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_31[1U];
        }
        s_1[1U][0U] = fma(av_31[1U], bv_30[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_31[1U], bv_30[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_31[1U], bv_30[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_31[1U], bv_30[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_31[2U];
        }
        s_1[2U][0U] = fma(av_31[2U], bv_30[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_31[2U], bv_30[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_31[2U], bv_30[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_31[2U], bv_30[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_31[3U];
        }
        s_1[3U][0U] = fma(av_31[3U], bv_30[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_31[3U], bv_30[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_31[3U], bv_30[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_31[3U], bv_30[3U], s_1[3U][3U]);
        thread array<float, int(4)> av_32;
        av_32[0U] = (*(&kernelContext_8)->qTileA_0)[31U][ty_0];
        av_32[1U] = (*(&kernelContext_8)->qTileA_0)[31U][_S41];
        av_32[2U] = (*(&kernelContext_8)->qTileA_0)[31U][_S42];
        av_32[3U] = (*(&kernelContext_8)->qTileA_0)[31U][_S43];
        thread array<float, int(4)> bv_31;
        bv_31[0U] = (*(&kernelContext_8)->qTileB_0)[31U][tx_0];
        bv_31[1U] = (*(&kernelContext_8)->qTileB_0)[31U][_S44];
        bv_31[2U] = (*(&kernelContext_8)->qTileB_0)[31U][_S45];
        bv_31[3U] = (*(&kernelContext_8)->qTileB_0)[31U][_S46];
        if(QFORMAT_0 == 2U)
        {
            sumA_1[0U] = sumA_1[0U] + av_32[0U];
        }
        s_1[0U][0U] = fma(av_32[0U], bv_31[0U], s_1[0U][0U]);
        s_1[0U][1U] = fma(av_32[0U], bv_31[1U], s_1[0U][1U]);
        s_1[0U][2U] = fma(av_32[0U], bv_31[2U], s_1[0U][2U]);
        s_1[0U][3U] = fma(av_32[0U], bv_31[3U], s_1[0U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[1U] = sumA_1[1U] + av_32[1U];
        }
        s_1[1U][0U] = fma(av_32[1U], bv_31[0U], s_1[1U][0U]);
        s_1[1U][1U] = fma(av_32[1U], bv_31[1U], s_1[1U][1U]);
        s_1[1U][2U] = fma(av_32[1U], bv_31[2U], s_1[1U][2U]);
        s_1[1U][3U] = fma(av_32[1U], bv_31[3U], s_1[1U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[2U] = sumA_1[2U] + av_32[2U];
        }
        s_1[2U][0U] = fma(av_32[2U], bv_31[0U], s_1[2U][0U]);
        s_1[2U][1U] = fma(av_32[2U], bv_31[1U], s_1[2U][1U]);
        s_1[2U][2U] = fma(av_32[2U], bv_31[2U], s_1[2U][2U]);
        s_1[2U][3U] = fma(av_32[2U], bv_31[3U], s_1[2U][3U]);
        if(QFORMAT_0 == 2U)
        {
            sumA_1[3U] = sumA_1[3U] + av_32[3U];
        }
        s_1[3U][0U] = fma(av_32[3U], bv_31[0U], s_1[3U][0U]);
        s_1[3U][1U] = fma(av_32[3U], bv_31[1U], s_1[3U][1U]);
        s_1[3U][2U] = fma(av_32[3U], bv_31[2U], s_1[3U][2U]);
        s_1[3U][3U] = fma(av_32[3U], bv_31[3U], s_1[3U][3U]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S47 = k0_0 + 32U;
        uint _S48 = _S47 % QBLOCK_0;
        if(_S48 == 0U)
        {
            uint _S49 = k0_0 / QBLOCK_0;
            uint _S50 = _S33 + tx_0;
            uint _S51 = min(_S50, (&kernelContext_8)->p_0->n_1 - 1U);
            float _S52 = blockParameter_0(_S51, _S49, 0U, &kernelContext_8);
            float _S53;
            if(QFORMAT_0 == 2U)
            {
                float _S54 = blockParameter_0(_S51, _S49, 1U, &kernelContext_8);
                _S53 = _S54;
            }
            else
            {
                _S53 = 0.0;
            }
            acc_1[0U][0U] = acc_1[0U][0U] + blockTerm_0(s_1[0U][0U], sumA_1[0U], _S52, _S53);
            s_1[0U][0U] = 0.0;
            acc_1[1U][0U] = acc_1[1U][0U] + blockTerm_0(s_1[1U][0U], sumA_1[1U], _S52, _S53);
            s_1[1U][0U] = 0.0;
            acc_1[2U][0U] = acc_1[2U][0U] + blockTerm_0(s_1[2U][0U], sumA_1[2U], _S52, _S53);
            s_1[2U][0U] = 0.0;
            acc_1[3U][0U] = acc_1[3U][0U] + blockTerm_0(s_1[3U][0U], sumA_1[3U], _S52, _S53);
            s_1[3U][0U] = 0.0;
            uint _S55 = min(_S50 + 16U, (&kernelContext_8)->p_0->n_1 - 1U);
            float _S56 = blockParameter_0(_S55, _S49, 0U, &kernelContext_8);
            float _S57;
            if(QFORMAT_0 == 2U)
            {
                float _S58 = blockParameter_0(_S55, _S49, 1U, &kernelContext_8);
                _S57 = _S58;
            }
            else
            {
                _S57 = 0.0;
            }
            acc_1[0U][1U] = acc_1[0U][1U] + blockTerm_0(s_1[0U][1U], sumA_1[0U], _S56, _S57);
            s_1[0U][1U] = 0.0;
            acc_1[1U][1U] = acc_1[1U][1U] + blockTerm_0(s_1[1U][1U], sumA_1[1U], _S56, _S57);
            s_1[1U][1U] = 0.0;
            acc_1[2U][1U] = acc_1[2U][1U] + blockTerm_0(s_1[2U][1U], sumA_1[2U], _S56, _S57);
            s_1[2U][1U] = 0.0;
            acc_1[3U][1U] = acc_1[3U][1U] + blockTerm_0(s_1[3U][1U], sumA_1[3U], _S56, _S57);
            s_1[3U][1U] = 0.0;
            uint _S59 = min(_S50 + 32U, (&kernelContext_8)->p_0->n_1 - 1U);
            float _S60 = blockParameter_0(_S59, _S49, 0U, &kernelContext_8);
            float _S61;
            if(QFORMAT_0 == 2U)
            {
                float _S62 = blockParameter_0(_S59, _S49, 1U, &kernelContext_8);
                _S61 = _S62;
            }
            else
            {
                _S61 = 0.0;
            }
            acc_1[0U][2U] = acc_1[0U][2U] + blockTerm_0(s_1[0U][2U], sumA_1[0U], _S60, _S61);
            s_1[0U][2U] = 0.0;
            acc_1[1U][2U] = acc_1[1U][2U] + blockTerm_0(s_1[1U][2U], sumA_1[1U], _S60, _S61);
            s_1[1U][2U] = 0.0;
            acc_1[2U][2U] = acc_1[2U][2U] + blockTerm_0(s_1[2U][2U], sumA_1[2U], _S60, _S61);
            s_1[2U][2U] = 0.0;
            acc_1[3U][2U] = acc_1[3U][2U] + blockTerm_0(s_1[3U][2U], sumA_1[3U], _S60, _S61);
            s_1[3U][2U] = 0.0;
            uint _S63 = min(_S50 + 48U, (&kernelContext_8)->p_0->n_1 - 1U);
            float _S64 = blockParameter_0(_S63, _S49, 0U, &kernelContext_8);
            float _S65;
            if(QFORMAT_0 == 2U)
            {
                float _S66 = blockParameter_0(_S63, _S49, 1U, &kernelContext_8);
                _S65 = _S66;
            }
            else
            {
                _S65 = 0.0;
            }
            acc_1[0U][3U] = acc_1[0U][3U] + blockTerm_0(s_1[0U][3U], sumA_1[0U], _S64, _S65);
            s_1[0U][3U] = 0.0;
            acc_1[1U][3U] = acc_1[1U][3U] + blockTerm_0(s_1[1U][3U], sumA_1[1U], _S64, _S65);
            s_1[1U][3U] = 0.0;
            acc_1[2U][3U] = acc_1[2U][3U] + blockTerm_0(s_1[2U][3U], sumA_1[2U], _S64, _S65);
            s_1[2U][3U] = 0.0;
            acc_1[3U][3U] = acc_1[3U][3U] + blockTerm_0(s_1[3U][3U], sumA_1[3U], _S64, _S65);
            s_1[3U][3U] = 0.0;
            sumA_1[0U] = 0.0;
            sumA_1[1U] = 0.0;
            sumA_1[2U] = 0.0;
            sumA_1[3U] = 0.0;
        }
        k0_0 = _S47;
    }
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                uint _S67 = _S32 + ty_0;
                _S29 = _S67;
                if(_S67 >= ((&kernelContext_8)->p_0->m_1))
                {
                    break;
                }
                uint _S68 = _S33 + tx_0;
                if(_S68 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S69 = _S30;
                    gemmStore_0(&_S69, _S67, _S68, acc_1[0U][0U], &kernelContext_8);
                }
                uint col_1 = _S68 + 16U;
                if(col_1 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S70 = _S30;
                    gemmStore_0(&_S70, _S67, col_1, acc_1[0U][1U], &kernelContext_8);
                }
                uint col_2 = _S68 + 32U;
                if(col_2 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S71 = _S30;
                    gemmStore_0(&_S71, _S67, col_2, acc_1[0U][2U], &kernelContext_8);
                }
                uint col_3 = _S68 + 48U;
                if(col_3 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S72 = _S30;
                    gemmStore_0(&_S72, _S67, col_3, acc_1[0U][3U], &kernelContext_8);
                }
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint r_4 = _S29 + 16U;
                if(r_4 >= ((&kernelContext_8)->p_0->m_1))
                {
                    break;
                }
                uint _S73 = _S33 + tx_0;
                if(_S73 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S74 = _S30;
                    gemmStore_0(&_S74, r_4, _S73, acc_1[1U][0U], &kernelContext_8);
                }
                uint col_4 = _S73 + 16U;
                if(col_4 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S75 = _S30;
                    gemmStore_0(&_S75, r_4, col_4, acc_1[1U][1U], &kernelContext_8);
                }
                uint col_5 = _S73 + 32U;
                if(col_5 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S76 = _S30;
                    gemmStore_0(&_S76, r_4, col_5, acc_1[1U][2U], &kernelContext_8);
                }
                uint col_6 = _S73 + 48U;
                if(col_6 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S77 = _S30;
                    gemmStore_0(&_S77, r_4, col_6, acc_1[1U][3U], &kernelContext_8);
                }
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint r_5 = _S29 + 32U;
                if(r_5 >= ((&kernelContext_8)->p_0->m_1))
                {
                    break;
                }
                uint _S78 = _S33 + tx_0;
                if(_S78 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S79 = _S30;
                    gemmStore_0(&_S79, r_5, _S78, acc_1[2U][0U], &kernelContext_8);
                }
                uint col_7 = _S78 + 16U;
                if(col_7 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S80 = _S30;
                    gemmStore_0(&_S80, r_5, col_7, acc_1[2U][1U], &kernelContext_8);
                }
                uint col_8 = _S78 + 32U;
                if(col_8 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S81 = _S30;
                    gemmStore_0(&_S81, r_5, col_8, acc_1[2U][2U], &kernelContext_8);
                }
                uint col_9 = _S78 + 48U;
                if(col_9 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S82 = _S30;
                    gemmStore_0(&_S82, r_5, col_9, acc_1[2U][3U], &kernelContext_8);
                }
                break;
            }
            break;
        }
        for(;;)
        {
            for(;;)
            {
                uint r_6 = _S29 + 48U;
                if(r_6 >= ((&kernelContext_8)->p_0->m_1))
                {
                    break;
                }
                uint _S83 = _S33 + tx_0;
                if(_S83 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S84 = _S30;
                    gemmStore_0(&_S84, r_6, _S83, acc_1[3U][0U], &kernelContext_8);
                }
                uint col_10 = _S83 + 16U;
                if(col_10 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S85 = _S30;
                    gemmStore_0(&_S85, r_6, col_10, acc_1[3U][1U], &kernelContext_8);
                }
                uint col_11 = _S83 + 32U;
                if(col_11 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S86 = _S30;
                    gemmStore_0(&_S86, r_6, col_11, acc_1[3U][2U], &kernelContext_8);
                }
                uint col_12 = _S83 + 48U;
                if(col_12 < ((&kernelContext_8)->p_0->n_1))
                {
                    thread GemmProblem_0 _S87 = _S30;
                    gemmStore_0(&_S87, r_6, col_12, acc_1[3U][3U], &kernelContext_8);
                }
                break;
            }
            break;
        }
        break;
    }
    return;
}

