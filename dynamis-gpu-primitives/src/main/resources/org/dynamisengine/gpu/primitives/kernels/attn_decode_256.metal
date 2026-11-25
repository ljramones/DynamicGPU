#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
float attnLo_0(uint w_0)
{
    return (as_type<half>((ushort)((w_0 & 65535U))));
}

float attnHi_0(uint w_1)
{
    return (as_type<half>((ushort)((w_1 >> 16U))));
}

float dg_negative_infinity_0()
{
    return (as_type<float>((4286578688U)));
}

bool dg_is_nan_0(float x_0)
{
    uint _S1 = (as_type<uint>((x_0)));
    bool _S2;
    if((_S1 & 2139095040U) == 2139095040U)
    {
        _S2 = (_S1 & 8388607U) != 0U;
    }
    else
    {
        _S2 = false;
    }
    return _S2;
}

float dg_max_0(float a_0, float b_0)
{
    if(dg_is_nan_0(a_0))
    {
        return a_0;
    }
    if(dg_is_nan_0(b_0))
    {
        return b_0;
    }
    bool _S3;
    if(a_0 == 0.0)
    {
        _S3 = b_0 == 0.0;
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        return (as_type<float>(((as_type<uint>((a_0))) & (as_type<uint>((b_0))))));
    }
    float _S4;
    if(a_0 > b_0)
    {
        _S4 = a_0;
    }
    else
    {
        _S4 = b_0;
    }
    return _S4;
}

float dg_op_max_0(float a_1, float b_1)
{
    return dg_max_0(a_1, b_1);
}

float dg_op_add_0(float a_2, float b_2)
{
    return a_2 + b_2;
}

uint qF32ToF16_0(float f_0)
{
    uint x_1 = (as_type<uint>((f_0)));
    uint sign_0 = (x_1 >> 16U) & 32768U;
    uint e_0 = (x_1 >> 23U) & 255U;
    uint mant_0 = x_1 & 8388607U;
    uint h_0;
    if(e_0 == 255U)
    {
        uint _S5 = sign_0 | 31744U;
        if(mant_0 != 0U)
        {
            h_0 = 512U | (mant_0 >> 13U);
        }
        else
        {
            h_0 = 0U;
        }
        return _S5 | h_0;
    }
    int exp_0 = int(e_0) - int(112);
    if(exp_0 >= int(31))
    {
        return sign_0 | 31744U;
    }
    bool _S6;
    if(exp_0 <= int(0))
    {
        if(exp_0 < int(-10))
        {
            return sign_0;
        }
        uint mant_1 = mant_0 | 8388608U;
        uint shift_0 = uint(int(14) - exp_0);
        uint h_1 = mant_1 >> shift_0;
        uint rem_0 = mant_1 & ((1U << shift_0) - 1U);
        uint halfway_0 = 1U << (shift_0 - 1U);
        if(rem_0 > halfway_0)
        {
            _S6 = true;
        }
        else
        {
            if(rem_0 == halfway_0)
            {
                _S6 = (h_1 & 1U) != 0U;
            }
            else
            {
                _S6 = false;
            }
        }
        if(_S6)
        {
            h_0 = h_1 + 1U;
        }
        else
        {
            h_0 = h_1;
        }
        return sign_0 | h_0;
    }
    uint h_2 = (uint(exp_0) << 10U) | (mant_0 >> 13U);
    uint rem_1 = mant_0 & 8191U;
    if(rem_1 > 4096U)
    {
        _S6 = true;
    }
    else
    {
        if(rem_1 == 4096U)
        {
            _S6 = (h_2 & 1U) != 0U;
        }
        else
        {
            _S6 = false;
        }
    }
    if(_S6)
    {
        h_0 = h_2 + 1U;
    }
    else
    {
        h_0 = h_2;
    }
    return sign_0 | h_0;
}

uint attnPack_0(float lo_0, float hi_0)
{
    return (qF32ToF16_0(lo_0)) | ((qF32ToF16_0(hi_0)) << 16U);
}

struct AttnParams_0
{
    uint heads_0;
    uint kvHeads_0;
    uint queries_0;
    uint keys_0;
    uint headDim_0;
    uint cacheKeys_0;
    float scale_0;
    uint causal_0;
    uint splits_0;
    uint splitKeys_0;
    uint rowBlocks_0;
    uint pageSize_0;
};

struct KernelContext_0
{
    AttnParams_0 constant* p_0;
    uint device* q_0;
    uint device* table_0;
    uint device* k_0;
    uint device* v_0;
    float device* partialML_0;
    float device* partialAcc_0;
    uint device* o_0;
    array<uint, int(1024)> threadgroup* qs_0;
    array<float, int(1024)> threadgroup* ss_0;
    array<float, int(128)> threadgroup* red_0;
    array<float, int(8)> threadgroup* rowM_0;
    array<float, int(8)> threadgroup* rowL_0;
};

[[kernel]] void attn_decode_256(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], AttnParams_0 constant* p_1 [[buffer(7)]], uint device* q_1 [[buffer(0)]], uint device* table_1 [[buffer(6)]], uint device* k_1 [[buffer(1)]], uint device* v_1 [[buffer(2)]], float device* partialML_1 [[buffer(4)]], float device* partialAcc_1 [[buffer(5)]], uint device* o_1 [[buffer(3)]])
{
    bool _S7;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->q_0 = q_1;
    (&kernelContext_0)->table_0 = table_1;
    (&kernelContext_0)->k_0 = k_1;
    (&kernelContext_0)->v_0 = v_1;
    (&kernelContext_0)->partialML_0 = partialML_1;
    (&kernelContext_0)->partialAcc_0 = partialAcc_1;
    (&kernelContext_0)->o_0 = o_1;
    threadgroup array<uint, int(1024)> qs_1;
    (&kernelContext_0)->qs_0 = &qs_1;
    threadgroup array<float, int(1024)> ss_1;
    (&kernelContext_0)->ss_0 = &ss_1;
    threadgroup array<float, int(128)> red_1;
    (&kernelContext_0)->red_0 = &red_1;
    threadgroup array<float, int(8)> rowM_1;
    (&kernelContext_0)->rowM_0 = &rowM_1;
    threadgroup array<float, int(8)> rowL_1;
    (&kernelContext_0)->rowL_0 = &rowL_1;
    uint t_0 = local_0.x;
    uint _S8 = group_0.y;
    uint split_0 = group_0.z;
    uint hpk_0 = p_1->heads_0 / p_1->kvHeads_0;
    uint _S9 = hpk_0 * (&kernelContext_0)->p_0->queries_0;
    uint d2_0 = (&kernelContext_0)->p_0->headDim_0 / 2U;
    uint _S10 = (&kernelContext_0)->p_0->keys_0 - (&kernelContext_0)->p_0->queries_0;
    uint kStart_0 = split_0 * (&kernelContext_0)->p_0->splitKeys_0;
    uint _S11 = min((&kernelContext_0)->p_0->keys_0, kStart_0 + (&kernelContext_0)->p_0->splitKeys_0);
    uint _S12;
    if(_S11 > kStart_0)
    {
        _S12 = _S11 - kStart_0;
    }
    else
    {
        _S12 = 0U;
    }
    bool visible_0;
    uint w_2;
    uint _S13 = (&kernelContext_0)->p_0->splitKeys_0;
    uint e_1 = t_0;
    for(;;)
    {
        if(e_1 < 1024U)
        {
        }
        else
        {
            break;
        }
        uint r_0 = e_1 / 128U;
        uint pr_0 = e_1 % 128U;
        if(r_0 < _S9)
        {
            visible_0 = pr_0 < d2_0;
        }
        else
        {
            visible_0 = false;
        }
        if(visible_0)
        {
            uint _S14 = _S8 * hpk_0;
            uint _S15 = r_0 / (&kernelContext_0)->p_0->queries_0;
            uint h_3 = _S14 + _S15;
            uint i_0 = r_0 % (&kernelContext_0)->p_0->queries_0;
            w_2 = *((&kernelContext_0)->q_0+((h_3 * (&kernelContext_0)->p_0->queries_0 + i_0) * d2_0 + pr_0));
        }
        else
        {
            w_2 = 0U;
        }
        (*(&kernelContext_0)->qs_0)[e_1] = w_2;
        e_1 = e_1 + 128U;
    }
    float acc_0;
    int _S16;
    int _S17;
    int _S18;
    int _S19;
    int _S20;
    int _S21;
    int _S22;
    int _S23;
    int _S24;
    int _S25;
    int _S26;
    int _S27;
    int _S28;
    int _S29;
    int _S30;
    int _S31;
    uint kw_0;
    uint kw_1;
    uint kw_2;
    uint kw_3;
    uint kw_4;
    uint kw_5;
    uint p0_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    w_2 = t_0;
    for(;;)
    {
        if(w_2 < _S12)
        {
        }
        else
        {
            break;
        }
        uint key_0 = kStart_0 + w_2;
        if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
        {
            e_1 = (_S8 * (&kernelContext_0)->p_0->cacheKeys_0 + key_0) * d2_0;
        }
        else
        {
            uint _S32 = key_0 / (&kernelContext_0)->p_0->pageSize_0;
            uint _S33 = (*((&kernelContext_0)->table_0+_S32) * p_1->kvHeads_0 + _S8) * (&kernelContext_0)->p_0->pageSize_0;
            uint _S34 = key_0 % (&kernelContext_0)->p_0->pageSize_0;
            e_1 = (_S33 + _S34) * d2_0;
        }
        thread array<float, int(8)> s_0;
        s_0[0U] = 0.0;
        s_0[1U] = 0.0;
        s_0[2U] = 0.0;
        s_0[3U] = 0.0;
        s_0[4U] = 0.0;
        s_0[5U] = 0.0;
        s_0[6U] = 0.0;
        s_0[7U] = 0.0;
        p0_0 = 0U;
        for(;;)
        {
            if(p0_0 < 128U)
            {
            }
            else
            {
                break;
            }
            if(p0_0 < d2_0)
            {
                kw_5 = *((&kernelContext_0)->k_0+(e_1 + p0_0));
            }
            else
            {
                kw_5 = 0U;
            }
            float _S35 = attnLo_0(kw_5);
            float _S36 = attnHi_0(kw_5);
            s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[p0_0]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[p0_0]), _S35, s_0[0U]));
            s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + p0_0]), _S35, s_0[1U]));
            s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + p0_0]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + p0_0]), _S35, s_0[2U]));
            s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + p0_0]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + p0_0]), _S35, s_0[3U]));
            s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + p0_0]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + p0_0]), _S35, s_0[4U]));
            s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + p0_0]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + p0_0]), _S35, s_0[5U]));
            s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + p0_0]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + p0_0]), _S35, s_0[6U]));
            s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + p0_0]), _S36, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + p0_0]), _S35, s_0[7U]));
            uint pr_1 = p0_0 + 1U;
            if(pr_1 < d2_0)
            {
                kw_4 = *((&kernelContext_0)->k_0+(e_1 + pr_1));
            }
            else
            {
                kw_4 = 0U;
            }
            float _S37 = attnLo_0(kw_4);
            float _S38 = attnHi_0(kw_4);
            s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_1]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_1]), _S37, s_0[0U]));
            s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_1]), _S37, s_0[1U]));
            s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_1]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_1]), _S37, s_0[2U]));
            s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_1]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_1]), _S37, s_0[3U]));
            s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_1]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_1]), _S37, s_0[4U]));
            s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_1]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_1]), _S37, s_0[5U]));
            s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_1]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_1]), _S37, s_0[6U]));
            s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_1]), _S38, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_1]), _S37, s_0[7U]));
            uint pr_2 = p0_0 + 2U;
            if(pr_2 < d2_0)
            {
                kw_3 = *((&kernelContext_0)->k_0+(e_1 + pr_2));
            }
            else
            {
                kw_3 = 0U;
            }
            float _S39 = attnLo_0(kw_3);
            float _S40 = attnHi_0(kw_3);
            s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_2]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_2]), _S39, s_0[0U]));
            s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_2]), _S39, s_0[1U]));
            s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_2]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_2]), _S39, s_0[2U]));
            s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_2]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_2]), _S39, s_0[3U]));
            s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_2]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_2]), _S39, s_0[4U]));
            s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_2]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_2]), _S39, s_0[5U]));
            s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_2]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_2]), _S39, s_0[6U]));
            s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_2]), _S40, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_2]), _S39, s_0[7U]));
            uint pr_3 = p0_0 + 3U;
            if(pr_3 < d2_0)
            {
                kw_2 = *((&kernelContext_0)->k_0+(e_1 + pr_3));
            }
            else
            {
                kw_2 = 0U;
            }
            float _S41 = attnLo_0(kw_2);
            float _S42 = attnHi_0(kw_2);
            s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_3]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_3]), _S41, s_0[0U]));
            s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_3]), _S41, s_0[1U]));
            s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_3]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_3]), _S41, s_0[2U]));
            s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_3]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_3]), _S41, s_0[3U]));
            s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_3]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_3]), _S41, s_0[4U]));
            s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_3]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_3]), _S41, s_0[5U]));
            s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_3]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_3]), _S41, s_0[6U]));
            s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_3]), _S42, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_3]), _S41, s_0[7U]));
            uint pr_4 = p0_0 + 4U;
            if(pr_4 < d2_0)
            {
                kw_1 = *((&kernelContext_0)->k_0+(e_1 + pr_4));
            }
            else
            {
                kw_1 = 0U;
            }
            float _S43 = attnLo_0(kw_1);
            float _S44 = attnHi_0(kw_1);
            s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_4]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_4]), _S43, s_0[0U]));
            s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_4]), _S43, s_0[1U]));
            s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_4]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_4]), _S43, s_0[2U]));
            s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_4]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_4]), _S43, s_0[3U]));
            s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_4]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_4]), _S43, s_0[4U]));
            s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_4]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_4]), _S43, s_0[5U]));
            s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_4]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_4]), _S43, s_0[6U]));
            s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_4]), _S44, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_4]), _S43, s_0[7U]));
            uint pr_5 = p0_0 + 5U;
            if(pr_5 < d2_0)
            {
                kw_0 = *((&kernelContext_0)->k_0+(e_1 + pr_5));
            }
            else
            {
                kw_0 = 0U;
            }
            float _S45 = attnLo_0(kw_0);
            float _S46 = attnHi_0(kw_0);
            s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_5]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_5]), _S45, s_0[0U]));
            s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_5]), _S45, s_0[1U]));
            s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_5]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_5]), _S45, s_0[2U]));
            s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_5]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_5]), _S45, s_0[3U]));
            s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_5]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_5]), _S45, s_0[4U]));
            s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_5]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_5]), _S45, s_0[5U]));
            s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_5]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_5]), _S45, s_0[6U]));
            s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_5]), _S46, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_5]), _S45, s_0[7U]));
            uint pr_6 = p0_0 + 6U;
            uint kw_6;
            if(pr_6 < d2_0)
            {
                kw_6 = *((&kernelContext_0)->k_0+(e_1 + pr_6));
            }
            else
            {
                kw_6 = 0U;
            }
            float _S47 = attnLo_0(kw_6);
            float _S48 = attnHi_0(kw_6);
            s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_6]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_6]), _S47, s_0[0U]));
            s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_6]), _S47, s_0[1U]));
            s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_6]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_6]), _S47, s_0[2U]));
            s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_6]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_6]), _S47, s_0[3U]));
            s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_6]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_6]), _S47, s_0[4U]));
            s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_6]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_6]), _S47, s_0[5U]));
            s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_6]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_6]), _S47, s_0[6U]));
            s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_6]), _S48, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_6]), _S47, s_0[7U]));
            uint pr_7 = p0_0 + 7U;
            uint kw_7;
            if(pr_7 < d2_0)
            {
                kw_7 = *((&kernelContext_0)->k_0+(e_1 + pr_7));
            }
            else
            {
                kw_7 = 0U;
            }
            float _S49 = attnLo_0(kw_7);
            float _S50 = attnHi_0(kw_7);
            s_0[0U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[pr_7]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[pr_7]), _S49, s_0[0U]));
            s_0[1U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[128U + pr_7]), _S49, s_0[1U]));
            s_0[2U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[256U + pr_7]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[256U + pr_7]), _S49, s_0[2U]));
            s_0[3U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[384U + pr_7]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[384U + pr_7]), _S49, s_0[3U]));
            s_0[4U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[512U + pr_7]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[512U + pr_7]), _S49, s_0[4U]));
            s_0[5U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[640U + pr_7]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[640U + pr_7]), _S49, s_0[5U]));
            s_0[6U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[768U + pr_7]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[768U + pr_7]), _S49, s_0[6U]));
            s_0[7U] = fma(attnHi_0((*(&kernelContext_0)->qs_0)[896U + pr_7]), _S50, fma(attnLo_0((*(&kernelContext_0)->qs_0)[896U + pr_7]), _S49, s_0[7U]));
            p0_0 = p0_0 + 8U;
        }
        for(;;)
        {
            for(;;)
            {
                for(;;)
                {
                    if(0U >= _S9)
                    {
                        _S31 = int(1);
                        break;
                    }
                    uint i_1 = 0U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_1 + _S10);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[0U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[w_2] = acc_0;
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
            for(;;)
            {
                for(;;)
                {
                    if(1U >= _S9)
                    {
                        _S29 = int(1);
                        break;
                    }
                    uint i_2 = 1U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_2 + _S10);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[1U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[_S13 + w_2] = acc_0;
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
            for(;;)
            {
                for(;;)
                {
                    if(2U >= _S9)
                    {
                        _S27 = int(1);
                        break;
                    }
                    uint i_3 = 2U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_3 + _S10);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[2U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[2U * _S13 + w_2] = acc_0;
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
            for(;;)
            {
                for(;;)
                {
                    if(3U >= _S9)
                    {
                        _S25 = int(1);
                        break;
                    }
                    uint i_4 = 3U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_4 + _S10);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[3U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[3U * _S13 + w_2] = acc_0;
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
            for(;;)
            {
                for(;;)
                {
                    if(4U >= _S9)
                    {
                        _S23 = int(1);
                        break;
                    }
                    uint i_5 = 4U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_5 + _S10);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[4U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[4U * _S13 + w_2] = acc_0;
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
            for(;;)
            {
                for(;;)
                {
                    if(5U >= _S9)
                    {
                        _S21 = int(1);
                        break;
                    }
                    uint i_6 = 5U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_6 + _S10);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[5U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[5U * _S13 + w_2] = acc_0;
                    _S21 = int(3);
                    break;
                }
                if(_S21 != int(3))
                {
                    _S20 = _S21;
                    break;
                }
                _S20 = int(2);
                break;
            }
            if(_S20 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(6U >= _S9)
                    {
                        _S19 = int(1);
                        break;
                    }
                    uint i_7 = 6U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_7 + _S10);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[6U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[6U * _S13 + w_2] = acc_0;
                    _S19 = int(3);
                    break;
                }
                if(_S19 != int(3))
                {
                    _S18 = _S19;
                    break;
                }
                _S18 = int(2);
                break;
            }
            if(_S18 != int(2))
            {
                break;
            }
            for(;;)
            {
                for(;;)
                {
                    if(7U >= _S9)
                    {
                        _S17 = int(1);
                        break;
                    }
                    uint i_8 = 7U % (&kernelContext_0)->p_0->queries_0;
                    if(((&kernelContext_0)->p_0->causal_0) == 0U)
                    {
                        visible_0 = true;
                    }
                    else
                    {
                        visible_0 = key_0 <= (i_8 + _S10);
                    }
                    if(visible_0)
                    {
                        acc_0 = s_0[7U] * (&kernelContext_0)->p_0->scale_0;
                    }
                    else
                    {
                        acc_0 = dg_negative_infinity_0();
                    }
                    (*(&kernelContext_0)->ss_0)[7U * _S13 + w_2] = acc_0;
                    _S17 = int(3);
                    break;
                }
                if(_S17 != int(3))
                {
                    _S16 = _S17;
                    break;
                }
                _S16 = int(2);
                break;
            }
            if(_S16 != int(2))
            {
                break;
            }
            break;
        }
        w_2 = w_2 + 128U;
    }
    float pj_0;
    float acc_1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p0_0 = 0U;
    for(;;)
    {
        if(p0_0 < _S9)
        {
        }
        else
        {
            break;
        }
        float _S51 = dg_negative_infinity_0();
        w_2 = t_0;
        acc_0 = _S51;
        for(;;)
        {
            if(w_2 < _S12)
            {
            }
            else
            {
                break;
            }
            float _S52 = max(acc_0, (*(&kernelContext_0)->ss_0)[p0_0 * _S13 + w_2]);
            w_2 = w_2 + 128U;
            acc_0 = _S52;
        }
        (*(&kernelContext_0)->red_0)[t_0] = acc_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S53 = t_0 < 64U;
        if(_S53)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S54 = t_0 < 32U;
        if(_S54)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S55 = t_0 < 16U;
        if(_S55)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S56 = t_0 < 8U;
        if(_S56)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S57 = t_0 < 4U;
        if(_S57)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S58 = t_0 < 2U;
        if(_S58)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S59 = t_0 < 1U;
        if(_S59)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_max_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S60 = t_0 == 0U;
        if(_S60)
        {
            (*(&kernelContext_0)->rowM_0)[p0_0] = (*(&kernelContext_0)->red_0)[int(0)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S61 = (*(&kernelContext_0)->rowM_0)[p0_0];
        kw_5 = t_0;
        acc_1 = 0.0;
        for(;;)
        {
            if(kw_5 < _S12)
            {
            }
            else
            {
                break;
            }
            float sv_0 = (*(&kernelContext_0)->ss_0)[p0_0 * _S13 + kw_5];
            if(((*(&kernelContext_0)->ss_0)[p0_0 * _S13 + kw_5]) == _S51)
            {
                pj_0 = 0.0;
            }
            else
            {
                pj_0 = exp(sv_0 - _S61);
            }
            (*(&kernelContext_0)->ss_0)[p0_0 * _S13 + kw_5] = pj_0;
            float _S62 = acc_1 + pj_0;
            kw_5 = kw_5 + 128U;
            acc_1 = _S62;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_0)->red_0)[t_0] = acc_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S53)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 64U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S54)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 32U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S55)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 16U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S56)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 8U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S57)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 4U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S58)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 2U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S59)
        {
            (*(&kernelContext_0)->red_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->red_0)[t_0], (*(&kernelContext_0)->red_0)[t_0 + 1U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S60)
        {
            (*(&kernelContext_0)->rowL_0)[p0_0] = (*(&kernelContext_0)->red_0)[int(0)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        p0_0 = p0_0 + 1U;
    }
    uint pair_0 = t_0 % 128U;
    uint part_0 = t_0 / 128U;
    thread array<float, int(8)> lo_1;
    thread array<float, int(8)> hi_1;
    lo_1[0U] = 0.0;
    hi_1[0U] = 0.0;
    lo_1[1U] = 0.0;
    hi_1[1U] = 0.0;
    lo_1[2U] = 0.0;
    hi_1[2U] = 0.0;
    lo_1[3U] = 0.0;
    hi_1[3U] = 0.0;
    lo_1[4U] = 0.0;
    hi_1[4U] = 0.0;
    lo_1[5U] = 0.0;
    hi_1[5U] = 0.0;
    lo_1[6U] = 0.0;
    hi_1[6U] = 0.0;
    lo_1[7U] = 0.0;
    hi_1[7U] = 0.0;
    if(pair_0 < d2_0)
    {
        kw_5 = part_0;
        for(;;)
        {
            bool _S63 = kw_5 < _S12;
            if(_S63)
            {
            }
            else
            {
                break;
            }
            thread array<uint, int(8)> vw_0;
            if(_S63)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    e_1 = (_S8 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + kw_5)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S64 = kStart_0 + kw_5;
                    uint _S65 = _S64 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S66 = (*((&kernelContext_0)->table_0+_S65) * p_1->kvHeads_0 + _S8) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S67 = _S64 % (&kernelContext_0)->p_0->pageSize_0;
                    e_1 = (_S66 + _S67) * d2_0 + pair_0;
                }
                e_1 = *((&kernelContext_0)->v_0+e_1);
            }
            else
            {
                e_1 = 0U;
            }
            vw_0[0U] = e_1;
            uint j_0 = kw_5 + 1U;
            if(j_0 < _S12)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    w_2 = (_S8 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_0)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S68 = kStart_0 + j_0;
                    uint _S69 = _S68 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S70 = (*((&kernelContext_0)->table_0+_S69) * p_1->kvHeads_0 + _S8) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S71 = _S68 % (&kernelContext_0)->p_0->pageSize_0;
                    w_2 = (_S70 + _S71) * d2_0 + pair_0;
                }
                w_2 = *((&kernelContext_0)->v_0+w_2);
            }
            else
            {
                w_2 = 0U;
            }
            vw_0[1U] = w_2;
            uint j_1 = kw_5 + 2U;
            if(j_1 < _S12)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    p0_0 = (_S8 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_1)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S72 = kStart_0 + j_1;
                    uint _S73 = _S72 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S74 = (*((&kernelContext_0)->table_0+_S73) * p_1->kvHeads_0 + _S8) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S75 = _S72 % (&kernelContext_0)->p_0->pageSize_0;
                    p0_0 = (_S74 + _S75) * d2_0 + pair_0;
                }
                p0_0 = *((&kernelContext_0)->v_0+p0_0);
            }
            else
            {
                p0_0 = 0U;
            }
            vw_0[2U] = p0_0;
            uint j_2 = kw_5 + 3U;
            if(j_2 < _S12)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_4 = (_S8 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_2)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S76 = kStart_0 + j_2;
                    uint _S77 = _S76 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S78 = (*((&kernelContext_0)->table_0+_S77) * p_1->kvHeads_0 + _S8) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S79 = _S76 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_4 = (_S78 + _S79) * d2_0 + pair_0;
                }
                kw_4 = *((&kernelContext_0)->v_0+kw_4);
            }
            else
            {
                kw_4 = 0U;
            }
            vw_0[3U] = kw_4;
            uint j_3 = kw_5 + 4U;
            if(j_3 < _S12)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_3 = (_S8 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_3)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S80 = kStart_0 + j_3;
                    uint _S81 = _S80 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S82 = (*((&kernelContext_0)->table_0+_S81) * p_1->kvHeads_0 + _S8) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S83 = _S80 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_3 = (_S82 + _S83) * d2_0 + pair_0;
                }
                kw_3 = *((&kernelContext_0)->v_0+kw_3);
            }
            else
            {
                kw_3 = 0U;
            }
            vw_0[4U] = kw_3;
            uint j_4 = kw_5 + 5U;
            if(j_4 < _S12)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_2 = (_S8 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_4)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S84 = kStart_0 + j_4;
                    uint _S85 = _S84 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S86 = (*((&kernelContext_0)->table_0+_S85) * p_1->kvHeads_0 + _S8) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S87 = _S84 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_2 = (_S86 + _S87) * d2_0 + pair_0;
                }
                kw_2 = *((&kernelContext_0)->v_0+kw_2);
            }
            else
            {
                kw_2 = 0U;
            }
            vw_0[5U] = kw_2;
            uint j_5 = kw_5 + 6U;
            if(j_5 < _S12)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_1 = (_S8 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_5)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S88 = kStart_0 + j_5;
                    uint _S89 = _S88 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S90 = (*((&kernelContext_0)->table_0+_S89) * p_1->kvHeads_0 + _S8) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S91 = _S88 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_1 = (_S90 + _S91) * d2_0 + pair_0;
                }
                kw_1 = *((&kernelContext_0)->v_0+kw_1);
            }
            else
            {
                kw_1 = 0U;
            }
            vw_0[6U] = kw_1;
            uint j_6 = kw_5 + 7U;
            if(j_6 < _S12)
            {
                if(((&kernelContext_0)->p_0->pageSize_0) == 0U)
                {
                    kw_0 = (_S8 * (&kernelContext_0)->p_0->cacheKeys_0 + (kStart_0 + j_6)) * d2_0 + pair_0;
                }
                else
                {
                    uint _S92 = kStart_0 + j_6;
                    uint _S93 = _S92 / (&kernelContext_0)->p_0->pageSize_0;
                    uint _S94 = (*((&kernelContext_0)->table_0+_S93) * p_1->kvHeads_0 + _S8) * (&kernelContext_0)->p_0->pageSize_0;
                    uint _S95 = _S92 % (&kernelContext_0)->p_0->pageSize_0;
                    kw_0 = (_S94 + _S95) * d2_0 + pair_0;
                }
                kw_0 = *((&kernelContext_0)->v_0+kw_0);
            }
            else
            {
                kw_0 = 0U;
            }
            vw_0[7U] = kw_0;
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        if(kw_5 >= _S12)
                        {
                            _S31 = int(1);
                            visible_0 = false;
                            break;
                        }
                        float _S96 = attnLo_0(vw_0[0U]);
                        float _S97 = attnHi_0(vw_0[0U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    bool _S98 = 0U >= _S9;
                                    _S7 = _S98;
                                    if(_S98)
                                    {
                                        _S31 = int(4);
                                        break;
                                    }
                                    float pj_1 = (*(&kernelContext_0)->ss_0)[kw_5];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[kw_5], _S96, lo_1[0U]);
                                    hi_1[0U] = fma(pj_1, _S97, hi_1[0U]);
                                    _S31 = int(6);
                                    break;
                                }
                                if(_S31 != int(6))
                                {
                                    _S30 = _S31;
                                    break;
                                }
                                _S30 = int(5);
                                break;
                            }
                            if(_S30 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S9)
                                    {
                                        _S29 = int(4);
                                        break;
                                    }
                                    float pj_2 = (*(&kernelContext_0)->ss_0)[_S13 + kw_5];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S13 + kw_5], _S96, lo_1[1U]);
                                    hi_1[1U] = fma(pj_2, _S97, hi_1[1U]);
                                    _S29 = int(6);
                                    break;
                                }
                                if(_S29 != int(6))
                                {
                                    _S28 = _S29;
                                    break;
                                }
                                _S28 = int(5);
                                break;
                            }
                            if(_S28 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S9)
                                    {
                                        _S27 = int(4);
                                        break;
                                    }
                                    float pj_3 = (*(&kernelContext_0)->ss_0)[2U * _S13 + kw_5];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S13 + kw_5], _S96, lo_1[2U]);
                                    hi_1[2U] = fma(pj_3, _S97, hi_1[2U]);
                                    _S27 = int(6);
                                    break;
                                }
                                if(_S27 != int(6))
                                {
                                    _S26 = _S27;
                                    break;
                                }
                                _S26 = int(5);
                                break;
                            }
                            if(_S26 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S9)
                                    {
                                        _S25 = int(4);
                                        break;
                                    }
                                    float pj_4 = (*(&kernelContext_0)->ss_0)[3U * _S13 + kw_5];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S13 + kw_5], _S96, lo_1[3U]);
                                    hi_1[3U] = fma(pj_4, _S97, hi_1[3U]);
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    _S24 = _S25;
                                    break;
                                }
                                _S24 = int(5);
                                break;
                            }
                            if(_S24 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S9)
                                    {
                                        _S23 = int(4);
                                        break;
                                    }
                                    float pj_5 = (*(&kernelContext_0)->ss_0)[4U * _S13 + kw_5];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S13 + kw_5], _S96, lo_1[4U]);
                                    hi_1[4U] = fma(pj_5, _S97, hi_1[4U]);
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    _S22 = _S23;
                                    break;
                                }
                                _S22 = int(5);
                                break;
                            }
                            if(_S22 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S9)
                                    {
                                        _S21 = int(4);
                                        break;
                                    }
                                    float pj_6 = (*(&kernelContext_0)->ss_0)[5U * _S13 + kw_5];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S13 + kw_5], _S96, lo_1[5U]);
                                    hi_1[5U] = fma(pj_6, _S97, hi_1[5U]);
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    _S20 = _S21;
                                    break;
                                }
                                _S20 = int(5);
                                break;
                            }
                            if(_S20 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S9)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_7 = (*(&kernelContext_0)->ss_0)[6U * _S13 + kw_5];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S13 + kw_5], _S96, lo_1[6U]);
                                    hi_1[6U] = fma(pj_7, _S97, hi_1[6U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S9)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_8 = (*(&kernelContext_0)->ss_0)[7U * _S13 + kw_5];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S13 + kw_5], _S96, lo_1[7U]);
                                    hi_1[7U] = fma(pj_8, _S97, hi_1[7U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S31 = int(3);
                        visible_0 = _S7;
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
                int _S99;
                int _S100;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_0 >= _S12)
                        {
                            _S29 = int(1);
                            break;
                        }
                        float _S101 = attnLo_0(vw_0[1U]);
                        float _S102 = attnHi_0(vw_0[1U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S29 = int(4);
                                        break;
                                    }
                                    float pj_9 = (*(&kernelContext_0)->ss_0)[j_0];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_0], _S101, lo_1[0U]);
                                    hi_1[0U] = fma(pj_9, _S102, hi_1[0U]);
                                    _S29 = int(6);
                                    break;
                                }
                                if(_S29 != int(6))
                                {
                                    _S28 = _S29;
                                    break;
                                }
                                _S28 = int(5);
                                break;
                            }
                            if(_S28 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S9)
                                    {
                                        _S27 = int(4);
                                        break;
                                    }
                                    float pj_10 = (*(&kernelContext_0)->ss_0)[_S13 + j_0];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S13 + j_0], _S101, lo_1[1U]);
                                    hi_1[1U] = fma(pj_10, _S102, hi_1[1U]);
                                    _S27 = int(6);
                                    break;
                                }
                                if(_S27 != int(6))
                                {
                                    _S26 = _S27;
                                    break;
                                }
                                _S26 = int(5);
                                break;
                            }
                            if(_S26 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S9)
                                    {
                                        _S25 = int(4);
                                        break;
                                    }
                                    float pj_11 = (*(&kernelContext_0)->ss_0)[2U * _S13 + j_0];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S13 + j_0], _S101, lo_1[2U]);
                                    hi_1[2U] = fma(pj_11, _S102, hi_1[2U]);
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    _S24 = _S25;
                                    break;
                                }
                                _S24 = int(5);
                                break;
                            }
                            if(_S24 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S9)
                                    {
                                        _S23 = int(4);
                                        break;
                                    }
                                    float pj_12 = (*(&kernelContext_0)->ss_0)[3U * _S13 + j_0];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S13 + j_0], _S101, lo_1[3U]);
                                    hi_1[3U] = fma(pj_12, _S102, hi_1[3U]);
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    _S22 = _S23;
                                    break;
                                }
                                _S22 = int(5);
                                break;
                            }
                            if(_S22 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S9)
                                    {
                                        _S21 = int(4);
                                        break;
                                    }
                                    float pj_13 = (*(&kernelContext_0)->ss_0)[4U * _S13 + j_0];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S13 + j_0], _S101, lo_1[4U]);
                                    hi_1[4U] = fma(pj_13, _S102, hi_1[4U]);
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    _S20 = _S21;
                                    break;
                                }
                                _S20 = int(5);
                                break;
                            }
                            if(_S20 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S9)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_14 = (*(&kernelContext_0)->ss_0)[5U * _S13 + j_0];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S13 + j_0], _S101, lo_1[5U]);
                                    hi_1[5U] = fma(pj_14, _S102, hi_1[5U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S9)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_15 = (*(&kernelContext_0)->ss_0)[6U * _S13 + j_0];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S13 + j_0], _S101, lo_1[6U]);
                                    hi_1[6U] = fma(pj_15, _S102, hi_1[6U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S9)
                                    {
                                        _S100 = int(4);
                                        break;
                                    }
                                    float pj_16 = (*(&kernelContext_0)->ss_0)[7U * _S13 + j_0];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S13 + j_0], _S101, lo_1[7U]);
                                    hi_1[7U] = fma(pj_16, _S102, hi_1[7U]);
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    _S99 = _S100;
                                    break;
                                }
                                _S99 = int(5);
                                break;
                            }
                            if(_S99 != int(5))
                            {
                                break;
                            }
                            break;
                        }
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
                int _S103;
                int _S104;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_1 >= _S12)
                        {
                            _S27 = int(1);
                            break;
                        }
                        float _S105 = attnLo_0(vw_0[2U]);
                        float _S106 = attnHi_0(vw_0[2U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S27 = int(4);
                                        break;
                                    }
                                    float pj_17 = (*(&kernelContext_0)->ss_0)[j_1];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_1], _S105, lo_1[0U]);
                                    hi_1[0U] = fma(pj_17, _S106, hi_1[0U]);
                                    _S27 = int(6);
                                    break;
                                }
                                if(_S27 != int(6))
                                {
                                    _S26 = _S27;
                                    break;
                                }
                                _S26 = int(5);
                                break;
                            }
                            if(_S26 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S9)
                                    {
                                        _S25 = int(4);
                                        break;
                                    }
                                    float pj_18 = (*(&kernelContext_0)->ss_0)[_S13 + j_1];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S13 + j_1], _S105, lo_1[1U]);
                                    hi_1[1U] = fma(pj_18, _S106, hi_1[1U]);
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    _S24 = _S25;
                                    break;
                                }
                                _S24 = int(5);
                                break;
                            }
                            if(_S24 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S9)
                                    {
                                        _S23 = int(4);
                                        break;
                                    }
                                    float pj_19 = (*(&kernelContext_0)->ss_0)[2U * _S13 + j_1];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S13 + j_1], _S105, lo_1[2U]);
                                    hi_1[2U] = fma(pj_19, _S106, hi_1[2U]);
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    _S22 = _S23;
                                    break;
                                }
                                _S22 = int(5);
                                break;
                            }
                            if(_S22 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S9)
                                    {
                                        _S21 = int(4);
                                        break;
                                    }
                                    float pj_20 = (*(&kernelContext_0)->ss_0)[3U * _S13 + j_1];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S13 + j_1], _S105, lo_1[3U]);
                                    hi_1[3U] = fma(pj_20, _S106, hi_1[3U]);
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    _S20 = _S21;
                                    break;
                                }
                                _S20 = int(5);
                                break;
                            }
                            if(_S20 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S9)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_21 = (*(&kernelContext_0)->ss_0)[4U * _S13 + j_1];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S13 + j_1], _S105, lo_1[4U]);
                                    hi_1[4U] = fma(pj_21, _S106, hi_1[4U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S9)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_22 = (*(&kernelContext_0)->ss_0)[5U * _S13 + j_1];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S13 + j_1], _S105, lo_1[5U]);
                                    hi_1[5U] = fma(pj_22, _S106, hi_1[5U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S9)
                                    {
                                        _S100 = int(4);
                                        break;
                                    }
                                    float pj_23 = (*(&kernelContext_0)->ss_0)[6U * _S13 + j_1];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S13 + j_1], _S105, lo_1[6U]);
                                    hi_1[6U] = fma(pj_23, _S106, hi_1[6U]);
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    _S99 = _S100;
                                    break;
                                }
                                _S99 = int(5);
                                break;
                            }
                            if(_S99 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S9)
                                    {
                                        _S104 = int(4);
                                        break;
                                    }
                                    float pj_24 = (*(&kernelContext_0)->ss_0)[7U * _S13 + j_1];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S13 + j_1], _S105, lo_1[7U]);
                                    hi_1[7U] = fma(pj_24, _S106, hi_1[7U]);
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    _S103 = _S104;
                                    break;
                                }
                                _S103 = int(5);
                                break;
                            }
                            if(_S103 != int(5))
                            {
                                break;
                            }
                            break;
                        }
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
                int _S107;
                int _S108;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_2 >= _S12)
                        {
                            _S25 = int(1);
                            break;
                        }
                        float _S109 = attnLo_0(vw_0[3U]);
                        float _S110 = attnHi_0(vw_0[3U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S25 = int(4);
                                        break;
                                    }
                                    float pj_25 = (*(&kernelContext_0)->ss_0)[j_2];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_2], _S109, lo_1[0U]);
                                    hi_1[0U] = fma(pj_25, _S110, hi_1[0U]);
                                    _S25 = int(6);
                                    break;
                                }
                                if(_S25 != int(6))
                                {
                                    _S24 = _S25;
                                    break;
                                }
                                _S24 = int(5);
                                break;
                            }
                            if(_S24 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S9)
                                    {
                                        _S23 = int(4);
                                        break;
                                    }
                                    float pj_26 = (*(&kernelContext_0)->ss_0)[_S13 + j_2];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S13 + j_2], _S109, lo_1[1U]);
                                    hi_1[1U] = fma(pj_26, _S110, hi_1[1U]);
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    _S22 = _S23;
                                    break;
                                }
                                _S22 = int(5);
                                break;
                            }
                            if(_S22 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S9)
                                    {
                                        _S21 = int(4);
                                        break;
                                    }
                                    float pj_27 = (*(&kernelContext_0)->ss_0)[2U * _S13 + j_2];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S13 + j_2], _S109, lo_1[2U]);
                                    hi_1[2U] = fma(pj_27, _S110, hi_1[2U]);
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    _S20 = _S21;
                                    break;
                                }
                                _S20 = int(5);
                                break;
                            }
                            if(_S20 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S9)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_28 = (*(&kernelContext_0)->ss_0)[3U * _S13 + j_2];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S13 + j_2], _S109, lo_1[3U]);
                                    hi_1[3U] = fma(pj_28, _S110, hi_1[3U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S9)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_29 = (*(&kernelContext_0)->ss_0)[4U * _S13 + j_2];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S13 + j_2], _S109, lo_1[4U]);
                                    hi_1[4U] = fma(pj_29, _S110, hi_1[4U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S9)
                                    {
                                        _S100 = int(4);
                                        break;
                                    }
                                    float pj_30 = (*(&kernelContext_0)->ss_0)[5U * _S13 + j_2];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S13 + j_2], _S109, lo_1[5U]);
                                    hi_1[5U] = fma(pj_30, _S110, hi_1[5U]);
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    _S99 = _S100;
                                    break;
                                }
                                _S99 = int(5);
                                break;
                            }
                            if(_S99 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S9)
                                    {
                                        _S104 = int(4);
                                        break;
                                    }
                                    float pj_31 = (*(&kernelContext_0)->ss_0)[6U * _S13 + j_2];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S13 + j_2], _S109, lo_1[6U]);
                                    hi_1[6U] = fma(pj_31, _S110, hi_1[6U]);
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    _S103 = _S104;
                                    break;
                                }
                                _S103 = int(5);
                                break;
                            }
                            if(_S103 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S9)
                                    {
                                        _S108 = int(4);
                                        break;
                                    }
                                    float pj_32 = (*(&kernelContext_0)->ss_0)[7U * _S13 + j_2];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S13 + j_2], _S109, lo_1[7U]);
                                    hi_1[7U] = fma(pj_32, _S110, hi_1[7U]);
                                    _S108 = int(6);
                                    break;
                                }
                                if(_S108 != int(6))
                                {
                                    _S107 = _S108;
                                    break;
                                }
                                _S107 = int(5);
                                break;
                            }
                            if(_S107 != int(5))
                            {
                                break;
                            }
                            break;
                        }
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
                int _S111;
                int _S112;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_3 >= _S12)
                        {
                            _S23 = int(1);
                            break;
                        }
                        float _S113 = attnLo_0(vw_0[4U]);
                        float _S114 = attnHi_0(vw_0[4U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S23 = int(4);
                                        break;
                                    }
                                    float pj_33 = (*(&kernelContext_0)->ss_0)[j_3];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_3], _S113, lo_1[0U]);
                                    hi_1[0U] = fma(pj_33, _S114, hi_1[0U]);
                                    _S23 = int(6);
                                    break;
                                }
                                if(_S23 != int(6))
                                {
                                    _S22 = _S23;
                                    break;
                                }
                                _S22 = int(5);
                                break;
                            }
                            if(_S22 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S9)
                                    {
                                        _S21 = int(4);
                                        break;
                                    }
                                    float pj_34 = (*(&kernelContext_0)->ss_0)[_S13 + j_3];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S13 + j_3], _S113, lo_1[1U]);
                                    hi_1[1U] = fma(pj_34, _S114, hi_1[1U]);
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    _S20 = _S21;
                                    break;
                                }
                                _S20 = int(5);
                                break;
                            }
                            if(_S20 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S9)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_35 = (*(&kernelContext_0)->ss_0)[2U * _S13 + j_3];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S13 + j_3], _S113, lo_1[2U]);
                                    hi_1[2U] = fma(pj_35, _S114, hi_1[2U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S9)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_36 = (*(&kernelContext_0)->ss_0)[3U * _S13 + j_3];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S13 + j_3], _S113, lo_1[3U]);
                                    hi_1[3U] = fma(pj_36, _S114, hi_1[3U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S9)
                                    {
                                        _S100 = int(4);
                                        break;
                                    }
                                    float pj_37 = (*(&kernelContext_0)->ss_0)[4U * _S13 + j_3];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S13 + j_3], _S113, lo_1[4U]);
                                    hi_1[4U] = fma(pj_37, _S114, hi_1[4U]);
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    _S99 = _S100;
                                    break;
                                }
                                _S99 = int(5);
                                break;
                            }
                            if(_S99 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S9)
                                    {
                                        _S104 = int(4);
                                        break;
                                    }
                                    float pj_38 = (*(&kernelContext_0)->ss_0)[5U * _S13 + j_3];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S13 + j_3], _S113, lo_1[5U]);
                                    hi_1[5U] = fma(pj_38, _S114, hi_1[5U]);
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    _S103 = _S104;
                                    break;
                                }
                                _S103 = int(5);
                                break;
                            }
                            if(_S103 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S9)
                                    {
                                        _S108 = int(4);
                                        break;
                                    }
                                    float pj_39 = (*(&kernelContext_0)->ss_0)[6U * _S13 + j_3];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S13 + j_3], _S113, lo_1[6U]);
                                    hi_1[6U] = fma(pj_39, _S114, hi_1[6U]);
                                    _S108 = int(6);
                                    break;
                                }
                                if(_S108 != int(6))
                                {
                                    _S107 = _S108;
                                    break;
                                }
                                _S107 = int(5);
                                break;
                            }
                            if(_S107 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S9)
                                    {
                                        _S112 = int(4);
                                        break;
                                    }
                                    float pj_40 = (*(&kernelContext_0)->ss_0)[7U * _S13 + j_3];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S13 + j_3], _S113, lo_1[7U]);
                                    hi_1[7U] = fma(pj_40, _S114, hi_1[7U]);
                                    _S112 = int(6);
                                    break;
                                }
                                if(_S112 != int(6))
                                {
                                    _S111 = _S112;
                                    break;
                                }
                                _S111 = int(5);
                                break;
                            }
                            if(_S111 != int(5))
                            {
                                break;
                            }
                            break;
                        }
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
                int _S115;
                int _S116;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_4 >= _S12)
                        {
                            _S21 = int(1);
                            break;
                        }
                        float _S117 = attnLo_0(vw_0[5U]);
                        float _S118 = attnHi_0(vw_0[5U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S21 = int(4);
                                        break;
                                    }
                                    float pj_41 = (*(&kernelContext_0)->ss_0)[j_4];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_4], _S117, lo_1[0U]);
                                    hi_1[0U] = fma(pj_41, _S118, hi_1[0U]);
                                    _S21 = int(6);
                                    break;
                                }
                                if(_S21 != int(6))
                                {
                                    _S20 = _S21;
                                    break;
                                }
                                _S20 = int(5);
                                break;
                            }
                            if(_S20 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S9)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_42 = (*(&kernelContext_0)->ss_0)[_S13 + j_4];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S13 + j_4], _S117, lo_1[1U]);
                                    hi_1[1U] = fma(pj_42, _S118, hi_1[1U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S9)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_43 = (*(&kernelContext_0)->ss_0)[2U * _S13 + j_4];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S13 + j_4], _S117, lo_1[2U]);
                                    hi_1[2U] = fma(pj_43, _S118, hi_1[2U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S9)
                                    {
                                        _S100 = int(4);
                                        break;
                                    }
                                    float pj_44 = (*(&kernelContext_0)->ss_0)[3U * _S13 + j_4];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S13 + j_4], _S117, lo_1[3U]);
                                    hi_1[3U] = fma(pj_44, _S118, hi_1[3U]);
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    _S99 = _S100;
                                    break;
                                }
                                _S99 = int(5);
                                break;
                            }
                            if(_S99 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S9)
                                    {
                                        _S104 = int(4);
                                        break;
                                    }
                                    float pj_45 = (*(&kernelContext_0)->ss_0)[4U * _S13 + j_4];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S13 + j_4], _S117, lo_1[4U]);
                                    hi_1[4U] = fma(pj_45, _S118, hi_1[4U]);
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    _S103 = _S104;
                                    break;
                                }
                                _S103 = int(5);
                                break;
                            }
                            if(_S103 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S9)
                                    {
                                        _S108 = int(4);
                                        break;
                                    }
                                    float pj_46 = (*(&kernelContext_0)->ss_0)[5U * _S13 + j_4];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S13 + j_4], _S117, lo_1[5U]);
                                    hi_1[5U] = fma(pj_46, _S118, hi_1[5U]);
                                    _S108 = int(6);
                                    break;
                                }
                                if(_S108 != int(6))
                                {
                                    _S107 = _S108;
                                    break;
                                }
                                _S107 = int(5);
                                break;
                            }
                            if(_S107 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S9)
                                    {
                                        _S112 = int(4);
                                        break;
                                    }
                                    float pj_47 = (*(&kernelContext_0)->ss_0)[6U * _S13 + j_4];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S13 + j_4], _S117, lo_1[6U]);
                                    hi_1[6U] = fma(pj_47, _S118, hi_1[6U]);
                                    _S112 = int(6);
                                    break;
                                }
                                if(_S112 != int(6))
                                {
                                    _S111 = _S112;
                                    break;
                                }
                                _S111 = int(5);
                                break;
                            }
                            if(_S111 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S9)
                                    {
                                        _S116 = int(4);
                                        break;
                                    }
                                    float pj_48 = (*(&kernelContext_0)->ss_0)[7U * _S13 + j_4];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S13 + j_4], _S117, lo_1[7U]);
                                    hi_1[7U] = fma(pj_48, _S118, hi_1[7U]);
                                    _S116 = int(6);
                                    break;
                                }
                                if(_S116 != int(6))
                                {
                                    _S115 = _S116;
                                    break;
                                }
                                _S115 = int(5);
                                break;
                            }
                            if(_S115 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S21 = int(3);
                        break;
                    }
                    if(_S21 != int(3))
                    {
                        _S20 = _S21;
                        break;
                    }
                    _S20 = int(2);
                    break;
                }
                if(_S20 != int(2))
                {
                    break;
                }
                int _S119;
                int _S120;
                for(;;)
                {
                    for(;;)
                    {
                        if(j_5 >= _S12)
                        {
                            _S19 = int(1);
                            break;
                        }
                        float _S121 = attnLo_0(vw_0[6U]);
                        float _S122 = attnHi_0(vw_0[6U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S19 = int(4);
                                        break;
                                    }
                                    float pj_49 = (*(&kernelContext_0)->ss_0)[j_5];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_5], _S121, lo_1[0U]);
                                    hi_1[0U] = fma(pj_49, _S122, hi_1[0U]);
                                    _S19 = int(6);
                                    break;
                                }
                                if(_S19 != int(6))
                                {
                                    _S18 = _S19;
                                    break;
                                }
                                _S18 = int(5);
                                break;
                            }
                            if(_S18 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S9)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_50 = (*(&kernelContext_0)->ss_0)[_S13 + j_5];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S13 + j_5], _S121, lo_1[1U]);
                                    hi_1[1U] = fma(pj_50, _S122, hi_1[1U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S9)
                                    {
                                        _S100 = int(4);
                                        break;
                                    }
                                    float pj_51 = (*(&kernelContext_0)->ss_0)[2U * _S13 + j_5];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S13 + j_5], _S121, lo_1[2U]);
                                    hi_1[2U] = fma(pj_51, _S122, hi_1[2U]);
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    _S99 = _S100;
                                    break;
                                }
                                _S99 = int(5);
                                break;
                            }
                            if(_S99 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S9)
                                    {
                                        _S104 = int(4);
                                        break;
                                    }
                                    float pj_52 = (*(&kernelContext_0)->ss_0)[3U * _S13 + j_5];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S13 + j_5], _S121, lo_1[3U]);
                                    hi_1[3U] = fma(pj_52, _S122, hi_1[3U]);
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    _S103 = _S104;
                                    break;
                                }
                                _S103 = int(5);
                                break;
                            }
                            if(_S103 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S9)
                                    {
                                        _S108 = int(4);
                                        break;
                                    }
                                    float pj_53 = (*(&kernelContext_0)->ss_0)[4U * _S13 + j_5];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S13 + j_5], _S121, lo_1[4U]);
                                    hi_1[4U] = fma(pj_53, _S122, hi_1[4U]);
                                    _S108 = int(6);
                                    break;
                                }
                                if(_S108 != int(6))
                                {
                                    _S107 = _S108;
                                    break;
                                }
                                _S107 = int(5);
                                break;
                            }
                            if(_S107 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S9)
                                    {
                                        _S112 = int(4);
                                        break;
                                    }
                                    float pj_54 = (*(&kernelContext_0)->ss_0)[5U * _S13 + j_5];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S13 + j_5], _S121, lo_1[5U]);
                                    hi_1[5U] = fma(pj_54, _S122, hi_1[5U]);
                                    _S112 = int(6);
                                    break;
                                }
                                if(_S112 != int(6))
                                {
                                    _S111 = _S112;
                                    break;
                                }
                                _S111 = int(5);
                                break;
                            }
                            if(_S111 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S9)
                                    {
                                        _S116 = int(4);
                                        break;
                                    }
                                    float pj_55 = (*(&kernelContext_0)->ss_0)[6U * _S13 + j_5];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S13 + j_5], _S121, lo_1[6U]);
                                    hi_1[6U] = fma(pj_55, _S122, hi_1[6U]);
                                    _S116 = int(6);
                                    break;
                                }
                                if(_S116 != int(6))
                                {
                                    _S115 = _S116;
                                    break;
                                }
                                _S115 = int(5);
                                break;
                            }
                            if(_S115 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(7U >= _S9)
                                    {
                                        _S120 = int(4);
                                        break;
                                    }
                                    float pj_56 = (*(&kernelContext_0)->ss_0)[7U * _S13 + j_5];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S13 + j_5], _S121, lo_1[7U]);
                                    hi_1[7U] = fma(pj_56, _S122, hi_1[7U]);
                                    _S120 = int(6);
                                    break;
                                }
                                if(_S120 != int(6))
                                {
                                    _S119 = _S120;
                                    break;
                                }
                                _S119 = int(5);
                                break;
                            }
                            if(_S119 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S19 = int(3);
                        break;
                    }
                    if(_S19 != int(3))
                    {
                        _S18 = _S19;
                        break;
                    }
                    _S18 = int(2);
                    break;
                }
                if(_S18 != int(2))
                {
                    break;
                }
                for(;;)
                {
                    for(;;)
                    {
                        if(j_6 >= _S12)
                        {
                            _S17 = int(1);
                            break;
                        }
                        float _S123 = attnLo_0(vw_0[7U]);
                        float _S124 = attnHi_0(vw_0[7U]);
                        for(;;)
                        {
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(visible_0)
                                    {
                                        _S17 = int(4);
                                        break;
                                    }
                                    float pj_57 = (*(&kernelContext_0)->ss_0)[j_6];
                                    lo_1[0U] = fma((*(&kernelContext_0)->ss_0)[j_6], _S123, lo_1[0U]);
                                    hi_1[0U] = fma(pj_57, _S124, hi_1[0U]);
                                    _S17 = int(6);
                                    break;
                                }
                                if(_S17 != int(6))
                                {
                                    _S16 = _S17;
                                    break;
                                }
                                _S16 = int(5);
                                break;
                            }
                            if(_S16 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(1U >= _S9)
                                    {
                                        _S100 = int(4);
                                        break;
                                    }
                                    float pj_58 = (*(&kernelContext_0)->ss_0)[_S13 + j_6];
                                    lo_1[1U] = fma((*(&kernelContext_0)->ss_0)[_S13 + j_6], _S123, lo_1[1U]);
                                    hi_1[1U] = fma(pj_58, _S124, hi_1[1U]);
                                    _S100 = int(6);
                                    break;
                                }
                                if(_S100 != int(6))
                                {
                                    _S99 = _S100;
                                    break;
                                }
                                _S99 = int(5);
                                break;
                            }
                            if(_S99 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(2U >= _S9)
                                    {
                                        _S104 = int(4);
                                        break;
                                    }
                                    float pj_59 = (*(&kernelContext_0)->ss_0)[2U * _S13 + j_6];
                                    lo_1[2U] = fma((*(&kernelContext_0)->ss_0)[2U * _S13 + j_6], _S123, lo_1[2U]);
                                    hi_1[2U] = fma(pj_59, _S124, hi_1[2U]);
                                    _S104 = int(6);
                                    break;
                                }
                                if(_S104 != int(6))
                                {
                                    _S103 = _S104;
                                    break;
                                }
                                _S103 = int(5);
                                break;
                            }
                            if(_S103 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(3U >= _S9)
                                    {
                                        _S108 = int(4);
                                        break;
                                    }
                                    float pj_60 = (*(&kernelContext_0)->ss_0)[3U * _S13 + j_6];
                                    lo_1[3U] = fma((*(&kernelContext_0)->ss_0)[3U * _S13 + j_6], _S123, lo_1[3U]);
                                    hi_1[3U] = fma(pj_60, _S124, hi_1[3U]);
                                    _S108 = int(6);
                                    break;
                                }
                                if(_S108 != int(6))
                                {
                                    _S107 = _S108;
                                    break;
                                }
                                _S107 = int(5);
                                break;
                            }
                            if(_S107 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(4U >= _S9)
                                    {
                                        _S112 = int(4);
                                        break;
                                    }
                                    float pj_61 = (*(&kernelContext_0)->ss_0)[4U * _S13 + j_6];
                                    lo_1[4U] = fma((*(&kernelContext_0)->ss_0)[4U * _S13 + j_6], _S123, lo_1[4U]);
                                    hi_1[4U] = fma(pj_61, _S124, hi_1[4U]);
                                    _S112 = int(6);
                                    break;
                                }
                                if(_S112 != int(6))
                                {
                                    _S111 = _S112;
                                    break;
                                }
                                _S111 = int(5);
                                break;
                            }
                            if(_S111 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(5U >= _S9)
                                    {
                                        _S116 = int(4);
                                        break;
                                    }
                                    float pj_62 = (*(&kernelContext_0)->ss_0)[5U * _S13 + j_6];
                                    lo_1[5U] = fma((*(&kernelContext_0)->ss_0)[5U * _S13 + j_6], _S123, lo_1[5U]);
                                    hi_1[5U] = fma(pj_62, _S124, hi_1[5U]);
                                    _S116 = int(6);
                                    break;
                                }
                                if(_S116 != int(6))
                                {
                                    _S115 = _S116;
                                    break;
                                }
                                _S115 = int(5);
                                break;
                            }
                            if(_S115 != int(5))
                            {
                                break;
                            }
                            for(;;)
                            {
                                for(;;)
                                {
                                    if(6U >= _S9)
                                    {
                                        _S120 = int(4);
                                        break;
                                    }
                                    float pj_63 = (*(&kernelContext_0)->ss_0)[6U * _S13 + j_6];
                                    lo_1[6U] = fma((*(&kernelContext_0)->ss_0)[6U * _S13 + j_6], _S123, lo_1[6U]);
                                    hi_1[6U] = fma(pj_63, _S124, hi_1[6U]);
                                    _S120 = int(6);
                                    break;
                                }
                                if(_S120 != int(6))
                                {
                                    _S119 = _S120;
                                    break;
                                }
                                _S119 = int(5);
                                break;
                            }
                            if(_S119 != int(5))
                            {
                                break;
                            }
                            int _S125;
                            for(;;)
                            {
                                int _S126;
                                for(;;)
                                {
                                    if(7U >= _S9)
                                    {
                                        _S126 = int(4);
                                        break;
                                    }
                                    float pj_64 = (*(&kernelContext_0)->ss_0)[7U * _S13 + j_6];
                                    lo_1[7U] = fma((*(&kernelContext_0)->ss_0)[7U * _S13 + j_6], _S123, lo_1[7U]);
                                    hi_1[7U] = fma(pj_64, _S124, hi_1[7U]);
                                    _S126 = int(6);
                                    break;
                                }
                                if(_S126 != int(6))
                                {
                                    _S125 = _S126;
                                    break;
                                }
                                _S125 = int(5);
                                break;
                            }
                            if(_S125 != int(5))
                            {
                                break;
                            }
                            break;
                        }
                        _S17 = int(3);
                        break;
                    }
                    if(_S17 != int(3))
                    {
                        _S16 = _S17;
                        break;
                    }
                    _S16 = int(2);
                    break;
                }
                if(_S16 != int(2))
                {
                    break;
                }
                break;
            }
            kw_5 = kw_5 + 8U;
        }
    }
    p0_0 = 0U;
    for(;;)
    {
        if(p0_0 < _S9)
        {
        }
        else
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(0U == p0_0)
        {
            acc_0 = lo_1[0U];
            acc_1 = hi_1[0U];
        }
        else
        {
            acc_0 = 0.0;
            acc_1 = 0.0;
        }
        float b_3;
        if(1U == p0_0)
        {
            pj_0 = lo_1[1U];
            b_3 = hi_1[1U];
        }
        else
        {
            pj_0 = acc_0;
            b_3 = acc_1;
        }
        float a_3;
        float b_4;
        if(2U == p0_0)
        {
            a_3 = lo_1[2U];
            b_4 = hi_1[2U];
        }
        else
        {
            a_3 = pj_0;
            b_4 = b_3;
        }
        float a_4;
        float b_5;
        if(3U == p0_0)
        {
            a_4 = lo_1[3U];
            b_5 = hi_1[3U];
        }
        else
        {
            a_4 = a_3;
            b_5 = b_4;
        }
        float a_5;
        float b_6;
        if(4U == p0_0)
        {
            a_5 = lo_1[4U];
            b_6 = hi_1[4U];
        }
        else
        {
            a_5 = a_4;
            b_6 = b_5;
        }
        float a_6;
        float b_7;
        if(5U == p0_0)
        {
            a_6 = lo_1[5U];
            b_7 = hi_1[5U];
        }
        else
        {
            a_6 = a_5;
            b_7 = b_6;
        }
        float a_7;
        float b_8;
        if(6U == p0_0)
        {
            a_7 = lo_1[6U];
            b_8 = hi_1[6U];
        }
        else
        {
            a_7 = a_6;
            b_8 = b_7;
        }
        float a_8;
        float b_9;
        if(7U == p0_0)
        {
            a_8 = lo_1[7U];
            b_9 = hi_1[7U];
        }
        else
        {
            a_8 = a_7;
            b_9 = b_8;
        }
        uint _S127 = 2U * pair_0;
        uint _S128 = part_0 * 256U + _S127;
        (*(&kernelContext_0)->ss_0)[_S128] = a_8;
        (*(&kernelContext_0)->ss_0)[_S128 + 1U] = b_9;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(part_0 != 0U)
        {
            visible_0 = true;
        }
        else
        {
            visible_0 = pair_0 >= d2_0;
        }
        if(visible_0)
        {
            p0_0 = p0_0 + 1U;
            continue;
        }
        float a_9 = (*(&kernelContext_0)->ss_0)[_S127];
        float b_10 = (*(&kernelContext_0)->ss_0)[_S127 + 1U];
        uint _S129 = _S8 * hpk_0;
        uint _S130 = p0_0 / (&kernelContext_0)->p_0->queries_0;
        uint h_4 = _S129 + _S130;
        uint i_9 = p0_0 % (&kernelContext_0)->p_0->queries_0;
        uint row_0 = h_4 * (&kernelContext_0)->p_0->queries_0 + i_9;
        if(((&kernelContext_0)->p_0->splits_0) > 1U)
        {
            uint pr_8 = split_0 * p_1->heads_0 * (&kernelContext_0)->p_0->queries_0 + row_0;
            if(pair_0 == 0U)
            {
                uint _S131 = 2U * pr_8;
                float device* _S132 = (&kernelContext_0)->partialML_0+_S131;
                bool _S133 = _S12 == 0U;
                float _S134;
                if(_S133)
                {
                    _S134 = dg_negative_infinity_0();
                }
                else
                {
                    _S134 = (*(&kernelContext_0)->rowM_0)[p0_0];
                }
                *_S132 = _S134;
                float device* _S135 = (&kernelContext_0)->partialML_0+(_S131 + 1U);
                float _S136;
                if(_S133)
                {
                    _S136 = 0.0;
                }
                else
                {
                    _S136 = (*(&kernelContext_0)->rowL_0)[p0_0];
                }
                *_S135 = _S136;
            }
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S127)) = a_9;
            *((&kernelContext_0)->partialAcc_0+(pr_8 * (&kernelContext_0)->p_0->headDim_0 + _S127 + 1U)) = b_10;
            p0_0 = p0_0 + 1U;
            continue;
        }
        *((&kernelContext_0)->o_0+(row_0 * d2_0 + pair_0)) = attnPack_0(a_9 / (*(&kernelContext_0)->rowL_0)[p0_0], b_10 / (*(&kernelContext_0)->rowL_0)[p0_0]);
        p0_0 = p0_0 + 1U;
    }
    return;
}

