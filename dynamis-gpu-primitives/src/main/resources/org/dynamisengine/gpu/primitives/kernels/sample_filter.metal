#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
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

float dg_negative_infinity_0()
{
    return (as_type<float>((4286578688U)));
}

float sanitize_0(float l_0)
{
    float _S3;
    if(dg_is_nan_0(l_0))
    {
        _S3 = dg_negative_infinity_0();
    }
    else
    {
        if(l_0 == 0.0)
        {
            _S3 = 0.0;
        }
        else
        {
            _S3 = l_0;
        }
    }
    return _S3;
}

uint key_0(float v_0)
{
    uint f_0 = (as_type<uint>((v_0)));
    uint _S4;
    if((f_0 & 2147483648U) != 0U)
    {
        _S4 = ~f_0;
    }
    else
    {
        _S4 = f_0 | 2147483648U;
    }
    return _S4;
}

bool before_0(float va_0, uint ia_0, float vb_0, uint ib_0)
{
    bool _S5;
    if(va_0 > vb_0)
    {
        _S5 = true;
    }
    else
    {
        if(va_0 == vb_0)
        {
            _S5 = ia_0 < ib_0;
        }
        else
        {
            _S5 = false;
        }
    }
    return _S5;
}

struct SampleParams_0
{
    uint rows_0;
    uint vocab_0;
    uint ldl_0;
    float invT_0;
    uint topK_0;
    float topP_0;
    uint seedLo_0;
    uint seedHi_0;
    uint step_0;
    uint chunks_0;
    uint restricted_0;
    uint lists_0;
    uint stride_0;
    uint head_0;
    uint perGroup_0;
};

struct KernelContext_0
{
    SampleParams_0 constant* p_0;
    float device* logits_0;
    uint4 device* info_0;
    float device* sortedV_0;
    uint device* sortedI_0;
    uint threadgroup* windowBound_0;
    array<float, int(1024)> threadgroup* sv_0;
    array<uint, int(1024)> threadgroup* si_0;
    array<uint, int(1024)> threadgroup* ranks_0;
    float threadgroup* cutoffV_0;
    uint threadgroup* cutoffI_0;
    uint threadgroup* survivors_0;
};

[[kernel]] void sample_filter(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], SampleParams_0 constant* p_1 [[buffer(4)]], float device* logits_1 [[buffer(0)]], uint4 device* info_1 [[buffer(3)]], float device* sortedV_1 [[buffer(1)]], uint device* sortedI_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->logits_0 = logits_1;
    (&kernelContext_0)->info_0 = info_1;
    (&kernelContext_0)->sortedV_0 = sortedV_1;
    (&kernelContext_0)->sortedI_0 = sortedI_1;
    threadgroup uint windowBound_1;
    (&kernelContext_0)->windowBound_0 = &windowBound_1;
    threadgroup array<float, int(1024)> sv_1;
    (&kernelContext_0)->sv_0 = &sv_1;
    threadgroup array<uint, int(1024)> si_1;
    (&kernelContext_0)->si_0 = &si_1;
    threadgroup array<uint, int(1024)> ranks_1;
    (&kernelContext_0)->ranks_0 = &ranks_1;
    threadgroup float cutoffV_1;
    (&kernelContext_0)->cutoffV_0 = &cutoffV_1;
    threadgroup uint cutoffI_1;
    (&kernelContext_0)->cutoffI_0 = &cutoffI_1;
    threadgroup uint survivors_1;
    (&kernelContext_0)->survivors_0 = &survivors_1;
    uint t_0 = local_0.x;
    uint k_0 = p_1->head_0;
    uint _S6 = group_0.x;
    uint j_0 = _S6 * 1024U + t_0;
    uint _S7 = group_0.y;
    uint chunk_0 = _S7 * p_1->chunks_0 + _S6;
    float v_1;
    if(j_0 < (p_1->vocab_0))
    {
        v_1 = sanitize_0(*((&kernelContext_0)->logits_0+(_S7 * (&kernelContext_0)->p_0->ldl_0 + j_0)));
    }
    else
    {
        v_1 = dg_negative_infinity_0();
    }
    uint idx_0;
    if(j_0 < (p_1->vocab_0))
    {
        idx_0 = j_0;
    }
    else
    {
        idx_0 = 4294967295U;
    }
    bool _S8 = t_0 == 0U;
    if(_S8)
    {
        *(&kernelContext_0)->windowBound_0 = key_0((as_type<float>(((*((&kernelContext_0)->info_0+_S7 * p_1->chunks_0)).z))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slots_0;
    uint size_0;
    if(t_0 < 128U)
    {
        float _S9 = dg_negative_infinity_0();
        float minBound_0;
        if(t_0 < (p_1->vocab_0))
        {
            minBound_0 = sanitize_0(*((&kernelContext_0)->logits_0+(_S7 * (&kernelContext_0)->p_0->ldl_0 + t_0)));
        }
        else
        {
            minBound_0 = _S9;
        }
        if(t_0 < (p_1->vocab_0))
        {
            slots_0 = t_0;
        }
        else
        {
            slots_0 = 4294967295U;
        }
        float _S10 = minBound_0;
        uint _S11 = slots_0;
        slots_0 = 1U + t_0;
        minBound_0 = _S9;
        float best_0 = _S10;
        size_0 = _S11;
        for(;;)
        {
            if(slots_0 < (p_1->chunks_0))
            {
            }
            else
            {
                break;
            }
            uint4 device* _S12 = (&kernelContext_0)->info_0+(_S7 * p_1->chunks_0 + slots_0);
            float _S13 = max(minBound_0, (as_type<float>(((*_S12).z))));
            float _S14 = (as_type<float>(((*_S12).x)));
            uint _S15 = (*_S12).y;
            if(before_0(_S14, _S15, best_0, size_0))
            {
                best_0 = _S14;
                size_0 = _S15;
            }
            slots_0 = slots_0 + 128U;
            minBound_0 = _S13;
        }
        uint _S16 = atomic_fetch_max_explicit(((atomic_uint threadgroup*)((&kernelContext_0)->windowBound_0)), key_0(minBound_0), memory_order_relaxed);
        (*(&kernelContext_0)->sv_0)[t_0] = best_0;
        (*(&kernelContext_0)->si_0)[t_0] = size_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint bucket_0 = t_0 / 8U;
    uint slice_0 = t_0 % 8U;
    float myV_0 = (*(&kernelContext_0)->sv_0)[bucket_0];
    uint myI_0 = (*(&kernelContext_0)->si_0)[bucket_0];
    if(before_0((*(&kernelContext_0)->sv_0)[slice_0], (*(&kernelContext_0)->si_0)[slice_0], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint other_0 = 8U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_0], (*(&kernelContext_0)->si_0)[other_0], myV_0, myI_0))
    {
        size_0 = 1U;
    }
    else
    {
        size_0 = 0U;
    }
    uint rank_0 = slots_0 + size_0;
    uint other_1 = 16U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_1], (*(&kernelContext_0)->si_0)[other_1], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_1 = rank_0 + slots_0;
    uint other_2 = 24U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_2], (*(&kernelContext_0)->si_0)[other_2], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_2 = rank_1 + slots_0;
    uint other_3 = 32U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_3], (*(&kernelContext_0)->si_0)[other_3], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_3 = rank_2 + slots_0;
    uint other_4 = 40U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_4], (*(&kernelContext_0)->si_0)[other_4], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_4 = rank_3 + slots_0;
    uint other_5 = 48U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_5], (*(&kernelContext_0)->si_0)[other_5], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_5 = rank_4 + slots_0;
    uint other_6 = 56U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_6], (*(&kernelContext_0)->si_0)[other_6], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_6 = rank_5 + slots_0;
    uint other_7 = 64U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_7], (*(&kernelContext_0)->si_0)[other_7], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_7 = rank_6 + slots_0;
    uint other_8 = 72U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_8], (*(&kernelContext_0)->si_0)[other_8], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_8 = rank_7 + slots_0;
    uint other_9 = 80U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_9], (*(&kernelContext_0)->si_0)[other_9], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_9 = rank_8 + slots_0;
    uint other_10 = 88U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_10], (*(&kernelContext_0)->si_0)[other_10], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_10 = rank_9 + slots_0;
    uint other_11 = 96U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_11], (*(&kernelContext_0)->si_0)[other_11], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_11 = rank_10 + slots_0;
    uint other_12 = 104U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_12], (*(&kernelContext_0)->si_0)[other_12], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_12 = rank_11 + slots_0;
    uint other_13 = 112U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_13], (*(&kernelContext_0)->si_0)[other_13], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_13 = rank_12 + slots_0;
    uint other_14 = 120U + slice_0;
    if(before_0((*(&kernelContext_0)->sv_0)[other_14], (*(&kernelContext_0)->si_0)[other_14], myV_0, myI_0))
    {
        slots_0 = 1U;
    }
    else
    {
        slots_0 = 0U;
    }
    uint rank_14 = rank_13 + slots_0;
    (*(&kernelContext_0)->ranks_0)[t_0] = rank_14;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(slice_0 < 4U)
    {
        (*(&kernelContext_0)->ranks_0)[t_0] = (*(&kernelContext_0)->ranks_0)[t_0] + (*(&kernelContext_0)->ranks_0)[t_0 + 4U];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(slice_0 < 2U)
    {
        (*(&kernelContext_0)->ranks_0)[t_0] = (*(&kernelContext_0)->ranks_0)[t_0] + (*(&kernelContext_0)->ranks_0)[t_0 + 2U];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(slice_0 < 1U)
    {
        (*(&kernelContext_0)->ranks_0)[t_0] = (*(&kernelContext_0)->ranks_0)[t_0] + (*(&kernelContext_0)->ranks_0)[t_0 + 1U];
    }
    bool _S17;
    if(slice_0 == 0U)
    {
        _S17 = ((*(&kernelContext_0)->ranks_0)[t_0]) == (k_0 - 1U);
    }
    else
    {
        _S17 = false;
    }
    if(_S17)
    {
        _S17 = myI_0 != 4294967295U;
    }
    else
    {
        _S17 = false;
    }
    if(_S17)
    {
        *(&kernelContext_0)->cutoffV_0 = myV_0;
        *(&kernelContext_0)->cutoffI_0 = myI_0;
    }
    if(_S8)
    {
        *(&kernelContext_0)->survivors_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(idx_0 != 4294967295U)
    {
        _S17 = (key_0(v_1)) >= (*(&kernelContext_0)->windowBound_0);
    }
    else
    {
        _S17 = false;
    }
    if(_S17)
    {
        _S17 = !before_0(*(&kernelContext_0)->cutoffV_0, *(&kernelContext_0)->cutoffI_0, v_1, idx_0);
    }
    else
    {
        _S17 = false;
    }
    if(_S17)
    {
        uint slot_0 = atomic_fetch_add_explicit(((atomic_uint threadgroup*)((&kernelContext_0)->survivors_0)), 1U, memory_order_relaxed);
        (*(&kernelContext_0)->sv_0)[slot_0] = v_1;
        (*(&kernelContext_0)->si_0)[slot_0] = idx_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_0 = *(&kernelContext_0)->survivors_0;
    slots_0 = 1U;
    for(;;)
    {
        if(slots_0 < n_0)
        {
        }
        else
        {
            break;
        }
        slots_0 = slots_0 * 2U;
    }
    if(t_0 >= n_0)
    {
        _S17 = t_0 < slots_0;
    }
    else
    {
        _S17 = false;
    }
    if(_S17)
    {
        (*(&kernelContext_0)->sv_0)[t_0] = dg_negative_infinity_0();
        (*(&kernelContext_0)->si_0)[t_0] = 4294967295U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    size_0 = 2U;
    for(;;)
    {
        if(size_0 <= slots_0)
        {
        }
        else
        {
            break;
        }
        uint stride_1 = size_0 / 2U;
        for(;;)
        {
            if(stride_1 > 0U)
            {
            }
            else
            {
                break;
            }
            uint partner_0 = t_0 ^ stride_1;
            if(t_0 < slots_0)
            {
                _S17 = partner_0 > t_0;
            }
            else
            {
                _S17 = false;
            }
            if(_S17)
            {
                float a_0 = (*(&kernelContext_0)->sv_0)[t_0];
                float b_0 = (*(&kernelContext_0)->sv_0)[partner_0];
                uint ai_0 = (*(&kernelContext_0)->si_0)[t_0];
                uint bi_0 = (*(&kernelContext_0)->si_0)[partner_0];
                bool swap_0;
                if((t_0 & size_0) == 0U)
                {
                    swap_0 = before_0(b_0, bi_0, a_0, ai_0);
                }
                else
                {
                    swap_0 = before_0(a_0, ai_0, b_0, bi_0);
                }
                if(swap_0)
                {
                    (*(&kernelContext_0)->sv_0)[t_0] = b_0;
                    (*(&kernelContext_0)->si_0)[t_0] = bi_0;
                    (*(&kernelContext_0)->sv_0)[partner_0] = a_0;
                    (*(&kernelContext_0)->si_0)[partner_0] = ai_0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            stride_1 = stride_1 / 2U;
        }
        size_0 = size_0 * 2U;
    }
    if(t_0 < k_0)
    {
        uint _S18 = chunk_0 * k_0 + t_0;
        float device* _S19 = (&kernelContext_0)->sortedV_0+_S18;
        bool _S20 = t_0 < n_0;
        if(_S20)
        {
            v_1 = (*(&kernelContext_0)->sv_0)[t_0];
        }
        else
        {
            v_1 = dg_negative_infinity_0();
        }
        *_S19 = v_1;
        uint device* _S21 = (&kernelContext_0)->sortedI_0+_S18;
        if(_S20)
        {
            idx_0 = (*(&kernelContext_0)->si_0)[t_0];
        }
        else
        {
            idx_0 = 4294967295U;
        }
        *_S21 = idx_0;
    }
    return;
}

