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

bool before_0(float va_0, uint ia_0, float vb_0, uint ib_0)
{
    bool _S4;
    if(va_0 > vb_0)
    {
        _S4 = true;
    }
    else
    {
        if(va_0 == vb_0)
        {
            _S4 = ia_0 < ib_0;
        }
        else
        {
            _S4 = false;
        }
    }
    return _S4;
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
    float device* sortedV_0;
    uint device* sortedI_0;
    uint4 device* info_0;
    uint _S5;
    uint _S6;
    uint threadgroup* irregular_0;
    array<float, int(1024)> threadgroup* sv_0;
    array<uint, int(1024)> threadgroup* si_0;
};

[[kernel]] void sample_shuffle(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], uint _builtinWaveLaneCount_0 [[threads_per_simdgroup]], uint _builtinWaveLaneIndex_0 [[thread_index_in_simdgroup]], SampleParams_0 constant* p_1 [[buffer(4)]], float device* logits_1 [[buffer(0)]], float device* sortedV_1 [[buffer(1)]], uint device* sortedI_1 [[buffer(2)]], uint4 device* info_1 [[buffer(3)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->logits_0 = logits_1;
    (&kernelContext_0)->sortedV_0 = sortedV_1;
    (&kernelContext_0)->sortedI_0 = sortedI_1;
    (&kernelContext_0)->info_0 = info_1;
    threadgroup uint irregular_1;
    (&kernelContext_0)->irregular_0 = &irregular_1;
    threadgroup array<float, int(1024)> sv_1;
    (&kernelContext_0)->sv_0 = &sv_1;
    threadgroup array<uint, int(1024)> si_1;
    (&kernelContext_0)->si_0 = &si_1;
    (&kernelContext_0)->_S5 = _builtinWaveLaneCount_0;
    (&kernelContext_0)->_S6 = _builtinWaveLaneIndex_0;
    uint t_0 = local_0.x;
    uint _S7 = group_0.y;
    uint _S8 = group_0.x;
    uint chunk_0 = _S7 * p_1->chunks_0 + _S8;
    uint j_0 = _S8 * 1024U + t_0;
    float v_0;
    if(j_0 < (p_1->vocab_0))
    {
        v_0 = sanitize_0(*((&kernelContext_0)->logits_0+(_S7 * (&kernelContext_0)->p_0->ldl_0 + j_0)));
    }
    else
    {
        v_0 = dg_negative_infinity_0();
    }
    uint size_0;
    if(j_0 < (p_1->vocab_0))
    {
        size_0 = j_0;
    }
    else
    {
        size_0 = 4294967295U;
    }
    uint _S9 = (&kernelContext_0)->_S5;
    uint _S10 = (&kernelContext_0)->_S6;
    bool _S11 = t_0 == 0U;
    if(_S11)
    {
        *(&kernelContext_0)->irregular_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint first_0 = (simd_shuffle((t_0), ushort((int(0)))));
    uint active_0 = simd_sum(1U);
    bool swap_0;
    if(active_0 != _S9)
    {
        swap_0 = true;
    }
    else
    {
        swap_0 = t_0 != (first_0 + _S10);
    }
    if(swap_0)
    {
        swap_0 = true;
    }
    else
    {
        uint _S12 = t_0 % _S9;
        swap_0 = _S10 != _S12;
    }
    if(swap_0)
    {
        uint _S13 = atomic_fetch_or_explicit(((atomic_uint threadgroup*)((&kernelContext_0)->irregular_0)), 1U, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S14;
    if((*(&kernelContext_0)->irregular_0) == 0U)
    {
        _S14 = _S9;
    }
    else
    {
        _S14 = 1U;
    }
    uint _S15 = size_0;
    size_0 = 2U;
    uint idx_0 = _S15;
    for(;;)
    {
        if(size_0 <= 256U)
        {
        }
        else
        {
            break;
        }
        uint stride_1 = size_0 / 2U;
        float v_1 = v_0;
        uint idx_1 = idx_0;
        for(;;)
        {
            if(stride_1 > 0U)
            {
            }
            else
            {
                break;
            }
            float other_0;
            uint otherIdx_0;
            if(stride_1 < _S14)
            {
                int _S16 = int(_S10 ^ stride_1);
                float _S17 = (simd_shuffle((v_1), ushort((_S16))));
                uint _S18 = (simd_shuffle((idx_1), ushort((_S16))));
                other_0 = _S17;
                otherIdx_0 = _S18;
            }
            else
            {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                (*(&kernelContext_0)->sv_0)[t_0] = v_1;
                (*(&kernelContext_0)->si_0)[t_0] = idx_1;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint _S19 = t_0 ^ stride_1;
                uint _S20 = (*(&kernelContext_0)->si_0)[_S19];
                other_0 = (*(&kernelContext_0)->sv_0)[_S19];
                otherIdx_0 = _S20;
            }
            if(((t_0 & stride_1) == 0U) == (((t_0 % 256U) & size_0) == 0U))
            {
                swap_0 = before_0(other_0, otherIdx_0, v_1, idx_1);
            }
            else
            {
                swap_0 = before_0(v_1, idx_1, other_0, otherIdx_0);
            }
            if(swap_0)
            {
                v_1 = other_0;
                idx_1 = otherIdx_0;
            }
            stride_1 = stride_1 / 2U;
        }
        size_0 = size_0 * 2U;
        v_0 = v_1;
        idx_0 = idx_1;
    }
    uint _S21 = chunk_0 * 1024U + t_0;
    *((&kernelContext_0)->sortedV_0+_S21) = v_0;
    *((&kernelContext_0)->sortedI_0+_S21) = idx_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((t_0 % 256U) == 0U)
    {
        (*(&kernelContext_0)->sv_0)[t_0 / 256U] = v_0;
        (*(&kernelContext_0)->si_0)[t_0 / 256U] = idx_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S11)
    {
        if(before_0((*(&kernelContext_0)->sv_0)[1U], (*(&kernelContext_0)->si_0)[1U], v_0, idx_0))
        {
            uint _S22 = (*(&kernelContext_0)->si_0)[1U];
            v_0 = (*(&kernelContext_0)->sv_0)[1U];
            idx_0 = _S22;
        }
        if(before_0((*(&kernelContext_0)->sv_0)[2U], (*(&kernelContext_0)->si_0)[2U], v_0, idx_0))
        {
            uint _S23 = (*(&kernelContext_0)->si_0)[2U];
            v_0 = (*(&kernelContext_0)->sv_0)[2U];
            idx_0 = _S23;
        }
        if(before_0((*(&kernelContext_0)->sv_0)[3U], (*(&kernelContext_0)->si_0)[3U], v_0, idx_0))
        {
            uint _S24 = (*(&kernelContext_0)->si_0)[3U];
            v_0 = (*(&kernelContext_0)->sv_0)[3U];
            idx_0 = _S24;
        }
        *((&kernelContext_0)->info_0+chunk_0) = uint4((as_type<uint>((v_0))), idx_0, 0U, 0U);
    }
    return;
}

