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
    uint4 device* info_0;
    array<float, int(1024)> threadgroup* sv_0;
    array<uint, int(1024)> threadgroup* si_0;
    array<float, int(1024)> threadgroup* windowMin_0;
};

[[kernel]] void sample_bounds(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], SampleParams_0 constant* p_1 [[buffer(2)]], float device* logits_1 [[buffer(0)]], uint4 device* info_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->logits_0 = logits_1;
    (&kernelContext_0)->info_0 = info_1;
    threadgroup array<float, int(1024)> sv_1;
    (&kernelContext_0)->sv_0 = &sv_1;
    threadgroup array<uint, int(1024)> si_1;
    (&kernelContext_0)->si_0 = &si_1;
    threadgroup array<float, int(1024)> windowMin_1;
    (&kernelContext_0)->windowMin_0 = &windowMin_1;
    uint c_0 = group_0.x;
    uint row_0 = group_0.y;
    uint t_0 = local_0.x;
    uint j_0 = c_0 * 1024U + t_0;
    bool valid_0 = j_0 < (p_1->vocab_0);
    float v_0;
    if(valid_0)
    {
        v_0 = sanitize_0(*((&kernelContext_0)->logits_0+(row_0 * (&kernelContext_0)->p_0->ldl_0 + j_0)));
    }
    else
    {
        v_0 = dg_negative_infinity_0();
    }
    uint idx_0;
    if(valid_0)
    {
        idx_0 = j_0;
    }
    else
    {
        idx_0 = 4294967295U;
    }
    (*(&kernelContext_0)->sv_0)[t_0] = v_0;
    (*(&kernelContext_0)->si_0)[t_0] = idx_0;
    (*(&kernelContext_0)->windowMin_0)[t_0] = v_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S5 = t_0 % 64U;
    if(_S5 < 32U)
    {
        (*(&kernelContext_0)->windowMin_0)[t_0] = min((*(&kernelContext_0)->windowMin_0)[t_0], (*(&kernelContext_0)->windowMin_0)[t_0 + 32U]);
    }
    bool _S6;
    if(t_0 < 512U)
    {
        uint _S7 = t_0 + 512U;
        _S6 = before_0((*(&kernelContext_0)->sv_0)[_S7], (*(&kernelContext_0)->si_0)[_S7], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        uint _S8 = t_0 + 512U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S8];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S8];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S5 < 16U)
    {
        (*(&kernelContext_0)->windowMin_0)[t_0] = min((*(&kernelContext_0)->windowMin_0)[t_0], (*(&kernelContext_0)->windowMin_0)[t_0 + 16U]);
    }
    if(t_0 < 256U)
    {
        uint _S9 = t_0 + 256U;
        _S6 = before_0((*(&kernelContext_0)->sv_0)[_S9], (*(&kernelContext_0)->si_0)[_S9], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        uint _S10 = t_0 + 256U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S10];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S10];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S5 < 8U)
    {
        (*(&kernelContext_0)->windowMin_0)[t_0] = min((*(&kernelContext_0)->windowMin_0)[t_0], (*(&kernelContext_0)->windowMin_0)[t_0 + 8U]);
    }
    if(t_0 < 128U)
    {
        uint _S11 = t_0 + 128U;
        _S6 = before_0((*(&kernelContext_0)->sv_0)[_S11], (*(&kernelContext_0)->si_0)[_S11], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        uint _S12 = t_0 + 128U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S12];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S12];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S5 < 4U)
    {
        (*(&kernelContext_0)->windowMin_0)[t_0] = min((*(&kernelContext_0)->windowMin_0)[t_0], (*(&kernelContext_0)->windowMin_0)[t_0 + 4U]);
    }
    if(t_0 < 64U)
    {
        uint _S13 = t_0 + 64U;
        _S6 = before_0((*(&kernelContext_0)->sv_0)[_S13], (*(&kernelContext_0)->si_0)[_S13], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        uint _S14 = t_0 + 64U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S14];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S14];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S5 < 2U)
    {
        (*(&kernelContext_0)->windowMin_0)[t_0] = min((*(&kernelContext_0)->windowMin_0)[t_0], (*(&kernelContext_0)->windowMin_0)[t_0 + 2U]);
    }
    if(t_0 < 32U)
    {
        uint _S15 = t_0 + 32U;
        _S6 = before_0((*(&kernelContext_0)->sv_0)[_S15], (*(&kernelContext_0)->si_0)[_S15], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        uint _S16 = t_0 + 32U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S16];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S16];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S5 < 1U)
    {
        (*(&kernelContext_0)->windowMin_0)[t_0] = min((*(&kernelContext_0)->windowMin_0)[t_0], (*(&kernelContext_0)->windowMin_0)[t_0 + 1U]);
    }
    if(t_0 < 16U)
    {
        uint _S17 = t_0 + 16U;
        _S6 = before_0((*(&kernelContext_0)->sv_0)[_S17], (*(&kernelContext_0)->si_0)[_S17], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        uint _S18 = t_0 + 16U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S18];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S18];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S19 = t_0 < 8U;
    if(_S19)
    {
        (*(&kernelContext_0)->windowMin_0)[t_0 * 64U] = max((*(&kernelContext_0)->windowMin_0)[t_0 * 64U], (*(&kernelContext_0)->windowMin_0)[(t_0 + 8U) * 64U]);
    }
    if(_S19)
    {
        uint _S20 = t_0 + 8U;
        _S6 = before_0((*(&kernelContext_0)->sv_0)[_S20], (*(&kernelContext_0)->si_0)[_S20], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        uint _S21 = t_0 + 8U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S21];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S21];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S22 = t_0 < 4U;
    if(_S22)
    {
        (*(&kernelContext_0)->windowMin_0)[t_0 * 64U] = max((*(&kernelContext_0)->windowMin_0)[t_0 * 64U], (*(&kernelContext_0)->windowMin_0)[(t_0 + 4U) * 64U]);
    }
    if(_S22)
    {
        uint _S23 = t_0 + 4U;
        _S6 = before_0((*(&kernelContext_0)->sv_0)[_S23], (*(&kernelContext_0)->si_0)[_S23], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        uint _S24 = t_0 + 4U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S24];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S24];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S25 = t_0 < 2U;
    if(_S25)
    {
        (*(&kernelContext_0)->windowMin_0)[t_0 * 64U] = max((*(&kernelContext_0)->windowMin_0)[t_0 * 64U], (*(&kernelContext_0)->windowMin_0)[(t_0 + 2U) * 64U]);
    }
    if(_S25)
    {
        uint _S26 = t_0 + 2U;
        _S6 = before_0((*(&kernelContext_0)->sv_0)[_S26], (*(&kernelContext_0)->si_0)[_S26], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        uint _S27 = t_0 + 2U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S27];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S27];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S28 = t_0 < 1U;
    if(_S28)
    {
        (*(&kernelContext_0)->windowMin_0)[t_0 * 64U] = max((*(&kernelContext_0)->windowMin_0)[t_0 * 64U], (*(&kernelContext_0)->windowMin_0)[(t_0 + 1U) * 64U]);
    }
    if(_S28)
    {
        uint _S29 = t_0 + 1U;
        _S6 = before_0((*(&kernelContext_0)->sv_0)[_S29], (*(&kernelContext_0)->si_0)[_S29], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        uint _S30 = t_0 + 1U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S30];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S30];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 == 0U)
    {
        *((&kernelContext_0)->info_0+(row_0 * (&kernelContext_0)->p_0->chunks_0 + c_0)) = uint4((as_type<uint>(((*(&kernelContext_0)->sv_0)[int(0)]))), (*(&kernelContext_0)->si_0)[int(0)], (as_type<uint>(((*(&kernelContext_0)->windowMin_0)[int(0)]))), 0U);
    }
    return;
}

