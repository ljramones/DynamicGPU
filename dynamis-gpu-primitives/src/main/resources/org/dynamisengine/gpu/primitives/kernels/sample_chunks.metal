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

float dg_op_add_0(float a_0, float b_0)
{
    return a_0 + b_0;
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
    array<float, int(1024)> threadgroup* sv_0;
    array<uint, int(1024)> threadgroup* si_0;
    float threadgroup* gMax_0;
    uint threadgroup* gIdx_0;
};

[[kernel]] void sample_chunks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], SampleParams_0 constant* p_1 [[buffer(4)]], float device* logits_1 [[buffer(0)]], uint4 device* info_1 [[buffer(3)]], float device* sortedV_1 [[buffer(1)]], uint device* sortedI_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->logits_0 = logits_1;
    (&kernelContext_0)->info_0 = info_1;
    (&kernelContext_0)->sortedV_0 = sortedV_1;
    (&kernelContext_0)->sortedI_0 = sortedI_1;
    threadgroup array<float, int(1024)> sv_1;
    (&kernelContext_0)->sv_0 = &sv_1;
    threadgroup array<uint, int(1024)> si_1;
    (&kernelContext_0)->si_0 = &si_1;
    threadgroup float gMax_1;
    (&kernelContext_0)->gMax_0 = &gMax_1;
    threadgroup uint gIdx_1;
    (&kernelContext_0)->gIdx_0 = &gIdx_1;
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S5 = t_0 < 512U;
    bool swap_0;
    if(_S5)
    {
        uint _S6 = t_0 + 512U;
        swap_0 = before_0((*(&kernelContext_0)->sv_0)[_S6], (*(&kernelContext_0)->si_0)[_S6], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        swap_0 = false;
    }
    if(swap_0)
    {
        uint _S7 = t_0 + 512U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S7];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S7];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S8 = t_0 < 256U;
    if(_S8)
    {
        uint _S9 = t_0 + 256U;
        swap_0 = before_0((*(&kernelContext_0)->sv_0)[_S9], (*(&kernelContext_0)->si_0)[_S9], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        swap_0 = false;
    }
    if(swap_0)
    {
        uint _S10 = t_0 + 256U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S10];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S10];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S11 = t_0 < 128U;
    if(_S11)
    {
        uint _S12 = t_0 + 128U;
        swap_0 = before_0((*(&kernelContext_0)->sv_0)[_S12], (*(&kernelContext_0)->si_0)[_S12], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        swap_0 = false;
    }
    if(swap_0)
    {
        uint _S13 = t_0 + 128U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S13];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S13];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S14 = t_0 < 64U;
    if(_S14)
    {
        uint _S15 = t_0 + 64U;
        swap_0 = before_0((*(&kernelContext_0)->sv_0)[_S15], (*(&kernelContext_0)->si_0)[_S15], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        swap_0 = false;
    }
    if(swap_0)
    {
        uint _S16 = t_0 + 64U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S16];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S16];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S17 = t_0 < 32U;
    if(_S17)
    {
        uint _S18 = t_0 + 32U;
        swap_0 = before_0((*(&kernelContext_0)->sv_0)[_S18], (*(&kernelContext_0)->si_0)[_S18], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        swap_0 = false;
    }
    if(swap_0)
    {
        uint _S19 = t_0 + 32U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S19];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S19];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S20 = t_0 < 16U;
    if(_S20)
    {
        uint _S21 = t_0 + 16U;
        swap_0 = before_0((*(&kernelContext_0)->sv_0)[_S21], (*(&kernelContext_0)->si_0)[_S21], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        swap_0 = false;
    }
    if(swap_0)
    {
        uint _S22 = t_0 + 16U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S22];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S22];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S23 = t_0 < 8U;
    if(_S23)
    {
        uint _S24 = t_0 + 8U;
        swap_0 = before_0((*(&kernelContext_0)->sv_0)[_S24], (*(&kernelContext_0)->si_0)[_S24], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        swap_0 = false;
    }
    if(swap_0)
    {
        uint _S25 = t_0 + 8U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S25];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S25];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S26 = t_0 < 4U;
    if(_S26)
    {
        uint _S27 = t_0 + 4U;
        swap_0 = before_0((*(&kernelContext_0)->sv_0)[_S27], (*(&kernelContext_0)->si_0)[_S27], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        swap_0 = false;
    }
    if(swap_0)
    {
        uint _S28 = t_0 + 4U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S28];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S28];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S29 = t_0 < 2U;
    if(_S29)
    {
        uint _S30 = t_0 + 2U;
        swap_0 = before_0((*(&kernelContext_0)->sv_0)[_S30], (*(&kernelContext_0)->si_0)[_S30], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        swap_0 = false;
    }
    if(swap_0)
    {
        uint _S31 = t_0 + 2U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S31];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S31];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S32 = t_0 < 1U;
    if(_S32)
    {
        uint _S33 = t_0 + 1U;
        swap_0 = before_0((*(&kernelContext_0)->sv_0)[_S33], (*(&kernelContext_0)->si_0)[_S33], (*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->si_0)[t_0]);
    }
    else
    {
        swap_0 = false;
    }
    if(swap_0)
    {
        uint _S34 = t_0 + 1U;
        (*(&kernelContext_0)->sv_0)[t_0] = (*(&kernelContext_0)->sv_0)[_S34];
        (*(&kernelContext_0)->si_0)[t_0] = (*(&kernelContext_0)->si_0)[_S34];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S35 = t_0 == 0U;
    if(_S35)
    {
        *(&kernelContext_0)->gMax_0 = (*(&kernelContext_0)->sv_0)[int(0)];
        *(&kernelContext_0)->gIdx_0 = (*(&kernelContext_0)->si_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx_0 = *(&kernelContext_0)->gMax_0;
    uint arg_0 = *(&kernelContext_0)->gIdx_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(valid_0)
    {
        swap_0 = mx_0 != (dg_negative_infinity_0());
    }
    else
    {
        swap_0 = false;
    }
    if(swap_0)
    {
        swap_0 = ((&kernelContext_0)->p_0->invT_0) > 0.0;
    }
    else
    {
        swap_0 = false;
    }
    float _S36;
    if(swap_0)
    {
        _S36 = exp((v_0 - mx_0) * (&kernelContext_0)->p_0->invT_0);
    }
    else
    {
        _S36 = 0.0;
    }
    (*(&kernelContext_0)->sv_0)[t_0] = _S36;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S5)
    {
        (*(&kernelContext_0)->sv_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->sv_0)[t_0 + 512U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S8)
    {
        (*(&kernelContext_0)->sv_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->sv_0)[t_0 + 256U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S11)
    {
        (*(&kernelContext_0)->sv_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->sv_0)[t_0 + 128U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S14)
    {
        (*(&kernelContext_0)->sv_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->sv_0)[t_0 + 64U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S17)
    {
        (*(&kernelContext_0)->sv_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->sv_0)[t_0 + 32U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S20)
    {
        (*(&kernelContext_0)->sv_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->sv_0)[t_0 + 16U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S23)
    {
        (*(&kernelContext_0)->sv_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->sv_0)[t_0 + 8U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S26)
    {
        (*(&kernelContext_0)->sv_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->sv_0)[t_0 + 4U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S29)
    {
        (*(&kernelContext_0)->sv_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->sv_0)[t_0 + 2U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S32)
    {
        (*(&kernelContext_0)->sv_0)[t_0] = dg_op_add_0((*(&kernelContext_0)->sv_0)[t_0], (*(&kernelContext_0)->sv_0)[t_0 + 1U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S35)
    {
        *((&kernelContext_0)->info_0+(row_0 * (&kernelContext_0)->p_0->chunks_0 + c_0)) = uint4((as_type<uint>((mx_0))), arg_0, (as_type<uint>(((*(&kernelContext_0)->sv_0)[int(0)]))), 0U);
    }
    if(((&kernelContext_0)->p_0->restricted_0) == 0U)
    {
        return;
    }
    uint stride_1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*(&kernelContext_0)->sv_0)[t_0] = v_0;
    (*(&kernelContext_0)->si_0)[t_0] = idx_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint size_0 = 2U;
    for(;;)
    {
        if(size_0 <= 256U)
        {
        }
        else
        {
            break;
        }
        stride_1 = size_0 / 2U;
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
            if(partner_0 > t_0)
            {
                float va_1 = (*(&kernelContext_0)->sv_0)[t_0];
                float vb_1 = (*(&kernelContext_0)->sv_0)[partner_0];
                uint ia_1 = (*(&kernelContext_0)->si_0)[t_0];
                uint ib_1 = (*(&kernelContext_0)->si_0)[partner_0];
                if((t_0 & size_0) == 0U)
                {
                    swap_0 = before_0(vb_1, ib_1, va_1, ia_1);
                }
                else
                {
                    swap_0 = before_0(va_1, ia_1, vb_1, ib_1);
                }
                if(swap_0)
                {
                    (*(&kernelContext_0)->sv_0)[t_0] = vb_1;
                    (*(&kernelContext_0)->sv_0)[partner_0] = va_1;
                    (*(&kernelContext_0)->si_0)[t_0] = ib_1;
                    (*(&kernelContext_0)->si_0)[partner_0] = ia_1;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            stride_1 = stride_1 / 2U;
        }
        size_0 = size_0 * 2U;
    }
    uint run_0 = t_0 / 256U;
    uint q_0 = t_0 % 256U;
    if((run_0 & 1U) == 0U)
    {
        stride_1 = t_0;
    }
    else
    {
        stride_1 = run_0 * 256U + (255U - q_0);
    }
    uint at_0 = (row_0 * (&kernelContext_0)->p_0->chunks_0 + c_0) * 1024U + t_0;
    *((&kernelContext_0)->sortedV_0+at_0) = (*(&kernelContext_0)->sv_0)[stride_1];
    *((&kernelContext_0)->sortedI_0+at_0) = (*(&kernelContext_0)->si_0)[stride_1];
    return;
}

