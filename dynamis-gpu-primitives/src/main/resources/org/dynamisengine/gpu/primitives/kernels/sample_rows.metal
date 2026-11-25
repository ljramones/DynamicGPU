#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint dg_linear_group_0(uint3 group_0, uint gridX_0)
{
    return group_0.y * gridX_0 + group_0.x;
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

bool before_0(float va_0, uint ia_0, float vb_0, uint ib_0)
{
    bool _S3;
    if(va_0 > vb_0)
    {
        _S3 = true;
    }
    else
    {
        if(va_0 == vb_0)
        {
            _S3 = ia_0 < ib_0;
        }
        else
        {
            _S3 = false;
        }
    }
    return _S3;
}

float dg_op_add_0(float a_0, float b_0)
{
    return a_0 + b_0;
}

uint fmix_0(uint h_0)
{
    uint _S4 = (h_0 ^ (h_0 >> 16U)) * 2246822507U;
    uint _S5 = (_S4 ^ (_S4 >> 13U)) * 3266489909U;
    return _S5 ^ (_S5 >> 16U);
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
    uint gridX_1;
};

struct KernelContext_0
{
    SampleParams_0 constant* p_0;
    float device* logits_0;
    uint device* tokens_0;
    array<float, int(1024)> threadgroup* sv_0;
    array<uint, int(1024)> threadgroup* si_0;
    float threadgroup* gMax_0;
    uint threadgroup* gIdx_0;
    float threadgroup* gTotal_0;
    uint threadgroup* gFound_0;
    uint threadgroup* gPrefix_0;
    uint threadgroup* gRemain_0;
    array<uint, int(256)> threadgroup* hist_0;
    float threadgroup* gU_0;
};

float rand01_0(uint row_0, KernelContext_0 thread* kernelContext_0)
{
    return float((fmix_0((kernelContext_0->p_0->seedLo_0) ^ (fmix_0((kernelContext_0->p_0->seedHi_0) ^ (fmix_0((row_0 * 2654435769U) ^ (fmix_0(kernelContext_0->p_0->step_0 + 1663821227U)))))))) >> 8U) * (as_type<float>((864026624U)));
}

uint key_0(float l_0)
{
    uint f_0 = (as_type<uint>((l_0)));
    uint _S6;
    if(dg_is_nan_0(l_0))
    {
        _S6 = 0U;
    }
    else
    {
        if((f_0 & 2147483648U) != 0U)
        {
            _S6 = ~f_0;
        }
        else
        {
            _S6 = f_0 | 2147483648U;
        }
    }
    return _S6;
}

uint dg_op_addu_0(uint a_1, uint b_1)
{
    return a_1 + b_1;
}

[[kernel]] void sample_rows(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], SampleParams_0 constant* p_1 [[buffer(2)]], float device* logits_1 [[buffer(0)]], uint device* tokens_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->p_0 = p_1;
    (&kernelContext_1)->logits_0 = logits_1;
    (&kernelContext_1)->tokens_0 = tokens_1;
    threadgroup array<float, int(1024)> sv_1;
    (&kernelContext_1)->sv_0 = &sv_1;
    threadgroup array<uint, int(1024)> si_1;
    (&kernelContext_1)->si_0 = &si_1;
    threadgroup float gMax_1;
    (&kernelContext_1)->gMax_0 = &gMax_1;
    threadgroup uint gIdx_1;
    (&kernelContext_1)->gIdx_0 = &gIdx_1;
    threadgroup float gTotal_1;
    (&kernelContext_1)->gTotal_0 = &gTotal_1;
    threadgroup uint gFound_1;
    (&kernelContext_1)->gFound_0 = &gFound_1;
    threadgroup uint gPrefix_1;
    (&kernelContext_1)->gPrefix_0 = &gPrefix_1;
    threadgroup uint gRemain_1;
    (&kernelContext_1)->gRemain_0 = &gRemain_1;
    threadgroup array<uint, int(256)> hist_1;
    (&kernelContext_1)->hist_0 = &hist_1;
    threadgroup float gU_1;
    (&kernelContext_1)->gU_0 = &gU_1;
    uint row_1 = dg_linear_group_0(group_1, p_1->gridX_1);
    if(row_1 >= (p_1->rows_0))
    {
        return;
    }
    bool restricted_0;
    uint t_0 = local_0.x;
    uint _S7 = row_1 * (&kernelContext_1)->p_0->ldl_0;
    uint n_0 = (&kernelContext_1)->p_0->vocab_0;
    float _S8 = dg_negative_infinity_0();
    uint j_0 = t_0;
    uint bi_0 = 4294967295U;
    float bv_0 = _S8;
    for(;;)
    {
        if(j_0 < n_0)
        {
        }
        else
        {
            break;
        }
        float device* _S9 = (&kernelContext_1)->logits_0+(_S7 + j_0);
        float l_1 = *_S9;
        if(!dg_is_nan_0(*_S9))
        {
            if(bi_0 == 4294967295U)
            {
                restricted_0 = true;
            }
            else
            {
                restricted_0 = l_1 > bv_0;
            }
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            bi_0 = j_0;
            bv_0 = l_1;
        }
        j_0 = j_0 + 1024U;
    }
    (*(&kernelContext_1)->sv_0)[t_0] = bv_0;
    (*(&kernelContext_1)->si_0)[t_0] = bi_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 512U)
    {
        uint _S10 = t_0 + 512U;
        float ov_0 = (*(&kernelContext_1)->sv_0)[_S10];
        uint oi_0 = (*(&kernelContext_1)->si_0)[_S10];
        if(((*(&kernelContext_1)->si_0)[_S10]) != 4294967295U)
        {
            if(((*(&kernelContext_1)->si_0)[t_0]) == 4294967295U)
            {
                restricted_0 = true;
            }
            else
            {
                restricted_0 = before_0(ov_0, oi_0, (*(&kernelContext_1)->sv_0)[t_0], (*(&kernelContext_1)->si_0)[t_0]);
            }
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            (*(&kernelContext_1)->sv_0)[t_0] = ov_0;
            (*(&kernelContext_1)->si_0)[t_0] = oi_0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 256U)
    {
        uint _S11 = t_0 + 256U;
        float ov_1 = (*(&kernelContext_1)->sv_0)[_S11];
        uint oi_1 = (*(&kernelContext_1)->si_0)[_S11];
        if(((*(&kernelContext_1)->si_0)[_S11]) != 4294967295U)
        {
            if(((*(&kernelContext_1)->si_0)[t_0]) == 4294967295U)
            {
                restricted_0 = true;
            }
            else
            {
                restricted_0 = before_0(ov_1, oi_1, (*(&kernelContext_1)->sv_0)[t_0], (*(&kernelContext_1)->si_0)[t_0]);
            }
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            (*(&kernelContext_1)->sv_0)[t_0] = ov_1;
            (*(&kernelContext_1)->si_0)[t_0] = oi_1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 128U)
    {
        uint _S12 = t_0 + 128U;
        float ov_2 = (*(&kernelContext_1)->sv_0)[_S12];
        uint oi_2 = (*(&kernelContext_1)->si_0)[_S12];
        if(((*(&kernelContext_1)->si_0)[_S12]) != 4294967295U)
        {
            if(((*(&kernelContext_1)->si_0)[t_0]) == 4294967295U)
            {
                restricted_0 = true;
            }
            else
            {
                restricted_0 = before_0(ov_2, oi_2, (*(&kernelContext_1)->sv_0)[t_0], (*(&kernelContext_1)->si_0)[t_0]);
            }
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            (*(&kernelContext_1)->sv_0)[t_0] = ov_2;
            (*(&kernelContext_1)->si_0)[t_0] = oi_2;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 64U)
    {
        uint _S13 = t_0 + 64U;
        float ov_3 = (*(&kernelContext_1)->sv_0)[_S13];
        uint oi_3 = (*(&kernelContext_1)->si_0)[_S13];
        if(((*(&kernelContext_1)->si_0)[_S13]) != 4294967295U)
        {
            if(((*(&kernelContext_1)->si_0)[t_0]) == 4294967295U)
            {
                restricted_0 = true;
            }
            else
            {
                restricted_0 = before_0(ov_3, oi_3, (*(&kernelContext_1)->sv_0)[t_0], (*(&kernelContext_1)->si_0)[t_0]);
            }
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            (*(&kernelContext_1)->sv_0)[t_0] = ov_3;
            (*(&kernelContext_1)->si_0)[t_0] = oi_3;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 32U)
    {
        uint _S14 = t_0 + 32U;
        float ov_4 = (*(&kernelContext_1)->sv_0)[_S14];
        uint oi_4 = (*(&kernelContext_1)->si_0)[_S14];
        if(((*(&kernelContext_1)->si_0)[_S14]) != 4294967295U)
        {
            if(((*(&kernelContext_1)->si_0)[t_0]) == 4294967295U)
            {
                restricted_0 = true;
            }
            else
            {
                restricted_0 = before_0(ov_4, oi_4, (*(&kernelContext_1)->sv_0)[t_0], (*(&kernelContext_1)->si_0)[t_0]);
            }
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            (*(&kernelContext_1)->sv_0)[t_0] = ov_4;
            (*(&kernelContext_1)->si_0)[t_0] = oi_4;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 16U)
    {
        uint _S15 = t_0 + 16U;
        float ov_5 = (*(&kernelContext_1)->sv_0)[_S15];
        uint oi_5 = (*(&kernelContext_1)->si_0)[_S15];
        if(((*(&kernelContext_1)->si_0)[_S15]) != 4294967295U)
        {
            if(((*(&kernelContext_1)->si_0)[t_0]) == 4294967295U)
            {
                restricted_0 = true;
            }
            else
            {
                restricted_0 = before_0(ov_5, oi_5, (*(&kernelContext_1)->sv_0)[t_0], (*(&kernelContext_1)->si_0)[t_0]);
            }
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            (*(&kernelContext_1)->sv_0)[t_0] = ov_5;
            (*(&kernelContext_1)->si_0)[t_0] = oi_5;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 8U)
    {
        uint _S16 = t_0 + 8U;
        float ov_6 = (*(&kernelContext_1)->sv_0)[_S16];
        uint oi_6 = (*(&kernelContext_1)->si_0)[_S16];
        if(((*(&kernelContext_1)->si_0)[_S16]) != 4294967295U)
        {
            if(((*(&kernelContext_1)->si_0)[t_0]) == 4294967295U)
            {
                restricted_0 = true;
            }
            else
            {
                restricted_0 = before_0(ov_6, oi_6, (*(&kernelContext_1)->sv_0)[t_0], (*(&kernelContext_1)->si_0)[t_0]);
            }
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            (*(&kernelContext_1)->sv_0)[t_0] = ov_6;
            (*(&kernelContext_1)->si_0)[t_0] = oi_6;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 4U)
    {
        uint _S17 = t_0 + 4U;
        float ov_7 = (*(&kernelContext_1)->sv_0)[_S17];
        uint oi_7 = (*(&kernelContext_1)->si_0)[_S17];
        if(((*(&kernelContext_1)->si_0)[_S17]) != 4294967295U)
        {
            if(((*(&kernelContext_1)->si_0)[t_0]) == 4294967295U)
            {
                restricted_0 = true;
            }
            else
            {
                restricted_0 = before_0(ov_7, oi_7, (*(&kernelContext_1)->sv_0)[t_0], (*(&kernelContext_1)->si_0)[t_0]);
            }
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            (*(&kernelContext_1)->sv_0)[t_0] = ov_7;
            (*(&kernelContext_1)->si_0)[t_0] = oi_7;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 2U)
    {
        uint _S18 = t_0 + 2U;
        float ov_8 = (*(&kernelContext_1)->sv_0)[_S18];
        uint oi_8 = (*(&kernelContext_1)->si_0)[_S18];
        if(((*(&kernelContext_1)->si_0)[_S18]) != 4294967295U)
        {
            if(((*(&kernelContext_1)->si_0)[t_0]) == 4294967295U)
            {
                restricted_0 = true;
            }
            else
            {
                restricted_0 = before_0(ov_8, oi_8, (*(&kernelContext_1)->sv_0)[t_0], (*(&kernelContext_1)->si_0)[t_0]);
            }
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            (*(&kernelContext_1)->sv_0)[t_0] = ov_8;
            (*(&kernelContext_1)->si_0)[t_0] = oi_8;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 1U)
    {
        uint _S19 = t_0 + 1U;
        float ov_9 = (*(&kernelContext_1)->sv_0)[_S19];
        uint oi_9 = (*(&kernelContext_1)->si_0)[_S19];
        if(((*(&kernelContext_1)->si_0)[_S19]) != 4294967295U)
        {
            if(((*(&kernelContext_1)->si_0)[t_0]) == 4294967295U)
            {
                restricted_0 = true;
            }
            else
            {
                restricted_0 = before_0(ov_9, oi_9, (*(&kernelContext_1)->sv_0)[t_0], (*(&kernelContext_1)->si_0)[t_0]);
            }
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            (*(&kernelContext_1)->sv_0)[t_0] = ov_9;
            (*(&kernelContext_1)->si_0)[t_0] = oi_9;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _S20 = t_0 == 0U;
    if(_S20)
    {
        *(&kernelContext_1)->gMax_0 = (*(&kernelContext_1)->sv_0)[int(0)];
        if(((*(&kernelContext_1)->si_0)[int(0)]) == 4294967295U)
        {
            j_0 = 0U;
        }
        else
        {
            j_0 = (*(&kernelContext_1)->si_0)[int(0)];
        }
        *(&kernelContext_1)->gIdx_0 = j_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx_0 = *(&kernelContext_1)->gMax_0;
    uint argmax_0 = *(&kernelContext_1)->gIdx_0;
    if(((&kernelContext_1)->p_0->invT_0) <= 0.0)
    {
        restricted_0 = true;
    }
    else
    {
        restricted_0 = ((&kernelContext_1)->p_0->topK_0) == 1U;
    }
    if(restricted_0)
    {
        restricted_0 = true;
    }
    else
    {
        restricted_0 = mx_0 == _S8;
    }
    if(restricted_0)
    {
        if(_S20)
        {
            *((&kernelContext_1)->tokens_0+row_1) = argmax_0;
        }
        return;
    }
    if(((&kernelContext_1)->p_0->topK_0) != 0U)
    {
        restricted_0 = true;
    }
    else
    {
        restricted_0 = ((&kernelContext_1)->p_0->topP_0) < 1.0;
    }
    float part_0;
    if(!restricted_0)
    {
        uint chunk_0 = (n_0 + 1024U - 1U) / 1024U;
        uint j0_0 = t_0 * chunk_0;
        uint _S21 = min(n_0, j0_0 + chunk_0);
        j_0 = j0_0;
        part_0 = 0.0;
        for(;;)
        {
            if(j_0 < _S21)
            {
            }
            else
            {
                break;
            }
            float _S22 = part_0 + exp((*((&kernelContext_1)->logits_0+(_S7 + j_0)) - mx_0) * (&kernelContext_1)->p_0->invT_0);
            j_0 = j_0 + 1U;
            part_0 = _S22;
        }
        (*(&kernelContext_1)->sv_0)[t_0] = part_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S23 = t_0 + 1U;
        uint _S24 = _S23 * 2U;
        uint dgI_0 = _S24 - 1U;
        bool _S25 = dgI_0 < 1024U;
        if(_S25)
        {
            (*(&kernelContext_1)->sv_0)[dgI_0] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_0 - 1U], (*(&kernelContext_1)->sv_0)[dgI_0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_1 = _S24 * 2U - 1U;
        bool _S26 = dgI_1 < 1024U;
        if(_S26)
        {
            (*(&kernelContext_1)->sv_0)[dgI_1] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_1 - 2U], (*(&kernelContext_1)->sv_0)[dgI_1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_2 = _S23 * 4U * 2U - 1U;
        bool _S27 = dgI_2 < 1024U;
        if(_S27)
        {
            (*(&kernelContext_1)->sv_0)[dgI_2] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_2 - 4U], (*(&kernelContext_1)->sv_0)[dgI_2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_3 = _S23 * 8U * 2U - 1U;
        bool _S28 = dgI_3 < 1024U;
        if(_S28)
        {
            (*(&kernelContext_1)->sv_0)[dgI_3] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_3 - 8U], (*(&kernelContext_1)->sv_0)[dgI_3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_4 = _S23 * 16U * 2U - 1U;
        bool _S29 = dgI_4 < 1024U;
        if(_S29)
        {
            (*(&kernelContext_1)->sv_0)[dgI_4] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_4 - 16U], (*(&kernelContext_1)->sv_0)[dgI_4]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_5 = _S23 * 32U * 2U - 1U;
        bool _S30 = dgI_5 < 1024U;
        if(_S30)
        {
            (*(&kernelContext_1)->sv_0)[dgI_5] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_5 - 32U], (*(&kernelContext_1)->sv_0)[dgI_5]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_6 = _S23 * 64U * 2U - 1U;
        bool _S31 = dgI_6 < 1024U;
        if(_S31)
        {
            (*(&kernelContext_1)->sv_0)[dgI_6] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_6 - 64U], (*(&kernelContext_1)->sv_0)[dgI_6]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_7 = _S23 * 128U * 2U - 1U;
        bool _S32 = dgI_7 < 1024U;
        if(_S32)
        {
            (*(&kernelContext_1)->sv_0)[dgI_7] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_7 - 128U], (*(&kernelContext_1)->sv_0)[dgI_7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_8 = _S23 * 256U * 2U - 1U;
        bool _S33 = dgI_8 < 1024U;
        if(_S33)
        {
            (*(&kernelContext_1)->sv_0)[dgI_8] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_8 - 256U], (*(&kernelContext_1)->sv_0)[dgI_8]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_9 = _S23 * 512U * 2U - 1U;
        bool _S34 = dgI_9 < 1024U;
        if(_S34)
        {
            (*(&kernelContext_1)->sv_0)[dgI_9] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_9 - 512U], (*(&kernelContext_1)->sv_0)[dgI_9]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S20)
        {
            (*(&kernelContext_1)->sv_0)[1023U] = 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S34)
        {
            float _S35 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_9], (*(&kernelContext_1)->sv_0)[dgI_9 - 512U]);
            (*(&kernelContext_1)->sv_0)[dgI_9 - 512U] = (*(&kernelContext_1)->sv_0)[dgI_9];
            (*(&kernelContext_1)->sv_0)[dgI_9] = _S35;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S33)
        {
            float _S36 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_8], (*(&kernelContext_1)->sv_0)[dgI_8 - 256U]);
            (*(&kernelContext_1)->sv_0)[dgI_8 - 256U] = (*(&kernelContext_1)->sv_0)[dgI_8];
            (*(&kernelContext_1)->sv_0)[dgI_8] = _S36;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S32)
        {
            float _S37 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_7], (*(&kernelContext_1)->sv_0)[dgI_7 - 128U]);
            (*(&kernelContext_1)->sv_0)[dgI_7 - 128U] = (*(&kernelContext_1)->sv_0)[dgI_7];
            (*(&kernelContext_1)->sv_0)[dgI_7] = _S37;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S31)
        {
            float _S38 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_6], (*(&kernelContext_1)->sv_0)[dgI_6 - 64U]);
            (*(&kernelContext_1)->sv_0)[dgI_6 - 64U] = (*(&kernelContext_1)->sv_0)[dgI_6];
            (*(&kernelContext_1)->sv_0)[dgI_6] = _S38;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S30)
        {
            float _S39 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_5], (*(&kernelContext_1)->sv_0)[dgI_5 - 32U]);
            (*(&kernelContext_1)->sv_0)[dgI_5 - 32U] = (*(&kernelContext_1)->sv_0)[dgI_5];
            (*(&kernelContext_1)->sv_0)[dgI_5] = _S39;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S29)
        {
            float _S40 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_4], (*(&kernelContext_1)->sv_0)[dgI_4 - 16U]);
            (*(&kernelContext_1)->sv_0)[dgI_4 - 16U] = (*(&kernelContext_1)->sv_0)[dgI_4];
            (*(&kernelContext_1)->sv_0)[dgI_4] = _S40;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S28)
        {
            float _S41 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_3], (*(&kernelContext_1)->sv_0)[dgI_3 - 8U]);
            (*(&kernelContext_1)->sv_0)[dgI_3 - 8U] = (*(&kernelContext_1)->sv_0)[dgI_3];
            (*(&kernelContext_1)->sv_0)[dgI_3] = _S41;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S27)
        {
            float _S42 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_2], (*(&kernelContext_1)->sv_0)[dgI_2 - 4U]);
            (*(&kernelContext_1)->sv_0)[dgI_2 - 4U] = (*(&kernelContext_1)->sv_0)[dgI_2];
            (*(&kernelContext_1)->sv_0)[dgI_2] = _S42;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S26)
        {
            float _S43 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_1], (*(&kernelContext_1)->sv_0)[dgI_1 - 2U]);
            (*(&kernelContext_1)->sv_0)[dgI_1 - 2U] = (*(&kernelContext_1)->sv_0)[dgI_1];
            (*(&kernelContext_1)->sv_0)[dgI_1] = _S43;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S25)
        {
            float _S44 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_0], (*(&kernelContext_1)->sv_0)[dgI_0 - 1U]);
            (*(&kernelContext_1)->sv_0)[dgI_0 - 1U] = (*(&kernelContext_1)->sv_0)[dgI_0];
            (*(&kernelContext_1)->sv_0)[dgI_0] = _S44;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(t_0 == 1023U)
        {
            *(&kernelContext_1)->gTotal_0 = (*(&kernelContext_1)->sv_0)[t_0] + part_0;
        }
        if(_S20)
        {
            *(&kernelContext_1)->gFound_0 = 4294967295U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S45 = rand01_0(row_1, &kernelContext_1);
        float u_0 = _S45 * *(&kernelContext_1)->gTotal_0;
        float run_0 = (*(&kernelContext_1)->sv_0)[t_0];
        if(part_0 > 0.0)
        {
            restricted_0 = u_0 < (run_0 + part_0);
        }
        else
        {
            restricted_0 = false;
        }
        if(restricted_0)
        {
            j_0 = j0_0;
            float run_1 = run_0;
            for(;;)
            {
                if(j_0 < _S21)
                {
                }
                else
                {
                    break;
                }
                float run_2 = run_1 + exp((*((&kernelContext_1)->logits_0+(_S7 + j_0)) - mx_0) * (&kernelContext_1)->p_0->invT_0);
                if(run_2 > u_0)
                {
                    uint _S46 = atomic_fetch_min_explicit(((atomic_uint threadgroup*)((&kernelContext_1)->gFound_0)), j_0, memory_order_relaxed);
                    break;
                }
                j_0 = j_0 + 1U;
                run_1 = run_2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S20)
        {
            uint device* _S47 = (&kernelContext_1)->tokens_0+row_1;
            if((*(&kernelContext_1)->gFound_0) == 4294967295U)
            {
                j_0 = argmax_0;
            }
            else
            {
                j_0 = *(&kernelContext_1)->gFound_0;
            }
            *_S47 = j_0;
        }
        return;
    }
    uint k_0;
    if(((&kernelContext_1)->p_0->topK_0) == 0U)
    {
        k_0 = min(n_0, 1024U);
    }
    else
    {
        k_0 = min((&kernelContext_1)->p_0->topK_0, n_0);
    }
    if(_S20)
    {
        *(&kernelContext_1)->gPrefix_0 = 0U;
        *(&kernelContext_1)->gRemain_0 = k_0;
    }
    uint equal_0;
    uint greater_0;
    uint shift_0 = 24U;
    for(;;)
    {
        uint b_2 = t_0;
        for(;;)
        {
            if(b_2 < 256U)
            {
            }
            else
            {
                break;
            }
            (*(&kernelContext_1)->hist_0)[b_2] = 0U;
            b_2 = b_2 + 1024U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint prefix_0 = *(&kernelContext_1)->gPrefix_0;
        if(shift_0 == 24U)
        {
            j_0 = 0U;
        }
        else
        {
            j_0 = 4294967295U << (shift_0 + 8U);
        }
        uint j_1 = t_0;
        for(;;)
        {
            if(j_1 < n_0)
            {
            }
            else
            {
                break;
            }
            uint kv_0 = key_0(*((&kernelContext_1)->logits_0+(_S7 + j_1)));
            if((kv_0 & j_0) == (prefix_0 & j_0))
            {
                uint _S48 = atomic_fetch_add_explicit(((atomic_uint threadgroup*)(&(*(&kernelContext_1)->hist_0)[(kv_0 >> shift_0) & 255U])), 1U, memory_order_relaxed);
            }
            j_1 = j_1 + 1024U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S20)
        {
            uint _S49 = *(&kernelContext_1)->gRemain_0;
            greater_0 = 255U;
            equal_0 = _S49;
            for(;;)
            {
                uint c_0 = (*(&kernelContext_1)->hist_0)[greater_0];
                if(((*(&kernelContext_1)->hist_0)[greater_0]) >= equal_0)
                {
                    restricted_0 = true;
                }
                else
                {
                    restricted_0 = greater_0 == 0U;
                }
                if(restricted_0)
                {
                    break;
                }
                uint remain_0 = equal_0 - c_0;
                greater_0 = greater_0 - 1U;
                equal_0 = remain_0;
            }
            *(&kernelContext_1)->gPrefix_0 = prefix_0 | (greater_0 << shift_0);
            *(&kernelContext_1)->gRemain_0 = equal_0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(shift_0 == 0U)
        {
            break;
        }
        shift_0 = shift_0 - 8U;
    }
    uint _S50 = *(&kernelContext_1)->gPrefix_0;
    uint ties_0 = *(&kernelContext_1)->gRemain_0;
    uint chunk_1 = (n_0 + 1024U - 1U) / 1024U;
    uint j0_1 = t_0 * chunk_1;
    uint _S51 = min(n_0, j0_1 + chunk_1);
    j_0 = j0_1;
    greater_0 = 0U;
    equal_0 = 0U;
    for(;;)
    {
        if(j_0 < _S51)
        {
        }
        else
        {
            break;
        }
        uint kv_1 = key_0(*((&kernelContext_1)->logits_0+(_S7 + j_0)));
        if(kv_1 > _S50)
        {
            bi_0 = 1U;
        }
        else
        {
            bi_0 = 0U;
        }
        uint greater_1 = greater_0 + bi_0;
        if(kv_1 == _S50)
        {
            shift_0 = 1U;
        }
        else
        {
            shift_0 = 0U;
        }
        uint equal_1 = equal_0 + shift_0;
        j_0 = j_0 + 1U;
        greater_0 = greater_1;
        equal_0 = equal_1;
    }
    (*(&kernelContext_1)->si_0)[t_0] = equal_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S52 = t_0 + 1U;
    uint _S53 = _S52 * 2U;
    uint dgI_10 = _S53 - 1U;
    bool _S54 = dgI_10 < 1024U;
    if(_S54)
    {
        (*(&kernelContext_1)->si_0)[dgI_10] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_10 - 1U], (*(&kernelContext_1)->si_0)[dgI_10]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint dgI_11 = _S53 * 2U - 1U;
    bool _S55 = dgI_11 < 1024U;
    if(_S55)
    {
        (*(&kernelContext_1)->si_0)[dgI_11] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_11 - 2U], (*(&kernelContext_1)->si_0)[dgI_11]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint dgI_12 = _S52 * 4U * 2U - 1U;
    bool _S56 = dgI_12 < 1024U;
    if(_S56)
    {
        (*(&kernelContext_1)->si_0)[dgI_12] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_12 - 4U], (*(&kernelContext_1)->si_0)[dgI_12]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint dgI_13 = _S52 * 8U * 2U - 1U;
    bool _S57 = dgI_13 < 1024U;
    if(_S57)
    {
        (*(&kernelContext_1)->si_0)[dgI_13] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_13 - 8U], (*(&kernelContext_1)->si_0)[dgI_13]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint dgI_14 = _S52 * 16U * 2U - 1U;
    bool _S58 = dgI_14 < 1024U;
    if(_S58)
    {
        (*(&kernelContext_1)->si_0)[dgI_14] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_14 - 16U], (*(&kernelContext_1)->si_0)[dgI_14]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint dgI_15 = _S52 * 32U * 2U - 1U;
    bool _S59 = dgI_15 < 1024U;
    if(_S59)
    {
        (*(&kernelContext_1)->si_0)[dgI_15] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_15 - 32U], (*(&kernelContext_1)->si_0)[dgI_15]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint dgI_16 = _S52 * 64U * 2U - 1U;
    bool _S60 = dgI_16 < 1024U;
    if(_S60)
    {
        (*(&kernelContext_1)->si_0)[dgI_16] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_16 - 64U], (*(&kernelContext_1)->si_0)[dgI_16]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint dgI_17 = _S52 * 128U * 2U - 1U;
    bool _S61 = dgI_17 < 1024U;
    if(_S61)
    {
        (*(&kernelContext_1)->si_0)[dgI_17] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_17 - 128U], (*(&kernelContext_1)->si_0)[dgI_17]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint dgI_18 = _S52 * 256U * 2U - 1U;
    bool _S62 = dgI_18 < 1024U;
    if(_S62)
    {
        (*(&kernelContext_1)->si_0)[dgI_18] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_18 - 256U], (*(&kernelContext_1)->si_0)[dgI_18]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint dgI_19 = _S52 * 512U * 2U - 1U;
    bool _S63 = dgI_19 < 1024U;
    if(_S63)
    {
        (*(&kernelContext_1)->si_0)[dgI_19] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_19 - 512U], (*(&kernelContext_1)->si_0)[dgI_19]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S20)
    {
        (*(&kernelContext_1)->si_0)[1023U] = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S63)
    {
        uint _S64 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_19], (*(&kernelContext_1)->si_0)[dgI_19 - 512U]);
        (*(&kernelContext_1)->si_0)[dgI_19 - 512U] = (*(&kernelContext_1)->si_0)[dgI_19];
        (*(&kernelContext_1)->si_0)[dgI_19] = _S64;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S62)
    {
        uint _S65 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_18], (*(&kernelContext_1)->si_0)[dgI_18 - 256U]);
        (*(&kernelContext_1)->si_0)[dgI_18 - 256U] = (*(&kernelContext_1)->si_0)[dgI_18];
        (*(&kernelContext_1)->si_0)[dgI_18] = _S65;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S61)
    {
        uint _S66 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_17], (*(&kernelContext_1)->si_0)[dgI_17 - 128U]);
        (*(&kernelContext_1)->si_0)[dgI_17 - 128U] = (*(&kernelContext_1)->si_0)[dgI_17];
        (*(&kernelContext_1)->si_0)[dgI_17] = _S66;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S60)
    {
        uint _S67 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_16], (*(&kernelContext_1)->si_0)[dgI_16 - 64U]);
        (*(&kernelContext_1)->si_0)[dgI_16 - 64U] = (*(&kernelContext_1)->si_0)[dgI_16];
        (*(&kernelContext_1)->si_0)[dgI_16] = _S67;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S59)
    {
        uint _S68 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_15], (*(&kernelContext_1)->si_0)[dgI_15 - 32U]);
        (*(&kernelContext_1)->si_0)[dgI_15 - 32U] = (*(&kernelContext_1)->si_0)[dgI_15];
        (*(&kernelContext_1)->si_0)[dgI_15] = _S68;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S58)
    {
        uint _S69 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_14], (*(&kernelContext_1)->si_0)[dgI_14 - 16U]);
        (*(&kernelContext_1)->si_0)[dgI_14 - 16U] = (*(&kernelContext_1)->si_0)[dgI_14];
        (*(&kernelContext_1)->si_0)[dgI_14] = _S69;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S57)
    {
        uint _S70 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_13], (*(&kernelContext_1)->si_0)[dgI_13 - 8U]);
        (*(&kernelContext_1)->si_0)[dgI_13 - 8U] = (*(&kernelContext_1)->si_0)[dgI_13];
        (*(&kernelContext_1)->si_0)[dgI_13] = _S70;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S56)
    {
        uint _S71 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_12], (*(&kernelContext_1)->si_0)[dgI_12 - 4U]);
        (*(&kernelContext_1)->si_0)[dgI_12 - 4U] = (*(&kernelContext_1)->si_0)[dgI_12];
        (*(&kernelContext_1)->si_0)[dgI_12] = _S71;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S55)
    {
        uint _S72 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_11], (*(&kernelContext_1)->si_0)[dgI_11 - 2U]);
        (*(&kernelContext_1)->si_0)[dgI_11 - 2U] = (*(&kernelContext_1)->si_0)[dgI_11];
        (*(&kernelContext_1)->si_0)[dgI_11] = _S72;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S54)
    {
        uint _S73 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_10], (*(&kernelContext_1)->si_0)[dgI_10 - 1U]);
        (*(&kernelContext_1)->si_0)[dgI_10 - 1U] = (*(&kernelContext_1)->si_0)[dgI_10];
        (*(&kernelContext_1)->si_0)[dgI_10] = _S73;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tiesBefore_0 = (*(&kernelContext_1)->si_0)[t_0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint takeTies_0;
    if(ties_0 > (*(&kernelContext_1)->si_0)[t_0])
    {
        takeTies_0 = min(equal_0, ties_0 - tiesBefore_0);
    }
    else
    {
        takeTies_0 = 0U;
    }
    (*(&kernelContext_1)->si_0)[t_0] = greater_0 + takeTies_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S54)
    {
        (*(&kernelContext_1)->si_0)[dgI_10] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_10 - 1U], (*(&kernelContext_1)->si_0)[dgI_10]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S55)
    {
        (*(&kernelContext_1)->si_0)[dgI_11] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_11 - 2U], (*(&kernelContext_1)->si_0)[dgI_11]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S56)
    {
        (*(&kernelContext_1)->si_0)[dgI_12] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_12 - 4U], (*(&kernelContext_1)->si_0)[dgI_12]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S57)
    {
        (*(&kernelContext_1)->si_0)[dgI_13] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_13 - 8U], (*(&kernelContext_1)->si_0)[dgI_13]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S58)
    {
        (*(&kernelContext_1)->si_0)[dgI_14] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_14 - 16U], (*(&kernelContext_1)->si_0)[dgI_14]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S59)
    {
        (*(&kernelContext_1)->si_0)[dgI_15] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_15 - 32U], (*(&kernelContext_1)->si_0)[dgI_15]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S60)
    {
        (*(&kernelContext_1)->si_0)[dgI_16] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_16 - 64U], (*(&kernelContext_1)->si_0)[dgI_16]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S61)
    {
        (*(&kernelContext_1)->si_0)[dgI_17] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_17 - 128U], (*(&kernelContext_1)->si_0)[dgI_17]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S62)
    {
        (*(&kernelContext_1)->si_0)[dgI_18] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_18 - 256U], (*(&kernelContext_1)->si_0)[dgI_18]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S63)
    {
        (*(&kernelContext_1)->si_0)[dgI_19] = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_19 - 512U], (*(&kernelContext_1)->si_0)[dgI_19]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S20)
    {
        (*(&kernelContext_1)->si_0)[1023U] = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S63)
    {
        uint _S74 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_19], (*(&kernelContext_1)->si_0)[dgI_19 - 512U]);
        (*(&kernelContext_1)->si_0)[dgI_19 - 512U] = (*(&kernelContext_1)->si_0)[dgI_19];
        (*(&kernelContext_1)->si_0)[dgI_19] = _S74;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S62)
    {
        uint _S75 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_18], (*(&kernelContext_1)->si_0)[dgI_18 - 256U]);
        (*(&kernelContext_1)->si_0)[dgI_18 - 256U] = (*(&kernelContext_1)->si_0)[dgI_18];
        (*(&kernelContext_1)->si_0)[dgI_18] = _S75;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S61)
    {
        uint _S76 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_17], (*(&kernelContext_1)->si_0)[dgI_17 - 128U]);
        (*(&kernelContext_1)->si_0)[dgI_17 - 128U] = (*(&kernelContext_1)->si_0)[dgI_17];
        (*(&kernelContext_1)->si_0)[dgI_17] = _S76;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S60)
    {
        uint _S77 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_16], (*(&kernelContext_1)->si_0)[dgI_16 - 64U]);
        (*(&kernelContext_1)->si_0)[dgI_16 - 64U] = (*(&kernelContext_1)->si_0)[dgI_16];
        (*(&kernelContext_1)->si_0)[dgI_16] = _S77;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S59)
    {
        uint _S78 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_15], (*(&kernelContext_1)->si_0)[dgI_15 - 32U]);
        (*(&kernelContext_1)->si_0)[dgI_15 - 32U] = (*(&kernelContext_1)->si_0)[dgI_15];
        (*(&kernelContext_1)->si_0)[dgI_15] = _S78;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S58)
    {
        uint _S79 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_14], (*(&kernelContext_1)->si_0)[dgI_14 - 16U]);
        (*(&kernelContext_1)->si_0)[dgI_14 - 16U] = (*(&kernelContext_1)->si_0)[dgI_14];
        (*(&kernelContext_1)->si_0)[dgI_14] = _S79;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S57)
    {
        uint _S80 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_13], (*(&kernelContext_1)->si_0)[dgI_13 - 8U]);
        (*(&kernelContext_1)->si_0)[dgI_13 - 8U] = (*(&kernelContext_1)->si_0)[dgI_13];
        (*(&kernelContext_1)->si_0)[dgI_13] = _S80;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S56)
    {
        uint _S81 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_12], (*(&kernelContext_1)->si_0)[dgI_12 - 4U]);
        (*(&kernelContext_1)->si_0)[dgI_12 - 4U] = (*(&kernelContext_1)->si_0)[dgI_12];
        (*(&kernelContext_1)->si_0)[dgI_12] = _S81;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S55)
    {
        uint _S82 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_11], (*(&kernelContext_1)->si_0)[dgI_11 - 2U]);
        (*(&kernelContext_1)->si_0)[dgI_11 - 2U] = (*(&kernelContext_1)->si_0)[dgI_11];
        (*(&kernelContext_1)->si_0)[dgI_11] = _S82;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S54)
    {
        uint _S83 = dg_op_addu_0((*(&kernelContext_1)->si_0)[dgI_10], (*(&kernelContext_1)->si_0)[dgI_10 - 1U]);
        (*(&kernelContext_1)->si_0)[dgI_10 - 1U] = (*(&kernelContext_1)->si_0)[dgI_10];
        (*(&kernelContext_1)->si_0)[dgI_10] = _S83;
    }
    bool take_0;
    uint tieSeen_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S84 = (*(&kernelContext_1)->si_0)[t_0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*(&kernelContext_1)->sv_0)[t_0] = _S8;
    (*(&kernelContext_1)->si_0)[t_0] = 4294967295U;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    j_0 = j0_1;
    uint tieSeen_1 = 0U;
    uint at_0 = _S84;
    for(;;)
    {
        if(j_0 < _S51)
        {
        }
        else
        {
            break;
        }
        float device* _S85 = (&kernelContext_1)->logits_0+(_S7 + j_0);
        float l_2 = *_S85;
        uint kv_2 = key_0(*_S85);
        if(kv_2 > _S50)
        {
            take_0 = true;
        }
        else
        {
            if(kv_2 == _S50)
            {
                uint _S86 = tieSeen_1 + 1U;
                restricted_0 = tieSeen_1 < takeTies_0;
                tieSeen_0 = _S86;
            }
            else
            {
                restricted_0 = false;
                tieSeen_0 = tieSeen_1;
            }
            take_0 = restricted_0;
            tieSeen_1 = tieSeen_0;
        }
        if(take_0)
        {
            if(dg_is_nan_0(l_2))
            {
                bv_0 = _S8;
            }
            else
            {
                bv_0 = l_2;
            }
            (*(&kernelContext_1)->sv_0)[at_0] = bv_0;
            (*(&kernelContext_1)->si_0)[at_0] = j_0;
            at_0 = at_0 + 1U;
        }
        j_0 = j_0 + 1U;
    }
    uint stride_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tieSeen_0 = 2U;
    for(;;)
    {
        if(tieSeen_0 <= 1024U)
        {
        }
        else
        {
            break;
        }
        stride_0 = tieSeen_0 / 2U;
        for(;;)
        {
            if(stride_0 > 0U)
            {
            }
            else
            {
                break;
            }
            uint partner_0 = t_0 ^ stride_0;
            if(partner_0 > t_0)
            {
                float va_1 = (*(&kernelContext_1)->sv_0)[t_0];
                float vb_1 = (*(&kernelContext_1)->sv_0)[partner_0];
                uint ia_1 = (*(&kernelContext_1)->si_0)[t_0];
                uint ib_1 = (*(&kernelContext_1)->si_0)[partner_0];
                if((t_0 & tieSeen_0) == 0U)
                {
                    take_0 = before_0(vb_1, ib_1, va_1, ia_1);
                }
                else
                {
                    take_0 = before_0(va_1, ia_1, vb_1, ib_1);
                }
                if(take_0)
                {
                    (*(&kernelContext_1)->sv_0)[t_0] = vb_1;
                    (*(&kernelContext_1)->sv_0)[partner_0] = va_1;
                    (*(&kernelContext_1)->si_0)[t_0] = ib_1;
                    (*(&kernelContext_1)->si_0)[partner_0] = ia_1;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            stride_0 = stride_0 / 2U;
        }
        tieSeen_0 = tieSeen_0 * 2U;
    }
    uint myIndex_0 = (*(&kernelContext_1)->si_0)[t_0];
    bool _S87 = t_0 < k_0;
    if(_S87)
    {
        part_0 = exp(((*(&kernelContext_1)->sv_0)[t_0] - mx_0) * (&kernelContext_1)->p_0->invT_0);
    }
    else
    {
        part_0 = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*(&kernelContext_1)->sv_0)[t_0] = part_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S54)
    {
        (*(&kernelContext_1)->sv_0)[dgI_10] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_10 - 1U], (*(&kernelContext_1)->sv_0)[dgI_10]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S55)
    {
        (*(&kernelContext_1)->sv_0)[dgI_11] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_11 - 2U], (*(&kernelContext_1)->sv_0)[dgI_11]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S56)
    {
        (*(&kernelContext_1)->sv_0)[dgI_12] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_12 - 4U], (*(&kernelContext_1)->sv_0)[dgI_12]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S57)
    {
        (*(&kernelContext_1)->sv_0)[dgI_13] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_13 - 8U], (*(&kernelContext_1)->sv_0)[dgI_13]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S58)
    {
        (*(&kernelContext_1)->sv_0)[dgI_14] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_14 - 16U], (*(&kernelContext_1)->sv_0)[dgI_14]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S59)
    {
        (*(&kernelContext_1)->sv_0)[dgI_15] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_15 - 32U], (*(&kernelContext_1)->sv_0)[dgI_15]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S60)
    {
        (*(&kernelContext_1)->sv_0)[dgI_16] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_16 - 64U], (*(&kernelContext_1)->sv_0)[dgI_16]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S61)
    {
        (*(&kernelContext_1)->sv_0)[dgI_17] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_17 - 128U], (*(&kernelContext_1)->sv_0)[dgI_17]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S62)
    {
        (*(&kernelContext_1)->sv_0)[dgI_18] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_18 - 256U], (*(&kernelContext_1)->sv_0)[dgI_18]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S63)
    {
        (*(&kernelContext_1)->sv_0)[dgI_19] = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_19 - 512U], (*(&kernelContext_1)->sv_0)[dgI_19]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S20)
    {
        (*(&kernelContext_1)->sv_0)[1023U] = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S63)
    {
        float _S88 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_19], (*(&kernelContext_1)->sv_0)[dgI_19 - 512U]);
        (*(&kernelContext_1)->sv_0)[dgI_19 - 512U] = (*(&kernelContext_1)->sv_0)[dgI_19];
        (*(&kernelContext_1)->sv_0)[dgI_19] = _S88;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S62)
    {
        float _S89 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_18], (*(&kernelContext_1)->sv_0)[dgI_18 - 256U]);
        (*(&kernelContext_1)->sv_0)[dgI_18 - 256U] = (*(&kernelContext_1)->sv_0)[dgI_18];
        (*(&kernelContext_1)->sv_0)[dgI_18] = _S89;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S61)
    {
        float _S90 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_17], (*(&kernelContext_1)->sv_0)[dgI_17 - 128U]);
        (*(&kernelContext_1)->sv_0)[dgI_17 - 128U] = (*(&kernelContext_1)->sv_0)[dgI_17];
        (*(&kernelContext_1)->sv_0)[dgI_17] = _S90;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S60)
    {
        float _S91 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_16], (*(&kernelContext_1)->sv_0)[dgI_16 - 64U]);
        (*(&kernelContext_1)->sv_0)[dgI_16 - 64U] = (*(&kernelContext_1)->sv_0)[dgI_16];
        (*(&kernelContext_1)->sv_0)[dgI_16] = _S91;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S59)
    {
        float _S92 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_15], (*(&kernelContext_1)->sv_0)[dgI_15 - 32U]);
        (*(&kernelContext_1)->sv_0)[dgI_15 - 32U] = (*(&kernelContext_1)->sv_0)[dgI_15];
        (*(&kernelContext_1)->sv_0)[dgI_15] = _S92;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S58)
    {
        float _S93 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_14], (*(&kernelContext_1)->sv_0)[dgI_14 - 16U]);
        (*(&kernelContext_1)->sv_0)[dgI_14 - 16U] = (*(&kernelContext_1)->sv_0)[dgI_14];
        (*(&kernelContext_1)->sv_0)[dgI_14] = _S93;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S57)
    {
        float _S94 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_13], (*(&kernelContext_1)->sv_0)[dgI_13 - 8U]);
        (*(&kernelContext_1)->sv_0)[dgI_13 - 8U] = (*(&kernelContext_1)->sv_0)[dgI_13];
        (*(&kernelContext_1)->sv_0)[dgI_13] = _S94;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S56)
    {
        float _S95 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_12], (*(&kernelContext_1)->sv_0)[dgI_12 - 4U]);
        (*(&kernelContext_1)->sv_0)[dgI_12 - 4U] = (*(&kernelContext_1)->sv_0)[dgI_12];
        (*(&kernelContext_1)->sv_0)[dgI_12] = _S95;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S55)
    {
        float _S96 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_11], (*(&kernelContext_1)->sv_0)[dgI_11 - 2U]);
        (*(&kernelContext_1)->sv_0)[dgI_11 - 2U] = (*(&kernelContext_1)->sv_0)[dgI_11];
        (*(&kernelContext_1)->sv_0)[dgI_11] = _S96;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S54)
    {
        float _S97 = dg_op_add_0((*(&kernelContext_1)->sv_0)[dgI_10], (*(&kernelContext_1)->sv_0)[dgI_10 - 1U]);
        (*(&kernelContext_1)->sv_0)[dgI_10 - 1U] = (*(&kernelContext_1)->sv_0)[dgI_10];
        (*(&kernelContext_1)->sv_0)[dgI_10] = _S97;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float incl_0 = (*(&kernelContext_1)->sv_0)[t_0] + part_0;
    if(t_0 == 1023U)
    {
        *(&kernelContext_1)->gTotal_0 = incl_0;
    }
    if(_S20)
    {
        *(&kernelContext_1)->gFound_0 = 4294967295U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_0 = *(&kernelContext_1)->gTotal_0;
    if(_S87)
    {
        restricted_0 = incl_0 >= ((&kernelContext_1)->p_0->topP_0 * total_0);
    }
    else
    {
        restricted_0 = false;
    }
    if(restricted_0)
    {
        if(_S20)
        {
            restricted_0 = true;
        }
        else
        {
            restricted_0 = ((*(&kernelContext_1)->sv_0)[t_0]) < ((&kernelContext_1)->p_0->topP_0 * total_0);
        }
    }
    else
    {
        restricted_0 = false;
    }
    if(restricted_0)
    {
        *(&kernelContext_1)->gFound_0 = t_0;
    }
    if(((&kernelContext_1)->p_0->topP_0) >= 1.0)
    {
        restricted_0 = _S20;
    }
    else
    {
        restricted_0 = false;
    }
    if(restricted_0)
    {
        *(&kernelContext_1)->gFound_0 = k_0 - 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((*(&kernelContext_1)->gFound_0) == 4294967295U)
    {
        stride_0 = k_0 - 1U;
    }
    else
    {
        stride_0 = *(&kernelContext_1)->gFound_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 == stride_0)
    {
        float _S98 = rand01_0(row_1, &kernelContext_1);
        *(&kernelContext_1)->gU_0 = _S98 * incl_0;
    }
    if(_S20)
    {
        *(&kernelContext_1)->gFound_0 = 4294967295U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float u_1 = *(&kernelContext_1)->gU_0;
    if(t_0 <= stride_0)
    {
        restricted_0 = incl_0 > u_1;
    }
    else
    {
        restricted_0 = false;
    }
    if(restricted_0)
    {
        if(_S20)
        {
            restricted_0 = true;
        }
        else
        {
            restricted_0 = ((*(&kernelContext_1)->sv_0)[t_0]) <= u_1;
        }
    }
    else
    {
        restricted_0 = false;
    }
    if(restricted_0)
    {
        uint _S99 = atomic_fetch_min_explicit(((atomic_uint threadgroup*)((&kernelContext_1)->gFound_0)), t_0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pick_0;
    if((*(&kernelContext_1)->gFound_0) == 4294967295U)
    {
        pick_0 = stride_0;
    }
    else
    {
        pick_0 = *(&kernelContext_1)->gFound_0;
    }
    if(t_0 == pick_0)
    {
        *((&kernelContext_1)->tokens_0+row_1) = myIndex_0;
    }
    return;
}

