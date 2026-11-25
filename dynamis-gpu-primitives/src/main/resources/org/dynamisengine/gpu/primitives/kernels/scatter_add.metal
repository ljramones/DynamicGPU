#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant bool fc_DG_DEBUG_0 [[function_constant(100)]];
constant bool DG_DEBUG_0 = is_function_constant_defined(fc_DG_DEBUG_0) ? fc_DG_DEBUG_0 : false;
uint dg_linear_group_0(uint3 group_0, uint gridX_0)
{
    return group_0.y * gridX_0 + group_0.x;
}

struct DgView_0
{
    uint offset_0;
    uint4 extent_0;
    uint4 stride_0;
};

DgView_0 dg_view_0(uint offset_1, uint4 extent_1, uint4 stride_1)
{
    thread DgView_0 v_0;
    (&v_0)->offset_0 = offset_1;
    (&v_0)->extent_0 = extent_1;
    (&v_0)->stride_0 = stride_1;
    return v_0;
}

uint3 dg_row_0(const DgView_0 thread* v_1, uint row_0)
{
    thread uint3 i_0;
    uint4 _S1 = v_1->extent_0;
    uint _S2 = v_1->extent_0.z;
    uint _S3 = row_0 % _S2;
    i_0.z = _S3;
    uint _S4 = row_0 / _S2;
    uint _S5 = _S1.y;
    uint _S6 = _S4 % _S5;
    i_0.y = _S6;
    uint _S7 = _S4 / _S5;
    i_0.x = _S7;
    return i_0;
}

uint dg_at_0(const DgView_0 thread* v_2, uint4 i_1)
{
    return v_2->offset_0 + i_1.x * v_2->stride_0.x + i_1.y * v_2->stride_0.y + i_1.z * v_2->stride_0.z + i_1.w * v_2->stride_0.w;
}

void dg_report_error_0(uint device* errors_0, uint code_0, uint index_0)
{
    uint _S8 = atomic_fetch_or_explicit(((atomic_uint device*)(errors_0+int(0))), code_0, memory_order_relaxed);
    uint _S9 = atomic_fetch_max_explicit(((atomic_uint device*)(errors_0+int(1))), ~index_0, memory_order_relaxed);
    uint _S10 = atomic_fetch_add_explicit(((atomic_uint device*)(errors_0+int(2))), 1U, memory_order_relaxed);
    return;
}

bool dg_in_range_0(uint device* errors_1, uint index_1, uint length_0, uint at_0)
{
    bool _S11;
    if(!DG_DEBUG_0)
    {
        _S11 = true;
    }
    else
    {
        _S11 = index_1 < length_0;
    }
    if(_S11)
    {
        return true;
    }
    dg_report_error_0(errors_1, 1U, at_0);
    return false;
}

uint dg_op_addu_0(uint a_0, uint b_0)
{
    return a_0 + b_0;
}

struct Params_0
{
    uint extent_2;
    uint gridX_1;
    uint n_0;
    uint accumulate_0;
    uint dxLength_0;
    uint dyLength_0;
    uint iOffset_0;
    uint iStride_0;
    uint dyOffset_0;
    uint dyExtent0_0;
    uint dyStride0_0;
    uint dyExtent1_0;
    uint dyStride1_0;
    uint dyExtent2_0;
    uint dyStride2_0;
    uint dyExtent3_0;
    uint dyStride3_0;
    uint dxOffset_0;
    uint dxExtent0_0;
    uint dxStride0_0;
    uint dxExtent1_0;
    uint dxStride1_0;
    uint dxExtent2_0;
    uint dxStride2_0;
    uint dxExtent3_0;
    uint dxStride3_0;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    uint device* errors_2;
    float device* dx_0;
    uint device* counts_0;
    uint device* indices_0;
    float device* dy_0;
    array<uint, int(256)> threadgroup* scan_0;
    array<uint, int(256)> threadgroup* list_0;
    uint threadgroup* chunkTotal_0;
};

[[kernel]] void scatter_add(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], Params_0 constant* p_1 [[buffer(5)]], uint device* errors_3 [[buffer(4)]], float device* dx_1 [[buffer(3)]], uint device* counts_1 [[buffer(2)]], uint device* indices_1 [[buffer(1)]], float device* dy_1 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->errors_2 = errors_3;
    (&kernelContext_0)->dx_0 = dx_1;
    (&kernelContext_0)->counts_0 = counts_1;
    (&kernelContext_0)->indices_0 = indices_1;
    (&kernelContext_0)->dy_0 = dy_1;
    threadgroup array<uint, int(256)> scan_1;
    (&kernelContext_0)->scan_0 = &scan_1;
    threadgroup array<uint, int(256)> list_1;
    (&kernelContext_0)->list_0 = &list_1;
    threadgroup uint chunkTotal_1;
    (&kernelContext_0)->chunkTotal_0 = &chunkTotal_1;
    uint k_0 = dg_linear_group_0(group_1, p_1->gridX_1);
    if(k_0 >= (p_1->extent_2))
    {
        return;
    }
    uint t_0 = local_0.x;
    DgView_0 xv_0 = dg_view_0((&kernelContext_0)->p_0->dxOffset_0, uint4((&kernelContext_0)->p_0->dxExtent0_0, (&kernelContext_0)->p_0->dxExtent1_0, (&kernelContext_0)->p_0->dxExtent2_0, (&kernelContext_0)->p_0->dxExtent3_0), uint4((&kernelContext_0)->p_0->dxStride0_0, (&kernelContext_0)->p_0->dxStride1_0, (&kernelContext_0)->p_0->dxStride2_0, (&kernelContext_0)->p_0->dxStride3_0));
    DgView_0 _S12 = dg_view_0((&kernelContext_0)->p_0->dyOffset_0, uint4((&kernelContext_0)->p_0->dyExtent0_0, (&kernelContext_0)->p_0->dyExtent1_0, (&kernelContext_0)->p_0->dyExtent2_0, (&kernelContext_0)->p_0->dyExtent3_0), uint4((&kernelContext_0)->p_0->dyStride0_0, (&kernelContext_0)->p_0->dyStride1_0, (&kernelContext_0)->p_0->dyStride2_0, (&kernelContext_0)->p_0->dyStride3_0));
    uint _S13 = xv_0.extent_0.x * xv_0.extent_0.y * xv_0.extent_0.z;
    uint base_0;
    if(((&kernelContext_0)->p_0->accumulate_0) == 0U)
    {
        base_0 = t_0;
        for(;;)
        {
            if(base_0 < _S13)
            {
            }
            else
            {
                break;
            }
            thread DgView_0 _S14 = xv_0;
            uint3 _S15 = dg_row_0(&_S14, base_0);
            uint4 _S16 = uint4(_S15, k_0);
            thread DgView_0 _S17 = xv_0;
            uint _S18 = dg_at_0(&_S17, _S16);
            bool _S19 = dg_in_range_0((&kernelContext_0)->errors_2, _S18, (&kernelContext_0)->p_0->dxLength_0, k_0);
            if(_S19)
            {
                *((&kernelContext_0)->dx_0+_S18) = 0.0;
            }
            base_0 = base_0 + 256U;
        }
    }
    uint _S20 = *((&kernelContext_0)->counts_0+k_0);
    base_0 = 0U;
    uint found_0 = 0U;
    for(;;)
    {
        bool _S21;
        if(base_0 < ((&kernelContext_0)->p_0->n_0))
        {
            _S21 = found_0 < _S20;
        }
        else
        {
            _S21 = false;
        }
        if(_S21)
        {
        }
        else
        {
            break;
        }
        uint j_0 = base_0 + t_0;
        bool _S22;
        if(j_0 < ((&kernelContext_0)->p_0->n_0))
        {
            _S22 = (*((&kernelContext_0)->indices_0+((&kernelContext_0)->p_0->iOffset_0 + j_0 * (&kernelContext_0)->p_0->iStride_0))) == k_0;
        }
        else
        {
            _S22 = false;
        }
        uint match_0;
        if(_S22)
        {
            match_0 = 1U;
        }
        else
        {
            match_0 = 0U;
        }
        (*(&kernelContext_0)->scan_0)[t_0] = match_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S23 = t_0 + 1U;
        uint _S24 = _S23 * 2U;
        uint dgI_0 = _S24 - 1U;
        bool _S25 = dgI_0 < 256U;
        if(_S25)
        {
            (*(&kernelContext_0)->scan_0)[dgI_0] = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_0 - 1U], (*(&kernelContext_0)->scan_0)[dgI_0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_1 = _S24 * 2U - 1U;
        bool _S26 = dgI_1 < 256U;
        if(_S26)
        {
            (*(&kernelContext_0)->scan_0)[dgI_1] = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_1 - 2U], (*(&kernelContext_0)->scan_0)[dgI_1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_2 = _S23 * 4U * 2U - 1U;
        bool _S27 = dgI_2 < 256U;
        if(_S27)
        {
            (*(&kernelContext_0)->scan_0)[dgI_2] = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_2 - 4U], (*(&kernelContext_0)->scan_0)[dgI_2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_3 = _S23 * 8U * 2U - 1U;
        bool _S28 = dgI_3 < 256U;
        if(_S28)
        {
            (*(&kernelContext_0)->scan_0)[dgI_3] = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_3 - 8U], (*(&kernelContext_0)->scan_0)[dgI_3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_4 = _S23 * 16U * 2U - 1U;
        bool _S29 = dgI_4 < 256U;
        if(_S29)
        {
            (*(&kernelContext_0)->scan_0)[dgI_4] = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_4 - 16U], (*(&kernelContext_0)->scan_0)[dgI_4]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_5 = _S23 * 32U * 2U - 1U;
        bool _S30 = dgI_5 < 256U;
        if(_S30)
        {
            (*(&kernelContext_0)->scan_0)[dgI_5] = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_5 - 32U], (*(&kernelContext_0)->scan_0)[dgI_5]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_6 = _S23 * 64U * 2U - 1U;
        bool _S31 = dgI_6 < 256U;
        if(_S31)
        {
            (*(&kernelContext_0)->scan_0)[dgI_6] = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_6 - 64U], (*(&kernelContext_0)->scan_0)[dgI_6]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dgI_7 = _S23 * 128U * 2U - 1U;
        bool _S32 = dgI_7 < 256U;
        if(_S32)
        {
            (*(&kernelContext_0)->scan_0)[dgI_7] = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_7 - 128U], (*(&kernelContext_0)->scan_0)[dgI_7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(t_0 == 0U)
        {
            (*(&kernelContext_0)->scan_0)[255U] = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S32)
        {
            uint _S33 = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_7], (*(&kernelContext_0)->scan_0)[dgI_7 - 128U]);
            (*(&kernelContext_0)->scan_0)[dgI_7 - 128U] = (*(&kernelContext_0)->scan_0)[dgI_7];
            (*(&kernelContext_0)->scan_0)[dgI_7] = _S33;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S31)
        {
            uint _S34 = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_6], (*(&kernelContext_0)->scan_0)[dgI_6 - 64U]);
            (*(&kernelContext_0)->scan_0)[dgI_6 - 64U] = (*(&kernelContext_0)->scan_0)[dgI_6];
            (*(&kernelContext_0)->scan_0)[dgI_6] = _S34;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S30)
        {
            uint _S35 = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_5], (*(&kernelContext_0)->scan_0)[dgI_5 - 32U]);
            (*(&kernelContext_0)->scan_0)[dgI_5 - 32U] = (*(&kernelContext_0)->scan_0)[dgI_5];
            (*(&kernelContext_0)->scan_0)[dgI_5] = _S35;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S29)
        {
            uint _S36 = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_4], (*(&kernelContext_0)->scan_0)[dgI_4 - 16U]);
            (*(&kernelContext_0)->scan_0)[dgI_4 - 16U] = (*(&kernelContext_0)->scan_0)[dgI_4];
            (*(&kernelContext_0)->scan_0)[dgI_4] = _S36;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S28)
        {
            uint _S37 = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_3], (*(&kernelContext_0)->scan_0)[dgI_3 - 8U]);
            (*(&kernelContext_0)->scan_0)[dgI_3 - 8U] = (*(&kernelContext_0)->scan_0)[dgI_3];
            (*(&kernelContext_0)->scan_0)[dgI_3] = _S37;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S27)
        {
            uint _S38 = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_2], (*(&kernelContext_0)->scan_0)[dgI_2 - 4U]);
            (*(&kernelContext_0)->scan_0)[dgI_2 - 4U] = (*(&kernelContext_0)->scan_0)[dgI_2];
            (*(&kernelContext_0)->scan_0)[dgI_2] = _S38;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S26)
        {
            uint _S39 = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_1], (*(&kernelContext_0)->scan_0)[dgI_1 - 2U]);
            (*(&kernelContext_0)->scan_0)[dgI_1 - 2U] = (*(&kernelContext_0)->scan_0)[dgI_1];
            (*(&kernelContext_0)->scan_0)[dgI_1] = _S39;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S25)
        {
            uint _S40 = dg_op_addu_0((*(&kernelContext_0)->scan_0)[dgI_0], (*(&kernelContext_0)->scan_0)[dgI_0 - 1U]);
            (*(&kernelContext_0)->scan_0)[dgI_0 - 1U] = (*(&kernelContext_0)->scan_0)[dgI_0];
            (*(&kernelContext_0)->scan_0)[dgI_0] = _S40;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(match_0 != 0U)
        {
            (*(&kernelContext_0)->list_0)[(*(&kernelContext_0)->scan_0)[t_0]] = j_0;
        }
        if(t_0 == 255U)
        {
            *(&kernelContext_0)->chunkTotal_0 = (*(&kernelContext_0)->scan_0)[t_0] + match_0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint threadgroup* _S41 = (&kernelContext_0)->chunkTotal_0;
        uint m_0 = *_S41;
        if((*_S41) != 0U)
        {
            uint e_0 = t_0;
            for(;;)
            {
                if(e_0 < _S13)
                {
                }
                else
                {
                    break;
                }
                thread DgView_0 _S42 = xv_0;
                uint3 _S43 = dg_row_0(&_S42, e_0);
                uint4 _S44 = uint4(_S43, k_0);
                thread DgView_0 _S45 = xv_0;
                uint _S46 = dg_at_0(&_S45, _S44);
                bool _S47 = dg_in_range_0((&kernelContext_0)->errors_2, _S46, (&kernelContext_0)->p_0->dxLength_0, k_0);
                if(!_S47)
                {
                    e_0 = e_0 + 256U;
                    continue;
                }
                float device* _S48 = (&kernelContext_0)->dx_0+_S46;
                uint q_0 = 0U;
                float acc_0 = *_S48;
                for(;;)
                {
                    if(q_0 < m_0)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint4 _S49 = uint4(_S43, (*(&kernelContext_0)->list_0)[q_0]);
                    thread DgView_0 _S50 = _S12;
                    uint _S51 = dg_at_0(&_S50, _S49);
                    bool _S52 = dg_in_range_0((&kernelContext_0)->errors_2, _S51, (&kernelContext_0)->p_0->dyLength_0, k_0);
                    if(_S52)
                    {
                        acc_0 = acc_0 + *((&kernelContext_0)->dy_0+_S51);
                    }
                    q_0 = q_0 + 1U;
                }
                *((&kernelContext_0)->dx_0+_S46) = acc_0;
                e_0 = e_0 + 256U;
            }
        }
        uint found_1 = found_0 + m_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        base_0 = base_0 + 256U;
        found_0 = found_1;
    }
    return;
}

