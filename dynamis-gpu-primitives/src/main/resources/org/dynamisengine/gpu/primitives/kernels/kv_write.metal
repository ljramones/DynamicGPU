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

struct KvWriteParams_0
{
    uint kvHeads_0;
    uint headDim_0;
    uint tokens_0;
    uint position_0;
    uint pageSize_0;
    uint headStride_0;
    uint tokenStride_0;
    uint gridX_1;
};

struct KernelContext_0
{
    KvWriteParams_0 constant* p_0;
    uint device* table_0;
    uint device* kPool_0;
    uint device* kSrc_0;
    uint device* vPool_0;
    uint device* vSrc_0;
};

[[kernel]] void kv_write(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], KvWriteParams_0 constant* p_1 [[buffer(5)]], uint device* table_1 [[buffer(4)]], uint device* kPool_1 [[buffer(2)]], uint device* kSrc_1 [[buffer(0)]], uint device* vPool_1 [[buffer(3)]], uint device* vSrc_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->table_0 = table_1;
    (&kernelContext_0)->kPool_0 = kPool_1;
    (&kernelContext_0)->kSrc_0 = kSrc_1;
    (&kernelContext_0)->vPool_0 = vPool_1;
    (&kernelContext_0)->vSrc_0 = vSrc_1;
    uint d2_0 = p_1->headDim_0 / 2U;
    uint id_0 = dg_linear_group_0(group_1, p_1->gridX_1) * 256U + local_0.x;
    if(id_0 >= (p_1->tokens_0 * p_1->kvHeads_0 * d2_0))
    {
        return;
    }
    uint pr_0 = id_0 % d2_0;
    uint _S1 = id_0 / d2_0;
    uint h_0 = _S1 % p_1->kvHeads_0;
    uint t_0 = id_0 / (d2_0 * p_1->kvHeads_0);
    uint src_0 = (h_0 * (&kernelContext_0)->p_0->headStride_0 + t_0 * (&kernelContext_0)->p_0->tokenStride_0) / 2U + pr_0;
    uint pos_0 = (&kernelContext_0)->p_0->position_0 + t_0;
    uint _S2 = pos_0 / (&kernelContext_0)->p_0->pageSize_0;
    uint _S3 = (*((&kernelContext_0)->table_0+_S2) * p_1->kvHeads_0 + h_0) * (&kernelContext_0)->p_0->pageSize_0;
    uint _S4 = pos_0 % (&kernelContext_0)->p_0->pageSize_0;
    uint dst_0 = (_S3 + _S4) * d2_0 + pr_0;
    *((&kernelContext_0)->kPool_0+dst_0) = *((&kernelContext_0)->kSrc_0+src_0);
    *((&kernelContext_0)->vPool_0+dst_0) = *((&kernelContext_0)->vSrc_0+src_0);
    return;
}

