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

uint dg_linear_thread_0(uint3 group_1, uint3 local_0, uint gridX_1, uint groupThreads_0)
{
    return dg_linear_group_0(group_1, gridX_1) * groupThreads_0 + local_0.x;
}

struct AttnBwdParams_0
{
    uint heads_0;
    uint kvHeads_0;
    uint queries_0;
    uint keys_0;
    uint headDim_0;
    uint cacheKeys_0;
    float scale_0;
    uint causal_0;
    uint gridX_2;
    uint lengthGroup_0;
    uint pad1_0;
    uint pad2_0;
};

struct KernelContext_0
{
    AttnBwdParams_0 constant* p_0;
    uint device* dD_0;
    uint device* lens_0;
    float device* o_0;
    float device* dO_0;
};

[[kernel]] void attn_bwd_dot(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], AttnBwdParams_0 constant* p_1 [[buffer(4)]], uint device* dD_1 [[buffer(2)]], uint device* lens_1 [[buffer(3)]], float device* o_1 [[buffer(0)]], float device* dO_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->dD_0 = dD_1;
    (&kernelContext_0)->lens_0 = lens_1;
    (&kernelContext_0)->o_0 = o_1;
    (&kernelContext_0)->dO_0 = dO_1;
    uint row_0 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    uint rows_0 = p_1->heads_0 * p_1->queries_0;
    if(row_0 >= rows_0)
    {
        return;
    }
    if(((&kernelContext_0)->p_0->lengthGroup_0) != 0U)
    {
        uint _S1 = (p_1->heads_0 - 1U) / (&kernelContext_0)->p_0->lengthGroup_0;
        if(row_0 <= _S1)
        {
            *((&kernelContext_0)->dD_0+(rows_0 + row_0)) = min(*((&kernelContext_0)->lens_0+row_0), (&kernelContext_0)->p_0->keys_0);
        }
    }
    uint d_0 = (&kernelContext_0)->p_0->headDim_0;
    uint _S2 = row_0 * (&kernelContext_0)->p_0->headDim_0;
    float acc_0 = 0.0;
    uint c_0 = 0U;
    for(;;)
    {
        if(c_0 < d_0)
        {
        }
        else
        {
            break;
        }
        uint _S3 = _S2 + c_0;
        uint _S4 = _S3 + 1U;
        float _S5 = acc_0 + *((&kernelContext_0)->dO_0+_S3) * *((&kernelContext_0)->o_0+_S3) + *((&kernelContext_0)->dO_0+_S4) * *((&kernelContext_0)->o_0+_S4);
        uint c_1 = c_0 + 2U;
        acc_0 = _S5;
        c_0 = c_1;
    }
    *((&kernelContext_0)->dD_0+row_0) = (as_type<uint>((acc_0)));
    return;
}

