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

float attnLo_0(uint w_0)
{
    return (as_type<half>((ushort)((w_0 & 65535U))));
}

float attnHi_0(uint w_1)
{
    return (as_type<half>((ushort)((w_1 >> 16U))));
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
    uint pad0_0;
    uint pad1_0;
    uint pad2_0;
};

struct KernelContext_0
{
    AttnBwdParams_0 constant* p_0;
    uint device* o_0;
    float device* dO_0;
    float device* dD_0;
};

[[kernel]] void attn_bwd_dot_h(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], AttnBwdParams_0 constant* p_1 [[buffer(3)]], uint device* o_1 [[buffer(0)]], float device* dO_1 [[buffer(1)]], float device* dD_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->o_0 = o_1;
    (&kernelContext_0)->dO_0 = dO_1;
    (&kernelContext_0)->dD_0 = dD_1;
    uint row_0 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(row_0 >= (p_1->heads_0 * p_1->queries_0))
    {
        return;
    }
    uint d_0 = (&kernelContext_0)->p_0->headDim_0;
    uint _S1 = row_0 * (&kernelContext_0)->p_0->headDim_0;
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
        uint device* _S2 = (&kernelContext_0)->o_0+(_S1 / 2U + c_0 / 2U);
        uint _S3 = _S1 + c_0;
        float _S4 = acc_0 + *((&kernelContext_0)->dO_0+_S3) * attnLo_0(*_S2) + *((&kernelContext_0)->dO_0+(_S3 + 1U)) * attnHi_0(*_S2);
        uint c_1 = c_0 + 2U;
        acc_0 = _S4;
        c_0 = c_1;
    }
    *((&kernelContext_0)->dD_0+row_0) = acc_0;
    return;
}

