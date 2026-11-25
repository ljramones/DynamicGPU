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

struct NormColumnsParams_0
{
    uint n_0;
    uint blocks_0;
    uint bias_0;
    uint gridX_2;
};

struct KernelContext_0
{
    NormColumnsParams_0 constant* p_0;
    float device* partW_0;
    float device* partB_0;
    float device* dW_0;
    float device* dB_0;
};

[[kernel]] void norm_backward_columns(uint3 group_2 [[threadgroup_position_in_grid]], uint3 local_1 [[thread_position_in_threadgroup]], NormColumnsParams_0 constant* p_1 [[buffer(4)]], float device* partW_1 [[buffer(0)]], float device* partB_1 [[buffer(1)]], float device* dW_1 [[buffer(2)]], float device* dB_1 [[buffer(3)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->partW_0 = partW_1;
    (&kernelContext_0)->partB_0 = partB_1;
    (&kernelContext_0)->dW_0 = dW_1;
    (&kernelContext_0)->dB_0 = dB_1;
    uint j_0 = dg_linear_thread_0(group_2, local_1, p_1->gridX_2, 256U);
    if(j_0 >= (p_1->n_0))
    {
        return;
    }
    float sw_0 = 0.0;
    float sb_0 = 0.0;
    uint k_0 = 0U;
    for(;;)
    {
        if(k_0 < ((&kernelContext_0)->p_0->blocks_0))
        {
        }
        else
        {
            break;
        }
        float _S1 = sw_0 + *((&kernelContext_0)->partW_0+(k_0 * p_1->n_0 + j_0));
        if(((&kernelContext_0)->p_0->bias_0) != 0U)
        {
            sb_0 = sb_0 + *((&kernelContext_0)->partB_0+(k_0 * p_1->n_0 + j_0));
        }
        uint _S2 = k_0 + 1U;
        sw_0 = _S1;
        k_0 = _S2;
    }
    *((&kernelContext_0)->dW_0+j_0) = *((&kernelContext_0)->dW_0+j_0) + sw_0;
    if(((&kernelContext_0)->p_0->bias_0) != 0U)
    {
        *((&kernelContext_0)->dB_0+j_0) = *((&kernelContext_0)->dB_0+j_0) + sb_0;
    }
    return;
}

