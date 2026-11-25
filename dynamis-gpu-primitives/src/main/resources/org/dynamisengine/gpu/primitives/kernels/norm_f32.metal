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

float roundIn_0(float s_0)
{
    return s_0;
}

float dg_op_add_0(float a_0, float b_0)
{
    return a_0 + b_0;
}

struct NormParams_0
{
    uint rows_0;
    uint n_0;
    uint ldx_0;
    uint ldr_0;
    uint ldh_0;
    uint ldy_0;
    float eps_0;
    uint layer_0;
    uint residual_0;
    uint bias_0;
    uint gridX_1;
    uint inPlace_0;
};

struct KernelContext_0
{
    NormParams_0 constant* p_0;
    float device* h_0;
    float device* x_0;
    float device* res_0;
    float device* w_0;
    float device* b_1;
    float device* y_0;
    array<float, int(256)> threadgroup* red_0;
    float threadgroup* rowStat_0;
};

float groupSum_0(uint t_0, float partial_0, KernelContext_0 thread* kernelContext_0)
{
    (*kernelContext_0->red_0)[t_0] = partial_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 128U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 128U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 64U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 64U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 32U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 32U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 16U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 16U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 8U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 8U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 4U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 4U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 2U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 2U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < 1U)
    {
        (*kernelContext_0->red_0)[t_0] = dg_op_add_0((*kernelContext_0->red_0)[t_0], (*kernelContext_0->red_0)[t_0 + 1U]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 == 0U)
    {
        *kernelContext_0->rowStat_0 = (*kernelContext_0->red_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threadgroup* _S1 = kernelContext_0->rowStat_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *_S1;
}

float value_0(uint r_0, uint j_0, KernelContext_0 thread* kernelContext_1)
{
    float _S2;
    if((kernelContext_1->p_0->residual_0) != 0U)
    {
        _S2 = *(kernelContext_1->h_0+(r_0 * kernelContext_1->p_0->ldh_0 + j_0));
    }
    else
    {
        _S2 = *(kernelContext_1->x_0+(r_0 * kernelContext_1->p_0->ldx_0 + j_0));
    }
    return _S2;
}

[[kernel]] void norm_f32(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], NormParams_0 constant* p_1 [[buffer(6)]], float device* h_1 [[buffer(2)]], float device* x_1 [[buffer(0)]], float device* res_1 [[buffer(1)]], float device* w_1 [[buffer(3)]], float device* b_2 [[buffer(4)]], float device* y_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->p_0 = p_1;
    (&kernelContext_2)->h_0 = h_1;
    (&kernelContext_2)->x_0 = x_1;
    (&kernelContext_2)->res_0 = res_1;
    (&kernelContext_2)->w_0 = w_1;
    (&kernelContext_2)->b_1 = b_2;
    (&kernelContext_2)->y_0 = y_1;
    threadgroup array<float, int(256)> red_1;
    (&kernelContext_2)->red_0 = &red_1;
    threadgroup float rowStat_1;
    (&kernelContext_2)->rowStat_0 = &rowStat_1;
    uint r_1 = dg_linear_group_0(group_1, p_1->gridX_1);
    if(r_1 >= (p_1->rows_0))
    {
        return;
    }
    float v_0;
    float xv_0;
    uint t_1 = local_0.x;
    uint n_1 = (&kernelContext_2)->p_0->n_0;
    float partial_1 = 0.0;
    uint j_1 = t_1;
    for(;;)
    {
        if(j_1 < n_1)
        {
        }
        else
        {
            break;
        }
        if(((&kernelContext_2)->p_0->residual_0) != 0U)
        {
            if(((&kernelContext_2)->p_0->inPlace_0) != 0U)
            {
                xv_0 = *((&kernelContext_2)->h_0+(r_1 * (&kernelContext_2)->p_0->ldh_0 + j_1));
            }
            else
            {
                xv_0 = *((&kernelContext_2)->x_0+(r_1 * (&kernelContext_2)->p_0->ldx_0 + j_1));
            }
            float v_1 = roundIn_0(xv_0 + *((&kernelContext_2)->res_0+(r_1 * (&kernelContext_2)->p_0->ldr_0 + j_1)));
            *((&kernelContext_2)->h_0+(r_1 * (&kernelContext_2)->p_0->ldh_0 + j_1)) = v_1;
            v_0 = v_1;
        }
        else
        {
            v_0 = *((&kernelContext_2)->x_0+(r_1 * (&kernelContext_2)->p_0->ldx_0 + j_1));
        }
        if(((&kernelContext_2)->p_0->layer_0) != 0U)
        {
            partial_1 = partial_1 + v_0;
        }
        else
        {
            partial_1 = partial_1 + v_0 * v_0;
        }
        j_1 = j_1 + 256U;
    }
    float _S3 = groupSum_0(t_1, partial_1, &kernelContext_2);
    float mean_0;
    if(((&kernelContext_2)->p_0->layer_0) != 0U)
    {
        float _S4 = float(n_1);
        float _S5 = _S3 / _S4;
        partial_1 = 0.0;
        j_1 = t_1;
        for(;;)
        {
            if(j_1 < n_1)
            {
            }
            else
            {
                break;
            }
            float _S6 = value_0(r_1, j_1, &kernelContext_2);
            float d_0 = _S6 - _S5;
            float _S7 = partial_1 + d_0 * d_0;
            uint j_2 = j_1 + 256U;
            partial_1 = _S7;
            j_1 = j_2;
        }
        float _S8 = groupSum_0(t_1, partial_1, &kernelContext_2);
        xv_0 = _S8 / _S4;
        mean_0 = _S5;
    }
    else
    {
        xv_0 = _S3 / float(n_1);
        mean_0 = 0.0;
    }
    float _S9 = rsqrt(xv_0 + (&kernelContext_2)->p_0->eps_0);
    j_1 = t_1;
    for(;;)
    {
        if(j_1 < n_1)
        {
        }
        else
        {
            break;
        }
        float _S10 = value_0(r_1, j_1, &kernelContext_2);
        float v_2 = (_S10 - mean_0) * _S9 * *((&kernelContext_2)->w_0+j_1);
        if(((&kernelContext_2)->p_0->bias_0) != 0U)
        {
            v_0 = v_2 + *((&kernelContext_2)->b_1+j_1);
        }
        else
        {
            v_0 = v_2;
        }
        *((&kernelContext_2)->y_0+(r_1 * (&kernelContext_2)->p_0->ldy_0 + j_1)) = v_0;
        j_1 = j_1 + 256U;
    }
    return;
}

