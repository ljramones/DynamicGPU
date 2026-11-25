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

float dg_op_add_0(float a_0, float b_0)
{
    return a_0 + b_0;
}

struct NormBackwardParams_0
{
    uint rows_0;
    uint n_0;
    uint ldv_0;
    uint lddy_0;
    uint lddv_0;
    float eps_0;
    uint layer_0;
    uint accumulate_0;
    uint gridX_1;
    uint partials_0;
};

struct KernelContext_0
{
    NormBackwardParams_0 constant* p_0;
    float device* v_0;
    float device* w_0;
    float device* dy_0;
    float device* dv_0;
    float device* partW_0;
    float device* partB_0;
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

[[kernel]] void norm_backward_f32(uint3 group_1 [[threadgroup_position_in_grid]], uint3 local_0 [[thread_position_in_threadgroup]], NormBackwardParams_0 constant* p_1 [[buffer(6)]], float device* v_1 [[buffer(0)]], float device* w_1 [[buffer(1)]], float device* dy_1 [[buffer(2)]], float device* dv_1 [[buffer(3)]], float device* partW_1 [[buffer(4)]], float device* partB_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->p_0 = p_1;
    (&kernelContext_1)->v_0 = v_1;
    (&kernelContext_1)->w_0 = w_1;
    (&kernelContext_1)->dy_0 = dy_1;
    (&kernelContext_1)->dv_0 = dv_1;
    (&kernelContext_1)->partW_0 = partW_1;
    (&kernelContext_1)->partB_0 = partB_1;
    threadgroup array<float, int(256)> red_1;
    (&kernelContext_1)->red_0 = &red_1;
    threadgroup float rowStat_1;
    (&kernelContext_1)->rowStat_0 = &rowStat_1;
    uint block_0 = dg_linear_group_0(group_1, p_1->gridX_1);
    uint r0_0 = block_0 * 16U;
    if(r0_0 >= (p_1->rows_0))
    {
        return;
    }
    uint _S2 = local_0.x;
    uint n_1 = (&kernelContext_1)->p_0->n_0;
    uint _S3 = min(p_1->rows_0, r0_0 + 16U);
    float _S4 = float((&kernelContext_1)->p_0->n_0);
    uint r_0 = r0_0;
    for(;;)
    {
        if(r_0 < _S3)
        {
        }
        else
        {
            break;
        }
        uint _S5 = r_0 * (&kernelContext_1)->p_0->ldv_0;
        uint _S6 = r_0 * (&kernelContext_1)->p_0->lddy_0;
        float partial_1 = 0.0;
        uint j_0 = _S2;
        for(;;)
        {
            if(j_0 < n_1)
            {
            }
            else
            {
                break;
            }
            float x_0 = *((&kernelContext_1)->v_0+(_S5 + j_0));
            if(((&kernelContext_1)->p_0->layer_0) != 0U)
            {
                partial_1 = partial_1 + x_0;
            }
            else
            {
                partial_1 = partial_1 + x_0 * x_0;
            }
            j_0 = j_0 + 256U;
        }
        float _S7 = groupSum_0(_S2, partial_1, &kernelContext_1);
        uint j_1;
        float var_0;
        float mean_0;
        if(((&kernelContext_1)->p_0->layer_0) != 0U)
        {
            float _S8 = _S7 / _S4;
            var_0 = 0.0;
            j_1 = _S2;
            for(;;)
            {
                if(j_1 < n_1)
                {
                }
                else
                {
                    break;
                }
                float d_0 = *((&kernelContext_1)->v_0+(_S5 + j_1)) - _S8;
                float _S9 = var_0 + d_0 * d_0;
                uint j_2 = j_1 + 256U;
                var_0 = _S9;
                j_1 = j_2;
            }
            float _S10 = groupSum_0(_S2, var_0, &kernelContext_1);
            var_0 = _S10 / _S4;
            mean_0 = _S8;
        }
        else
        {
            var_0 = _S7 / _S4;
            mean_0 = 0.0;
        }
        float _S11 = rsqrt(var_0 + (&kernelContext_1)->p_0->eps_0);
        float pgx_0 = 0.0;
        float pg_0 = 0.0;
        j_1 = _S2;
        for(;;)
        {
            if(j_1 < n_1)
            {
            }
            else
            {
                break;
            }
            float g_0 = *((&kernelContext_1)->w_0+j_1) * *((&kernelContext_1)->dy_0+(_S6 + j_1));
            float _S12 = pgx_0 + g_0 * ((*((&kernelContext_1)->v_0+(_S5 + j_1)) - mean_0) * _S11);
            float _S13 = pg_0 + g_0;
            uint j_3 = j_1 + 256U;
            pgx_0 = _S12;
            pg_0 = _S13;
            j_1 = j_3;
        }
        float _S14 = groupSum_0(_S2, pgx_0, &kernelContext_1);
        float _S15 = _S14 / _S4;
        float _S16;
        if(((&kernelContext_1)->p_0->layer_0) != 0U)
        {
            float _S17 = groupSum_0(_S2, pg_0, &kernelContext_1);
            _S16 = _S17 / _S4;
        }
        else
        {
            _S16 = 0.0;
        }
        uint j_4 = _S2;
        for(;;)
        {
            if(j_4 < n_1)
            {
            }
            else
            {
                break;
            }
            float device* _S18 = (&kernelContext_1)->dy_0+(_S6 + j_4);
            float d_1 = *_S18;
            float xh_0 = (*((&kernelContext_1)->v_0+(_S5 + j_4)) - mean_0) * _S11;
            float grad_0 = _S11 * (*((&kernelContext_1)->w_0+j_4) * *_S18 - _S16 - xh_0 * _S15);
            uint at_0 = r_0 * (&kernelContext_1)->p_0->lddv_0 + j_4;
            float device* _S19 = (&kernelContext_1)->dv_0+at_0;
            float _S20;
            if(((&kernelContext_1)->p_0->accumulate_0) != 0U)
            {
                _S20 = *((&kernelContext_1)->dv_0+at_0) + grad_0;
            }
            else
            {
                _S20 = grad_0;
            }
            *_S19 = _S20;
            if(((&kernelContext_1)->p_0->partials_0) != 0U)
            {
                uint pa_0 = block_0 * n_1 + j_4;
                float cw_0 = d_1 * xh_0;
                float device* _S21 = (&kernelContext_1)->partW_0+pa_0;
                bool _S22 = r_0 == r0_0;
                float _S23;
                if(_S22)
                {
                    _S23 = cw_0;
                }
                else
                {
                    _S23 = *((&kernelContext_1)->partW_0+pa_0) + cw_0;
                }
                *_S21 = _S23;
                if(((&kernelContext_1)->p_0->layer_0) != 0U)
                {
                    float device* _S24 = (&kernelContext_1)->partB_0+pa_0;
                    float _S25;
                    if(_S22)
                    {
                        _S25 = d_1;
                    }
                    else
                    {
                        _S25 = *((&kernelContext_1)->partB_0+pa_0) + d_1;
                    }
                    *_S24 = _S25;
                }
            }
            j_4 = j_4 + 256U;
        }
        r_0 = r_0 + 1U;
    }
    return;
}

