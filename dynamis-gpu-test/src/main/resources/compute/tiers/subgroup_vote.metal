#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct Params_0
{
    uint n_0;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    uint device* o_0;
    uint _S1;
    uint _S2;
};

[[kernel]] void subgroup_vote(uint3 id_0 [[thread_position_in_grid]], uint _builtinWaveLaneIndex_0 [[thread_index_in_simdgroup]], uint _builtinWaveLaneCount_0 [[threads_per_simdgroup]], Params_0 constant* p_1 [[buffer(1)]], uint device* o_1 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->o_0 = o_1;
    (&kernelContext_0)->_S1 = _builtinWaveLaneIndex_0;
    (&kernelContext_0)->_S2 = _builtinWaveLaneCount_0;
    uint t_0 = id_0.x;
    bool _S3 = simd_all(_builtinWaveLaneIndex_0 < _builtinWaveLaneCount_0);
    uint _S4;
    if(_S3)
    {
        _S4 = 1U;
    }
    else
    {
        _S4 = 0U;
    }
    bool _S5 = _builtinWaveLaneIndex_0 == 0U;
    bool _S6 = simd_any(_S5);
    uint _S7;
    if(_S6)
    {
        _S7 = 2U;
    }
    else
    {
        _S7 = 0U;
    }
    uint _S8 = _S4 | _S7;
    bool _S9 = simd_all(_S5);
    if(_S9)
    {
        _S4 = 4U;
    }
    else
    {
        _S4 = 0U;
    }
    uint bits_0 = _S8 | _S4;
    if(t_0 >= ((&kernelContext_0)->p_0->n_0))
    {
        return;
    }
    *((&kernelContext_0)->o_0+t_0) = bits_0;
    return;
}

