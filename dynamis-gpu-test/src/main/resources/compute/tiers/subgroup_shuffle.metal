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

[[kernel]] void subgroup_shuffle(uint3 id_0 [[thread_position_in_grid]], uint _builtinWaveLaneCount_0 [[threads_per_simdgroup]], uint _builtinWaveLaneIndex_0 [[thread_index_in_simdgroup]], Params_0 constant* p_1 [[buffer(1)]], uint device* o_1 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->p_0 = p_1;
    (&kernelContext_0)->o_0 = o_1;
    (&kernelContext_0)->_S1 = _builtinWaveLaneCount_0;
    (&kernelContext_0)->_S2 = _builtinWaveLaneIndex_0;
    uint t_0 = id_0.x;
    uint _S3 = (_builtinWaveLaneIndex_0 + 1U) % _builtinWaveLaneCount_0;
    uint next_0 = (simd_shuffle((t_0), ushort((int(_S3)))));
    if(t_0 >= ((&kernelContext_0)->p_0->n_0))
    {
        return;
    }
    uint _S4 = 3U * t_0;
    *((&kernelContext_0)->o_0+_S4) = _builtinWaveLaneCount_0;
    *((&kernelContext_0)->o_0+(_S4 + 1U)) = _builtinWaveLaneIndex_0;
    *((&kernelContext_0)->o_0+(_S4 + 2U)) = next_0;
    return;
}

