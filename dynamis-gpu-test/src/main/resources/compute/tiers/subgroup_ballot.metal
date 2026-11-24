#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint4 WaveActiveBallot_0(bool condition_0)
{
    ulong _S1 = (uint64_t(simd_ballot((condition_0))));
    return uint4(uint(_S1 & 4294967295ULL), uint(_S1 >> 32U), 0U, 0U);
}

struct Params_0
{
    uint n_0;
};

struct KernelContext_0
{
    Params_0 constant* p_0;
    uint device* o_0;
    uint _S2;
    uint _S3;
};

uint _WaveCountBits_0(uint4 value_0, KernelContext_0 thread* kernelContext_0)
{
    switch((kernelContext_0->_S3 - 1U) / 32U)
    {
    case 0U:
    default:
        {
            return ((uint)popcount((value_0.x)));
        }
    case 1U:
        {
            return ((uint)popcount((value_0.x))) + ((uint)popcount((value_0.y)));
        }
    case 2U:
        {
            return ((uint)popcount((value_0.x))) + ((uint)popcount((value_0.y))) + ((uint)popcount((value_0.z)));
        }
    case 3U:
        {
            return ((uint)popcount((value_0.x))) + ((uint)popcount((value_0.y))) + ((uint)popcount((value_0.z))) + ((uint)popcount((value_0.w)));
        }
    }
}

uint WaveActiveCountBits_0(bool value_1, KernelContext_0 thread* kernelContext_1)
{
    uint4 _S4 = WaveActiveBallot_0(value_1);
    uint _S5 = _WaveCountBits_0(_S4, kernelContext_1);
    return _S5;
}

[[kernel]] void subgroup_ballot(uint3 id_0 [[thread_position_in_grid]], uint _builtinWaveLaneIndex_0 [[thread_index_in_simdgroup]], uint _builtinWaveLaneCount_0 [[threads_per_simdgroup]], Params_0 constant* p_1 [[buffer(1)]], uint device* o_1 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->p_0 = p_1;
    (&kernelContext_2)->o_0 = o_1;
    (&kernelContext_2)->_S2 = _builtinWaveLaneIndex_0;
    (&kernelContext_2)->_S3 = _builtinWaveLaneCount_0;
    uint t_0 = id_0.x;
    uint _S6 = WaveActiveCountBits_0((_builtinWaveLaneIndex_0 % 3U) == 0U, &kernelContext_2);
    uint first_0 = simd_broadcast_first(t_0);
    if(t_0 >= ((&kernelContext_2)->p_0->n_0))
    {
        return;
    }
    uint _S7 = 3U * t_0;
    *((&kernelContext_2)->o_0+_S7) = _builtinWaveLaneIndex_0;
    *((&kernelContext_2)->o_0+(_S7 + 1U)) = _S6;
    *((&kernelContext_2)->o_0+(_S7 + 2U)) = first_0;
    return;
}

