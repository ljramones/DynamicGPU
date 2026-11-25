// GradientClip.squaredNorm, pass 2 (1.14.0): out[0] = the sum of the n partials, one group: thread t folds partials
// t, t + 256, ... in order, then the fixed tree. The same bits on every device.
#include "dynamis/kit.slang"

struct Params
{
    uint n;
    uint pLength;   // for debug builds
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> partial;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> out;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;
groupshared float red[THREADS];

[shader("compute")]
[numthreads(256, 1, 1)]
void clip_combine(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = local.x;
    float acc = 0.0f;
    for (uint i = t; i < p.n; i += THREADS)
    {
        if (dg_in_range(errors, i, p.pLength, 0)) acc = acc + partial[i];
    }
    red[t] = acc;
    DG_GROUP_REDUCE(red, t, THREADS, dg_op_add)
    if (t == 0) out[0] = red[0];
}
