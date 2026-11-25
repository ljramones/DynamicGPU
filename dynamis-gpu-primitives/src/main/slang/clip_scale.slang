// GradientClip.clip (1.14.0): x *= f in place over a view, with f = maxNorm / sqrt(s) when sqrt(s) > maxNorm and 1
// otherwise (also when s is NaN), s the squared norm in sumSq[0] (GPU-resident: no host round trip). One thread per
// element; f is computed by every thread from the same value, so it is uniform.
#include "dynamis/kit.slang"

struct Params
{
    uint count;
    uint gridX;
    float maxNorm;
    uint xLength;   // for debug builds
    DG_VIEW_FIELDS4(x)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> sumSq;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> x;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void clip_scale(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint i = dg_linear_thread(group, local, p.gridX, THREADS);
    if (i >= p.count) return;
    float norm = sqrt(sumSq[0]);
    if (!(norm > p.maxNorm)) return;   // nothing to scale (NaN norm too)
    float f = p.maxNorm / norm;
    uint xi = dg_at_linear(DG_VIEW4(p, x), i);
    if (dg_in_range(errors, xi, p.xLength, i)) x[xi] = x[xi] * f;
}
