// GradientClip.squaredNorm, pass 1 (1.14.0): the sum of squares of a view's elements, in chunks of CHUNK elements (in
// row-major order), one group per chunk: thread t folds x^2 for elements t, t + 256, ... of its chunk in order, the
// group combines the 256 partials in DG_GROUP_REDUCE's fixed tree, and writes partial[base + chunk]. Each tensor of the
// list is one dispatch at its own base. The same bits on every device (GradientClip.referenceSquaredNorm).
#include "dynamis/kit.slang"

struct Params
{
    uint count;      // elements of the view
    uint gridX;
    uint chunks;     // groups
    uint base;       // this tensor's first partial
    uint xLength;    // for debug builds
    uint pLength;
    DG_VIEW_FIELDS4(x)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> partial;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;
static const uint CHUNK = 4096;
groupshared float red[THREADS];

[shader("compute")]
[numthreads(256, 1, 1)]
void clip_sumsq(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint chunk = dg_linear_group(group, p.gridX);
    if (chunk >= p.chunks) return;   // uniform across the group
    uint t = local.x;
    DgView xv = DG_VIEW4(p, x);
    uint first = chunk * CHUNK, end = min(p.count, first + CHUNK);
    float acc = 0.0f;
    for (uint i = first + t; i < end; i += THREADS)
    {
        uint xi = dg_at_linear(xv, i);
        if (dg_in_range(errors, xi, p.xLength, chunk))
        {
            float v = x[xi];
            acc = acc + v * v;
        }
    }
    red[t] = acc;
    DG_GROUP_REDUCE(red, t, THREADS, dg_op_add)
    if (t == 0 && dg_in_range(errors, p.base + chunk, p.pLength, chunk)) partial[p.base + chunk] = red[0];
}
