// Reduce.maxBackward (1.14.0): the gradient of a max along the last dimension of a rank-4 view (the reduced axis moved
// last on the host): dx[.., j] = j == indices[..] ? dy[..] : 0, routing each row's gradient to the index its argmax
// chose (the first largest, the first NaN). One thread per dx element. indices and dy are broadcast inputs (an offset
// and strides, stride 0 along the reduced axis), read with dx's coordinates.
#include "dynamis/kit.slang"

#define RB_BROADCAST_FIELDS(V) uint V##Offset; uint V##Stride0; uint V##Stride1; uint V##Stride2; uint V##Stride3;
#define RB_BROADCAST_AT(P, V, C) (P.V##Offset + C.x * P.V##Stride0 + C.y * P.V##Stride1 + C.z * P.V##Stride2 + C.w * P.V##Stride3)

struct Params
{
    uint count;
    uint gridX;
    uint iLength;   // buffer lengths in elements, for debug builds
    uint dyLength;
    uint dxLength;
    RB_BROADCAST_FIELDS(i)
    RB_BROADCAST_FIELDS(dy)
    DG_VIEW_FIELDS4(dx)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dy;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dx;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> errors;   // debug builds only
[[vk::push_constant]] ConstantBuffer<Params> p;

static const uint THREADS = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void reduce_max_backward(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint id = dg_linear_thread(group, local, p.gridX, THREADS);
    if (id >= p.count) return;
    DgView dv = DG_VIEW4(p, dx);
    uint4 c = dg_coords(dv, id);
    uint ii = RB_BROADCAST_AT(p, i, c), gi = RB_BROADCAST_AT(p, dy, c), xi = dg_at(dv, c);
    if (!dg_in_range(errors, ii, p.iLength, id) || !dg_in_range(errors, gi, p.dyLength, id) || !dg_in_range(errors, xi, p.dxLength, id)) return;
    dx[xi] = c.w == indices[ii] ? dy[gi] : 0.0f;
}
