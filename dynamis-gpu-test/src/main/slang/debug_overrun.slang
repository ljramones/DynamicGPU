// Debug builds (1.7.0): o[i + shift] = a[i] for i < n, every write checked against o's length by the kit's
// dg_in_range. With DG_DEBUG set, the writes past the end are reported (DG_ERROR_INDEX, the smallest i) and skipped.
// Tests run a shift past the end only in debug pipelines: a normal pipeline would write out of range.
#include "dynamis/kit.slang"
struct Params { uint n; uint shift; uint oLength; };
[[vk::binding(0, 0)]] RWStructuredBuffer<float> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> o;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> errors;
[[vk::push_constant]] ConstantBuffer<Params> p;
[shader("compute")] [numthreads(64, 1, 1)]
void debug_overrun(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= p.n) return;
    uint at = i + p.shift;
    if (!dg_in_range(errors, at, p.oLength, i)) return;
    o[at] = a[i];
}
