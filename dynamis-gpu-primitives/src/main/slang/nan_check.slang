// NanCheck: reports every NaN or infinity of a rank-4 view to a KernelErrors buffer, with code DG_ERROR_NONFINITE and
// the element's linear (row-major) index in the view; the buffer keeps the smallest. One thread per element.
#include "dynamis/kit.slang"

struct Params
{
    uint count;
    uint gridX;
    DG_VIEW_FIELDS4(x)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> errors;
[[vk::push_constant]] ConstantBuffer<Params> p;

[shader("compute")]
[numthreads(256, 1, 1)]
void nan_check(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint i = dg_linear_thread(group, local, p.gridX, 256);
    if (i >= p.count) return;
    if (!dg_is_finite(x[dg_at_linear(DG_VIEW4(p, x), i)])) dg_report_error(errors, DG_ERROR_NONFINITE, i);
}
