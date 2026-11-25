// Softmax's backward pass (1.12.0), shared by softmax_backward_f32.slang and _f16.slang (which define IN_T, the
// type of y, and ENTRY): for each row r of y = softmax(s) and the gradient dy (both at ldy, strideY), with the
// forward pass's mask,
//     dx[j] = y[j] (dy[j] - sum_j' dy[j'] y[j'])   over the unmasked j,   dx[j] = 0 where masked,
// written at lds, strideS (the scores' layout). One group of 256 threads per row (group x = row, group z = matrix);
// thread t takes columns t, t + 256, .. in order and the sum is combined by the same fixed tree as the forward pass,
// so the result does not depend on scheduling and every step is one correctly rounded operation: bounded by
// |y| (k 2^-24 sum |dy y| + 2^-24 |dy - sum|) + 2^-24 |dx|, k = ceil(n / 256) + 9.
struct SoftmaxParams
{
    uint m;
    uint n;
    uint lds;
    uint ldy;
    uint causal;
    uint causalOffset;
    uint strideS;
    uint strideY;
    uint lengths;       // 1.14.0: valid lengths, as the forward pass
    uint lengthGroup;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<IN_T> y;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dy;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dx;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> lens;
[[vk::push_constant]] ConstantBuffer<SoftmaxParams> p;

static const uint THREADS = 256;
groupshared float red[THREADS];
groupshared float rowSum;

[shader("compute")]
[numthreads(256, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = local.x, r = group.x;
    uint rowX = group.z * p.strideS + r * p.lds, rowY = group.z * p.strideY + r * p.ldy;
    uint limit = p.causal != 0 ? min(p.n, r + p.causalOffset + 1) : p.n;
    if (p.lengths != 0) limit = min(limit, lens[group.z / p.lengthGroup]);

    float acc = 0.0f;
    for (uint j = t; j < limit; j += THREADS) acc = acc + dy[rowY + j] * float(y[rowY + j]);
    red[t] = acc;
    [ForceUnroll]
    for (uint width = THREADS / 2; width > 0; width /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        if (t < width) red[t] = red[t] + red[t + width];
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) rowSum = red[0];
    GroupMemoryBarrierWithGroupSync();
    float sum = rowSum;
    for (uint j = t; j < p.n; j += THREADS)
        dx[rowX + j] = j < limit ? float(y[rowY + j]) * (dy[rowY + j] - sum) : 0.0f;
}
