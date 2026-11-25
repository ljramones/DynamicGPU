// Row-wise softmax, shared by softmax_f32.slang and softmax_f16.slang (which define OUT_T and ENTRY): for each row r
// of an m x n matrix of FP32 scores (leading dimension lds; matrix z at z * strideS), with an optional causal mask,
//     y[j] = exp(s[j] - max) / sum_j' exp(s[j'] - max)   over the unmasked j,   y[j] = 0 where masked.
// Causal: column j is masked when j > r + causalOffset (query r sees keys 0 .. r + offset; with a KV cache of
// n keys and m new queries the offset is n - m). Valid lengths (1.14.0, a padding mask): with `lengths` set, matrix z
// also masks columns at or past lens[z / lengthGroup] (keys past a sequence's length). A row with every column masked
// is all zeros, and so (1.14.0) is a row whose unmasked scores are all -infinity, as an additive mask makes them.
//
// One group of 256 threads per row (group x = row, group z = matrix). Thread t takes columns t, t + 256, ...;
// the max and the sum are combined by a fixed tree over the threads, so the result does not depend on scheduling.
// exp is the device's: within Vulkan's 3 + 2|x| ulp (Metal compiles with precise functions), so results are bounded,
// not bit-identical across devices. Built by scripts/build-slang-kernels.sh.
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
    uint lengths;       // 1.14.0: 1 when lens holds valid lengths
    uint lengthGroup;   // matrices per length (for example heads per sequence)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> s;
[[vk::binding(1, 0)]] RWStructuredBuffer<OUT_T> y;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> lens;
[[vk::push_constant]] ConstantBuffer<SoftmaxParams> p;

static const uint THREADS = 256;
groupshared float red[THREADS];
// The row's max and sum get their own group variables: Slang's SPIR-V target may reread a local copy of a
// groupshared value after a barrier, so red[0] must not be copied and then overwritten.
groupshared float rowMax;
groupshared float rowSum;

// Tree reduction over red[0 .. 255] into red[0], max or sum, in a fixed order.
void reduce(uint t, bool isMax)
{
    [ForceUnroll]
    for (uint width = THREADS / 2; width > 0; width /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        if (t < width)
        {
            float a = red[t], b = red[t + width];
            red[t] = isMax ? max(a, b) : a + b;
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint t = local.x, r = group.x;
    uint rowS = group.z * p.strideS + r * p.lds, rowY = group.z * p.strideY + r * p.ldy;
    // columns 0 .. limit - 1 are unmasked (the host keeps r + causalOffset + 1 within 32 bits)
    uint limit = p.causal != 0 ? min(p.n, r + p.causalOffset + 1) : p.n;
    if (p.lengths != 0) limit = min(limit, lens[group.z / p.lengthGroup]);

    float local = asfloat(0xFF800000u); // -infinity
    for (uint j = t; j < limit; j += THREADS) local = max(local, s[rowS + j]);
    red[t] = local;
    reduce(t, true);
    if (t == 0) rowMax = red[0];
    GroupMemoryBarrierWithGroupSync();
    float mx = rowMax;

    local = 0.0f;
    for (uint j = t; j < limit; j += THREADS) local += exp(s[rowS + j] - mx);
    red[t] = local;
    reduce(t, false);
    if (t == 0) rowSum = red[0];
    GroupMemoryBarrierWithGroupSync();
    float sum = rowSum;

    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    bool none = asuint(mx) == 0xFF800000u;   // nothing unmasked, or every unmasked score -infinity: zeros
    for (uint j = t; j < p.n; j += THREADS)
        y[rowY + j] = OUT_T(j < limit && !none ? exp(s[rowS + j] - mx) * inv : 0.0f);
}
