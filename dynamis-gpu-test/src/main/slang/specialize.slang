// Specialization (1.7.0): o[i] = +-(a[i] * SCALE + OFFSET + BIAS), with one constant of each type (uint, float, bool,
// int) and 64 floats of group-shared staging, so tests can check both specialization and the reported limits.
struct Params { uint n; };
[[vk::binding(0, 0)]] RWStructuredBuffer<float> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> o;
[[vk::push_constant]] ConstantBuffer<Params> p;
[vk::constant_id(0)] const uint OFFSET = 0;
[vk::constant_id(1)] const float SCALE = 1.0;
[vk::constant_id(2)] const bool NEGATE = false;
[vk::constant_id(3)] const int BIAS = 0;
groupshared float stage[64];
[shader("compute")] [numthreads(64, 1, 1)]
void specialize(uint3 id : SV_DispatchThreadID, uint3 lt : SV_GroupThreadID) {
    uint i = id.x;
    stage[lt.x] = i < p.n ? a[i] : 0.0f;
    GroupMemoryBarrierWithGroupSync();
    if (i >= p.n) return;
    float v = stage[lt.x] * SCALE + float(OFFSET) + float(BIAS);
    o[i] = NEGATE ? -v : v;
}
