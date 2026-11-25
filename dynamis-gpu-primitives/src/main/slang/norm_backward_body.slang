// Norms.backward (1.12.0), shared by norm_backward_f32.slang and _f16.slang (which define IN_T, the type of v, and
// ENTRY): for each row of v (the values the forward pass normalized), the output gradient dy and the weight w,
//     g = w dy,  xhat = (v - mean) inv,  inv = rsqrt(var + eps)   (RMSNorm: mean = 0, var = sum v^2 / n)
//     RMSNorm:   dv = inv (g - xhat sum(g xhat) / n)
//     LayerNorm: dv = inv (g - sum(g) / n - xhat sum(g xhat) / n)
// written to dv, or added to it (accumulate: the residual stream's gradient). The weight's and bias's gradients,
// sum over rows of dy xhat and of dy, go to per-block partials: a group takes ROWS consecutive rows, and the thread
// that owns column j adds row after row into partial[block][j] in order; norm_backward_columns.slang then adds the
// blocks in order. Sums over a row are the kit's fixed tree, so a device's result does not depend on scheduling;
// rsqrt and the divisions are the device's, so the result is bounded.
#include "dynamis/kit.slang"

struct NormBackwardParams
{
    uint rows;
    uint n;
    uint ldv;
    uint lddy;
    uint lddv;
    float eps;
    uint layer;        // 0 RMSNorm, 1 LayerNorm
    uint accumulate;   // dv += instead of dv =
    uint gridX;
    uint partials;     // write the weight and bias partials
};

[[vk::binding(0, 0)]] RWStructuredBuffer<IN_T> v;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> w;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dy;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dv;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> partW;   // [blocks][n]
[[vk::binding(5, 0)]] RWStructuredBuffer<float> partB;   // [blocks][n], LayerNorm
[[vk::push_constant]] ConstantBuffer<NormBackwardParams> p;

static const uint THREADS = 256;
static const uint ROWS = 16;
groupshared float red[THREADS];
groupshared float rowStat;

float groupSum(uint t, float partial)
{
    red[t] = partial;
    DG_GROUP_REDUCE(red, t, THREADS, dg_op_add);
    DG_GROUP_BROADCAST(red, t, rowStat);
    float s = rowStat;
    GroupMemoryBarrierWithGroupSync();
    return s;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint block = dg_linear_group(group, p.gridX);
    uint r0 = block * ROWS;
    if (r0 >= p.rows) return;   // uniform across the group
    uint t = local.x, n = p.n, rEnd = min(p.rows, r0 + ROWS);
    float fn = float(n);
    for (uint r = r0; r < rEnd; r++)
    {
        uint vb = r * p.ldv, db = r * p.lddy;
        // the forward pass's statistics, as it computed them
        float partial = 0.0f;
        for (uint j = t; j < n; j += THREADS) { float x = float(v[vb + j]); partial = p.layer != 0 ? partial + x : partial + x * x; }
        float first = groupSum(t, partial);
        float mean = 0.0f, var;
        if (p.layer != 0)
        {
            mean = first / fn;
            partial = 0.0f;
            for (uint j = t; j < n; j += THREADS) { float d = float(v[vb + j]) - mean; partial = partial + d * d; }
            var = groupSum(t, partial) / fn;
        }
        else var = first / fn;
        float inv = rsqrt(var + p.eps);
        // sum(g xhat) and, for LayerNorm, sum(g)
        float pgx = 0.0f, pg = 0.0f;
        for (uint j = t; j < n; j += THREADS)
        {
            float g = w[j] * dy[db + j], xh = (float(v[vb + j]) - mean) * inv;
            pgx = pgx + g * xh;
            pg = pg + g;
        }
        float sgx = groupSum(t, pgx) / fn;
        float sg = p.layer != 0 ? groupSum(t, pg) / fn : 0.0f;
        for (uint j = t; j < n; j += THREADS)
        {
            float d = dy[db + j], g = w[j] * d, xh = (float(v[vb + j]) - mean) * inv;
            float grad = inv * (g - sg - xh * sgx);
            uint at = r * p.lddv + j;
            dv[at] = p.accumulate != 0 ? dv[at] + grad : grad;
            if (p.partials != 0)
            {
                uint pa = block * n + j;
                float cw = d * xh;
                partW[pa] = r == r0 ? cw : partW[pa] + cw;
                if (p.layer != 0) partB[pa] = r == r0 ? d : partB[pa] + d;
            }
        }
    }
}
