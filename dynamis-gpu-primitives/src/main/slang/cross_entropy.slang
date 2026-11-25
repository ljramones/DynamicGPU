// CrossEntropy (1.12.0): with lse[r] = log sum exp of row r's logits (Reduce.logSumExp) and target t = targets[r],
//     loss[r] = lse[r] - logits[r][t]            (0 for an ignored row, t = 0xFFFFFFFF)
//     dl[r][j] = scale (exp(logits[r][j] - lse[r]) - [j == t])   (0 for an ignored row)
// grad = 0: a thread per row, the loss only. grad = 1: a thread per element; the target's thread writes the loss.
// In place, dl is the logits (read through dl, each element by the thread that overwrites it). A target past the
// vocabulary fails the submission (KernelErrors) and its row is left unwritten.
#include "dynamis/kit.slang"

struct CeParams
{
    uint rows;
    uint vocab;
    uint ld;        // logits' leading dimension
    uint lddl;      // dl's
    float scale;
    uint gridX;
    uint inPlace;
    uint grad;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> logits;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> targets;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> lse;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> loss;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> dl;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> errors;
[[vk::push_constant]] ConstantBuffer<CeParams> p;

static const uint IGNORE = 0xFFFFFFFFu;

[shader("compute")]
[numthreads(256, 1, 1)]
void cross_entropy(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint id = dg_linear_thread(group, local, p.gridX, 256);
    if (p.grad == 0)
    {
        if (id >= p.rows) return;
        uint t = targets[id];
        if (t == IGNORE) { loss[id] = 0.0f; return; }
        if (t >= p.vocab) { dg_report_error(errors, DG_ERROR_INDEX, id); return; }
        loss[id] = lse[id] - logits[id * p.ld + t];
        return;
    }
    if (id >= p.rows * p.vocab) return;
    uint r = id / p.vocab, j = id % p.vocab, t = targets[r];
    bool ignored = t == IGNORE;
    if (!ignored && t >= p.vocab) { if (j == 0) dg_report_error(errors, DG_ERROR_INDEX, r); return; }
    uint at = r * p.lddl + j;
    float l = p.inPlace != 0 ? dl[at] : logits[r * p.ld + j];
    if (ignored) { if (j == 0) loss[r] = 0.0f; dl[at] = 0.0f; return; }
    float m = lse[r];
    if (j == t) loss[r] = m - l;
    dl[at] = p.scale * (exp(l - m) - (j == t ? 1.0f : 0.0f));
}
