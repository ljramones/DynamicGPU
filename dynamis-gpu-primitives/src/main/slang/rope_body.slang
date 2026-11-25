// Rope (1.10.0): rotary position embedding of each head of each token, shared by rope_<in>[_h].slang (IN_T, OUT_T,
// ENTRY). Token t's heads start at t * ldx (head h at h * strideHead); element pairs (a, b) of a head rotate by the
// table's (cos, sin) for the token's position (positions[t], from a GPU buffer) and pair i:
//     a' = a cos - b sin,    b' = a sin + b cos
// with the pair i < rotaryDim / 2 at (2i, 2i + 1) (layout 0, interleaved, GPT-J) or (i, i + rotaryDim / 2) (layout 1,
// split halves, GPT-NeoX and llama). Elements from rotaryDim to headDim pass through. The table (cos, sin as FP32,
// positions x rotaryDim / 2) is computed on the host in double precision, so the kernel is two correctly rounded
// products and one unfused add each: the same bits on every device (Rope.reference). y may be x itself, bound once
// (inPlace; x then read through y).
//
// One thread per pair (and per two passed-through elements). A position beyond the table is reported
// (DG_ERROR_INDEX, index = the token) and its token left untouched.
#include "dynamis/kit.slang"

struct RopeParams
{
    uint tokens;
    uint heads;
    uint headDim;
    uint rotaryDim;
    uint ldx;          // elements between tokens of x
    uint ldy;          // and of y
    uint strideHead;   // elements between heads
    uint layout;       // 0 interleaved, 1 split halves
    uint positions;    // rows of the table
    uint count;        // tokens * heads * headDim / 2 threads
    uint gridX;
    uint inPlace;      // y is x (one binding: x is read through y)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<IN_T> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> pos;
[[vk::binding(2, 0)]] RWStructuredBuffer<float2> table;
[[vk::binding(3, 0)]] RWStructuredBuffer<OUT_T> y;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> errors;
[[vk::push_constant]] ConstantBuffer<RopeParams> p;

static const uint THREADS = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void ENTRY(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint id = dg_linear_thread(group, local, p.gridX, THREADS);
    if (id >= p.count) return;
    uint halfHead = p.headDim / 2, i = id % halfHead, head = (id / halfHead) % p.heads, tok = id / halfHead / p.heads;
    uint position = pos[tok];
    if (position >= p.positions)
    {
        if (i == 0 && head == 0) dg_report_error(errors, DG_ERROR_INDEX, tok);
        return;
    }
    uint bx = tok * p.ldx + head * p.strideHead, by = tok * p.ldy + head * p.strideHead;
    uint halfRot = p.rotaryDim / 2;
    if (i < halfRot)
    {
        uint ia = p.layout == 0 ? 2 * i : i, ib = p.layout == 0 ? 2 * i + 1 : i + halfRot;
        float a = p.inPlace != 0 ? float(y[by + ia]) : float(x[bx + ia]);
        float b = p.inPlace != 0 ? float(y[by + ib]) : float(x[bx + ib]);
        float2 cs = table[position * halfRot + i];
        y[by + ia] = OUT_T(a * cs.x - b * cs.y);
        y[by + ib] = OUT_T(a * cs.y + b * cs.x);
    }
    else
    {
        uint e = p.rotaryDim + 2 * (i - halfRot);   // two passed-through elements
        if (p.inPlace == 0)
        {
            y[by + e] = OUT_T(float(x[bx + e]));
            y[by + e + 1] = OUT_T(float(x[bx + e + 1]));
        }
    }
}
