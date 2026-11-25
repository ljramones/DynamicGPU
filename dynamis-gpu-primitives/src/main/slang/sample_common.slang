// What the two sampling kernels share (1.10.0): the parameters, the order-preserving key, the draw, and the ordering
// of candidates. Logits are sanitized first: NaN counts as -infinity and -0 as +0, so the order by value, the order
// by key and the reference's sort all agree.
#include "dynamis/kit.slang"

struct SampleParams
{
    uint rows;
    uint vocab;
    uint ldl;
    float invT;      // 1 / temperature, 0 for greedy
    uint topK;       // 0: none; else 1 .. 1024
    float topP;      // >= 1: none
    uint seedLo;
    uint seedHi;
    uint step;
    uint chunks;     // ceil(vocab / 1024)
    uint restricted; // top-k or top-p
    uint lists;      // sample_merge: input lists per row; sample_final: 1
    uint stride;     // elements between lists (1024 for the chunks, head for merged lists)
    uint head;       // entries of a list that matter: min(k, 1024)
    uint perGroup;   // sample_merge: lists one group merges
};

static const uint THREADS = 1024;
static const uint CHUNK = 1024;
static const uint RUN = 256;     // stage 1's sorted runs
static const uint NONE = 0xFFFFFFFFu;

float sanitize(float l) { return dg_is_nan(l) ? dg_negative_infinity() : (l == 0.0f ? 0.0f : l); }
// order-preserving key of a sanitized value
uint key(float v) { uint f = asuint(v); return (f & 0x80000000u) != 0 ? ~f : f | 0x80000000u; }
// (va, ia) before (vb, ib): larger value first, then lower index (NONE last)
bool before(float va, uint ia, float vb, uint ib) { return va > vb || (va == vb && ia < ib); }

uint fmix(uint h) { h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16; return h; }
float rand01(uint seedLo, uint seedHi, uint row, uint step)
{
    uint h = fmix(seedLo ^ fmix(seedHi ^ fmix(row * 0x9E3779B9u ^ fmix(step + 0x632BE5ABu))));
    return float(h >> 8) * asfloat(0x33800000u);   // 2^-24
}
