// The sampler's merge tree (1.10.1), shared by sample_merge.slang and sample_final.slang, which declare SLOTS (a
// multiple of THREADS) and the group arrays sv and si of SLOTS entries before including it.
//
// mergeTree: the first `slots` entries hold sorted lists of `c` entries each (value descending, then index; a power of
// two of lists); at each level, lists A and B (adjacent, equal lengths) become one, each entry moving to its index plus
// the count of the other list's entries that go before it (a binary search; for equal entries, A's go first, so
// places are unique). The order is total, so the result is the reference's sort.
#pragma once

static const uint PER_THREAD = SLOTS / THREADS;

// entries of the sorted list at [b, b + len) that go before (v, i): strictly before, or (orEqual) not after
uint countBefore(uint b, uint len, float v, uint i, bool orEqual)
{
    uint lo = 0, hi = len;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2;
        float mv = sv[b + mid]; uint mi = si[b + mid];
        bool goesBefore = before(mv, mi, v, i) || (orEqual && !before(v, i, mv, mi));
        if (goesBefore) lo = mid + 1; else hi = mid;
    }
    return lo;
}

void mergeTree(uint t, uint c, uint slots)
{
    for (uint len = c; len < slots; len *= 2)
    {
        float v[PER_THREAD]; uint ix[PER_THREAD], to[PER_THREAD];
        [ForceUnroll]
        for (uint r = 0; r < PER_THREAD; r++)
        {
            uint s = t + r * THREADS;
            to[r] = NONE; v[r] = 0.0f; ix[r] = NONE;
            if (s >= slots) continue;
            uint pb = (s / (2 * len)) * 2 * len, inPair = s % (2 * len);
            bool inA = inPair < len;
            uint q = inA ? inPair : inPair - len;
            v[r] = sv[s]; ix[r] = si[s];
            // A's entries count B's strictly before them; B's count A's not after them
            uint other = inA ? countBefore(pb + len, len, v[r], ix[r], false) : countBefore(pb, len, v[r], ix[r], true);
            to[r] = pb + q + other;
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint r = 0; r < PER_THREAD; r++) if (to[r] != NONE) { sv[to[r]] = v[r]; si[to[r]] = ix[r]; }
        GroupMemoryBarrierWithGroupSync();
    }
}

// Loads the first c entries of `count` lists (list l of the row at (row * lists + first + l) * stride) into the
// first `slots` slots, the rest -infinity
#define LOAD_LISTS(inV, inI, row, lists, first, count, stride, c, slots, t)          \
    [ForceUnroll]                                                                     \
    for (uint dgR = 0; dgR < PER_THREAD; dgR++)                                      \
    {                                                                                 \
        uint dgS = (t) + dgR * THREADS, dgL = dgS / (c), dgQ = dgS % (c);             \
        bool dgValid = dgS < (slots) && dgL < (count);                                \
        uint dgAt = ((row) * (lists) + (first) + dgL) * (stride) + dgQ;               \
        sv[dgS] = dgValid ? inV[dgAt] : dg_negative_infinity();                       \
        si[dgS] = dgValid ? inI[dgAt] : NONE;                                         \
    }                                                                                 \
    GroupMemoryBarrierWithGroupSync();
