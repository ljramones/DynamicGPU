// Fixed-order group reductions and scans over a groupshared array of THREADS elements (a power of two). The order of
// combination is fixed by the tree, never by scheduling, so a float sum is the same bits on every run and device
// that rounds IEEE operations as written (ShaderBuild's NoContraction and -O0 rules).
//
// Groupshared rules (Slang's SPIR-V target may reread a local copy of a groupshared value after a barrier):
// - after DG_GROUP_REDUCE, read the result from red[0] into a dedicated groupshared variable from thread 0, barrier,
//   and read that variable (DG_GROUP_BROADCAST) before writing `red` again;
// - never copy a groupshared element to a local and then overwrite the element while the local is live.
//
// OP is a function name: dg_op_add, dg_op_max, dg_op_min, or the caller's own (T, T) -> T.
#pragma once
#include "dynamis/numeric.slang"

float dg_op_add(float a, float b) { return a + b; }
// Java semantics (dg_max, dg_min), so a reduction gives the same bits everywhere, NaNs and zeros included.
float dg_op_max(float a, float b) { return dg_max(a, b); }
float dg_op_min(float a, float b) { return dg_min(a, b); }
uint dg_op_addu(uint a, uint b) { return a + b; }

// red[0] = red[0] OP red[1] OP ... in a fixed tree: pairs (t, t + width) for width = THREADS/2 .. 1. Every thread
// of the group must execute it. Ends with a barrier.
#define DG_GROUP_REDUCE(red, t, THREADS, OP)                         \
    {                                                                \
        [ForceUnroll]                                                \
        for (uint dgWidth = (THREADS) / 2; dgWidth > 0; dgWidth /= 2) \
        {                                                            \
            GroupMemoryBarrierWithGroupSync();                       \
            if ((t) < dgWidth)                                       \
                red[t] = OP(red[t], red[(t) + dgWidth]);             \
        }                                                            \
        GroupMemoryBarrierWithGroupSync();                           \
    }

// shared = red[0] from thread 0, then a barrier; read `shared` afterwards. Every thread must execute it.
#define DG_GROUP_BROADCAST(red, t, shared)                           \
    {                                                                \
        if ((t) == 0) shared = red[0];                               \
        GroupMemoryBarrierWithGroupSync();                           \
    }

// An exclusive scan of buf[0 .. THREADS-1] in place (Blelloch: an up-sweep tree, then a down-sweep), so
// buf[i] = buf[0] OP ... OP buf[i-1] and buf[0] = IDENTITY. The combination order is fixed. Every thread must execute
// it. Ends with a barrier.
#define DG_GROUP_SCAN_EXCLUSIVE(buf, t, THREADS, OP, IDENTITY)        \
    {                                                                \
        [ForceUnroll]                                                \
        for (uint dgD = 1; dgD < (THREADS); dgD *= 2)                \
        {                                                            \
            GroupMemoryBarrierWithGroupSync();                       \
            uint dgI = ((t) + 1) * dgD * 2 - 1;                      \
            if (dgI < (THREADS))                                     \
                buf[dgI] = OP(buf[dgI - dgD], buf[dgI]);             \
        }                                                            \
        GroupMemoryBarrierWithGroupSync();                           \
        if ((t) == 0) buf[(THREADS) - 1] = IDENTITY;                 \
        [ForceUnroll]                                                \
        for (uint dgD2 = (THREADS) / 2; dgD2 > 0; dgD2 /= 2)         \
        {                                                            \
            GroupMemoryBarrierWithGroupSync();                       \
            uint dgI2 = ((t) + 1) * dgD2 * 2 - 1;                    \
            if (dgI2 < (THREADS))                                    \
            {                                                        \
                uint dgL = dgI2 - dgD2;                              \
                /* combine before overwriting: no local copy of */   \
                /* a groupshared element outlives its overwrite */   \
                let dgSum = OP(buf[dgI2], buf[dgL]);                 \
                buf[dgL] = buf[dgI2];                                \
                buf[dgI2] = dgSum;                                   \
            }                                                        \
        }                                                            \
        GroupMemoryBarrierWithGroupSync();                           \
    }

// Subgroup reductions where the pipeline requires SUBGROUP_ARITHMETIC. Integer sums are exact in any order. Float max
// and min are too for ordinary values, but the hardware's handling of NaN and signed zeros is the device's, so a
// reduction that must match dg_max/dg_min bit for bit uses the tree above. Float sums always use the tree.
#ifdef DG_SUBGROUP_ARITHMETIC
float dg_subgroup_max(float x) { return WaveActiveMax(x); }
float dg_subgroup_min(float x) { return WaveActiveMin(x); }
uint dg_subgroup_addu(uint x) { return WaveActiveSum(x); }
#endif
