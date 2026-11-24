// The kernel error buffer (Java: org.dynamisengine.gpu.api.compute.KernelErrors). Bind it as RWStructuredBuffer<uint>.
// dg_report_error: code is a nonzero, application-defined bit mask; the buffer keeps the OR of the codes, the smallest
// index and the count, whatever order the threads run in.
#pragma once

// Codes the kit's own kernels report.
static const uint DG_ERROR_INDEX = 1u;      // an index read from data is out of range (debug builds)
static const uint DG_ERROR_NONFINITE = 2u;  // a NaN or infinity was found

void dg_report_error(RWStructuredBuffer<uint> errors, uint code, uint index)
{
    InterlockedOr(errors[0], code);
    InterlockedMax(errors[1], ~index);
    InterlockedAdd(errors[2], 1u);
}
