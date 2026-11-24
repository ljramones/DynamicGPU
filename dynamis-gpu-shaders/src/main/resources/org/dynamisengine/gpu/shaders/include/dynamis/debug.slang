// Debug builds (Java: kit.KitOptions.debug). DG_DEBUG is a specialization constant: normal pipelines leave it false and
// the backend compiler removes every check; debug pipelines set it and check each buffer access against the buffer's
// length (in elements, a parameter), reporting DG_ERROR_INDEX with the caller's index (a thread or row) through the
// kernel error buffer instead of reading or writing out of range. With pooled memory an out-of-range write lands in a
// neighbouring buffer, so these checks catch kernel indexing bugs that would otherwise corrupt other data silently.
#pragma once
#include "dynamis/errors.slang"

// Specialization-constant id 100 is reserved for DG_DEBUG in every kit kernel.
[vk::constant_id(100)] const bool DG_DEBUG = false;

// Whether index < length; in a debug build, an index out of range is reported (at `at`) and false returned.
bool dg_in_range(RWStructuredBuffer<uint> errors, uint index, uint length, uint at)
{
    if (!DG_DEBUG || index < length) return true;
    dg_report_error(errors, DG_ERROR_INDEX, at);
    return false;
}
