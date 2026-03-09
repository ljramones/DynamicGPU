# Ray Tracing BLAS Capability Foundation

## Scope

This pass adds a minimal, pull-compatible BLAS capability foundation inside DynamisGPU.

It introduces:

- `RayTracingBlasCapability`
- `RayTracingBlasWork`
- `GpuRayTracingBlasPayload`
- `GpuRayTracingBlasResource`
- `VulkanRayTracingBlasCapability`

The capability consumes an uploaded `GpuRayTracingGeometryResource` and produces a BLAS-side resource seam.

## What This Pass Implements

This pass implements a **BLAS build-preparation seam**, not full acceleration-structure execution.

Current behavior:

1. validate BLAS work input and source geometry resource state
2. derive deterministic BLAS build-prep payload metadata
3. upload build-prep payload bytes into a Vulkan GPU buffer
4. return `GpuRayTracingBlasResource` containing:
   - uploaded buffer handle
   - build-prep payload metadata
   - source geometry resource linkage

BLAS payload layout v1:

- `regionCount` (int32)
- `regionsStrideBytes` (int32)
- `regionsByteSize` (int32)
- `reservedFlags` (int32)

## Why This Shape

This keeps RT work aligned with existing subsystem boundaries:

- MeshForge prepares RT geometry metadata.
- DynamisGPU ingests/uploads and executes backend capability slices.
- Renderer policy remains outside this subsystem pass.

The BLAS seam is intentionally narrow so TLAS and later RT execution layers can build on a stable contract.

## Validation Added

Focused tests cover:

- BLAS work contract validation
- BLAS payload invariants
- BLAS resource lifecycle semantics
- Vulkan capability behavior for:
  - valid execution
  - closed geometry input rejection
  - upload size mismatch rejection

## Deferred

Still deferred by design:

- real Vulkan BLAS build command execution
- TLAS construction
- shader binding table work
- RT renderer/frame-graph integration
- material/shader policy integration

