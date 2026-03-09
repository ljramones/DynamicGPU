# Meshlet Bounds Upload Path (Phase 2.2)

## Purpose
Add the minimal DynamisGPU-side upload/create path that turns validated meshlet bounds payloads into GPU-managed buffer resources.

This pass does not add compute visibility execution.

## Previous Step
Phase 2.1 established:
- `GpuMeshletBoundsPayload`
- `GpuMeshletBoundsResource`
- `MeshletBoundsPayloadIngestion`

## What This Pass Adds

### 1) Upload Contract
`GpuMeshletBoundsUploader` defines the upload path interface:
- input: validated `GpuMeshletBoundsPayload`
- output: uploaded `GpuMeshletBoundsResource`

### 2) Vulkan Upload Implementation
`VulkanMeshletBoundsUploader` now:
- accepts payload bytes from `GpuMeshletBoundsPayload`
- creates a device-local Vulkan buffer using existing staging/copy conventions
- validates uploaded size matches payload byte size
- returns `GpuMeshletBoundsResource` bound to the real uploaded buffer

### 3) Resource Lifecycle
`GpuMeshletBoundsResource` now owns a `GpuBuffer` and is `AutoCloseable`, aligning with existing DynamisGPU resource lifecycle conventions.

## Layout Preserved
No reinterpretation of the MeshForge contract:
- 6 floats per meshlet
- order: `minX, minY, minZ, maxX, maxY, maxZ`
- little-endian bytes

## Deferred
Still intentionally deferred:
- compute dispatch
- visibility flags/list generation
- compaction/indirect draw integration
- renderer coupling

## Next Step
Phase 2.3: consume uploaded meshlet bounds resource in a minimal compute visibility pass.

