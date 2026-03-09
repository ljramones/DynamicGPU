# Ray Tracing Resource Foundation

This pass adds the DynamisGPU-side RT geometry resource seam.

## Ownership Boundary

- MeshForge owns RT metadata preparation and payload contract definition.
- DynamisGPU now owns RT payload ingestion/validation and GPU resource creation.
- BLAS/TLAS construction remains deferred.
- Renderer integration remains deferred.
- Material/shader policy remains deferred.

## Added API Contracts

- `GpuRayTracingGeometryPayload`
  - little-endian payload contract
  - v1 layout: 5 int32 values per region
  - order: `submeshIndex`, `firstIndex`, `indexCount`, `materialSlot`, `flags`
- `RayTracingGeometryPayloadIngestion`
  - ingestion seam for upstream metadata/byte consistency checks
  - resource materialization helpers from `GpuBuffer`/`GpuBufferHandle`
- `GpuRayTracingGeometryResource`
  - GPU-managed resource representation with payload metadata
- `GpuRayTracingGeometryUploader`
  - upload interface for backend implementations

## Vulkan Upload Path

- `VulkanRayTracingGeometryUploader`
  - creates/uploads non-empty RT payload buffers using existing Vulkan memory/upload conventions
  - validates uploaded buffer size against payload byte size

## Deferred

- BLAS construction capability
- TLAS construction capability
- shader binding table resources
- renderer/frame-graph integration
- RT material/shader policy

This establishes the RT geometry resource boundary so later phases can implement acceleration-structure execution without moving responsibilities across subsystem boundaries.

