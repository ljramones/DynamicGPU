# Meshlet Streaming Resource Foundation

This pass adds the DynamisGPU-side geometry streaming resource seam.

## Ownership Boundary

- MeshForge owns streaming metadata preparation and payload contract definition.
- DynamisGPU now owns streaming payload ingestion/validation and GPU resource creation.
- Residency/orchestration policy remains deferred.
- Renderer integration remains deferred.
- Visibility/LOD/streaming fusion remains deferred.

## Added API Contracts

- `GpuMeshletStreamingPayload`
  - little-endian payload contract
  - v1 layout: 5 int32 values per stream unit
  - order: `streamUnitId`, `meshletStart`, `meshletCount`, `payloadByteOffset`, `payloadByteSize`
- `MeshletStreamingPayloadIngestion`
  - ingestion seam for upstream metadata/byte consistency checks
  - resource materialization helpers from `GpuBuffer`/`GpuBufferHandle`
- `GpuMeshletStreamingResource`
  - GPU-managed resource representation with payload metadata
- `GpuMeshletStreamingUploader`
  - upload interface for backend implementations

## Vulkan Upload Path

- `VulkanMeshletStreamingUploader`
  - creates/uploads non-empty streaming payload buffers using existing Vulkan memory/upload conventions
  - validates uploaded buffer size against payload byte size

## Deferred

- Residency state machine and request orchestration
- Streaming scheduler/paging policy
- renderer/frame-graph integration
- Visibility/LOD/streaming composition

This establishes the streaming resource boundary so later phases can implement residency execution and policy without moving responsibilities across subsystem boundaries.
