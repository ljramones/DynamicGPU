# Meshlet LOD Resource Foundation

This pass adds the DynamisGPU-side Meshlet LOD resource seam.

## Ownership Boundary

- MeshForge owns Meshlet LOD metadata preparation and payload contract definition.
- DynamisGPU now owns LOD payload ingestion/validation and GPU resource creation.
- LOD selection policy remains deferred.
- Renderer integration remains deferred.
- Streaming remains deferred.

## Added API Contracts

- `GpuMeshletLodPayload`
  - little-endian payload contract
  - v1 layout: 4 int32 values per level
  - order: `lodLevel`, `meshletStart`, `meshletCount`, `geometricErrorBits`
- `MeshletLodPayloadIngestion`
  - ingestion seam for upstream metadata/byte consistency checks
  - resource materialization helpers from `GpuBuffer`/`GpuBufferHandle`
- `GpuMeshletLodResource`
  - GPU-managed resource representation with payload metadata
- `GpuMeshletLodUploader`
  - upload interface for backend implementations

## Vulkan Upload Path

- `VulkanMeshletLodUploader`
  - creates/uploads non-empty LOD payload buffers using existing Vulkan memory/upload conventions
  - validates uploaded buffer size against payload byte size

## Deferred

- GPU-side LOD selection execution
- LOD policy (screen-space/error thresholds)
- renderer/frame-graph integration
- streaming integration

This establishes the LOD resource boundary so later phases can implement execution and policy without moving responsibilities across subsystem boundaries.
