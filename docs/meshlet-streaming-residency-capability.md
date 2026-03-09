# Meshlet Streaming Residency Capability

This pass adds a minimal standalone DynamisGPU streaming residency capability.

## Ownership Boundary

- MeshForge owns stream-unit metadata preparation and payload contract definition.
- DynamisGPU now owns execution of a pull-compatible residency capability over uploaded streaming metadata.
- Streaming scheduler/orchestration remains deferred.
- Renderer integration remains deferred.
- Visibility/LOD/streaming fusion remains deferred.

## Added Contracts

- `MeshletStreamingResidencyCapability`
  - schedulable capability contract for stream-unit resolution
- `MeshletStreamingResidencyWork`
  - work-unit input with `GpuMeshletStreamingResource` + `targetStreamUnitId`
- `GpuResolvedMeshletStreamingPayload`
  - output payload contract for one resolved stream unit
  - v1 layout: `streamUnitId`, `meshletStart`, `meshletCount`, `payloadByteOffset`, `payloadByteSize`
- `GpuResolvedMeshletStreamingResource`
  - GPU-managed resolved residency output resource

## Vulkan Execution

- `VulkanMeshletStreamingResidencyCapability`
  - resolves the exact requested `targetStreamUnitId` from uploaded stream-unit metadata
  - uploads selected result payload to a storage buffer using existing Vulkan memory/upload conventions
  - validates output buffer size against payload byte size

## Validation Coverage

- work-model validation (`targetStreamUnitId >= 0`)
- valid stream-unit resolution across multiple units
- invalid stream-unit id rejection
- output metadata consistency (`meshletStart/count`, `payloadByteOffset/size`)
- output buffer size mismatch rejection

## Deferred

- residency state machine and eviction policy
- streaming request scheduler/orchestration
- visibility + LOD + streaming composition
- renderer/frame-graph integration

This establishes a correctness-first streaming residency execution seam that remains pull-compatible and ready for later orchestration phases.

