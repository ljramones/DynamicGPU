# Ray Tracing TLAS Capability Foundation

## Scope

This pass adds a minimal pull-compatible TLAS capability that consumes built BLAS instances and
builds a real Vulkan TLAS when RT procedures are available.

## Added Models

- `GpuRayTracingTlasInstanceMetadata`
  - BLAS reference
  - 3x4 row-major transform (12 floats)
  - instance custom index
  - mask
  - SBT record offset
  - flags
- `RayTracingTlasWork`
- `RayTracingTlasCapability`
- `GpuRayTracingTlasPayload`
- `GpuRayTracingTlasResource`

## Vulkan TLAS Execution Path

`VulkanRayTracingTlasCapability` now:

1. validates required Vulkan RT procs:
   - `vkCreateAccelerationStructureKHR`
   - `vkGetAccelerationStructureBuildSizesKHR`
   - `vkCmdBuildAccelerationStructuresKHR`
   - `vkGetAccelerationStructureDeviceAddressKHR`
   - `vkGetBufferDeviceAddress`
2. resolves BLAS device addresses from input BLAS handles
3. serializes `VkAccelerationStructureInstanceKHR` data
4. uploads instance buffer with device-address-capable staging path
5. configures TLAS build geometry/range
6. queries TLAS build sizes
7. allocates TLAS storage + scratch buffers
8. creates TLAS handle and records/submits build command
9. returns `GpuRayTracingTlasResource` with a real TLAS handle

## Validation

Focused tests cover:

- TLAS instance metadata invariants
- TLAS work contract validation
- TLAS resource lifecycle/metadata
- capability contract via deterministic injected executor path

## Deferred

- shader binding/material policy
- renderer/frame-graph integration
- full RT pipeline integration

