# Ray Tracing BLAS Execution

## Scope

This pass upgrades BLAS from build-prep metadata upload to a real Vulkan BLAS build path when
required RT procedures are available on the device.

## What Changed

### BLAS Work Input

`RayTracingBlasWork` now consumes a resolved build-input resource:

- `GpuRayTracingBuildInputResource`

This gives BLAS execution direct access to:

- validated RT geometry metadata
- resolved vertex/index device addresses
- build-range assumptions prepared at ingestion

### BLAS Capability Execution

`VulkanRayTracingBlasCapability` now:

1. checks required Vulkan RT procedures are available:
   - `vkCreateAccelerationStructureKHR`
   - `vkGetAccelerationStructureBuildSizesKHR`
   - `vkCmdBuildAccelerationStructuresKHR`
   - `vkGetBufferDeviceAddress`
2. parses RT regions into build ranges
3. creates BLAS geometry/build structures
4. queries build sizes
5. allocates:
   - acceleration-structure storage buffer
   - scratch buffer
   both with device-address-capable allocation semantics
6. creates acceleration structure
7. records/submits `vkCmdBuildAccelerationStructuresKHR`
8. returns a `GpuRayTracingBlasResource` with a real acceleration-structure handle

### Resource Semantics

`GpuRayTracingBlasResource` now carries:

- optional `accelerationStructureHandle`
- `hasAccelerationStructure()` helper
- close behavior that can destroy backend AS resources through a provided close action

## Validation

Focused tests cover:

- BLAS work contract changes
- BLAS resource handle semantics
- capability execution contract through injected executor paths
- prerequisite build-input and resolver tests from prior slice remain active

## Deferred

Still deferred by design:

- TLAS construction
- shader binding / material policy
- renderer integration

