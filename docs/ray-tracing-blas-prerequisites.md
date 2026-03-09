# Ray Tracing BLAS Prerequisites

## Why This Pass Exists

Real Vulkan BLAS execution requires more than RT region metadata. The previous foundation carried
region descriptors but did not provide build-input-ready buffer linkage or device-address seams.

This pass adds the minimum prerequisites so the next slice can implement real BLAS execution
without contract ambiguity.

## Added In This Pass

### 1. RT Build-Input-Ready Contract

New API types:

- `GpuRayTracingBuildInputPayload`
- `GpuRayTracingBuildInputResource`
- `RayTracingBuildInputIngestion`

These extend the RT metadata path with:

- source `GpuRayTracingGeometryResource`
- explicit vertex/index `GpuBufferHandle` linkage
- vertex stride + max vertex index
- vertex/index data offsets
- validated per-region index range assumptions (`firstIndex >= 0`, `indexCount > 0`)

### 2. Vulkan Device Address Resolution Seam

New Vulkan type:

- `VulkanRayTracingBuildInputResolver`

This resolves backend-usable buffer device addresses from the linked buffer handles and produces
`GpuRayTracingBuildInputResource` with:

- vertex buffer device address
- index buffer device address

### 3. Device-Address-Capable Allocation Helpers

`VulkanMemoryOps` now includes:

- `createBuffer(..., memoryAllocateFlags)` overload
- `createDeviceAddressBufferWithStaging(...)`
- `getBufferDeviceAddress(...)`

This provides the minimal allocation/query seam needed for later BLAS storage/scratch resources
that require device address capability.

## Deferred (Still Intentionally Not In Scope)

- real BLAS build command recording/execution
- TLAS construction
- renderer integration
- shader binding/material policy

## Immediate Next Step

Implement real Vulkan BLAS execution using:

- `GpuRayTracingBuildInputResource` as the build-input seam
- device-address-capable allocation/query helpers in `VulkanMemoryOps`

