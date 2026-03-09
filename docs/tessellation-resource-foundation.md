# Tessellation Resource Foundation

Date: 2026-03-09

## Purpose

This pass adds the DynamisGPU-side tessellation resource foundation that consumes the finalized MeshForge tessellation payload contract and represents it as a GPU-managed resource.

Scope is intentionally limited to ingestion, validation, upload, and resource ownership.

## Added API Contracts

- `GpuTessellationPayload`
- `TessellationPayloadIngestion`
- `GpuTessellationResource`
- `GpuTessellationUploader`

Payload layout (int32 per region, stride 6):

1. `submeshIndex`
2. `firstIndex`
3. `indexCount`
4. `patchControlPoints`
5. `tessLevelBits` (`Float.floatToRawIntBits`)
6. `flags`

## Vulkan Upload Seam

Added:

- `VulkanTessellationUploader`

Behavior:

- accepts validated `GpuTessellationPayload`
- rejects empty payload uploads
- uploads little-endian region bytes to a device-local GPU buffer using staging
- returns `GpuTessellationResource`

## Validation and Tests

Coverage added for:

- payload metadata/stride/byte contract
- ingestion count and size consistency checks
- resource metadata propagation and lifecycle
- Vulkan uploader success and failure cases

## Boundary

MeshForge remains responsible for tessellation metadata preparation and payload definition.

DynamisGPU now owns:

- tessellation payload ingestion
- tessellation payload validation
- tessellation resource upload/representation

## Deferred

Still deferred by design:

- GPU tessellation/subdivision execution
- adaptive tessellation policy
- renderer/frame-graph integration
- material/shader tessellation policy
