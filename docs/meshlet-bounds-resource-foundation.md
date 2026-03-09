# Meshlet Bounds Resource Foundation

## Purpose
Define the DynamisGPU-side resource ingestion seam for meshlet bounds produced by MeshForge.

This pass introduces resource modeling and contract ingestion only. It does not add visibility compute execution.

## Boundary Ownership
- MeshForge owns payload preparation and contract definition.
- DynamisGPU owns GPU resource creation, upload, binding, dispatch, synchronization, and execution.

This document describes the DynamisGPU side of that handoff.

## MeshForge Contract Consumed
Authoritative upstream payload contract:
- 6 floats per meshlet
- order: `minX, minY, minZ, maxX, maxY, maxZ`
- stride: 6 floats / 24 bytes
- little-endian direct `ByteBuffer` export
- explicit count/stride/size metadata

## DynamisGPU Foundation Introduced

### 1) `GpuMeshletBoundsPayload` (API resource model)
Immutable ingestion model with:
- `meshletCount`
- `boundsOffsetFloats`
- `boundsStrideFloats`
- read-only little-endian bounds bytes
- metadata helpers:
  - `boundsStrideBytes()`
  - `boundsFloatCount()`
  - `expectedBoundsFloatCount()`
  - `boundsByteSize()`

Validation enforces count/stride/byte consistency at ingestion.

### 2) `MeshletBoundsPayloadIngestion` (contract ingestion seam)
Consumes upstream contract metadata and byte payload as authoritative input:
- `meshletCount`
- `boundsOffsetFloats`
- `boundsStrideFloats`
- `boundsFloatCount`
- `expectedBoundsFloatCount`
- little-endian bounds bytes

The ingestion seam validates metadata consistency before creating `GpuMeshletBoundsPayload`.

### 3) `GpuMeshletBoundsResource` (GPU-managed representation)
Resource model carrying:
- `GpuBufferHandle`
- `GpuMeshletBoundsPayload`
This provides the minimal GPU-side identity + payload metadata seam.

## Deferred (Not Implemented in This Pass)
- compute shader dispatch
- visibility flag/list generation on GPU
- draw compaction or indirect draw integration
- renderer coupling

## Next Step
Implement compute visibility in DynamisGPU against `GpuMeshletBoundsPayload`/`GpuMeshletBoundsResource` as the formal handoff boundary.
