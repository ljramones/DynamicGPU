# Meshlet LOD Selection Capability

This pass adds a minimal standalone DynamisGPU Meshlet LOD selection capability.

## What was added

- `MeshletLodSelectionCapability`
- `MeshletLodSelectionWork`
- `GpuSelectedMeshletLodPayload`
- `GpuSelectedMeshletLodResource`
- `VulkanMeshletLodSelectionCapability`

## Input contract

- uploaded `GpuMeshletLodResource`
- `targetLodLevel` (integer)

## Selection rule (v1)

The implementation uses a deterministic correctness-first rule:

- select the exact level where `lodLevel == targetLodLevel`
- return that level's `meshletStart`, `meshletCount`, and `geometricErrorBits`
- fail with `IllegalArgumentException` when the target level is not present

This keeps policy minimal and testable while preserving pull-compatible scheduling.

## Output contract

`GpuSelectedMeshletLodResource` contains one selected level payload:

- `selectedLodLevel`
- `meshletStart`
- `meshletCount`
- `geometricErrorBits`

Payload layout is little-endian int32 x4.

## Deferred

- LightEngine/renderer policy integration
- visibility + LOD fusion
- advanced screen-space/error policy
- streaming integration

This capability is intentionally standalone and below renderer policy.
