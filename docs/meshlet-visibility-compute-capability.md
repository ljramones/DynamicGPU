# Meshlet Visibility Compute Capability (Phase 2.3)

## Scope

This pass adds the first minimal GPU visibility execution capability in DynamisGPU.

It consumes:

- uploaded meshlet bounds resources (`GpuMeshletBoundsResource`)
- frustum plane payload (`MeshletVisibilityFrustum`, 6 planes x 4 floats)
- meshlet work descriptor (`MeshletVisibilityWork`)

It produces:

- one visibility flag per meshlet (`GpuMeshletVisibilityFlagsResource`)
  - `1` = visible
  - `0` = culled

## Added Capability/Work Contracts

- `MeshletVisibilityCapability`
- `MeshletVisibilityFrustum`
- `MeshletVisibilityWork`
- `GpuMeshletVisibilityFlagsPayload`
- `GpuMeshletVisibilityFlagsResource`

The work is represented explicitly, so the capability can be scheduled by a pull loop in a later phase without introducing renderer-facing push APIs.

## Vulkan Execution Slice

`VulkanMeshletVisibilityCapability` is the minimal Vulkan backend implementation for this phase.

Behavior per meshlet:

1. Read one AABB from meshlet bounds payload (authoritative MeshForge layout).
2. Test AABB against 6 frustum planes.
3. Write one visibility flag byte.
4. Upload visibility flags into a GPU-managed storage buffer.

## Correctness Validation

Focused tests validate:

- mixed visibility output
- all-visible and all-culled cases
- output size equals meshlet count
- lifecycle guard (capability cannot execute after close)

Validation uses controlled synthetic bounds/frustum data to compare expected flags.

## Deferred Items

This pass intentionally does **not** include:

- generic scheduler integration
- visible-list compaction
- indirect draw generation
- renderer/frame-graph integration
- occlusion, cone culling, hierarchy/BVH, LOD, or streaming

These remain for later phases.

