# Meshlet Visibility Compaction Capability (Phase 2.4)

## Scope

This pass adds the first minimal visible-list compaction capability in DynamisGPU.

It consumes:

- `GpuMeshletVisibilityFlagsResource` (1 byte flag per meshlet)
- `MeshletVisibilityCompactionWork` (flags resource + source meshlet count)

It produces:

- `GpuVisibleMeshletListResource`
  - compacted little-endian `int32` meshlet indices
  - visible meshlet count metadata

## Added Capability/Work Contracts

- `MeshletVisibilityCompactionCapability`
- `MeshletVisibilityCompactionWork`
- `GpuVisibleMeshletListPayload`
- `GpuVisibleMeshletListResource`

The work is represented explicitly so this remains compatible with pull scheduling.

## Vulkan Execution Slice

`VulkanMeshletVisibilityCompactionCapability` is the minimal backend implementation for this phase.

Compaction behavior:

1. Read one visibility flag per meshlet.
2. For each visible meshlet (`flag != 0`), append meshlet index to output list.
3. Emit compacted visible index payload and visible count metadata.
4. Upload compacted index bytes into a GPU-managed storage buffer.

This is correctness-first and intentionally avoids broader scan/compaction frameworks.

## Correctness Validation

Focused tests validate:

- mixed visibility compaction preserves stable index order
- all-culled output is empty (`visibleCount = 0`)
- all-visible output is `[0..n-1]`
- capability lifecycle guard (no execution after close)

## Deferred Items

This pass intentionally does **not** include:

- indirect draw command generation
- renderer/frame-graph integration
- visibility list consumption by draw submission
- occlusion/cull cone/hierarchy/LOD/streaming
- broad scheduler redesign

Next phase can consume this compacted visible list for indirect draw generation.

