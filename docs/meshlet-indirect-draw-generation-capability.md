# Meshlet Indirect Draw Generation Capability (Phase 2.5)

## Scope

This pass adds the minimal GPU capability that turns compact visible meshlet indices into
indexed-indirect draw commands.

It consumes:

- `GpuVisibleMeshletListResource` (compact visible meshlet indices)
- `GpuMeshletDrawMetadataPayload` (per-meshlet draw fields)
- `MeshletIndirectDrawGenerationWork` (resource + metadata + command count)

It produces:

- `GpuMeshletIndirectDrawResource`
  - packed `VkDrawIndexedIndirectCommand`-layout records
  - command count metadata

## Added Contracts

- `MeshletIndirectDrawGenerationCapability`
- `MeshletIndirectDrawGenerationWork`
- `GpuMeshletDrawMetadataPayload`
- `GpuMeshletIndirectDrawPayload`
- `GpuMeshletIndirectDrawResource`

This remains pull-compatible and scheduler-friendly.

## Command Format and Assumptions

Each generated command uses this field order:

1. `indexCount`
2. `instanceCount` (fixed to `1` in this phase)
3. `firstIndex`
4. `vertexOffset`
5. `firstInstance` (fixed to `0` in this phase)

Draw metadata is supplied by `GpuMeshletDrawMetadataPayload`.

## Vulkan Execution Slice

`VulkanMeshletIndirectDrawGenerationCapability` is the minimal backend implementation for this phase.

Behavior:

1. Read compact visible meshlet indices in stable order.
2. Look up per-meshlet draw metadata.
3. Emit one packed indirect command per visible meshlet.
4. Upload command bytes to a GPU-managed indirect buffer resource.

## Correctness Validation

Focused tests validate:

- mixed visible list produces expected command sequence and count
- empty visible list produces zero commands
- capability lifecycle guard (no execution after close)

## Deferred Items

This pass intentionally does **not** include:

- renderer or frame-graph integration
- material/state sorting and batching policies
- multi-draw optimization policies
- occlusion/cull cone/hierarchy/LOD/streaming
- broad scheduler redesign

This completes the basic GPU chain:
meshlet bounds -> visibility flags -> compact visible list -> indirect draw command resource.

