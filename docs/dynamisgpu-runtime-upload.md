# DynamisGPU Runtime Upload (Phase 1)

## Scope
Phase 1 establishes a minimal, backend-neutral runtime geometry upload foundation:

- consume opaque packed geometry bytes plus explicit metadata
- create GPU-owned mesh resources
- keep ownership and disposal explicit

Phase 1 does not include bindless integration, meshlets, streaming systems, or renderer graph orchestration.

## Architectural Boundaries
- Mesh/domain semantics are upstream concerns (MeshForge and bridge layers).
- DynamisGPU upload contracts only transport:
  - vertex bytes
  - optional index bytes
  - vertex layout metadata
  - index type metadata
  - submesh draw ranges
- Backend-specific mapping (for example Vulkan usage/memory flags) remains in backend modules.

## Ownership Model
- `GpuGeometryUploadPlan` is caller-owned input.
- `GpuUploadExecutor` consumes a plan and returns `GpuMeshResource`.
- `GpuMeshResource` owns resulting GPU resources and must be closed by the caller.
- `GpuBuffer` instances are resource owners and must be closed.

## Canonical Phase 1 Types
Public API contracts are split into focused packages:

- `org.dynamisengine.gpu.api.buffer`
  - `GpuBuffer`
  - `GpuBufferHandle`
  - `GpuBufferUsage`
  - `GpuMemoryLocation`
- `org.dynamisengine.gpu.api.layout`
  - `VertexFormat`
  - `VertexAttribute`
  - `VertexLayout`
  - `IndexType`
  - `SubmeshRange`
- `org.dynamisengine.gpu.api.upload`
  - `GpuGeometryUploadPlan`
  - `GpuUploadExecutor`
- `org.dynamisengine.gpu.api.resource`
  - `GpuMeshResource`

## Reuse and Deferred Integration
- Vulkan backend should build on existing memory primitives (`VulkanMemoryOps`, `VulkanBufferAlloc`).
- Existing specialized contracts (indirect/culling/skinning/bindless) remain valid for their domains but are not the canonical runtime-geometry upload API.
- Descriptor and bindless systems are deferred to later phases.

## Integration Harness Seam
The benchmark module provides a fixture-independent ingestion harness with a plan-supplier seam:

- `GpuGeometryUploadPlanSupplier` supplies plans from any source
- `GpuIngestionHarnessRunner` measures:
  - upload-plan intake and validation
  - upload execution
  - total executor time
- `DefaultGeometryUploadValidation` validates:
  - vertex count and stride
  - index count and index type
  - submesh range bounds
  - uploaded resource metadata consistency

This seam is intentionally generic so MeshForge wiring can be added without redesigning the harness shape.

## MeshForge Binding
When MeshForge modules are available, the benchmark harness can bind real classes through:

- `RuntimeGeometryLoader` (cache-hit load path)
- `RuntimeGeometryPayload`
- `MeshForgeRuntimePlanAdapter` (payload -> DynamisGPU `GpuGeometryUploadPlan`)
- `MeshForgeIngestionHarnessRunner` (cache-hit load + bridge + upload + total timing)

Current adapter behavior:
- maps MeshForge layout entries to DynamisGPU vertex attributes in deterministic entry order
- maps MeshForge index types to DynamisGPU index types
- maps submesh index ranges with `baseVertex=0` for Phase 1
