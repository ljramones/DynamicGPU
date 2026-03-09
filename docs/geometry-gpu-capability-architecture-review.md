# Geometry/GPU Capability Architecture Review

Date: 2026-03-09

## 1. Implemented Capability Inventory

### 1.1 MeshForge Prepared-Data Contracts

| Part | Repo/Module | Responsibility | Boundary Type |
|---|---|---|---|
| `MgiMeshDataCodec` + `RuntimeDecodeResult` | `MeshForge/meshforge-mgi` | MGI encode/decode adapter and runtime metadata exposure | Public cross-subsystem contract surface |
| `ops/cull/ViewFrustum`, `MeshletFrustumCuller` | `MeshForge/meshforge` | CPU baseline meshlet culling and measured correctness baseline | Internal implementation baseline |
| `MeshletLodMetadata`, `MeshletLodLevelMetadata` | `MeshForge/meshforge` | Prepared meshlet LOD metadata model | Public handoff contract |
| `MeshletStreamingMetadata`, `MeshletStreamUnitMetadata` | `MeshForge/meshforge` | Prepared streaming unit metadata model | Public handoff contract |
| `RayTracingGeometryMetadata`, `RayTracingGeometryRegionMetadata` | `MeshForge/meshforge` | RT region metadata for later BLAS/TLAS work | Public handoff contract |
| `TessellationMetadata`, `TessellationRegionMetadata` | `MeshForge/meshforge` | Tessellation/subdivision region metadata | Public handoff contract |
| `RuntimePayloadCompressionMode`, `RuntimePayloadCompression`, `CompressedRuntimePayload` | `MeshForge/meshforge` | Optional compression seam for selected payloads | Public handoff contract |

### 1.2 MGI Optional Chunks/Extensions

`MgiChunkType` and `MgiStaticMeshCodec` currently preserve optional metadata chunks for:

- `MESHLET_DESCRIPTORS`, `MESHLET_VERTEX_REMAP`, `MESHLET_TRIANGLES`, `MESHLET_BOUNDS`
- `MESHLET_LOD_LEVELS`
- `MESHLET_STREAM_UNITS`
- `RAY_TRACING_REGIONS`
- `TESSELLATION_REGIONS`

Ownership: MeshForge format layer (`meshforge-mgi`).
Boundary: Public persisted format contract.

### 1.3 MeshForge GPU Handoff Payloads

| Part | Repo/Module | Responsibility | Boundary Type |
|---|---|---|---|
| `GpuMeshletVisibilityPayload` + `MeshletVisibilityUploadPrep` | `MeshForge/meshforge` | Flatten bounds payload for GPU visibility | Public cross-repo handoff contract |
| `GpuMeshletLodPayload` + `MeshletLodUploadPrep` | `MeshForge/meshforge` | Flatten LOD levels payload | Public cross-repo handoff contract |
| `GpuMeshletStreamingPayload` + `MeshletStreamingUploadPrep` | `MeshForge/meshforge` | Flatten stream-unit payload | Public cross-repo handoff contract |
| `GpuRayTracingGeometryPayload` + `RayTracingGeometryUploadPrep` | `MeshForge/meshforge` | Flatten RT region payload | Public cross-repo handoff contract |
| `GpuTessellationPayload` + `TessellationUploadPrep` | `MeshForge/meshforge` | Flatten tessellation region payload | Public cross-repo handoff contract |

### 1.4 DynamisGPU Ingestion/Validation Seams

| Part | Repo/Module | Responsibility | Boundary Type |
|---|---|---|---|
| `MeshletBoundsPayloadIngestion` | `DynamisGPU/dynamis-gpu-api` | Ingest/validate bounds payload (+ optional compressed mode) | Public boundary adapter |
| `MeshletLodPayloadIngestion` | `DynamisGPU/dynamis-gpu-api` | Ingest/validate LOD payload | Public boundary adapter |
| `MeshletStreamingPayloadIngestion` | `DynamisGPU/dynamis-gpu-api` | Ingest/validate streaming payload (+ optional compressed mode) | Public boundary adapter |
| `RayTracingGeometryPayloadIngestion` | `DynamisGPU/dynamis-gpu-api` | Ingest/validate RT geometry payload | Public boundary adapter |
| `RayTracingBuildInputIngestion` | `DynamisGPU/dynamis-gpu-api` | Ingest/validate BLAS build-input payload | Public boundary adapter |
| `TessellationPayloadIngestion` | `DynamisGPU/dynamis-gpu-api` | Ingest/validate tessellation payload | Public boundary adapter |

### 1.5 DynamisGPU Upload/Resource Seams

| Part | Repo/Module | Responsibility | Boundary Type |
|---|---|---|---|
| `GpuMeshlet*Resource`, `GpuRayTracing*Resource`, `GpuTessellationResource` | `dynamis-gpu-api` | Stable GPU-managed resource holders over payload + buffer/handle | Public subsystem contract |
| `GpuMeshlet*Uploader`, `GpuRayTracingGeometryUploader`, `GpuTessellationUploader` | `dynamis-gpu-api` | Upload/create interfaces for validated payloads | Public subsystem contract |
| `VulkanMeshlet*Uploader`, `VulkanRayTracingGeometryUploader`, `VulkanTessellationUploader` | `dynamis-gpu-vulkan` | Vulkan upload implementation | Internal backend implementation detail |

### 1.6 Standalone GPU Capabilities (DynamisGPU)

| Capability | Repo/Module | Responsibility | Boundary Type |
|---|---|---|---|
| `MeshletVisibilityCapability` | `dynamis-gpu-api` + Vulkan impl | Per-meshlet frustum visibility flags | Public pull-compatible capability contract |
| `MeshletVisibilityCompactionCapability` | `dynamis-gpu-api` + Vulkan impl | Flags -> compact visible meshlet list | Public pull-compatible capability contract |
| `MeshletIndirectDrawGenerationCapability` | `dynamis-gpu-api` + Vulkan impl | Visible meshlets -> indirect draw commands | Public pull-compatible capability contract |
| `MeshletLodSelectionCapability` | `dynamis-gpu-api` + Vulkan impl | Minimal standalone LOD level selection | Public pull-compatible capability contract |
| `MeshletStreamingResidencyCapability` | `dynamis-gpu-api` + Vulkan impl | Minimal standalone streaming unit resolution | Public pull-compatible capability contract |
| `RayTracingBlasCapability` | `dynamis-gpu-api` + Vulkan impl | BLAS build path (real execution when supported) | Public pull-compatible capability contract |
| `RayTracingTlasCapability` | `dynamis-gpu-api` + Vulkan impl | TLAS build path (real execution when supported) | Public pull-compatible capability contract |

### 1.7 Validation Harnesses / Measurement Passes

| Part | Repo/Module | Responsibility | Boundary Type |
|---|---|---|---|
| `GpuDrivenClusterVisibilityValidationHarness` | `dynamis-gpu-api` | End-to-end correctness/timing for visibility->compaction->indirect chain | Internal validation harness |
| Compression activation measurements docs + tests | MeshForge + DynamisGPU docs/tests | Validate optional compression tradeoff | Internal decision artifact |
| MeshForge CPU culling benchmark docs | `MeshForge/docs` | Baseline and regression anchor for visibility behavior | Internal validation artifact |

## 2. Subsystem Ownership Map

### MeshForge owns

- Source ingest, canonicalization, and MGI persistence.
- Optional MGI chunk evolution for geometry metadata.
- Runtime metadata models for meshlet LOD/streaming/RT/tessellation.
- Deterministic CPU-side GPU payload shaping (`*UploadPrep` + payload records).
- Optional payload compression decisioning and encoding on MeshForge side.

MeshForge should not own:

- GPU upload orchestration.
- GPU execution capabilities (visibility, compaction, BLAS/TLAS, etc.).
- Renderer/frame-graph policy.

### DynamisGPU owns

- Ingestion and validation of finalized MeshForge payload contracts.
- Upload/create seams and GPU-managed resource lifecycle.
- Pull-compatible standalone compute capabilities.
- Backend-specific Vulkan execution details.

DynamisGPU should not own:

- Source mesh semantics or MGI authoring policy.
- High-level render planning policy.
- Scene-level feature policy (when to enable LOD/streaming/RT/tessellation).

### Deferred to DynamisLightEngine / higher policy layer

- Feature policy and composition decisions:
  - visibility + LOD + streaming fusion policy
  - pass/material/state policy
  - RT pipeline usage policy (SBT/material mapping, pass integration)
  - tessellation runtime policy (adaptive factors, pass selection)
- Renderer/frame-graph integration and frame-level orchestration.

## 3. Data Flow and Dependency Direction

Current intended direction is consistently one-way:

1. MeshForge core mesh -> MGI encode (`MgiStaticMeshCodec`)
2. MGI decode (`MgiMeshDataCodec`) -> runtime metadata
3. MeshForge `*UploadPrep` -> deterministic payload bytes/arrays
4. DynamisGPU ingestion (`*PayloadIngestion`) -> validated canonical payload
5. DynamisGPU upload (`*Uploader`) -> `Gpu*Resource`
6. DynamisGPU standalone capability -> output resources
7. Higher-level engine consumes outputs later (deferred)

Strong layering points:

- MeshForge payload contracts are explicit and deterministic.
- DynamisGPU ingestion boundaries are narrow and defensive.
- Capability contracts are pull-compatible and below renderer integration.
- Compression awareness is localized at ingestion boundaries in DynamisGPU.

Provisional/transitional points:

- Capability composition is partly explicit (validation harness) but not yet a single generalized composition runtime.
- Several resource/capability slices are intentionally standalone and not yet policy-integrated.
- Tessellation is currently resource-foundation only (no execution capability yet).

## 4. Boundary Findings: Clean vs Transitional

### Clean boundaries confirmed

- MeshForge defines payload shape; DynamisGPU consumes without reinterpreting semantics.
- MGI optional chunks preserve backward compatibility while enabling feature metadata growth.
- DynamisGPU compute capabilities are isolated from renderer integration concerns.
- RT progression is layered correctly: geometry resource -> build input -> BLAS -> TLAS.

### Transitional / awkward areas

1. Repeated payload/resource/ingestion boilerplate
- `Gpu*Payload`, `*PayloadIngestion`, `Gpu*Resource`, `Vulkan*Uploader` follow near-identical patterns.
- This is good for clarity now, but maintenance cost is increasing.

2. Naming divergence across slices
- Some classes use domain prefixes (`Meshlet*`, `RayTracing*`, `Tessellation*`) while similar concerns (payload count/stride/bytes) are not standardized by shared terminology.

3. Compression seam asymmetry
- Compression was activated selectively and measured, but only some payload categories support optional compressed ingestion.
- This is intentional, but easy to misinterpret later as partial implementation debt.

4. Potential policy leakage risk at integration time
- LightEngine integration could accidentally bypass capability seams and call backend-specific paths directly.
- Current code is mostly clean, but this risk appears at the next phase boundary.

5. Harnesses are strong but scattered
- Validation/perf docs and harnesses are spread across multiple files and repos.
- Discoverability for new contributors is improving but still fragmented.

## 5. Recommendations

### Keep as-is

- Keep MeshForge as canonical payload-authoring owner.
- Keep DynamisGPU ingestion boundary strict and local.
- Keep standalone capability model in DynamisGPU (no renderer API leakage).
- Keep compression optional and selective for now.

### Watch later

- Watch code duplication in payload/ingestion/uploader classes as feature count grows.
- Watch naming drift before LightEngine integration starts.
- Watch for accidental policy coupling in future composed capability work.

### Candidate post-roadmap refactor

1. Introduce a small internal template/pattern for repeated payload ingestion/resource checks (not a broad generic framework).
2. Standardize payload metadata naming conventions (`count/offset/stride/byteSize`) across all new slices.
3. Add one central index doc for capability chain status and ownership links across MeshForge + DynamisGPU docs.

### Do not do yet

- Do not collapse all payload/resource types into one generic abstraction.
- Do not move policy decisions into DynamisGPU.
- Do not perform package-wide renames while LightEngine integration design is still pending.
- Do not broaden compression rollout without fresh end-to-end evidence in target runtime context.

## 6. Deferred to LightEngine / Policy Layers (Explicit)

The following remain intentionally out of MeshForge and DynamisGPU foundation slices:

- Renderer/frame-graph wiring.
- Feature activation policy (visibility/LOD/streaming interplay).
- RT material/shader-binding policy and render-pass integration.
- Tessellation adaptive policy and pass-level execution planning.
- Scene-level residency/orchestration state machines.

## 7. Review Conclusion

Current MeshForge ↔ DynamisGPU architecture is coherent and layered correctly for the completed roadmap work.

Most boundaries are clean and intentionally narrow.

The primary near-term risk is not missing capability primitives; it is integration-time policy leakage and duplication creep.

Recommended next step: **LightEngine integration planning**, with explicit guardrails to preserve existing ownership boundaries, while deferring broad refactors until integration pressure identifies concrete pain points.
