# DynamisGPU

Backend-neutral GPU abstraction and Vulkan runtime upload implementation for Dynamis Engine.

## Modules
- `dynamis-gpu-api`: public GPU contracts (`org.dynamisengine.gpu.api`)
- `dynamis-gpu-vulkan`: LWJGL/Vulkan backend (`org.dynamisengine.gpu.vulkan`)
- `dynamis-gpu-test`: mock/testing implementations
- `dynamis-gpu-bench`: ingestion and benchmark runners

## Build And Test
From repo root:

```bash
mvn clean verify
```

Useful module-scoped commands:

```bash
mvn -pl dynamis-gpu-vulkan -am test
mvn -pl dynamis-gpu-bench -am package
```

## Runtime Geometry Upload Path
Current canonical flow:

```text
RuntimeGeometryLoader
-> RuntimeGeometryPayload
-> MeshForgeGpuBridge
-> GpuGeometryUploadPlan
-> VulkanGpuUploadExecutor
-> GpuMeshResource
```

`VulkanGpuUploadExecutor` supports two internal modes:
- `SIMPLE`: baseline, per-upload staging/allocation path
- `OPTIMIZED`: Phase 2 path with reusable upload arena, batched copy submit, and initial device-local allocation reuse

## macOS Vulkan Setup (MoltenVK)
Known-good environment pattern:

```bash
source "$HOME/VulkanSDK/1.4.341.0/setup-env.sh"
export VK_DRIVER_FILES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib:$DYLD_LIBRARY_PATH"
```

## Run MeshForge Ingestion Harness

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanIngestionMain \
  -Dexec.args="--upload-mode=optimized ../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

Switch mode for comparison:
- `--upload-mode=simple`
- `--upload-mode=optimized`

## Run Vulkan Probe

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.VulkanProbeMain \
  -Dexec.args="--debug" \
  -Dexec.classpathScope=runtime
```

## Docs
- [Baseline ingestion note](docs/dynamisgpu-ingestion-baseline.md)
- [Phase 2 upload optimization note](docs/dynamisgpu-phase2-upload-optimization.md)
