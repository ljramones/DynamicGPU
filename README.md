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

`VulkanGpuUploadExecutor` supports three internal modes:
- `SIMPLE`: baseline, per-upload staging/allocation path
- `OPTIMIZED`: Phase 2 path with reusable upload arena, batched copy submit, and initial device-local allocation reuse
- `OPTIMIZED_DEFERRED`: Phase 3 deferred submission + fence-retirement path for sustained throughput scenarios

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
- `--upload-mode=optimized_deferred`

## Run Sustained Upload Benchmarks (Phase 3)

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanSustainedMain \
  -Dexec.args="../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

## Run Vulkan Probe

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.VulkanProbeMain \
  -Dexec.args="--debug" \
  -Dexec.classpathScope=runtime
```

Minimal SDK-only runtime selection (macOS):

```bash
source "$HOME/VulkanSDK/1.4.341.0/setup-env.sh"
export VK_DRIVER_FILES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export VK_ADD_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib"

mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.VulkanProbeMain \
  -Dexec.args="--debug" \
  -Dexec.classpathScope=runtime
```

## Run Vulkan Parity Probe

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.VulkanParityProbeMain \
  -Dexec.args="--debug" \
  -Dexec.classpathScope=runtime
```

Useful side-by-side checks:

```bash
env | grep '^VK_'
env | grep '^DYLD'
vulkaninfo | head -40
```

## Docs
- [Baseline ingestion note](docs/dynamisgpu-ingestion-baseline.md)
- [Phase 2 upload optimization note](docs/dynamisgpu-phase2-upload-optimization.md)
- [Phase 3 deferred upload note](docs/dynamisgpu-phase3-deferred-upload.md)
- [Phase 3 profiling workflow + decisions](docs/dynamisgpu-phase3-profiling-plan.md)
- [Phase 4 overlap + streaming experiment plan](docs/dynamisgpu-phase4-experiment-plan.md)
- [LWJGL optimization decision matrix](docs/lwjgl-optimization-decision-matrix.md)
