# DynamisGPU Phase 2 Upload Optimization (Vulkan)

## Scope Completed
- Added persistent staging arena: `VulkanUploadArena`
- Added initial device-local allocation reuse pool: `VulkanDeviceLocalBufferPool`
- Added optimized batched-copy path in `VulkanGpuUploadExecutor` (single command buffer submit per upload batch)
- Kept fallback reference path in `VulkanGpuUploadExecutor.UploadPathMode.SIMPLE`
- Added harness switch: `--upload-mode=simple|optimized`

## Implementation Notes
- Public Phase 1 API was unchanged.
- Optimizations are backend-internal (`dynamis-gpu-vulkan`) only.
- `VulkanGpuBuffer` now supports custom close behavior so pooled allocations can be returned to the pool instead of destroyed each upload.
- Ownership remains explicit:
  - `GpuMeshResource` owns returned buffers.
  - Closing `GpuMeshResource` closes buffers.
  - In optimized mode, close returns pooled allocations to the device-local pool.

## Validation Commands (macOS)
```bash
source "$HOME/VulkanSDK/1.4.341.0/setup-env.sh"
export VK_DRIVER_FILES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib:$DYLD_LIBRARY_PATH"

mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanIngestionMain \
  -Dexec.args="--upload-mode=simple ../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime

mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanIngestionMain \
  -Dexec.args="--upload-mode=optimized ../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

## Current Runtime Result in This Environment
- Both modes currently report `PREUPLOAD_SUCCESS_UPLOAD_BLOCKED` with:
  - `GpuException: stage=instance_create result=-9 requestedInstanceExtensions=[VK_KHR_portability_enumeration, VK_EXT_debug_utils]`
- This run environment could not provide a usable Vulkan/MoltenVK device path, so upload-time deltas (`SIMPLE` vs `OPTIMIZED`) are not measurable here.
- Pre-upload ingestion stages (cache-hit load + bridge) still run and are valid.

## Deferred
- True deferred completion/fence-driven retirement (current optimized path uses single submit + wait-idle completion per batch).
- Broader allocator policies (defragmentation/residency/multi-page heuristics).
- Multi-queue transfer orchestration.
