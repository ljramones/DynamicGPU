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

## Benchmark Comparison Summary
Phase 2 Vulkan upload optimizations reduced GPU upload cost by **~20% for large assets and ~45–50% for medium assets** on Apple M4 Max (MoltenVK).

### Environment
- Machine: Apple M4 Max
- OS: macOS 26.3
- JDK: Temurin 25.0.1
- LWJGL: 3.4.1
- Vulkan SDK: 1.4.341
- ICD: MoltenVK (`MoltenVK_icd.json`)
- Loader: SDK-local `libvulkan.1.dylib`

### Benchmark Command
Simple path:

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanIngestionMain \
  -Dexec.args="--upload-mode=simple ../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

Optimized path:

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanIngestionMain \
  -Dexec.args="--upload-mode=optimized ../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

### Results

| Fixture           | Upload Mode | Load ms | Bridge ms | Upload ms | Total ms |
| ----------------- | ----------- | ------: | --------: | --------: | -------: |
| RevitHouse.obj    | SIMPLE      |   7.547 |     2.309 |    21.822 |   31.678 |
| RevitHouse.obj    | OPTIMIZED   |   7.186 |     2.154 |    17.503 |   26.843 |
| lucy.obj          | SIMPLE      |   0.580 |     0.115 |     1.879 |    2.574 |
| lucy.obj          | OPTIMIZED   |   0.654 |     0.122 |     1.057 |    1.834 |
| xyzrgb_dragon.obj | SIMPLE      |   1.106 |     0.130 |     2.254 |    3.491 |
| xyzrgb_dragon.obj | OPTIMIZED   |   1.004 |     0.118 |     1.090 |    2.213 |

### Upload Improvement

| Fixture           | Simple Upload ms | Optimized Upload ms | Improvement | Reduction |
| ----------------- | ---------------: | ------------------: | ----------: | --------: |
| RevitHouse.obj    |           21.822 |              17.503 |    4.319 ms |     19.8% |
| lucy.obj          |            1.879 |               1.057 |    0.822 ms |     43.7% |
| xyzrgb_dragon.obj |            2.254 |               1.090 |    1.164 ms |     51.6% |

### Total Ingestion Improvement

| Fixture           | Simple Total ms | Optimized Total ms | Improvement | Reduction |
| ----------------- | --------------: | -----------------: | ----------: | --------: |
| RevitHouse.obj    |          31.678 |             26.843 |    4.835 ms |     15.3% |
| lucy.obj          |           2.574 |              1.834 |    0.740 ms |     28.8% |
| xyzrgb_dragon.obj |           3.491 |              2.213 |    1.278 ms |     36.6% |

### Interpretation
- Phase 2 optimization produced a clear and measurable reduction in upload cost across all tested fixtures.
- Persistent staging and allocation reuse reduced upload overhead substantially.
- Improvement was strongest for small and medium assets, indicating that Phase 1 cost included significant fixed per-upload overhead.
- Large-asset upload also improved meaningfully, but remains dominated by bulk transfer and driver/runtime cost.
- Cache-hit load and MeshForge bridge times remain low relative to upload, confirming that the dominant remaining ingestion cost is GPU upload rather than CPU-side geometry preparation.

### Architectural Takeaway
- Upstream geometry processing did not need redesign.
- MeshForge -> `GpuGeometryUploadPlan` remains inexpensive.
- The correct optimization target was the Vulkan upload backend.
- Persistent staging, pooled allocation, and optimized copy submission materially improved ingestion performance without changing the public upload API.

### Current Conclusion
- After Phase 2, the runtime ingestion path is significantly more credible as an engine-grade upload pipeline.
- For larger assets, the next limiting factors are likely bulk transfer cost, synchronization/submission behavior, and device-local allocation strategy under sustained or repeated load.

### Suggested Next Benchmarks
- Repeated upload of one medium asset.
- Batch upload of many medium assets.
- Batch upload of many small assets.
- Synthetic large upload cases (for example 25 MB, 50 MB, 100 MB, 250 MB).

### Suggested Next Optimization Focus
- Reduce blocking synchronization in benchmark/runtime upload flow.
- Improve sustained-transfer behavior for larger uploads.
- Strengthen pooled allocation and reuse under repeated scene ingestion.
- Measure batch-submit effectiveness under multi-asset workloads.

## Deferred
- True deferred completion/fence-driven retirement (current optimized path uses single submit + wait-idle completion per batch).
- Broader allocator policies (defragmentation/residency/multi-page heuristics).
- Multi-queue transfer orchestration.
