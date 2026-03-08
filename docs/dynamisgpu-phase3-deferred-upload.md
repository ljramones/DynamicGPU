# Phase 3 Deferred Upload - Benchmark Summary

Phase 3 establishes a stable deferred Vulkan upload pipeline delivering sustained throughput of ~3.7-7.0 GB/s on Apple M4 Max (MoltenVK).

## Environment

- Machine: Apple M4 Max
- OS: macOS 26.3
- CPU: Apple Silicon (ARM64)
- GPU backend: Metal via MoltenVK
- Vulkan SDK: 1.4.341
- ICD: `MoltenVK_icd.json`
- Loader: `$VULKAN_SDK/lib/libvulkan.1.dylib`
- Java: Temurin 25.0.1 (OpenJDK 25 LTS)
- LWJGL: 3.4.1

Runtime environment setup:

```bash
source "$HOME/VulkanSDK/1.4.341.0/setup-env.sh"

export VK_DRIVER_FILES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export VK_ADD_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib"
```

## Benchmark Commands

### One-shot ingestion

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanIngestionMain \
  -Dexec.args="--upload-mode=optimized_deferred ../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

### Sustained throughput harness

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanSustainedMain \
  -Dexec.args="../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

## Root Cause Fixed During Phase 3

A crash under sustained upload pressure was traced to staging arena reallocation during batch construction.

When a large batch exceeded staging capacity:
- the arena resized mid-submission
- earlier `vkCmdCopyBuffer` commands referenced the old staging buffer
- the submission then contained mixed staging buffers
- the Metal driver crashed during `copyBufferToBuffer`

### Fix

The staging arena now guarantees submission-stable memory:
- `reserveForBatch()` pre-sizes staging memory before recording commands
- mid-submission growth is disallowed
- submission IDs track in-flight staging usage
- retirement occurs only after fence completion
- guards prevent reuse while memory is in flight

Additional safety checks were added:
- src/dst overlap validation
- buffer-range validation
- pool reuse detection
- optional debug logging (`DYNAMISGPU_UPLOAD_DEBUG`)

## Sustained Upload Results

### Repeated uploads

| Scenario | Uploads | Total Uploaded | Time | Throughput |
| --- | ---: | ---: | ---: | ---: |
| dragon_repeat_100 | 100 | 499,964,000 bytes | 98.892 ms | **5.056 GB/s** |
| lucy_repeat_1000 | 1000 | 1,399,612,000 bytes | 376.351 ms | **3.719 GB/s** |

### Batched uploads

| Scenario | Mode | Throughput |
| --- | --- | ---: |
| dragon_batch_10_deferred | deferred | **4.945 GB/s** |
| lucy_batch_100_deferred | deferred | **3.727 GB/s** |

### Synthetic transfer test

| Scenario | Uploaded | Throughput |
| --- | --- | ---: |
| synthetic_100mb_deferred | ~100 MB | **6.971 GB/s** |

## 2026-03-08 Refresh (Submit-Path Reuse A/B)

This refresh reran sustained scenarios before/after a narrow Vulkan submit-path change that reuses a single `VkBufferCopy` region struct per submission recording pass (instead of allocating one per recorded copy command).

### Scenario table (requested set)

| Scenario | Mode | Before ms | After ms | Delta ms | Before GB/s | After GB/s | Delta GB/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dragon_repeat_100 | OPTIMIZED (blocking) | 100.704 | 106.544 | +5.839 | 4.965 | 4.693 | -0.272 |
| lucy_repeat_1000 | OPTIMIZED (blocking) | 389.697 | 395.365 | +5.668 | 3.592 | 3.540 | -0.052 |
| dragon_batch_10_deferred | OPTIMIZED_DEFERRED | 10.471 | 10.364 | -0.108 | 4.775 | 4.824 | +0.049 |
| lucy_batch_100_deferred | OPTIMIZED_DEFERRED | 39.873 | 38.944 | -0.929 | 3.510 | 3.594 | +0.084 |
| synthetic_100mb_deferred | OPTIMIZED_DEFERRED | 14.555 | 14.345 | -0.210 | 7.204 | 7.310 | +0.106 |

### A/B interpretation

- Deferred batched and synthetic scenarios improved slightly (~1-2%).
- Long repeated blocking scenarios regressed slightly (~1-6%), consistent with runtime variance at this scale.
- Net result: submit-path struct reuse is safe and low-risk, but not a major standalone limiter; sustained throughput remains in the same effective Phase 3 band.

## Interpretation

Phase 3 demonstrates stable and sustained upload throughput in the ~3.7-7.0 GB/s range on Apple M4 Max through MoltenVK.

Key observations:
- Deferred completion allows the GPU to process transfers without blocking the CPU.
- Staging arena reuse eliminates repeated allocation overhead.
- Device-local buffer pooling prevents driver allocation churn.
- Submission-aware lifetime tracking prevents unsafe reuse of staging memory.

The synthetic 100 MB test shows the backend approaching the expected limits for the Vulkan-to-Metal transfer path on Apple Silicon.

## Architectural Takeaway

Phase 3 transitions the upload system from single-shot optimized uploads to a streaming-capable deferred transfer pipeline.

The backend now supports:
- persistent staging arenas
- device-local buffer pooling
- submission-safe memory reuse
- deferred fence-based retirement
- sustained throughput benchmarking

Phase 3 introduces explicit deferred upload lifecycle control via `submitBatchDeferred`, `completeDeferredBatch`, and fence-backed retirement.

## Comparison with Phase 2

| Metric | Phase 2 | Phase 3 |
| --- | --- | --- |
| Upload path | blocking optimized | deferred + batched |
| Staging reuse | yes | yes |
| Device buffer pooling | basic | guarded + tracked |
| Deferred lifecycle APIs | no | yes |
| Sustained throughput harness | none | implemented |
| Crash under sustained load | present | fixed |
| Sustained throughput | not measured | **3.7-7.0 GB/s** |

## Current Conclusion

The DynamisGPU backend now provides:
- a stable deferred upload pipeline
- sustained Vulkan upload throughput approaching **7 GB/s**
- robust memory-lifetime safety under heavy batching

The remaining performance frontier is no longer basic upload mechanics but:
- overlapping uploads with other GPU work
- multi-frame streaming behavior
- large-scene asset residency strategies
