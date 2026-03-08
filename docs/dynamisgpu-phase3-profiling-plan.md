# DynamisGPU Phase 3 Profiling Plan

## 1. Purpose

This document defines a repeatable profiling workflow for Phase 3 sustained upload paths so LWJGL/native optimization decisions are based on measured data.

Primary question:

> Which Java-side paths in the sustained upload hot loop are still worth touching after Phase 3?

## 2. Known-Good Environment Setup

```bash
source "$HOME/VulkanSDK/1.4.341.0/setup-env.sh"

export VK_DRIVER_FILES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export VK_ADD_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib"
```

Notes:
- Keep probe and harness on the same bootstrap path.
- Use an SDK-local loader + SDK ICD manifests to avoid mixed-runtime drift.

## 3. JFR Command(s)

Use a separate Java process (`exec:exec`) so JVM startup flags reliably apply.

```bash
mvn -q -pl dynamis-gpu-bench exec:exec \
  -Dexec.executable=java \
  -Dexec.classpathScope=runtime \
  -Dexec.args='-XX:StartFlightRecording=filename=/tmp/dgpu-phase3-sustained.jfr,settings=profile,dumponexit=true -cp %classpath org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanSustainedMain /Users/larrymitchell/Dynamis/MeshForge/fixtures/baseline'
```

Inspect the recording:

```bash
jfr summary /tmp/dgpu-phase3-sustained.jfr
jfr view --width 180 hot-methods /tmp/dgpu-phase3-sustained.jfr
jfr view --width 180 allocation-by-site /tmp/dgpu-phase3-sustained.jfr
jfr print --events jdk.ExecutionSample --stack-depth 64 /tmp/dgpu-phase3-sustained.jfr > /tmp/dgpu-phase3-samples.txt
```

## 4. Scenario Set

The sustained harness run includes these anchor scenarios:

- `dragon_repeat_100`
- `lucy_repeat_1000`
- `dragon_batch_10_deferred`
- `lucy_batch_100_deferred`
- `synthetic_100mb_deferred`

## 5. Target Methods / Classes

Primary upload-path targets:

- `VulkanGpuUploadExecutor.submitBatchDeferred(...)`
- `VulkanGpuUploadExecutor.completeDeferredBatch(...)`
- `VulkanGpuUploadExecutor.tryCollectDeferredBatch()`
- `VulkanGpuUploadExecutor.drainDeferredSubmissions()`
- `VulkanGpuUploadExecutor.preparePlanForBatch(...)`
- `VulkanGpuUploadExecutor.submitCopies(...)`
- `VulkanGpuUploadExecutor.recordCopy(...)`
- `VulkanGpuUploadExecutor.waitFenceOrThrow(...)`

Struct/build churn targets:

- `java.nio.ByteBuffer.allocateDirect(...)`
- `VulkanGpuUploadExecutor.toDirectCopy(...)`
- `MeshForgeVulkanSustainedMain.copyToDirect(...)`

## 6. What To Inspect In JFR

- CPU sample concentration (`hot-methods`) for Java-side churn.
- Allocation pressure by site (`allocation-by-site`) for repeated helper overhead.
- Inclusive sample presence in upload methods from `jdk.ExecutionSample` stacks.
- Whether LWJGL internals appear as a measurable Java-side hotspot in this workload.

## 7. Decision Thresholds

Use these thresholds for Java-side helper churn share of total sampled time:

- `<3%`: leave alone
- `3-8%`: wrap/cache in Dynamis
- `>8-10%` in a narrow repeated helper path: surgical patch or FFM experiment can be justified

## 8. Action Table Template

| Path | Measured Share | Action | Notes |
| --- | ---: | --- | --- |
| Java helper churn (alloc/copy/build) | | leave / wrap / patch / FFM | |
| Deferred submission bookkeeping | | leave / wrap / patch / FFM | |
| Completion/retirement path | | leave / wrap / patch / FFM | |
| LWJGL helper internals | | leave / wrap / patch / FFM | |
| Native driver/runtime time | | not Java-side target | |

## 9. Initial Measured Action Table (2026-03-08)

Recording:
- `/tmp/dgpu-phase3-sustained.jfr`
- `jdk.ExecutionSample` count: `46`
- Duration: `~1s` sustained harness run

Measured shares:
- `hot-methods` top-of-stack:
  - `ByteBuffer.allocateDirect`: `32.61%`
  - `VulkanGpuUploadExecutor.toDirectCopy`: `8.70%`
  - `MeshForgeVulkanSustainedMain.copyToDirect`: `6.52%`
- Inclusive sample presence (`/tmp/dgpu-phase3-samples.txt`):
  - `VulkanGpuUploadExecutor.submitBatchDeferred`: `19.57%` (9/46 samples contain this frame)
  - `VulkanGpuUploadExecutor.preparePlanForBatch`: `19.57%` (9/46)
  - `VulkanGpuUploadExecutor.completeDeferredBatch`: `2.17%` (1/46)
  - `VulkanGpuUploadExecutor.waitFenceOrThrow`: `2.17%` (1/46)
  - `VulkanGpuUploadExecutor.tryCollectDeferredBatch`: `0.00%`
  - `VulkanGpuUploadExecutor.drainDeferredSubmissions`: `0.00%`

First action table:

| Path | Measured Share | Action | Notes |
| --- | ---: | --- | --- |
| Java helper churn (`allocateDirect` + copy helpers) | `47.83%` top-of-stack (32.61 + 8.70 + 6.52) | **wrap/cache** | High measured churn; prioritize reducing direct-buffer clone/allocation volume in harness/executor paths. |
| Deferred submission bookkeeping (`submitBatchDeferred`, `preparePlanForBatch`) | `19.57%` inclusive presence each | **wrap/cache** | Worth optimization in object/plan prep path before considering lower-level binding changes. |
| Completion/retirement (`completeDeferredBatch`, `waitFenceOrThrow`) | `2.17%` each inclusive | **leave** | Below threshold; no immediate optimization needed. |
| LWJGL helper internals (Java-side) | No dominant LWJGL Java helper in hot-methods | **leave** | Current capture does not justify patching LWJGL internals. |
| FFM replacement candidates | Not isolated as dominant measured hotspot | **defer** | Gather microbench evidence first; do not replace LWJGL broadly. |

## 10. Caveats

- This initial profile is short (`~1s`) and sample-based; treat it as directional.
- `hot-methods` is top-of-stack sampled time; inclusive stack-presence values are separate and should not be summed with top-of-stack percentages.
- Repeat this workflow after any upload-path change before deciding on LWJGL patching or FFM experiments.

## 11. Phase 3.5 Rerun (Bounded Direct-Buffer Reuse Pass)

Change scope:
- Removed sustained-harness plan cloning direct-buffer churn (reuse template plans for repeated/batch scenarios).
- Added executor-side reusable direct-copy staging (`toReusableDirectCopy`) for upload prep.
- No public API changes.

Rerun recording:
- `/tmp/dgpu-phase3-sustained-phase35.jfr`
- `jdk.ExecutionSample` count: `25`

Measured deltas versus 2026-03-08 baseline:

| Metric | Baseline | Phase 3.5 | Delta |
| --- | ---: | ---: | ---: |
| `ByteBuffer.allocateDirect` (hot-methods top-of-stack) | `32.61%` | `0.00%` | `-32.61pp` |
| `VulkanGpuUploadExecutor.toDirectCopy` (hot-methods top-of-stack) | `8.70%` | `0.00%` | `-8.70pp` |
| `MeshForgeVulkanSustainedMain.copyToDirect` (hot-methods top-of-stack) | `6.52%` | `0.00%` | `-6.52pp` |
| Combined direct-buffer/copy helper churn (top-of-stack) | `47.83%` | `0.00%` | `-47.83pp` |
| `submitBatchDeferred` (inclusive presence) | `19.57%` | `28.00%` | `+8.43pp` |
| `preparePlanForBatch` (inclusive presence) | `19.57%` | `24.00%` | `+4.43pp` |
| `toReusableDirectCopy` (inclusive presence) | `0.00%` | `8.00%` | `+8.00pp` |

Updated action read:
- Direct-buffer helper churn target is materially reduced; initial Phase 3.5 goal is met.
- Deferred bookkeeping is now relatively more visible and remains the next wrap/cache target.
- No new evidence justifies LWJGL patching or FFM replacement.
