# DynamisGPU Phase 3 Deferred Upload

Phase 3 introduces deferred upload completion and sustained-throughput benchmarking, enabling direct comparison of blocking vs deferred Vulkan upload behavior under repeated, batched, and synthetic workloads.

Phase 3 deferred upload hardening fixed a staging-lifetime bug and enabled stable sustained upload throughput of ~3.7–7.0 GB/s.

## Benchmark Summary

### Environment
- Machine: Apple M4 Max
- OS: macOS 26.3
- JDK: Temurin 25.0.1
- LWJGL: 3.4.1
- Vulkan SDK: 1.4.341
- ICD: MoltenVK (`MoltenVK_icd.json`)
- Loader: SDK-local `libvulkan.1.dylib`

### Root Cause Fixed
- Under sustained deferred load, staging arena growth could occur mid-batch.
- That changed the source staging buffer handle while copy ops for the same submission were already recorded.
- Result: invalid source-buffer lifetime for some copy commands and eventual native Metal crash.

Fix:
- Reserve staging arena capacity once per batch (`reserveForBatch(...)`).
- Forbid mid-submission arena growth.
- Tie staging retirement to submission completion (`markSubmissionInFlight` / `retireSubmission`).
- Add pool in-use guards and copy overlap/offset validation.

### Sustained Throughput Highlights

| Scenario | Mode | Completion | Uploaded Bytes | Total ms | GB/s |
| --- | --- | --- | ---: | ---: | ---: |
| dragon_repeat_100 | OPTIMIZED | BLOCKING | 499,964,000 | 98.892 | 5.056 |
| lucy_repeat_1000 | OPTIMIZED | BLOCKING | 1,399,612,000 | 376.351 | 3.719 |
| dragon_batch_10_deferred | OPTIMIZED_DEFERRED | DEFERRED | 49,996,400 | 10.111 | 4.945 |
| lucy_batch_100_deferred | OPTIMIZED_DEFERRED | DEFERRED | 139,961,200 | 37.553 | 3.727 |
| synthetic_100mb_deferred | OPTIMIZED_DEFERRED | DEFERRED | 104,857,588 | 15.041 | 6.971 |

### Interpretation
- Phase 3 is now stable under sustained scenarios that previously crashed.
- Deferred mode preserves strong throughput while making completion/retirement explicit.
- Large synthetic transfer behavior is competitive (~7 GB/s peak observed), indicating the backend is no longer limited by one-shot setup overhead.
- Remaining limits are expected to be transfer bandwidth, fence cadence, and allocator behavior under heavier multi-submission pressure.

## Scope
- Added third Vulkan upload mode: `OPTIMIZED_DEFERRED`
- Added fence-backed pending submission tracking and retirement
- Kept `SIMPLE` and `OPTIMIZED` as reference/baseline paths
- Added sustained benchmark harness for repeated and batched scenarios
- Extended report output with:
  - upload mode
  - completion mode
  - uploaded bytes
  - effective upload bandwidth (GB/s, when upload timing is available)

## Upload Modes
- `SIMPLE`
  - Per-upload staging/allocation path, fully blocking.
- `OPTIMIZED`
  - Reusable staging arena + pool reuse, blocking completion.
- `OPTIMIZED_DEFERRED`
  - Reusable staging/pool with deferred submit/retire model backed by fences.

## Deferred Completion Model
- `submitBatchDeferred(plans)` records and submits transfer work without immediate wait.
- `completeDeferredBatch(batch, timeoutNanos)` waits and materializes resources.
- `tryCollectDeferredBatch()` polls completion non-blocking.
- `drainDeferredSubmissions()` waits/drains all pending submissions.

Current constraints:
- Single pending submission at a time in this phase (intentional scope control).
- Fence-based retirement (timeline semaphores deferred).

## Bench Entry Points

One-shot ingestion harness:

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanIngestionMain \
  -Dexec.args="--upload-mode=optimized_deferred ../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

Sustained throughput harness:

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanSustainedMain \
  -Dexec.args="../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

## Probe-First Run Protocol
Always validate runtime first:

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.VulkanProbeMain \
  -Dexec.args="--debug" \
  -Dexec.classpathScope=runtime
```

If probe fails at instance/device stage, do not treat upload benchmark output as authoritative.

## Suggested Scenario Matrix
- One-shot fixture comparison:
  - `simple`, `optimized`, `optimized_deferred`
  - `RevitHouse.obj`, `lucy.obj`, `xyzrgb_dragon.obj`
- Sustained scenarios:
  - repeated medium upload (many iterations)
  - repeated small upload (many iterations)
  - batched medium uploads
  - batched small uploads
  - synthetic sizes: 25 MB / 50 MB / 100 MB

## Expected Phase 3 Effect
- Single isolated upload latency: may improve modestly.
- Repeated/batched workloads: expected stronger gains.
- Effective GB/s: expected improvement in sustained scenarios.

## Deferred
- Multi-queue orchestration
- Timeline semaphore path
- Full streaming/residency manager
- Renderer-coupled scheduling
