# DynamisGPU Phase 3 Deferred Upload

Phase 3 introduces deferred upload completion and sustained-throughput benchmarking, enabling direct comparison of blocking vs deferred Vulkan upload behavior under repeated, batched, and synthetic workloads.

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
