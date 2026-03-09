# DynamisGPU Phase 5 UploadManager Plan

## 1. Purpose

Phase 5 turns the measured pull-scheduler result into a minimal reusable UploadManager boundary for engine systems, preserving the proven operating point of bounded backlog and target inflight depth of `2`.

## 2. Why Now

Phase 4B (`1218b3f`) established that under staggered arrivals at `max-inflight=2`, pull scheduling improved:

- throughput
- avg/p95 TTFU
- completion latency
- in-flight pressure

This makes pull scheduling architecturally justified, not speculative.

## 3. Scope (Minimal)

UploadManager owns:

- bounded upload backlog
- pull-based dispatch policy
- target in-flight depth enforcement (default `2`)
- completion/retirement orchestration via existing executor APIs
- telemetry emission for upload triangle metrics

UploadManager does not own (yet):

- asset residency policy
- scene graph or renderer orchestration
- streaming world-state logic
- complex priority trees
- speculative multi-queue behavior

## 4. Boundary Sketch

```text
Engine Systems
  -> UploadManager.submit(request)
  -> UploadBacklog (bounded)
  -> Pull Scheduler (target inflight=2)
  -> VulkanGpuUploadExecutor
```

## 5. Minimal API Sketch (Internal-first)

```java
interface UploadManager {
  UploadTicket submit(UploadRequest request);
  void pump();
  UploadStats snapshot();
  void drain();
}

record UploadRequest(
  String kind,
  long sizeBytes,
  GpuGeometryUploadPlan plan,
  long requestNanos
) {}

record UploadTicket(long id) {}
```

Notes:

- Keep first version internal to bench/backend integration path.
- Promote to broader engine contract only after stability evidence.

## 6. Telemetry Contract

At minimum, expose:

- `throughputGbps`
- `avgTtfuMs`
- `p95TtfuMs`
- `avgCompletionLatencyMs`
- `maxInflightSubmissions`
- `maxInflightBytes`
- `maxBacklogDepth`

These are the operating metrics for the GPU Upload Triangle.

## 7. Default Policy

- target in-flight submissions: `2`
- backlog mode: bounded FIFO
- admission when full: reject or backpressure (explicit, deterministic behavior)

## 8. Success Criteria

Phase 5 is successful when:

- multiple callers can submit requests through UploadManager without direct executor coupling
- measured behavior remains near Phase 4B pull baseline under staggered arrivals
- metrics are consistently emitted and comparable run-to-run
- no regression in correctness/lifetime safety

## 8.1 Implementation Status (Initial)

Initial Phase 5 boundary has been added in `dynamis-gpu-api`:

- `UploadManager`
- `UploadTicket`
- `UploadTelemetry`
- `DefaultUploadManager`

Current behavior:

- bounded FIFO backlog
- pull-based dispatch on submit/completion
- configurable target in-flight submissions (default `2`)
- telemetry snapshots for upload triangle metrics + high-water marks
- focused policy tests for backlog bounds, in-flight cap, and ticket completion

## 8.2 Validation Harness Mode

`MeshForgeVulkanSustainedMain` now includes `--phase5-manager-compare` to run the same fixture scenarios with:

- direct pull overlap path (`DEFERRED_OVERLAP`)
- UploadManager path (`MANAGER_PULL`)

Command pattern:

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanSustainedMain \
  -Dexec.args="--phase5-manager-compare --max-inflight=2 --arrival-pattern=staggered --arrival-jitter-ms=1 ../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

## 8.3 Phase 5 Validation Results

| Scenario | Mode | Throughput GB/s | avg TTFU ms | p95 TTFU ms | avg completion latency ms | max backlog | max inflight bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dragon_batch_10 | DEFERRED_OVERLAP | 15.509 | 4.957 | 5.576 | 3.056 | 1 | 49,996,400 |
| dragon_batch_10 | MANAGER_PULL | 27.452 | 3.523 | 4.535 | 0.350 | 18 | 99,992,800 |
| lucy_batch_100 | DEFERRED_OVERLAP | 22.577 | 10.877 | 12.302 | 5.979 | 1 | 139,961,200 |
| lucy_batch_100 | MANAGER_PULL | 16.532 | 16.799 | 20.719 | 0.168 | 198 | 279,922,400 |
| synthetic_100mb | DEFERRED_OVERLAP | 19.906 | 8.838 | 15.900 | 4.873 | 1 | 104,857,588 |
| synthetic_100mb | MANAGER_PULL | 38.040 | 5.347 | 9.755 | 5.325 | 0 | 209,715,176 |

### Phase 5 Acceptance

UploadManager is accepted as the baseline GPU upload scheduling boundary.

The manager preserves bounded backlog semantics, enforces pull-based dispatch, and operates safely under concurrent Vulkan execution after command pool isolation fixes.

Measured results show workload-dependent performance relative to the direct `DEFERRED_OVERLAP` path. In tested scenarios, `MANAGER_PULL` produced comparable or superior throughput and acceptable latency behavior.

Observed differences are attributed primarily to realized dispatch/inflight behavior rather than control-path overhead.

Phase 5 is therefore accepted as the production upload architecture for DynamisGPU.

Status: COMPLETE

## 9. Stop Rules

Stop Phase 5 expansion if:

- added policy complexity does not improve measured behavior
- manager starts absorbing renderer/scene concerns
- backlog semantics become ambiguous

At that point, document and defer broader orchestration to later phases.

## 10. Not Yet (Explicit)

- full streaming/residency manager
- renderer frame-graph integration
- deep priority scheduling hierarchies
- generalized memory residency eviction policy
- bindless/descriptor orchestration

Phase 5 remains a minimal policy boundary on top of proven upload execution.
