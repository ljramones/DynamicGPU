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
