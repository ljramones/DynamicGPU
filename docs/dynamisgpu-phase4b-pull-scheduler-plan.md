# DynamisGPU Phase 4B Pull-Scheduler Plan

## 1. Purpose

Phase 4B evaluates a bounded backlog-driven transfer scheduler (pull model) on top of the existing Vulkan upload executor.

Key principle:

> The scheduler does not make uploads faster in isolation; it makes upload behavior more engine-appropriate by balancing throughput, TTFU, and in-flight pressure.

## 2. Why Phase 4B

From Phase 4A.1:

- inflight depth `1 -> 2` improved throughput materially
- inflight depth `2 -> 3` was flat/slightly regressive
- completion latency increased with deeper inflight depth

This indicates scheduling policy is now a primary optimization frontier.

## 3. Scope

Include:

- bounded upload backlog
- transfer scheduling policy (`targetInflight`, batching policy)
- measured push-vs-pull comparisons

Exclude:

- public API redesign
- LWJGL patching/forking
- FFM experiments
- renderer feature expansion

## 4. Minimal Architecture (Experiment)

```text
UploadRequest
  -> UploadBacklog
  -> TransferScheduler
  -> VulkanGpuUploadExecutor
```

Minimal request fields:

- `sizeBytes`
- `priority`
- `requestTime`
- `GpuGeometryUploadPlan`

## 5. Experiment Matrix

### 4B.1 Push vs Pull

Compare:

- immediate push submission
- bounded pull scheduler (`targetInflight=2` baseline)

Arrival patterns:

- burst
- staggered
- microburst

### 4B.2 Priority behavior

- `HIGH` vs `LOW` request classes
- verify high-priority TTFU improvement without collapse in throughput

### 4B.3 Mixed scene streaming

- many small
- many medium
- mixed small/medium/large batches

## 6. Metrics

- throughput (`GB/s`)
- avg/p95 TTFU
- max backlog depth
- max in-flight submissions
- max in-flight bytes
- submit/completion timing split

## 7. Success Criteria

At least one must hold:

- same or better throughput vs direct overlap mode
- lower avg/p95 TTFU under irregular arrival
- better pressure bounds under mixed workloads

## 8. Stop Rules

Stop Phase 4B when:

- scheduler complexity exceeds measured benefit
- gains flatten across realistic workload mix
- dominant bottleneck shifts to residency/renderer integration

## 9. Next Decision Gate

If 4B is successful:

- promote experimental scheduler into a reusable UploadManager track

If 4B is inconclusive:

- retain current overlap strategy (`max-inflight≈2`) and shift focus to streaming/residency integration experiments.
