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

## 10. Initial Push vs Pull Result (staggered, inflight=2)

Command:

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanSustainedMain \
  -Dexec.args="--phase4-4b-push-pull --max-inflight=2 --arrival-pattern=staggered --arrival-jitter-ms=1 /Users/larrymitchell/Dynamis/MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

| Scenario | Mode | Throughput (GB/s) | avg TTFU | p95 TTFU | max backlog depth | max inflight bytes | avg completion latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dragon_batch_10` | `push` | 15.780 | 6.184 ms | 6.916 ms | 0 | 99,992,800 | 4.295 ms |
| `dragon_batch_10` | `pull` | 16.745 | 4.628 ms | 5.311 ms | 1 | 49,996,400 | 2.937 ms |
| `lucy_batch_100` | `push` | 22.031 | 12.465 ms | 13.074 ms | 0 | 279,922,400 | 7.395 ms |
| `lucy_batch_100` | `pull` | 23.818 | 10.416 ms | 10.876 ms | 1 | 139,961,200 | 5.832 ms |
| `synthetic_100mb` | `push` | 19.624 | 10.183 ms | 15.939 ms | 0 | 209,715,176 | 6.158 ms |
| `synthetic_100mb` | `pull` | 22.206 | 7.988 ms | 8.141 ms | 1 | 104,857,588 | 4.570 ms |

Decision:

- Pull scheduling is justified in this workload shape: it improved throughput and reduced avg/p95 TTFU while halving observed in-flight bytes.

## 11. Phase 4B Outcome

The pull scheduler was confirmed as the correct upload policy and became the foundation for the Phase 5 UploadManager boundary.

The Phase 5 implementation preserved the pull model and validated it under bounded backlog control and real benchmark workloads.

See:

- `docs/dynamisgpu-phase5-upload-manager-plan.md`
