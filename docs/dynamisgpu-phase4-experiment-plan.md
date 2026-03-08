# DynamisGPU Phase 4 Experiment Plan

## 1. Purpose

Phase 4 shifts focus from reducing upload cost to **hiding upload cost through overlap and streaming-oriented behavior**.

Phase 4 focuses on engine-level upload behavior:

- upload overlap
- multi-frame in-flight behavior
- streaming-oriented submission patterns
- residency-adjacent pressure measurements

Phase 4 explicitly does **not** focus on:

- additional Java-side micro-tuning
- LWJGL patching or forking
- FFM replacement experiments
- renderer feature expansion

## 2. Baseline Assumptions

- Phase 3 established stable deferred sustained throughput.
- Phase 3.5 removed direct-buffer prep churn from the sustained hot path.
- Phase 3.6 did not justify further Java-side prep/bookkeeping micro-optimization.
- The next optimization frontier is higher-level upload overlap, in-flight behavior, and streaming shape.

## 3. Experiment Questions

- How much wall-clock ingestion cost can be hidden by overlap?
- How many uploads can be in flight safely before throughput flattens or latency spikes?
- Does batching across frames improve effective throughput or just shift cost?
- What submission granularity works best for scene-streaming patterns?
- At what point does residency pressure become the dominant limiter?

## 4. Experiment Tracks

### Track A: Multi-Frame In-Flight Uploads

Scenarios:

- 1 frame in flight
- 2 frames in flight
- 3 frames in flight
- 4 frames in flight

Measure:

- uploaded bytes
- submit time
- completion time
- effective GB/s
- max in-flight bytes
- fence retirement lag

### Track B: Mixed Scene Streaming

Scenarios:

- many small meshes
- many medium meshes
- mixed small/medium/large sets
- scene-chunk style batches

Measure:

- total scene ingestion time
- time to first usable subset
- per-batch latency
- throughput stability

### Track C: Pressure and Residency-Adjacent Behavior

Scenarios:

- repeated scene turnover
- upload and release cycles
- increasing in-flight memory pressure
- synthetic large scene chunks

Measure:

- allocator reuse behavior
- in-flight pool pressure
- throughput collapse thresholds
- staging/pool memory high-water marks

## 5. Standard Reporting Metrics

Use one reporting shape across all experiments:

- mode
- completion mode
- uploaded bytes
- upload GB/s
- submit ms
- completion ms
- total ms
- in-flight bytes
- in-flight submission count
- staging high-water mark
- device-local pool high-water mark

GPU Upload Triangle for interpretation:

- Throughput (GB/s)
- Time-to-first-usable (TTFU / completion latency)
- In-flight pressure (submissions/bytes)

## 6. Success Criteria

Phase 4 is successful if at least one is demonstrated with measured evidence:

- better effective throughput under overlap
- lower exposed wall-clock ingestion time
- stable multi-frame in-flight behavior
- clear operating thresholds for streaming-style workloads

## 7. Stop Rules

Stop Phase 4 backend experiments when one or more become true:

- overlap gains flatten across additional in-flight depth
- residency pressure is the dominant limiter
- results indicate the next work belongs in renderer integration or streaming management layers

## 8. First Experiment (4A.1)

Run current sustained scenarios while varying allowed in-flight submissions:

- scenarios:
  - dragon batched
  - lucy batched
  - synthetic 100 MB
- in-flight depths:
  - 1
  - 2
  - 3

Goal:

- quickly determine whether overlap yields meaningful wall-clock relief versus single in-flight behavior.

## 9. Execution Notes

- Keep public APIs stable during Phase 4 experiments.
- Keep upload semantics unchanged while varying overlap/in-flight policy.
- Prefer benchmark-first changes; avoid architecture expansion until measurements indicate direction.
- Model non-ideal request arrival with harness knobs:
  - `--arrival-pattern=burst|staggered|microburst`
  - `--arrival-jitter-ms=<ms>`
  - `--microburst-size=<N>` (when `microburst` is selected)

## 10. 4A.1 Initial Results (Apple M4 Max / MoltenVK)

Command pattern:

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanSustainedMain \
  -Dexec.args="--phase4-4a1 --max-inflight=<N> /Users/larrymitchell/Dynamis/MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

Measured results:

| Scenario | Inflight=1 GB/s | Inflight=2 GB/s | Inflight=3 GB/s | Latency trend |
| --- | ---: | ---: | ---: | --- |
| `dragon_batch_10_overlap` | 18.749 | 24.280 | 24.033 | increases (`0.877ms` -> `1.973ms` -> `3.857ms`) |
| `lucy_batch_100_overlap` | 22.739 | 26.830 | 26.199 | increases (`1.368ms` -> `5.010ms` -> `9.933ms`) |
| `synthetic_100mb_overlap` | 21.032 | 22.905 | 22.013 | increases (`1.379ms` -> `4.247ms` -> `8.442ms`) |

Interpretation:

- Throughput improves substantially from inflight depth `1 -> 2`.
- Depth `2 -> 3` is flat/slightly regressive on throughput.
- Completion latency rises meaningfully with inflight depth.
- Current best operating point for these scenarios appears near `max-inflight=2`.

## 11. Arrival Jitter Smoke Run

Command:

```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanSustainedMain \
  -Dexec.args="--phase4-4a1 --max-inflight=2 --arrival-pattern=staggered --arrival-jitter-ms=1 /Users/larrymitchell/Dynamis/MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

Observed direction vs burst (`max-inflight=2`):

- throughput decreased (e.g., `dragon_batch_10_overlap`: `24.280 -> 15.567 GB/s`)
- completion latency increased (e.g., `dragon_batch_10_overlap`: `1.973ms -> 3.122ms`)

This confirms the harness now exposes non-ideal arrival behavior and provides a realistic next step for push-vs-pull scheduling experiments.
