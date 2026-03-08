# DynamisGPU Phase 4 Experiment Plan

## 1. Purpose

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
