# Runtime Payload Compression Activation Measurements

This pass measures activation-path impact for optional compressed payload ingestion in DynamisGPU.

## Scope

Compared at ingestion boundary:

- uncompressed path
- compressed path (`DEFLATE`)

Payload classes:

- meshlet visibility bounds payload
- meshlet streaming payload

Measured path includes ingestion + validation (with decompression for compressed mode).

## Scenarios

- Visibility scenario:
  - `meshletCount = 20,000`
  - canonical bytes = `480,000`
- Streaming scenario:
  - `streamUnitCount = 2,048`
  - canonical bytes = `40,960`

Compression mode:

- `DEFLATE` (`BEST_SPEED`)

Test harness:

- `RuntimePayloadCompressionActivationMeasurementTest`
- warmup: 8 runs
- measured: 25 runs
- reported: median and p95 ingestion microseconds

## Results

### Visibility

- Raw bytes: `480,000`
- Compressed bytes: `154,055`
- Compression ratio: `0.321`
- Ingest uncompressed median/p95: `102.125us / 119.166us`
- Ingest compressed median/p95: `1099.750us / 1204.208us`
- Median delta: `+997.625us`

### Streaming

- Raw bytes: `40,960`
- Compressed bytes: `17,567`
- Compression ratio: `0.429`
- Ingest uncompressed median/p95: `2.917us / 6.917us`
- Ingest compressed median/p95: `105.875us / 130.875us`
- Median delta: `+102.958us`

## Interpretation

Compression materially reduces payload byte size for both classes, but decompression cost dominates
ingestion-path timing in this measurement slice.

At the current ingestion boundary:

- compressed ingestion is slower than uncompressed ingestion for both measured payload classes
- size reduction is strong, but CPU overhead is non-trivial

## Recommendation

Keep compression optional and available, but do **not** broaden rollout or discuss default-on policy yet.

Next decision gate should be based on end-to-end system context:

1. measure where byte-size savings matter most (e.g. transport/storage constrained paths)
2. if needed, evaluate faster codec options and/or different decompression placement
3. otherwise pause compression expansion and prioritize higher-leverage roadmap items

## Deferred

- default-on compression policy
- broader payload class rollout
- alternate codec adoption
- renderer-level policy integration

