# DynamisGPU 1.15: the Metal matrix kernel's row edges, and the resident FP16 control

- **Date:** 2026-11-26
- **Branch:** `perf/1.15`. It follows the recommendations in [benchmarks-1.15-step3-review.md](benchmarks-1.15-step3-review.md).
- **Machine:** Apple M4 Max, as in [benchmarks-1.15-phase0.md](benchmarks-1.15-phase0.md).
- **Data:** [evidence/benchmarks-1.15-step4/](evidence/benchmarks-1.15-step4/), Metal unless noted:

| File | What it holds |
|---|---|
| `kernel-ab-final.csv` | The 1.14.1 and 1.15 matrix kernels in the same alternating rounds: 30 shapes, m = 32–512 |
| `kernel-ab-heldout.csv` | The same at the held-out m = 129, 191, 192 and 193: 24 shapes |
| `kernel-ab-first-version.csv` | The first row-edge version against 1.14.1, which slowed every tile (section 2) |
| `pad-before.csv`, `pad-after.csv` | The padding experiment, on the 1.14.1 and the 1.15 kernels |
| `prefill-gate.csv` | Q4K, Q4, Q8 and Q6K: quantized TILED, two-pass and direct, m = 17–2048, including 191–193 |
| `selector-refit.csv` | AUTO and every forced dense kernel on 156 shapes, the 1.15 kernels: the cost-model fixture |
| `selector-final.csv` | The same with the final cost model |
| `block-compare.csv` | The whole 32-layer Q4K prefill: TILED, two-pass, direct and resident FP16, alternating |
| `block-compare-earlier-run.csv` | The same, 45 minutes earlier, with the GPU hotter |

Labels as before: **Measured** means in the CSVs; **Derived** means computed from them.

## Summary

- **The kernel change:** row-only edge tiles of the Metal matrix kernel load A directly, like interior tiles.
  - A row-only edge is a tile whose columns are complete and whose k is aligned, but whose rows pass m.
  - Results are bit for bit the same as before.
  - The partial-row products run in 0.35–0.72 of their 1.14.1 time.
  - Full-tile shapes are unchanged: 0.99–1.00, measured in the same rounds.
- **A first version slowed every tile by 1.19–1.29**, interior tiles included. Measuring the two kernels in the same
  rounds caught it; separate runs could not, because they drifted by up to ±20%. The committed version keeps the
  interior code as it was.
- **The selector:**
  - AUTO now picks the matrix kernel below 64 rows wherever its column tiles are complete.
  - The cost model has a new long-k term for SMALL with B transposed.
  - The fixture is refitted on the 1.15 kernels: 0 misses on 156 shapes, with and without the matrix kernel.
- **Quantized prefill (direct path, single products):**
  - It now beats quantized TILED from m = 17 for nearly every format and shape.
  - The dip at m = 112–127 is gone.
  - The held-out m = 191–193 behave as the tuned rows do.
- **Unpacking on demand against resident FP16 (whole pass):**
  - the direct path costs 1.09 of resident FP16 at 512 tokens (63–79 ms per pass);
  - 1.01–1.03 at 2048 tokens.
- **Unchanged:** `QuantizedGemm`'s default. The direct path is still harness-only, per the review.

## 1. The padding experiment (Measured; wall time, µs)

These are four arms in the same rounds, on the same real rows and weights:
- the matrix kernel at m;
- the same product with A padded by zero rows to the next multiple of 64;
- that padded product including the copies in and out (`Cast`);
- dense AUTO.

Every arm's logical rows were checked against the exact product, and guard words past C stayed untouched.

| n × k | m | 1.14.1 at m | 1.14.1 padded | 1.15 at m | 1.15 padded | 1.15 padded + copies |
|---|---:|---:|---:|---:|---:|---:|
| 4096 × 4096 | 17 | 586 | 294 | 354 | 283 | 304 |
| 4096 × 4096 | 32 | 714 | 288 | 353 | 280 | 296 |
| 4096 × 4096 | 63 | 995 | 289 | 358 | 280 | 302 |
| 4096 × 4096 | 65 | 691 | 503 | 576 | 492 | 514 |
| 4096 × 4096 | 127 | 970 | 511 | 588 | 492 | 527 |
| 14336 × 4096 | 17 | 1,286 | 720 | 852 | 706 | 723 |
| 14336 × 4096 | 63 | 2,211 | 712 | 1,011 | 833 | 873 |
| 14336 × 4096 | 127 | 2,595 | 1,332 | 1,795 | 1,490 | 1,562 |
| 4096 × 14336 | 17 | 2,216 | 991 | 1,225 | 960 | 974 |
| 4096 × 14336 | 63 | 3,798 | 995 | 1,279 | 994 | 1,030 |
| 4096 × 14336 | 127 | 4,141 | 1,749 | 2,198 | 1,847 | 1,902 |

- **Before the change**, padding (with its copies) was 1.3–3.7 times faster than the product at m. The edge tiles
  staged A and B through threadgroup memory, k in steps of 32.
- **After the change**, the gap is 1.14–1.28 times (0.78–0.88 padded over at m). Padding with copies still wins by
  10–20%.
  - That remainder is the row-edge tile itself: shifted loads and a remapped store.
  - Compiling the interior loops out of the edge copy changed nothing (0.996–1.011).
  - It is open: section 6.
- The two halves of this table come from separate runs. The same-round kernel comparison is section 2.

## 2. The kernel change

**Scope.** The change covers a row-only edge: A not transposed, m ≥ 8, a complete column tile and k a multiple of 8.
- **Loads:** each 8-row block of A loads directly from row `min(block, m − 8)`, so every load stays inside A. The
  k-loop is the interior loop.
- **Stores:** a block that m cuts then holds its valid rows as its last ones. It goes through the SIMD-group's share
  of `tileA`, which is free by then, to its place in `tileC`. A block wholly past m is not stored.
- **Result bits:** rows of a product are independent and the sums keep the interior's order, so the result is bit for
  bit the result with A padded by zero rows.
- **Other edges:** columns past n, unaligned k and transposed A keep the staged path.

**Structure.** The row-edge code is a template parameter (`EDGE`), and the kernel entry sends only the last row of
tiles to the `EDGE` copy:
- **Why a separate copy:** the first version put the path in the same function. Every tile then ran 1.19–1.29× its
  1.14.1 time (geometric mean 1.23 over 18 full-tile shapes, `kernel-ab-first-version.csv`). Its loops, or a variant
  that shared one loop with the interior, cost the interior tiles too.
- **Why it went unnoticed at first:** separate runs disagreed by ±20% between passes. Only `-Dperf.only=kernelab`
  exposed it. That section compiles two kernel sources side by side through a package-private seam,
  `GemmF16.createSimdgroup`, checks that their C is bit for bit the same, and times them in the same rounds.
- **The bundle version** goes from 3 to 4.

**Same-round results (Measured; GPU median, 1.15 over 1.14.1; C bit for bit the same on every shape):**

| m | 4096 × 4096 | 14336 × 4096 | 4096 × 14336 | Tile rows |
|---|---:|---:|---:|---|
| 32 | 0.37–0.51 | 0.41–0.55 | 0.35–0.46 | edge only |
| 64, 128, 512 | 0.99–1.00 | 0.99–1.00 | 0.99–1.00 | full |
| 127 | 0.48–0.63 | 0.44–0.60 | 0.43–0.52 | 1 full + 1 edge |
| 129 (held out, wall) | 0.64–0.89 | 0.65–0.89 | 0.66–0.86 | 2 full + 1 edge (1 row) |
| 191 (held out, wall) | 0.50–0.69 | 0.55–0.72 | 0.49–0.62 | 2 full + 1 edge |
| 192 (held out, wall) | 1.00 | 1.00 | 1.00 | full |
| 193 (held out, wall) | 0.66–0.90 | 0.73–0.93 | 0.59–0.87 | 3 full + 1 edge (1 row) |

Ranges span the two B layouts; ordinary B gains more, because its 1.14.1 edge tiles were slower.

**Conformance:**
- **New test:** `GemmF16Conformance.rowEdgeMatchesPaddedRows` covers 960 cases. Each one is bit for bit equal to the
  same product with A padded by zero rows. The cases cover:
  - FP16 and FP32 inputs, FP32 and FP16 C;
  - m ∈ {1, 7, 8, 9, 17, 33, 63, 65, 100, 127};
  - three n × k pairs, including ragged n and k = 100;
  - both B layouts;
  - no epilogue, column bias, residual and gated SiLU;
  - ldc = n + 8, batch 2 and guard words past C.
- **Suites:**

| Suite | Result | Notes |
|---|---|---|
| Portable build | passes | |
| Native Metal | passes | 17 `MetalGemmF16Test` tests, `MetalQuantizedTest` |
| MoltenVK | passes | no matrix kernel there; the cost-model change applies |
| lavapipe (`scripts/test-lavapipe.sh`) | passes | |

## 3. The selector

**The change in `GemmF16.Costs`:**
- **SIMDGROUP_MATRIX below 64 rows:** with A not transposed, m ≥ 8 and k a multiple of 8, each complete column tile
  costs 0.65 of TILED (0.55 with B transposed); a partial column tile costs 1.5. Before, every tile below 64 rows cost
  1.5 (0.75 with B transposed and m ≤ 16). The measured matrix-over-TILED ratios at m = 8–32 on full column
  tiles are 0.60–0.70 with B and 0.52–0.54 with B transposed.
- **SMALL with B transposed:** k beyond 4096 multiplies the estimate by 1 + (k − 4096)/16384.
  - Measured SMALL time over its estimate is about 1.3 at k = 4096 and 1.5–1.8 at k = 11008.
  - Without the term, AUTO picked SMALL over SKINNY at 16 × 4096 × 11008 with B transposed. SKINNY was faster in all
    three runs that timed it: by 1.08 (step 3), 1.10 and 1.30.
  - That gap moves with the GPU's state. SMALL's and TILED's times rose by about 25% between two runs while SKINNY's
    did not, so a row sitting on this boundary can flip between runs.

**The fixture.** `GemmF16Test.costModelPicksTheMeasuredFastest` now holds the 156 shapes of `selector-refit.csv`:
- the shapes: the 1.8.0 sweep, the 1.14.1 decode grid, the Phase 0 selector shapes and the 3- to 6-row projections;
- the timing: every dense kernel in the same rounds, on the 1.15 kernels;
- the result: 0 misses (regret over 1.2), with the matrix kernel and without;
- against the previous fixture's 146 archived rows, the new model misses 4. All four are shapes the kernel change sped
  up (m = 32 and m = 16 tiles, which the old rows timed staged), so those rows no longer describe the kernel.

**AUTO measured (Measured; alternating, against the fastest forced kernel).** This is `selector-final.csv`: the
final kernels and cost model, 156 shapes.

| | Cases | Worst regret | Geometric mean | Same-kernel spread | Misses (over 1.15) |
|---|---:|---:|---:|---:|---:|
| Metal, step 3 | 164 | 1.08 | – | ≤ 4.3% | 0 |
| Metal, now | 156 | 1.054 | 1.001 | ≤ 2.8% | 0 |

- **Where AUTO differs from the fastest kernel**, there are four cases, all near-ties of 1.03–1.054:
  - SKINNY against SMALL at 5 rows with ordinary B;
  - SKINNY against SMALL, and against the matrix kernel, at 16 rows with B^T.
- **Case counts:** the 164 of step 3 count repeated shapes from two sections twice; this run has 156 distinct shapes.

**MoltenVK (Derived):** applied to step 3's alternating MoltenVK timings (156 shapes), the new model:
- fixes the 16 × 4096 × 11008 B^T row;
- adds no misses;
- keeps the worst regret at 1.14, on a shape the term does not touch.

## 4. Quantized prefill, single products (Measured; direct time over quantized TILED time, wall)

The direct path is Quantize.unpackF16 followed by the FP16 GEMM's AUTO. AUTO now chooses the matrix kernel at every m
in this table. Each path's output was checked before timing; the worst sampled error was 0.018 of the loose bound.

| Format, n × k | m = 17 | 32 | 63 | 64 | 65 | 127 | 128 | 191 | 192 | 193 | 512 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4K 4096 × 4096 | 0.94 | 0.92 | 0.87 | 0.73 | 0.61 | 0.60 | 0.52 | 0.55 | 0.48 | 0.49 | 0.39 | 0.35 |
| Q4K 14336 × 4096 | 0.74 | 0.73 | 0.69 | 0.61 | 0.52 | 0.53 | 0.47 | 0.45 | 0.42 | 0.43 | 0.36 | 0.34 |
| Q4K 4096 × 14336 | 0.87 | 0.86 | 0.79 | 0.68 | 0.58 | 0.57 | 0.50 | 0.53 | 0.46 | 0.47 | 0.38 | 0.34 |
| Q4 4096 × 4096 | 1.05 | 1.03 | 0.97 | 0.81 | 0.69 | 0.68 | 0.59 | 0.64 | 0.56 | 0.56 | 0.43 | 0.40 |
| Q4 14336 × 4096 | 0.86 | 0.85 | 0.80 | 0.73 | 0.60 | 0.60 | 0.55 | 0.53 | 0.49 | 0.49 | 0.41 | 0.38 |
| Q4 4096 × 14336 | 1.00 | 0.97 | 0.91 | 0.78 | 0.66 | 0.66 | 0.58 | 0.61 | 0.55 | 0.55 | 0.43 | 0.39 |
| Q8, all three | 0.78–0.98 | 0.78–0.96 | 0.74–0.90 | 0.65–0.75 | 0.57–0.65 | 0.58–0.64 | 0.52–0.56 | 0.51–0.60 | 0.47–0.53 | 0.48–0.53 | 0.41–0.43 | 0.38–0.39 |
| Q6K, all three | 0.81–0.97 | 0.81–0.96 | 0.77–0.90 | 0.68–0.76 | 0.58–0.64 | 0.57–0.63 | 0.52–0.56 | 0.51–0.60 | 0.46–0.52 | 0.47–0.53 | 0.39–0.42 | 0.36–0.37 |

Against step 3 (same harness; Metal):
- m ≤ 63 was 1.15–1.76 (a loss) and is now 0.69–1.05;
- m = 112–127 was 0.75–1.16 and is now 0.53–0.68;
- the curve now falls steadily with m;
- the held-out 191–193 sit between their neighbours and show no new dip.

**Still to measure before any AUTO cutoff for the direct path:**
- Q4M and Q5K at these m (and the block size);
- FP16 C and the stack's epilogues on the deciding rows;
- the same gate on MoltenVK with the new selector.

Q4 at m = 17–32 on 4096 × 4096 still loses (1.03–1.05), so a cutoff at 17 would not hold for every format.

## 5. The whole 32-layer Q4K prefill, with the resident FP16 control (Measured; wall, median of 10 alternating rounds)

The fourth arm unpacks the same Q4K bytes into resident FP16 weights once, before timing, and then runs the same
dense GEMM, inputs, epilogues and KV setup as the direct arm. Every arm's layer-0 stages were checked; the worst
check was 0.19 of the sanity bound.

| Tokens | Quantized TILED | Two-pass | Direct (unpackF16) | Resident FP16 | Direct over TILED | Direct over resident |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 1,790 ms (286 tok/s) | 985 ms (520) | 783 ms (654) | 720 ms (711) | 0.44 | 1.087 (63 ms) |
| 2048 | 9,325 ms (220 tok/s) | 3,787 ms (541) | 3,579 ms (572) | 3,535 ms (579) | 0.38 | 1.013 (44 ms) |
| 512, earlier run | 2,359 ms | 1,203 ms | 930 ms | 852 ms | 0.39 | 1.092 (79 ms) |
| 2048, earlier run | 8,509 ms | 3,557 ms | 3,277 ms | 3,181 ms | 0.39 | 1.030 (96 ms) |

**What the control shows (Derived):**
- **Cost of unpacking on demand:** about 9% of the pass at 512 tokens (63–79 ms; about 2–2.5 ms per layer). It falls
  to 1–3% at 2048 tokens, where the GEMMs dominate.
- **The direct path's ceiling:** the remaining gap to quantized TILED is the dense GEMMs themselves, which resident
  FP16 shares. A fused dequantizing GEMM could recover at most the 9% at 512 tokens.

**Absolute times move between runs; ratios within a run are the comparison:**
- Quantized TILED, which nothing in this step changed, ran 1,761 ms (step 3), 2,359 ms (earlier run) and 1,790 ms
  (this run) at 512 tokens.
- Direct ran 682, 930 and 783 ms.
- Its ratio to TILED in the same rounds was 0.39, 0.39 and 0.44.
- Nothing on the direct path changed at these lengths:
  - all of its products have m = 512 or 2048, so they are full tiles, which the kernel A/B measured at 0.99–1.00;
  - the attention kernels are untouched.
- This run's direct arm also spread more within its rounds (GPU p10 657 ms against a median of 778).
- The 20-minute duty cycle (section 6) is the place to pin this variation down.
- Decode stays on SKINNY: 52.1 tok/s at 512 context.

## 6. Open

1. **The row-edge tile** still costs 1.14–1.28 times a padded product (section 1). Candidates:
   - a 32-row tile variant for the last tile row;
   - a cheaper remapped store.
2. **An AUTO policy for the direct path:**
   - a caller-selected precision policy, per the review;
   - plus the rows of section 4's last list.
3. **The selector's held-out generalization set:** 31/32/33 column groups in both layouts, and N = 496/512/528 and
   1984/2048/2112.
4. **The 20-minute duty cycle with telemetry.** Today's absolute times drifted by up to 34% over the session (quantized
   TILED, which nothing here changed: section 5), while ratios within a round held.
5. **The decode trace** at 4K and 32K.
6. **A training-step trace.**

## Reproducing

```sh
# kernel A/B (Metal): two MSL sources, same rounds, C compared bit for bit
mvn -o -pl dynamis-gpu-metal,dynamis-gpu-primitives -am -Pmetal-native,metal-hardware test -Dtest=MetalPerfProbeTest \
    -Dperf.probe=true -Dperf.only=kernelab -Dperf.kernelA=<1.14.1 .metal> -Dperf.kernelB=<1.15 .metal> -Dperf.abM=129,191,192,193
# padding experiment, gate, selector
... -Dperf.only=pad -Dperf.padM=17,32,48,63,65,96,112,127
... -Dperf.only=gate -Dperf.gateM=17,32,48,63,64,65,96,112,127,128,191,192,193,256,512,2048
... -Dperf.only=selector
# whole pass with the resident control
mvn ... -Dtest=MetalTransformerBlockTest -Dblock.benchmark=true -Dblock.weights=Q4K -Dblock.quantPrefill=compare
```
