# DynamisGPU 1.15: selector fix and the quantized-prefill experiment

- **Date:** 2026-11-25
- **Branch:** `perf/1.15`:
  - the selector fix is `f3d6440`;
  - the prefill experiment is test code only, committed with this file.
- **Machine:** Apple M4 Max, as in [benchmarks-1.15-phase0.md](benchmarks-1.15-phase0.md).
- **Data:** [evidence/benchmarks-1.15-step2/](evidence/benchmarks-1.15-step2/):
  - `selector-after-fix.csv`: 116 selector and sweep cases per backend, AUTO against every forced kernel;
  - `quant-switch-m8-16.csv`: Q4K, Q5K and Q6K, SKINNY against TILED at m = 8 to 16;
  - `prefill-dequant.csv`: 240 rows, the unpack-and-GEMM sequence against quantized TILED;
  - `block-q4k-dequant.csv`: the whole 32-layer Q4K pass with that path.

This is steps 1 and 2 of the order agreed after the Phase 0 review. Labels are as before: **Measured** means in the
CSVs; **Derived** means computed from them, with the method given.

## Step 1: the selector misses, fixed with existing kernels (`f3d6440`)

### Cause

`GemmF16.Costs.skinnyApplies` offered SKINNY only with at least 32 groups of output columns. That withheld it from
two regions:
- small-n products with B transposed (n ≤ 32: one or two groups);
- m = 1 with n up to 1024.

In both regions SKINNY wins. The cost-model unit test hid this: it scored only the kernels the registry offered, so a
withheld winner never counted as a miss.

### Change

- The 32-group rule now applies only with B not transposed and m > 1. There SMALL does win, by 1.2–4×.
- With B transposed and fewer than 32 groups, SKINNY's cost is the larger of full-rate streaming and a latency term,
  1.2 + 1.5 m + 0.00064 m k µs. The term is fitted to the probe's one- and two-group rows.
- From 32 groups the costs are unchanged. The first version of the change used the latency model everywhere, and
  regressed 16 × 1024 × 1024 with B transposed (regret 1.25); restricting it fixed that.
- `GemmF16Test.costModelPicksTheMeasuredFastest` now takes the fastest over every kernel that ran, and includes the 72
  probed shapes: 118 shapes in all, with zero misses with and without the matrix kernel. The model applied offline to
  the 72 MoltenVK rows also has zero misses.
- **Q6K:** quantized SKINNY against TILED (Measured, both backends): Q6K ties at m = 14 (SKINNY at 1.01–1.02× TILED's
  time), loses by 13–15% at m = 16, and wins by 18–20% at m = 12. Q4K and Q5K keep SKINNY through m = 16, winning by
  20–35%.
  - `QuantizedGemm.skinnyMaxM(Format)` is 13 for Q6K and 16 otherwise.
  - `QuantizedGemm.algorithm(Shape, QuantizedMatrix)` is new.
  - `encode` uses the format-aware choice.
  - SKINNY and TILED sum in different fixed orders, so Q6K results at m = 14–16 change bits. Both are within the
    existing contract.

### Result (Measured; AUTO against the fastest forced kernel, seeded inputs, 10 paired rounds)

| | Cases | Misses (regret > 1.15) | Worst regret | Geometric-mean regret |
|---|---:|---:|---:|---:|
| Metal, before | 72 | 33 | 6.9 | – |
| Metal, after (72 probed + 44 sweep cases) | 116 | 0 | 1.02 | 1.000 |
| MoltenVK, after | 116 | 3 (see below) | 1.82 | 1.011 |

The three MoltenVK cases are not selection errors: AUTO and the fastest are the same SMALL kernel. They are
128 × 128 × 128 with B and with B transposed, and 96 × 160 × 96 with B transposed: 5–17 µs products with wide spread
(p90 up to 1.7× the median). This is an open measurement issue on MoltenVK for tiny dispatches; on Metal the same
pairs agree within 8%.

Suites: the GEMM and quantized hardware suites pass on Metal and MoltenVK (16 and 5 tests each), as does the portable
build.

## Step 2: Q4K prefill through existing kernels

### What was tried

For quantized products with m above the SKINNY limit:
1. Unpack the packed weights to FP32 (`Quantize.unpack`).
2. Cast them to FP16 (`Cast`).
3. Run the FP16 GEMM with AUTO selection: SIMDGROUP_MATRIX on Metal, TILED on MoltenVK.

It was timed two ways:
- as one sequence on single products (`-Dperf.only=prefill`);
- inside the whole 32-layer pass (`-Dblock.quantPrefill=dequant`), with every stage of layer 0 checked as before.

The path is test code only; nothing in the library changed for it.

### Whole pass, Q4K prefill (Measured; 32 layers, 10 rounds, wall time)

| Backend | Tokens | Before: quantized TILED | After: unpack + cast + GEMM | Speed-up | FP16 weights, for reference |
|---|---:|---:|---:|---:|---:|
| Metal | 512 | 1,838 ms (279 tok/s) | 894 ms (573 tok/s) | 2.06× | 613 ms (835 tok/s) |
| Metal | 2048 | 8,407 ms (244 tok/s) | 3,065 ms (668 tok/s) | 2.74× | 2,818 ms (727 tok/s) |
| MoltenVK | 512 | 2,410 ms (212 tok/s) | 2,033 ms (252 tok/s) | 1.19× | 1,631 ms (314 tok/s) |
| MoltenVK | 2048 | 10,785 ms (190 tok/s) | 7,133 ms (287 tok/s) | 1.51× | 6,386 ms (321 tok/s) |

- Stage checks passed: the worst was 0.189–0.193 of the sanity bound.
- Q4K decode stays on SKINNY. It ran at 52.9 tok/s on Metal at 512 context here, against 55.2 in Phase 0.
- GPT ultra predicted about 550 tok/s for Metal at 512 tokens if the projections ran twice as fast. Measured: 573.

### Single products (Measured; µs, median; Metal)

Each cell compares quantized TILED with the unpack, cast and GEMM sequence. The sequence costs about the GEMM alone
plus the unpack-and-cast time in the last column.

| Format, n × k | m = 64 | m = 512 | m = 2048 | Unpack + cast alone |
|---|---:|---:|---:|---:|
| Q4K 4096 × 4096 | 571 against 946 (0.60×) | 4,067 against 2,057 (1.98×) | 16,348 against 6,071 (2.69×) | 668 |
| Q4K 14336 × 4096 | 0.62× | 2.02× | 2.67× | 2,292 |
| Q4K 4096 × 14336 | 0.61× | 1.88× | 2.45× | 2,303 |
| Q4, Q8, Q6K (all shapes) | 0.55–0.60× | 1.69–1.84× | 2.16–2.37× | 604–2,310 |

On MoltenVK, the FP16 GEMM is TILED, at 3.3–4.2 TFLOP/s for these shapes:
- **K formats win:** Q4K by 1.15–1.62× from m = 512.
- **Q4 and Q8 lose:** 0.70–0.92×, because their quantized TILED kernel is already about as fast as FP16 TILED.
- **Q6K:** about even at m = 2048 (0.97–1.09×) and loses at 512 (0.82–0.92×).
- **Rule:** any switch must be per format and per backend.

Sampled errors against the exact dequantized weights, which include the FP16 rounding of W, are at most 0.0176 of the
loose bound 4 k 2⁻²⁴ Σ|a w|.

### Where the remaining time goes (Derived)

- **The unpack and cast pass is the overhead.**
  - It costs about 36–40 µs per million weights on Metal: 0.6–0.67 ms per 4096 × 4096 and 2.1–2.3 ms per 14336 × 4096.
  - The 32-layer stack has 218 M weights per layer, so that is about 8.5 ms per layer and 270–280 ms per pass.
  - That matches the measured gap: 894 ms against FP16's 613 ms at 512 tokens.
- **Its traffic is the cause.** It moves about 10.6 bytes per weight: the packed read, an FP32 write and read, and the
  FP16 write. That is about 270 GB/s.
- **Crossover:** below m ≈ 125–135 the sequence loses on Metal, across the three shapes. That figure is interpolated
  from the m = 64 and m = 512 per-row costs, not measured.

### What this suggests next (Hypotheses, to be measured)

1. **An unpack kernel that writes FP16 directly.**
   - It is a variant of `quant_unpack`, with no FP32 scratch and no cast pass: about 2.6 bytes per weight instead of
     10.6.
   - Expected: about 4× less unpack time, putting Metal Q4K prefill at 512 tokens near 680 ms (about 750 tok/s,
     against FP16's 835).
   - Small, low-risk, and it lowers the crossover m.
2. **Then ship the path as a `QuantizedGemm` algorithm:**
   - chosen per format, backend and m from measured crossovers (m = 64, 128, 256 to be measured);
   - a pooled FP16 scratch of up to n·k halves, 117 MB for the FFN projections;
   - its own accuracy contract: the FP16 rounding of each weight, plus the FP16 GEMM's.
3. **Later, a fused kernel** that dequantizes tiles into threadgroup memory inside the matrix GEMM. It removes the
   scratch and the unpack pass entirely. Judge it against item 2's whole-pass time, not against quantized TILED.

## Step 3, not started: long-context decode

From the Phase 0 trace, Q4K decode from 4K to 32K context on Metal:
- added 10.17 ms of wall time;
- added 10.0 ms of profiled attention;
- left the products unchanged at 11.9 ms.

GPT ultra's suggestion stands: look at KV traffic first. Separately, get a submit-and-completion trace of the 2.8 ms
wall-against-GPU gap at 4K before attributing it to Java.
