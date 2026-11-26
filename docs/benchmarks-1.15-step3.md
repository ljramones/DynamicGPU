# DynamisGPU 1.15: selector repair, FP16 unpack, and the quantized-prefill gates

- **Date:** 2026-11-25
- **Branch:** `perf/1.15`
- **Machine:** Apple M4 Max, as in [benchmarks-1.15-phase0.md](benchmarks-1.15-phase0.md)
- **Data:** [evidence/benchmarks-1.15-step3/](evidence/benchmarks-1.15-step3/), all on both backends:

| File | What it holds |
|---|---|
| `block-compare.csv` | Whole 32-layer Q4K prefill: quantized TILED, two-pass and direct, alternating |
| `prefill-gate.csv` | The three paths on single products, m = 17–2048 (Q4K, Q4, Q8, Q6K) |
| `prefill-gate-boundary.csv` | The same at m = 56–129 |
| `prefill-gate-q4m-q5k.csv` | Q4M and Q5K at m = 17–2048 |
| `selector-alternating.csv` | AUTO against every forced kernel, alternating: 164 cases |
| `selector-sequential.csv` | The earlier sequential probe, kept to show its drift |
| `rows-check.csv` | 3-, 5- and 6-row projections |

Labels: **Measured** means in the CSVs; **Derived** means computed from them; **Proposal** is not implemented.

## What is decided and what is not

| Item | State |
|---|---|
| Selector repair (`4bef072`) | Committed; measured on both backends |
| `Quantize.unpackF16` (`226c7a3`) | Implemented and verified; a public method, opt-in |
| Using unpackF16 automatically inside `QuantizedGemm` | **Not decided, not implemented** |

On the last item: `QuantizedGemm` is unchanged. Its AUTO still picks SKINNY or TILED. The direct path exists only in
the test harness (`-Dperf.only=gate`, `-Dblock.quantPrefill=dequant|compare`). Enabling it is a separate decision:
- it rounds each reconstructed weight to FP16, which is a new accuracy contract;
- it needs an n·k FP16 scratch buffer;
- its crossover depends on the backend, the format and the tile position of m (below).

## 1. Selector repair (`4bef072`)

**The bug.** `f3d6440` dropped the transposed-B SKINNY estimate for 32 or more groups. Those shapes took the
non-transposed formula, so 4- and 8-row decode at 4096² would have picked SMALL (about 300 µs) over SKINNY (124 and
194 µs). GPT ultra found it in review.

**The fix.**
- From 32 groups the transposed-B estimate is 1.14.1's again. The latency model applies only below 32 groups.
- The non-transposed per-row penalty is refitted from 0.45 to 0.40. With the repair held fixed, that changes three of
  the 146 fixture rows: 4-row, ordinary-B projections, all from SMALL to SKINNY. SKINNY wins all three in the archived
  timings, by 1.12–1.22× on Metal and 1.20–1.37× on MoltenVK.
- The cost-model fixture now has 146 measured rows, with 0 misses with and without the matrix kernel:
  - 44 from the 1.8.0 sweep;
  - 30 from the 1.14.1 decode grid, which the broken estimate misses on 7 rows;
  - 72 from the Phase 0 probe.

**Independent check (Measured; seeded inputs).** m = 3, n = 3456, k = 4096 with ordinary B is a shape the refit
changes but did not fit. AUTO picks SKINNY:
- Metal: 192 µs, against 259 µs for SMALL;
- MoltenVK: 204 µs, against 275 µs.

Across the 3-, 5- and 6-row projections, the largest regret where AUTO picked a different kernel from the fastest is
1.11 (MoltenVK, 5 × 4096 × 4096, B).

**AUTO measured alternating (Measured).** The probe now times AUTO and every forced kernel in the same rounds,
rotating the order. That covers the selector cases, the 1.8.0 sweep, the 1.14.1 decode grid and the row checks: 164
cases per backend.

| Backend | Cases | Worst regret | Same-kernel spread (AUTO picked the fastest kernel) | Misses over 1.15 |
|---|---:|---:|---:|---:|
| Metal | 164 | 1.08 | ≤ 4.3% | 0 |
| MoltenVK | 164 | 1.13 | ≤ 2.2% | 0 |

The earlier sequential probe (`selector-sequential.csv`) showed "misses" up to 1.53×. Every one of them either paired
AUTO with the same kernel it had picked, or was a SKINNY/SMALL near-tie at m = 16. That was drift between sequential
measurements late in a long session; measured alternately, it is gone.

## 2. `Quantize.unpackF16` (`226c7a3`)

- **What it does:** dequantizes straight to FP16, each value rounded once to nearest even. The output is bit for bit
  `Float.floatToFloat16(QuantizedMatrix.dequantize)`, and bit for bit `unpack` followed by `Cast`.
- **Kernel structure:** `quant_unpack_body.slang` holds the FP32 and FP16 entries.
- **Performance:**
  - The first FP16 version, 2 elements per thread, was about 2× faster than unpack and cast: 325 against 670 µs per
    4096². It was arithmetic-bound, not memory-bound.
  - The committed version decodes 8 consecutive elements per thread. They share one block, one code word (two for
    Q8), one sub-block scale and one high-bits word, with the same per-element arithmetic.
  - Its unpack-only time was not timed separately; the gates below time it inside the whole path.

**Precision gate (Measured; Metal and MoltenVK identical).**
- `unpackF16MatchesUnpackThenCast` covers all 18 format and scale pairs, with weights at σ 0.02, 1e-5 and 2e4.
- Large Q4M uses a deterministic spread over [30000, 70000]. Gaussian values that large give a block minimum, Q4M's
  offset, below FP16's range, which `quantize` rejects. That pair was silently skipped before; now any quantize
  failure fails the test.
- Results: 4,718,592 values bit for bit against both references, including 213,600 exact FP16 ties, 1,277,550 FP16
  subnormals and 27,785 finite values beyond FP16's range (which become infinity).
- `packsAndUnpacksBitForBit` also checks the FP16 output for every format and block size.

## 3. Quantized prefill: the alternating gates (Measured)

Each path's output was checked before timing: the worst sampled error was 0.016 of the loose bound, and the worst
whole-pass stage check 0.19. The three paths then ran in the same rounds, rotating the order. Ratios below are of
median wall times over 10 alternating rounds.

### Whole 32-layer Q4K prefill

| Backend | Tokens | Quantized TILED | Two-pass | Direct (unpackF16) | Direct against TILED |
|---|---:|---:|---:|---:|---:|
| Metal | 512 | 1,761 ms (291 tok/s) | 922 ms (555 tok/s) | 682 ms (751 tok/s) | 2.58× |
| Metal | 2048 | 8,263 ms (248 tok/s) | 3,426 ms (598 tok/s) | 3,131 ms (654 tok/s) | 2.64× |
| MoltenVK | 512 | 2,491 ms (206 tok/s) | 1,927 ms (266 tok/s) | 1,701 ms (301 tok/s) | 1.46× |
| MoltenVK | 2048 | 11,750 ms (174 tok/s) | 8,077 ms (254 tok/s) | 7,813 ms (262 tok/s) | 1.50× |

- On Metal at 512 tokens, the direct path saves 240 ms against the two-pass path.
- FP16 weights did 835 tok/s (512) and 727 tok/s (2048) in Phase 0. That was a different run, so it is not an
  alternating comparison.

### Single products: direct time over quantized TILED time

| Backend, format | m ≤ 63 | m = 64–65 | m = 112–127 (Q4K, Q4, Q8, Q6K only) | m = 128 | m = 2048 |
|---|---:|---:|---:|---:|---:|
| Metal, all six formats | 1.15–1.76 (loses) | 0.58–0.83 | 0.75–1.16 | 0.46–0.59 | 0.34–0.41 |
| MoltenVK, Q4K, Q4M, Q5K | 0.77–0.93 | 0.72–0.93 | 0.73–0.78 | 0.73–0.78 | 0.62–0.68 |
| MoltenVK, Q6K | 1.22–1.44 | 1.05–1.44 | 1.05–1.16 | 1.07–1.16 | 0.88–0.95 (wins from m = 512) |
| MoltenVK, Q4, Q8 | 1.29–1.64 | 1.21–1.65 | 1.21–1.33 | 1.23–1.33 | 1.07–1.19 (never wins) |

- **The steps on Metal (Derived) follow the two GEMMs' 64-row tiles:**
  - Below 64 rows, the FP16 AUTO choice is TILED: about 3.0–3.3 ms for 4096 × 14336, against 1.8 ms for quantized
    TILED.
  - At m = 64 it switches to the matrix kernel, at 1.6 ms. That is why the win starts exactly at 64.
  - At m = 65 quantized TILED steps up a tile, from 1.97 to 3.97 ms, so the win continues: there is no dip just after
    the boundary.
  - Within the second tile, the matrix kernel's partial tile takes 3.1 ms at m = 65 and 4.5 ms at m = 127. The
    long-k shape (4096 × 14336) loses there: at m = 127 by 1.16× (Q4), 1.09× (Q8), 1.05× (Q4K) and 1.03× (Q6K).
    m = 128 wins again, at 0.49–0.59×.
- **The cause** is the matrix kernel's edge tiles, the same weakness behind the m = 32 dip in Phase 0.

### Proposal: when AUTO would pick the direct path (not implemented)

**Metal:** the direct path for m ≥ 64, any format, with one of these options:
- **(a) Accept the known window.** Losses of up to 1.16× at m = 113–127 for long-k projections, in exchange for
  0.34–0.6× elsewhere.
- **(b) Use a cost rule.** Unpack time plus the matrix GEMM per 64-row tile, against quantized TILED per tile, fitted
  to these CSVs.
- **(c) Fix the matrix kernel's edge tiles first.** That also lifts FP16 at m = 17–63 and makes the crossover smooth.

  My preference is (c) followed by (a): (b) fits a curve that (c) would change.

**MoltenVK (portable kernels):**
- Q4K, Q4M and Q5K: the direct path for every m above SKINNY's range.
- Q6K: the direct path from m = 512.
- Q4 and Q8: never.

**Either way, the contract and the memory:**
- **Contract:** FP16 rounding of each reconstructed weight, plus the FP16 GEMM's own bound. That is one rounding of W
  to FP16 (a relative error of at most 2⁻¹¹ per weight) on top of the GEMM's contract with FP16 inputs.
- **Memory:** a pooled FP16 scratch of up to n·k halves, 117 MB for the FFN projections.

## Next

1. The decision above.
2. The matrix kernel's edge tiles (Proposal (c)).
3. Long-context decode, starting with KV traffic. From Phase 0: Q4K at 32K adds 10 ms of attention per token.
4. The 2.8 ms wall-against-GPU decode gap, with a submit-and-completion trace.
5. A training-step trace, to set the mixed-precision priority.
