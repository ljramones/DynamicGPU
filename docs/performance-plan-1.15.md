# DynamisGPU performance plan after the 1.14.1 benchmarks

- **Date:** 2026-11-25
- **Inputs:**
  - [benchmarks-1.14.1.md](benchmarks-1.14.1.md) and its CSV;
  - two external reviews of that data;
  - GPT ultra's review of the data, the source and an earlier draft of this plan;
  - a check of the reviews' hypotheses against the 1.14.1 source.
- **Status:** a plan. Nothing in it is implemented yet. Every phase ends with a gate measured by the same benchmark.

## Constraints every item keeps

These come from the project's decisions. The reviews did not know them.

- **Determinism:**
  - Every kernel sums in a fixed order. No float atomics by default; at most an opt-in, as recorded for 1.12.1.
  - A new reduction structure is allowed. It changes which bits come out, so the kernel's `AccuracyContract`, its
    tests and the docs change with it.
- **Accuracy contracts:** a faster path that rounds differently is a different contract, not a quiet replacement.
  Examples are FP16 operands in a quantized GEMM, or a different order for the norm.
  - It ships as its own algorithm, or as a documented change, with the error measured against the current kernel.
- **No untestable code:** nothing ships that no available device can run. Today that means native Metal, MoltenVK and
  lavapipe.
  - Before any `VK_KHR_cooperative_matrix` work, check whether lavapipe or MoltenVK exposes the extension. If
    neither does, that item waits for a device that has it.
- **Releases:** each phase is a DynamisGPU minor release, published on GitHub only.

## What the reviews got right, and what the source says

| Review claim | Verdict | Evidence in 1.14.1 |
|---|---|---|
| F1: quantized prefill runs at the TILED rate | **Confirmed** | `QuantizedGemm` has two algorithms only: SKINNY for m ≤ 16 and TILED beyond (`QuantizedGemm.java:67`, `:119`). There is no matrix-hardware path. |
| F2: the small-m/large-m switch is wrong for some formats | **Confirmed** | `SKINNY_MAX_M` is one constant (16) for every format; Q6K is slower at m = 16 than at m = 64. |
| F3 cause: backward uses TILED products and doesn't skip causal tiles | **Wrong** | Metal already runs a SIMD-group matrix backward (`attn_bwd_simdgroup.metal`, 1.12.1). It skips causal tiles (`kEnd = min(kEnd, iMax + offset + 1)`) and is FlashAttention-2-style: it recomputes P per tile and never materializes it. R3.2 and R3.3 are therefore done already. The gap has other causes (below). |
| F3 R3.1: instrument backward | **Partly exists** | The benchmark already prints per-stage times (`attention backward stages`). It lacks a per-dispatch kernel name and work counts. |
| F4: whole-tensor reductions are slow | **Confirmed; sharper hypothesis** | `clip_sumsq` and `nan_check` map every element through the generic rank-4 view (`dg_at_linear`), with a range check per element and one element per step. The same per-element addressing is in `Cast`, `Gather` and the axis reductions, which are also at 150–300 GB/s. A contiguous fast path is the first experiment. |
| F4 R4.3: fold the clip into the optimizer | **Half exists** | `Optimizer.adamW`/`sgd` already take a GPU `scale` buffer. What's missing is a clip that writes its factor into that buffer instead of scaling the gradients in place. |
| F5: multi-tensor operations pay per tensor | **Confirmed** | `GradientClip` dispatches per `Part`; `Optimizer` takes one tensor. |
| F6: FP32 SIMD-group GEMM is broken, not hardware-limited | **Plausible; investigate** | The FP32 path uses `simdgroup_matrix<float, 8, 8>` operands loaded straight from device memory: twice the bytes per block, against the FP16 path whose own comment records the direct loads as the reason it is fast. |
| F6 R6.1: never select SIMD-group for FP32 | **Already done** | `GemmF16.java:400` and `:472` exclude it. |
| F7: selection is unmeasured | **Confirmed** | The sweep prints each algorithm, not `GemmF16.algorithm(Shape)`'s choice. The 67× worst case matters only if the cost model (`GemmF16.Costs`) can pick it. Measure before redesigning. |
| F8: MoltenVK gaps | **Confirmed; mostly expected** | MoltenVK runs the portable Slang kernels: no matrix GEMM, matrix backward or `decode.short`. The open question is what Vulkan is for (R8.1). |
| F9b: paged decode is 47% slower than contiguous | **Unproven** | The two numbers come from different harnesses (median per call in a sweep against best-of-4). Measure both in one harness first. |
| B5: some GB/s exceed the bandwidth roof | **Agreed** | The report caveats them; the harness should compute masked traffic and assert the roof. |
| Second review: end-to-end, sustained, AUTO, MPS reference | **Agreed** | These are the main gaps in the evidence. |
| GPT ultra: the GEMM benchmarks time uninitialized inputs | **Confirmed** | `GemmF16Conformance.benchmark` and `shapeSweep` allocate private A and B and never fill them. That breaks AGENTS.md's rule; the 12.5 TFLOP/s headline needs a rerun on seeded inputs. |
| GPT ultra: GPU and wall times are not paired | **Confirmed** | The GEMM benchmark takes the minimum of each independently across rounds. |
| GPT ultra: the masked-prefill label is wrong | **Confirmed; fixed** | The fixture is 16 heads without GQA (`Shape.of(16, 512, 512, 128)`), not 32/8. The report and CSV are corrected. |
| GPT ultra: the quantized comparison is unfair | **Confirmed** | The quantized weights are zero-filled and rotate through over 256 MiB; the FP16 reference reuses one buffer. |
| GPT ultra: SKINNY already shares decoded codes across rows | **Confirmed** | `gemm_q_skinny_body.slang` decodes each unit once and keeps `acc[MAXM][CPT]` for all rows. The review's R2.3 is done; the m = 16 growth needs another explanation (Phase 2). |
| GPT ultra: the 64 "tensors" are adjacent views of one buffer, aligned to the 4,096-element chunks | **Confirmed** | That is how the harness built them. An order-preserving single dispatch is possible for that case. |
| GPT ultra: keep `NanCheck` separate from the norm | **Accepted** | A finite 1e20f overflows when squared, and `NanCheck`'s callers can locate the first non-finite element. The earlier draft's "the norm doubles as the finite check" is withdrawn. |
| GPT ultra: no hard ceiling on effective GB/s | **Accepted** | Cache-resident working sets can legitimately exceed the DRAM streaming rate. Flag such rows instead, and size bandwidth rows beyond the cache. |
| Section 6 (compressed payload ingestion) | **Separate track** | C1, a bounded inflate, is a robustness and security fix. It should not wait for this plan. |

## Why attention backward is slow (replacing F3's hypothesis)

- **More work than the 2.5× accounting assumes.** The deterministic two-kernel design computes S = QKᵀ and
  dP = dO Vᵀ in both the dK/dV kernel and the dQ kernel. That is seven products against the forward's two, so
  about 3.5 times the forward's work, not 2.5. At equal throughput the floor is about 3.5× forward time; today it is
  7.4×.
- **Small tiles:** dK/dV takes 32 keys × 16 rows and dQ 32 × 32, against the forward's larger blocks.
- **More FP16 staging:** P, dS and dO are rounded to FP16 for the products, with more threadgroup traffic per FLOP.
- **Profile split:** in the 2048-token profile the dK/dV kernel takes 32 ms of 57 profiled, so it goes first.
- **Gate:** ≤ 4.5× the forward time on Metal at 512–2048 tokens, from 6.8–7.4×; about 23 ms at 2048 tokens, from
  38. A later, fused one-kernel backward without atomics is out of scope.

## Phases

### Phase 0 (1.15.0): the measurement we need first

The micro-benchmarks can't rank the work on their own. Every later gate depends on this phase.

1. **Correct the existing harness first:**
   - seeded, finite inputs everywhere, with sampled outputs checked outside the timer;
   - GPU and wall time kept from the same round;
   - shape labels generated from the actual invocation;
   - a fair quantized comparison: representative nonzero packed weights, the same cache conditions for FP16 and
     quantized weights, and the FP16 AUTO choice at every m.
2. **One complete transformer block through the normal Java API:** resident weights and KV, AUTO selection, checked
   outputs, and whole-submission GPU and wall times. Prefill and decode are reported separately, and the same workload
   runs for 20 minutes. This comes before a multi-layer stack, because it already ranks dispatch, quantization and
   attention work.
3. **End-to-end rows in the benchmark:**
   - one decode token through a Llama-8B-shaped layer stack (resident weights; FP16, Q8, Q4, Q4K) at 512, 4K and
     32K keys of context;
   - prefill tokens/s at 512 and 2048;
   - one training step of a small GPT-2-sized model and of one 8B-shaped layer;
   - each reported as tokens/s, plus a per-kernel time breakdown from the GPU timestamps.
4. **AUTO rows:**
   - `GemmF16.algorithm(Shape)`'s choice and its time, next to every algorithm in the shape sweep and both decode
     grids, with regret against the best;
   - the same for `QuantizedGemm`.
5. **Aligned grids:** m ∈ {1, 2, 4, 8, 16, 32, 64, 512} in both decode sections, with an FP16 reference (the
   fastest eligible FP16 algorithm) at every m.
6. **One statistic:** median, min and p10/p90 with n written into each row. Machine-readable columns: m, n, k,
   batch, orientation, dtype, kernel_selected.
7. **Run conditions:**
   - an environment header: commit, OS build, JDK, MoltenVK, power source, thermal state;
   - a cool-down between suites;
   - a separate 20-minute sustained run reporting median throughput over time.
8. **Correct GB/s accounting** for masked and data-dependent kernels. Flag rows above the streaming rate rather than
   asserting a ceiling, and size the bandwidth rows beyond the system-level cache.
9. **External baselines on the same machine:**
   - `MPSMatrixMultiplication` through the Metal bridge, for GEMM in FP16 and FP32;
   - llama.cpp and MLX tokens/s on one reference model per format, as a script outside the library (needs both
     installed).
10. **Regression gate:** key rows tracked per commit on the self-hosted Apple runner. With 5% run-to-run spread, fail
   only on a median regression over 10%.

**Gate:** the CSV has AUTO rows, FP16 references at every m, the transformer-block, end-to-end and sustained rows,
and the external baselines. Every GEMM row is measured on seeded inputs, with paired times.

### Phase 1 (with 1.15.0): selection and cheap wins, no new kernels

- **Two targeted selector experiments before any autotuner.** GPT ultra's replay of the current FP16 rules
  against the forced-kernel timings (derived from the CSV's rounded ratios, so unconfirmed) puts 42 of 44 sweep cases
  within 15% of the fastest kernel; the 67× case already selects
  SMALL. Two misses remain:
  - 8×8×4096 with B transposed: the rules pick SMALL (about 103 µs), while forced SKINNY took about 33 µs, on both
    backends;
  - 1×1024×1024 with B not transposed.
  Measure AUTO on these and their neighbors, including the absolute time saved, before changing `GemmF16.Costs`.
- **Quantized switch per format:** force SKINNY and TILED for Q6K at m = 8 and 16 on the fair fixture. If the data
  supports it, make the SKINNY/TILED switch a per-format table.
- **Gradient norm and clip (the first implementation win):**
  - `squaredNorm` over adjacent views of one allocation whose boundaries fall on the 4,096-element chunks becomes one
    partial-reduction dispatch, in the same reduction order and so with the same result. Judge it against the 320 µs
    single-view time, then extend it to irregular views and to separate buffers.
  - `GradientClip` gets a factor output: `sumSq → scale` buffer. The optimizer consumes it through its existing
    `scale` input, which drops the in-place scaling pass for callers that don't need clipped gradients elsewhere.
  - `NanCheck` stays a separate pass. A finite gradient can overflow when squared, and its callers can locate the
    first non-finite element.
- **Transfers:** serial by default on unified memory; the two-queue path only where it measures faster.

**Gate:** the two selector misses fixed, with no new regressions on the sweep (AUTO regret ≤ 1.15× everywhere);
64-view squaredNorm within 10% of one view, with bit-identical results; clip + AdamW within 1.2× of AdamW alone.

### Phase 2 (1.16.0): inference kernels

1. **Quantized prefill on the matrix hardware:**
   - **Stopgap first:** dequantize to a pooled FP16 scratch, then run the SIMD-group FP16 GEMM. Expected 3.6 →
     about 1.8 ms at m = 512 on 4096².
   - **Then:** a fused kernel that dequantizes each weight tile into threadgroup memory as half and multiplies with
     SIMD-group matrices, accumulating in FP32 (the structure of llama.cpp's `kernel_mul_mm`).
   - Time the whole unpack, cast and GEMM sequence, scratch included, not the GEMM alone.
   - Both round W to FP16 operands, which is a new contract. Ship it as its own algorithm, with its measured error
     against the current kernel per format.
   - Set the whole-prefill target from the transformer-block trace. A 2× faster projection gives 1.8× faster prefill
     only if projections are about 89% of prefill time.
2. **Small-m quantized decode:** SKINNY already decodes each unit once for all rows, so the cost growth to m = 16
   has another cause. Hypotheses to test: the 16-row tier's registers (`acc[16][2]` plus the K formats' per-row
   partial sums) lowering occupancy, and its per-row group reductions. Try an 8-row tier and an earlier per-format
   switch to TILED.
3. **Decode-step fusion:**
   - RMSNorm into the next GEMM's prologue;
   - RoPE into the QKV epilogue;
   - a recorded and replayed decode step. MoltenVK's per-dispatch cost is about 8 µs against Metal's 4 µs, so
     dispatch count matters.
4. **`decode.short` for Vulkan:** a Slang port of the direct short decode.
5. **Paged against contiguous decode:** measure both in one harness, then fix page-table resolution if the gap is
   real.

**Gate:**
- quantized m = 512 ≥ 9 TFLOP/s on Metal;
- no quantized format slower than 1.15× FP16 at m ∈ {8, 16, 32};
- quantized prefill tokens/s against Phase 0, with a target set from the block trace;
- decode tokens/s not regressed.

### Phase 3 (1.17.0): training

1. **Mixed precision (NNJava G-T9).** The largest training lever in the data: FP32 GEMMs run at 5.2 TFLOP/s, against
   12.5 for FP16 on the matrix hardware.
   - FP16 or BF16 operands with FP32 master weights;
   - the optimizer's FP16 gradients and shadow already exist;
   - loss scaling uses the optimizer's GPU `scale` input.
2. **BF16 on the matrix hardware:** `simdgroup_bfloat8x8` (MSL 3.1). The M4 qualifies; verify with the device-family
   check. BF16 is the safer training format, since it has no loss-scale overflow.
3. **FP32 SIMD-group GEMM:** a time-boxed investigation, comparing against `MPSMatrixMultiplication` FP32 on the same
   shapes. Try staged threadgroup tiles for float operands and a smaller tile. If MPS is also near 2.8 TFLOP/s, the
   limit is the hardware, so close it.
4. **Attention backward:** the dK/dV kernel first. Try larger tiles, fewer FP16 staging round trips, and reusing S
   and dP within a tile pair. Gate as above.
5. **Kit kernels:**
   - contiguous fast paths with vector loads and several elements per thread, for `clip_sumsq`, `nan_check`,
     `Cast`, `Gather`, `Scan` and the reductions;
   - axis-0 reduction as row sweeps with a second-stage combine;
   - a fused two-pass norm backward, with a target of ≤ 3× forward.
6. **Multi-tensor operations:**
   - segment-table entry points (`squaredNorm`, `adamW` and `sgd` over a table of offset, length and per-segment
     hyperparameters), each one dispatch, in a fixed order;
   - NNJava's adapter allocates parameters, gradients and moments in flat arenas, so the common case is one segment.

**Gate:**
- training-step tokens/s against Phase 0;
- FP16 and BF16 training GEMMs ≥ 10 TFLOP/s at 2048³;
- attention backward ≤ 4.5× forward;
- squaredNorm ≥ 430 GB/s;
- 64-tensor rows within 10% of one tensor;
- norm backward ≤ 3× forward.

### Phase 4: Vulkan, after a decision

- **Decide the role of Vulkan:** decide what it is for (Linux and Windows GPUs, or Apple portability) before tuning
  MoltenVK-only gaps. Translation overhead doesn't carry over to native drivers.
- **If native Vulkan matters:**
  - acquire a device with `VK_KHR_cooperative_matrix`;
  - map the matrix GEMM and attention onto it;
  - measure there.
- **MoltenVK-specific items, if MoltenVK stays a target:**
  - diff the generated MSL for the Q4K/Q4M/Q5K kernels, which lose 1.3–1.8× where Q4/Q8/Q6K don't;
  - reduce per-dispatch state changes.

## Order and expected payoff

| Phase | Payoff (from the 1.14.1 data) | Risk |
|---|---|---|
| 0 | Makes every later gate measurable; may reorder Phases 2 and 3 | Low |
| 1 | Hygiene pass about 6× cheaper (962 → about 145 µs per 16M parameters); avoids bad picks | Low |
| 2 | Quantized prefill about 2× (3.6 → about 1.8 ms per 4096² at m = 512); batched quantized decode up to 2× | Medium: new accuracy contract |
| 3 | Training GEMMs about 2.4× (mixed precision); attention backward about 1.6×; kit kernels 2–3× | Medium to high |
| 4 | Depends on the Vulkan decision | Needs hardware |

Phases 2 and 3 are independent. Phase 2 first suits inference with quantized models. Phase 3 first suits NNJava
training. Phase 0's transformer-block rows should decide which comes first.

**Release sequence:** the corrected harness and the verified small wins (the gradient norm and clip, and the two
selector fixes) ship in 1.15.0, before any model runner exists. The larger choice between quantized inference and
training follows the first whole-block measurements.

**The number to publish eventually:** once NNJava's GPU adapter and model reader are ready, sustained tokens/s
against llama.cpp on the same model, quantization, prompt and context, next to the large GEMM result.
