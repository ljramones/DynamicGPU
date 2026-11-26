# DynamisGPU 1.15 Step 3 findings and recommendations

**Handoff for Claude Code, 2026-11-26.** Reviewed `perf/1.15` at `c320c15` against the checked-in benchmark CSVs and source. The hardware results below come from the recorded M4 Max runs. This review adds calculations and proposed experiments. [Step 3 report][step3] · [Phase 0 report][phase0]

## Decision for 1.15

| Item | Recommendation | Reason |
|---|---|---|
| Dense GEMM selector repair, `4bef072` | Accept for normal release validation. | Worst measured AUTO regret was 1.08× on Metal and 1.13× on MoltenVK against a 1.15× gate. |
| Public `Quantize.unpackF16`, `226c7a3` | Accept as an explicit conversion. | GPU output matched both the Java reference and GPU unpack followed by cast, including ties, subnormals and overflow. |
| Automatic unpack inside `QuantizedGemm` | Preserve the current default. Add a caller-selected precision policy if the composite path ships. Within that policy, let AUTO pick from measured, eligible paths. | The operation changes the weights used in multiplication, needs temporary storage and has shape-dependent costs. Current callers get fixed-order quantized algorithms. |
| Next performance experiment | Take the Metal partial-row tile first. Add the resident-FP16 control to the next whole-stack run. | The same long-K projection took twice as long at 127 rows as at 128. A same-round FP16 control can isolate the cost that remains after direct unpack. |

The selector and the explicit unpack can proceed through release validation independently of model-level testing. A default precision change deserves separate evidence and a maintainer decision. [Selector results][selector] · [Quantized GEMM API][qgemm] · [Conversion conformance][qtest]

## 1. What the throughput numbers establish

Each cell below uses **median wall time** over ten rounds. The three paths ran in rotating order with the same packed Q4K weights and seeded activations. Two-pass reconstructs FP32 weights, casts to FP16, then runs dense GEMM. Direct reconstructs FP16 in one dispatch and runs the same dense GEMM.

| Backend | Prompt tokens | Quantized TILED | Two-pass | Direct FP16 unpack | Direct speedup over TILED | Time saved by direct over two-pass |
|---|---:|---:|---:|---:|---:|---:|
| Metal | 512 | 291 tok/s | 555 tok/s | **751 tok/s** | 2.58× | 240 ms |
| Metal | 2,048 | 248 tok/s | 598 tok/s | **654 tok/s** | 2.64× | 296 ms |
| MoltenVK | 512 | 206 tok/s | 266 tok/s | **301 tok/s** | 1.46× | 226 ms |
| MoltenVK | 2,048 | 174 tok/s | 254 tok/s | **262 tok/s** | 1.50× | 264 ms |

At 2,048 tokens, moving to two-pass already captures 94% of the total time saved on Metal and 93% on MoltenVK, calculated as `(TILED − two-pass) / (TILED − direct)`. Dense FP16 multiplication is the main reason long-prompt prefill improves. Eliminating the intermediate FP32 write and read then saves roughly another quarter-second per stack pass. [Whole-pass CSV][block]

**Suggested wording for the result.** “On M4 Max, direct FP16 reconstruction reached 751 prompt tok/s at 512 tokens and 654 at 2,048 in a 32-layer, Llama-3-8B-shaped Q4K prefill benchmark.” The fixture reuses one physical layer’s weights over 32 layers with separate KV state. Embedding, tokenizer, output head and sampling sit outside this measurement. The public unpack primitive exists today. The combined path used for these figures is composed in the test harness. The earlier resident-FP16 rates of 835 and 727 tok/s came from a separate session. [Fixture and scope][phase0] · [Whole-pass implementation][stack]

## 2. Accuracy and public API

The conversion contract should read **“each FP32 reconstructed weight is converted with the result specified by Java’s `Float.floatToFloat16`.”** The two hardware suites compared 4,718,592 values per backend, including 213,600 ties, 1,277,550 FP16 subnormal results and 27,785 finite inputs that rounded to infinity. For finite weights in the normal FP16 range, round-to-nearest gives a relative error bound of 2⁻¹¹. In the subnormal range the general bound is absolute, 2⁻²⁵. A blanket bound relative to the format’s original quantization error fails even for Q8. For example, the valid Q8 scale 1 + 2⁻¹⁰ times code 3 reconstructs exactly 3.0029296875. If that was the original weight, its quantization error is zero. FP16 rounds the tie to 3.00390625. [Conversion results][step3] · [Conformance fixture][qtest]

For the **composite GEMM**, form a CPU reference by reconstructing the actual packed weights and rounding each to FP16 first. Check the selected dense GEMM’s contract using those operands, the requested output type and the real epilogue. For a plain dot product, the additional absolute error relative to unrounded reconstructed weights is bounded by `Σ |aᵢ| · |wᵢ − round16(wᵢ)|` before adding the GEMM’s error on its rounded inputs. Bias, activation, residual and output rounding follow their respective contracts. Preserve the current quantized CPU reference for existing algorithms. [Quantized reference][qgemm] · [Dense GEMM][gemm]

The performance fixture serves a different purpose. It checks 32 sampled dot products per microbenchmark and samples stages of layer zero before timing the full stack. The largest microbenchmark fraction across the three Step 3 gate CSVs is **0.0186** at Q4K, `m=63, n=k=4096`, including the boundary sweep. The whole-stack fixture’s largest recorded layer-zero fraction is **0.193**. A value of 1 would fail the respective sanity check. Use full-output conformance cases for a published numerical contract. [Boundary CSV][edgecsv] · [Probe oracle][probe] · [Whole-pass CSV][block]

If a public composite is added, exercise these four conditions in the public API.

1. **Permission and reproducibility.** Let callers explicitly permit FP16 weight reconstruction. Preserve forced SKINNY/TILED and reproducible AUTO, including a caller entering a scoped reproducible mode around encode. Expose the path actually used so callers can pin golden tests and diagnose changes at prompt-length boundaries.
2. **Weight eligibility.** For imported weights, compute or conservatively bound the maximum absolute reconstructed weight once while loading. Require a finite result that safely converts to FP16. A conservative ceiling of 65,504 is straightforward. Tie eligibility to the actual packed contents, slice and layout. `QuantizedMatrix` contains the descriptor and callers pass the buffer separately. Changed or unverified contents need revalidation or the existing quantized path. The standalone conversion still follows Java’s overflow semantics.
3. **Supported operands and economics.** The measured dense path uses FP16 activations and weights. Preserve BF16 and other existing callers. Check dense GEMM capability and select on measured backend/device, format, block size and shape. On measured MoltenVK projections beyond SKINNY’s range, Q4K, Q4M/32 and Q5K favor direct. Q6K crosses later, around 512 rows on this grid. Q4 and Q8 favor quantized TILED across the tested sizes. Q4M/128 needs its own performance rows. Native Vulkan devices need independent calibration.
4. **Scratch and layout.** Dense weight scratch needs `2 × n × k` bytes, **112 MiB** at 4,096 × 14,336. The measured stack reuses one buffer. Check allocation and storage-binding limits, acquire scratch before encoding unpack, and fall back on a genuine allocation shortage. Preserve the underlying error for invalid arguments or device failures. Keep the workspace live and correctly ordered across recorded or overlapping submissions. Give the dense weight operand canonical transposed strides (`ldb=k`, shared batch weights) while retaining the caller’s A/C strides, alpha, beta, gate and residual. Exercise a forced shortage and padded, batched inputs through the public entry point.

[Quantized descriptor][matrix] · [Current selection and encoding][qgemm] · [Dense GEMM capabilities][gemm] · [Harness scratch reuse][stack]

**Evidence for a future default.** Run identical teacher-forced token IDs on the same real packed weights with current quantized TILED and the direct path. Compare next-token distributions with mean and tail KL divergence, top-1 agreement and non-finite counts. Also compare the original model weights to quantized TILED, where available, to put any added deviation in context. Compare one-shot and chunked prompts that cross the candidate cutoff, followed by the same decode tokens. Agree on acceptable model-quality thresholds before seeing the result. This work fits NNJava once its model adapter is available.

## 3. Next experiment on Metal

The 64 × 64 matrix kernel already bounds-checks output stores. Its direct device-load path also requires a complete row tile, a complete column tile and K divisible by eight. A partial M tile sends **both A and B** through threadgroup staging and barriers, even when the B tile is fully in bounds. For Q4K with `n=4096, k=14336`, the current direct path took **4.493 ms wall at m=127 and 2.207 ms at m=128**. The measurements establish the discontinuity. They leave room to test which part of the edge branch causes it. [Metal kernel][msl] · [Boundary CSV][edgecsv]

1. **Test physical padding in the harness first.** Use the same 127 real rows and weights in both arms. In the second arm allocate a valid 128th A row and space for the output, zero the added inputs and provide valid padded epilogue operands. Check all 127 logical output rows and untouched guard regions. Report both isolated product time and wall time including padding or copying. This estimates the benefit available from reaching the interior path.
2. **If the padding result pays off, isolate the kernel change.** For a row-only edge with full N and aligned K, try direct B loads and stage or zero-fill the incomplete A fragments safely. Keep the existing general edge path for the other boundaries. Exercise both B layouts, ragged N/K, padded leading dimensions, FP16 and FP32 outputs and the production residual/gated epilogues. Retain the appropriate dense GEMM accuracy bounds and verify padding is untouched.
3. **Measure what AUTO would deliver.** Rotate baseline and candidate, forced matrix and dense AUTO, against quantized TILED. Start with the three existing projection shapes, `m=17,32,48,63,64,65,112,127,128,129`, and hold out **191/192/193** until after tuning. Include Q4M/128 and record block size. The current single-product gate uses FP32 C, transposed B, batch one and preallocated scratch. Repeat the decision-making rows with FP16 C and stack epilogues. Accept a cutoff where wall-time improvement exceeds paired run variation, the aligned shapes hold their performance, and the ordinary Java path selects the measured winner. A win at 17 rows remains an experiment.

[Single-product gate][probe] · [Per-format gate][formats] · [Full-stack implementation][stack]

## 4. Measurements that settle the remaining questions

| Question | Smallest decisive measurement | How to use the result |
|---|---|---|
| How much does unpack cost inside prefill? | Add a fourth arm to the same rotation. Before timing, unpack the **same Q4K bytes** into resident FP16. Use the same dense GEMM, inputs, epilogues and KV setup as direct. Separately time the committed eight-elements-per-thread direct unpack, FP32 unpack + cast, and GEMM alone on the three projection shapes. | Direct minus resident wall time estimates the end-to-end cost of on-demand reconstruction under those cache conditions. Weigh the isolated results by actual projection counts as a cross-check. The earlier 670 µs and 280 ms estimates concern other measurements. |
| Why does time per token change with prompt length? | Retain round index, path order, wall, GPU and encode times at 512 and 2,048. Take separate diagnostic stage profiles at both lengths. | Metal time per token increases **14.8%** for direct and **17.3%** for quantized TILED, and falls **7.1%** for two-pass. Report each path separately. Keep unprofiled whole submissions as the acceptance measure. |
| Does the selector generalize? | Freeze the fitted model and a seed for previously unused shapes. Rotate AUTO with every forceable candidate. Exercise 31/32/33 column groups in both B layouts, short M and the row-tile seams. With batch one, example N values are 496/512/528 for transposed B and 1984/2048/2112 for ordinary B. | Retain the 164-case regression set and report held-out results separately. Use the existing 1.15× worst-regret target. Compare AUTO with the fastest forced GPU median and report same-selected-kernel variation. The held-out three-row win already corroborates the 0.40 refit. |
| Is the 1.53× sequential variation thermal or allocator-related? | Repeat the existing 20-minute duty cycle with Q4K direct. Log elapsed time, power/thermal state where available, pool used and reserved bytes, driver allocation count and sampled high-water marks. Recheck the same warmed shape early and late. | Phase 0’s sustained data covers FP16 and fixed-position decode. The sequential selector discrepancy disappears under alternation. Investigate persistent growth or slowdown that survives controls before attributing it to the library. |
| Where do the extra decode milliseconds go? | On Q4K at 4K and 32K, capture paired host timestamps for encode, submit return and wait completion, plus submission GPU time and a separate attention/KV profile. Record actual KV data read or label calculated bytes as estimates. | Source shows **one submit and one wait per 32-layer token**. At 4K the separate medians are 18.852 ms wall, 16.060 ms GPU and 0.875 ms encode. Going to 32K adds 10.167 ms wall, alongside a 10.002 ms increase in the reported Metal attention stages. The same trace can investigate both the fixed host cost and the context-dependent KV cost. |

Use the timed submission for token latency. Per-dispatch intervals can overlap, and independently sorted medians lose the per-round pairing needed to attribute queue wait. MoltenVK’s per-dispatch timestamps under-report in this fixture. Use whole-submission GPU time and caller wall time for its performance decisions. A measured training step and real-model comparison against llama.cpp remain separate milestones. [Phase 0 timings][phasecsv] · [Stage timings][stages] · [Sustained CSV][sustain] · [Paired timing code][timer] · [Memory counters][memory]

## Reproducing the baseline and reporting back

From the DynamisGPU root on the configured Apple runner, this command exercises the **existing** three-path gate, on the tuning shapes. The test writes timing rows to Surefire output.

```sh
mvn -o -pl dynamis-gpu-metal,dynamis-gpu-primitives -am \
  -Pmetal-native,metal-hardware test -Dsurefire.failIfNoSpecifiedTests=false \
  -Dtest=MetalPerfProbeTest -Dperf.probe=true -Dperf.only=gate -Dperf.rounds=10 \
  -Dperf.gateFormats=Q4K,Q4,Q8,Q6K \
  -Dperf.gateM=17,32,48,63,64,65,112,127,128,129
```

After freezing the candidate, use `-Dperf.gateM=191,192,193` on both revisions for the held-out comparison. Use `-Dperf.only=selector` for the existing alternating selector run. For the current whole-stack comparison, use `MetalTransformerBlockTest` with `-Dblock.benchmark=true -Dblock.weights=Q4K -Dblock.prefill=512,2048 -Dblock.quantPrefill=compare`. The repository gives the corresponding Vulkan profile, class names and ICD/loader setup. The fourth resident-FP16 arm, forced-matrix comparison inside the prefill gate, block-size parameter and raw round/telemetry output need harness additions. [Project run instructions][agents] · [Probe][probe] · [Stack benchmark][stack]

For the next review, record the commit, environment and benchmark invocation alongside before/after CSVs, accuracy results and actual selected paths. Lead with caller wall time at the problematic edge and on the full stack. Record losing variants as well as improvements. For a production kernel change, run the portable build and conformance on native Metal, MoltenVK and lavapipe. For any Slang edit, require a successful kernel rebuild before timing. [Project validation instructions][agents]

[step3]: benchmarks-1.15-step3.md
[phase0]: benchmarks-1.15-phase0.md
[selector]: evidence/benchmarks-1.15-step3/selector-alternating.csv
[block]: evidence/benchmarks-1.15-step3/block-compare.csv
[edgecsv]: evidence/benchmarks-1.15-step3/prefill-gate-boundary.csv
[formats]: evidence/benchmarks-1.15-step3/prefill-gate-q4m-q5k.csv
[phasecsv]: evidence/benchmarks-1.15-phase0/block.csv
[stages]: evidence/benchmarks-1.15-phase0/block-stages.csv
[sustain]: evidence/benchmarks-1.15-phase0/sustained.csv
[agents]: ../AGENTS.md
[gemm]: ../dynamis-gpu-primitives/src/main/java/org/dynamisengine/gpu/primitives/GemmF16.java
[qgemm]: ../dynamis-gpu-primitives/src/main/java/org/dynamisengine/gpu/primitives/QuantizedGemm.java
[matrix]: ../dynamis-gpu-primitives/src/main/java/org/dynamisengine/gpu/primitives/QuantizedMatrix.java
[qtest]: ../dynamis-gpu-primitives/src/test/java/org/dynamisengine/gpu/primitives/QuantizedConformance.java
[probe]: ../dynamis-gpu-primitives/src/test/java/org/dynamisengine/gpu/primitives/PerfProbeConformance.java
[stack]: ../dynamis-gpu-primitives/src/test/java/org/dynamisengine/gpu/primitives/TransformerBlockConformance.java
[timer]: ../dynamis-gpu-primitives/src/test/java/org/dynamisengine/gpu/primitives/GpuBenchmark.java
[msl]: ../dynamis-gpu-primitives/src/main/resources/org/dynamisengine/gpu/primitives/metal/gemm_f16_simdgroup.metal
[memory]: ../dynamis-gpu-api/src/main/java/org/dynamisengine/gpu/api/compute/MemoryStats.java
