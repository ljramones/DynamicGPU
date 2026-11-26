# DynamisGPU Phase 0 measurements (1.15 plan)

- **Date:** 2026-11-25
- **Code:** branch `perf/1.15` (base 1.14.1, `95c38e7`). Harness added in this phase: `PerfProbeConformance`,
  `TransformerBlockConformance`, and `GpuBenchmark.paired`/`Stats`.
- **Machine:** Apple M4 Max, macOS 27.0.1, Temurin 25.0.4. Native Metal, and MoltenVK 1.4.1 through the Vulkan SDK
  1.4.341 loader.
- **Data:** [evidence/benchmarks-1.15-phase0/](evidence/benchmarks-1.15-phase0/), CSV files:
  - `probe.csv`: 996 rows, GEMM and quantized cases;
  - `block.csv`: whole forward passes;
  - `block-selected.csv`: what AUTO picked for each projection;
  - `block-stages.csv`: profiled time per stage;
  - `sustained.csv`: per-minute medians.
- **Baseline:** the earlier [1.14.1 report](benchmarks-1.14.1.md) is kept unchanged as the historical baseline.
- **Plan:** [performance-plan-1.15.md](performance-plan-1.15.md).

## The answer first: what a Java caller gets

A Llama-3-8B-shaped model is pushed through the public Java API. It has 32 layers of the same block, AUTO kernel
selection, a paged FP16 KV cache, and checked outputs. Each pass is one submission, timed GPU and wall from the same
round.

| | Metal | MoltenVK |
|---|---:|---:|
| FP16 prefill, 512 tokens, warmed | 835 tok/s (613 ms) | 314 tok/s (1,631 ms) |
| FP16 prefill, 512 tokens, sustained, minutes 3–20 | 558–652 tok/s (785–917 ms) | 258–287 tok/s (1,782–1,988 ms) |
| FP16 decode, 4K context, warmed | 20.7 tok/s (48.4 ms) | 19.6 tok/s (51.1 ms) |
| FP16 decode, 4K context, sustained, minutes 1–20 | 21.4–21.7 tok/s (46.1–46.7 ms) | 20.0–20.5 tok/s (48.7–50.0 ms) |
| Q4K prefill, 512 tokens, warmed | 279 tok/s (1,838 ms) | 212 tok/s (2,410 ms) |
| Q4K decode, 4K context, warmed | 53.0 tok/s (18.9 ms) | 41.4 tok/s (24.2 ms) |
| Q4K decode, 32K context, warmed | 34.5 tok/s (29.0 ms) | 29.3 tok/s (34.1 ms) |

**Scope of these numbers:**
- They come from a synthetic model: random N(0, 0.02) weights and random KV history, with no tokenizer, embedding or
  output head.
- One layer's weights serve all 32 layers. That is 436 MB per layer in FP16, about 123 MB in Q4K. Decode reads them at
  330 and 257 GB/s, below the streaming rate, so the system cache isn't inflating them.
- Not yet compared with llama.cpp, MLX or MPS on the same machine. That is still open in Phase 0.

## Findings

Every claim below is **Measured** (in the CSVs) unless it is marked **Hypothesis**.

### 1. The GEMM headline holds on seeded inputs; FP32 on the matrix path is confirmed slow

- 4096³ FP16 with AUTO runs at 12.53 TFLOP/s on Metal, by median GPU time over 10 paired rounds. Forced
  SIMDGROUP_MATRIX gives 12.59.
  - Seeded, checked inputs leave the headline unchanged from 1.14.1 (12.50).
  - Wall time is 1.009× GPU time in the same rounds.
- FP32 on the matrix path is 2.87 TFLOP/s against 5.23 for TILED. AUTO picks TILED correctly. BF16 has only TILED,
  at 4.92.
- On MoltenVK, TILED FP16 at 4096³ is 4.85 TFLOP/s on seeded inputs, against 5.31 in the 1.14.1 run on uninitialized
  memory. Metal didn't change.
  - **Hypothesis:** data-dependent power or clock behavior.
  - Unexplained; worth one controlled rerun.

### 2. AUTO misses a whole region of small-n and m = 1 products

AUTO misses on 33 of the 72 probed shapes on both backends, with a regret over 1.15.
- **First region:** m = 4–16, n ≤ 32, k ≥ 2048, B transposed. AUTO picks SMALL, and SKINNY is 1.3–6.9× faster.
  - Example: m = 4, n = 8, k = 8192 takes 204 µs, against 30 µs.
- **Second region:** m = 1 with B not transposed and n, k ≤ 2048. AUTO picks SMALL, and SKINNY is 1.3–1.6× faster.
- With m = 1 and B transposed, AUTO is right.
- **Where it matters:** LoRA-rank products (n = 8–32) and small models such as GPT-2 sizes. In the 8B block every
  projection's selection is the right one (`block-selected.csv`).
- **Hypothesis:** at n ≤ 32 both kernels scale linearly in k, with few output elements to spread across the GPU.
  4 × 8 × 4096 takes ≥ 18 µs for 131k multiply-adds. A split-k path would cut that further than any selector change.

### 3. The fair quantized comparison moves the crossover down to m ≈ 8

The setup: 4096 × 4096, representative nonzero weights quantized once, and FP16 and packed weights both rotating
through more than 256 MiB.
- **FP16 SKINNY on rotating weights:** 101.9 µs at m = 1, against 89.8 µs when one buffer was reused. The 1.14.1
  FP16 reference was cache-assisted.
- **m = 1:** Q4 is 3.46× faster than FP16, and Q4K 3.36×.
- **m = 8:** the K formats already lose to FP16 (Q4K 0.97×, Q5K 0.93×, Q6K 0.83×). Q8 and Q4 still win by 1.14× and
  1.30×.
- **m = 16:** for Q6K, AUTO picks SKINNY (579 µs) where TILED takes 509 µs.
- **m = 64 and 512:** quantized TILED runs at 0.48–0.54× and 0.34–0.40× of FP16. Quantized TILED is nearly flat from
  m = 4 to 32, at about 460–520 µs.
- **FP16 has its own dip:** AUTO at m = 32 takes 549 µs (SMALL), against 284 µs at m = 64 (SIMDGROUP_MATRIX). In the
  1.14.1 sweep the matrix kernel at m = 32 took 1.02× TILED, against 0.40× at m = 64.
  - **Hypothesis:** below m = 64 every 64 × 64 tile is an edge tile and takes the staged path that the kernel's own
    comment measures at 3.7 TFLOP/s.
  - This affects batched decode of 17–63 sequences and speculative decoding.

### 4. Where a whole pass spends its time

- **Prefill:** the products take 92–99% of the GPU time.
  - Q4K prefill takes 3.0× FP16's time (1,838 against 613 ms at 512 tokens) because quantized products run on TILED.
  - That makes quantized prefill on the matrix hardware a ~3× prefill item for Q4K. Attention is 2–7%.
- **FP16 decode:** the products take 73–95%; attention takes 24% at 32K keys.
- **Q4K decode at 32K keys:** attention decode takes 49% of the step (12.5 ms), as much as the products. For
  quantized long-context decode, attention decode is the item.
- **Norms:** RMSNorm costs 1.0 ms per decode token, 64 dispatches of about 16 µs each. That is 6–7% of a Q4K token,
  so fusing the norm into the next product's prologue is worth that much.
- **Not a factor:** RoPE, the KV-cache write and the prefill transposes are each under 1%.

### 5. A Java caller pays about 3 ms per decode token beyond GPU time

- On Metal, decode wall time exceeds GPU time by 2.8–3.8 ms per token. That is 15% of a Q4K token at 4K context
  (18.9 against 16.1 ms).
- Host encoding is 0.8–1.3 ms of it. The rest, about 2 ms, falls between submission and completion.
- **Hypothesis:** submit and completion latency for a 450-dispatch submission.
- **Levers:**
  - replaying a recorded decode step;
  - encoding token n + 1 while token n runs;
  - fewer dispatches (fusion).
- For prefill the overhead is about 1.5% or less.

### 6. Sustained load throttles prefill, not decode

- **Decode** held steady for 20 minutes: 46.1–46.7 ms per token on Metal and 48.7–50.0 ms on MoltenVK.
- **Prefill** slowed from minute 3 on. On Metal it went from 613 ms warmed and 645 ms in minute 1 to 785–917 ms in
  minutes 3–20, which is 28–50% slower than warmed. MoltenVK went from about 1,600 ms in minutes 1–2 to 1,782–1,988 ms, which is 9–22% slower than its warmed 1,631 ms.
- Compute-bound work loses clock under sustained load; bandwidth-bound decode doesn't.
- Publish both numbers.

### 7. MoltenVK's per-dispatch profile under-reports

- On MoltenVK the profiled dispatch times sum to 40–90% of the whole-submission GPU time; on Metal they match within
  5%. MoltenVK's stage shares are therefore omitted.
- Acceptance uses whole-submission times, as planned.

## What Phase 0 has done, and what is still open

**Done:**
- seeded inputs, checked outputs and paired timings (probe and block);
- AUTO against forced kernels for the decisive cases;
- the fair quantized fixture;
- the whole-block benchmark, prefill and decode, with its stage profile;
- the 20-minute sustained run;
- machine-readable CSVs.

**Open:**
- External baselines on the same machine: MPS GEMM, llama.cpp and MLX tokens/s.
- AUTO rows and aligned m grids in the existing sweep and decode benchmarks. The existing GEMM benchmark is still
  unseeded; the probe stands in for it.
- An environment header written by the harness itself; power source and thermal state are not recorded yet.
- GB/s flags for cache-resident rows.
- The per-commit regression gate.

## What this changes in the plan

Kept: Phase 1 (selection and gradient wins) and the Phase 0 items still open.

1. **Quantized prefill on the matrix hardware is confirmed as the largest inference item:** about 3× for Q4K
   prefill, with products 97–99% of the pass.
2. **New selector items for Phase 1:**
   - the small-n region (SKINNY over SMALL, a fix in the cost model);
   - Q6K's switch at m = 16;
   - investigate FP16 at m = 17–63, where the matrix kernel's edge-tile path makes m = 32 slower than m = 64.
3. **New kernel candidates:**
   - split-k for small-n, long-k products;
   - full-tile handling for m < 64 in the matrix kernel.
4. **Decode:** for Q4K at long context, attention decode (49% at 32K) and the per-token host and submit overhead
   (about 3 ms) now rank above norm fusion (about 1 ms).
5. **Training:** this phase measured no training step yet. The block benchmark has no backward pass. Mixed precision
   stays the leading training candidate on the GEMM evidence (FP16 12.5 against FP32 5.2 TFLOP/s). A training-step
   row is the next measurement for it.

## Results

### Whole forward pass, warmed (32 layers, AUTO selection, checked)

| Backend | Weights | Phase | Tokens | Context | GPU ms (median, p10–p90) | Wall ms | Host encode ms | Tokens/s (wall) | Worst check |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Metal | FP16 | prefill | 512 | 512 | 604.1 (603.2–642.1) | 613.4 | 2.78 | 834.6 | 0.19 |
| Metal | FP16 | prefill | 2048 | 2048 | 2,811.6 (2,688.0–2,844.8) | 2,818.5 | 3.33 | 726.6 | 0.18 |
| Metal | FP16 | decode | 1 | 512 | 42.4 (42.2–43.0) | 46.2 | 1.29 | 21.6 | 0.17 |
| Metal | FP16 | decode | 1 | 4096 | 44.8 (44.3–45.3) | 48.4 | 1.28 | 20.7 | 0.13 |
| Metal | FP16 | decode | 1 | 32768 | 55.0 (54.6–55.5) | 58.2 | 1.23 | 17.2 | 0.12 |
| Metal | Q4K | prefill | 512 | 512 | 1,835.0 (1,732.3–1,854.1) | 1,838.4 | 1.26 | 278.5 | 0.19 |
| Metal | Q4K | prefill | 2048 | 2048 | 8,401.6 (7,868.7–8,533.6) | 8,406.8 | 1.77 | 243.6 | 0.18 |
| Metal | Q4K | decode | 1 | 512 | 15.3 (15.1–15.6) | 18.1 | 0.86 | 55.2 | 0.15 |
| Metal | Q4K | decode | 1 | 4096 | 16.1 (16.0–16.2) | 18.9 | 0.88 | 53.0 | 0.12 |
| Metal | Q4K | decode | 1 | 32768 | 26.2 (26.0–26.3) | 29.0 | 0.80 | 34.5 | 0.16 |
| MoltenVK | FP16 | prefill | 512 | 512 | 1,619.9 (1,560.5–1,650.1) | 1,630.7 | 2.80 | 314.0 | 0.19 |
| MoltenVK | FP16 | prefill | 2048 | 2048 | 6,378.7 (6,343.7–6,408.7) | 6,385.9 | 3.27 | 320.7 | 0.18 |
| MoltenVK | FP16 | decode | 1 | 512 | 45.5 (44.9–46.1) | 50.0 | 1.41 | 20.0 | 0.17 |
| MoltenVK | FP16 | decode | 1 | 4096 | 47.0 (46.4–47.3) | 51.1 | 1.21 | 19.6 | 0.13 |
| MoltenVK | FP16 | decode | 1 | 32768 | 57.3 (56.6–57.6) | 61.4 | 1.21 | 16.3 | 0.12 |
| MoltenVK | Q4K | prefill | 512 | 512 | 2,405.7 (2,398.5–2,456.9) | 2,410.2 | 1.24 | 212.4 | 0.19 |
| MoltenVK | Q4K | prefill | 2048 | 2048 | 10,779.9 (10,703.6–11,002.7) | 10,784.5 | 1.57 | 189.9 | 0.18 |
| MoltenVK | Q4K | decode | 1 | 512 | 20.5 (20.1–20.9) | 23.8 | 0.73 | 42.0 | 0.15 |
| MoltenVK | Q4K | decode | 1 | 4096 | 21.0 (20.7–21.1) | 24.2 | 0.71 | 41.4 | 0.12 |
| MoltenVK | Q4K | decode | 1 | 32768 | 31.1 (31.0–31.5) | 34.1 | 0.62 | 29.3 | 0.16 |

### Where a pass spends its GPU time (Metal, profiled pass)

Shares of the profiled dispatch times. On Metal they sum to within 5% of the whole-submission GPU time (profiling adds a little). On MoltenVK they sum to 40–90% of it, so MoltenVK stage shares are left out; use its whole-pass times.

| Weights | Phase | Context | Products | Attention | Norms | RoPE | KV write | Transposes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | prefill | 512 | 97% (610.2 ms) | 2% (15.5 ms) | 0% (1.6 ms) | 0% (1.1 ms) | 0% (0.3 ms) | 0% (3.1 ms) |
| FP16 | prefill | 2048 | 92% (2634.6 ms) | 7% (188.7 ms) | 0% (5.9 ms) | 0% (9.8 ms) | 0% (1.1 ms) | 0% (13.0 ms) |
| FP16 | decode | 512 | 95% (40.0 ms) | 2% (0.8 ms) | 2% (1.0 ms) | 1% (0.2 ms) | 0% (0.1 ms) | 0% (0.0 ms) |
| FP16 | decode | 4096 | 91% (40.8 ms) | 6% (2.6 ms) | 2% (1.1 ms) | 1% (0.2 ms) | 0% (0.1 ms) | 0% (0.0 ms) |
| FP16 | decode | 32768 | 73% (40.1 ms) | 24% (13.2 ms) | 2% (1.1 ms) | 0% (0.3 ms) | 0% (0.1 ms) | 0% (0.0 ms) |
| Q4K | prefill | 512 | 99% (1831.1 ms) | 1% (15.9 ms) | 0% (1.8 ms) | 0% (1.2 ms) | 0% (0.3 ms) | 0% (3.2 ms) |
| Q4K | prefill | 2048 | 97% (8320.8 ms) | 2% (203.7 ms) | 0% (6.2 ms) | 0% (4.3 ms) | 0% (7.4 ms) | 0% (14.0 ms) |
| Q4K | decode | 512 | 85% (12.4 ms) | 6% (0.9 ms) | 7% (1.0 ms) | 2% (0.2 ms) | 1% (0.1 ms) | 0% (0.0 ms) |
| Q4K | decode | 4096 | 76% (11.9 ms) | 16% (2.5 ms) | 6% (1.0 ms) | 1% (0.2 ms) | 1% (0.1 ms) | 0% (0.0 ms) |
| Q4K | decode | 32768 | 46% (11.9 ms) | 49% (12.5 ms) | 4% (1.0 ms) | 1% (0.2 ms) | 1% (0.1 ms) | 0% (0.0 ms) |

### Sustained: prefill of 512 tokens, then 32 decode tokens at 4K context, repeated for 20 minutes (FP16)

| Minute | Metal prefill ms | Metal prefill tok/s | Metal decode ms | MoltenVK prefill ms | MoltenVK prefill tok/s | MoltenVK decode ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 645.2 | 794 | 46.4 | 1,601.6 | 320 | 50.0 |
| 2 | 678.2 | 755 | 46.6 | 1,599.4 | 320 | 49.2 |
| 3 | 860.2 | 595 | 46.5 | 1,849.8 | 277 | 49.0 |
| 4 | 916.9 | 558 | 46.7 | 1,939.9 | 264 | 48.8 |
| 5 | 859.1 | 596 | 46.5 | 1,899.3 | 270 | 48.7 |
| 6 | 822.7 | 622 | 46.4 | 1,826.7 | 280 | 48.9 |
| 7 | 818.2 | 626 | 46.5 | 1,782.1 | 287 | 48.9 |
| 8 | 817.6 | 626 | 46.5 | 1,808.7 | 283 | 48.9 |
| 9 | 825.8 | 620 | 46.5 | 1,811.7 | 283 | 49.0 |
| 10 | 839.2 | 610 | 46.7 | 1,835.2 | 279 | 48.9 |
| 11 | 826.7 | 619 | 46.6 | 1,849.8 | 277 | 49.1 |
| 12 | 814.1 | 629 | 46.5 | 1,861.8 | 275 | 49.1 |
| 13 | 800.1 | 640 | 46.5 | 1,884.4 | 272 | 49.1 |
| 14 | 784.8 | 652 | 46.1 | 1,898.3 | 270 | 49.3 |
| 15 | 786.5 | 651 | 46.1 | 1,922.5 | 266 | 49.5 |
| 16 | 787.0 | 651 | 46.3 | 1,988.3 | 258 | 49.6 |
| 17 | 806.7 | 635 | 46.4 | 1,985.2 | 258 | 49.6 |
| 18 | 811.6 | 631 | 46.3 | 1,943.8 | 263 | 49.5 |
| 19 | 805.7 | 635 | 46.2 | 1,883.8 | 272 | 49.2 |
| 20 | 802.9 | 638 | 46.1 | 1,887.1 | 271 | 49.2 |

### Square GEMM on seeded inputs (TFLOP/s by median GPU time; wall/GPU from the same rounds)

| Backend | Size | Layout | Input | AUTO picks | AUTO TFLOP/s | Best forced | Best TFLOP/s | Wall / GPU | Worst check |
|---|---|---|---|---|---:|---|---:|---:|---:|
| Metal | 2048³ | B | FP16 | SIMDGROUP_MATRIX | 12.40 | SIMDGROUP_MATRIX | 12.47 | 1.063 | 2.2e-04 |
| Metal | 2048³ | B^T | FP16 | SIMDGROUP_MATRIX | 12.39 | SIMDGROUP_MATRIX | 12.39 | 1.027 | 2.1e-04 |
| Metal | 2048³ | B | FP32 | TILED | 5.18 | TILED | 5.17 | 1.017 | 1.4e-04 |
| Metal | 2048³ | B^T | FP32 | TILED | 4.92 | TILED | 4.91 | 1.016 | 1.6e-04 |
| Metal | 2048³ | B | BF16 | TILED | 4.95 | TILED | 4.96 | 1.014 | 1.6e-04 |
| Metal | 2048³ | B^T | BF16 | TILED | 4.90 | TILED | 4.91 | 1.014 | 1.0e-04 |
| Metal | 4096³ | B | FP16 | SIMDGROUP_MATRIX | 12.53 | SIMDGROUP_MATRIX | 12.59 | 1.009 | 1.2e-04 |
| Metal | 4096³ | B^T | FP16 | SIMDGROUP_MATRIX | 12.23 | SIMDGROUP_MATRIX | 12.43 | 1.009 | 5.3e-05 |
| Metal | 4096³ | B | FP32 | TILED | 5.23 | TILED | 5.20 | 1.011 | 1.1e-04 |
| Metal | 4096³ | B^T | FP32 | TILED | 4.79 | TILED | 4.75 | 1.010 | 9.1e-05 |
| Metal | 4096³ | B | BF16 | TILED | 4.92 | TILED | 4.92 | 1.010 | 6.5e-05 |
| Metal | 4096³ | B^T | BF16 | TILED | 4.76 | TILED | 4.70 | 1.010 | 2.5e-05 |
| MoltenVK | 2048³ | B | FP16 | TILED | 5.17 | TILED | 5.16 | 1.075 | 2.2e-04 |
| MoltenVK | 2048³ | B^T | FP16 | TILED | 3.85 | TILED | 3.87 | 1.045 | 2.1e-04 |
| MoltenVK | 2048³ | B | FP32 | TILED | 4.28 | TILED | 4.28 | 1.034 | 1.4e-04 |
| MoltenVK | 2048³ | B^T | FP32 | TILED | 4.34 | TILED | 4.27 | 1.030 | 1.6e-04 |
| MoltenVK | 2048³ | B | BF16 | TILED | 4.39 | TILED | 4.33 | 1.028 | 1.6e-04 |
| MoltenVK | 2048³ | B^T | BF16 | TILED | 4.28 | TILED | 4.37 | 1.029 | 1.0e-04 |
| MoltenVK | 4096³ | B | FP16 | TILED | 4.85 | TILED | 4.86 | 1.019 | 1.2e-04 |
| MoltenVK | 4096³ | B^T | FP16 | TILED | 4.77 | TILED | 4.72 | 1.023 | 5.3e-05 |
| MoltenVK | 4096³ | B | FP32 | TILED | 4.96 | TILED | 4.87 | 1.020 | 1.1e-04 |
| MoltenVK | 4096³ | B^T | FP32 | TILED | 4.75 | TILED | 4.65 | 1.019 | 9.1e-05 |
| MoltenVK | 4096³ | B | BF16 | TILED | 4.81 | TILED | 4.58 | 1.020 | 6.5e-05 |
| MoltenVK | 4096³ | B^T | BF16 | TILED | 4.58 | TILED | 4.58 | 1.020 | 2.5e-05 |

### Selector cases (FP16): AUTO against the fastest forced kernel

| Backend | m | n | k | Layout | AUTO picks | AUTO µs | Fastest | Fastest µs | Regret |
|---|---:|---:|---:|---|---|---:|---|---:|---:|
| Metal | 4 | 8 | 2048 | B^T | SMALL | 51.4 | SKINNY | 12.8 | 4.01 **miss** |
| Metal | 4 | 8 | 2048 | B | SMALL | 59.7 | SMALL | 59.7 | 1.00 |
| Metal | 4 | 8 | 4096 | B^T | SMALL | 102.6 | SKINNY | 18.6 | 5.53 **miss** |
| Metal | 4 | 8 | 4096 | B | SMALL | 119.4 | SMALL | 116.8 | 1.02 |
| Metal | 4 | 8 | 8192 | B^T | SMALL | 204.4 | SKINNY | 29.6 | 6.91 **miss** |
| Metal | 4 | 8 | 8192 | B | SMALL | 229.5 | SMALL | 229.5 | 1.00 |
| Metal | 4 | 16 | 2048 | B^T | SMALL | 50.8 | SKINNY | 13.2 | 3.83 **miss** |
| Metal | 4 | 16 | 2048 | B | SMALL | 57.7 | SMALL | 58.1 | 0.99 |
| Metal | 4 | 16 | 4096 | B^T | SMALL | 102.1 | SKINNY | 19.5 | 5.24 **miss** |
| Metal | 4 | 16 | 4096 | B | SMALL | 117.2 | SMALL | 114.6 | 1.02 |
| Metal | 4 | 16 | 8192 | B^T | SMALL | 198.9 | SKINNY | 31.3 | 6.35 **miss** |
| Metal | 4 | 16 | 8192 | B | SMALL | 228.3 | SMALL | 231.3 | 0.99 |
| Metal | 4 | 32 | 2048 | B^T | SMALL | 51.0 | SKINNY | 13.3 | 3.82 **miss** |
| Metal | 4 | 32 | 2048 | B | SMALL | 58.6 | SMALL | 58.4 | 1.00 |
| Metal | 4 | 32 | 4096 | B^T | SMALL | 100.6 | SKINNY | 19.3 | 5.21 **miss** |
| Metal | 4 | 32 | 4096 | B | SMALL | 115.9 | SMALL | 116.0 | 1.00 |
| Metal | 4 | 32 | 8192 | B^T | SMALL | 197.8 | SKINNY | 31.4 | 6.30 **miss** |
| Metal | 4 | 32 | 8192 | B | SMALL | 230.6 | SMALL | 230.8 | 1.00 |
| Metal | 8 | 8 | 2048 | B^T | SMALL | 51.9 | SKINNY | 23.5 | 2.21 **miss** |
| Metal | 8 | 8 | 2048 | B | SMALL | 59.9 | SMALL | 59.1 | 1.01 |
| Metal | 8 | 8 | 4096 | B^T | SMALL | 103.1 | SKINNY | 34.7 | 2.97 **miss** |
| Metal | 8 | 8 | 4096 | B | SMALL | 117.9 | SMALL | 119.5 | 0.99 |
| Metal | 8 | 8 | 8192 | B^T | SMALL | 204.3 | SKINNY | 56.5 | 3.61 **miss** |
| Metal | 8 | 8 | 8192 | B | SMALL | 236.4 | SMALL | 234.4 | 1.01 |
| Metal | 8 | 16 | 2048 | B^T | SMALL | 51.4 | SKINNY | 23.9 | 2.15 **miss** |
| Metal | 8 | 16 | 2048 | B | SMALL | 59.7 | SMALL | 58.7 | 1.02 |
| Metal | 8 | 16 | 4096 | B^T | SMALL | 101.8 | SKINNY | 35.0 | 2.91 **miss** |
| Metal | 8 | 16 | 4096 | B | SMALL | 116.3 | SMALL | 115.9 | 1.00 |
| Metal | 8 | 16 | 8192 | B^T | SMALL | 200.8 | SKINNY | 57.3 | 3.51 **miss** |
| Metal | 8 | 16 | 8192 | B | SMALL | 231.5 | SMALL | 231.8 | 1.00 |
| Metal | 8 | 32 | 2048 | B^T | SMALL | 51.7 | SKINNY | 24.0 | 2.16 **miss** |
| Metal | 8 | 32 | 2048 | B | SMALL | 59.6 | SMALL | 59.5 | 1.00 |
| Metal | 8 | 32 | 4096 | B^T | SMALL | 101.9 | SKINNY | 35.1 | 2.91 **miss** |
| Metal | 8 | 32 | 4096 | B | SMALL | 117.7 | SMALL | 117.7 | 1.00 |
| Metal | 8 | 32 | 8192 | B^T | SMALL | 201.2 | SKINNY | 57.3 | 3.51 **miss** |
| Metal | 8 | 32 | 8192 | B | SMALL | 234.9 | SMALL | 234.8 | 1.00 |
| Metal | 16 | 8 | 2048 | B^T | SMALL | 56.6 | SKINNY | 44.4 | 1.28 **miss** |
| Metal | 16 | 8 | 2048 | B | SMALL | 60.0 | SMALL | 60.0 | 1.00 |
| Metal | 16 | 8 | 4096 | B^T | SMALL | 112.5 | SKINNY | 65.0 | 1.73 **miss** |
| Metal | 16 | 8 | 4096 | B | SMALL | 119.6 | SMALL | 119.7 | 1.00 |
| Metal | 16 | 8 | 8192 | B^T | SMALL | 220.3 | SKINNY | 106.6 | 2.07 **miss** |
| Metal | 16 | 8 | 8192 | B | SMALL | 238.2 | SMALL | 238.7 | 1.00 |
| Metal | 16 | 16 | 2048 | B^T | SMALL | 56.8 | SKINNY | 44.8 | 1.27 **miss** |
| Metal | 16 | 16 | 2048 | B | SMALL | 59.9 | SMALL | 59.9 | 1.00 |
| Metal | 16 | 16 | 4096 | B^T | SMALL | 112.8 | SKINNY | 65.7 | 1.72 **miss** |
| Metal | 16 | 16 | 4096 | B | SMALL | 118.7 | SMALL | 118.4 | 1.00 |
| Metal | 16 | 16 | 8192 | B^T | SMALL | 220.5 | SKINNY | 107.7 | 2.05 **miss** |
| Metal | 16 | 16 | 8192 | B | SMALL | 236.8 | SMALL | 236.9 | 1.00 |
| Metal | 16 | 32 | 2048 | B^T | SMALL | 57.0 | SKINNY | 45.0 | 1.27 **miss** |
| Metal | 16 | 32 | 2048 | B | SMALL | 60.3 | SMALL | 60.3 | 1.00 |
| Metal | 16 | 32 | 4096 | B^T | SMALL | 112.6 | SKINNY | 65.6 | 1.72 **miss** |
| Metal | 16 | 32 | 4096 | B | SMALL | 119.6 | SMALL | 119.5 | 1.00 |
| Metal | 16 | 32 | 8192 | B^T | SMALL | 220.2 | SKINNY | 107.7 | 2.04 **miss** |
| Metal | 16 | 32 | 8192 | B | SMALL | 238.9 | SMALL | 238.3 | 1.00 |
| Metal | 1 | 512 | 512 | B | SMALL | 16.2 | SKINNY | 12.2 | 1.33 **miss** |
| Metal | 1 | 512 | 512 | B^T | SKINNY | 3.8 | SKINNY | 3.8 | 1.01 |
| Metal | 1 | 512 | 1024 | B | SMALL | 30.5 | SKINNY | 20.1 | 1.52 **miss** |
| Metal | 1 | 512 | 1024 | B^T | SKINNY | 4.3 | SKINNY | 4.3 | 1.00 |
| Metal | 1 | 512 | 2048 | B | SMALL | 59.3 | SKINNY | 36.7 | 1.62 **miss** |
| Metal | 1 | 512 | 2048 | B^T | SKINNY | 5.6 | SKINNY | 5.5 | 1.02 |
| Metal | 1 | 1024 | 512 | B | SMALL | 16.9 | SKINNY | 12.2 | 1.39 **miss** |
| Metal | 1 | 1024 | 512 | B^T | SKINNY | 4.1 | SKINNY | 4.1 | 1.01 |
| Metal | 1 | 1024 | 1024 | B | SMALL | 32.2 | SKINNY | 21.0 | 1.53 **miss** |
| Metal | 1 | 1024 | 1024 | B^T | SKINNY | 4.9 | SKINNY | 4.9 | 0.99 |
| Metal | 1 | 1024 | 2048 | B | SMALL | 61.8 | SKINNY | 37.8 | 1.63 **miss** |
| Metal | 1 | 1024 | 2048 | B^T | SKINNY | 6.0 | SKINNY | 6.1 | 0.99 |
| Metal | 1 | 2048 | 512 | B | SKINNY | 12.2 | SKINNY | 12.3 | 1.00 |
| Metal | 1 | 2048 | 512 | B^T | SKINNY | 8.0 | SKINNY | 8.0 | 1.00 |
| Metal | 1 | 2048 | 1024 | B | SKINNY | 20.6 | SKINNY | 20.7 | 1.00 |
| Metal | 1 | 2048 | 1024 | B^T | SKINNY | 10.0 | SKINNY | 10.0 | 1.00 |
| Metal | 1 | 2048 | 2048 | B | SKINNY | 37.3 | SKINNY | 37.3 | 1.00 |
| Metal | 1 | 2048 | 2048 | B^T | SKINNY | 18.0 | SKINNY | 17.7 | 1.02 |
| MoltenVK | 4 | 8 | 2048 | B^T | SMALL | 56.9 | SKINNY | 18.6 | 3.06 **miss** |
| MoltenVK | 4 | 8 | 2048 | B | SMALL | 63.5 | SMALL | 63.4 | 1.00 |
| MoltenVK | 4 | 8 | 4096 | B^T | SMALL | 107.1 | SKINNY | 24.3 | 4.41 **miss** |
| MoltenVK | 4 | 8 | 4096 | B | SMALL | 122.5 | SMALL | 121.9 | 1.00 |
| MoltenVK | 4 | 8 | 8192 | B^T | SMALL | 206.7 | SKINNY | 35.0 | 5.90 **miss** |
| MoltenVK | 4 | 8 | 8192 | B | SMALL | 236.7 | SMALL | 237.0 | 1.00 |
| MoltenVK | 4 | 16 | 2048 | B^T | SMALL | 56.1 | SKINNY | 18.8 | 2.98 **miss** |
| MoltenVK | 4 | 16 | 2048 | B | SMALL | 62.9 | SMALL | 62.9 | 1.00 |
| MoltenVK | 4 | 16 | 4096 | B^T | SMALL | 106.1 | SKINNY | 24.9 | 4.27 **miss** |
| MoltenVK | 4 | 16 | 4096 | B | SMALL | 119.8 | SMALL | 120.5 | 0.99 |
| MoltenVK | 4 | 16 | 8192 | B^T | SMALL | 204.1 | SKINNY | 37.4 | 5.45 **miss** |
| MoltenVK | 4 | 16 | 8192 | B | SMALL | 234.6 | SMALL | 234.7 | 1.00 |
| MoltenVK | 4 | 32 | 2048 | B^T | SMALL | 56.3 | SKINNY | 19.2 | 2.94 **miss** |
| MoltenVK | 4 | 32 | 2048 | B | SMALL | 63.8 | SMALL | 63.8 | 1.00 |
| MoltenVK | 4 | 32 | 4096 | B^T | SMALL | 106.4 | SKINNY | 25.0 | 4.26 **miss** |
| MoltenVK | 4 | 32 | 4096 | B | SMALL | 121.7 | SMALL | 121.9 | 1.00 |
| MoltenVK | 4 | 32 | 8192 | B^T | SMALL | 204.3 | SKINNY | 37.2 | 5.50 **miss** |
| MoltenVK | 4 | 32 | 8192 | B | SMALL | 237.2 | SMALL | 237.5 | 1.00 |
| MoltenVK | 8 | 8 | 2048 | B^T | SMALL | 57.1 | SKINNY | 30.2 | 1.89 **miss** |
| MoltenVK | 8 | 8 | 2048 | B | SMALL | 63.7 | SMALL | 63.8 | 1.00 |
| MoltenVK | 8 | 8 | 4096 | B^T | SMALL | 107.7 | SKINNY | 40.9 | 2.64 **miss** |
| MoltenVK | 8 | 8 | 4096 | B | SMALL | 122.6 | SMALL | 123.0 | 1.00 |
| MoltenVK | 8 | 8 | 8192 | B^T | SMALL | 209.1 | SKINNY | 62.8 | 3.33 **miss** |
| MoltenVK | 8 | 8 | 8192 | B | SMALL | 240.0 | SMALL | 240.2 | 1.00 |
| MoltenVK | 8 | 16 | 2048 | B^T | SMALL | 56.9 | SKINNY | 30.8 | 1.85 **miss** |
| MoltenVK | 8 | 16 | 2048 | B | SMALL | 63.9 | SMALL | 64.0 | 1.00 |
| MoltenVK | 8 | 16 | 4096 | B^T | SMALL | 107.7 | SKINNY | 41.9 | 2.57 **miss** |
| MoltenVK | 8 | 16 | 4096 | B | SMALL | 121.7 | SMALL | 121.7 | 1.00 |
| MoltenVK | 8 | 16 | 8192 | B^T | SMALL | 207.9 | SKINNY | 63.8 | 3.26 **miss** |
| MoltenVK | 8 | 16 | 8192 | B | SMALL | 238.4 | SMALL | 238.5 | 1.00 |
| MoltenVK | 8 | 32 | 2048 | B^T | SMALL | 57.1 | SKINNY | 30.9 | 1.85 **miss** |
| MoltenVK | 8 | 32 | 2048 | B | SMALL | 64.9 | SMALL | 64.9 | 1.00 |
| MoltenVK | 8 | 32 | 4096 | B^T | SMALL | 107.5 | SKINNY | 40.7 | 2.64 **miss** |
| MoltenVK | 8 | 32 | 4096 | B | SMALL | 123.1 | SMALL | 124.7 | 0.99 |
| MoltenVK | 8 | 32 | 8192 | B^T | SMALL | 207.9 | SKINNY | 63.6 | 3.27 **miss** |
| MoltenVK | 8 | 32 | 8192 | B | SMALL | 241.6 | SMALL | 241.8 | 1.00 |
| MoltenVK | 16 | 8 | 2048 | B^T | SMALL | 61.9 | SKINNY | 52.6 | 1.18 **miss** |
| MoltenVK | 16 | 8 | 2048 | B | SMALL | 65.0 | SMALL | 64.8 | 1.00 |
| MoltenVK | 16 | 8 | 4096 | B^T | SMALL | 117.0 | SKINNY | 73.7 | 1.59 **miss** |
| MoltenVK | 16 | 8 | 4096 | B | SMALL | 125.6 | SMALL | 125.2 | 1.00 |
| MoltenVK | 16 | 8 | 8192 | B^T | SMALL | 225.7 | SKINNY | 113.8 | 1.98 **miss** |
| MoltenVK | 16 | 8 | 8192 | B | SMALL | 244.1 | SMALL | 244.1 | 1.00 |
| MoltenVK | 16 | 16 | 2048 | B^T | SMALL | 62.6 | SKINNY | 53.1 | 1.18 **miss** |
| MoltenVK | 16 | 16 | 2048 | B | SMALL | 65.1 | SMALL | 65.3 | 1.00 |
| MoltenVK | 16 | 16 | 4096 | B^T | SMALL | 119.1 | SKINNY | 74.4 | 1.60 **miss** |
| MoltenVK | 16 | 16 | 4096 | B | SMALL | 124.4 | SMALL | 125.0 | 1.00 |
| MoltenVK | 16 | 16 | 8192 | B^T | SMALL | 227.1 | SKINNY | 115.3 | 1.97 **miss** |
| MoltenVK | 16 | 16 | 8192 | B | SMALL | 243.0 | SMALL | 243.0 | 1.00 |
| MoltenVK | 16 | 32 | 2048 | B^T | SMALL | 62.7 | SKINNY | 53.3 | 1.18 **miss** |
| MoltenVK | 16 | 32 | 2048 | B | SMALL | 65.5 | SMALL | 65.5 | 1.00 |
| MoltenVK | 16 | 32 | 4096 | B^T | SMALL | 119.4 | SKINNY | 74.7 | 1.60 **miss** |
| MoltenVK | 16 | 32 | 4096 | B | SMALL | 125.8 | SMALL | 126.3 | 1.00 |
| MoltenVK | 16 | 32 | 8192 | B^T | SMALL | 227.3 | SKINNY | 116.0 | 1.96 **miss** |
| MoltenVK | 16 | 32 | 8192 | B | SMALL | 245.1 | SMALL | 245.1 | 1.00 |
| MoltenVK | 1 | 512 | 512 | B | SMALL | 21.1 | SKINNY | 17.8 | 1.19 **miss** |
| MoltenVK | 1 | 512 | 512 | B^T | SKINNY | 9.5 | SKINNY | 9.5 | 1.01 |
| MoltenVK | 1 | 512 | 1024 | B | SMALL | 36.1 | SKINNY | 25.8 | 1.40 **miss** |
| MoltenVK | 1 | 512 | 1024 | B^T | SKINNY | 10.0 | SKINNY | 9.7 | 1.03 |
| MoltenVK | 1 | 512 | 2048 | B | SMALL | 64.9 | SKINNY | 40.0 | 1.62 **miss** |
| MoltenVK | 1 | 512 | 2048 | B^T | SKINNY | 11.1 | SKINNY | 11.0 | 1.01 |
| MoltenVK | 1 | 1024 | 512 | B | SMALL | 22.0 | SKINNY | 17.9 | 1.23 **miss** |
| MoltenVK | 1 | 1024 | 512 | B^T | SKINNY | 9.8 | SKINNY | 9.7 | 1.01 |
| MoltenVK | 1 | 1024 | 1024 | B | SMALL | 37.1 | SKINNY | 24.4 | 1.52 **miss** |
| MoltenVK | 1 | 1024 | 1024 | B^T | SKINNY | 11.0 | SKINNY | 11.0 | 1.00 |
| MoltenVK | 1 | 1024 | 2048 | B | SMALL | 67.1 | SKINNY | 42.6 | 1.58 **miss** |
| MoltenVK | 1 | 1024 | 2048 | B^T | SKINNY | 12.5 | SKINNY | 12.4 | 1.00 |
| MoltenVK | 1 | 2048 | 512 | B | SKINNY | 18.1 | SKINNY | 18.3 | 0.99 |
| MoltenVK | 1 | 2048 | 512 | B^T | SKINNY | 14.0 | SKINNY | 14.5 | 0.96 |
| MoltenVK | 1 | 2048 | 1024 | B | SKINNY | 26.5 | SKINNY | 26.6 | 1.00 |
| MoltenVK | 1 | 2048 | 1024 | B^T | SKINNY | 16.5 | SKINNY | 16.6 | 1.00 |
| MoltenVK | 1 | 2048 | 2048 | B | SKINNY | 44.2 | SKINNY | 44.1 | 1.00 |
| MoltenVK | 1 | 2048 | 2048 | B^T | SKINNY | 24.2 | SKINNY | 24.8 | 0.98 |

Misses (regret > 1.15): Metal 33 of 72, MoltenVK 33 of 72.

### Fair quantized comparison, 4096 × 4096 (µs, median; both FP16 and packed weights rotate through > 256 MiB)

**Metal** (each cell: the faster of SKINNY (S) and TILED (T), and its speed-up over FP16 AUTO; `!` = AUTO more than 10% slower than the fastest)

| m | FP16 AUTO (picks) | Q8 | Q4 | Q4M | Q4K | Q5K | Q6K |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 101.9 (SKINNY) | 45.4 S (2.25×) | 29.5 S (3.46×) | 30.3 S (3.37×) | 30.4 S (3.36×) | 37.9 S (2.69×) | 41.3 S (2.47×) |
| 2 | 109.3 (SKINNY) | 59.9 S (1.82×) | 44.6 S (2.45×) | 50.4 S (2.17×) | 54.6 S (2.00×) | 58.5 S (1.87×) | 67.8 S (1.61×) |
| 4 | 132.8 (SKINNY) | 72.8 S (1.82×) | 62.2 S (2.13×) | 67.3 S (1.97×) | 72.3 S (1.84×) | 76.6 S (1.73×) | 97.1 S (1.37×) |
| 8 | 204.6 (SKINNY) | 179.9 S (1.14×) | 157.9 S (1.30×) | 191.3 S (1.07×) | 210.9 S (0.97×) | 220.0 S (0.93×) | 246.2 S (0.83×) |
| 16 | 322.6 (SMALL) | 336.7 S (0.96×) | 303.8 S (1.06×) | 381.9 S (0.84×) | 413.1 S (0.78×) | 428.5 S (0.75×) | 508.6 T (0.63×)! |
| 32 | 549.3 (SMALL) | 492.2 T (1.12×) | 486.1 T (1.13×) | 534.7 T (1.03×) | 533.2 T (1.03×) | 549.7 T (1.00×) | 516.4 T (1.06×) |
| 64 | 283.7 (SIMDGROUP_MATRIX) | 528.2 T (0.54×) | 525.5 T (0.54×) | 577.3 T (0.49×) | 576.0 T (0.49×) | 586.1 T (0.48×) | 558.2 T (0.51×) |
| 512 | 1409.2 (SIMDGROUP_MATRIX) | 3547.2 T (0.40×) | 3556.7 T (0.40×) | 4058.8 T (0.35×) | 4011.2 T (0.35×) | 4108.4 T (0.34×) | 3975.9 T (0.35×) |

**MoltenVK** (each cell: the faster of SKINNY (S) and TILED (T), and its speed-up over FP16 AUTO; `!` = AUTO more than 10% slower than the fastest)

| m | FP16 AUTO (picks) | Q8 | Q4 | Q4M | Q4K | Q5K | Q6K |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 107.6 (SKINNY) | 48.9 S (2.20×) | 34.5 S (3.12×) | 39.8 S (2.70×) | 44.2 S (2.43×) | 50.6 S (2.12×) | 48.1 S (2.24×) |
| 2 | 116.6 (SKINNY) | 69.6 S (1.68×) | 55.4 S (2.10×) | 76.7 S (1.52×) | 82.0 S (1.42×) | 86.2 S (1.35×) | 75.9 S (1.54×) |
| 4 | 141.3 (SKINNY) | 90.2 S (1.57×) | 79.9 S (1.77×) | 118.2 S (1.20×) | 123.6 S (1.14×) | 128.6 S (1.10×) | 109.6 S (1.29×) |
| 8 | 213.9 (SKINNY) | 198.6 S (1.08×) | 178.0 S (1.20×) | 275.4 S (0.78×) | 300.8 S (0.71×) | 306.2 S (0.70×) | 276.6 S (0.77×) |
| 16 | 332.8 (SMALL) | 378.2 S (0.88×) | 354.9 S (0.94×) | 643.2 S (0.52×) | 659.2 S (0.50×) | 669.0 S (0.50×) | 602.1 T (0.55×)! |
| 32 | 570.3 (SMALL) | 552.4 T (1.03×) | 550.6 T (1.04×) | 968.3 T (0.59×) | 960.8 T (0.59×) | 963.7 T (0.59×) | 613.2 T (0.93×) |
| 64 | 684.1 (TILED) | 568.2 T (1.20×) | 570.1 T (1.20×) | 1000.1 T (0.68×) | 990.5 T (0.69×) | 995.3 T (0.69×) | 636.9 T (1.07×) |
| 512 | 3553.4 (TILED) | 3285.3 T (1.08×) | 3309.7 T (1.07×) | 5712.7 T (0.62×) | 5724.3 T (0.62×) | 5750.7 T (0.62×) | 4001.2 T (0.89×) |


## Method

- **Probe (`-Dperf.probe=true`, `PerfProbeConformance`):**
  - Every case uploads seeded inputs and fills C with NaN.
  - It runs once and checks 64 sampled elements of C (32 for quantized) against a double-precision product. The
    bound is the loose 4 k 2⁻²⁴ Σ|a b|, which a wrong kernel or garbage input exceeds by orders of magnitude. Every
    case is far inside it: the worst is 8.4e-4 of the bound.
  - It then times 10 paired rounds of at least 20 ms each, after a ramp, keeping GPU, encode and wall time from the
    same submission.
  - For quantized cases, the weights are normal values quantized once by `QuantizedMatrix.quantize`. FP16 and packed
    weights rotate through copies totalling more than 256 MiB.
- **Block (`-Dblock.benchmark=true`, `TransformerBlockConformance`):**
  - Each layer: RMSNorm; Q, K and V projections; RoPE; KV-cache write; fused causal attention over the paged cache
    (prefill transposes Q to head-major and O back with `Cast`); the output projection with the residual in its
    epilogue; RMSNorm; up; gate with SiLU and the up result as its gate; down with the residual.
  - Before timing, one layer runs alone and every stage is checked on 24 sampled rows against a double-precision
    computation from that stage's own GPU inputs: norms, products, RoPE and attention, including the random KV
    history. This is a sanity bound, not the kernels' contracts. The worst is 0.19 of it.
  - Each phase is timed as 10 whole-pass submissions after 2 warm ones.
  - The profiled pass is a separate submission.
- **Sustained:** prefill 512, then 32 decode tokens at 4K context, repeated for 20 minutes, with per-minute medians.
- **Order and conditions:** Metal ran, then MoltenVK, with cool-downs of 60–120 s between suites. The runs were not
  interleaved and the machine was otherwise idle. Power source and thermal state were not recorded.

## Reproduce

```sh
props="-Dperf.probe=true -Dblock.benchmark=true -Dsurefire.failIfNoSpecifiedTests=false"
mvn -o -pl dynamis-gpu-metal,dynamis-gpu-primitives -am -Pmetal-native,metal-hardware test ${=props} \
  '-Dtest=MetalPerfProbeTest,MetalTransformerBlockTest'
# sustained: add -Dblock.prefill=512 -Dblock.decode=4096 -Dblock.weights=FP16 -Dblock.sustainMinutes=20
# MoltenVK: the Vulkan command from benchmarks-1.14.1.md with VulkanPerfProbeTest,VulkanTransformerBlockTest
```
