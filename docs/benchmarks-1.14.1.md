# DynamisGPU 1.14.1 benchmarks

Every benchmark in the repository, plus a new harness for the training and kit kernels, run on DynamisGPU 1.14.1
(`95c38e7`) on 2026-11-25. All tables come from one dataset: [results.csv](evidence/benchmarks-1.14.1/results.csv)
(2,242 values, including the GEMM shape sweep and the transposed-B and wall-time rates that the tables leave out).

**Machine:** Apple M4 Max, macOS 27.0.1, Temurin 25.0.4. Two backends on the same GPU:
- **Metal:** native, with the hand-written SIMD-group matrix kernels.
- **MoltenVK:** 1.4.1, through the Vulkan SDK 1.4.341 loader. It runs the portable Slang kernels: no SIMD-group matrix GEMM, matrix attention backward or direct short decode.

## Summary

- **Large matrix multiplies:** Metal's SIMD-group matrix GEMM reaches 12.5 TFLOP/s in FP16 at 4096³, 2.4 times the
  portable TILED kernel. TILED gives 5.3 TFLOP/s on both backends.
- **Decode products:** SKINNY reads FP16 weights at 70–79% of streaming bandwidth for one token with transposed B.
  The share falls as m grows (34–41% at m = 8). From m = 8 with B not transposed, SMALL is faster.
- **Quantized decode:** at m = 1, Q4 is 3.1–4.2 times faster than FP16 on Metal and Q8 2.1–2.4 times. By m = 16
  the quantized products are at most 16% faster than FP16 and mostly slower; Q6K takes 1.7–2.5 times as long.
- **Attention:** fused prefill reaches 13.3 TFLOP/s dense-equivalent on Metal at 2048 tokens, against 7.3 on
  MoltenVK. The training backward takes 7.4 times the forward on Metal. On MoltenVK, which has no matrix backward, it is
  4.1 times slower than on Metal.
- **Masked prefill (new in 1.14.1):** at 512 tokens the fused kernel is 1.5 times faster than the composed path on
  Metal and 2.4 times faster on MoltenVK.
- **Long-context decode:** 32K-key decode reads K and V at 403 GB/s, about 80% of streaming. Batched paged decode
  cuts host encoding from 49–294 µs to about 6 µs.
- **Training kernels at bandwidth:** the elementwise ops, row reductions, AdamW and SGD (474–482 GB/s, about 95% of
  streaming), RoPE and the logic ops all run at or near memory bandwidth.
- **Slowest relative to their traffic:** these are the places to optimize first (best Metal time):
  - norm backward: RMS 208 µs, 7 times its forward, at 121 GB/s;
  - `Scan.rows`: 794 µs Metal and 1,171 µs MoltenVK, 115–169 GB/s;
  - column reduction (`Reduce.axis` over axis 0): 382 µs, 2.7 times the row reduction;
  - `Reduce.maxBackward`: 357 µs, 2.5 times `argmaxAxis`;
  - `Gather.gather`: 158 GB/s;
  - the read-only `NanCheck` and `GradientClip.squaredNorm`: about 210 GB/s, roughly 42% of streaming;
  - softmax over 32,000-wide rows: 227 GB/s;
  - `Cast`: 260–300 GB/s.
- **Many-tensor clipping:** `GradientClip.squaredNorm` over 64 tensors costs 81 µs of host encoding and 2.2 times the
  GPU time of one tensor of the same total size.
- **Runtime overhead:** MoltenVK's submission overhead is about 3 times Metal's: 50 dispatches take 759 µs from encode
  to done, against 244 µs.

## Method

- **Existing benchmarks:** the opt-in hardware tests (`gemm.benchmark`, `gemm.decode`, `gemm.shapes`,
  `inference.benchmark`, `quant.decode`, `attentionMask.benchmark`, `pagedDecode.benchmark`, `compute.benchmark`), as
  they stand at 1.14.1.
- **Timing:**
  - GPU time comes from `GpuSubmission.gpuTime()` over back-to-back dispatches in one submission. Each round lasts
    at least 20 ms, and the result is the best of four warmed rounds.
  - The masked prefill and paged working-set tables are medians of four paired rounds.
  - The runtime overhead figures are the third (warm) run of each test.
- **New harness (`KitBenchmarkConformance`, `-Dkit.benchmark=true`):** times the kernels that nothing else times,
  through the same `GpuBenchmark` helper:
  - softmax and cross-entropy;
  - the forward and backward passes of the norms;
  - elementwise, reduce, scan, gather and scatter-add;
  - RoPE backward;
  - the optimizers, clipping, logic, cast and NaN checks.
  It ran four times per backend, alternating backends, and the tables keep each kernel's best time.
- **Thermal effect:** a fifth run, made straight after the 2.5-minute GEMM suites, was up to 42% slower on Metal and
  was discarded. In the kept runs, the slowest is within 5% of the best for all but five kernels; the largest spread is 13%.
- **Agreement with the release notes:** the existing suites reproduce the 1.14.1 notes to within 2%. Masked prefill
  is 235.8 against 238.5 µs on Metal and 275.0 against 275.2 µs on MoltenVK; 32K decode is 332.6 against 337.7 µs.
- **Known harness limits:**
  - The new kit harness initializes every input. The existing GEMM throughput and shape-sweep benchmarks don't: they
    time private buffers whose contents are undefined, against AGENTS.md's rule.
  - The GEMM throughput benchmark takes the best GPU time and the best wall time independently across its rounds, so
    the small gap between them is not a paired measurement of Java-side overhead.
  - The quantized decode benchmark rotates through over 256 MiB of zero-filled packed weights, while its FP16
    reference reuses one buffer. Cache conditions differ, so treat its speed-up column as indicative.
  - The fixes are Phase 0 of [the performance plan](performance-plan-1.15.md).
- **Inputs:** the kit harness and the attention, inference and sampling benchmarks use initialized inputs.
  Data-dependent kernels use the inputs noted in their rows.

## Results

### GEMM throughput (B not transposed, GFLOP/s by GPU time)

| Algorithm | Input | Backend | 512³ | 1024³ | 2048³ | 4096³ |
|---|---|---|---:|---:|---:|---:|
| TILED | FP16 | Metal | 3,572 | 4,606 | 5,215 | 5,299 |
| TILED | BF16 | Metal | 3,393 | 4,447 | 5,028 | 5,098 |
| TILED | FP32 | Metal | 3,745 | 4,631 | 5,241 | 5,214 |
| SIMDGROUP_MATRIX | FP16 | Metal | 6,805 | 10,752 | 12,421 | 12,504 |
| SIMDGROUP_MATRIX | FP32 | Metal | 1,357 | 2,338 | 2,923 | 2,837 |
| SMALL | FP16 | Metal | 2,112 | 2,178 | 2,166 | 2,177 |
| SMALL | BF16 | Metal | 1,925 | 2,024 | 2,026 | 2,030 |
| SMALL | FP32 | Metal | 2,027 | 2,125 | 2,143 | 2,127 |
| TILED | FP16 | MoltenVK | 3,363 | 4,546 | 5,215 | 5,314 |
| TILED | BF16 | MoltenVK | 3,201 | 4,375 | 4,997 | 5,085 |
| TILED | FP32 | MoltenVK | 3,557 | 4,568 | 5,189 | 5,260 |
| SMALL | FP16 | MoltenVK | 1,959 | 2,141 | 2,178 | 2,195 |
| SMALL | BF16 | MoltenVK | 1,867 | 2,044 | 2,069 | 2,081 |
| SMALL | FP32 | MoltenVK | 1,962 | 2,155 | 2,197 | 2,193 |

With B transposed (`Shape.ofTransposedB`) the rates are 0–8% lower from 1024 up (FP32 TILED loses the most, 5–8%) and up to 10% lower at 512. Wall-time rates are within 2.5% of GPU-time rates from 1024 up, and 2–8% lower at 512. Both are in the CSV.

### GEMM decode, FP16 weights (µs, and share of streaming read bandwidth)

SKINNY is the decode kernel. The last column is the fastest other algorithm on Metal; from m = 8 with B not transposed, SMALL beats SKINNY. `B^T` (`Shape.ofTransposedB`) stores the weights n × k, as a linear layer stores [out][in] weights.

| m | n × k | Layout | Metal SKINNY µs | % stream | MoltenVK SKINNY µs | % stream | Metal: fastest other |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | 4096 × 4096 | B^T | 89.0 | 74% | 90.9 | 75% | SMALL 298.1 |
| 1 | 4096 × 4096 | B | 113.4 | 58% | 117.9 | 58% | SMALL 299.4 |
| 2 | 4096 × 4096 | B^T | 96.8 | 68% | 100.2 | 68% | SMALL 297.5 |
| 2 | 4096 × 4096 | B | 163.0 | 41% | 152.7 | 45% | SMALL 297.2 |
| 4 | 4096 × 4096 | B^T | 124.0 | 53% | 128.1 | 53% | SMALL 298.9 |
| 4 | 4096 × 4096 | B | 247.0 | 27% | 231.8 | 29% | SMALL 301.2 |
| 8 | 4096 × 4096 | B^T | 193.8 | 34% | 205.6 | 33% | SMALL 300.0 |
| 8 | 4096 × 4096 | B | 443.5 | 15% | 449.2 | 15% | SMALL 305.9 |
| 16 | 4096 × 4096 | B^T | 314.5 | 21% | 325.4 | 21% | SMALL 312.6 |
| 16 | 4096 × 4096 | B | 841.4 | 8% | 824.4 | 8% | SMALL 312.7 |
| 1 | 11008 × 4096 | B^T | 255.4 | 70% | 257.5 | 71% | SMALL 700.3 |
| 1 | 11008 × 4096 | B | 315.6 | 56% | 320.1 | 57% | SMALL 684.4 |
| 2 | 11008 × 4096 | B^T | 255.9 | 69% | 262.5 | 70% | SMALL 700.0 |
| 2 | 11008 × 4096 | B | 407.9 | 44% | 407.6 | 45% | SMALL 679.9 |
| 4 | 11008 × 4096 | B^T | 306.7 | 58% | 324.0 | 56% | SMALL 698.6 |
| 4 | 11008 × 4096 | B | 580.7 | 31% | 580.6 | 31% | SMALL 684.0 |
| 8 | 11008 × 4096 | B^T | 435.1 | 41% | 453.8 | 40% | SMALL 707.4 |
| 8 | 11008 × 4096 | B | 990.0 | 18% | 999.9 | 18% | SMALL 696.4 |
| 16 | 11008 × 4096 | B^T | 689.6 | 26% | 722.6 | 25% | SMALL 735.8 |
| 16 | 11008 × 4096 | B | 1,934 | 9% | 1,960 | 9% | SMALL 719.4 |
| 1 | 4096 × 11008 | B^T | 228.9 | 78% | 231.0 | 79% | SMALL 874.4 |
| 1 | 4096 × 11008 | B | 341.5 | 52% | 347.6 | 53% | SMALL 804.1 |
| 2 | 4096 × 11008 | B^T | 238.7 | 74% | 246.7 | 74% | SMALL 868.2 |
| 2 | 4096 × 11008 | B | 470.6 | 38% | 466.2 | 39% | SMALL 801.5 |
| 4 | 4096 × 11008 | B^T | 328.5 | 54% | 338.2 | 54% | SMALL 865.1 |
| 4 | 4096 × 11008 | B | 720.6 | 25% | 710.5 | 26% | SMALL 809.3 |
| 8 | 4096 × 11008 | B^T | 530.2 | 34% | 546.2 | 33% | SMALL 862.0 |
| 8 | 4096 × 11008 | B | 1,276 | 14% | 1,280 | 14% | SMALL 817.2 |
| 16 | 4096 × 11008 | B^T | 880.3 | 20% | 878.0 | 21% | SMALL 925.7 |
| 16 | 4096 × 11008 | B | 2,347 | 8% | 2,394 | 8% | SMALL 846.8 |

### Quantized GEMM decode (µs per product; speed-up over FP16 SKINNY in brackets)

**Metal**

| m | n × k | FP16 | Q8 | Q4 | Q4M | Q4M/128 | Q4K | Q5K | Q6K |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 × 4096 | 89.8 | 42.9 (2.09×) | 28.9 (3.11×) | 30.2 (2.97×) | 37.8 (2.37×) | 30.3 (2.96×) | 37.7 (2.38×) | 41.0 (2.19×) |
| 4 | 4096 × 4096 | 120.1 | 72.5 (1.66×) | 62.4 (1.93×) | 67.3 (1.78×) | 83.8 (1.43×) | 72.4 (1.66×) | 75.3 (1.59×) | 96.0 (1.25×) |
| 16 | 4096 × 4096 | 329.1 | 330.4 (1.00×) | 302.1 (1.09×) | 381.5 (0.86×) | 344.5 (0.96×) | 413.7 (0.80×) | 410.0 (0.80×) | 584.3 (0.56×) |
| 64 | 4096 × 4096 | – | 527.4 | 524.5 | 577.5 | 584.1 | 576.3 | 581.6 | 548.8 |
| 512 | 4096 × 4096 | – | 3,555 | 3,559 | 4,037 | 3,711 | 4,032 | 4,117 | 3,788 |
| 1 | 11008 × 4096 | 252.7 | 105.6 (2.39×) | 60.5 (4.18×) | 66.5 (3.80×) | 76.6 (3.30×) | 67.8 (3.73×) | 84.8 (2.98×) | 91.9 (2.75×) |
| 4 | 11008 × 4096 | 302.4 | 152.0 (1.99×) | 132.1 (2.29×) | 141.2 (2.14×) | 159.2 (1.90×) | 153.3 (1.97×) | 164.2 (1.84×) | 216.3 (1.40×) |
| 16 | 11008 × 4096 | 705.9 | 792.1 (0.89×) | 715.7 (0.99×) | 977.6 (0.72×) | 790.8 (0.89×) | 1,053 (0.67×) | 1,045 (0.68×) | 1,473 (0.48×) |
| 64 | 11008 × 4096 | – | 1,337 | 1,325 | 1,525 | 1,404 | 1,521 | 1,541 | 1,407 |
| 512 | 11008 × 4096 | – | 9,596 | 9,580 | 10,968 | 10,018 | 10,946 | 11,169 | 10,195 |
| 1 | 4096 × 11008 | 230.5 | 111.8 (2.06×) | 68.9 (3.34×) | 72.6 (3.17×) | 88.0 (2.62×) | 74.3 (3.10×) | 94.2 (2.45×) | 100.4 (2.29×) |
| 4 | 4096 × 11008 | 328.1 | 172.7 (1.90×) | 140.9 (2.33×) | 154.1 (2.13×) | 194.0 (1.69×) | 166.0 (1.98×) | 177.9 (1.84×) | 232.3 (1.41×) |
| 16 | 4096 × 11008 | 874.6 | 832.9 (1.05×) | 752.8 (1.16×) | 997.1 (0.88×) | 915.5 (0.96×) | 1,075 (0.81×) | 1,068 (0.82×) | 1,486 (0.59×) |
| 64 | 4096 × 11008 | – | 1,379 | 1,381 | 1,539 | 1,551 | 1,539 | 1,551 | 1,455 |
| 512 | 4096 × 11008 | – | 9,560 | 9,540 | 10,866 | 9,962 | 10,808 | 11,090 | 10,188 |

**MoltenVK**

| m | n × k | FP16 | Q8 | Q4 | Q4M | Q4M/128 | Q4K | Q5K | Q6K |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 × 4096 | 95.2 | 47.3 (2.01×) | 34.2 (2.79×) | 40.6 (2.35×) | 43.5 (2.19×) | 44.0 (2.16×) | 50.0 (1.90×) | 47.7 (2.00×) |
| 4 | 4096 × 4096 | 129.0 | 90.2 (1.43×) | 79.5 (1.62×) | 117.9 (1.09×) | 127.2 (1.01×) | 124.4 (1.04×) | 126.9 (1.02×) | 110.0 (1.17×) |
| 16 | 4096 × 4096 | 332.4 | 373.7 (0.89×) | 356.4 (0.93×) | 633.2 (0.52×) | 549.5 (0.60×) | 651.3 (0.51×) | 666.3 (0.50×) | 707.4 (0.47×) |
| 64 | 4096 × 4096 | – | 569.9 | 571.3 | 986.1 | 876.6 | 990.6 | 997.7 | 637.1 |
| 512 | 4096 × 4096 | – | 3,281 | 3,286 | 5,680 | 5,032 | 5,723 | 5,733 | 4,000 |
| 1 | 11008 × 4096 | 254.8 | 110.3 (2.31×) | 67.4 (3.78×) | 81.2 (3.14×) | 86.1 (2.96×) | 90.6 (2.81×) | 106.4 (2.39×) | 101.9 (2.50×) |
| 4 | 11008 × 4096 | 316.2 | 182.6 (1.73×) | 162.7 (1.94×) | 256.8 (1.23×) | 261.9 (1.21×) | 273.8 (1.15×) | 281.8 (1.12×) | 243.2 (1.30×) |
| 16 | 11008 × 4096 | 711.8 | 855.1 (0.83×) | 837.9 (0.85×) | 1,574 (0.45×) | 1,297 (0.55×) | 1,648 (0.43×) | 1,674 (0.43×) | 1,786 (0.40×) |
| 64 | 11008 × 4096 | – | 1,311 | 1,315 | 2,269 | 2,020 | 2,286 | 2,293 | 1,514 |
| 512 | 11008 × 4096 | – | 8,681 | 8,703 | 14,951 | 13,297 | 14,996 | 15,082 | 10,690 |
| 1 | 4096 × 11008 | 235.5 | 113.3 (2.08×) | 72.3 (3.26×) | 84.8 (2.78×) | 97.3 (2.42×) | 98.6 (2.39×) | 115.0 (2.05×) | 110.1 (2.14×) |
| 4 | 4096 × 11008 | 333.1 | 191.2 (1.74×) | 164.8 (2.02×) | 264.9 (1.26×) | 293.3 (1.14×) | 280.2 (1.19×) | 291.3 (1.14×) | 253.5 (1.31×) |
| 16 | 4096 × 11008 | 895.8 | 881.6 (1.02×) | 836.6 (1.07×) | 1,603 (0.56×) | 1,425 (0.63×) | 1,647 (0.54×) | 1,681 (0.53×) | 1,713 (0.52×) |
| 64 | 4096 × 11008 | – | 1,478 | 1,465 | 2,634 | 2,345 | 2,650 | 2,662 | 1,702 |
| 512 | 4096 × 11008 | – | 8,868 | 8,865 | 15,324 | 13,570 | 15,435 | 15,469 | 10,670 |

### Attention prefill, FP16 Q/K/V

Composed = GEMM, softmax, GEMM with the scores in memory (Metal uses SIMDGROUP_MATRIX GEMMs, MoltenVK TILED). Fused = the flash kernels.

| Operation | Shape | Metal ms | MoltenVK ms | Metal GFLOP/s dense-equiv. | MoltenVK GFLOP/s dense-equiv. |
|---|---|---:|---:|---:|---:|
| composed (SIMDGROUP_MATRIX GEMM + softmax) | 32 heads × 512 × 128, causal | 0.68 | – | 6,277 | – |
| composed (SIMDGROUP_MATRIX GEMM + softmax) | 32 heads × 1024 × 128, causal | 2.23 | – | 7,709 | – |
| composed (SIMDGROUP_MATRIX GEMM + softmax) | 32 heads × 2048 × 128, causal | 8.48 | – | 8,108 | – |
| fused (flash) | 32 heads × 512 × 128, causal | 0.42 | 0.64 | 10,223 | 6,659 |
| fused (flash) | 32 heads × 1024 × 128, causal | 1.39 | 2.43 | 12,330 | 7,064 |
| fused (flash) | 32 heads × 2048 × 128, causal | 5.15 | 9.47 | 13,338 | 7,260 |
| composed (TILED GEMM + softmax) | 32 heads × 512 × 128, causal | – | 1.21 | – | 3,543 |
| composed (TILED GEMM + softmax) | 32 heads × 1024 × 128, causal | – | 4.31 | – | 3,987 |
| composed (TILED GEMM + softmax) | 32 heads × 2048 × 128, causal | – | 16.38 | – | 4,195 |

### Attention training (deterministic backward, no atomics)

Metal runs the SIMD-group matrix backward (1.12.1); MoltenVK runs the portable kernels.

| Operation | Shape | Metal ms | MoltenVK ms | Metal GFLOP/s dense-equiv. | MoltenVK GFLOP/s dense-equiv. |
|---|---|---:|---:|---:|---:|
| forward with LSE | 32 heads × 512 × 128, causal | 0.42 | 0.65 | – | – |
| forward with LSE | 32 heads × 1024 × 128, causal | 1.39 | 2.43 | – | – |
| forward with LSE | 32 heads × 2048 × 128, causal | 5.15 | 9.44 | – | – |
| backward (dQ, dK, dV) | 32 heads × 512 × 128, causal | 2.86 | 12.68 | 3,753 | 847 |
| backward (dQ, dK, dV) | 32 heads × 1024 × 128, causal | 10.16 | 44.39 | 4,228 | 967 |
| backward (dQ, dK, dV) | 32 heads × 2048 × 128, causal | 38.04 | 154.97 | 4,517 | 1,109 |

### Attention decode (one sequence, FP16 KV)

MoltenVK has no direct short-decode kernel, so 512 keys runs the split decode there.

| Operation | Shape | Metal µs | MoltenVK µs | Metal GB/s of K and V | MoltenVK GB/s of K and V |
|---|---|---:|---:|---:|---:|
| fused decode | 32/8 heads, 512 keys, dim 128 | 9.9 | 40.0 | 212 | 52 |
| fused decode | 32/8 heads, 4096 keys, dim 128 | 46.6 | 62.0 | 360 | 271 |
| fused decode | 32/8 heads, 32768 keys, dim 128 | 332.6 | 349.4 | 403 | 384 |

### Four paged decodes (four layers, 512 keys) in one submission

| Operation | Shape | Metal GPU µs | MoltenVK GPU µs | Metal host encode µs | MoltenVK host encode µs | Metal wall µs | MoltenVK wall µs |
|---|---|---:|---:|---:|---:|---:|---:|
| four paged decodes (4 layers) | 512 keys | 60.9 | 182.8 | 16.8 | 20.5 | 84.0 | 235.8 |

### Batched paged decode (`encodeBatch`) against one encode per sequence

| Operation | Shape | Metal GPU µs | MoltenVK GPU µs | Metal host µs | MoltenVK host µs | Metal GB/s | MoltenVK GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| encodeBatch | 8 sequences × 4096 keys | 423.7 | 439.2 | 5.3 | 5.6 | 317 | 306 |
| encodeBatch | 32 sequences × 1024 keys | 418.6 | 428.4 | 5.2 | 5.7 | 321 | 313 |
| encodeBatch | 32 sequences × 4096 keys | 1,547 | 1,575 | 6.6 | 6.7 | 347 | 341 |
| encodeBatch | 64 sequences × 512 keys | 372.7 | 383.8 | 5.9 | 6.0 | 360 | 350 |
| one encode per sequence | 8 sequences × 4096 keys | 579.0 | 696.1 | 48.6 | 92.4 | – | – |
| one encode per sequence | 32 sequences × 1024 keys | 1,263 | 1,612 | 176.9 | 182.4 | – | – |
| one encode per sequence | 32 sequences × 4096 keys | 2,326 | 2,748 | 183.6 | 385.2 | – | – |
| one encode per sequence | 64 sequences × 512 keys | 1,416 | 3,103 | 293.9 | 378.7 | – | – |

### Masked prefill (new in 1.14.1)

Median of four paired rounds. The composed path is how padded attention ran in 1.14.0.

| Operation | Shape | Metal GPU µs | MoltenVK GPU µs | Metal wall µs | MoltenVK wall µs |
|---|---|---:|---:|---:|---:|
| fused (1.14.1) | 16 heads (no GQA) × 512 × 128, causal, lengths 512/384/128/0 per 4 heads | 235.8 | 275.0 | 259.0 | 314.1 |
| composed GEMM + valid-length softmax (1.14.0 path) | 16 heads (no GQA) × 512 × 128, causal, lengths 512/384/128/0 per 4 heads | 354.3 | 649.1 | 375.8 | 680.7 |

### Paged decode: reused pages against pages rotating through every layer

Median of four rounds, per layer call. 32 query / 8 KV heads, dim 128, 64-token pages, FP16 cache.

| Operation | Shape | Metal GPU µs per layer call | MoltenVK GPU µs per layer call |
|---|---|---:|---:|
| reused pages | 4096 keys, batch 1, 16 MiB KV | 68.3 | 76.2 |
| reused pages | 4096 keys, batch 2, 32 MiB KV | 121.2 | 139.1 |
| reused pages | 32768 keys, batch 1, 128 MiB KV | 379.3 | 406.5 |
| reused pages | 32768 keys, batch 2, 256 MiB KV | 807.9 | 819.6 |
| rotating pages | 4096 keys, batch 1, 320 MiB KV | 72.5 | 88.1 |
| rotating pages | 4096 keys, batch 2, 320 MiB KV | 131.8 | 145.6 |
| rotating pages | 32768 keys, batch 1, 384 MiB KV | 378.8 | 408.1 |
| rotating pages | 32768 keys, batch 2, 512 MiB KV | 805.8 | 820.8 |

### Inference step kernels (Llama-like sizes)

| Operation | Shape | Metal µs | MoltenVK µs |
|---|---|---:|---:|
| residual add + RMSNorm | 1 × 4096 | 8.3 | 14.8 |
| residual add + RMSNorm | 512 × 4096 | 32.5 | 46.5 |
| RoPE | 1 tokens × 32 × 128 | 14.0 | 20.3 |
| RoPE | 512 tokens × 32 × 128 | 32.9 | 33.8 |
| embedding row lookup | vocab 128256, dim 4096 | 16.8 | 23.1 |
| log-sum-exp | vocab 128256 | 9.7 | 20.2 |
| sample: greedy | vocab 128256 | 11.9 | 20.9 |
| sample: temperature | vocab 128256 | 17.0 | 29.1 |
| sample: top-k 40 + top-p | vocab 128256 | 44.2 | 71.0 |

### Sampling by logit distribution (temperature 0.8, top-k 40, top-p 0.95)

| Operation | Shape | Metal µs | MoltenVK µs |
|---|---|---:|---:|
| top-k 40 + top-p 0.95 | vocab 128256, tied logits | 44.3 | 64.1 |
| top-k 40 + top-p 0.95 | vocab 128256, varied logits | 41.6 | 63.4 |
| top-k 40 + top-p 0.95 | vocab 128256, skewed logits | 41.2 | 63.9 |
| top-k 40 + top-p 0.95 | vocab 128256, concentrated logits | 56.2 | 76.3 |
| top-k 40 + top-p 0.95 | vocab 128256, weak-bound logits | 41.8 | 63.2 |
| top-k 40 + top-p 0.95 | vocab 128256, ascending logits | 41.5 | 63.1 |

### Training and kit kernels (FP32 unless stated)

Best GPU time of four runs per backend; spread is how much slower the slowest run was. GB/s is the minimum traffic (each input read once, each output written once) over the best time. Rates above the ~500 GB/s streaming read are not DRAM bandwidth: the causal softmax probably skips its masked half, the constant-mask select takes 21% less time than the alternating one (so it reads less), and working sets of 16–34 MB (norms, RoPE, gather) can stay in the system-level cache between repetitions.

| Kernel | Shape | Metal µs | Metal GB/s | Metal spread | MoltenVK µs | MoltenVK GB/s | MoltenVK spread | Host encode µs (Metal) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Elementwise: unary COPY | 16M fp32 | 344.7 | 389 | +4% | 343.7 | 390 | +1% | 17.6 |
| Elementwise: unary RELU | 16M fp32 | 346.0 | 388 | +4% | 345.8 | 388 | +0% | 6.7 |
| Elementwise: unary EXP | 16M fp32 | 348.4 | 385 | +4% | 347.3 | 386 | +1% | 5.6 |
| Elementwise: unary SILU | 16M fp32 | 372.4 | 360 | +2% | 379.1 | 354 | +1% | 5.3 |
| Elementwise: unary GELU | 16M fp32 | 379.4 | 354 | +3% | 394.0 | 341 | +1% | 4.2 |
| Elementwise: unary SIGMOID | 16M fp32 | 358.1 | 375 | +3% | 358.9 | 374 | +1% | 4.2 |
| Elementwise: unary TANH | 16M fp32 | 395.9 | 339 | +3% | 401.6 | 334 | +1% | 4.3 |
| Elementwise: binary ADD | 16M fp32 | 434.0 | 464 | +5% | 418.2 | 481 | +1% | 6.7 |
| Elementwise: binary MUL | 16M fp32 | 433.1 | 465 | +5% | 414.5 | 486 | +2% | 6.8 |
| Elementwise: binary SWIGLU | 16M fp32 | 439.6 | 458 | +4% | 419.4 | 480 | +2% | 4.2 |
| Elementwise: binary SILU_BACKWARD | 16M fp32 | 438.0 | 460 | +5% | 420.9 | 478 | +1% | 3.6 |
| Elementwise: binary GELU_BACKWARD | 16M fp32 | 438.7 | 459 | +4% | 443.5 | 454 | +1% | 3.4 |
| Elementwise: binary TANH_BACKWARD | 16M fp32 | 439.7 | 458 | +3% | 420.3 | 479 | +1% | 3.2 |
| Reduce: rows SUM | 4096x4096 fp32 | 139.8 | 480 | +4% | 141.2 | 475 | +3% | 2.3 |
| Reduce: rows MAX | 4096x4096 fp32 | 139.1 | 482 | +3% | 143.8 | 467 | +2% | 2.0 |
| Reduce: axis 0 MEAN | 4096x4096 fp32 | 382.2 | 176 | +4% | 385.3 | 174 | +1% | 2.7 |
| Reduce: logSumExp rows | 4096x4096 fp32 | 166.3 | 404 | +3% | 195.9 | 343 | +3% | 3.9 |
| Reduce: argmaxAxis 1 | 4096x4096 fp32 | 140.3 | 478 | +4% | 144.5 | 464 | +2% | 1.9 |
| Reduce: maxBackward axis 1 | 4096x4096 fp32 | 357.4 | 188 | +4% | 362.7 | 185 | +0% | 3.6 |
| Scan: rows SUM | 4096x4096 fp32 | 793.7 | 169 | +3% | 1,171 | 115 | +1% | 2.8 |
| Softmax: forward causal | 32x512x512 | 162.6 | 413 | +1% | 205.8 | 326 | +1% | 0.6 |
| Softmax: backward causal | 32x512x512 | 146.5 | 687 | +4% | 156.7 | 643 | +2% | 0.8 |
| Softmax: forward | 512x32000 | 577.3 | 227 | +3% | 576.8 | 227 | +2% | 0.8 |
| CrossEntropy: loss | 512x32000 | 193.5 | 339 | +3% | 224.0 | 293 | +1% | 8.8 |
| CrossEntropy: loss + gradient | 512x32000 | 492.0 | 266 | +3% | 517.6 | 253 | +1% | 7.8 |
| Norms: RMS forward | 512x4096 fp32 | 29.9 | 562 | +4% | 40.7 | 412 | +3% | 1.9 |
| Norms: RMS forward + residual | 512x4096 fp32 | 44.6 | 753 | +3% | 55.0 | 610 | +3% | 1.0 |
| Norms: LayerNorm forward | 512x4096 fp32 | 39.2 | 428 | +4% | 55.9 | 300 | +2% | 1.1 |
| Norms: RMS backward | 512x4096 fp32 | 208.2 | 121 | +3% | 285.9 | 88 | +3% | 9.7 |
| Norms: LayerNorm backward | 512x4096 fp32 | 259.9 | 97 | +4% | 356.1 | 71 | +3% | 9.1 |
| Rope: forward | 512x32x128 fp32 | 31.8 | 528 | +9% | 39.5 | 425 | +8% | 1.8 |
| Rope: backward | 512x32x128 fp32 | 33.7 | 498 | +2% | 39.9 | 421 | +1% | 1.6 |
| Gather: gather rows | 512 of 8192x4096 | 105.9 | 158 | +4% | 110.4 | 152 | +2% | 2.9 |
| Gather: scatterAdd accumulate | 512 into 8192x4096 | 72.5 | 347 | +4% | 96.3 | 261 | +6% | 4.2 |
| Gather: scatterAdd overwrite | 512 into 8192x4096 | 470.4 | 303 | +3% | 502.2 | 284 | +4% | 4.5 |
| Optimizer: AdamW fp32 grads | 16M params | 991.4 | 474 | +4% | 984.6 | 477 | +5% | 4.3 |
| Optimizer: AdamW fp16 grads + fp16 shadow | 16M params | 984.9 | 477 | +4% | 979.3 | 480 | +5% | 4.3 |
| Optimizer: SGD momentum | 16M params | 703.2 | 477 | +5% | 696.6 | 482 | +5% | 2.9 |
| GradientClip: squaredNorm | 16M fp32 | 319.6 | 210 | +2% | 336.3 | 200 | +2% | 3.6 |
| GradientClip: clip (scale in place) | 16M fp32 | 324.7 | 413 | +4% | 325.3 | 413 | +3% | 2.2 |
| GradientClip: squaredNorm, 64 tensors | 16M fp32 | 712.1 | 94 | +2% | 953.0 | 70 | +10% | 81.2 |
| Logic: compare LESS fp32 | 16M | 438.0 | 460 | +3% | 417.2 | 483 | +3% | 4.0 |
| Logic: select fp32, constant mask | 16M | 442.1 | 607 | +6% | 416.6 | 644 | +1% | 3.9 |
| Logic: select fp32, alternating mask | 16M | 561.6 | 478 | +13% | 550.1 | 488 | +4% | 3.8 |
| Cast: FP32 to FP16 | 16M | 362.5 | 278 | +3% | 361.3 | 279 | +1% | 3.5 |
| Cast: FP16 to FP32 | 16M | 388.2 | 259 | +3% | 386.8 | 260 | +3% | 2.5 |
| Cast: FP32 to BF16 | 16M | 333.3 | 302 | +4% | 331.9 | 303 | +2% | 2.4 |
| NanCheck: check | 16M fp32 | 317.2 | 212 | +2% | 319.5 | 210 | +2% | 3.9 |

### Runtime overhead (third, warm run)

| Measurement | Metal | MoltenVK |
|---|---:|---:|
| 50 dispatches, one submission, record | 17 µs | 17 µs |
| 50 dispatches, one submission, submit | 52 µs | 178 µs |
| 50 dispatches, one submission, submit-to-done | 287 µs | 544 µs |
| 50 dispatches, one submission, GPU | 193 µs | 411 µs |
| encode + submit + wait, 50 dispatches | 244 µs | 759 µs |
| replay recording + wait, 50 dispatches | 216 µs | 661 µs |
| replay submit only | 53 µs | 125 µs |
| allocate + close (pooled) | 299 ns | 336 ns |
| 4 KiB upload + download round trip | 131 µs | 192 µs |
| transfer overlap: serial | 15.4 ms | 17.5 ms |
| transfer overlap: two queues | 16.4 ms | 17.2 ms |

### Memory baseline

| Measurement | Metal | MoltenVK |
|---|---:|---:|
| streaming read (row sums, 256 MiB) | 507 GB/s | 494 GB/s |
| buffer copy, read + write | 432 GB/s | 431 GB/s |

## Reproduce

Run each backend separately, from the repository root:

```sh
props="-Dgemm.benchmark=true -Dgemm.decode=true -Dgemm.shapes=true -Dinference.benchmark=true -Dquant.decode=true \
  -DattentionMask.benchmark=true -DpagedDecode.benchmark=true -Dcompute.benchmark=true -Dkit.benchmark=true \
  -Dsurefire.failIfNoSpecifiedTests=false"

mvn -o -pl dynamis-gpu-metal,dynamis-gpu-primitives -am -Pmetal-native,metal-hardware test $props \
  '-Dtest=MetalGemmF16Test#benchmark+decodeBandwidth+shapeSweep,MetalAttentionTest#benchmark*,MetalInferenceTest#*enchmark,MetalQuantizedTest#decodeBandwidth,MetalComputeTest#*Benchmark,MetalKitBenchmarkTest'

sdk="$HOME/VulkanSDK/1.4.341.0/macOS"
VK_DRIVER_FILES="$sdk/share/vulkan/icd.d/MoltenVK_icd.json" \
  mvn -o -pl dynamis-gpu-vulkan,dynamis-gpu-primitives -am -Pvulkan-hardware test $props \
  "-Ddynamisgpu.vk.loader=$sdk/lib/libvulkan.1.dylib" \
  '-Dtest=VulkanGemmF16Test#benchmark+decodeBandwidth+shapeSweep,VulkanAttentionTest#benchmark*,VulkanInferenceTest#*enchmark,VulkanQuantizedTest#decodeBandwidth,VulkanComputeTest#*Benchmark,VulkanKitBenchmarkTest'
```

In zsh, write `${=props}` so the properties split into separate arguments. The run takes about five minutes per
backend. Leave the machine idle, and allow a cool-down between the GEMM suites and the kit harness.
