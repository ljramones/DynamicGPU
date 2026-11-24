# Throughput primitives

`dynamis-gpu-primitives` (module `org.dynamisengine.gpu.primitives`) holds throughput kernels on the compute API,
written in Slang and built with `dynamis-gpu-shaders`. Their contract is an accuracy bound, not bit-exactness. The
exact GPU twins of Vectrix's kernels are vectrix-gpu.

| Primitive | Does | Since |
|---|---|---|
| `GemmF16` | half-precision matrix multiply: transposed B, batches, FP32 or FP16 output | 1.2.0 (forms 1.4.0) |
| `Softmax` | row-wise softmax of FP32 scores, causal mask, FP32 or FP16 output | 1.4.0 |
| `AttentionF16` | scaled dot-product attention for every head: `softmax(scale Q K^T + mask) V` | 1.4.0 |
| `Elementwise` | unary and binary maps over strided FP32 views, with broadcasting | 1.7.0 |
| `Reduce` | fixed-order sum, max, min and argmax along the last dimension | 1.7.0 |
| `Scan` | fixed-order inclusive sum, max and min scans along the last dimension | 1.7.0 |
| `NanCheck` | reports NaNs and infinities through a `KernelErrors` buffer; tripwires | 1.7.0 |

Every primitive since 1.7.0 publishes its accuracy contract as data (`contract(...)`, a kit `AccuracyContract`):
- **What it says:** the input types, the accumulation type, the output rounding, the order class and the bound.
  - **Order class:** `FIXED` gives the same bits on every device; `DEVICE` means the result is bounded.
  - **Bound:** exact against a named CPU reference, within ULPs, or a formula.
- **Measured:** the tests check each primitive against its contract on native Metal, MoltenVK and lavapipe.

Output types are `Precision.FP32` and `Precision.FP16`. FP16 output is the FP32 result rounded once to the nearest
half.

## `GemmF16`

`C = alpha * A * B + beta * C`:
- **Layout:**
  - A is `m x k` in FP16.
  - B is `k x n`, or `n x k` when transposed, in FP16.
  - C is `m x n`, in FP32 or FP16.
  - All are row-major with leading dimensions in elements (`GemmF16.Shape`).
- **Use:** it is a dense layer with half-precision weights and activations.
  - **Linear layers:** with transposed B it computes a linear layer's `x W^T`, with the weights as stored.
  - **Attention:** with a batch, it runs the per-head products of attention in one dispatch.

```java
try (GemmF16 gemm = GemmF16.create(device)) {              // FP32 C; create(device, Output.FP16) for FP16 C
    var shape = GemmF16.Shape.of(m, n, k, 1f, 0f);          // dense: lda = k, ldb = ldc = n
    gemm.encode(encoder, a, b, c, shape);                    // a, b: >= shape.aBytes(), bBytes(); c: cBytes(output)
}
var linear = GemmF16.Shape.ofTransposedB(tokens, out, in, 1f, 0f);       // C = X W^T, W stored out x in
var heads = GemmF16.Shape.ofTransposedB(seq, seq, headDim, scale, 0f).batched(numHeads);   // one Q K^T per head
var shared = GemmF16.Shape.of(m, n, k, 1f, 0f).withBatch(count, m * k, 0, m * n);         // stride 0: one B for all
```

- **Transposed B:** `Shape.ofTransposedB`, or `transposeB = true` with `ldb >= k`.
- **Batches:** `batched(count)` packs the matrices densely one after another; `withBatch(count, strideA, strideB,
  strideC)` sets the strides.
  - **Sharing:** a stride of 0 shares one matrix across the batch.
  - **Overlap:** C matrices may not overlap.
  - **Dispatch:** the batch is the dispatch's z dimension.
- **FP16 output:** `Output.FP16` computes in FP32 as above, then rounds the result once to the nearest half.
  Overflow gives infinity.

- **`beta == 0`:** C is not read (as in BLAS), so it may hold anything.
- **Buffer sizes:** transfers move whole 32-bit words, so round the FP16 buffers up to a multiple of 4 bytes.
- **Refusal:** `create` throws `UNSUPPORTED` on a device without `FP16`.
- **Indexing:** element indices are 32-bit, so each operand, batch included, spans fewer than 2^32 elements.

### Algorithms

`GemmF16.create(device)` picks the fastest algorithm the device has; `create(device, algorithm)` chooses one.

| Algorithm | Runs on | Kernel |
|---|---|---|
| `TILED` | every device with `FP16` | Slang (`gemm_body.slang`, one entry per output type): shared FP32 tiles and fused multiply-adds |
| `SIMDGROUP_MATRIX` | native Metal on Apple7+ GPUs (`SUBGROUP_MATRIX_F16`) | MSL (`metal/gemm_f16_simdgroup.metal`, one entry per output type): 8 x 8 hardware multiply-accumulate |

- **The matrix kernel:** a group of four SIMD-groups computes a 64 x 64 tile, each SIMD-group a 32 x 32 quarter.
  - **Interior tiles** (`k` a multiple of 8) load 8 x 8 blocks straight from device memory; the caches serve the
    reuse. Transposed B loads its blocks transposed, in a separate loop: testing the layout inside the loop cost
    about 15 %.
  - **Edge tiles** stage zero-filled copies in threadgroup memory.
  - **Why the direct loads:** staging every tile in threadgroup memory, a half at a time, ran at 3.7 TFLOP/s.

### Accuracy and reproducibility

- **Contract:** before `alpha` and `beta`, `|C - exact| <= k * 2^-24 * sum |a b|`. Later kernel versions may sum in
  another order, for example on matrix hardware, within the same bound.
- **The tiled algorithm is reproducible across devices:**
  - **Why:** products of halves are exact in FP32, so `fma(a, b, acc)` rounds once whether or not the device fuses
    it.
  - **The sum:** each element is `acc = fl(acc + a b)`, over `k` in order.
  - **The epilogue:** it is `fl(fl(alpha acc) + fl(beta c))`, with the add marked `NoContraction`.
  - **Checking it:** `GemmF16.reference` computes the same bits on the CPU, and the tests require equality on native
    Metal, MoltenVK and Mesa lavapipe. lavapipe doesn't fuse `fma`, so it's a meaningful check.
  - **On the CPU:** NNJava can reproduce a GPU layer bit for bit.
- **The matrix algorithm:** it is held to the bound only.
  - **Measured:** on an M4 Max it matched `GemmF16.reference` bit for bit on every test shape: 0 differences in
    about 62,000 elements, interior and edge tiles. So that GPU's multiply-accumulate sums exact products in order
    with FP32 rounding.
  - **Not promised:** Apple does not document this, and other GPU generations are unverified.

### Performance

Measured on an Apple M4 Max (40-core GPU, macOS 27). The figures are GPU time (`GpuSubmission.gpuTime`) over
back-to-back multiplies in one submission (`-Dgemm.benchmark=true`). MoltenVK runs the tiled kernel.

| Size (m = n = k) | Metal, `SIMDGROUP_MATRIX` | Metal, `TILED` | MoltenVK, `TILED` |
|---|---|---|---|
| 512 | 7.8 TFLOP/s | 3.9 TFLOP/s | 3.0 TFLOP/s |
| 1024 | 11.3 TFLOP/s | 5.0 TFLOP/s | 4.6 TFLOP/s |
| 2048 | 12.8 TFLOP/s | 5.6 TFLOP/s | 5.4 TFLOP/s |
| 4096 | 12.9 TFLOP/s | 5.7 TFLOP/s | 5.4 TFLOP/s |

- **The kernel:** a group of 16 x 16 threads computes a 64 x 64 tile, each thread 4 x 4 outputs. k advances in steps
  of 16 through shared FP32 tiles.
- **Alternatives measured on the same GPU:**
  - 128 x 128 tiles with 8 x 8 per thread: 3.6 TFLOP/s;
  - steps of 32: 4.2 TFLOP/s.
- **Unrolling:** the fixed loops are `[ForceUnroll]`ed. The shader build compiles at `-O0`, and without unrolling the
  SPIR-V keeps the accumulators as indexed arrays. MoltenVK then ran 14 times slower: 376 against 5,261 GFLOP/s.
- **Peak:** the tiled kernel reaches about a third of the GPU's FP32 peak, and the matrix kernel about three
  quarters.

## `Softmax`

`y[j] = exp(s[j] - max) / sum exp(s[j'] - max)` over each row of an `m x n` matrix of FP32 scores. Masked columns
are 0.

```java
try (Softmax softmax = Softmax.create(device, Precision.FP16)) {      // FP16 output feeds GemmF16 directly
    softmax.encode(encoder, scores, probs, Softmax.Shape.of(rows, cols).causal(cols - rows).batched(heads));
}
```

- **Causal mask:** row `r` sees columns `0 .. r + causalOffset`. With a cache of `n` keys and `m` new queries the
  offset is `n - m`.
- **Batches:** the batch is the dispatch's z dimension.
- **The kernel:** one group of 256 threads per row. The max and the sum are combined by a fixed tree, so a device's
  result doesn't depend on scheduling. The row's max and sum live in their own group variables, because Slang's
  SPIR-V target may reread a local copy of a groupshared value after a barrier.
- **Accuracy:** `exp` is the device's own, so results are bounded, not bit-identical across devices. For an unmasked
  element, with `x = s - max` and `xbar` the softmax-weighted mean of `|x|`:
  `|y - exact| <= (n + 10 + 2|x| + 2 xbar) 2^-24 exact + 2^-126`, plus one rounding to half for FP16. The `2^-126`
  allows for Apple GPUs flushing subnormal results to zero.
- **Measured:** the tests check every element on native Metal, MoltenVK and lavapipe. FP32 output stays within a
  quarter of the bound; FP16 output is dominated by its final rounding.

## `AttentionF16`

`O = softmax(scale Q K^T + mask) V` for every head.
- **Layout:** Q is `[heads][queries][headDim]`, K and V `[heads][keys][headDim]`, all FP16, densely packed. O has the
  layout of Q, in FP16 or FP32.

```java
try (AttentionF16 attention = AttentionF16.create(device, Precision.FP16)) {
    var shape = AttentionF16.Shape.of(heads, queries, keys, headDim).asCausal();   // scale = 1/sqrt(headDim)
    attention.encode(encoder, q, k, v, o, scores, probs, shape);   // scores: shape.scoresBytes(); probs: probsBytes()
}
```

- **The steps:** it records three dispatches.
  1. `S = scale Q K^T`: `GemmF16` with transposed B, FP32.
  2. `P = softmax(S)`: FP16, causal if asked.
  3. `O = P V`.
- **Algorithm:** both products use the fastest `GemmF16` algorithm the device has.
- **Memory:** the scratch buffers hold `heads * queries * keys` scores and probabilities, so memory grows with the
  square of the sequence. A fused (flash) kernel is planned.
- **Causal:** query `i` sees keys `0 .. i + keys - queries`, so new queries at the end of a KV cache see the whole
  cache. One decode step (`queries = 1`) sees every key.
- **Accuracy:** the error of O is the composition of the steps' bounds.
  - The scores' `GemmF16` error perturbs P by at most `2 P max|dS|`.
  - `Softmax` adds its bound, and P adds one rounding to half.
  - The value product adds `(keys + 2) 2^-24 sum P |V|`, then the rounding of O.
  - The tests check every element of O against that composition, computed in double precision from the same half
    inputs: on native Metal, MoltenVK and lavapipe, for causal and unmasked shapes and both outputs. The worst
    measured error is 37 % of the bound.
- **Measured** on an Apple M4 Max (native Metal, `SIMDGROUP_MATRIX`), 32 heads, head dimension 128, causal, FP16, by
  GPU time:

  | Sequence | Time | Dense-equivalent rate |
  |---|---|---|
  | 512 | 0.82 ms | 5.3 TFLOP/s |
  | 1024 | 2.1 ms | 8.1 TFLOP/s |
  | 2048 | 7.4 ms | 9.3 TFLOP/s |

  The rate counts both products in full. A causal mask makes half of the scores unnecessary, so a kernel that skips
  them would do better.

## Planned

- **Flash attention:** fused attention that keeps the scores on chip, with memory linear in the sequence, and skips
  masked blocks.
- **GEMM:** a Vulkan cooperative-matrix path, once there is hardware to test it on.
- **Other primitives:** layer and RMS norms, reductions, and int8 and BF16 weight formats.

## Kit primitives: `Elementwise`, `Reduce`, `Scan`, `NanCheck` (1.7.0)

They take kit `StridedView`s of rank up to 4 over FP32 buffers, so dense, sliced, transposed, batched and
interleaved data need no copies. The operation is a specialization constant, so each one compiles to its own folded
pipeline on first use.

```java
try (var ew = Elementwise.create(device); var reduce = Reduce.create(device)) {
    var x = StridedView.matrix(0, rows, cols, ld);                       // a sliced matrix
    ew.binary(e, Elementwise.Binary.ADD, a, x, bias, StridedView.contiguous(cols), y, StridedView.matrix(rows, cols));
    reduce.rows(e, Reduce.Op.MAX, y, StridedView.matrix(rows, cols), m, StridedView.contiguous(rows));
}
```

- **`Elementwise`:**
  - **Unary:** COPY, NEG, ABS, SQUARE, RELU, SQRT, EXP, LOG, RECIPROCAL.
  - **Binary:** ADD, SUB, MUL, DIV, MAX, MIN.
  - **Broadcasting:** inputs broadcast along extent-1 dimensions, aligned at the last dimension as in NumPy.
  - **Accuracy:**
    - **Exact:** the exact operations, plus ADD, SUB, MUL, MAX and MIN, match `Elementwise.apply` bit for bit.
      MAX and MIN follow Java's `Math.max` and `Math.min` (NaN propagates; max(-0, +0) = +0) on every device,
      not the hardware's.
    - **Device functions:** SQRT, EXP, LOG, RECIPROCAL and DIV use the device's own functions, within the Vulkan
      and Metal precision requirements: 5, max(4, 3 + 2|x|), 4 (or 2^-21 absolute near 1), 2.5 and 2.5 ULP.
  - **Measured worst error:**

    | Device | SQRT | DIV | RECIPROCAL | EXP | LOG |
    |---|---|---|---|---|---|
    | M4 Max (Metal and MoltenVK) | 0 | 0 | 0 | 1 | 2 |
    | lavapipe | 0 | 0 | 0 | 7 | 8 (inside [0.5, 2]) |
- **`Reduce`:** sum, max and min along the last dimension, and argmax (the first largest; NaN counts as the largest).
  - **The order:** one group of 256 threads per row. Thread `t` folds columns `t, t + 256, ...`, then the partials
    combine in a fixed tree.
  - **Exactness:** the result is the same bits on every device, and `Reduce.reference` reproduces it on the CPU.
  - **Against the exact sum:** `|err| <= (ceil(n / 256) + 8) u sum |x|`.
- **`Scan`:** inclusive sum, max and min scans along the last dimension.
  - **The order:** blocks of 256 are scanned in a fixed Blelloch tree and combined with the carry of the blocks
    before them.
  - **Exactness:** the same bits on every device, reproduced by `Scan.reference`.
  - **Against the exact prefix sum:** `|err_j| <= (18 + ceil((j + 1) / 256)) u sum_{i <= j} |x_i|`.
- **`NanCheck`:** it reports every NaN and infinity of a view, with code `NanCheck.NONFINITE` and the smallest
  row-major index in the view.
  - **`tripwire`:** it resets the error buffer, checks the view and records `checkErrors`. The submission then fails
    with `KERNEL_ERROR` if the dispatch before it produced a non-finite value, with no readback.
- **The Slang side:** these kernels are written on the kit's Slang library, which ShaderBuild bundles:
  - views (`dg_at`, `dg_coords`, `dg_row`, broadcasting);
  - fixed-tree reductions and scans (`DG_GROUP_REDUCE`, `DG_GROUP_SCAN_EXCLUSIVE`);
  - Java-semantics `dg_max` and `dg_min`;
  - classification by bits (`dg_is_nan`);
  - `dg_report_error`.

  Any kernel built with ShaderBuild gets them through `#include "dynamis/kit.slang"`.

