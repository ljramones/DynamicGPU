# Throughput primitives

`dynamis-gpu-primitives` (module `org.dynamisengine.gpu.primitives`) holds throughput kernels on the compute API,
written in Slang and built with `dynamis-gpu-shaders`. Their contract is an accuracy bound, not bit-exactness. The
exact GPU twins of Vectrix's kernels are vectrix-gpu.

| Primitive | Does | Since |
|---|---|---|
| `GemmF16` | the GEMM family: FP16, BF16 or FP32 inputs, transposed A or B, batches, fused epilogues, skinny decode, small and grouped multiplies; FP32 or FP16 output | 1.2.0 (forms 1.4.0, family 1.8.0) |
| `Softmax` | row-wise softmax of FP32 scores, causal mask, FP32 or FP16 output | 1.4.0 |
| `AttentionF16` | scaled dot-product attention for every head: `softmax(scale Q K^T + mask) V` | 1.4.0 |
| `Elementwise` | unary and binary maps over strided FP32 views, with broadcasting | 1.7.0 |
| `Reduce` | fixed-order sum, max, min and argmax along the last dimension | 1.7.0 |
| `Scan` | fixed-order inclusive sum, max and min scans along the last dimension | 1.7.0 |
| `NanCheck` | reports NaNs and infinities through a `KernelErrors` buffer; tripwires | 1.7.0 |
| `QuantizedMatrix`, `Quantize` | block-quantized weights (Q8, Q4, Q4M): the CPU encoder and decoder, and the GPU's, bit for bit | 1.9.0 |
| `QuantizedGemm` | matrix multiply with quantized weights, dequantized in the kernel, for decode and prefill | 1.9.0 |

Every primitive since 1.7.0 publishes its accuracy contract as data (`contract(...)`, a kit `AccuracyContract`):
- **What it says:** the input types, the accumulation type, the output rounding, the order class and the bound.
  - **Order class:** `FIXED` gives the same bits on every device; `DEVICE` means the result is bounded.
  - **Bound:** exact against a named CPU reference, within ULPs, or a formula.
- **Measured:** the tests check each primitive against its contract on native Metal, MoltenVK and lavapipe.

Output types are `Precision.FP32` and `Precision.FP16`. FP16 output is the FP32 result rounded once to the nearest
half.

## `GemmF16`: the GEMM family

`C = epilogue(alpha * op(A) * op(B) + beta * C)`:
- **Layout:**
  - A is `m x k`, or `k x m` when transposed; B is `k x n`, or `n x k` when transposed.
  - A and B are FP16, BF16 or FP32 (`GemmF16.Input`, 1.8.0); C is `m x n`, in FP32 or FP16 (`Precision`).
  - All are row-major with leading dimensions in elements (`GemmF16.Shape`).
- **Use:**
  - **Linear layers:** with transposed B it computes `x W^T`, with the weights as stored, and a fused bias,
    activation, gate or residual.
  - **Decode:** for `m` up to 16 the skinny kernel streams the weights at close to memory bandwidth.
  - **Attention:** with a batch, it runs the per-head products in one dispatch.
  - **Backward pass:** with transposed A it computes the weight gradient `dY^T X`.
  - **Mixture of experts:** the grouped form runs every expert's multiply in one dispatch.
  - **Small networks:** batches of 16 x 16 or 32 x 32 multiplies, such as game-AI layers, run on the small kernel.

```java
try (GemmF16 gemm = GemmF16.create(device)) {              // FP16 A and B, FP32 C; create(device, Input, Precision)
    var shape = GemmF16.Shape.of(m, n, k, 1f, 0f);          // dense: lda = k, ldb = ldc = n
    gemm.encode(encoder, a, b, c, shape);                    // a, b: >= shape.aBytes(input), bBytes(input); c: cBytes(output)
}
var linear = GemmF16.Shape.ofTransposedB(tokens, out, in, 1f, 0f);       // C = X W^T, W stored out x in
var heads = GemmF16.Shape.ofTransposedB(seq, seq, headDim, scale, 0f).batched(numHeads);   // one Q K^T per head
var shared = GemmF16.Shape.of(m, n, k, 1f, 0f).withBatch(count, m * k, 0, m * n);         // stride 0: one B for all
var gradient = GemmF16.Shape.ofTransposedA(out, in, tokens, 1f, 0f);    // dW = dY^T X, dY stored tokens x out
```

- **Transposed B:** `Shape.ofTransposedB`, or `transposeB = true` with `ldb >= k`.
- **Transposed A** (1.8.0): `Shape.ofTransposedA`, or `transposeA = true` with `lda >= m`. Every algorithm but
  SKINNY takes it.
- **Batches:** `batched(count)` packs the matrices densely one after another; `withBatch(count, strideA, strideB,
  strideC)` sets the strides.
  - **Sharing:** a stride of 0 shares one matrix across the batch.
  - **Overlap:** C matrices may not overlap.
  - **Dispatch:** the batch is the dispatch's z dimension.
- **Column-major operands** (1.8.0): there are no column-major kernels. A column-major product is the row-major
  `C^T = op(B)^T op(A)^T`, which `Shape.columnMajor(m, n, k, lda, ldb, ldc, alpha, beta, transposeA, transposeB)`
  describes. Encode it with the buffers of A and B swapped.
- **FP16 output:** `Precision.FP16` computes in FP32 as above, then rounds the result once to the nearest half.
  Overflow gives infinity.
- **`beta == 0`:** C is not read (as in BLAS), so it may hold anything.
- **Buffer sizes:** transfers move whole 32-bit words, so round 2-byte buffers up to a multiple of 4 bytes.
- **Refusal:** `create` throws `UNSUPPORTED` on a device without `FP16` when the inputs or C are FP16.
- **Indexing:** element indices are 32-bit, so each operand, batch included, spans fewer than 2^32 elements.

### Algorithms

`GemmF16.create(device)` compiles every algorithm the device runs and picks one per shape;
`create(device, algorithm, input, output)` runs one algorithm for every shape.

| Algorithm | Runs on | Kernel | Order |
|---|---|---|---|
| `TILED` | every device | Slang (`gemm_body.slang`, `gemm_tile.slang`): 64 x 64 tiles, shared FP32 tiles and fused multiply-adds | fixed: `reference` |
| `SIMDGROUP_MATRIX` | native Metal on Apple7+ GPUs (`SUBGROUP_MATRIX_F16`); FP16 and FP32 inputs | MSL (`metal/gemm_f16_simdgroup.metal`): 8 x 8 hardware multiply-accumulate | the hardware's |
| `SKINNY` (1.8.0) | every device; `m <= 16`, A not transposed | Slang (`gemm_skinny_body.slang`): streams B once with every row of A | fixed: `referenceSkinny` |
| `SMALL` (1.8.0) | every device | Slang (`gemm_small_body.slang`): 16 x 16 tiles, one element per thread | fixed: `reference` |

- **The matrix kernel:** a group of four SIMD-groups computes a 64 x 64 tile, each SIMD-group a 32 x 32 quarter.
  - **Interior tiles** (`k` a multiple of 8) load 8 x 8 blocks straight from device memory; the caches serve the
    reuse. Transposed operands load their blocks transposed, in separate loops: testing the layout inside the loop
    cost about 15 %.
  - **Edge tiles** stage zero-filled copies in threadgroup memory.
  - **Why the direct loads:** staging every tile in threadgroup memory, a half at a time, ran at 3.7 TFLOP/s.
- **The skinny kernel** (the decode step of an LLM):
  - **B transposed** (the weights of a linear layer): a group of 256 threads computes 16 columns. Each column's k
    elements are split over 32 lanes in chunks of 8, and each thread computes 2 columns, so a value of A serves both.
  - **B not transposed:** a group computes 64 columns, 16 lanes of 16 threads, each thread 4 adjacent columns.
  - **The vector build:** with B transposed, every row on a multiple of 8 elements and the buffers at 16-byte
    offsets, A and B are read 16 bytes per lane. Otherwise they are read as 32-bit words. Both builds sum in the
    same order.
  - **Lanes:** they combine in a fixed tree in group memory, never through subgroup operations, whose width differs
    between devices.
- **The small kernel:** each thread sums its element's k products in order, as TILED does, so it is bit for bit
  `reference`. It wins wherever TILED's 64 x 64 tiles would be mostly padding or too few to fill the GPU.

### Inputs (1.8.0)

`GemmF16.Input`: FP16, BF16 or FP32. Every family has a kernel per input and output type, 30 Slang kernels in all.
- **FP16 and BF16:** they widen exactly to FP32 in the kernel (BF16 by a shift), and their products are exact in
  FP32. The fixed-order algorithms are therefore bit for bit their references on every device.
  - **BF16** is never narrowed to FP16, whose exponent range is smaller.
  - **Subnormals:** devices that flush FP32 subnormals (Apple GPUs) also flush BF16 subnormal inputs.
- **FP32:** products round, so each step is one fused multiply-add.
  - **The references** use `Math.fma`, and are bit for bit where the device reports `FUSED_FMA` (native Metal and
    MoltenVK).
  - **On lavapipe,** which rounds `fma` twice, results are within the bound.
- **References:** `reference(algorithm, Elements a, Elements b, ...)` reads A and B through `GemmF16.Elements`
  (`fp16`, `bf16`, `fp32`). `bf16ToFloat` and `floatToBf16` (ties to even) convert.
- **The matrix kernel** takes FP16 and FP32. FP32 SIMD-group matrices ran at only 2.7 to 3.0 TFLOP/s at 4096^3,
  against the tiled kernel's 4.9 to 5.3, so shape dispatch leaves them out for FP32 inputs.

### Fused epilogues (1.8.0)

`GemmF16.Epilogue` is applied to each element in FP32, in this order, before the one rounding to C's type:
1. `r = fl(fl(alpha acc) + fl(beta c))`, as without an epilogue;
2. `r + bias`, per column (`bias[j]`, a linear layer's bias) or per row (`bias[i]`), shared by every matrix of a
   batch;
3. the activation: ReLU (`Math.max(r, 0)`, so NaN stays NaN), GELU in its tanh form, or SiLU;
4. `r * gate`, the gated linear unit: SiLU with the up projection as gate is SwiGLU (`Epilogue.swiGlu()`);
5. `r + residual`.

```java
var ep = GemmF16.Epilogue.NONE.withBias(GemmF16.Epilogue.Bias.COLUMN).withActivation(GemmF16.Epilogue.Activation.GELU);
gemm.encode(encoder, x, w, h, GemmF16.Shape.ofTransposedB(tokens, hidden, dim, 1f, 0f), ep, bias, null, null);
```

- **Operands:** gate and residual are laid out as C (`ldc`, `strideC`), and bias, gate and residual have C's
  element type. Each is passed exactly when the epilogue uses it, else null; an unused slot binds a placeholder.
- **Exactness:** bias, ReLU, gate and residual round as written, so TILED, SKINNY and SMALL stay bit for bit with
  `reference(algorithm, ..., epilogue, bias, gate, residual)` (`Epilogue.exact()`).
- **GELU and SiLU** use the device's `exp`, written so that its overflow and underflow give the right limits; they
  are held to the bound. `contract(algorithm, input, output, epilogue)` states it.

### Grouped GEMM (1.8.0)

`encodeGrouped(encoder, a, b, c, problems, grouped, epilogue, bias, gate, residual)` runs many multiplies of
different shapes in one dispatch, such as the experts of a mixture-of-experts layer.
- **The table:** `problems` is a buffer of `Problem` records (48 bytes each: sizes, leading dimensions and element
  offsets into the shared A, B, C and bias buffers). A kernel can write it, a router for example; `Problem.encode`
  writes it from the host.
- **Shared by all problems:** alpha, beta, the layouts and the epilogue (`Grouped`). `maxM` and `maxN` bound the
  problems and size the dispatch.
- **The kernel:** each problem runs the tiled kernel, so it is bit for bit `referenceGrouped`. Empty problems (an
  expert with no tokens) cost nothing.
- **Safety:** a problem that reaches outside a buffer, or beyond `maxM` or `maxN`, is skipped and reported through a
  `KernelErrors` buffer, so the submission fails with `KERNEL_ERROR`.

### Accuracy and reproducibility

- **Contract:** before `alpha` and `beta`, `|C - exact| <= k * 2^-24 * sum |a b|`. Later kernel versions may sum in
  another order, for example on matrix hardware, within the same bound.
- **The accumulation** is FP32 for every input. A narrower accumulation would be a separate, named variant; there
  is none. Every contract states its accumulation type and order (`contract(...)`).
- **The fixed-order algorithms are reproducible across devices** that round FP32 add and multiply correctly:
  - **TILED and SMALL:** each element is `acc = fma(a, b, acc)` over `k` in order. For FP16 and BF16 the product is
    exact, so this is `fl(acc + a b)` whether or not the device fuses `fma`.
  - **SKINNY:** element `q` of k goes to lane `(q / chunk) % L`, with `L` 32 and `chunk` 8 when B is transposed, 16
    and 2 when not. Each lane sums in order of `q`, then lane `t` takes lane `t + w` for `w = L/2, ..., 1`.
  - **The epilogue:** `fl(fl(alpha acc) + fl(beta c))`, with the add marked `NoContraction`, then the steps above.
  - **Checking it:** the references compute the same bits on the CPU, and the tests require equality on native
    Metal, MoltenVK and Mesa lavapipe (FP32 inputs where `FUSED_FMA` holds).
  - **On the CPU:** NNJava can reproduce a GPU layer bit for bit.
- **The matrix algorithm:** it is held to the bound only.
  - **Measured:** on an M4 Max it matched `GemmF16.reference` bit for bit on every test shape, FP16 and FP32 inputs
    alike. So that GPU's multiply-accumulate sums in order with FP32 rounding.
  - **Not promised:** Apple does not document this, and other GPU generations are unverified.

### Shape dispatch

`GemmF16.create(device)` compiles every algorithm the device runs for the input type, all or none, and `encode`
picks one per shape (`algorithm(shape)`) from `GemmF16.REGISTRY`, a kit `ShapeRegistry` (1.7.0).
- **Reproducible mode:** it is TILED only.
- **An explicit algorithm:** `create(device, algorithm, input, output)` runs that algorithm for every shape.
- **The costs** (1.8.0) are estimated GPU times (`GemmF16.Costs`), fitted to sweeps on an Apple M4 Max
  (`-Dgemm.shapes=true`, `-Dgemm.decode=true`; native Metal and MoltenVK agree):
  - **TILED:** 5.5 TFLOP/s over its 64-padded work when the GPU is full, which takes about 100 groups. A 128^3
    multiply (4 groups) runs at 4 % of that.
  - **SMALL:** 2.15 TFLOP/s over its 16-padded work, full from about 190 groups.
  - **SKINNY:** B at 400 GB/s when B is transposed (300 when not), slower by 17 % (45 %) per extra row of A. It is
    offered only for decode-sized products: k of at least 256 and at least 32 groups.
  - **SIMDGROUP_MATRIX:** 0.5 of TILED's time from `m = 64`, 0.75 with B transposed and `m <= 16`, else 1.5.
  - **Checked:** on all 46 measured shapes the model picks the fastest algorithm or one within 1.2 times of it,
    with the matrix kernel and without (`GemmF16Test.costModelPicksTheMeasuredFastest`).
- **Some measured choices** (GPU time relative to TILED, M4 Max):

  | Shape (m x n x k x batch) | B | Fastest | Its time | Matrix kernel |
  |---|---|---|---|---|
  | 1 x 4096 x 4096 | transposed | SKINNY | 0.14 | 0.68 |
  | 16 x 4096 x 4096 | as stored | SMALL | 0.59 | 1.63 |
  | 128 x 128 x 128 | as stored | SMALL | 0.29 | 0.76 |
  | 16 x 16 x 16 x 4096 | as stored | SMALL | 0.26 | 2.25 |
  | 512 x 512 x 512 | as stored | matrix | 0.52 | 0.52 |
  | 2048 x 2048 x 128 | as stored | matrix | 0.57 | 0.57 |

- **Checking it:** `REGISTRY.checkLimits(device, shape)` compiles every variant the device supports and checks it
  against the pipeline's and the device's limits. The tests run it on every backend.

### Decode bandwidth (1.8.0)

`-Dgemm.decode=true` measures the decode products and reports them as GB/s of B and as a share of the device's
streaming-read bandwidth, measured in the same run as a row sum over 256 MiB. On an M4 Max that measurement varies
between runs, from 415 to 516 GB/s (nominal 546).

| m = 1, n = k = 4096 | B transposed | B as stored |
|---|---|---|
| TILED (1.7.0) | 53 GB/s | 67 GB/s |
| SKINNY, FP16 weights | 378-408 GB/s, 77-96 % | 304-345 GB/s, 59-67 % |
| SKINNY, BF16 weights | 368 GB/s, 89 % | 294 GB/s, 71 % |
| SKINNY, FP32 weights | 395 GB/s, 80 % | 511 GB/s |

- **Rows of A:** each extra row costs about 17 % with B transposed: 52 % of TILED's time at `m = 16`.
- **MoltenVK** runs the same kernels at the same rates.

### Performance

Measured on an Apple M4 Max (40-core GPU, macOS 27). The figures are GPU time (`GpuSubmission.gpuTime`) over
back-to-back multiplies in one submission (`-Dgemm.benchmark=true`; `-Dgemm.input` picks the input type). MoltenVK
runs the Slang kernels.

| Size (m = n = k) | Metal, `SIMDGROUP_MATRIX` | Metal, `TILED` | MoltenVK, `TILED` |
|---|---|---|---|
| 512 | 7.8 TFLOP/s | 3.9 TFLOP/s | 3.0 TFLOP/s |
| 1024 | 11.3 TFLOP/s | 5.0 TFLOP/s | 4.6 TFLOP/s |
| 2048 | 12.8 TFLOP/s | 5.6 TFLOP/s | 5.4 TFLOP/s |
| 4096 | 12.7 TFLOP/s | 5.3-5.7 TFLOP/s | 5.4 TFLOP/s |

- **Inputs at 4096^3:** TILED runs 5.3 TFLOP/s with FP16, 5.1 with BF16 and 5.3 with FP32; the matrix kernel 12.7
  with FP16 and 2.7 to 3.0 with FP32.
- **The tiled kernel:** a group of 16 x 16 threads computes a 64 x 64 tile, each thread 4 x 4 outputs. k advances in
  steps of 16 through shared FP32 tiles.
- **Alternatives measured on the same GPU:**
  - 128 x 128 tiles with 8 x 8 per thread: 3.6 TFLOP/s;
  - steps of 32: 4.2 TFLOP/s.
- **Unrolling:** the fixed loops are `[ForceUnroll]`ed. The shader build compiles at `-O0`, and without unrolling the
  SPIR-V keeps the accumulators as indexed arrays. MoltenVK then ran 14 times slower: 376 against 5,261 GFLOP/s.
- **Epilogues** cost nothing measurable: the 4096^3 rates are the same with the 1.8.0 parameter block.
- **Peak:** the tiled kernel reaches about a third of the GPU's FP32 peak, and the matrix kernel about three
  quarters.

## Quantized weights (1.9.0)

Decode is limited by memory bandwidth, so weights in fewer bytes are faster: 4-bit weights with block 32 take 0.28
of FP16's bytes.

### `QuantizedMatrix`: the block formats

A `rows x cols` matrix (a linear layer's `n x k` weights) is quantized by blocks of 32 or 64 along its rows. Each
block holds integer codes and FP16 parameters; the formats are compatible in meaning with the common Q8_0, Q4_0 and
Q4_1 schemes.

| Format | Value | Codes | Parameters | Bytes per weight (block 32) |
|---|---|---|---|---|
| `Q8` | `q d` | -127 .. 127 | `d = fp16(max abs x * (1/127))` | 1.0625 |
| `Q4` | `q d` | -8 .. 7 | `d = fp16(v * (-1/8))`, `v` the value of largest magnitude, which maps to -8 | 0.5625 |
| `Q4M` | `fl(q d + m)` | 0 .. 15 | `m = fp16(min)`, `d = fp16((max - min) * (1/15))` | 0.625 |

- **Codes:** each is the integer nearest to `x / d` (Q4M: `fl(x - m) / d`), ties to even, then clamped. A block with
  `d = 0` has codes 0.
- **The same bits on every device:**
  - **Scales:** one correctly rounded multiply, never a division (GPU division is not correctly rounded). The
    reciprocal constants are Java's `1f / 127f` and `1f / 15f`.
  - **Codes:** an estimate, corrected by comparing with the exact midpoints `(2c + 1) d / 2`.
  - **FP16 rounding:** in integer arithmetic on the GPU, as `Float.floatToFloat16`.
- **Layout:** codes row-major (Q8 one byte, Q4 and Q4M a nibble each, element `2t` low), then from a 16-byte boundary
  the block parameters, row-major (`d`, or `d, m`). `bytes()`, `parameterOffset()` and `rowCodeBytes()` give it.
- **Refusals:** values must be finite and each block's scale must fit FP16.
- **Accuracy:** `errorBound()`: half a step inside the code range, plus Q4M's FP16 rounding of the minimum.
- **Apple GPUs** flush FP32 subnormals, so `Quantize` flushes subnormal weights, which the CPU encoder does not.

```java
var w = new QuantizedMatrix(QuantizedMatrix.Format.Q4, 32, n, k);
byte[] packed = w.quantize(weights);                       // or Quantize.pack on the GPU: the same bytes
```

### `Quantize`: on the GPU

`pack(encoder, values, matrix, packed)` and `unpack(encoder, matrix, packed, values)` quantize FP32 weights already on
the GPU and dequantize them, bit for bit as `QuantizedMatrix.quantize` and `dequantize`. A non-finite value, or a
scale beyond FP16, fails the submission with `KERNEL_ERROR`.

### `QuantizedGemm`

`C = epilogue(alpha * A * W^T + beta * C)`, with A FP16 or BF16 activations and W a packed `QuantizedMatrix`,
dequantized inside the kernel.

```java
try (var gemm = QuantizedGemm.create(device, GemmF16.Input.FP16, Precision.FP32)) {
    gemm.encode(encoder, x, w, packedBuffer, y, GemmF16.Shape.ofTransposedB(tokens, n, k, 1f, 0f));
}
```

- **Shapes:** a `GemmF16.Shape` with B transposed, whose `n` and `k` are W's. A batch of A and C shares W. The
  epilogues are `GemmF16`'s.
- **Algorithms:** `SKINNY` for `m <= 16` (decode) and `TILED` beyond (prefill).
  - **SKINNY:** a row of W's blocks is dealt to 32 lanes, each lane adding its blocks in order, then a fixed tree.
  - **TILED:** 64 x 64 tiles, k in steps of 32, each block's term added as the block ends.
  - **Specialization:** the pipelines are compiled on first use per format, block size and (SKINNY) rows tier (1, 4
    or 16). A decode step then holds one row of accumulators instead of sixteen.
- **Accuracy: the same bits on every device,** native Metal, MoltenVK and lavapipe:
  - **Exact products:** an activation (11 or 8 significant bits) times a code (at most 8) is exact in FP32. So each
    block's `s = sum a q` depends only on its order, which both algorithms fix.
  - **Unfused terms:** the block adds `fl(d s)`, and Q4M also `fl(m sum a)`, with the adds unfused.
  - **The reference:** `QuantizedGemm.reference` computes the same bits. Against A times the dequantized W, the error
    is within `(k + 2) 2^-24 sum |a w|`.
- **Decode, measured** (`-Dquant.decode=true`): M4 Max, m = 1, GPU time per product. The quantized weights rotate
  through copies over 256 MiB so the system cache cannot serve them.

  | n x k | FP16 (`GemmF16` SKINNY) | Q8 | Q4 | Q4M |
  |---|---|---|---|---|
  | 4096 x 4096 | 83 us | 43 us (1.9x) | 29 us (2.9x) | 35 us (2.4x) |
  | 11008 x 4096 | 250 us | 105 us (2.4x) | 62 us (4.0x) | 68 us (3.7x) |
  | 4096 x 11008 | 228 us | 110 us (2.1x) | 70 us (3.3x) | 73 us (3.1x) |

  - **More rows:** at m = 4 the quantized product runs 1.3 to 2.2 times faster than FP16. At m = 16 it is about
    even (0.74 to 1.16).
- **Prefill:** TILED runs about 4.6 TFLOP/s at 512 x 4096 x 4096, a little below the FP16 tiled kernel's 5.3. The matrix kernel's
  12.7 TFLOP/s for FP16 weights has no quantized form yet.

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
- **GEMM:** a Vulkan cooperative-matrix path, once there is hardware to test it on; a faster FP32 matrix kernel.
- **Other primitives:** layer and RMS norms (1.10).
- **Quantized weights:** a quantized form of the matrix kernel for prefill.

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

## Debug builds and reproducible mode (1.7.0)

- **Debug builds:** `-Ddynamisgpu.kernels.debug=true`, or `new KitOptions(false, true).call(() -> Reduce.create(device))`
  for one primitive. The kit primitives then compile with their bounds checks.
  - **Checks:** every buffer access is checked against the buffer's length.
  - **Out-of-range accesses:** they are skipped and reported with code 1 (`DG_ERROR_INDEX`) and the thread or row
    index. The submission fails with `KERNEL_ERROR`.
  - **Why:** buffers share pooled blocks, so an out-of-range write would otherwise corrupt a neighbouring buffer.
  - **Normal builds:** the checks are compiled out through a specialization constant.
  - **Kernel authors:** guard accesses with `dg_in_range(errors, index, length, at)` from `dynamis/kit.slang`.
- **Reproducible mode:** `-Ddynamisgpu.reproducible=true`, or `new KitOptions(true, false)`. Primitives that have a
  fixed-order algorithm choose it: `GemmF16` and `AttentionF16` use `TILED`. Results are then the same bits on every
  device, as their contracts state (`contract().reproducible()`).
- **The cross-backend harness:** `org.dynamisengine.gpu.test.compute.CrossBackend` runs an operation on two devices
  and compares the results under its contract.
  - **EXACT:** the same bits.
  - **ULPS:** within twice the bound.
  - **FORMULA:** a per-element tolerance.

  It reports the first element beyond the contract.

