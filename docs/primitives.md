# Throughput primitives

`dynamis-gpu-primitives` (module `org.dynamisengine.gpu.primitives`) holds throughput kernels on the compute API,
written in Slang and built with `dynamis-gpu-shaders`. Their contract is an accuracy bound, not bit-exactness. The
exact GPU twins of Vectrix's kernels are vectrix-gpu. The first primitive is a half-precision matrix multiply.

## `GemmF16`

`C = alpha * A * B + beta * C`:
- **Layout:** A is `m x k` and B is `k x n`, both FP16; C is `m x n` in FP32. All are row-major with leading
  dimensions in elements (`GemmF16.Shape`).
- **Use:** it is a dense layer with half-precision weights and activations.

```java
try (GemmF16 gemm = GemmF16.create(device)) {              // needs GpuCapabilities.FP16
    var shape = GemmF16.Shape.of(m, n, k, 1f, 0f);          // dense: lda = k, ldb = ldc = n
    gemm.encode(encoder, a, b, c, shape);                    // a, b: >= shape.aBytes(), bBytes(); c: cBytes()
}
```

- **`beta == 0`:** C is not read (as in BLAS), so it may hold anything.
- **Buffer sizes:** transfers move whole 32-bit words, so round the FP16 buffers up to a multiple of 4 bytes.
- **Refusal:** `create` throws `UNSUPPORTED` on a device without `FP16`.
- **Indexing:** element indices are 32-bit, so each matrix holds fewer than 2^32 elements.

### Algorithms

`GemmF16.create(device)` picks the fastest algorithm the device has; `create(device, algorithm)` chooses one.

| Algorithm | Runs on | Kernel |
|---|---|---|
| `TILED` | every device with `FP16` | Slang (`gemm_f16.slang`): shared FP32 tiles and fused multiply-adds |
| `SIMDGROUP_MATRIX` | native Metal on Apple7+ GPUs (`SUBGROUP_MATRIX_F16`) | MSL (`metal/gemm_f16_simdgroup.metal`): 8 x 8 hardware multiply-accumulate |

- **The matrix kernel:** a group of four SIMD-groups computes a 64 x 64 tile, each SIMD-group a 32 x 32 quarter.
  - **Interior tiles** (`k` a multiple of 8) load 8 x 8 blocks straight from device memory; the caches serve the
    reuse.
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

## Planned

- **GEMM:**
  - FP16 output;
  - transposed B (the `x W^T` of a linear layer);
  - batched and strided multiplies;
  - a Vulkan cooperative-matrix path, once there is hardware to test it on.
- **Other primitives:** reductions, softmax and attention, and int8 and BF16 weight formats.
