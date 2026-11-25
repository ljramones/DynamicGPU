# DynamisGPU optimization review of 1.11.0 and 1.10.1

On the Apple M4 Max, the combined branch cut 2048-token prefill time by 3.6% on Metal and 26% on MoltenVK. At 512 keys, Metal decode fell from 63.5 to 9.7 µs. At 32768 keys it fell from 411.2 to 330.9 µs. Sampling 128256 varied logits with top-k 40 and top-p 0.95 took 41.32 µs on Metal, down from 75.09 µs. Every tested attention element stayed within the existing bound, and every tested sampler draw matched the CPU reference.

## Release against the combined branch

Both checkouts ran the same warmed benchmark driver with initialized inputs. The control used production code from `18dddb6`, including the 1.10.1 sampler. Times are whole-submission GPU times per encode except where host and wall time are explicitly listed. Each attention shape uses dimension 128, 32 query heads and FP16 output. Decode uses eight KV heads. The paged test records four layers of the same 512-key sequence.

| Backend | Workload | 1.11.0 | Review branch |
|---|---|---:|---:|
| Metal | Causal prefill at 512 / 1024 / 2048 tokens (ms) | 0.42 / 1.44 / 5.31 | 0.41 / 1.38 / 5.12 |
| Metal | Decode at 512 / 4096 / 32768 keys (µs) | 63.5 / 89.8 / 411.2 | 9.7 / 44.3 / 330.9 |
| Metal | Top-k 40 with top-p 0.95, varied logits (µs) | 75.09 | 41.32 |
| Metal | Four paged decodes, host encoding / GPU / wall (µs) | 3915.5 / 345.7 / 4320.9 | 17.2 / 60.2 / 84.9 |
| MoltenVK | Causal prefill at 512 / 1024 / 2048 tokens (ms) | 0.92 / 3.40 / 12.82 | 0.64 / 2.42 / 9.45 |
| MoltenVK | Decode at 512 / 4096 / 32768 keys (µs) | 77.8 / 116.7 / 429.5 | 39.5 / 58.5 / 340.8 |
| MoltenVK | Top-k 40 with top-p 0.95, varied logits (µs) | 97.84 | 56.39 |
| MoltenVK | Four paged decodes, host encoding / GPU / wall (µs) | 4426.7 / 373.3 / 4944.4 | 20.8 / 180.3 / 235.0 |

Repeating the native Metal comparison gave 65.3 → 9.7 µs for 512-key decode and 4219.4 → 89.3 µs wall time for four paged layers. The latter includes the host savings from caching immutable shader metadata.

## Isolated optimization trials

| Backend and workload | Release | Selected change |
|---|---:|---:|
| Metal causal prefill, 512 / 1024 / 2048 tokens | 0.426 / 1.440 / 5.291 ms | 0.411 / 1.376 / 5.118 ms |
| MoltenVK causal prefill, same shapes | 0.921 / 3.373 / 12.822 ms | 0.646 / 2.412 / 9.385 ms |
| arm64 lavapipe prefill, 8 heads, 128 / 256 tokens | 1.994 / 6.712 ms | 1.595 / 4.949 ms |
| Metal decode, 512 / 4096 / 32768 keys | 64.8 / 94.6 / 410.4 µs | 14.0 / 65.3 / 329.3 µs |
| MoltenVK decode, same shapes | 50.6 / 78.7 / 432.9 µs | 39.9 / 58.0 / 335.6 µs |
| Metal sampling, k=40 / 1024, varied logits | 75.12 / 121.85 µs | 41.46 / 111.86 µs |
| MoltenVK sampling, k=40, varied / concentrated logits | 96.05 / 94.66 µs | 63.26 / 75.91 µs |
| Metal, four 512-key paged decodes, cached shader metadata on both sides | 224 µs | 195 µs with immutable page tables |

The attention benchmarks use dimension 128 and FP16 output. The M4 Max tests use 32 query heads and decode uses eight KV heads. Each pair uses the same inputs, times the whole submission after warm-up, and reports the best of four rounds of at least 20 ms. The MoltenVK decode control includes the metadata cache on both sides. Separate trial series use their own controls. The decode benchmark repeatedly reads the same K/V buffers. The reported byte rate can include cache traffic.

## Leads and precision

- **Prefill.** Checking visibility once for a complete key block won on Metal. Register softmax passed the elementwise tests on the M4 Max but ran slower. Overlapping K/V loads, larger blocks and the dimension-256 matrix path also lost. Pre-scaling half Q for exp2 introduced a rounding and possible overflow without a repeatable gain. On Vulkan, processing 16 lanes per row and interleaving two independent dot products won together. The original accumulation order is preserved. The guide records the before-and-after times for each trial. [Prefill measurements](primitives.md#attentionf16)
- **Decode.** Short native Metal contexts can use a single workgroup per output row. Longer contexts benefit from specializing the row count and using subgroup reductions. The new paths can change the order of FP32 additions in the dot products, probability sums or value sums. The elementwise tests now reserve the FP16 probability allowance for the matrix path. The scalar bound covers both new schedules. A padded matrix decode lost at medium and long contexts. The atomic last-group combine produced no valid Metal executable with the required memory ordering, so it has no after timing. [Decode measurements](primitives.md#attentionf16)
- **Sampler.** Large rows with small k use a lower bound on the kth logit, compact the candidates and sort them. A simpler bound was faster on random inputs but took 94.28 µs on an adversarial distribution where the release took 75.65 µs. Ranking bucket winners and using window minima improved all six measured distributions. Larger k can use subgroup shuffles for the short bitonic strides. Both paths preserve tie order, the arithmetic of the draw and its CPU reference. The original path remains available when the optional kernels exceed a device limit. [Sampler measurements](primitives.md#sampler)
- **Scratch and uploads.** Immutable page-table snapshots survive page changes, sequence closure and recorded use. Repeated sampler draws benefited from reusing scratch in one encoder, but a single draw and reduction in a fresh encoder showed no repeatable host gain. That trial would also retain the final buffers beyond the encoder's lifetime. [Scratch measurements](primitives.md#host-cost-of-sampler-and-log-sum-exp-scratch) · [Page-table measurements](primitives.md#pagedkvcache)

## Verification

All committed Slang outputs were rebuilt with the Vulkan SDK 1.4.341.0 toolchain. The script exited zero. The JDK 25 portable build passed with 285 tests executed and 221 skipped. Native Metal and MoltenVK each passed all 22 attention and inference tests including opt-in benchmarks. The full Docker lavapipe suite passed on LLVM 20.1.2. Its four opt-in benchmarks across the attention and inference suites were skipped.

On each backend the largest measured fused attention error was 0.898 of the bound. Each backend matched all 1440 fixed-seed sampler draws, plus the cutoff, tie, partial-chunk and optional-pipeline fallback cases. The page-table lifetime test records two different mappings and closes the cache before submission.

```sh
SDK="$HOME/VulkanSDK/1.4.341.0/macOS"
PATH="$SDK/bin:$PATH" sh scripts/build-slang-kernels.sh
mvn -o verify
mvn -o -pl dynamis-gpu-primitives -am -Pmetal-native,metal-hardware test \
    -Dtest=MetalAttentionTest,MetalInferenceTest -Dsurefire.failIfNoSpecifiedTests=false \
    -Dgemm.benchmark=true -Dinference.benchmark=true
VK_DRIVER_FILES="$SDK/share/vulkan/icd.d/MoltenVK_icd.json" mvn -o -pl dynamis-gpu-primitives -am \
    -Pvulkan-hardware -Ddynamisgpu.vk.loader="$SDK/lib/libvulkan.1.dylib" test \
    -Dtest=VulkanAttentionTest,VulkanInferenceTest -Dsurefire.failIfNoSpecifiedTests=false \
    -Dgemm.benchmark=true -Dinference.benchmark=true
sh scripts/test-lavapipe.sh
```
