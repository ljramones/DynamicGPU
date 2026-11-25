# Long-context paged decode and KV precision

On an Apple M4 Max, repeatedly decoding against the same FP16 pages and cycling through up to 512 MiB of distinct pages took nearly the same GPU time at 32,768 tokens. At 4,096 tokens, cycling through layers cost 6–13% across the tested batches and backends. Two candidate 4-bit value encodings reduced the raw cache payload by over half, but each produced about 11% relative attention-output error on a synthetic long-context outlier fixture.

**Decision.** Keep the FP16 cache and proceed with fused attention masks. Defer a packed GPU append/decode prototype until a model workload can establish an acceptable loss or a concrete cache-capacity requirement. The timing experiment measures FP16 reuse. Measuring the latency of a packed cache would require its own GPU implementation, including the cost of appending a token.

## FP16 working set

The base is DynamisGPU v1.14.0 (`31210b3`). Native Metal and MoltenVK 1.4.1 ran on the same M4 Max, macOS 27.0.1, Temurin 25.0.4. Each decode used 32 query heads, eight KV heads, dimension 128, one query per sequence, 64-token pages and FP16 output. Batch one used the single-sequence paged entry point. Batch two used batched paged decode with equal lengths and distinct pages for each sequence.

Each schedule performed the same number of calls on the same allocated cache. One used layer zero on every call. The other visited each layer once. Q was initialized once. Each layer's K and V pools had their own deterministic 1 MiB tile of varied FP16 data repeated across the pool. One V column identified its layer and was checked for every query head and sequence before timing. Both schedules reused their page tables and excluded allocation, initialization and readback.

Times below are microseconds **per layer call**. Each pair of columns lists reused / rotating. Host encoding and elapsed wall time are measured independently. Raw observations are whole-submission measurements normalized per sweep. Each table entry is the median of eight observations divided by the calls per sweep. Those observations came from two JVM processes per backend, with four warmed, paired rounds per process. Both modes used the same repetition count within a pair, alternated measurement order and ran for at least 20 ms per measured submission.

| Backend | Tokens | Batch | Calls/sweep | Distinct KV MiB | GPU µs | Host encode µs | Wall µs | GPU change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Metal | 4,096 | 1 | 20 | 16 / 320 | 67.6 / 73.4 | 16.8 / 16.8 | 100.8 / 108.3 | +8.6% |
| Metal | 4,096 | 2 | 10 | 32 / 320 | 122.5 / 132.0 | 8.7 / 8.9 | 141.7 / 150.0 | +7.7% |
| Metal | 32,768 | 1 | 3 | 128 / 384 | 379.3 / 379.0 | 9.3 / 9.0 | 424.3 / 410.0 | −0.1% |
| Metal | 32,768 | 2 | 2 | 256 / 512 | 804.7 / 803.7 | 8.4 / 8.5 | 827.2 / 825.7 | −0.1% |
| MoltenVK | 4,096 | 1 | 20 | 16 / 320 | 76.3 / 86.3 | 21.8 / 21.6 | 122.2 / 134.5 | +13.1% |
| MoltenVK | 4,096 | 2 | 10 | 32 / 320 | 136.2 / 145.1 | 9.2 / 8.5 | 158.2 / 166.1 | +6.5% |
| MoltenVK | 32,768 | 1 | 3 | 128 / 384 | 402.5 / 402.7 | 9.6 / 9.2 | 429.5 / 427.1 | +0.1% |
| MoltenVK | 32,768 | 2 | 2 | 256 / 512 | 824.0 / 821.1 | 9.7 / 9.2 | 857.3 / 854.8 | −0.4% |

At 32K and batch one, a single layer already contains 128 MiB of FP16 KV. The per-process GPU changes for that shape ranged from −0.6% to +0.9% across both backends. The byte counts describe the distinct K/V address ranges touched. Physical DRAM traffic and full-model token latency would require separate measurements. [All GPU rounds](evidence/paged-kv/gpu-rounds.csv)

## CPU codec comparison

Two candidate formats use groups of 32. Magnitude's affine format stores unsigned 8-bit keys and 4-bit values with FP16 scale and offset for each group. The Vectrix candidate uses the released `BlockQuant.encodeBlock` and `decode` for symmetric Q8 keys and Q4M values. At width 128, the combined raw payload per KV head-token is 512 bytes for FP16, 224 for affine K8/V4 and 216 for Q8/Q4M. For one 32K sequence with 32 layers and eight KV heads, these amount to 4.00, 1.75 and 1.6875 GiB respectively, before page slack, tables and allocator overhead. [Magnitude reference](https://github.com/magnitudedev/magnitude/blob/78c92e4f21e38f9528f21e09c5e43936a04b6b0d/inference/engine/kernels/tests/attention_k8v4.rs#L46)

One dimension-128 KV head served four query heads for each of three seeds. Keys and values were sampled independently and rounded to FP16. The centered fixture used unit Gaussian samples. The shifted fixture chose a mean of +3 or −3 per group with Gaussian standard deviation 0.35. The outlier fixture added +12 or −12 to a centered sample with probability 1/128. Two query heads per seed used unit Gaussian scale and two used twice that scale. The 4K history was the prefix of the 32K history.

The reference used the original FP16 operands with stable softmax and double-precision accumulation. Each candidate packed its codes into bytes, decoded them, then ran exactly the same attention. The percentages are the L2 norm of the attention-output difference divided by the L2 norm of the reference output, pooled across heads, seeds and coordinates.

| Fixture | 4K affine K8/V4 | 4K Q8/Q4M | 32K affine K8/V4 | 32K Q8/Q4M |
|---|---:|---:|---:|---:|
| Centered | 7.61% | 7.66% | 8.12% | 7.98% |
| Shifted groups | 1.05% | 2.11% | 0.89% | 1.74% |
| Sparse outliers | 9.08% | 9.33% | 10.86% | 11.14% |

With the 32K outlier fixture and the original FP16 keys, quantizing just V produced **10.69%** relative output error for both codecs. Quantizing just K produced **2.15%** for affine K8 and **2.56%** for symmetric Q8. With shifted groups, the key-only errors were **0.27%** and **1.53%**. The experiments identify the value precision as the first variable to revisit. The next quality comparison should use captured post-RoPE keys, values and queries and check the effect on model logits. The synthetic measurements establish neither a model-quality tolerance nor a new GPU accuracy bound. [All CPU errors, including absolute and per-operand errors](evidence/paged-kv/cpu-codecs.csv)

## Reproduce

The GPU benchmark is opt-in. Run each backend separately. The optional `-DpagedDecode.layers=N` fixes the layer count to an integer of at least two. With the default layer selection the largest KV allocation is 512 MiB.

```sh
mvn -o -pl dynamis-gpu-primitives -am -Pmetal-native,metal-hardware test \
  '-Dtest=MetalAttentionTest#benchmarkPagedDecodeWorkingSet' \
  -Dsurefire.failIfNoSpecifiedTests=false -DpagedDecode.benchmark=true

sdk="$HOME/VulkanSDK/1.4.341.0/macOS"
VK_DRIVER_FILES="$sdk/share/vulkan/icd.d/MoltenVK_icd.json" \
  mvn -o -pl dynamis-gpu-primitives -am -Pvulkan-hardware test \
  "-Ddynamisgpu.vk.loader=$sdk/lib/libvulkan.1.dylib" \
  '-Dtest=VulkanAttentionTest#benchmarkPagedDecodeWorkingSet' \
  -Dsurefire.failIfNoSpecifiedTests=false -DpagedDecode.benchmark=true
```

The CPU experiment uses JDK 25 and locally installed Vectrix release jars. Both 4.1.0 and the 4.0.0 version pinned by DynamisGPU 1.14.0 reproduced the committed CSV byte for byte.

```sh
sh scripts/kv-study/run.sh 4.1.0 > /tmp/dynamis-kv-codecs.csv
cmp /tmp/dynamis-kv-codecs.csv docs/evidence/paged-kv/cpu-codecs.csv
```
