# DynamisGPU roadmap

Agreed 2026-11-24, after release 1.4.0 (published).

**Goal:** build DynamisGPU into a complete, layered GPU management system for compute, so that throughput work
(neural networks and LLMs) and the engine's GPU work run on one runtime.

**Order:**
1. Build the lower layers first.
2. Then flow the new capabilities into Vectrix, and move reusable GPU plumbing out of Vectrix into DynamisGPU.
3. Then plan NNJava, which stays untouched until then.

Graphics (images, render passes, the LightEngine breakup) follows the same order once compute is complete.

**How to work on it:**
- [AGENTS.md](../AGENTS.md) covers the environment and the test commands for each backend, including the lavapipe
  container.
- It also covers the kernel-writing rules and Slang traps, the testing conventions, the repository rules and the
  release checklist.
- Every milestone below ends with a **Where to start** list of the files involved.

## The layers

Each layer depends only on those below it, and each is tested on native Metal, MoltenVK and lavapipe.

| Layer | Module | Owns | State after 1.4.0 |
|---|---|---|---|
| 5. Bridges | `dynamis-gpu-vectrix`, vectrix-gpu | GPU data from Vectrix types; Vectrix's exact twins | layouts done; the twins run on DynamisGPU |
| 4. Primitives | `dynamis-gpu-primitives` | throughput kernels, accuracy-bounded | `GemmF16`, `Softmax`, `AttentionF16` |
| 3. Kernel kit | `dynamis-gpu-api` (`compute.kit`), `dynamis-gpu-shaders` | the Slang toolchain, the kernel library, and the shared plumbing every kernel family needs | toolchain and library done; plumbing duplicated in vectrix-gpu and the primitives |
| 2. Runtime | `dynamis-gpu-api` (`compute`), backends | memory, queues and sync, command recording and replay, pipelines, profiling, errors | one queue, an allocation per buffer, submissions rebuilt every time |
| 1. Devices | backends | adapters, capability tiers, measured features | tiers, timestamps and Metal matrices done |

The work below fills layers 2 to 4 bottom-up. Every milestone is a release.

## Milestones

### 1.5: Runtime I, memory and replay

The two biggest costs outside the kernels today are allocation and re-recording.
- **Memory pools:**
  - Sub-allocate buffers from large device allocations (a per-placement pool with size classes and a free list).
  - Recycle the staging memory that uploads and downloads use today, one fresh buffer per transfer, through a ring.
  - `GpuDevice.memoryStats()` reports what is in use and reserved.
  - **Acceptance:** allocating and freeing many small buffers no longer reaches the driver each time, and uploads
    don't allocate.
- **Recorded command replay:**
  - Record an encoder once as a `GpuRecording`.
  - Replay it with new parameter blocks and rebound buffers, without re-validating or re-recording.
  - **Backends:** Vulkan reuses its command buffer and descriptor sets; Metal re-encodes from a frozen command list
    with nothing validated again.
  - **Use:** an LLM decode step is dozens of small dispatches per token, and this removes the per-token host cost.
  - **Acceptance:** replaying a 50-dispatch recording costs a small fraction of building it. The cost is measured
    and documented.
- **Per-dispatch timing:**
  - Timestamps around each dispatch where the backend allows it (Vulkan everywhere; Metal at encoder boundaries on
    Apple GPUs).
  - `GpuSubmission.dispatchTimes()` returns them.
- **Persistent pipeline cache:** Vulkan `VkPipelineCache` and Metal binary archives, saved to a cache directory.
  Start-up compiles only once per driver version.

- **Where to start:**
  - **Memory:** buffers are allocated in `VulkanComputeDriver.allocate` (one `vkAllocateMemory` each) and
    `MetalDriver.allocate` (`dg_buffer_create`).
  - **Staging:** it is created per transfer in `VulkanComputeDriver.submit` (`Upload`, `Download`), in
    `MetalDriver.submit` (`Download`), and in the bridge's `dg_upload` and `dg_fill` (`newBufferWithBytes` per
    call).
  - **Validation and handles:** `ComputeDevice` (spi) validates and tracks handles; keep pooling below the SPI, so
    `GpuBuffer` semantics don't change.
  - **Replay:** encoders collect validated `ComputeDriver.Command`s (`ComputeDevice.Encoder`), and drivers record them
    in `submit(List<Command>)`. Replay needs a driver-level "prepare once, submit many": add it to the SPI, with a
    default that re-submits the command list.
  - **Per-dispatch timing:**
    - **Vulkan:** a query pool of 2 entries per dispatch around `vkCmdDispatch`.
    - **Metal:** a `MTLCounterSampleBuffer` (common timestamp counter set) attached through
      `MTLComputePassDescriptor.sampleBufferAttachments` at stage boundaries. Each dispatch is already its own
      compute encoder (`dg_compute_begin`); GPU timestamps convert to nanoseconds with `sampleTimestamps`. This is
      a bridge ABI bump.
  - **Pipeline cache:**
    - **Vulkan:** `vkCreateComputePipelines` passes `VK_NULL_HANDLE` today; pass a `VkPipelineCache` loaded from and
      saved to disk.
    - **Metal:** `MTLBinaryArchive`.

### 1.6: Runtime II, queues and synchronization

- **Multiple queues:** a compute queue and a transfer queue where the device has them (Vulkan queue families; Metal
  command queues). `GpuDevice.queue(QueueKind)` selects one.
- **Cross-queue dependencies:** timeline semaphores on Vulkan, `MTLSharedEvent` on Metal.
  - **API:** a `GpuFence` that one submission signals and another waits on, without the host.
- **Async transfer:** uploads of the next batch overlap compute of the current one.
- **Acceptance:**
  - A stream of uploads interleaved with dependent compute is correct on every backend.
  - On hardware with a separate transfer engine, uploads overlap compute; the overlap is measured.

- **Where to start:**
  - **Queue selection:** `VulkanProvider.family` picks the first compute-capable family. Add a transfer-only family
    where one exists.
  - **The queue model:** `GpuDevice.queue()` returns the single `GpuQueue`, and `ComputeDevice` keeps one pending
    list.
  - **Metal:** `DGDevice` holds one `MTLCommandQueue`; shared events need the bridge (ABI bump).
  - **Test devices:** lavapipe exposes one queue family, so its tests cover the single-queue fallback, and Apple GPUs
    test real queues.

### 1.7: Kernel kit

- **Shared plumbing:**
  - **Parameter-block builder:** fields named and checked against the manifest.
  - **Grids:** linear-to-2D grids that respect the group-count limits.
  - **Checks:** 32-bit index limits, and extent, matrix and SoA checks.
  - **Lifecycle:** pipeline sets that compile together and close together, and placeholder buffers for absent
    optional arguments.
  - These are taken from vectrix-gpu's `Kernels`, `Rows` and `EngineBuffers` and the primitives' copies.
- **A Slang include library:**
  - fixed-order workgroup reductions (max, sum) and scans;
  - the groupshared rules that avoid Slang's reread bug;
  - subgroup fast paths behind the capability tiers.
- **Specialization:**
  - Specialization constants on Vulkan.
  - Metal function constants, which ABI 1 rejected; the bridge gains them.
  - Then a kernel can be compiled for a shape class instead of branching at run time.
- **Shape-dispatched kernels:** a small registry that picks a kernel variant by shape and capability, with a
  measured heuristic, so callers keep one entry point.

- **Where to start:**
  - **The copies to unify:**
    - in vectrix-gpu: `Kernels` (`Params`, `grid`, `matrix`, `MAX_INDEX`), `Rows` (`compile`, `placeholder`, `close`,
      `shape`) and `EngineBuffers` (`soa`, `distinct`, `groups`);
    - in the primitives: the `check` methods and parameter `ByteBuffer`s in `GemmF16`, `Softmax` and `AttentionF16`.
  - **Where it goes:** a public `org.dynamisengine.gpu.api.compute.kit` package (or a `dynamis-gpu-kit` module if it
    needs more than the API).
  - **Validating parameter blocks:** `KernelLibrary.parameterOffsets` and `parameterTypes` already give names,
    offsets and types to check them against.
  - **Function constants:** `ComputeDevice` rejects Metal specialization today (ABI 1 note in
    `docs/compute-runtime.md`). The bridge's `dg_pipeline_create` needs an `MTLFunctionConstantValues` argument.

### 1.8: GEMM family

On the kit, `GemmF16` grows into a family. Each form is tested bit for bit (tiled) or within its bound, on every
backend.
- **Skinny GEMM / GEMV** (`m` from 1 to 16): one SIMD-group streams a slice of the weights once.
  - **Use:** the decode step of an LLM.
  - **Acceptance:** at `m = 1` it is limited by memory bandwidth, and its rate is reported as a share of the
    device's bandwidth.
- **Small batched tiles** (16 x 16 and 32 x 32): many tiny multiplies, such as game-AI networks.
- **Grouped GEMM:** many multiplies with different shapes in one dispatch, from an offsets buffer (mixture-of-experts
  layers).
- **Transposed A** (`A^T B`): the weight gradient of the backward pass.
- **Fused epilogues:**
  - bias (per row or column);
  - activation (ReLU, GELU, SiLU);
  - residual add;
  - gated SwiGLU.

  All are applied in FP32 before the output rounding. Bias, ReLU and residual keep the tiled kernel bit-exact; GELU
  and SiLU use the device's transcendentals and are bounded.
- **More precisions:**
  - FP32 inputs (Metal float SIMD-group matrices);
  - BF16 inputs, converted exactly to FP32 in the kernel.

  BF16 is never narrowed to FP16, whose exponent range is smaller.
- **Column-major:** no new kernels. `C^T = B^T A^T` maps column-major onto the row-major forms; a helper and the
  documentation cover it.
- **Vulkan cooperative matrices:** once there is NVIDIA, AMD or Intel hardware to test on.

- **Where to start:**
  - **Existing code:** `GemmF16` (Java), `gemm_body.slang` (tiled, one entry per `Precision`) and
    `metal/gemm_f16_simdgroup.metal` (matrix, a template).
  - **Shape and parameter block:** `GemmF16.Shape` carries the shape, B layout and batch. The parameter block is
    12 fields (`KernelLibrary` reports it).
  - **The skinny kernel:** add it as a third `Algorithm`, chosen by `create` and `encode` from the shape (the 1.7
    registry).
  - **Epilogues:**
    - Add an epilogue description to `Shape`, or to a new `Epilogue` record, plus optional bias and residual buffers.
    - Absent buffers bind a placeholder (the kit).
    - Extend `GemmF16.reference` and `referenceF16` so the tiled kernel stays bit for bit for bias, ReLU and residual.

### 1.9: Quantized weights and packing

- **Weight-only quantization:**
  - int8 and int4 weights in blocks with FP16 scales (and zero points for int4), dequantized inside the GEMM and GEMV
    kernels.
  - **Formats:** block sizes 32 and 64, compatible in meaning with the common Q8_0, Q4_0 and Q4_1 schemes, with
    layouts chosen for the kernels.
  - **Why:** decode is limited by memory bandwidth, so 4-bit weights are roughly 4 times faster than FP16.
- **Packing:** a one-time packing of weights, at model load, into the blocked layout the kernels read: quantized
  blocks and FP16 tiles for the matrix kernel. Activations stay in plain row-major.
- **Quantize and dequantize kernels**, with the CPU reference bit for bit.

- **Where to start:**
  - **CPU side:** the block formats need a CPU encoder and decoder. Put them in Vectrix `nn.Formats` (which has int8
    today) and test the GPU against them bit for bit.
  - **Kernels:** the dequantizing kernels are new entries of the GEMM family, reading packed blocks. The packer is a
    Java class producing the packed buffer.

### 1.10: Inference kernels

- **Norms:** RMSNorm and LayerNorm, optionally fused with the residual add.
- **RoPE:** rotary position embedding, with the common rotation layouts.
- **Elementwise activations:** SiLU, GELU and SwiGLU as standalone operations, plus residual add.
- **Embedding gather:** FP16 and quantized tables.
- **Sampling on the GPU:**
  - argmax, top-k, top-p and temperature, with a seeded generator;
  - logits never return to the host.
- **Reductions and scans** as public primitives.

- **Where to start:**
  - **Row kernels:** they follow `softmax_body.slang`: one 256-thread group per row, fixed-tree reductions, and
    results in their own group variables. After 1.7 the reductions come from the Slang include library.
  - **Twins:** vectrix-gpu has exact twins of several of these (`GpuNorms`, `GpuActivations`, `GpuSampler`). The
    throughput versions here are bounded, tested against double references, and may be faster.

### 1.11: Attention at scale

- **Flash attention:**
  - fused `softmax(Q K^T) V` that keeps scores on chip;
  - memory linear in the sequence;
  - skips masked blocks, roughly halving causal work;
  - the Metal matrix path first.
- **Paged KV cache:** fixed-size pages in a pool, an append kernel, and attention over the page table.
- **Decode attention:** one query against a long cache, split across the keys and combined.

- **Where to start:**
  - **The current version:** `AttentionF16` composes `GemmF16`, `Softmax` and `GemmF16`, and its test checks the
    composed bound in double precision (`AttentionConformance`).
  - **Flash attention:** replaces the composition behind the same class, and keeps that test as the oracle.
  - **First kernel:** the Metal matrix path first (hand-written MSL, like `gemm_f16_simdgroup.metal`).

### 1.12: Training kernels

- **Backward passes:** softmax, the norms, attention, and the activations.
- **Transposed-A GEMM** (from 1.8), and gradient accumulation with float atomics (`ATOMIC_FLOAT32_ADD`).
- **Optimizers:** AdamW and SGD as throughput kernels. The exact twins stay in vectrix-gpu.

- **Where to start:**
  - **Exact versions:** vectrix-gpu's `GpuOptim`, `GpuLoss` and `GpuNorms` hold exact versions of several of these.
    Use them as references where they apply.
  - **Gradient checks:** compare finite differences in double against the forward kernels.

## Flowing into Vectrix

These follow as the layers land.
- **vectrix-gpu moves onto the kit** (after 1.7): its `Kernels`, `Rows` and `EngineBuffers` plumbing is replaced by
  DynamisGPU's. Its 58 exact twins and their CPU models stay in vectrix-gpu.
- **vectrix-gpu uses the runtime** (after 1.5 and 1.6): pooled buffers, recorded replay for its multi-dispatch
  operations, and per-dispatch timing in its benchmarks.
- **Vectrix's 4.0 cleanup:** the deprecated GPU layout classes are removed, and the jar is split into
  `vectrix-core`, `-render`, `-signal` and `-ml`.
- **CPU references:**
  - Quantization formats and their CPU encode and decode belong in Vectrix (`nn.Formats` already holds int8).
  - DynamisGPU's quantize kernels are tested against them, the same way `GemmF16.reference` works today.

## After this

- **NNJava:** planned on top of the finished stack: tensors, autograd and models, choosing exact (vectrix-gpu) or
  throughput (primitives) kernels per operation.
- **Graphics:**
  - images and samplers;
  - API consolidation (retiring the Vulkan-shaped `api.buffer` and `api.gpu`);
  - graphics pipelines and the swapchain;
  - the LightEngine breakup;
  - more backends (an OpenGL fallback, WebGPU).

## Rules that hold throughout

- **Testing:** every capability is tested on native Metal, MoltenVK and lavapipe. Where a driver lacks it, it must
  be refused rather than emulated silently.
- **Accuracy contracts:** every kernel states one, either bit for bit with a CPU reference or a stated bound. The
  tests check every element against it.
- **Performance claims:** they are measured by GPU time on named hardware and kept in the docs with the alternatives
  that lost.
- **Untestable code:** none is shipped. Code without a device here to run it, such as Vulkan cooperative matrices,
  waits.
