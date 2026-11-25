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

**Progress (branch `feat/runtime-memory`):**
- **Memory pools:** done.
- **Measurements changed the order.** In a 50-dispatch submission on an M4 Max, recording costs only 15–18 µs. The
  costs were polling waits (about 200 µs beyond GPU time), per-submission setup on MoltenVK (about 190 µs) and GPU
  time per dependent dispatch.
- **Done so far:** blocking waits, Metal encoder sharing, and narrower Vulkan barriers. The results are in
  `docs/compute-runtime.md`, under submission costs.
- **Replay:** done (`GpuRecording`).
  - **lavapipe:** submitting fell from about 200 to 30–35 µs.
  - **MoltenVK:** about a third, because it translates to Metal on every submit.
  - **Native Metal:** only the validation is saved.
- **Memory budget:** done (`MemoryStats.budgetBytes`, `deviceUsageBytes`, `headroomBytes()`).
- **Per-dispatch timing:** done (`profile()`, `dispatchTimes()`).
- **Pipeline cache:** done for Vulkan (`-Ddynamisgpu.pipelineCache`).
  - **Metal binary archives:** deferred. Metal pipeline creation measured 3.7 ms and is already cached in-process.
- **Milestone 1.5:** complete.
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
  - **Lifetime:** a recording retains every buffer and pipeline it uses until it is closed and its last submission
    has completed, as submissions do now. Closing the host-side `GpuBuffer` while a recording holds it is legal, and
    the storage stays until the recording lets go. Rebinding swaps what is retained.
  - **One submission at a time:** a recording is submitted at most once at a time, and submitting it again waits for
    the previous run. That keeps its descriptor sets and staging single-buffered.
  - **Acceptance:** replaying a 50-dispatch recording costs a small fraction of building it. The cost is measured
    and documented.
- **Memory budget:** `MemoryStats` gains the device's budget and current usage, from `VK_EXT_memory_budget` or the
  heap sizes on Vulkan, and `recommendedMaxWorkingSetSize` and `currentAllocatedSize` on Metal. Large-weight code can
  then check headroom before allocating.
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

**Status: complete, released as 1.6.0.** Done:
- labels, names, tags, memory classes, queue counters and backend debug labels;
- timeline fences (`GpuFence`);
- the transfer queue;
- indirect dispatch;
- transient arenas (`GpuArena`);
- interop (`GpuInterop`): native handles for devices, buffers and fences, and on Vulkan, recording into a caller's
  command buffer.

Not done, deliberately:
- **A same-family second Vulkan queue:** no available device could test it.
- **Metal external command buffers:** the bridge owns encoding. A Metal engine shares buffers and orders its work
  with the exported `MTLSharedEvent`.

**First, observability for cost models** (from the token-efficiency review, 2026-11-24). The runtime provides generic
hooks; serving vocabulary (prefill, decode, KV, tokens) stays in the layers above.
- **Labels:** a label on an encoder, a dispatch or a recording.
  - **Timing:** `dispatchTimes()` reports (label, duration).
  - **Debug labels:** labels become Vulkan debug-utils labels and Metal debug groups, so they appear in Xcode GPU
    captures and RenderDoc.
  - **Primitives:** they label their own dispatches by default (`gemm.tiled`, `gemm.skinny`, `softmax`, ...), and
    callers can override.
- **Memory classes:** a label on each allocation (`BufferDescriptor`), with `memoryStats()` broken down per label.
  For example, a serving layer labels KV pages and a game labels textures.
- **A user tag on `GpuRecording`,** for higher layers to map recordings to their own domains, such as a prefix cache.
- **Queue counters:** submissions pending, submitted and completed (running totals), and bytes in flight.
- **Rules for counters:**
  - All are host-side bookkeeping, readable without a device sync or queue drain.
  - Unread, they cost nothing measurable.
- **Cost accounting uses labelled submissions or recordings timed with `gpuTime()`.** Per-dispatch timing perturbs
  execution (on Metal it splits encoders and includes queued time), so it stays an opt-in diagnostic.

- **Multiple queues:** a compute queue and a transfer queue where the device has them (Vulkan queue families; Metal
  command queues). `GpuDevice.queue(QueueKind)` selects one.
- **Cross-queue dependencies:** timeline semaphores on Vulkan, `MTLSharedEvent` on Metal.
  - **API:** a `GpuFence` that one submission signals and another waits on, without the host.
- **Async transfer:** uploads of the next batch overlap compute of the current one.
- **Indirect dispatch:** group counts read from a GPU buffer.
  - **Use:** GPU-driven work whose size is only known on the GPU: culling, compaction, mixture-of-experts token
    routing.
  - **Backends:** `vkCmdDispatchIndirect`, and Metal `dispatchThreadgroupsWithIndirectBuffer`.
- **Memory II, residency and aliasing:**
  - **Residency** stays explicit through `Placement`: `PRIVATE` is device-resident (weights, KV pages, game resources
    kept across frames), and `SHARED` is host-visible (staging, readbacks).
  - **Transient arenas** are a scoped allocator whose buffers alias the same memory across scopes whose lifetimes
    don't overlap. Examples are an attention layer's scratch reused by the next layer, or a frame's temporaries.
    Aliasing is safe only between scopes ordered by the queue, and the arena checks that ordering. It is deliberate,
    never implicit.
  - **Lifetime across queues:** a buffer used by submissions on several queues is held until all of them complete.
- **Interop with the engine (sketch, for game compute before the graphics breakup):**
  - **Native handles:** export a buffer as `(VkBuffer, offset, size)` or `(MTLBuffer, offset, size)`, so
    LightEngine's existing Vulkan and Metal code can bind DynamisGPU buffers. With pooling, the offset is not zero.
  - **Vulkan, external command buffers:** record a DynamisGPU compute encoder into a command buffer the caller owns,
    so compute runs inside a frame's command stream.
  - **Synchronizing with graphics:** the shared fences from cross-queue dependencies, so compute and graphics
    submissions order each other without the host.
- **Acceptance:**
  - A stream of uploads interleaved with dependent compute is correct on every backend.
  - On hardware with a separate transfer engine, uploads overlap compute; the overlap is measured.

- **Demotion (for example of KV pages under memory pressure) is an ordinary transfer:** a copy into `SHARED` memory
  on the transfer queue, ordered by a `GpuFence`. On Apple's unified memory, `SHARED` already is the CPU tier. No
  special runtime operation is added.
- **Where to start:**
  - **Queue selection:** `VulkanProvider.family` picks the first compute-capable family. Add a transfer-only family
    where one exists.
  - **The queue model:** `GpuDevice.queue()` returns the single `GpuQueue`, and `ComputeDevice` keeps one pending
    list.
  - **Metal:** `DGDevice` holds one `MTLCommandQueue`; shared events need the bridge (ABI bump).
  - **Test devices:** lavapipe exposes one queue family, so its tests cover the single-queue fallback, and Apple GPUs
    test real queues.

### 1.7: Kernel kit

**Status: complete, released as 1.7.0.** Done:
- the Java kit (`ParameterLayout`, `StridedView`, `Grid`, `PipelineSet`, `Checks`);
- specialization on Metal (function constants, bridge ABI 6);
- device and pipeline limits;
- the Slang include library;
- `AccuracyContract`, with contracts for every primitive;
- `Elementwise`, `Reduce`, `Scan` and `NanCheck` with tripwires;
- debug builds (`DG_DEBUG`);
- reproducible mode (`KitOptions`);
- the cross-backend harness (`CrossBackend`);
- `ShapeRegistry`, with `GemmF16` dispatching per shape from a measured rule.

Not done, deliberately:
- **Moving vectrix-gpu's helpers onto the kit:** that is the Vectrix flow, after the stack.
- **A shader printf:** there is no portable channel for it, so the error buffer remains the channel.

- **Shared plumbing:**
  - **Parameter-block builder:** fields named and checked against the manifest.
  - **Grids:** linear-to-2D grids that respect the group-count limits.
  - **Checks:** 32-bit index limits, and extent, matrix and SoA checks.
  - **Strided views as the norm:** a view type with a base offset, extents and per-dimension strides (in elements).
    - **Shapes it covers:** dense, sliced, transposed, batched and interleaved or SoA layouts alike.
    - **Checks:** one bounds check, on the largest index the view can reach.
    - **Parameters:** parameter-block helpers that write a view's fields.
    - **Rule:** new primitives take views; none re-implements stride arithmetic. A Slang header gives the matching
      index arithmetic. This matters most for game data, which is often packed or interleaved.
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
- **Resource limits for the heuristics:**
  - **Device limits:** report what the shape registry and tile choices need. That is the maximum group-shared memory
    (Vulkan `maxComputeSharedMemorySize`, Metal `maxThreadgroupMemoryLength`), the maximum threads per group, and the
    subgroup size range.
  - **Pipeline limits:** report what the compiled pipeline allows (Metal `maxTotalThreadsPerThreadgroup` and
    `threadExecutionWidth`, which reflect register pressure; on Vulkan, the reflected group-shared usage).
  - **Tests:** check that every registered variant respects them on each backend.
- **Elementwise operations and reductions, early:** they are the building blocks that epilogues, norms and attention
  share, so they come before the GEMM family.
  - **Elementwise maps** (unary and binary, with broadcasting over views).
  - **Fixed-order reductions** (sum, max, min, argmax) along an axis, and **scans**.
  - **A NaN and infinity check.**
  - Each is a public primitive and a Slang include. 1.10 keeps the model-specific kernels (norms, RoPE, sampling).
- **Validation and debugging:**
  - **A debug build of each kernel family**, selected with a system property. It adds bounds checks that report
    through the existing kernel error buffers (`KernelErrors`), and is compiled out normally.
  - **A reproducible mode** that forces the fixed-order algorithms (for example `GemmF16.Algorithm.TILED`) everywhere.
  - **NaN tripwires:** the NaN check can be armed after chosen dispatches.
  - **A cross-backend harness:** run one operation on two devices, for example native Metal and MoltenVK, and
    report the first differing element against the operation's contract.
  - **No printf:** shader printf is only available through Vulkan validation layers, so the error buffer is the
    portable channel.
- **Fusion and attribution:** timing granularity is the dispatch, so a fused kernel declares what it fuses in its
  label.
  - **Fusion stays allowed:** mixing prefill and decode tokens in one GEMM (continuous batching, chunked prefill) is a
    major serving win.
  - **Splitting the cost:** the serving layer does it by token count or FLOPs; kernels don't avoid fusion for the
    sake of accounting.
- **Accuracy contracts as API:** every primitive publishes its contract as data. That is the input types, the
  accumulation type, the output rounding, the order class (fixed and reproducible, or hardware order) and the error
  bound, through an `AccuracyContract` record next to the documentation table. NNJava can then compare GPU results
  (gradients included) with CPU references, using the stated tolerance instead of guessed ones.

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

**Status: complete, released as 1.8.0.** Done:
- **Skinny GEMM** (`Algorithm.SKINNY`, `m <= 16`): a fixed split order, bit for bit `referenceSkinny` on every
  device. At `m = 1` with B transposed it streams 378-408 GB/s on an M4 Max, 77-96 % of the measured streaming-read
  bandwidth (`-Dgemm.decode=true` reports both), about 7 times the tiled kernel.
- **Small multiplies** (`Algorithm.SMALL`): 16 x 16 tiles, bit for bit `reference`. On batches of 16 x 16 and
  32 x 32 multiplies it takes 0.26 and 0.62 of TILED's time.
- **Grouped GEMM** (`encodeGrouped`): a `Problem` table the GPU may write, tiled per problem, bit for bit
  `referenceGrouped`; out-of-range problems fail the submission.
- **Transposed A** (`Shape.transposeA`) on every algorithm but SKINNY.
- **Fused epilogues** (`Epilogue`): bias per column or row, ReLU, GELU (tanh form), SiLU, gate (SwiGLU) and
  residual, in FP32 before the output rounding. The exact ones keep the fixed-order kernels bit for bit.
- **Inputs** (`Input`): FP16, BF16 and FP32 on every family; FP32 bit for bit where the device fuses fma. The Metal
  matrix kernel takes FP32 too.
- **Accumulation policy:** every contract states its accumulation (FP32 for every input) and order class.
- **Column-major:** `Shape.columnMajor` maps it onto `C^T = op(B)^T op(A)^T`.
- **Shape dispatch:** the registry's costs are estimated GPU times (`GemmF16.Costs`). On all 46 measured shapes
  they pick the fastest algorithm or one within 1.2 times of it.
- **Found and fixed:** the Vulkan backend failed past about 1,170 dispatches per submission (a thread-stack frame
  per submission); now a frame per dispatch, with a conformance test.

Not done:
- **Vulkan cooperative matrices:** no NVIDIA, AMD or Intel hardware to test on.
- **A fast FP32 matrix kernel:** Metal's FP32 SIMD-group matrices ran at only 2.7 to 3.0 TFLOP/s at 4096^3, against
  the tiled kernel's 5.3. Shape dispatch leaves them out for FP32 inputs.
- **The matrix kernel** has no BF16 entry (Metal's `bfloat` matrices need a newer language version) and no grouped
  form; grouped problems run the tiled kernel.

The plan as agreed:

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
- **Accumulation policy:**
  - Every form states its accumulation type and order class in its `AccuracyContract`.
  - FP32 accumulation is the default for FP16, BF16 and quantized inputs.
  - Any narrower accumulation is a separate, explicitly named variant, never a silent choice.
  - Fused epilogues state where their rounding happens: in FP32, before the output rounding.
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

**Status: complete, released as 1.9.0; extended in 1.9.1.** Done:
- **Formats** (`QuantizedMatrix`): Q8, Q4 and Q4M (the meanings of Q8_0, Q4_0 and Q4_1), blocks of 32 and 64, FP16
  parameters, with the CPU encoder and decoder.
- **Quantize and dequantize kernels** (`Quantize`): bit for bit the CPU's on every device. Scales come from one
  correctly rounded multiply, and codes from exact midpoint comparisons.
- **The quantized GEMM** (`QuantizedGemm`): FP16 or BF16 activations, dequantized in the kernel, for decode
  (`SKINNY`) and prefill (`TILED`), with the epilogues. It is bit for bit `QuantizedGemm.reference` on every device,
  lavapipe included: activation-code products are exact.
- **Measured:** decode at m = 1 with Q4 weights streamed from memory runs 2.9 to 4.0 times faster than FP16 weights on
  an M4 Max (Q8 1.9 to 2.4).
- **Packing:** the packed layout is the format's (`QuantizedMatrix.quantize`, or `Quantize.pack` on the GPU), made
  once at model load.

Differences from the plan:
- **The CPU formats live in DynamisGPU** (`QuantizedMatrix`), not Vectrix `nn.Formats`. The primitives do not depend
  on Vectrix, and putting them there would have needed a Vectrix release first. Vectrix can adopt them in the
  Vectrix flow; its int8 format (FP32 scales) is a different format.
- **No FP16 tile packing:** the matrix kernel reads row-major `W^T` at full rate (12.7 TFLOP/s), so a tiled layout
  has nothing to gain.

Not done:
- **A quantized matrix kernel** for prefill: `TILED` runs about 4.6 TFLOP/s against the FP16 matrix kernel's 12.7.

**1.9.1** (agreed after 1.9.0, from a survey of the local models: most GGUF models ship as Q4_K with Q6_K):
- **K formats** `Q4K`, `Q5K`, `Q6K`: llama.cpp's dequantization bit for bit, in the quantized GEMM and `Quantize.unpack`.
- **Block size 128** for Q8, Q4 and Q4M (the group size of GPTQ and AWQ models).
- **`GgufBlocks`:** GGML Q4_0, Q4_1, Q8_0, Q4_K, Q5_K and Q6_K to and from `QuantizedMatrix`, without loss. Checked
  against gguf-py on synthetic blocks and on real models.
- **Not included:** Q5_0 and Q5_1 (seen in one local model), the IQ and 2- to 3-bit formats, FP8 weights, and a GGUF
  container reader (left to NNJava).

The plan as agreed:

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
  - logits never return to the host. The sampled token stays in a GPU buffer, a host readback is optional, and the
    next step's recording reads it directly.
- **Reductions and scans:** these arrive in 1.7; here they gain model-oriented fused forms such as a log-sum-exp.

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
  - **Sharing:** pages are reference-counted and shared across sequences, with copy-on-write. That is the mechanism
    prefix caching needs. The prefix index itself (hashing token IDs, a radix tree) belongs to the serving layer.
  - **Facts, readable without a sync:** pages and tokens per sequence, pinned (active) against evictable, and total
    against free. Recompute cost depends on the model and the hardware, so the serving layer's cost model derives it
    from these facts.
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

## Review notes (2026-11-24)

An independent review of this roadmap raised seven points, all adopted.

| Point | Where it went |
|---|---|
| Strided, offset-rich views | 1.7 (kit) |
| Residency and aliasing | 1.5 (recording lifetime, budget), 1.6 (arenas) |
| Validation and debugging | 1.7 |
| Elementwise operations and reductions earlier | 1.7 |
| Resource limits | 1.7 |
| Accumulation policy as contract | 1.7 (`AccuracyContract`), 1.8 (per form) |
| Coexistence with game compute | 1.6 (indirect dispatch, native handles, external command buffers, shared fences) |

## Rules that hold throughout

- **Counters and queries:** every counter or query meant for cost or admission decisions is readable on the host
  without a device sync or queue drain. It costs nothing measurable when unread. GPU-side figures arrive through an
  asynchronous readback, never a blocking read. Anything that perturbs execution is opt-in and documented as such.

- **Testing:** every capability is tested on native Metal, MoltenVK and lavapipe. Where a driver lacks it, it must
  be refused rather than emulated silently.
- **Accuracy contracts:** every kernel states one, either bit for bit with a CPU reference or a stated bound. The
  tests check every element against it.
- **Performance claims:** they are measured by GPU time on named hardware and kept in the docs with the alternatives
  that lost.
- **Untestable code:** none is shipped. Code without a device here to run it, such as Vulkan cooperative matrices,
  waits.
