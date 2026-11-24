# Headless compute runtime, contract 1

DynamisGPU provides devices, storage, transfers, ordered queues, shader pipelines,
bindings, dispatch, and completion independently of geometry uploads. Use JDK 25.
The runtime has no NeuralNetworks or Vectrix dependency. NeuralNetworks owns tensor
metadata, automatic differentiation, ML kernels, numerical policy, and its adapter.

## Modules and provider selection

- `dynamis-gpu-api`: dependency-free `api.compute` contracts and backend SPI.
- `dynamis-gpu-metal`: JDK FFM bindings and a native Metal C bridge, ABI 1.
- `dynamis-gpu-vulkan`: headless LWJGL compute alongside existing graphics services.
- `dynamis-gpu-test`: deterministic asynchronous mock, generic shaders, and shared
  JUnit 5 `ComputeConformance` suite. JUnit is a dependency of this testing artifact.
- `dynamis-gpu-bench`: common example and cost-separated benchmark probe.

Construct `MetalProvider` or `VulkanProvider`, call `adapters()`, inspect identity
and capabilities, and explicitly select an adapter before `open()`. Construction
alone does not initialize native backends. There is no automatic CPU fallback or
backend switching. Native Metal and Vulkan through MoltenVK are different providers.
Discovery failures carry backend diagnostics; a required hardware test fails if
there is no usable adapter.

## Build and hardware validation

Portable builds do not compile Metal or require GPUs. Hardware tests are explicitly
reported as skipped. Maven uses release 25 consistently, including inherited compiler
configuration; Surefire runs tests on the class path with native access enabled.

```sh
java -version
mvn clean verify
mvn -pl dynamis-gpu-vulkan -am test
mvn -pl dynamis-gpu-metal -am -Pmetal-native,metal-hardware test
mvn -pl dynamis-gpu-vulkan -am -Pvulkan-hardware test
```

On macOS install the Apple command-line tools. The native profile builds the bridge
with `xcrun clang`, Foundation and Metal, minimum deployment target macOS 13.0, for
the host's arm64 or x86_64 architecture. It embeds
`native/macos-<arch>/libdynamis_metal.dylib` in the Metal JAR. Loading checks ABI 1;
Java copies the embedded library to a temporary file before loading. Override with
`-Ddynamisgpu.metal.library=/absolute/path/libdynamis_metal.dylib` if needed.

When Xcode's Metal tools are installed, the same build compiles the test MSL to AIR
and packages `vector_add.metallib` beside the bridge. Tests/examples prefer this
binary; without that toolchain they explicitly use development-time `MSL_SOURCE`
compilation on the actual Metal device. Require offline compilation for release:

```sh
DYNAMIS_METAL_REQUIRE_OFFLINE=1 mvn -pl dynamis-gpu-metal -am -Pmetal-native package
sh scripts/build-compute-shaders.sh
```

The shader script regenerates the committed Vulkan SPIR-V using glslang and runs
`spirv-val` when installed. Ordinary builds consume the committed binary.
MSL and SPIR-V are separate artifacts; no shader translation is implicit.

For a Vulkan SDK installation on macOS, select its matching loader and driver;
stale manifests under `/usr/local/share/vulkan` may reference missing libraries:

```sh
export DYNAMIS_VULKAN_LOADER="$HOME/VulkanSDK/1.4.341.0/macOS/lib/libvulkan.1.4.341.dylib"
export VK_DRIVER_FILES="$HOME/VulkanSDK/1.4.341.0/macOS/share/vulkan/icd.d/MoltenVK_icd.json"
mvn -pl dynamis-gpu-vulkan -am -Pvulkan-hardware \
  -Ddynamisgpu.vk.loader="$DYNAMIS_VULKAN_LOADER" test
```

Standalone class-path applications need `--enable-native-access=ALL-UNNAMED`.
For named modules enable `org.dynamisengine.gpu.metal` for FFM and `org.lwjgl` for
LWJGL. No preview flag is needed by the runtime; the benchmark launcher also enables
preview/vector modules required by its existing MeshForge dependency.

## Common public example and benchmark

```sh
mvn -Pmetal-native install
sh scripts/run-compute.sh example metal 0 131
sh scripts/run-compute.sh example vulkan 0 131
sh scripts/run-compute.sh benchmark metal 262144 100
sh scripts/run-compute.sh benchmark vulkan 262144 100
```

`ComputeExampleMain` selects a provider/device, allocates shared and private buffers,
loads a kernel bundle/pipeline, uploads, binds, runs `(a+b)+b` in consecutive GPU
dispatches, awaits once, verifies final readback, and closes everything. Only the
provider and bundle executable differ between backends. The benchmark reports
backend/device/software/workload, warmup and iterations, allocation/release,
empty submission call, transfer round trip, resident dispatch submission, and
resident dispatch completion. Measurements use the host clock, include driver and
polling overhead, and are not GPU timestamp measurements or tuning claims.

## Shader ABI and capabilities

Each `KernelBundle` has a logical name/version and backend-specific immutable bytes.
`PipelineDescriptor` specifies the entry point, `BindingLayout`, fixed local size,
32-bit specialization values, and required types/features. Binding slots are explicit:
Vulkan uses storage buffers in descriptor set 0; Metal uses buffer indices. Every
bound slice declares a read, write, or read/write access and a minimum byte range.
Writable overlaps are rejected, including distinct slices of the same allocation.

Scalar parameters are a four-byte-aligned byte block of exactly `parameterBytes`.
Vulkan uses push constants at offset zero; Metal uses `setBytes` at one past the
highest buffer index (`parameterSlot()`). Scalar field offsets, widths and padding
are part of the application's kernel schema; both executables must implement that
same schema. Use little-endian FP32/INT32 and 32-bit BOOL masks for the initial
portable baseline. Vector-add slots 0/1 read FP32, slot 2 writes FP32, and its scalar
block is a single unsigned 32-bit logical element count. The kernel guards the final
partial group. Group counts are independent of the fixed local size (64,1,1 here).

Check `GpuCapabilities` before allocation and pipeline creation. Limits cover byte
sizes/storage ranges, binding offsets/counts, scalar parameters, local dimensions,
threads/group and group counts. Type/feature sets conservatively report what this
runtime enables.

**Capability tiers (1.2.0).** Beyond FP32, INT32 and BOOL, a device reports the optional data types and shader
features it supports. The backend enables them when it opens the device, and a pipeline lists the ones it uses in
`requiredDataTypes` and `requiredFeatures`.

| Tier | Meaning | Vulkan source | Metal |
|---|---|---|---|
| `FP16` | half arithmetic and storage | `shaderFloat16`, 16-bit storage (storage and uniform-and-storage) | always |
| `INT16`, `INT8` | 16- and 8-bit integer arithmetic and storage | `shaderInt16` / `shaderInt8`, 16- / 8-bit storage | always |
| `INT64` | 64-bit integers | `shaderInt64` | Apple7+, Mac2 |
| `FP64` | doubles | `shaderFloat64` | never |
| `SUBGROUP_BASIC`, `_VOTE`, `_ARITHMETIC`, `_BALLOT`, `_SHUFFLE` | subgroup (SIMD-group) operations in compute | `VkPhysicalDeviceSubgroupProperties` | Apple7+, Mac2 |
| `ATOMIC_FLOAT32_ADD` | float atomic add in storage buffers | `VK_EXT_shader_atomic_float` | Metal 3 |
| `ATOMIC_INT64` | 64-bit integer atomics in storage buffers | `shaderBufferInt64Atomics` | never (Metal has 64-bit min/max only) |
| `SUBGROUP_MATRIX_F16` (1.3.0) | hardware matrix multiply-accumulate, FP16 into FP32 | not yet (see below) | Apple7+ (`simdgroup_matrix`) |

- **Subgroup sizes:** `minSubgroupSize()` and `maxSubgroupSize()` bound the lane count a kernel can see. They are 0
  without subgroups. A kernel must work at any size in the range; MoltenVK reports 4 to 32.
- **Measured on our targets:**

  | Target | Reports |
  |---|---|
  | Apple M4 Max, native Metal | everything but `FP64` and `ATOMIC_INT64`; subgroups of 32 |
  | MoltenVK on the same machine | the same, with subgroups of 4 to 32 |
  | Mesa lavapipe | all 13 optional tiers; subgroups of 4 |
  | mock | none |

- **Undeclared needs are refused too.** On Vulkan, a SPIR-V module whose `OpCapability` list needs a feature the device
  did not enable fails with `UNSUPPORTED` even if its descriptor does not declare the tier. Without that check the
  driver's behaviour would be undefined. DynamisGPU never enables `StoragePushConstant16`/`8` or the subgroup
  shuffle-relative, clustered and quad operations, so modules using them are refused.
- **BF16** is not a tier: no portable API exposes it. Store it as 16-bit words and compute in FP32.
- **Conformance:** `CapabilityTierConformance` (dynamis-gpu-test) runs one Slang kernel per tier and checks its
  results exactly where the tier is reported, and checks pipeline refusal where it is not.
  - **Exact checks:** correctly rounded half and double add and multiply; wrapping 8-, 16- and 64-bit integers;
    subgroup reductions, scans, shuffles, ballots and votes; exact float and 64-bit atomic sums.
  - **Backends:** every backend's conformance test inherits it.
- **Matrix hardware (1.3.0):** `SUBGROUP_MATRIX_F16` reports Metal SIMD-group matrices on Apple7 and later GPUs.
  - **Writing the kernels:** they are written in MSL by hand, because Slang cannot emit Metal SIMD-group matrices.
    The conformance kernel is `compute/matrix/subgroup_matrix.metal`.
  - **Vulkan cooperative matrices:** Slang emits `CooperativeMatrixKHR`, but neither MoltenVK nor lavapipe supports
    `VK_KHR_cooperative_matrix`, so no device here can run such a kernel. They are left until there is NVIDIA, AMD or
    Intel hardware to test on.

**GPU timestamps (1.3.0).** `GpuSubmission.gpuTime()` gives the GPU execution time of a completed submission, from the
start of its first command to the end of its last. It excludes queueing and host work, so use it for kernel
throughput.
- **Where it's available:** devices whose `GpuCapabilities.timestamps()` is true.
  - **Metal:** always. It uses the command buffer's `GPUStartTime` and `GPUEndTime`, through bridge ABI 3.
  - **Vulkan:** where the compute queue family has timestamp bits. Each submission writes a timestamp at its start
    and end into its own query pool, and the time is scaled by `timestampPeriod`.
  - **Measured here:** native Metal, MoltenVK and lavapipe report it; the mock does not.
- **When it's empty:** while the submission is pending, when it failed before its commands ran, and without
  timestamps.
- **Example:** twenty 1M-element vector adds take 1.1 ms of GPU time on an M4 Max, against 8 to 10 ms of wall time
  around submit and wait.

**Measured features (1.1.0).** `GpuDevice.capabilities()` adds features the device can only report by running
code; the adapter's capabilities never include them.
- **`GpuCapabilities.FUSED_FMA`:** `fma` computes `a*b + c` with one rounding. The first call on the device's owner
  thread runs one probe dispatch (`fma(x, x, c)` with `c = -(x*x)` rounded on the host) and waits for it, and later
  calls are free. A failed probe reports the feature absent and logs a warning.
- **Where it holds:**
  - native Metal: yes;
  - MoltenVK on Apple GPUs: yes;
  - Mesa's lavapipe: no;
  - the mock: not reported.
- **Using it:** list it in `PipelineDescriptor.requiredFeatures` and pipeline creation fails with `UNSUPPORTED` where it
  is absent. Only such pipelines trigger the probe.

Metal additionally checks each
pipeline's maximum threads. Vulkan verifies the SPIR-V entry point and fixed
`LocalSize`; `LocalSizeId` is explicitly unsupported. Vulkan supports 32-bit
specialization constants; Metal ABI 1 rejects specialization requests and accepts
pre-specialized executables.

Floating point follows IEEE 754 by default, so a kernel computes what its source says: no reassociation, NaN and
infinities preserved, and a product rounded before it is added unless the source asks for a fused multiply-add. On
Metal and MoltenVK that holds for all arithmetic. On other Vulkan drivers it holds for operations the SPIR-V marks
`NoContraction` (`precise` in GLSL), because GLSL and SPIR-V let a driver reassociate and fuse unmarked arithmetic,
and Mesa does.
- **Metal runtime compilation (macOS 15+):** MSL source is compiled with `MTLMathModeSafe` and precise math
  functions, and with fp contraction off. `MTLMathModeSafe` alone still fuses `a*b + c`, so the bridge places
  `#pragma METAL fp contract(off)` before the source (with `#line 1`, so diagnostics keep their line numbers). A
  kernel can opt back in with its own `#pragma METAL fp contract(fast)`.
- **Metal before macOS 15:** only fast math is disabled.
- **Offline `metallib`s:** these keep whatever their compiler was told. Put the pragma in the source.
- **Vulkan:** the SPIR-V controls contraction and reassociation. Mark every operation whose rounding matters
  `NoContraction`, for example with `precise` in GLSL or by decorating the module after compilation; drivers must
  then round it as written.
- **MoltenVK:** it would otherwise compile every shader with Metal fast math, so the provider switches that off with
  `VK_EXT_layer_settings` (`MVK_CONFIG_FAST_MATH_ENABLED = 0`). Other drivers ignore the setting.

**Fused multiply-add.** Vulkan does not require `fma` to be fused: its precision rules treat it as a multiply
followed by an add, and lavapipe rounds twice. Metal's `fma` is fused. Check `GpuCapabilities.FUSED_FMA` before
relying on one rounding.

Mark `fma` calls `NoContraction` too when their rounding matters. Mesa otherwise rewrites expressions such as
`fma(k, a, fma(k, b, c))` into `fma(k, a + b, c)`. glslang does not mark `fma` for `precise`, so decorate the module
after compilation; vectrix-gpu's `build-kernels.py` shows how.

`ComputeConformance.floatingPointFollowsIeeeByDefault` checks these properties on each hardware backend: the
marked operations everywhere, and unmarked ones where the backend promises strict defaults (Metal and MoltenVK).
CI runs the Vulkan suite on Mesa's lavapipe, a CPU implementation of Vulkan. Results
can still differ between backends in built-in transcendental functions, in division on drivers that don't round it
correctly, and in subnormals, which Apple GPUs flush to zero. Applications own their numerical tolerances.

`PipelineDescriptor.cacheIdentity(backend, deviceId)` combines backend/device,
kernel name/version, executable SHA-256, entry point, layout, local size,
specialization, and required numerical/features options. No implicit cache is
installed. An application persisting native pipeline caches must also include its
runtime/driver/OS versions and numerical policy.

## Memory, synchronization and errors

A `BufferDescriptor` combines usage flags with shared/private placement. Check byte
ranges with long arithmetic; transfers use four-byte-aligned offsets and lengths.
Storage bindings use the device's reported offset alignment. Buffers must have a
positive allocation size; zero-element tensors should avoid dispatch and use an
adapter-owned padded allocation if a storage handle is needed. Host arrays and
mapped ByteBuffer views are limited by Java's integer indexing; larger allocations
can be transferred/mapped in slices.

**Memory pools (1.5.0).** Buffers are ranges of pooled blocks, so most allocations never reach the driver.
- **Blocks:** each block is one large backend buffer, a 64 MiB `VkBuffer` with its memory or an `MTLBuffer`, kept per
  placement. Buffers start at a multiple of 256 bytes (or the device's larger offset alignment).
- **Large buffers:** buffers over 16 MiB get a dedicated block, freed with them.
- **Reuse:** freed ranges merge with their neighbours, and one empty block is kept to absorb churn.
- **Staging:** uploads, downloads and fills use pooled shared staging, and the Metal bridge no longer creates a
  buffer per transfer.
- **Contents:** a new buffer's contents are undefined, because memory is reused. Fill or upload before reading.
- **Stats:** `GpuDevice.memoryStats()` reports the blocks held, the bytes used, the live allocations and the driver
  allocations made so far.
- **Budget:** it also reports the device's memory budget and the process's current usage (`headroomBytes()` is the
  difference).
  - **Vulkan:** `VK_EXT_memory_budget`, enabled where present, over the device-local heaps. Without it, the heap sizes
    and unknown usage.
  - **Metal:** `recommendedMaxWorkingSetSize` and `currentAllocatedSize`.
  - **Measured:** about 115 GB on an M4 Max, natively and through MoltenVK, and 14.9 GB on lavapipe in the
    container.
- **Measured on an M4 Max:**
  - 1,000 small buffers cost one driver allocation.
  - Allocating and closing a buffer went from 1,781 to 336 ns on native Metal, and from 9,908 to 782 ns on MoltenVK.
  - A small upload-and-download round trip is unchanged: its time is in submission and waiting, not allocation.
- **The allocator:** `MemoryPool` (spi) is backend-neutral. Every backend allocates through it, and each backend
  only creates and destroys its blocks.

**Queues (1.6.0).** `GpuDevice.queue(QueueKind.TRANSFER)` is a queue for uploads, downloads, copies and fills. It
rejects dispatches and recordings.
- **Which hardware queue:** `GpuQueue.dedicated()` says whether it runs on its own hardware queue.
  - **Native Metal:** a second command queue.
  - **Vulkan:** a dedicated transfer family where the device has one, as discrete GPUs do; buffers are then shared
    concurrently between the families.
  - **Otherwise:** transfers share the compute queue. That is the case on MoltenVK by default and on lavapipe.
- **Ordering:** within a queue it is automatic. Across queues, order work with fences, for example an upload that
  signals and a dispatch that waits.
- **Not used:** a second queue within the compute family, because no device available for testing exposes one.
- **Testing the dedicated Vulkan path:** MoltenVK runs it with `MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1`, which
  exposes a transfer-only family. CI runs the Vulkan suite that way too.
- **Measured overlap on an M4 Max:** a 64 MiB upload next to 40 large dispatches took about 16–18 ms either way. On
  unified memory the copy engine and the shaders share one memory system, and staging is a CPU copy in both cases.
  The gain belongs to discrete GPUs with DMA engines, which were not available for measurement.

**Timeline fences (1.6.0).** `GpuDevice.fence()` makes a `GpuFence`: a 64-bit value that only grows.
- **On the GPU:** `encoder.waitFor(fence, v)` holds a submission, before its first command, until the value reaches
  `v`; `encoder.signal(fence, v)` sets it once the commands complete.
- **On the host:** `value()` reads without blocking, `signal(v)` sets it, and `await(v, timeout)` blocks without
  polling.
- **Recordings:** `waitFor` and `signal` set the values for their next runs, because a timeline value is used once.
- **Backends:** Vulkan timeline semaphores (`timelineSemaphore`, enabled where supported); Metal `MTLSharedEvent`
  with an event listener for host waits (`dg_event_*`, bridge ABI 5). The mock has none (`UNSUPPORTED`).
- **Caution:** a submission that waits for a value nobody signals never runs. Closing the device then times out, so
  pair every wait with a signal.

**Observability for cost models (1.6.0).** These are host-side hooks, so reading them never waits for the device.
- **Labels:**
  - `encoder.label("decode")` labels the dispatches that follow.
  - `dispatch(..., "gemm.tiled")` gives one dispatch its own label.
  - The two join as `decode/gemm.tiled`.
  - Primitives label their dispatches (`gemm.tiled`, `gemm.simdgroup`, `softmax`).
  - `GpuSubmission.dispatchTimings()` returns (index, label, time) for profiled work.
- **Names and tags:**
  - `encoder.name("step")` names the submission or recording (`GpuSubmission.name()`, `GpuRecording.name()`).
  - `GpuRecording.tag(Object)` holds a caller's tag, such as a prefix-cache domain.
- **Memory classes:** `BufferDescriptor.withLabel("kv")` puts a buffer in a memory class, and
  `GpuDevice.memoryByLabel()` gives the live buffers and requested bytes per class.
- **Queue counters:** `GpuQueue.stats()` gives pending, submitted, completed and failed submissions, and the transfer
  bytes in flight. Reading them polls completions without blocking.
- **Debug labels:** labelled dispatches carry their label into GPU captures.
  - **Vulkan:** `VK_EXT_debug_utils` regions, where the loader offers the extension (the Vulkan SDK's loader does,
    with MoltenVK).
  - **Metal:** debug groups on the compute encoder (`dg_compute_label`, bridge ABI 5).
- **Use for cost accounting:** time labelled submissions or recordings with `gpuTime()`. Per-dispatch timing perturbs
  execution (see below), so it is a diagnostic.

**Per-dispatch timing (1.5.0).** `encoder.profile()` asks for one GPU time per dispatch, returned by
`GpuSubmission.dispatchTimes()` in recording order. A recording made from a profiled encoder times every run.
- **Vulkan:** a timestamp pair around each dispatch, in the submission's query pool.
- **Metal:** Apple GPUs sample counters only at encoder boundaries, so a profiled batch gives each dispatch its own
  encoder with start and end samples. GPU ticks convert to nanoseconds through the CPU clock (`sampleTimestamps` at
  commit and completion).
  - **Caveat:** an encoder's start sample is taken when it is scheduled. On an M4 Max, identical back-to-back 1M-element
    dispatches read 70, 165 and 221 µs on native Metal, where MoltenVK reads 60–70 µs each. So Metal's figures include
    queued time; use them to compare dispatches, not as exact durations.
- **Unprofiled work and devices without `timestamps()`:** an empty list.

**Pipeline cache (1.5.0).**
- **Vulkan:** every device creates its pipelines through a `VkPipelineCache`.
- **Persisting it:** with `-Ddynamisgpu.pipelineCache=<directory>`, the cache is loaded from and saved (atomically, on
  close) to a file per device and driver build. Data from another device or build is ignored.
- **Measured** on MoltenVK (M4 Max):
  - repeating a pipeline in the same process fell from 4.3 to about 1 ms;
  - three new pipelines in a later run fell from 21.6 to 13.3 ms.
- **Metal:** it has no archive yet. Creating a pipeline from MSL source measured 3.7 ms in every process, and 0.4 ms
  for a repeat in the same process. A binary archive would skip only the back-end compile; it is left until a
  workload shows the cost.

**Submission costs (1.5.0).** Measured on an M4 Max for one submission of 50 small dependent dispatches, in steady
state:

| | Record | Submit | Wait | GPU time |
|---|---|---|---|---|
| Native Metal | 18 µs | 49 µs | 320 µs | 195 µs |
| MoltenVK | 14 µs | 191 µs | 582 µs | 415 µs |

- **Blocking waits:** `await` blocks on the driver's completion signal instead of polling every 100 µs, which on macOS
  overshot.
  - **Vulkan:** `vkWaitForFences`.
  - **Metal:** a command-buffer completion handler and a semaphore, `dg_wait`, bridge ABI 4.
  - **Effect:** the wait beyond GPU time fell from 203 to 124 µs on Metal and from 205 to 167 µs on MoltenVK. A 4 KiB
    upload-and-download round trip fell from 155 to 103 µs on Metal and from 249 to 202 µs on MoltenVK.
- **Metal encoders:** consecutive dispatches share one serial compute encoder, which is ended before a blit and at
  commit. In serial dispatch each dispatch completes, with its writes visible, before the next begins.
  - **GPU time:** unchanged. About 4 µs per small dependent dispatch is the floor of a serial chain on Apple GPUs.
- **Vulkan barriers:** they cover compute and transfer stages, not all commands and the host. Host writes made before
  submission are visible through `vkQueueSubmit`.
  - **MoltenVK:** no change, since it ends a Metal encoder at every barrier, about 8 µs per dispatch.
- **Host cost:** on native Metal it is now about 67 µs of a 390 µs submission. On MoltenVK, submitting (descriptor
  pools, command buffers, fences per submission) is the largest host cost, which recorded replay addresses.

**Recordings (1.5.0).** `encoder.record()` ends an encoder as a `GpuRecording`, and `queue.submit(recording)` runs
it as often as needed. Between runs, `parameters(i, bytes)`, `bind(i, slot, buffer)` and `groups(i, dims)` change
dispatch `i`, each validated like the original dispatch.

```java
GpuRecording step;
try (var e = device.encoder()) {
    e.dispatch(attention, ...).dispatch(mlp, ...);   // a decode step
    step = e.record();
}
for (int token = 0; token < n; token++) {
    try (var m = input.map(Access.WRITE)) { m.bytes().put(nextToken()); }   // allowed between runs
    step.parameters(0, positionParameters(token));
    queue.submit(step).await(timeout);
}
step.close();
```

- **No readbacks:** a recording cannot contain downloads or error checks, because readbacks are single-use. Uploads
  replay the bytes they recorded.
- **One run at a time:** submitting, changing or closing a recording waits for its run in flight.
- **Lifetime:** a recording holds its buffers and pipelines until it is closed, even if their handles are closed.
  They count as in use, blocking host mapping, only while a run is in flight, and submitting while one of them is
  mapped is rejected.
- **Backends:**
  - **Vulkan:** each recording keeps its command buffer, descriptor sets, staging, fence and query pool, and
    re-records only after a change. A descriptor update invalidates command buffers, so rebinding re-records too.
  - **Metal:** command buffers are single-use, so Metal re-encodes the already-validated list.
- **Measured** (50 small dispatches):

  | Driver | Submit, re-encoding | Submit, replay | Encode, submit and wait | Replay and wait |
  |---|---|---|---|---|
  | lavapipe | ~200 µs | 30–35 µs | | |
  | MoltenVK | 177 µs | 119 µs | 762 µs | 665 µs |
  | native Metal | | | 419 µs | 375 µs |

  MoltenVK translates the command buffer to Metal on every submit, which limits it.

Devices and children are confined to the creating thread. Each encoder snapshots
upload/parameter bytes at recording time and retains allocations/pipelines.
Slices retain storage independently. Closing a handle is idempotent and invalidates
that handle; storage remains until the last slice, mapping, recorded use, and
submitted use is released. Submission consumes the encoder. Abandoning an encoder
releases its references and fails its readbacks. The device owns submitted work
until terminal completion, even when callers discard submission handles.

One compute queue orders all commands and submissions. Vulkan inserts conservative
transfer/compute/host memory barriers, uses a fence per submission, flushes or
invalidates whole mapped noncoherent allocations, and uses one queue family so no
cross-family ownership transfer is needed. Metal uses tracked resources and ends
blit/compute encoders between dependent operations. Command buffers and Java
retentions preserve native resources through asynchronous execution; upload data
is copied into retained staging. Neither backend waits for an entire queue after
every operation.

Shared mapping is exclusive: it rejects outstanding recorded/submitted GPU uses.
Read mappings are read-only. Close the mapping before recording GPU commands;
retained ByteBuffer references must never be accessed after mapping close or device
close. Metal shared memory still requires GPU completion. Private memory is not
mapped and uses explicit blit/staging transfers. Managed Metal storage is not exposed.

`status()` progresses completion and retirement in order. `await(Duration)` returns
false at its deadline and throws a `ComputeException` for execution failures.
Readback bytes become available after polling/awaiting the producing submission.
Timeout/interruption leaves ownership with the device. Submission/device close
waits at most 30 seconds per pending submission; a timeout keeps the device alive
for another drain attempt. A terminal queue error poisons later work, fails relevant
readbacks and is reported on device close after cleanup. Errors retain stable
categories and native operation diagnostics. No Objective-C exception crosses the
C ABI; bridge calls scope autorelease pools and copy thread-local error text into
Java immediately. Native handles are typed and stay inside backend implementations.

## Shader toolchain: Slang and KernelLibrary

Write kernels once in Slang, and build them with `dynamis-gpu-shaders` (`ShaderBuild`):
- **Outputs:** per kernel, `name.spv` (Vulkan) and `name.metal` (Metal source), plus a manifest recording each kernel's
  entry point, group size, buffer slots and parameter block. At runtime, `KernelLibrary` (dynamis-gpu-api) loads them
  from the class path as `KernelBundle`s and `PipelineDescriptor`s, so the Java side never restates the ABI.
- **Committed outputs:** they are committed, so neither the build nor CI needs slangc. A test recomputes the
  manifest's source hash, so a stale build fails.
- **Rounding rules:** the build keeps every rounding as written.
  - **Optimizer:** slangc runs at `-O0`, because its optimizer folds `(x + c) - c` to `x`.
  - **SPIR-V:** every float add, subtract and `fma` is marked `NoContraction`. Multiplies stay unmarked, because
    SPIRV-Cross turns a marked multiply into `fma(a, b, 0)`, which loses `-0`.
  - **Metal:** the source starts with `contract(off)`.
- **Binding order:** the build checks that Metal buffer indices follow the slots. Declare buffers in slot order, with
  the push-constant block last.
- **Running it:** `java -cp dynamis-gpu-shaders.jar org.dynamisengine.gpu.shaders.ShaderBuild --src DIR --out DIR
  --kernels FILE` (`--include DIR` and `--hash DIR` are repeatable). The kernels file has one
  `name source entry [targets]` line per kernel. `targets` is `spirv`, `metal` or both (the default). An FP64 kernel
  is `spirv` only, because Metal has no `double`. slangc is found through `$SLANGC`, `$VULKAN_SDK` or `~/VulkanSDK`.
- **DynamisGPU's own kernels** (`fma_probe`) and the capability-tier conformance kernels (`dynamis-gpu-test/src/main/slang`)
  are built by `scripts/build-slang-kernels.sh`.
- **Not converted:** the older conformance kernels (`vector_add`, `fp_strict`, `error_report`) stay hand-written GLSL
  and MSL, because they test what drivers do with specific SPIR-V, including unmarked arithmetic.
- **Equivalence:** the tool is a port of vectrix-gpu's `build-kernels.py`. On vectrix-gpu's 58 kernels its output is
  byte-identical.

## Kernel error buffers

Some arguments live in device memory, such as embedding indices, joint indices and data to be quantized, so the host
cannot check them before a dispatch. A kernel reports such a data-dependent error into a `KernelErrors` buffer, and
the submission fails with `ComputeException.Code.KERNEL_ERROR` instead of producing silently wrong results.

- **Buffer:** `KernelErrors.descriptor()` gives a device-private buffer of four 32-bit words.
  - word 0: the OR of the reported codes;
  - word 1: the complement of the smallest reported index;
  - word 2: the report count;
  - word 3: reserved.

  Each word is updated with an atomic chosen so the report does not depend on thread order. Codes are
  application-defined, nonzero bit masks.
- **Shader helpers:** `reportError(code, index)` ships as resources beside `KernelErrors` in `dynamis-gpu-api`:
  `shader/kernel_errors.glsl` (declare `kernelErrors[4]`, then `#include` it), `.metal` (prepend it) and `.slang`.
  The conformance suite runs the GLSL and Metal helpers on every hardware backend, and the Slang helper compiles to
  the same atomics.
- **Recording:** call `resetErrors(errors)` before the dispatches that report and `checkErrors(errors)` after them.
  Bind the buffer `READ_WRITE`.
- **On failure:**
  - **Scope:** only that submission fails. The queue is not poisoned, and the device stays usable.
  - **Results:** the submission's other readbacks stay readable, so results and a separate `download` of the error
    buffer can be inspected.
  - **The report:** `KernelErrors.parse` decodes the buffer, and the exception message carries the report.

## Migration and adapter handoff

Existing `api.buffer`, `api.gpu.VkCommandBuffer`, mesh upload/tickets, staging, and
Vulkan frame synchronization remain source-compatible. Existing geometry tests
exercise these surfaces. New compute contracts live in `api.compute` to avoid
collisions and avoid imposing new lifetimes on current graphics callers. Existing
one-shot graphics helpers wait for queue-idle and carry frame/graphics assumptions;
they were retained while compute uses submission-scoped allocation and synchronization.
A future graphics adapter can adopt this foundation without breaking that surface.

The NeuralNetworks adapter should hold one device/queue per context, express tensor
storage as retained slices, preflight operation requirements, supply its own
kernel bundles/layout schemas, batch dependent operations, and wait only at host
observation boundaries. It owns casting/layout/index validation, mutation rules,
autograd, kernel error buffers for data-dependent errors, scratch pooling and
numerical validation. G3 ML primitives and G4 tuning remain in that repository.

See [implementation evidence](compute-implementation-record.md) for measured results,
platform coverage, design decisions, and outstanding external validation.
