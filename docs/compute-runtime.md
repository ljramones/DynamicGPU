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
runtime enables. FP16, BF16, FP64, subgroups, atomics, optimized matrix paths, and
GPU timestamps are not advertised in contract 1. Metal additionally checks each
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
