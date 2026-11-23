# Compute and Metal commission implementation record

Executed 2026-11-23 against the NeuralNetworks compute commission, linked GPU plan,
backend contract and architecture. All implementation changes are in DynamisGPU;
`AGENTS.md` is unchanged. The initial working tree was clean.

## Delivered runtime

G0 introduces the `api.compute` contract, shared lifecycle/validation SPI, asynchronous
mock, and reusable JUnit suite. G1 adds `dynamis-gpu-metal`, JDK 25 FFM, ABI-checked
opaque handles, exception-contained Objective-C bridge, discovery, shared/private
buffers, blits, pipeline/library loading, dispatch, polling and deferred release.
G2 adds real headless Vulkan compute discovery, storage descriptors, SPIR-V pipelines,
barriers, coherent/noncoherent host access, and submission fences. Both backends
run consecutive dependent compute operations through the same public control flow.

The benchmark module supplies `ComputeExampleMain` and `ComputeBenchmarkMain`.
[The runtime guide](compute-runtime.md) documents selection, shader/parameter ABI,
cache identity, capabilities, ownership, native flags, packaging, and adapter handoff.
[Captured evidence](evidence/compute-2026-11-23.txt) preserves verification summaries,
actual device diagnostics, example results, and unedited benchmark phase measurements.

## Baseline and build migration

Environment: Temurin **25.0.4+7**, Maven **3.10.2**, macOS **27.0.1**, arm64,
Apple **M4 Max**, Apple Clang **21.0.0**, LWJGL **3.4.1**, glslang **16.6.0**,
Vulkan SDK **1.4.341.0**. Vulkan reports vendor 4203, device 452985353,
API integer 4198734 and driver integer 10401. Native Metal bridge ABI is 1.

Before changes:

```sh
java -version
mvn clean verify
mvn -pl dynamis-gpu-vulkan -am test
```

The full baseline passed **165 tests**: API 118, mocks 1, Vulkan 43, benchmark 3.
The Vulkan-focused baseline also passed. The initial sandboxed Maven invocation
could not open its external dependency-cache lock; rerunning with permitted cache
access passed. The inherited parent already specified compiler release 25, while
this POM declared source/target 21. The revision uses one release property, **25**,
and enforces JDK 25+. No original graphics source signatures were changed.

Portable final verification: `mvn clean verify` passed **179 executed tests** and
explicitly skipped **26 hardware tests** (205 discovered). Surefire uses the class
path to avoid the JUnit launcher/module-path access failure encountered when
exposing the shared suite in the testing artifact; production modules still compile
as JPMS modules. Forked JVM native-access flags are configured in Maven.

## Native verification

These commands passed on the M4 Max:

```sh
mvn -pl dynamis-gpu-metal -am -Pmetal-native,metal-hardware test
export VK_DRIVER_FILES="$HOME/VulkanSDK/1.4.341.0/macOS/share/vulkan/icd.d/MoltenVK_icd.json"
export DYNAMIS_VULKAN_LOADER="$HOME/VulkanSDK/1.4.341.0/macOS/lib/libvulkan.1.4.341.dylib"
mvn -pl dynamis-gpu-vulkan -am -Pvulkan-hardware \
  -Ddynamisgpu.vk.loader="$DYNAMIS_VULKAN_LOADER" test
mvn -Pmetal-native,metal-hardware,vulkan-hardware \
  -Ddynamisgpu.vk.loader="$DYNAMIS_VULKAN_LOADER" install
mvn -Pmetal-native,metal-hardware,vulkan-hardware \
  -Ddynamisgpu.vk.loader="$DYNAMIS_VULKAN_LOADER" clean verify
sh scripts/run-compute.sh example metal 0 131
sh scripts/run-compute.sh example vulkan 0 131
sh scripts/build-compute-shaders.sh
```

The combined install and final clean verification passed **205 tests**, none skipped: API 118, testing artifact
15 (14 compute plus one original), Vulkan 56 (13 compute plus 43 original), Metal
13, benchmark 3. Each backend passed copy/fill, uploads/downloads, shared/private
storage, nonzero slice offsets, mapping visibility/exclusion, multiple submissions,
partial workgroups, two dependent dispatches, ownership after handle closure,
repeated allocation/release, invalid ranges/bindings/aliases, cross-device misuse,
unsupported features, and shader failures. The mock additionally injects bounded
wait timeout, pending mapping/cancellation, and terminal device loss to verify
failure reporting and resource retirement without provoking a hardware fault.

Both standalone examples verified **131 FP32 results** for `(a+b)+b` using a single
final readback. Metal diagnostics identify `METAL`, M4 Max, and **MSL_SOURCE**;
Vulkan identifies `VULKAN`, M4 Max, and **SPIRV**. Native Metal uses its own bridge
and Metal command queue, without LWJGL/Vulkan/MoltenVK. Vulkan execution uses
MoltenVK and a compute-capable queue with no presentation requirements.

The first Vulkan attempts correctly failed required tests: the default loader was
absent, then stale system ICD manifests referenced absent libraries. Explicitly
selecting the SDK loader and its matching manifest resolved initialization. Required
hardware profiles never silently skip an unavailable backend.

## Measured probe results

Commands, run sequentially, with 10 warmup iterations per phase and 100 measured:

```sh
sh scripts/run-compute.sh benchmark metal 262144 100
sh scripts/run-compute.sh benchmark vulkan 262144 100
```

Workload: 262,144 FP32 elements, 1,048,576 bytes per buffer, 64 threads per group.
These are **host-wall-clock microseconds**, not GPU timestamps. Completion includes
submission/driver/polling latency; transfer includes staging, native work and host
copies. Allocation includes release. Resident dispatch excludes host transfers and
allocation of the application buffers, but backend command/descriptor allocation
is part of submission. These short probes establish separable measurement paths;
they are not an optimized Metal/Vulkan performance comparison.

| Phase | Metal median / p95 (µs) | Vulkan median / p95 (µs) |
| --- | ---: | ---: |
| Allocation and release | 28.250 / 70.667 | 78.334 / 266.875 |
| Empty submission call | 12.041 / 32.083 | 28.792 / 71.667 |
| Upload/download complete | 850.334 / 966.583 | 996.375 / 1196.209 |
| Resident dispatch submit call | 51.416 / 126.333 | 92.250 / 182.625 |
| Resident dispatch complete | 184.209 / 372.584 | 358.375 / 453.125 |

## Platform and artifact coverage

| Platform/path | Build and execution status |
| --- | --- |
| macOS arm64, native Metal | Bridge built and embedded in JAR; real GPU suite/example/probe passed on macOS 27.0.1. Deployment target is 13.0; older OS versions remain untested. |
| macOS arm64, Vulkan/MoltenVK | Real headless compute suite/example/probe passed with the selected SDK. |
| macOS x86_64 | Native build path and LWJGL classifier configured; hardware/OS validation pending. |
| Linux x86_64 / arm64, Windows x86_64 | Java/provider and LWJGL classifier paths configured; builds/hardware validation on those systems pending. Portable builds do not invoke Apple tools. |
| Offline Metal executables | Build path implemented; artifact generation/loading remains unverified because Xcode/Metal tools are absent. |

`xcode-select -p` reports `/Library/Developer/CommandLineTools`; Xcode.app is absent.
`xcrun --find metal` fails. The bridge builds with Clang, and runtime MSL compilation
executes on native Metal. The explicit release prerequisite check:

```sh
DYNAMIS_METAL_REQUIRE_OFFLINE=1 sh dynamis-gpu-metal/src/main/native/build.sh
```

failed with exit 1 and `Offline Metal compiler required; install Xcode Metal
toolchain.` The profile is prepared to compile AIR/metallib and package it beside
the architecture-specific bridge when that external toolchain is installed. The
commission's **offline Metal artifact acceptance remains incomplete**, and the
commission must not be treated as fully verified across untested platforms.

## Material design choices and integration work

- Compute types are additive under `api.compute`, avoiding collisions with legacy
  buffer/mesh contracts. The graphics API remains a compatibility surface without
  a premature forced migration. All 165 original tests still pass, including the benchmark module consuming
  `meshforge-dynamisgpu:1.1.0` through the existing geometry contracts.
- The shared device/driver SPI resides in the dependency-free API artifact, avoiding
  another Maven module split. Device operations are explicitly thread-confined;
  multiple queues and concurrent recording are deferred until their semantics can
  be tested. Native libraries initialize only on explicit provider discovery.
- Compute has its own submission-scoped allocator/queue path. Existing graphics
  one-shot helpers wait for queue idle and cannot supply compute's asynchronous
  retention semantics without changing consumers; those helpers remain intact.
- Scalar layouts use explicit raw byte size and documented application-owned field
  offsets. Metal uses one dedicated buffer slot; Vulkan uses push constants. Vulkan
  specialization is supported; Metal function constants/LocalSizeId are actionable
  unsupported requests in this version.
- Conservative FP32/INT32/32-bit BOOL capabilities and host-clock profiling are
  delivered. GPU timestamps, extra numeric types/features, allocation pools,
  native pipeline caching, multiqueue scheduling, and optimized matrix extensions
  are later opt-in work, with capability and cache-identity hooks in place.
- Complete offline Metal packaging validation with Xcode, then validate any newly
  declared OS/architecture support. Real device-loss/timeout faults were not
  induced; deterministic injection is covered by the mock.

The NeuralNetworks handoff is the version-1 `api.compute` surface described in the
runtime guide. Its adapter can own an ordered context, retained tensor slices,
operation preflight, kernel bundles, scratch pooling, batching and host observation
boundaries immediately. NeuralNetworks owns G3 ML kernels, gradients/optimizer
validation against its CPU backend, and G4 numerical/performance tuning; no tensor,
autograd, Vectrix, or training dependency was introduced into DynamisGPU.
