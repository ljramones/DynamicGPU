# Repository guidelines

DynamisGPU is the GPU layer of the Dynamis engine and of its throughput (neural-network) work. It provides:
- a backend-neutral compute runtime (`dynamis-gpu-api`, `compute` packages);
- backends for native Metal and Vulkan (MoltenVK on macOS, any Vulkan driver elsewhere);
- a Slang shader toolchain;
- capability tiers;
- throughput primitives;
- the older Vulkan-shaped graphics helpers (`api.buffer`, `api.gpu`, upload), which are to be consolidated later.

**Start here:**
- **The plan:** [docs/roadmap.md](docs/roadmap.md) lists the milestones in order, each with acceptance criteria
  and where to start.
- **The runtime contract:** [docs/compute-runtime.md](docs/compute-runtime.md) covers the ABI, capability tiers,
  timestamps, kernel errors and the shader toolchain.
- **The primitives:** [docs/primitives.md](docs/primitives.md) covers the kernels, their accuracy contracts and
  measurements.
- **Releases:** `docs/releases/<version>.md` has the notes for each release.

## Modules

| Module | Package | Holds |
|---|---|---|
| `dynamis-gpu-api` | `org.dynamisengine.gpu.api` | compute contracts (`compute`), SPI (`compute.spi`: `ComputeDevice` validates, `ComputeDriver` is what a backend implements), `KernelLibrary`, layouts (`layout`: `ShaderLayout`, `VertexLayout`), legacy graphics types |
| `dynamis-gpu-vulkan` | `org.dynamisengine.gpu.vulkan` | `compute.VulkanProvider`, `VulkanComputeDriver`, `VulkanFeatures` (tiers, enabled features, SPIR-V capability check); legacy Vulkan helpers |
| `dynamis-gpu-metal` | `org.dynamisengine.gpu.metal` | `MetalProvider`, `MetalDriver`, `MetalNative` (FFM binding), and the Objective-C bridge `src/main/native/dynamis_metal.m` (ABI 3) |
| `dynamis-gpu-shaders` | `org.dynamisengine.gpu.shaders` | `ShaderBuild`: Slang to SPIR-V + Metal with the rounding rules, plus a manifest (build-time tool) |
| `dynamis-gpu-primitives` | `org.dynamisengine.gpu.primitives` | `GemmF16`, `Softmax`, `AttentionF16`, `Precision` |
| `dynamis-gpu-vectrix` | `org.dynamisengine.gpu.vectrix` | GPU data written from Vectrix types (the only module depending on Vectrix) |
| `dynamis-gpu-test` | `org.dynamisengine.gpu.test` | the mock backend, and the conformance suites every backend inherits (`ComputeConformance`, `CapabilityTierConformance`) |
| `dynamis-gpu-bench` | | JMH benchmarks of the legacy upload paths |

## Environment

- **JDK and Maven:** JDK 25 and Maven 3.9.
  - **Development machine:** macOS on arm64 with an Apple GPU (an M4 Max so far).
  - **Parent POM:** the external parent `org.dynamisengine:dynamis-parent:1.0.1` comes from Maven Central.
  - **Vectrix:** `dynamis-gpu-vectrix` needs `org.dynamisengine:vectrix` (version `vectrix.version` in its POM) from
    the [Vectrix GitHub release](https://github.com/ljramones/vectrix/releases). Vectrix 3.x is not on Maven
    Central. CI installs it with `mvn install:install-file`.
- **Vulkan SDK:** it lives in `~/VulkanSDK/<version>/macOS` (1.4.341.0 at the time of writing).
  - **What it provides:** the MoltenVK driver, the loader, `spirv-val`, `spirv-dis` and `slangc` 2026.1.
  - **Which slangc:** use the SDK's `slangc`. `/usr/local/bin/slangc` on the development machine is broken, because
    a library is missing. `ShaderBuild` finds slangc through `--slangc`, `$SLANGC`, `$VULKAN_SDK`, `~/VulkanSDK`,
    then `PATH`.
- **Docker:** needed only for the lavapipe harness.
- **Xcode:** full Xcode is not installed, so there is no offline `metallib`. Metal kernels compile at run time from
  MSL source.

## Building and testing

| What | Command |
|---|---|
| Portable build (mock backend; hardware tests skip) | `mvn clean verify` |
| Native Metal hardware suites | `mvn -pl dynamis-gpu-metal,dynamis-gpu-primitives -am -Pmetal-native,metal-hardware test` |
| MoltenVK hardware suites | `SDK=~/VulkanSDK/1.4.341.0/macOS; VK_DRIVER_FILES=$SDK/share/vulkan/icd.d/MoltenVK_icd.json mvn -pl dynamis-gpu-vulkan,dynamis-gpu-primitives -am -Pvulkan-hardware -Ddynamisgpu.vk.loader=$SDK/lib/libvulkan.1.dylib test` |
| Mesa lavapipe (Linux, CPU Vulkan; what CI runs) | `sh scripts/test-lavapipe.sh` (Docker) |
| One test class | add `-Dtest=ClassName -Dsurefire.failIfNoSpecifiedTests=false` |
| Benchmarks | add `-Dgemm.benchmark=true` (GEMM, attention) or `-Dcompute.benchmark=true` (allocation, dispatch overhead, replay) to a hardware run; results print to the test report XML |
| Rebuild the Slang kernels (after editing any `.slang`) | `sh scripts/build-slang-kernels.sh`, then commit the regenerated resources |
| Regenerate the `ShaderLayout` glslang golden | `python3 scripts/gen-shader-layout-golden.py dynamis-gpu-api/src/test/resources/gpu/shader-layout-golden.txt` |

- **Profiles:**
  - **`metal-native`** builds the bridge dylib into `dynamis-gpu-metal/target/classes/native/macos-arm64`.
  - **`metal-hardware`** and **`vulkan-hardware`** set the `compute.metal.required` and `compute.vulkan.required`
    system properties that the hardware test classes are gated on.
- **Reading the output:** test output (capabilities, `worst error/bound`, benchmark rates) goes to stdout, which
  surefire keeps in `target/surefire-reports/TEST-*.xml`.

### CI (`.github/workflows/ci.yml`)

There are three jobs, all on every push:
- **Build:** the portable build. It installs Vectrix from its GitHub release first.
- **Vulkan conformance:** lavapipe on Ubuntu.
- **Apple GPU:** native Metal and MoltenVK, on the self-hosted runner `MacBook-Pro-2-metal`.
  - **Labels:** `metal` and `moltenvk`.
  - **Where it runs:** a launchd service in `~/actions-runner/dynamisgpu` on the development machine.
  - **Pushes only:** it never runs for pull requests, so code from forks can't reach the machine.

A new hardware test module must be added to the `-pl` lists of the hardware jobs.

## Writing kernels

- **Language:** Slang. Sources live in a module's `src/main/slang` and are listed in its `kernels.list`, one
  `name source entry [targets]` line each.
  - **Targets:** `targets` is `spirv`, `metal` or both (the default). FP64 kernels are `spirv` only.
  - **Building:** `scripts/build-slang-kernels.sh` builds each module's list into its resources, with a
    `kernels.properties` manifest.
  - **Loading:** `KernelLibrary.load(anchor, dir, manifest, prefix)` loads them and gives bundles, entry points,
    group sizes, buffer slots and the parameter-block layout.
- **ABI:**
  - Declare buffers with `[[vk::binding(i, 0)]]` in slot order, and the `[[vk::push_constant]]` parameter block
    last. Metal buffer indices then equal the slots, and the parameters take the next index, which `ShaderBuild`
    checks.
  - Slang numbers Metal buffers in declaration order, includes too. An include that declares buffers or the
    parameters must come after the buffers before it: the GEMM kernels declare A and B (`gemm_inputs.slang`), then C
    and the epilogue operands (`gemm_common.slang`), then any extra buffers, then the parameters
    (`gemm_epilogue.slang`).
  - Parameter blocks are 32-bit fields. Java writes them little-endian, in declaration order.
- **Variants:** share one body file. Each entry file does `#define OUT_T` / `#define ENTRY` and then
  `#include "x_body.slang"` (see `gemm_body.slang`). Slang `import` is not transitive.
  - **The GEMM family** has one four-line entry file per family, input and output type:
    `gemm_<family>[_bf16|_f32][_h].slang`, defining `IN_KIND` (0 FP16, 1 BF16, 2 FP32), `OUT_T`, `ENTRY` and, for
    the skinny vector build, `VEC`. The tiled family is `gemm_f16`, `gemm_bf16` and `gemm_f32`. `GemmF16.kernel`
    builds the names, and `GemmF16Test` checks that every one exists with the shared ABI.
- **What the build does for rounding:**
  - It compiles at `-O0`, because the optimizer folds `(x + c) - c`.
  - It marks SPIR-V `FAdd`, `FSub` and `Fma` `NoContraction`, but not `FMul`: SPIRV-Cross turns a marked multiply
    into `fma(a, b, 0)`, which loses -0.
  - It prepends the Metal prelude (`fp math_mode(safe)`, `fp contract(off)`).
  - Write `fma` explicitly where you want one.
- **Performance at -O0:**
  - Put `[ForceUnroll]` on every fixed-trip loop. Without it the SPIR-V keeps per-thread arrays in memory, and
    MoltenVK ran a GEMM 14 times slower.
  - Hoist uniform branches out of inner loops. Testing `transB` inside the k loop cost the matrix GEMM about 15 %.
- **Check the kernel build's exit status.** A failed `spirv-val` stops `build-slang-kernels.sh` partway, and the
  kernels after the failing one keep their old binaries: three rounds of 1.11 measurements ran stale kernels. For
  example, a `break` inside a `[ForceUnroll]` loop gave "ID ... does not dominate its use"; use a guard instead.
- **Metal limits on generated code:** a fully unrolled 128-step `fma` chain nests one expression past Metal's 256
  brackets. Unroll an inner block of 8 inside a plain loop.
- **Occupancy follows group memory:** the decode kernel ran 550 us with 16 KiB of scores per group and 396 us with
  4 KiB. One reduction tree for all rows, with 8 times the group memory, was slower than eight trees.
- **simdgroup_matrix kernels:** stage blocks that several SIMD-groups read in threadgroup memory, with 16-byte copies.
  Loading K and V from device memory, one product per load, made fused attention slower than the scalar kernel.
- **Group-wide work on one row** (the 1.10.1 sampler and log-sum-exp):
  - **Spread across groups:** one group per row leaves the GPU idle when there are few rows. Split the row into
    chunks, one group each, and combine the chunks in a second dispatch.
  - **Barriers cost:** a 1024-thread bitonic step costs about 1 us across 126 groups on an M4 Max. Sort only what
    must be sorted (runs, then merges), and size scans and trees to the live slots.
  - **Group-memory atomics on skewed keys contend:** a radix-select histogram of logits puts most keys in a few bins.
    A merge by binary search was faster.
- **Slang traps:**
  - **Groupshared rereads:** Slang's SPIR-V target may reread a local copy of a groupshared value after a barrier.
    Never copy a group variable and then overwrite it; give results their own group variable (`rowMax`, `rowSum`).
  - **Folding:** Slang folds `0.0f + v` even at -O0.
  - **No Metal forms:** Slang has no Metal `WaveActiveAllEqual`, no Metal `double`, and no Metal `CoopMat`.
    Cooperative matrices are SPIR-V only, with `using namespace linalg; CoopMat<...>`.
  - **Capabilities:** 16- and 8-bit buffers declare the `UniformAndStorageBuffer16/8BitAccess` capabilities.
  - **Constants:** build infinities from bits, for example `asfloat(0xFF800000u)`, not `1.0 / 0.0`.
  - **Specialization constants:** a `static const` computed from one (`static const bool QK = QFORMAT >= 3`) aborts
    the SPIR-V emitter ("Unhandled global inst"). Use a macro or a function.
- **Hand-written MSL** is for what Slang cannot emit, such as `simdgroup_matrix`. Put it under
  `src/main/resources/.../metal/`, use the same ABI as the Slang kernel, and build a `KernelBundle` with only a
  `METAL` executable (see `GemmF16.simdgroupBundle`).
- **Changing the Metal bridge ABI:** bump `dg_abi()` in `dynamis_metal.m`, `MetalNative.ABI` and the comment in
  `dynamis_metal.h` together. A stale dylib is then refused rather than crashing.

## Device facts that change results

- **Apple GPUs flush FP32 subnormals to zero.** Bounds need a `2^-126` floor; exact tests avoid subnormal results.
- **Metal fuses `a*b + c`** even in safe math mode, unless the source has `#pragma METAL fp contract(off)`. The
  bridge adds that pragma to all runtime-compiled MSL.
- **MoltenVK compiles with fast math unless told otherwise.** `VulkanProvider` turns it off through
  `VK_EXT_layer_settings`.
- **Mesa lavapipe:**
  - It does not fuse `fma` (`GpuCapabilities.FUSED_FMA` is absent).
  - It folds `float(half(s))` back to `s`, so a value rounded to FP16 in registers may not be. Where the rounded
    value matters (the norms' residual stream), round in integer arithmetic (`qF32ToF16` in `quant_common.slang`).
  - It reassociates unmarked arithmetic.
  - Mesa merges `fma(k, a, fma(k, b, c))` unless the `fma` is marked `NoContraction`.
- **Subgroups:** 32 lanes on native Metal, 4 to 32 on MoltenVK, 4 on lavapipe. Kernels must work at any size in
  `[minSubgroupSize, maxSubgroupSize]`.
- **Cooperative matrices:** neither MoltenVK nor lavapipe supports `VK_KHR_cooperative_matrix`. Don't ship a
  cooperative-matrix kernel until there is hardware to run it.
- **Tiers measured on each target:** see the table in `docs/compute-runtime.md`.

## Runtime traps

- **LWJGL's `MemoryStack` is 64 KiB per thread.** Allocate per-command structures in a frame per command
  (`try (var local = MemoryStack.stackPush())`), never in the frame that spans a whole command list: the Vulkan
  backend failed past about 1,170 dispatches per submission until 1.8.0 (`ComputeConformance.runsLongCommandLists`).
  The error is LWJGL's `OutOfMemoryError("Out of stack space.")`, which kills the surefire fork.
- **Don't index a vector by a runtime component** (`v[w & 3]` on a `uint4`) in a hot loop: the compiler spills the
  vector to memory. The first quantized decode kernel ran at 52 GB/s for that reason; read scalar words instead.
- **Size register arrays to what the dispatch needs.** A kernel that keeps 16 rows of accumulators for a one-row
  decode runs with few threads in flight. Bound the arrays with a specialization constant (`QROWS` in
  `gemm_q_skinny_body.slang`), and the driver drops the dead ones.
- **Measuring bandwidth:** warm the GPU up first (its clocks ramp up under load; the first shape measured ran 1.7
  times slow). Rotate through copies of the data larger than the system cache (256 MiB on an M4 Max is enough), or a
  repeated matrix is served from the cache, faster than memory.
  - **Round length:** size each timed round to about 20 ms of GPU work, after an untimed round. Rounds of 1 ms after
    host-side setup ran up to twice slow, and a busy desktop (WindowServer composites on the GPU) makes them noisier.
- **Read per-tile parameters with the tiles.** A load issued after a step's last barrier, and used at once, leaves
  every thread waiting on it: staging the quantized kernel's scales with the tiles took Q4K prefill from 64 ms to
  4.0.
- **Neighbouring lanes, different branches:** a branch that depends on the lane's data (the K scales' `j < 4`)
  diverges within a SIMD group. Compute both sides and select.
- **Writable bindings may not overlap another binding** ("Writable dispatch bindings overlap"). For an in-place
  update, bind the buffer once and read through the writable slot (`inPlace` in the norm and RoPE kernels), and give
  an unused writable slot its own scratch buffer rather than the shared read placeholder.
- **Near zero, compare absolutely:** the ULP distance from a tiny value to 0 counts every float in between (about
  850 million from 1.5e-8), so a bound with an absolute term must be checked as an absolute difference.
- **Benchmarks record many dispatches per submission.** Long command lists are a good stress test; keep
  `-Dgemm.shapes=true` and `-Dgemm.benchmark=true` running on MoltenVK as well as Metal.

## Testing conventions

- **Backend suites:**
  - `ComputeConformance` and `CapabilityTierConformance` (dynamis-gpu-test) are abstract JUnit classes.
    `MetalComputeTest`, `VulkanComputeTest` and `MockComputeTest` extend them, so every backend runs the same
    contract.
  - Primitives follow the same pattern: an abstract `XConformance` plus `MetalXTest` and `VulkanXTest`, gated with
    `@EnabledIfSystemProperty(named = "compute.metal.required" / "compute.vulkan.required", matches = "true")`.
- **Accuracy contracts:** every kernel has one, and the tests check every output element against it.
  - **Bit for bit with a CPU reference:** for example `GemmF16.reference`. Comparing the whole buffer also proves
    that padding and gaps were not written.
  - **Or a stated bound,** computed in double precision from the same inputs.
  - **Report:** print the worst `error / bound`.
- **Capabilities:**
  - A reported capability must be exercised, by a tier kernel with exact checks.
  - An unreported one must be refused: pipeline creation fails with `ComputeException.Code.UNSUPPORTED`.
- **Benchmarks:**
  - They are opt-in (`-Dgemm.benchmark=true`).
  - They measure GPU time (`GpuSubmission.gpuTime()`) over back-to-back dispatches in one submission, and print wall
    time too.
  - Record measured numbers in the docs, with the alternatives that lost.
- **Transfers:** uploads and downloads move whole 32-bit words, so pad FP16 arrays to an even length.

## Repository rules

- **Branches:** do work on a branch and push branches. The maintainer merges to `main` and publishes releases;
  agents do not push `main`.
- **Commits:** Conventional Commits (`feat(scope):`, `fix:`, `docs:`, `ci:`, `chore: release X`), in the imperative.
- **Don't run `mvn install` between releases.** The project version stays at the last release until the release
  commit, so installing overwrites the released artifacts in `~/.m2`. Releases are on GitHub, so a mistake can be
  repaired from the release zip.
- **Testing a consumer** (vectrix-gpu) against unreleased DynamisGPU: copy `~/.m2/repository` to a scratch
  directory, `mvn -Dmaven.repo.local=<copy> install -DskipTests` there, and test the consumer with the same
  `-Dmaven.repo.local`.
- **Maven Central is not used:** distribution is GitHub releases plus local installation. Don't run any deploy or
  Central profile (`-Prelease`).
- **Untestable code:** don't ship code that no available device can run.

## Releasing

1. **Branch** `release/X.Y.Z` from the finished work.
2. **Bump the version:**
   - the root POM `<version>` and `<scm><tag>`;
   - the parent version in every module POM. `dynamis-gpu-bench`'s `meshforge-dynamisgpu` dependency is external,
     so leave it alone.
   - the README "Release" section.
3. **Finish `docs/releases/X.Y.Z.md`** in the shape of the previous release. It needs:
   - New;
   - Behaviour changes to check;
   - Install and consume (every artifact);
   - Verification, with measured test counts per target.
4. **Verify:**
   - the portable build;
   - the Metal and MoltenVK hardware suites;
   - `scripts/test-lavapipe.sh`;
   - vectrix-gpu against the new version (`mvn -Pmetal,vulkan -Ddynamisgpu.version=X.Y.Z test` in
     `vectrix/vectrix-gpu`, with the MoltenVK environment), once the packaging step has installed it.
5. **Commit** `chore: release X.Y.Z`, then package. Run
   `VK_DRIVER_FILES=... sh scripts/package-release.sh -q -Ddynamisgpu.vk.loader=...`.
   - **Hardware suites:** it re-runs both and deploys to `target/release/repository`.
   - **Outputs:** `dynamisgpu-X.Y.Z-maven.zip`, `SHA256SUMS` and the jars.
   - **Local install:** it also installs X.Y.Z locally, which is correct for a new version.
6. **Push the branch** and wait for CI to pass.
7. **The maintainer publishes:**

   ```sh
   git switch main && git merge --ff-only release/X.Y.Z && git push origin main
   git tag vX.Y.Z && git push origin vX.Y.Z
   cd target/release && gh release create vX.Y.Z --title "DynamisGPU X.Y.Z" \
     --notes-file ../../docs/releases/X.Y.Z.md dynamisgpu-X.Y.Z-maven.zip SHA256SUMS *.jar
   ```

**Coupling with Vectrix** (`Layer1-Foundation/vectrix` in the Dynamis monorepo):
- **vectrix-gpu → DynamisGPU:** vectrix-gpu pins `dynamisgpu.version`, and Vectrix CI downloads that DynamisGPU
  bundle from its GitHub release. So a DynamisGPU release must be published before Vectrix can move to it.
- **DynamisGPU → Vectrix:** `dynamis-gpu-vectrix` pins `vectrix.version`.
- **Shared shader build:** vectrix-gpu builds its kernels with this repository's `ShaderBuild`
  (`vectrix-gpu/scripts/build-kernels.sh`).
- **Who does what:**
  - vectrix-gpu holds the exact (bit-for-bit) twins of Vectrix's CPU kernels.
  - DynamisGPU holds the throughput primitives and all reusable GPU plumbing.

## State at the time of writing (2026-11-24)

- **Published:** 1.0.x to 1.10.1 are on GitHub.
  - **1.5.0:** memory pools, blocking waits, recordings, the memory budget, per-dispatch timing and the Vulkan
    pipeline cache.
  - **1.6.0:** queues and synchronization, indirect dispatch, arenas and interop.
  - **1.7.0:** the kernel kit, and the `Elementwise`, `Reduce`, `Scan` and `NanCheck` primitives.
- **1.8.0:** the GEMM family.
- **1.9.0:** quantized weights.
- **1.9.1:** K formats, block 128 and GGUF import. `scripts/ggml-oracle.py` (with gguf-py) writes the GGML test
  samples.
- **1.10.0:** inference kernels (norms, RoPE, embedding, sampling, activations, log-sum-exp).
- **1.10.1:** fast sampling and log-sum-exp.
- **1.11.0 (attention at scale):** built on branch `feat/attention-at-scale`; see `docs/releases/1.11.0.md`.
- **Next:** milestone 1.12 (training kernels).
