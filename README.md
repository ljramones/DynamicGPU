# DynamisGPU

GPU utilities for the Dynamis ecosystem: persistent staging buffers, ring-buffer uploads, allocation-free `DeviceBuffer`, `ResourceHandle` lifecycle and caching, `DescriptorBuilder`, and indirect/compute helpers.

The library is backend-neutral (Vulkan/OpenGL via LWJGL), zero-copy focused, and multi-frame safe. It targets Java 21+ and serves as the foundation for particles, mesh streaming, and compute workloads.

## Java Baseline
- Java version: JDK 21+
- Base package for all modules: `org.dynamisgpu`

## Multi-Module Layout
This repository is intentionally limited to **four modules**:

1. `dynamis-gpu-api`
- Pure Java interfaces and value types
- No native dependencies
- Defines: `StagingBuffer`, `DeviceBuffer`, `IndirectCommandBuffer`, `BindlessHeap`, `GpuResourceHandle`, `GpuResourceLifetime`, enums, and exceptions
- Compile-time target for downstream libraries (for example, DynamisParticles)

2. `dynamis-gpu-vulkan`
- Vulkan implementations of API contracts
- Depends on `dynamis-gpu-api` and LWJGL Vulkan
- Runtime backend for Vulkan engine integrations

3. `dynamis-gpu-test`
- In-memory/mock API implementations for unit tests
- No GPU and no LWJGL required
- Shared test-scope dependency for downstream repos

4. `dynamis-gpu-bench`
- JMH benchmark suite
- Measures staging throughput, ring-buffer contention, resource lifecycle latency, and bindless allocation churn
- Depends on `dynamis-gpu-vulkan`
- Runs separately from the standard build

## Dependency Graph
```text
dynamis-gpu-api
       ↑
dynamis-gpu-vulkan    dynamis-gpu-test
       ↑                    ↑
dynamis-gpu-bench    (test scope in consumers)
```

## Parent Build Policy
The parent `pom.xml` owns shared dependency and plugin management:
- LWJGL BOM (kept in lockstep with DynamisLightEngine; currently `3.3.6`)
- JMH version
- Java 21+ enforcement
- Checkstyle
- `dynamis-gpu-bench` is opt-in via Maven profile: `mvn -Pbench test`

If work does not fit cleanly into one of these four modules, it likely belongs in DynamisLightEngine or a future dedicated library.
