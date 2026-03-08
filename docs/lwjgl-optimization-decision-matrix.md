# LWJGL Optimization Strategy for Dynamis

Dynamis uses LWJGL as the primary binding layer for Vulkan and related native APIs.

LWJGL already provides a thin, high-performance bridge to native code, so broad forking or rewriting the binding layer is not a goal.

Instead, Dynamis follows a four-tier decision strategy for performance optimization.

---

## Tier 1 - Leave Alone

Use this when profiling shows that the cost is dominated by:

- GPU execution
- Vulkan driver behavior
- command submission
- synchronization

Typical examples:

- raw Vulkan function entry points
- device/instance creation
- rarely executed code paths

Rule:

> If the hotspot is not in Java-side binding logic, do not optimize the binding.

---

## Tier 2 - Wrap

The default strategy.

Wrap LWJGL functionality inside Dynamis-owned helpers to gain:

- memory ownership control
- layout caching
- consistent fast-path usage
- future flexibility

Example packages:

```text
org.dynamisengine.gpu.nativefast
org.dynamisengine.gpu.memory
org.dynamisengine.gpu.vulkan.fastpath
```

Examples of wrapped areas:

- loader/bootstrap policy
- upload staging helpers
- struct/layout caching
- submission builders

Rule:

> Wrap first when the goal is control or caching rather than fixing LWJGL itself.

---

## Tier 3 - Patch LWJGL (Surgically)

Only do this when profiling proves a specific LWJGL helper path is a measurable hotspot.

Acceptable targets:

- small memory helper behavior
- bootstrap configuration behavior
- repeated struct marshalling paths

Avoid:

- altering generated bindings
- diverging broadly from upstream LWJGL

Rule:

> Patch only when the benefit is measurable and the patch is tiny.

---

## Tier 4 - Replace With FFM

JDK 25 includes the finalized Foreign Function & Memory API.

This allows writing native bindings without JNI.

However, Dynamis only uses FFM when:

- a hotspot is proven to be binding overhead
- the native call surface is small and stable
- a microbenchmark proves a win

Examples:

- repeated struct packing/unpacking
- small native kernels used at very high frequency

Rule:

> Use FFM as a scalpel, not a replacement for LWJGL.

---

## Current Status

Phase 3 of DynamisGPU introduced:

- persistent staging arenas
- device-local buffer pooling
- deferred upload submission
- sustained throughput benchmarks (~3.7-7.0 GB/s on Apple M4 Max)

Remaining optimization work will focus on:

- submission overlap
- struct/layout caching
- reduced temporary native allocations

Not rewriting LWJGL.

---

## Summary

Dynamis treats LWJGL as a stable outer shell.

Optimization effort focuses on:

1. architecture
2. memory lifetime
3. batching and submission
4. carefully measured binding overhead

rather than replacing the entire binding stack.
