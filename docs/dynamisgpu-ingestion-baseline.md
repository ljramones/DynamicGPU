# MeshForge -> DynamisGPU -> Vulkan Baseline (macOS)

## Environment Snapshot
- Machine: Apple M4 Max (user-reported)
- OS: macOS 26.3
- JDK: Temurin 25.0.1
- LWJGL: 3.4.1
- Vulkan SDK: 1.4.341.0
- Driver path: MoltenVK ICD forced via `VK_DRIVER_FILES`
- Vulkan loader path: `/usr/local/lib/libvulkan.1.dylib`

## Build Reference
- DynamisGPU commit: `bb5cf7e`
- MeshForge commit: `2603f06`
- Vectrix: not linked in this baseline run path
- Build state: local working tree (uncommitted benchmark/harness changes present)

## Vulkan Runtime Stack
- Loader: `/usr/local/lib/libvulkan.1.dylib`
- ICD: `MoltenVK_icd.json` via `VK_DRIVER_FILES`
- Backend: Metal via MoltenVK (Apple Silicon path)

## Environment Setup (macOS)
```bash
source "$HOME/VulkanSDK/1.4.341.0/setup-env.sh"
export VK_DRIVER_FILES="$HOME/VulkanSDK/1.4.341.0/macOS/share/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="/usr/local/lib:${DYLD_LIBRARY_PATH:-}"
```

## Benchmark Command
```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.MeshForgeVulkanIngestionMain \
  -Dexec.args="../MeshForge/fixtures/baseline" \
  -Dexec.classpathScope=runtime
```

## Diagnostic Probe Command
```bash
mvn -q -pl dynamis-gpu-bench exec:java \
  -Dexec.mainClass=org.dynamisengine.gpu.bench.ingest.meshforge.VulkanProbeMain \
  -Dexec.args="--debug" \
  -Dexec.classpathScope=runtime
```

## Fixture Results

| Fixture | Status | Source | Vertices | Indices | Stride | Index Type | Submeshes | Cache-Hit Load (ms) | Bridge (ms) | Upload (ms) | Total (ms) |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| RevitHouse.obj | PREUPLOAD_SUCCESS_UPLOAD_BLOCKED | CACHE | 1,242,180 | 1,236,357 | 16 | UINT32 | 1 | 7.623 | 2.410 | NA | 10.033 |
| lucy.obj | PREUPLOAD_SUCCESS_UPLOAD_BLOCKED | CACHE | 49,987 | 299,910 | 16 | UINT16 | 1 | 0.539 | 0.135 | NA | 0.674 |
| xyzrgb_dragon.obj | PREUPLOAD_SUCCESS_UPLOAD_BLOCKED | CACHE | 125,066 | 749,646 | 16 | UINT32 | 1 | 3.706 | 0.139 | NA | 3.845 |

Upload blocked reason in this run:
- `GpuException: stage=instance_create result=-9 requestedInstanceExtensions=[VK_KHR_portability_enumeration, VK_EXT_debug_utils]`

## Derived Metrics
- RevitHouse vertex buffer size: `1,242,180 * 16 = 19,874,880 bytes` (~18.95 MiB)
- lucy vertex buffer size: `49,987 * 16 = 799,792 bytes` (~0.76 MiB)
- xyzrgb_dragon vertex buffer size: `125,066 * 16 = 2,001,056 bytes` (~1.91 MiB)
- Upload throughput (indices/sec or MB/sec): `NA` in this run because upload execution is blocked in-process.

## Interpretation
- Cache-hit load is fast for all tested fixtures.
- Bridge conversion is cheap (sub-millisecond for `lucy` and `xyzrgb_dragon`, low single-digit ms for `RevitHouse`).
- Upload is expected to be the dominant ingestion cost once Vulkan device/runtime compatibility is available in-process; this run could not measure upload due runtime driver/device blocking.

Architectural takeaway: CPU-side ingestion (cache load + bridge conversion) is inexpensive; GPU upload is the intended dominant ingestion stage once runtime/device access is available.

## Observed Pipeline Characteristics
- MeshForge cache path performs well, including large assets.
- MeshForge payload -> `GpuGeometryUploadPlan` conversion remains a low-cost stage.
- Architecture is correctly isolating ingestion work so GPU upload becomes the dominant cost center once runtime/device access is available.

## Phase 1 Note
- `baseVertex=0` remains acceptable for the current single-submesh fixture set in this baseline.

## Future Work
- Measure full upload throughput once in-process Vulkan device enumeration is available.
- Evaluate staging-buffer reuse to reduce upload overhead.
- Expand fixture set with multi-submesh assets and validate non-zero `baseVertex` handling.

## First-Order Scaling Estimate (Engineering Planning)
This is a first-order estimate from current baseline behavior (not a hard guarantee).

Assumptions:
- Phase 1 path: CPU payload -> staging -> device-local upload
- Typical runtime layout near 16-byte vertex stride
- Index buffer included in total upload bytes
- Effective sustained upload planning anchor: ~1.0 GB/s (conservative)

| Total Upload Size | Estimated Upload Time | Planning Note |
|---:|---:|---|
| 5 MB | ~2–5 ms | Low impact load-time upload |
| 10 MB | ~5–10 ms | Still comfortable at load-time |
| 25 MB | ~20–25 ms | Noticeable ingestion cost |
| 50 MB | ~40–55 ms | Prefer async/background upload |
| 100 MB | ~85–120 ms | Streaming/chunking candidate |
| 250 MB | ~220–320 ms | Strong streaming requirement |
| 500 MB | ~450–650 ms | Large-asset residency/streaming territory |

### Vertex-Scale Heuristic
Rule of thumb under current packing assumptions:

| Vertex Count | Likely Total Upload Bytes | Estimated Upload Time |
|---:|---:|---:|
| 1M | ~20–30 MB | ~20–30 ms |
| 5M | ~100–150 MB | ~90–170 ms |
| 10M | ~200–300 MB | ~180–340 ms |
| 20M | ~400–600 MB | ~350–700 ms |
| 50M | ~1.0–1.5 GB | ~0.9–1.8 s |
