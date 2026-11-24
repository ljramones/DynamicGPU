# GPU Data Contracts: Vectrix ↔ DynamisGPU, LightEngine, MeshForge, and Verified Defects

Status: **planned**. Vectrix 2.4.0 defines the contracts below; vectrix never calls a GPU API.

The ownership rule is `/Users/larrymitchell/Dynamis/docs/architecture/math-gpu-ownership.md`, section 3: vectrix
owns the layout math and encodings, and DynamisGPU binds them.

**Verified on 2026-11-22 against:**
- DynamisGPU `c5bd756` and MeshForge `8ea9354`.
- DynamisLightEngine `7017278e`, plus its uncommitted working tree (71 modified files of skinned-mesh debug work).

Every LightEngine finding below says whether it is in the committed code (**HEAD**), the uncommitted code (**WT**),
or both.

**How it was checked:**
- **Shader layouts:** the shader text, a Java text block, was compiled with glslang and the layout read back by
  spirv-cross reflection.
- **Pure Java logic:** run against the compiled classes, or as verbatim copies where `engine-impl-vulkan` could not
  be built (see defect 15).
- **No GPU was needed.**

## 1. What to adopt

| Today | Vectrix 2.4.0 |
|---|---|
| Hand-counted `putFloat`/`putInt` offsets for GlobalData (13 shader copies), ObjectData, MorphWeights, DrawMeta, MeshBounds and the culling push constants (LightEngine) | `gpu.ShaderLayout`: declare the struct once, then write at `block.offset("path")`. `toGlsl` emits the shader declaration, so the 13 hand-written copies become one generated string, and a contract test compares writer and shader. |
| Three vertex-layout types (vectrix, DynamisGPU `layout.VertexLayout`, MeshForge `pack.layout.VertexLayout`), plus LightEngine's hard-coded descriptors | Pick one owner (open decision). Vectrix proposes the format and offset math, and DynamisGPU maps formats to `VkFormat`. Decide together with the semantic → location table (defect 3). |
| Transform writers using the buffer's byte order | vectrix `GpuTransformLayout`/`GpuTransformWriteKernels` (always little-endian from 2.4.0); `AFFINE_3X4_ROW_MAJOR` = `VkTransformMatrixKHR` |
| Four frustum-plane packings | `FrustumPlanes.getVec4Array` (`vec4 planes[6]`); `FrustumPlanes.set(m, normalize, zZeroToOne = true)` for [0, 1] depth |
| Sphere × affine using the largest column length (`VulkanMeshBoundsBuffer`) | `BoundingVolumesf.transformSphere`: exact largest singular value, rounded up (the column shortcut under-covers under shear) |
| CPU AABB-versus-plane tests (`VulkanMeshletVisibilityCapability`, MeshForge `ViewFrustum`) | `geometry.CullingKernels.classifyAabb` |
| MeshForge meshlet cones (mean normal, cosine cutoff, no culling test) | `gpu.MeshletCones.normalCone` / `isBackfacing` / `quantize` |
| MeshForge's duplicate snorm8, Morton code and non-minimal bounding sphere | `PackedNorm`, `SpatialCodes.morton3d64`, `BoundingVolumesf.minimalSphere` |
| Gamma approximations written into `R8G8B8A8_SRGB` images | `PackedFormats.encodeSrgb8` / `decodeSrgb8` |
| Missing codecs | `PackedFormats`: 10_10_10_2, RGB9E5, R11G11B10F, QTangent |
| Camera depth-range formulas (defect 6) | `camera.DepthRange`, `Projections`, `ShadowCascades` |
| `normalize(mat3(model) * n)` (defect 11) | a normal matrix: the inverse transpose of the linear part |

## 2. Verified defects

### Data that reaches the GPU wrong

| # | Defect | Evidence | Status |
|---|---|---|---|
| 1 | **Meshlet bounds byte-swapped** (DynamisGPU). `GpuMeshletBoundsPayload.java:60`: `sourceBytes.duplicate()` always returns a big-endian buffer, so the little-endian branch at `:62` is dead, and every input is read big-endian and written little-endian. The 8 sibling payloads correctly use `.duplicate().order(sourceBytes.order())`. The test `preservesLittleEndianLayoutAndCounts` checks counts only. | `fromLittleEndianBytes` of little-endian floats 1..6 → `4.6006E-41 9.0E-44 2.3049E-41 …` (every float byte-swapped). | **Confirmed** |
| 2 | **Morph weights stride** (LightEngine, HEAD and WT). `VulkanMorphWeightUniforms.java:42-43, 89, 140-144` allocates, binds and packs 256 floats tightly (1024 B), and the bindless range is also 1024 (`VulkanSceneMeshLifecycle.java:577, 613`). All four morph shaders declare `uniform MorphWeights { float weights[256]; }`, which is std140 by default. | glslang + spirv-cross on HEAD and WT: `block_size 4096`, `weights` `array_stride 16` in every variant. Shader `weights[t]` reads host float `4t`: host `{0.25, 0.5, 0.75, 1.0}` is seen as `{0.25, 0, 0, 0}`, and `t ≥ 64` reads past the bound range. `ShaderLayout` gives the stride; alternatively use `vec4 weights[64]` or std430. | **Confirmed** |
| 3 | **MeshForge payloads decoded as all-float** (LightEngine, HEAD). `VulkanMeshAssetLoader.java:97-144` reads normals, UVs and tangents with `getFloat` whatever their format. `MeshForgeAssetService.java:106, 127` packs with `Packers.realtimeFast()` (SNORM8x4 normals and tangents, F16x2 UVs), and `:161` drops joints, weights and morphs (`skinned = false`). | Real MeshForge 1.1.0 packer + verbatim decode: normal (0, 0, 1) and uv (0.25, 0.5) come back as `nrm = 1.1663E-38, 1.7014E38, 3.06E-5`, `uv = 3.06E-5, 1.58E-41`. **Latent today:** only `.mgi` reaches this path. The default loaders handle obj/stl/ply/off but not glTF, OBJ fails at `Ops.tangents()` (`Missing required attribute: UV[0]`), and no `.mgi` assets exist. Separately, the bench adapter `MeshForgeRuntimePlanAdapter.java:49-63` assigns tangent = 2 and uv = 3 while the shaders expect uv = 2 and tangent = 3, but LightEngine does not use that adapter. | **Confirmed** (latent) |
| 4 | **Data textures uploaded as sRGB** (LightEngine, HEAD and WT). `VulkanTextureResourceOps.java:144, 179, 273, 320` always use `R8G8B8A8_SRGB`; the `normalMap` flag only picks the 1×1 fallback. This covers normal, metallic-roughness and occlusion maps (`VulkanSceneMeshLifecycle.java:151-159`), the IBL maps, and the **BRDF LUT** (`:98-100`). | A flat normal texel 0x80 decodes to 0.2159, and `*2 − 1` gives −0.568. Flat (0, 0, 1) becomes about (−0.57, −0.57, 1), a 39° tilt. | **Confirmed** |
| 5 | **Object matrix transposed** (LightEngine, **WT only**). `VulkanUniformWriters.java:270-280` writes `m[0], m[4], m[8], m[12], …`; HEAD (`:126`) writes it verbatim, which is correct. The data is column-major (`model.get(new float[16])`, `VulkanEngineRuntimeSceneAssembly:76`), as are GlobalData, the instance buffer, `VulkanMeshBoundsBuffer` and OpenGL (`transpose = false`). Also, the joint-palette doc (`engine-api EngineRuntime.java:109-110`) says "row-major", but the palette is uploaded verbatim into a column-major `mat4[]`. | `translation(2, 3, 4)`, shader `uModel * vec4(1, 0, 0, 1)`: WT gives (1, 0, 0, 3), HEAD gives (3, 3, 4, 1). | **Confirmed** (WT); the joint-palette doc is wrong in both |
| 9 | **Joint indices truncated at 255** (MeshForge `VertexWriteOps.java:93-96, 103-106`; LightEngine `VulkanGltfMeshParser.java` HEAD :345-350, WT :351-356). LightEngine also reinterprets the 4 joint bytes as a float (`Float.intBitsToFloat`). | Joint 300 is packed as 44. Joints (1, 0, 128, 127) form 0x7F800001, a signalling NaN. It survives today's copies bit-exactly, but any float arithmetic or double conversion could change it. | **Confirmed** (truncation); the reinterpretation is fragile |
| 11 | **Normals transformed by `mat3(model)`** in every Vulkan vertex shader and in the GL shaders (`GlShaderSources.java:36`), HEAD and WT. No inverse transpose or normal matrix exists anywhere. | Scale (1, 4, 1), normal (1, 1, 0)/√2: `mat3` gives (0.243, 0.970, 0), and the transformed tangent · normal = −15 (not perpendicular). The inverse transpose gives (0.970, 0.243, 0). | **Confirmed** |

### Depth range and culling

| # | Defect | Evidence | Status |
|---|---|---|---|
| 6 | **GL depth formulas with a [0, 1] projection.** WT switches the camera to `perspective(…, zZeroToOne = true)` with a Y flip (`VulkanEngineRuntimeCameraMath.java:44-45`); HEAD uses the GL [−1, 1] form. `VulkanMath.projectionNear` (`:31-41`) assumes [−1, 1] and feeds the cascade splits (`VulkanShadowMatrixBuilder:207-208`). The cascade split remap `farNdc * 0.5 + 0.5` (`VulkanShadowMatrixBuilder.java:257`) is compared with `gl_FragCoord.z`, but Vulkan applies no such remap. **That part is wrong in HEAD and WT.** | n = 0.1, f = 100. A [−1, 1] projection gives near 0.1000 / far 100.00 ✓. A [0, 1] projection gives near **0.0500** (`nf/(2f−n)`); the correct [0, 1] values are near = b/a, far = b/(a + 1). The splits move (7.745 vs 7.916). Cascade 0's split compares at 0.99404 instead of 0.98808, which is about 14.4 instead of 7.7 view units. `viewDistanceToNdcDepth` (`:43-50`) is correct. The culling near plane (`VulkanCullingComputePass.java:462`, row3 + row2) is right for HEAD. Under WT it is conservative (about 0.05 instead of 0.1): it culls too little, never wrongly. | **Partly confirmed:** the near/far error is WT; the split remap is both |
| 13 | **HEAD: a GL projection under Vulkan.** HEAD uses a [−1, 1] projection with no Y flip and no negative viewport. Vulkan clips z < 0, which with n = 0.1, f = 100 clips view depths below 2nf/(f + n) ≈ 0.2. WT addresses this but introduces defect 6. | Code + arithmetic | **Confirmed** (HEAD) |
| 7 | **Culling and indirect-command API mismatches.** (a) DynamisGPU: the API doc (`ComputeCullingDispatch.java:12`) says "6 x vec4", but `VulkanComputeCullingDispatch.java:22-23` requires exactly 6 floats; LightEngine never calls this implementation, since its own pass builds 24 floats of planes. (b) **The VFX draw path throws** (HEAD). `VulkanIndirectDrawBuffer.writeCommand` (`:157-158`) throws `UnsupportedOperationException`, and `VulkanVfxDrawContextAdapter.java:34` forwards DynamisVFX's `writeCommand` calls (`VulkanVfxIndirectWriter.java:23, 35, 44`) to it without a catch. | (a) 6 floats are accepted; 16 or 24 throw `frustumPlanes6 must contain exactly 6 values`. (b) Code. | **Partly confirmed:** (a) is an API inconsistency only; (b) is worse than suspected |

### Ray tracing, validation and loaders

| # | Defect | Evidence | Status |
|---|---|---|---|
| 8 | **Ray-tracing index type and bit fields** (DynamisGPU, latent). The BLAS hard-codes `VK_INDEX_TYPE_UINT32` and `firstIndex * 4` (`VulkanRayTracingBlasCapability.java:323, 343`); the build-input payloads carry no index type, while MeshForge emits UINT16 when it can. TLAS metadata (`GpuRayTracingTlasInstanceMetadata.java:56-67`) checks `customIndex`, `sbtOffset` and `flags` only for `≥ 0`. | With LWJGL: `customIndex` 0x01000005 reads back 0x5, `sbtOffset` 0x01000007 reads back 0x7, `flags` 0x1FF reads back 0xFF, all silently. | **Confirmed** (latent: nothing connects MeshForge index buffers to BLAS builds yet) |
| 10 | **Implicit byte order and weak layout validation.** LightEngine's Vulkan writers use `nativeOrder()` 16 times against 2 explicit little-endian uses; MeshForge and DynamisGPU force little-endian. All real targets are little-endian, so this is latent. DynamisGPU `layout/VertexLayout.java:24-35` checks only duplicate locations and `offset + size > stride`. | Accepted: overlapping attributes (legal in Vulkan), offset 1 with stride 13 (rejected by MoltenVK alignment rules), and **`offset = Integer.MAX_VALUE − 2`, because `offset + 12` overflows**. | **Partly confirmed:** the overflow is a real validation hole; byte order is latent |
| 12 | **MeshForge meshopt decoder** (`meshforge-loader/.../gltf/read/MeshoptDecoder.java`). It is wired only to `KHR_meshopt_compression` (`GltfBufferOps.java:81-111`); `EXT_meshopt_compression` is never handled. "NONE" is a plain LZ4 block decompress; "OCTAHEDRAL" decompresses LZ4, decodes two u16s via vectrix `OctaNormal` and always emits float3. Compatibility with real meshopt files is unverified. **Separate bug:** `:53` reuses a grow-only ThreadLocal scratch buffer, but the LZ4 decompressor requires the exact output size (`Lz4BlockDecompressor.java:86-89`). | On one thread: OCTAHEDRAL with 3 elements (12 B) works; then 2 elements (8 B) throws `LZ4 output size mismatch: expected 12 bytes, decoded 8 bytes`. | **Partly confirmed;** the scratch-buffer bug is confirmed |

### LightEngine build and shader state

| # | Defect | Evidence | Status |
|---|---|---|---|
| 14 | **Bindless vertex shaders don't compile** (HEAD and WT). Seven shader sources use `gl_DrawID` under `#version 450` with `GL_ARB_shader_draw_parameters`, where the name is `gl_DrawIDARB` (or use `#version 460`). The shadow pipeline catches the failure (`:119-123`), but `VulkanMainPipelineBuilder` (`:280-322`, rethrown at `:461-523`) fails main pipeline creation whenever a bindless layout exists. | glslangValidator 1.4.335 (vulkan1.1, the same target as `VulkanShaderCompiler:64`): `'gl_DrawID' : undeclared identifier`. | **Confirmed** |
| 15 | **engine-impl-vulkan imports a package that no longer exists** (HEAD and WT). 13 files import `org.vectrix.core.*`, and `module-info.java` has `requires org.vectrix`. No vectrix release (1.10.13–2.3.0) contains that package, which is now `org.dynamisengine.vectrix`, and no LightEngine POM declares vectrix at all. The module also requires `org.dynamisvfx.api/vulkan`, while DynamisVFX is `org.dynamisengine.vfx.api`. | Inspection of every local vectrix jar and all LightEngine POMs. A build was not run because offline dependencies (lwjgl-stb, jackson, ui-*) are missing. | **Confirmed by inspection** |
| 16 | **WT: the skinned-morph shader fails with skinning enabled.** With `DLE_VK_DEBUG_BYPASS_SKINNING=false`, it fails to compile: `'uModel' : no such field in structure 'obj'` (WT ObjectData is `vec4 c0..c10`). | glslang | **Confirmed** (WT) |
| 17 | **WT: debug flags default to true.** About 18 `DLE_VK_DEBUG_*` flags default to true, including bypassing skinning, flat colour, centring meshes, forcing direct draws, and per-object or per-frame `LOG.info`. | Code | **Confirmed** (WT; expected while debugging, but must not ship) |
| 18 | **Asset loading falls back to placeholders silently.** `VulkanMeshAssetLoader` swallows `MeshLoadException` (`:69-71`) and returns a placeholder triangle or quad for OBJ, STL and PLY (`:175-180`). | Code + the defect 3 reproduction | **Confirmed** (HEAD) |

Defects 1-12 come from the 2026-11-22 survey; 13-18 were found during verification.

**Fixed in DynamisGPU (branch `fix/defects-ci`, 2026-11-24):**
- **1:** meshlet bounds keep the source byte order.
- **7a:** the culling frustum is specified as 24 floats, `ComputeCullingDispatch.FRUSTUM_PLANE_FLOATS`, and
  enforced.
- **8:** BLAS build inputs carry an `IndexType`, and TLAS instance fields are checked against their 24-, 24- and
  8-bit Vulkan widths.
- **10:** `VertexLayout` computes an attribute's end in `long`.

Each fix has a test that failed before it.

## 3. Fixing order

1. **LightEngine build health (15, 14).** Fix the package names and the `gl_DrawIDARB` name; nothing else can be
   tested until the module builds and its pipelines compile.
2. **Wrong data on the GPU (2, 4, 1, 11).**
   - Morph stride: use `ShaderLayout`, `vec4[64]` or std430.
   - Linear (UNORM) formats for data textures and the BRDF LUT.
   - The one-line `.order(...)` fix for meshlet bounds.
   - A normal matrix.
3. **Depth range (6, 13).** Settle [0, 1] depth with vectrix `camera`, then fix `projectionNear/Far` and the split
   remap together.
4. **Before committing the WT: 5, 16 and 17.** Drop the transpose, make the morph shader match the new ObjectData,
   and default the debug flags off.
5. **Latent and validation issues (3, 7b, 8, 9, 10, 12, 18).**

Every fix should come with its reproduction as a regression test. For shader layouts, add a contract test that
compiles the shader text with glslang and compares the result with `ShaderLayout`.
