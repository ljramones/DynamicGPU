# GPU data layouts

How CPU data is laid out in GPU buffers: interface blocks, vertex inputs, and instance transforms written from
Vectrix types. These moved from Vectrix (`org.dynamisengine.vectrix.gpu`) in DynamisGPU 1.2.0. Vectrix keeps the
numeric encodings they use: `Half`, `PackedNorm`, `OctaNormal`, `QuatCompression` and `PackedAffineArray`.

| Job | Class | Module |
|---|---|---|
| Lay out a uniform or storage block | `org.dynamisengine.gpu.api.layout.ShaderLayout` | `dynamis-gpu-api` |
| Describe a pipeline's vertex input | `org.dynamisengine.gpu.api.layout.VertexLayout`, `VertexAttribute`, `VertexFormat` | `dynamis-gpu-api` |
| Write instance transforms as TRS structs | `org.dynamisengine.gpu.vectrix.GpuTransformLayout` | `dynamis-gpu-vectrix` |
| Write or expand packed 3x4 affines | `org.dynamisengine.gpu.vectrix.GpuTransformWriteKernels` | `dynamis-gpu-vectrix` |
| Run the per-frame instance path | `org.dynamisengine.gpu.vectrix.InstanceSubmissionPipeline` | `dynamis-gpu-vectrix` |

`dynamis-gpu-vectrix` is the only DynamisGPU module that depends on Vectrix; `dynamis-gpu-api` stays
dependency-free.

## Uniform and storage block layouts

A shader reads a uniform or storage block at byte offsets fixed by its layout rules, and the CPU writer must use
the same offsets. The rules differ mainly in padding:

- **std140** (uniform blocks): arrays and structs are aligned to 16 bytes, so a `float[4]` takes 64 bytes.
- **std430** (storage blocks): no rounding to 16 bytes, so a `float[4]` takes 16 bytes.
- **scalar** (`VK_EXT_scalar_block_layout`): every member is aligned to its component size, with no padding.

Under std140 and std430 a `vec3` is aligned to 16 bytes but is only 12 bytes long, so a following `float` fills its
fourth slot.

Use `ShaderLayout` for real blocks. You describe the block with `ShaderLayout.struct`, `ShaderLayout.member` and the
type helpers (`ShaderLayout.vec`, `ShaderLayout.mat`, `ShaderLayout.array`, `ShaderLayout.rowMajor`,
`ShaderLayout.FLOAT`, `ShaderLayout.INT`, ...), then call `ShaderLayout.layout` with one of the
`ShaderLayout.Rules`. The returned `ShaderLayout.Block` gives:

- `offset("lights[3].color")`: the byte offset of any member path, including array elements and nested structs;
- `arrayStride(path)` and `matrixStride(path)`;
- `size()` (as SPIR-V reflection reports it), `paddedSize()` (the stride of an array of these blocks) and
  `alignment()`;
- `entries()`: every member with its offset, size and strides;
- `toGlsl(storage, blockName, instanceName, binding)`: the GLSL declaration, so the shader and the Java writer
  share one definition. It adds `#extension GL_EXT_scalar_block_layout : require` where the rules need it.

`ShaderLayoutTest` checks `ShaderLayout` against glslang's own layout on randomly generated blocks. It allocates, so
build layouts at startup and keep the offsets. `toGlsl` takes `"uniform"` or `"buffer"` as the storage and rejects
two different structs with the same name, since GLSL has one declaration per struct name.

Regenerate the glslang golden with `scripts/gen-shader-layout-golden.py`.

## Vertex inputs

`VertexLayout` is a pipeline's vertex input: a stride and `VertexAttribute`s, each a shader `location`, a byte
`offsetBytes` and a `VertexFormat`.
- **Formats:** each `VertexFormat` names the component count and type together, as WebGPU does: `FLOAT3`, `SNORM8X4`,
  `FLOAT16X2`, `UINT32X4` and so on.
- **Checks:** the constructor rejects duplicate locations and attributes that run past the stride.
- **What it isn't:** mesh semantics (position, normal, UV sets) and packing choices belong to MeshForge's layout,
  which maps onto this one.

## Instance transforms

The default instance format is Vectrix's packed 3x4 affine (`PackedAffineArray`, 12 floats per instance, built and
gathered into draw order with `PackedAffineKernels`). `GpuTransformWriteKernels.writeMatrix4fFromPackedAffine`
expands it to 16-float column-major `mat4`s for shaders that need full matrices.

`InstanceSubmissionPipeline` chains the whole per-frame path in one call: skinning, TRS to packed affine, bounds
and upload. Bounds and upload are both written per draw slot: slot `i` gets the transform and the world bounds of
instance `order[i]`. Its `InstanceSubmissionPipeline.Path` selects the packed-affine path or a `Matrix4f` fallback
(also forced by the `dynamisgpu.instanceSubmission.forceMatrix` system property; the old `vectrix.runtime.instanceSubmission.forceMatrix` still works) for comparing the two. It
checks every argument before writing and reuses internal scratch arrays, growing them when a frame has more
instances than before (the `Matrix4f` scratch only when the fallback runs), so use one instance per thread.

### TRS structs

`GpuTransformLayout` writes a transform as separate translation, rotation and scale instead of a matrix. A layout
names a stride, the three field offsets, a `GpuTransformLayout.RotationEncoding` (`FLOAT4` or the smallest-three
`QUAT_SMALLEST3_16`) and a `GpuTransformLayout.ScaleEncoding` (`FLOAT3` or `HALF3`). The constructor rejects a
layout whose fields overlap, run past the stride, or are not aligned to their component size. Two layouts are
predefined:

| Layout | Stride | GLSL (`layout(scalar)`) |
|---|---|---|
| `GpuTransformLayout.floatTRS` | 40 bytes | `struct { vec3 t; vec4 q; vec3 s; }` |
| `GpuTransformLayout.compactTRS` | 28 bytes | `struct { vec3 t; uvec2 q; uint16_t s[3]; }`: smallest-three rotation, half-float scale |

Neither fits std430, where the `vec4` would need 16-byte alignment, so the shader needs the scalar block layout.
`GpuTransformLayout.write` writes instance `index` at byte offset `index * strideBytes`, each field in the
encoding the layout names, so a mixed layout such as `FLOAT4` rotation with `HALF3` scale is written as it says. The
bytes are always little-endian, and a record that does not fit in the buffer throws `IllegalArgumentException`
before anything is written. `GpuTransformLayout.requiredBytes` gives the buffer size and throws
`ArithmeticException` if it overflows an `int`. `GpuTransformWriteKernels.writeTrs` writes a whole array in draw
order.

The shader has to rebuild a matrix from TRS, and the compact form also has to decode the rotation, so prefer the
packed affine unless memory size is the constraint.

