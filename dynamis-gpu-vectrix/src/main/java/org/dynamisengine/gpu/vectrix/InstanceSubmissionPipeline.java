/*
 * The MIT License
 *
 * Copyright (c) 2024 Vectrix
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package org.dynamisengine.gpu.vectrix;

import org.dynamisengine.vectrix.affine.PackedAffineArray;
import org.dynamisengine.vectrix.affine.PackedAffineKernels;
import org.dynamisengine.vectrix.affine.Transformf;
import org.dynamisengine.vectrix.core.Matrix4f;
import org.dynamisengine.vectrix.core.Vector3f;
import org.dynamisengine.vectrix.soa.SkinningKernels;
import org.dynamisengine.vectrix.soa.TransformSoA;

/**
 * A reference per-frame submission path: skinning, instance transforms, world-space bounds and upload staging in one
 * call ({@link #processFrame}).
 * <p>
 * The default path builds packed 3x4 affine transforms ({@link PackedAffineKernels}). A matrix path that builds
 * {@link Matrix4f} instances and uploads column-major 4x4 matrices is kept for comparison and debugging; select it
 * per call with {@link Path#MATRIX_FALLBACK}, or for {@link Path#AUTO} with the system property
 * {@value #PROP_FORCE_MATRIX_FALLBACK}.
 * <p>
 * An instance owns internal scratch (packed transforms, a joint palette, and the {@code Matrix4f} array of the
 * matrix path, created the first time that path runs) and is not thread-safe: use one per thread.
 * {@code processFrame} does not allocate unless the instance count or the joint count exceeds the capacity reached so
 * far on the path taken, in which case it grows that path's scratch.
 */
public final class InstanceSubmissionPipeline {
    /**
     * System property that, when {@code "true"} at construction time, makes {@link Path#AUTO} take the matrix path.
     */
    public static final String PROP_FORCE_MATRIX_FALLBACK = "dynamisgpu.instanceSubmission.forceMatrix";

    /** The property's name from when this class was in Vectrix; still honoured. */
    public static final String LEGACY_PROP_FORCE_MATRIX_FALLBACK = "vectrix.runtime.instanceSubmission.forceMatrix";

    /** The transform path {@link #processFrame} takes. */
    public enum Path {
        /** {@link #PACKED_AFFINE}, unless {@link #PROP_FORCE_MATRIX_FALLBACK} was set when the pipeline was made. */
        AUTO,
        /** Packed 3x4 affine transforms; uploads 12 floats per instance. */
        PACKED_AFFINE,
        /** {@link Matrix4f} transforms; uploads 16 floats per instance, column-major. */
        MATRIX_FALLBACK
    }

    private final boolean forceMatrixFallback;
    private PackedAffineArray packed;
    private Matrix4f[] matrices = new Matrix4f[0];
    private float[] skinningMatrixPalette12;
    private final Vector3f tmpMin = new Vector3f();
    private final Vector3f tmpMax = new Vector3f();

    /**
     * Creates a pipeline with packed scratch for {@code initialCapacity} instances and reads
     * {@link #PROP_FORCE_MATRIX_FALLBACK} once.
     *
     * @param initialCapacity the initial instance capacity; values below 1 are treated as 1
     */
    public InstanceSubmissionPipeline(int initialCapacity) {
        this.forceMatrixFallback = Boolean.getBoolean(PROP_FORCE_MATRIX_FALLBACK)
                || Boolean.getBoolean(LEGACY_PROP_FORCE_MATRIX_FALLBACK);
        int cap = java.lang.Math.max(1, initialCapacity);
        this.packed = new PackedAffineArray(cap);
        this.skinningMatrixPalette12 = new float[12];
    }

    /**
     * Runs one frame: skins the vertices, builds the instance transforms, transforms the bounds and stages the
     * upload.
     * <ol>
     * <li>Skinning: builds a matrix palette with each joint's full {@code T·R·S}
     * ({@link SkinningKernels#buildMatrixPalette12}) from all {@code joints.size()} joints, then skins
     * {@code vertexCount} positions with four weighted joints each, from {@code in*} to {@code out*}.</li>
     * <li>Transforms: instance {@code j} gets {@code T * R * S} of {@code transforms[j]}, for {@code j} in
     * {@code [0, instanceCount)}.</li>
     * <li>Bounds and upload, per slot {@code i} in {@code [0, instanceCount)}, with {@code j = order[i]}: the local
     * box {@code (min*[j], max*[j])} is transformed by instance {@code j}, its world-space axis-aligned bounds are
     * written to {@code outMin*[i]} and {@code outMax*[i]}, and slot {@code i} of the upload array receives the
     * transform of instance {@code j}. The bounds and the upload share the slot index, so every slot written this
     * frame has bounds from this frame.</li>
     * </ol>
     * Every argument, including every {@code order} entry and joint index, is checked before anything is written.
     *
     * @param transforms the instance transforms; at least {@code instanceCount} entries, none of them null
     * @param order the instance index of each upload slot; at least {@code instanceCount} entries, each in
     *        {@code [0, instanceCount)}
     * @param minX the local box minimum x per instance
     * @param minY the local box minimum y per instance
     * @param minZ the local box minimum z per instance
     * @param maxX the local box maximum x per instance
     * @param maxY the local box maximum y per instance
     * @param maxZ the local box maximum z per instance
     * @param outMinX receives the world box minimum x per slot; must not be one of the input box arrays
     * @param outMinY receives the world box minimum y per slot
     * @param outMinZ receives the world box minimum z per slot
     * @param outMaxX receives the world box maximum x per slot
     * @param outMaxY receives the world box maximum y per slot
     * @param outMaxZ receives the world box maximum z per slot
     * @param joints the joint transforms of the skeleton
     * @param jointIndices four joint indices per vertex, vertex {@code v} at {@code 4 * v}, each in
     *        {@code [0, joints.size())}
     * @param jointWeights four weights per vertex, matching {@code jointIndices}
     * @param inX the bind-pose vertex x coordinates
     * @param inY the bind-pose vertex y coordinates
     * @param inZ the bind-pose vertex z coordinates
     * @param outX receives the skinned x coordinates; may be {@code inX}
     * @param outY receives the skinned y coordinates; may be {@code inY}
     * @param outZ receives the skinned z coordinates; may be {@code inZ}
     * @param packedUpload receives 12 floats per slot (packed 3x4 affine, row-major) on the packed path; not used,
     *        and may be null, on the matrix path
     * @param matrixUpload receives 16 floats per slot (column-major 4x4) on the matrix path; not used, and may be
     *        null, on the packed path
     * @param instanceCount the number of instances and upload slots, at least 0
     * @param vertexCount the number of vertices to skin, at least 0
     * @param path the path to take
     * @return {@code packedUpload} or {@code matrixUpload}, whichever was written
     * @throws IllegalArgumentException if an argument is null (other than the unused upload array), a count is
     *         negative, an array is too short, or an {@code order} entry or joint index is out of range; nothing is
     *         written then
     */
    public float[] processFrame(
            Transformf[] transforms,
            int[] order,
            float[] minX,
            float[] minY,
            float[] minZ,
            float[] maxX,
            float[] maxY,
            float[] maxZ,
            float[] outMinX,
            float[] outMinY,
            float[] outMinZ,
            float[] outMaxX,
            float[] outMaxY,
            float[] outMaxZ,
            TransformSoA joints,
            int[] jointIndices,
            float[] jointWeights,
            float[] inX,
            float[] inY,
            float[] inZ,
            float[] outX,
            float[] outY,
            float[] outZ,
            float[] packedUpload,
            float[] matrixUpload,
            int instanceCount,
            int vertexCount,
            Path path) {
        if (path == null) {
            throw new IllegalArgumentException("path must not be null");
        }
        Path effective = resolvePath(path);
        checkInstances(transforms, order, instanceCount);
        checkLength(minX, instanceCount, "minX");
        checkLength(minY, instanceCount, "minY");
        checkLength(minZ, instanceCount, "minZ");
        checkLength(maxX, instanceCount, "maxX");
        checkLength(maxY, instanceCount, "maxY");
        checkLength(maxZ, instanceCount, "maxZ");
        checkLength(outMinX, instanceCount, "outMinX");
        checkLength(outMinY, instanceCount, "outMinY");
        checkLength(outMinZ, instanceCount, "outMinZ");
        checkLength(outMaxX, instanceCount, "outMaxX");
        checkLength(outMaxY, instanceCount, "outMaxY");
        checkLength(outMaxZ, instanceCount, "outMaxZ");
        if (effective == Path.MATRIX_FALLBACK) {
            checkLength(matrixUpload, 16L * instanceCount, "matrixUpload");
        } else {
            checkLength(packedUpload, 12L * instanceCount, "packedUpload");
        }
        checkSkinning(joints, jointIndices, jointWeights, inX, inY, inZ, outX, outY, outZ, vertexCount);

        ensureSkinningPaletteCapacity(joints.size());
        SkinningKernels.buildMatrixPalette12(joints, skinningMatrixPalette12, joints.size());
        SkinningKernels.skinLbs4MatrixPalette(skinningMatrixPalette12, jointIndices, jointWeights, inX, inY, inZ,
                outX, outY, outZ, vertexCount);

        if (effective == Path.MATRIX_FALLBACK) {
            runMatrixPath(transforms, order, minX, minY, minZ, maxX, maxY, maxZ,
                    outMinX, outMinY, outMinZ, outMaxX, outMaxY, outMaxZ, matrixUpload, instanceCount);
            return matrixUpload;
        }

        runPackedPath(transforms, order, minX, minY, minZ, maxX, maxY, maxZ,
                outMinX, outMinY, outMinZ, outMaxX, outMaxY, outMaxZ, packedUpload, instanceCount);
        return packedUpload;
    }

    private Path resolvePath(Path requested) {
        if (requested == Path.MATRIX_FALLBACK || requested == Path.PACKED_AFFINE) {
            return requested;
        }
        return forceMatrixFallback ? Path.MATRIX_FALLBACK : Path.PACKED_AFFINE;
    }

    private void runPackedPath(
            Transformf[] transforms,
            int[] order,
            float[] minX,
            float[] minY,
            float[] minZ,
            float[] maxX,
            float[] maxY,
            float[] maxZ,
            float[] outMinX,
            float[] outMinY,
            float[] outMinZ,
            float[] outMaxX,
            float[] outMaxY,
            float[] outMaxZ,
            float[] packedUpload,
            int count) {
        if (count > packed.size()) {
            packed = new PackedAffineArray(count);
        }
        PackedAffineKernels.trsToPackedAffineBatch(transforms, packed, count);
        float[] t = packed.raw();
        for (int i = 0; i < count; i++) {
            int idx = order[i];
            int o = packed.offsetOf(idx);
            float r00 = t[o];
            float r01 = t[o + 1];
            float r02 = t[o + 2];
            float tx = t[o + 3];
            float r10 = t[o + 4];
            float r11 = t[o + 5];
            float r12 = t[o + 6];
            float ty = t[o + 7];
            float r20 = t[o + 8];
            float r21 = t[o + 9];
            float r22 = t[o + 10];
            float tz = t[o + 11];

            // centre through the full transform, half extents through |R|: the tightest box around the moved box
            float cx = (minX[idx] + maxX[idx]) * 0.5f;
            float cy = (minY[idx] + maxY[idx]) * 0.5f;
            float cz = (minZ[idx] + maxZ[idx]) * 0.5f;
            float ex = (maxX[idx] - minX[idx]) * 0.5f;
            float ey = (maxY[idx] - minY[idx]) * 0.5f;
            float ez = (maxZ[idx] - minZ[idx]) * 0.5f;
            float ncx = r00 * cx + r01 * cy + r02 * cz + tx;
            float ncy = r10 * cx + r11 * cy + r12 * cz + ty;
            float ncz = r20 * cx + r21 * cy + r22 * cz + tz;
            float nex = java.lang.Math.abs(r00) * ex + java.lang.Math.abs(r01) * ey + java.lang.Math.abs(r02) * ez;
            float ney = java.lang.Math.abs(r10) * ex + java.lang.Math.abs(r11) * ey + java.lang.Math.abs(r12) * ez;
            float nez = java.lang.Math.abs(r20) * ex + java.lang.Math.abs(r21) * ey + java.lang.Math.abs(r22) * ez;
            outMinX[i] = ncx - nex;
            outMinY[i] = ncy - ney;
            outMinZ[i] = ncz - nez;
            outMaxX[i] = ncx + nex;
            outMaxY[i] = ncy + ney;
            outMaxZ[i] = ncz + nez;

            System.arraycopy(t, o, packedUpload, i * PackedAffineArray.STRIDE_FLOATS, PackedAffineArray.STRIDE_FLOATS);
        }
    }

    private void runMatrixPath(
            Transformf[] transforms,
            int[] order,
            float[] minX,
            float[] minY,
            float[] minZ,
            float[] maxX,
            float[] maxY,
            float[] maxZ,
            float[] outMinX,
            float[] outMinY,
            float[] outMinZ,
            float[] outMaxX,
            float[] outMaxY,
            float[] outMaxZ,
            float[] matrixUpload,
            int count) {
        ensureMatrixCapacity(count);
        for (int i = 0; i < count; i++) {
            Transformf t = transforms[i];
            matrices[i].translationRotateScale(t.translation, t.rotation, t.scale);
        }
        for (int i = 0; i < count; i++) {
            int idx = order[i];
            Matrix4f m = matrices[idx];
            m.transformAab(minX[idx], minY[idx], minZ[idx], maxX[idx], maxY[idx], maxZ[idx], tmpMin, tmpMax);
            outMinX[i] = tmpMin.x;
            outMinY[i] = tmpMin.y;
            outMinZ[i] = tmpMin.z;
            outMaxX[i] = tmpMax.x;
            outMaxY[i] = tmpMax.y;
            outMaxZ[i] = tmpMax.z;
            int base = i << 4;
            matrixUpload[base] = m.m00();
            matrixUpload[base + 1] = m.m01();
            matrixUpload[base + 2] = m.m02();
            matrixUpload[base + 3] = m.m03();
            matrixUpload[base + 4] = m.m10();
            matrixUpload[base + 5] = m.m11();
            matrixUpload[base + 6] = m.m12();
            matrixUpload[base + 7] = m.m13();
            matrixUpload[base + 8] = m.m20();
            matrixUpload[base + 9] = m.m21();
            matrixUpload[base + 10] = m.m22();
            matrixUpload[base + 11] = m.m23();
            matrixUpload[base + 12] = m.m30();
            matrixUpload[base + 13] = m.m31();
            matrixUpload[base + 14] = m.m32();
            matrixUpload[base + 15] = m.m33();
        }
    }

    private void ensureMatrixCapacity(int count) {
        if (count <= matrices.length) {
            return;
        }
        Matrix4f[] next = new Matrix4f[count];
        System.arraycopy(matrices, 0, next, 0, matrices.length);
        for (int i = matrices.length; i < count; i++) {
            next[i] = new Matrix4f();
        }
        matrices = next;
    }

    private void ensureSkinningPaletteCapacity(int jointCount) {
        int needed = jointCount * 12;
        if (skinningMatrixPalette12.length < needed) {
            skinningMatrixPalette12 = new float[needed];
        }
    }

    private static void checkInstances(Transformf[] transforms, int[] order, int count) {
        if (count < 0) {
            throw new IllegalArgumentException("instanceCount must not be negative: " + count);
        }
        if (transforms == null || transforms.length < count) {
            throw new IllegalArgumentException("transforms must hold " + count + " entries");
        }
        for (int i = 0; i < count; i++) {
            if (transforms[i] == null) {
                throw new IllegalArgumentException("transforms[" + i + "] is null");
            }
        }
        if (order == null || order.length < count) {
            throw new IllegalArgumentException("order must hold " + count + " entries");
        }
        for (int i = 0; i < count; i++) {
            if (order[i] < 0 || order[i] >= count) {
                throw new IllegalArgumentException("order[" + i + "] = " + order[i] + " is outside [0, " + count + ")");
            }
        }
    }

    private static void checkSkinning(TransformSoA joints, int[] jointIndices, float[] jointWeights, float[] inX,
            float[] inY, float[] inZ, float[] outX, float[] outY, float[] outZ, int vertexCount) {
        if (joints == null) {
            throw new IllegalArgumentException("joints must not be null");
        }
        if (vertexCount < 0) {
            throw new IllegalArgumentException("vertexCount must not be negative: " + vertexCount);
        }
        long n = 4L * vertexCount;
        if (jointIndices == null || jointIndices.length < n) {
            throw new IllegalArgumentException("jointIndices must hold " + n + " entries");
        }
        int jointCount = joints.size();
        for (int i = 0; i < n; i++) {
            if (jointIndices[i] < 0 || jointIndices[i] >= jointCount) {
                throw new IllegalArgumentException("jointIndices[" + i + "] = " + jointIndices[i]
                        + " is outside [0, " + jointCount + ")");
            }
        }
        checkLength(jointWeights, n, "jointWeights");
        checkLength(inX, vertexCount, "inX");
        checkLength(inY, vertexCount, "inY");
        checkLength(inZ, vertexCount, "inZ");
        checkLength(outX, vertexCount, "outX");
        checkLength(outY, vertexCount, "outY");
        checkLength(outZ, vertexCount, "outZ");
    }

    private static void checkLength(float[] a, long n, String name) {
        if (a == null || a.length < n) {
            throw new IllegalArgumentException(name + " must hold " + n + " values");
        }
    }
}
