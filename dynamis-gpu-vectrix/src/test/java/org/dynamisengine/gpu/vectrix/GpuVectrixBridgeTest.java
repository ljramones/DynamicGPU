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

import org.dynamisengine.gpu.vectrix.GpuTransformLayout.RotationEncoding;
import org.dynamisengine.gpu.vectrix.GpuTransformLayout.ScaleEncoding;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.dynamisengine.vectrix.affine.PackedAffineArray;
import org.dynamisengine.vectrix.affine.PackedAffineKernels;
import org.dynamisengine.vectrix.affine.Transformf;
import org.dynamisengine.vectrix.core.Matrix4f;
import org.dynamisengine.vectrix.core.Quaternionf;
import org.dynamisengine.vectrix.core.Vector3f;
import org.dynamisengine.vectrix.gpu.Half;
import org.dynamisengine.vectrix.gpu.QuatCompression;
import org.dynamisengine.vectrix.soa.TransformSoA;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The transform layout, transform writers and instance pipeline, moved from Vectrix with their tests. */
class GpuVectrixBridgeTest {

    // from vectrix CoverageGapsTest.java
    @Test
    void instancePipelinePathsAgree() {
        int n = 5;
        Transformf[] t = new Transformf[n];
        int[] order = {4, 0, 3, 1, 2};
        float[][] bounds = new float[6][n];
        for (int i = 0; i < n; i++) {
            t[i] = new Transformf(new Vector3f(i, -i, 2 * i), new Quaternionf().rotationXYZ(0.1f * i, 0.2f, -0.3f * i),
                    new Vector3f(1 + 0.1f * i, 1, 2));
            bounds[0][i] = -1; bounds[1][i] = -2; bounds[2][i] = -0.5f;
            bounds[3][i] = 1; bounds[4][i] = 0.5f; bounds[5][i] = 3;
        }
        TransformSoA joints = new TransformSoA(1);
        joints.qw[0] = 1;
        float[] vin = {1, 2, 3, 4}, vout = new float[4];
        float[][] boxPacked = new float[6][n], boxMatrix = new float[6][n];
        float[] packedUpload = new float[12 * n], matrixUpload = new float[16 * n];
        InstanceSubmissionPipeline pipe = new InstanceSubmissionPipeline(2); // grows
        pipe.processFrame(t, order, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
                boxPacked[0], boxPacked[1], boxPacked[2], boxPacked[3], boxPacked[4], boxPacked[5],
                joints, new int[16], new float[] {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, vin, vin, vin, vout, vout, vout,
                packedUpload, matrixUpload, n, 4, InstanceSubmissionPipeline.Path.PACKED_AFFINE);
        pipe.processFrame(t, order, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
                boxMatrix[0], boxMatrix[1], boxMatrix[2], boxMatrix[3], boxMatrix[4], boxMatrix[5],
                joints, new int[16], new float[] {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, vin, vin, vin, vout, vout, vout,
                packedUpload, matrixUpload, n, 4, InstanceSubmissionPipeline.Path.MATRIX_FALLBACK);
        for (int k = 0; k < 6; k++) {
            assertArrayEquals(boxMatrix[k], boxPacked[k], 1e-4f, "bounds " + k);
        }
        // the packed upload, expanded by the GPU writer, is exactly the matrix upload
        float[] expanded = new float[16 * n];
        int[] identityOrder = {0, 1, 2, 3, 4};
        GpuTransformWriteKernels.writeMatrix4fFromPackedAffine(new PackedAffineArray(packedUpload), identityOrder, expanded, 0, n);
        assertArrayEquals(matrixUpload, expanded, 1e-5f);
        assertArrayEquals(vin, vout, 0f, "identity joint leaves vertices in place");
    }

    // from vectrix GpuPackingTest.java
    @Test
    void gpuTransformLayoutFloatAndCompactWrite() {
        Transformf t = new Transformf();
        t.translation.set(1.25f, -2.5f, 3.75f);
        t.rotation.set(new Quaternionf().rotationXYZ(0.2f, -0.4f, 0.8f));
        t.scale.set(0.5f, 2.0f, 4.0f);

        GpuTransformLayout full = GpuTransformLayout.floatTRS();
        ByteBuffer a = ByteBuffer.allocate(full.requiredBytes(1)).order(ByteOrder.LITTLE_ENDIAN);
        full.write(a, 0, t);
        assertEquals(1.25f, a.getFloat(0), 0.0f);
        assertEquals(-2.5f, a.getFloat(4), 0.0f);
        assertEquals(3.75f, a.getFloat(8), 0.0f);
        assertEquals(0.5f, a.getFloat(28), 0.0f);
        assertEquals(2.0f, a.getFloat(32), 0.0f);
        assertEquals(4.0f, a.getFloat(36), 0.0f);

        GpuTransformLayout compact = GpuTransformLayout.compactTRS();
        ByteBuffer b = ByteBuffer.allocate(compact.requiredBytes(1)).order(ByteOrder.LITTLE_ENDIAN);
        compact.write(b, 0, t);
        assertEquals(1.25f, b.getFloat(0), 0.0f);
        assertEquals(-2.5f, b.getFloat(4), 0.0f);
        assertEquals(3.75f, b.getFloat(8), 0.0f);
        long packedQ = b.getLong(12);
        Quaternionf dq = QuatCompression.unpackSmallest3(packedQ, new Quaternionf());
        assertTrue(QuatCompression.angularErrorDegrees(t.rotation, dq) < 0.05f);
        assertEquals(t.scale.x, Half.unpack(b.getShort(20)), 0.001f);
        assertEquals(t.scale.y, Half.unpack(b.getShort(22)), 0.001f);
        assertEquals(t.scale.z, Half.unpack(b.getShort(24)), 0.001f);
        // GPUs read little-endian: a default (big-endian) buffer receives exactly the same bytes
        for (GpuTransformLayout layout : new GpuTransformLayout[] {full, compact}) {
            ByteBuffer le = ByteBuffer.allocate(layout.requiredBytes(1)).order(ByteOrder.LITTLE_ENDIAN);
            ByteBuffer be = ByteBuffer.allocate(layout.requiredBytes(1));
            layout.write(le, 0, t);
            layout.write(be, 0, t);
            assertEquals(le.rewind(), be.rewind());
        }
    }

    // from vectrix GpuSoaNnCurveFixesTest.java
    @Test
    void gpuTransformLayoutValidatesItsDescription() {
        RotationEncoding f4 = RotationEncoding.FLOAT4;
        ScaleEncoding f3 = ScaleEncoding.FLOAT3;
        ScaleEncoding h3 = ScaleEncoding.HALF3;
        assertThrows(IllegalArgumentException.class, () -> new GpuTransformLayout(40, 0, 12, 28, null, f3));
        assertThrows(IllegalArgumentException.class, () -> new GpuTransformLayout(40, 0, 12, 28, f4, null));
        assertThrows(IllegalArgumentException.class, () -> new GpuTransformLayout(0, 0, 12, 28, f4, f3));
        assertThrows(IllegalArgumentException.class, () -> new GpuTransformLayout(-40, 0, 12, 28, f4, f3));
        assertThrows(IllegalArgumentException.class, () -> new GpuTransformLayout(42, 0, 12, 28, f4, f3)); // stride
        assertThrows(IllegalArgumentException.class, () -> new GpuTransformLayout(36, 0, 12, 28, f4, f3)); // past end
        assertThrows(IllegalArgumentException.class, () -> new GpuTransformLayout(40, 0, 8, 28, f4, f3)); // overlap
        assertThrows(IllegalArgumentException.class, () -> new GpuTransformLayout(40, 0, 12, 24, f4, f3)); // overlap
        new GpuTransformLayout(40, 28, 12, 0, f4, f3); // any order of fields
        assertThrows(IllegalArgumentException.class, () -> new GpuTransformLayout(40, -4, 12, 28, f4, f3));
        assertThrows(IllegalArgumentException.class, () -> new GpuTransformLayout(40, 0, 14, 28, f4, f3)); // align
        assertThrows(IllegalArgumentException.class, () -> new GpuTransformLayout(36, 0, 12, 29, f4, h3)); // align
        assertThrows(IllegalArgumentException.class,
                () -> new GpuTransformLayout(40, 0, 12, Integer.MAX_VALUE - 2, f4, f3)); // overflowing end
        new GpuTransformLayout(36, 0, 12, 30, f4, h3); // half scale needs 2-byte alignment only
        assertThrows(IllegalArgumentException.class, () -> GpuTransformLayout.floatTRS().requiredBytes(-1));
        assertEquals(0, GpuTransformLayout.floatTRS().requiredBytes(0));
    }

    // from vectrix GpuSoaNnCurveFixesTest.java
    @Test
    void gpuTransformLayoutWritesEveryFieldInItsOwnEncoding() {
        Transformf t = new Transformf();
        t.translation.set(1.25f, -2.5f, 3.75f);
        t.rotation.set(new Quaternionf().rotationXYZ(0.2f, -0.4f, 0.8f));
        t.scale.set(0.5f, 2.0f, 4.0f);
        // FLOAT4 rotation with HALF3 scale used to be written entirely in the compact form
        GpuTransformLayout mixed = new GpuTransformLayout(36, 0, 12, 28, RotationEncoding.FLOAT4, ScaleEncoding.HALF3);
        ByteBuffer b = ByteBuffer.allocate(mixed.requiredBytes(2)).order(ByteOrder.LITTLE_ENDIAN);
        mixed.write(b, 1, t);
        assertEquals(1.25f, b.getFloat(36));
        assertEquals(t.rotation.x, b.getFloat(48));
        assertEquals(t.rotation.w, b.getFloat(60));
        assertEquals(2.0f, Half.unpack(b.getShort(66)));
        assertEquals(0, b.getShort(70), "the padding is left alone");
        // and the other mix: compact rotation with float scale
        GpuTransformLayout mixed2 = new GpuTransformLayout(32, 20, 0, 8, RotationEncoding.QUAT_SMALLEST3_16,
                ScaleEncoding.FLOAT3);
        ByteBuffer c = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
        mixed2.write(c, 0, t);
        assertEquals(QuatCompression.packSmallest3(t.rotation), c.getLong(0));
        assertEquals(4.0f, c.getFloat(16));
        assertEquals(3.75f, c.getFloat(28));
        // a record that does not fit is rejected before any byte is written
        ByteBuffer small = ByteBuffer.allocate(mixed.requiredBytes(2) - 1);
        assertThrows(IllegalArgumentException.class, () -> mixed.write(small, 1, t));
        assertThrows(IllegalArgumentException.class, () -> mixed.write(small, -1, t));
        assertThrows(IllegalArgumentException.class, () -> mixed.write(null, 0, t));
        assertThrows(IllegalArgumentException.class, () -> mixed.write(small, 0, null));
        assertEquals(0, firstNonZero(small));
    }

    // from vectrix GpuSoaNnCurveFixesTest.java
    @Test
    void transformWriteKernelsCheckEverythingFirst() {
        Transformf[] tr = {new Transformf(), new Transformf(), null};
        tr[0].translation.set(1, 2, 3);
        tr[1].translation.set(4, 5, 6);
        GpuTransformLayout layout = GpuTransformLayout.floatTRS();
        ByteBuffer out = ByteBuffer.allocate(layout.requiredBytes(2)).order(ByteOrder.LITTLE_ENDIAN);
        GpuTransformWriteKernels.writeTrs(layout, tr, new int[] {1, 0}, out, 2);
        assertEquals(4f, out.getFloat(0));
        assertEquals(1f, out.getFloat(40));
        ByteBuffer fresh = ByteBuffer.allocate(layout.requiredBytes(2));
        assertThrows(IllegalArgumentException.class, () -> GpuTransformWriteKernels.writeTrs(layout, tr,
                new int[] {0, 3}, fresh, 2)); // index out of range: was written up to the bad slot
        assertThrows(IllegalArgumentException.class, () -> GpuTransformWriteKernels.writeTrs(layout, tr,
                new int[] {0, 2}, fresh, 2)); // null transform
        assertThrows(IllegalArgumentException.class, () -> GpuTransformWriteKernels.writeTrs(layout, tr,
                new int[] {0, 1}, fresh, 3)); // order too short
        assertThrows(IllegalArgumentException.class, () -> GpuTransformWriteKernels.writeTrs(layout, tr,
                new int[] {0, 1}, ByteBuffer.allocate(79), 2));
        assertThrows(IllegalArgumentException.class, () -> GpuTransformWriteKernels.writeTrs(layout, tr,
                new int[] {0, 1}, fresh, -1));
        assertEquals(0, firstNonZero(fresh));

        PackedAffineArray packed = new PackedAffineArray(2);
        packed.set(1, 1, 0, 0, 7, 0, 1, 0, 8, 0, 0, 1, 9);
        float[] m = new float[3 + 32];
        GpuTransformWriteKernels.writeMatrix4fFromPackedAffine(packed, new int[] {1, 1}, m, 3, 2);
        assertEquals(7f, m[3 + 12]);
        assertEquals(9f, m[3 + 16 + 14]);
        float[] untouched = new float[32];
        assertThrows(IllegalArgumentException.class, () -> GpuTransformWriteKernels.writeMatrix4fFromPackedAffine(
                packed, new int[] {0, 2}, untouched, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> GpuTransformWriteKernels.writeMatrix4fFromPackedAffine(
                packed, new int[] {0, 1}, untouched, 1, 2));
        assertArrayEquals(new float[32], untouched);
    }

    // from vectrix GpuSoaNnCurveFixesTest.java
    @Test
    void instancePipelineWritesBoundsPerSlotAndKeepsMatrixScratchOffThePackedPath() throws Exception {
        int n = 3;
        Transformf[] tr = new Transformf[n];
        for (int i = 0; i < n; i++) {
            tr[i] = new Transformf();
            tr[i].translation.set(10 * (i + 1), 0, 0);
            tr[i].scale.set(i + 1, 1, 1);
        }
        int[] order = {2, 2, 0}; // instance 1 is not drawn; instance 2 twice
        float[] minX = {-1, -1, -1}, minY = {-1, -1, -1}, minZ = {-1, -1, -1};
        float[] maxX = {1, 1, 1}, maxY = {1, 1, 1}, maxZ = {1, 1, 1};
        TransformSoA joints = new TransformSoA(1);
        float[] v = new float[1];
        for (InstanceSubmissionPipeline.Path path : new InstanceSubmissionPipeline.Path[] {
            InstanceSubmissionPipeline.Path.PACKED_AFFINE, InstanceSubmissionPipeline.Path.MATRIX_FALLBACK}) {
            float[][] out = new float[6][n];
            for (float[] o : out) {
                Arrays.fill(o, Float.NaN); // stale data
            }
            InstanceSubmissionPipeline pipe = new InstanceSubmissionPipeline(1);
            float[] upload = pipe.processFrame(tr, order, minX, minY, minZ, maxX, maxY, maxZ, out[0], out[1], out[2],
                    out[3], out[4], out[5], joints, new int[4], new float[] {1, 0, 0, 0}, v, v, v, v, v, v,
                    new float[12 * n], new float[16 * n], n, 1, path);
            for (int i = 0; i < n; i++) {
                int j = order[i];
                // slot i holds instance j's bounds (they used to be written at index j, leaving slot 1 stale)
                assertEquals(10 * (j + 1) - (j + 1), out[0][i], 1e-5f, path + " slot " + i);
                assertEquals(10 * (j + 1) + (j + 1), out[3][i], 1e-5f, path + " slot " + i);
                float tx = path == InstanceSubmissionPipeline.Path.PACKED_AFFINE ? upload[12 * i + 3] : upload[16 * i + 12];
                assertEquals(10 * (j + 1), tx, path + " upload " + i);
            }
            Field f = InstanceSubmissionPipeline.class.getDeclaredField("matrices");
            f.setAccessible(true);
            int matrices = ((Matrix4f[]) f.get(pipe)).length;
            assertEquals(path == InstanceSubmissionPipeline.Path.PACKED_AFFINE ? 0 : n, matrices, path.toString());
        }
        // a bad order entry, joint index or short array throws before anything is written
        float[] outMin = new float[n];
        float[] skinned = {42};
        InstanceSubmissionPipeline pipe = new InstanceSubmissionPipeline(4);
        assertThrows(IllegalArgumentException.class, () -> pipe.processFrame(tr, new int[] {0, 1, 3}, minX, minY,
                minZ, maxX, maxY, maxZ, outMin, outMin, outMin, outMin, outMin, outMin, joints, new int[4],
                new float[4], v, v, v, skinned, skinned, skinned, new float[12 * n], null, n, 1,
                InstanceSubmissionPipeline.Path.PACKED_AFFINE));
        assertThrows(IllegalArgumentException.class, () -> pipe.processFrame(tr, order, minX, minY, minZ, maxX,
                maxY, maxZ, outMin, outMin, outMin, outMin, outMin, outMin, joints, new int[] {0, 0, 1, 0},
                new float[4], v, v, v, skinned, skinned, skinned, new float[12 * n], null, n, 1,
                InstanceSubmissionPipeline.Path.PACKED_AFFINE));
        assertThrows(IllegalArgumentException.class, () -> pipe.processFrame(tr, order, minX, minY, minZ, maxX,
                maxY, maxZ, outMin, outMin, outMin, outMin, outMin, outMin, joints, new int[4], new float[4], v, v, v,
                skinned, skinned, skinned, new float[12 * n - 1], null, n, 1,
                InstanceSubmissionPipeline.Path.PACKED_AFFINE));
        assertEquals(42f, skinned[0], "skinning did not run");
        assertArrayEquals(new float[n], outMin);
    }

    // from vectrix LowFindingsTest.java
    @Test
    void gpuLayoutSizeOverflowIsDetected() {
        GpuTransformLayout layout = GpuTransformLayout.compactTRS();
        assertEquals(28 * 1000, layout.requiredBytes(1000));
        assertThrows(ArithmeticException.class, () -> layout.requiredBytes(100_000_000)); // was -1494967296
    }

    // from vectrix RenderingMathReferenceTest.java
    @Test
    void packedAffineExpandsToTheSameMatrixAsJoml() {
        Transformf[] t = new Transformf[3];
        t[0] = new Transformf(new Vector3f(1, 2, 3), new Quaternionf().rotationAxis(0.7f, 0.3f, 0.9f, -0.2f).normalize(),
                new Vector3f(1, 1, 1));
        t[1] = new Transformf(new Vector3f(-4, 0.5f, 9), new Quaternionf().rotationXYZ(1.1f, -0.4f, 2.3f),
                new Vector3f(2, 0.5f, 3));
        t[2] = new Transformf();
        PackedAffineArray packed = new PackedAffineArray(3);
        PackedAffineKernels.trsToPackedAffineBatch(t, packed, 3);
        float[] out = new float[48];
        GpuTransformWriteKernels.writeMatrix4fFromPackedAffine(packed, new int[] {0, 1, 2}, out, 0, 3);
        for (int i = 0; i < 3; i++) {
            float[] expected = new Matrix4f().translationRotateScale(t[i].translation, t[i].rotation, t[i].scale)
                    .get(new float[16]);
            float[] actual = new float[16];
            System.arraycopy(out, i * 16, actual, 0, 16);
            assertArrayEquals(expected, actual, 1e-5f, "transform " + i);
        }
        // The origin must map to the translation (it mapped to 0 before the fix).
        Matrix4f m = new Matrix4f().set(java.util.Arrays.copyOfRange(out, 0, 16));
        Vector3f origin = m.transformPosition(new Vector3f());
        assertEquals(new Vector3f(1, 2, 3).distance(origin), 0.0f, 1e-5f);
    }

    private static int firstNonZero(ByteBuffer b) {
        for (int i = 0; i < b.limit(); i++) {
            if (b.get(i) != 0) {
                return i + 1;
            }
        }
        return 0;
    }
}
