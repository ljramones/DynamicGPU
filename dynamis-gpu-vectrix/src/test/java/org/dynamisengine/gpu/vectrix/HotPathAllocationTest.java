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

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import org.dynamisengine.gpu.vectrix.GpuTransformLayout.RotationEncoding;
import org.dynamisengine.gpu.vectrix.GpuTransformLayout.ScaleEncoding;
import org.dynamisengine.vectrix.affine.PackedAffineArray;
import org.dynamisengine.vectrix.affine.Transformf;
import org.dynamisengine.vectrix.soa.TransformSoA;
import org.junit.jupiter.api.Test;

/** The transform writers and the instance pipeline allocate nothing per call (moved from Vectrix's allocation test). */
class HotPathAllocationTest {
    private static final Object THREADS;
    private static final Method ALLOCATED;

    static {
        try {
            THREADS = Class.forName("java.lang.management.ManagementFactory").getMethod("getThreadMXBean").invoke(null);
            ALLOCATED = Class.forName("com.sun.management.ThreadMXBean").getMethod("getCurrentThreadAllocatedBytes");
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private static double bytesPerCall(Runnable r) {
        try {
            for (int i = 0; i < 2000; i++) {
                r.run();
            }
            int n = 20000;
            double best = Double.MAX_VALUE;
            for (int round = 0; round < 6 && best >= 1.0; round++) {
                long before = (Long) ALLOCATED.invoke(THREADS);
                for (int i = 0; i < n; i++) {
                    r.run();
                }
                long after = (Long) ALLOCATED.invoke(THREADS);
                best = Math.min(best, (double) (after - before) / n);
            }
            return best;
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(e);
        }
    }

    private static void assertNoAllocation(String what, Runnable r) {
        double b = bytesPerCall(r);
        assertTrue(b < 1.0, what + " allocates " + b + " bytes per call");
    }

    @Test
    void transformWritersAndInstancePipelineDoNotAllocate() {
        int count = 64;
        Transformf[] tr = new Transformf[count];
        int[] order = new int[count];
        for (int i = 0; i < count; i++) {
            tr[i] = new Transformf();
            tr[i].translation.set(i, 1, 2);
            order[i] = count - 1 - i;
        }
        GpuTransformLayout mixed = new GpuTransformLayout(36, 0, 12, 28, RotationEncoding.FLOAT4, ScaleEncoding.HALF3);
        ByteBuffer buf = ByteBuffer.allocate(mixed.requiredBytes(count));
        assertNoAllocation("GpuTransformLayout.write", () -> mixed.write(buf, 7, tr[3]));
        assertNoAllocation("GpuTransformWriteKernels.writeTrs", () -> GpuTransformWriteKernels.writeTrs(mixed, tr, order, buf, count));
        PackedAffineArray packed = new PackedAffineArray(count);
        float[] m16 = new float[16 * count + 5];
        assertNoAllocation("GpuTransformWriteKernels.writeMatrix4fFromPackedAffine",
                () -> GpuTransformWriteKernels.writeMatrix4fFromPackedAffine(packed, order, m16, 5, count));

        TransformSoA joints = new TransformSoA(8);
        int[] ji = new int[4 * count];
        float[] jw = new float[4 * count];
        float[] px = new float[count];
        float[] py = new float[count];
        float[] pz = new float[count];
        InstanceSubmissionPipeline pipe = new InstanceSubmissionPipeline(count);
        float[] b0 = new float[count];
        float[] b1 = new float[count];
        float[] up = new float[12 * count];
        assertNoAllocation("InstanceSubmissionPipeline.processFrame", () -> pipe.processFrame(tr, order, b0, b0, b0, b1,
                b1, b1, px, py, pz, px, py, pz, joints, ji, jw, b0, b0, b0, b1, b1, b1, up, null, count, count,
                InstanceSubmissionPipeline.Path.PACKED_AFFINE));
    }
}
