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
import org.dynamisengine.vectrix.affine.Transformf;

import java.nio.ByteBuffer;

/**
 * Write kernels that stage per-instance transforms for upload, in draw order.
 * <p>
 * Every kernel takes an {@code order} array: output slot {@code i} receives source transform {@code order[i]}, for
 * {@code i} in {@code [0, count)}. Outputs are written densely. The arguments, including every {@code order} entry,
 * are checked before anything is written: a bad one throws {@link IllegalArgumentException} and leaves the output
 * unchanged. The kernels are stateless, safe to call from any thread on separate outputs, and do not allocate.
 * <p>
 * To copy packed affines in draw order without expanding them, use
 * {@link org.dynamisengine.vectrix.affine.PackedAffineKernels#uploadPackedAffine}.
 */
public final class GpuTransformWriteKernels {
    private GpuTransformWriteKernels() {
    }

    /**
     * Expand packed affines into column-major 4x4 matrices (GLSL {@code mat4} / JOML {@code Matrix4f} order).
     * Each packed transform holds rows {@code [r00 r01 r02 tx, r10 r11 r12 ty, r20 r21 r22 tz]} (see
     * {@link PackedAffineArray}), so the output is
     * {@code [r00 r10 r20 0, r01 r11 r21 0, r02 r12 r22 0, tx ty tz 1]}. {@code out} must not be
     * {@code src.raw()}.
     *
     * @param src the packed affine transforms
     * @param order the source transform index of each output slot, each in {@code [0, src.size())}; at least
     *        {@code count} entries
     * @param out receives 16 floats per slot, slot {@code i} at {@code outOff + 16 * i}
     * @param outOff the index of slot 0 in {@code out}
     * @param count the number of slots to write, at least 0
     * @throws IllegalArgumentException if an argument is null, {@code count} is negative, {@code order} is shorter
     *         than {@code count} or names a transform outside {@code src}, or {@code out} cannot hold
     *         {@code 16 * count} floats from {@code outOff}; nothing is written then
     * @since 3.0.0
     */
    public static void writeMatrix4fFromPackedAffine(PackedAffineArray src, int[] order, float[] out, int outOff,
            int count) {
        if (src == null || out == null) {
            throw new IllegalArgumentException("src and out must not be null");
        }
        checkOrder(order, count, src.size());
        if (outOff < 0 || outOff > out.length - 16L * count) {
            throw new IllegalArgumentException("out must hold " + 16L * count + " floats from offset " + outOff);
        }
        float[] data = src.raw();
        for (int i = 0; i < count; i++) {
            int srcBase = src.offsetOf(order[i]);
            int dst = outOff + (i << 4);
            out[dst] = data[srcBase];
            out[dst + 1] = data[srcBase + 4];
            out[dst + 2] = data[srcBase + 8];
            out[dst + 3] = 0.0f;
            out[dst + 4] = data[srcBase + 1];
            out[dst + 5] = data[srcBase + 5];
            out[dst + 6] = data[srcBase + 9];
            out[dst + 7] = 0.0f;
            out[dst + 8] = data[srcBase + 2];
            out[dst + 9] = data[srcBase + 6];
            out[dst + 10] = data[srcBase + 10];
            out[dst + 11] = 0.0f;
            out[dst + 12] = data[srcBase + 3];
            out[dst + 13] = data[srcBase + 7];
            out[dst + 14] = data[srcBase + 11];
            out[dst + 15] = 1.0f;
        }
    }

    /**
     * Writes {@code transforms[order[i]]} into record {@code i} of {@code out} with {@link GpuTransformLayout#write},
     * for {@code i} in {@code [0, count)}: every field in the encoding {@code layout} names, in little-endian bytes.
     * Record 0 starts at byte 0 of {@code out} (absolute puts; the position is neither used nor moved); write from
     * another byte offset through {@link ByteBuffer#slice(int, int)}.
     *
     * @param layout the record layout
     * @param transforms the source transforms
     * @param order the source transform index of each record, each in {@code [0, transforms.length)} and naming a
     *        non-null transform; at least {@code count} entries
     * @param out the buffer, at least {@code layout.requiredBytes(count)} bytes up to its limit
     * @param count the number of records to write, at least 0
     * @throws IllegalArgumentException if an argument is null, {@code count} is negative, {@code order} is shorter
     *         than {@code count} or names a missing transform, or {@code out} is too small; nothing is written then
     * @since 3.0.0
     */
    public static void writeTrs(GpuTransformLayout layout, Transformf[] transforms, int[] order, ByteBuffer out,
            int count) {
        if (layout == null || transforms == null || out == null) {
            throw new IllegalArgumentException("layout, transforms and out must not be null");
        }
        checkOrder(order, count, transforms.length);
        for (int i = 0; i < count; i++) {
            if (transforms[order[i]] == null) {
                throw new IllegalArgumentException("transforms[" + order[i] + "] is null");
            }
        }
        if ((long) count * layout.strideBytes > out.limit()) {
            throw new IllegalArgumentException(count + " records of " + layout.strideBytes
                    + " bytes do not fit in a buffer of limit " + out.limit());
        }
        for (int i = 0; i < count; i++) {
            layout.writeRecord(out, i * layout.strideBytes, transforms[order[i]]);
        }
    }

    private static void checkOrder(int[] order, int count, int sourceCount) {
        if (count < 0) {
            throw new IllegalArgumentException("count must not be negative: " + count);
        }
        if (order == null || order.length < count) {
            throw new IllegalArgumentException("order must hold " + count + " entries");
        }
        for (int i = 0; i < count; i++) {
            int k = order[i];
            if (k < 0 || k >= sourceCount) {
                throw new IllegalArgumentException("order[" + i + "] = " + k + " is outside [0, " + sourceCount + ")");
            }
        }
    }
}
