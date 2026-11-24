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

import org.dynamisengine.vectrix.gpu.Half;
import org.dynamisengine.vectrix.gpu.QuatCompression;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.dynamisengine.vectrix.affine.Transformf;

/**
 * Layout and writer for per-instance translation, rotation and scale (TRS) records in a GPU buffer.
 * <p>
 * A layout is a record stride, the byte offsets of the three fields inside a record and the encodings of the
 * rotation and the scale; the translation is always three floats. Instance {@code i} starts at byte
 * {@code i * strideBytes} of the buffer (absolute indexing: the buffer position is neither used nor moved).
 * {@link #write} honours every field of the layout and always produces little-endian bytes. The constructor
 * validates the layout, so every instance describes fields that fit in the stride, do not overlap and are aligned to
 * their component size. Instances are immutable; the writer does not allocate.
 */
public final class GpuTransformLayout {
    /** How the rotation quaternion is stored. */
    public enum RotationEncoding {
        /** Four 32-bit floats, {@code x, y, z, w} (16 bytes, 4-byte aligned). */
        FLOAT4(16, 4),
        /**
         * The 64-bit smallest-three code of {@link QuatCompression#packSmallest3}, written as one little-endian
         * 64-bit word (a GLSL {@code uvec2}, low word first; 8 bytes, 4-byte aligned).
         */
        QUAT_SMALLEST3_16(8, 4);

        private final int bytes;
        private final int alignment;

        RotationEncoding(int bytes, int alignment) {
            this.bytes = bytes;
            this.alignment = alignment;
        }

        /**
         * The size of the encoded rotation.
         *
         * @return the size in bytes
         * @since 3.0.0
         */
        public int bytes() {
            return bytes;
        }

        /**
         * The alignment the layout requires for the rotation offset: the size of one component.
         *
         * @return the alignment in bytes
         * @since 3.0.0
         */
        public int alignment() {
            return alignment;
        }
    }

    /** How the scale is stored. */
    public enum ScaleEncoding {
        /** Three 32-bit floats (12 bytes, 4-byte aligned). */
        FLOAT3(12, 4),
        /** Three IEEE half floats from {@link Half#pack} (6 bytes, 2-byte aligned). */
        HALF3(6, 2);

        private final int bytes;
        private final int alignment;

        ScaleEncoding(int bytes, int alignment) {
            this.bytes = bytes;
            this.alignment = alignment;
        }

        /**
         * The size of the encoded scale.
         *
         * @return the size in bytes
         * @since 3.0.0
         */
        public int bytes() {
            return bytes;
        }

        /**
         * The alignment the layout requires for the scale offset: the size of one component.
         *
         * @return the alignment in bytes
         * @since 3.0.0
         */
        public int alignment() {
            return alignment;
        }
    }

    /** The size of the translation field: three floats. */
    private static final int TRANSLATION_BYTES = 12;

    /** The size of one record in bytes: the distance between consecutive instances. */
    public final int strideBytes;
    /** The byte offset of the translation (three floats) inside a record. */
    public final int translationOffset;
    /** The byte offset of the rotation inside a record. */
    public final int rotationOffset;
    /** The byte offset of the scale inside a record. */
    public final int scaleOffset;
    /** How {@link #write} stores the rotation. */
    public final RotationEncoding rotationEncoding;
    /** How {@link #write} stores the scale. */
    public final ScaleEncoding scaleEncoding;

    /**
     * Creates and validates a layout.
     *
     * @param strideBytes the record size in bytes, positive and a multiple of 4 (the alignment of the float
     *        translation)
     * @param translationOffset the byte offset of the translation in a record, a multiple of 4
     * @param rotationOffset the byte offset of the rotation in a record, a multiple of
     *        {@code rotationEncoding.alignment()}
     * @param scaleOffset the byte offset of the scale in a record, a multiple of {@code scaleEncoding.alignment()}
     * @param rotationEncoding the rotation encoding
     * @param scaleEncoding the scale encoding
     * @throws IllegalArgumentException if an encoding is null, the stride is not positive, an offset is negative or
     *         misaligned, a field does not end within the stride, or two fields overlap
     */
    public GpuTransformLayout(int strideBytes, int translationOffset, int rotationOffset, int scaleOffset,
            RotationEncoding rotationEncoding, ScaleEncoding scaleEncoding) {
        if (rotationEncoding == null || scaleEncoding == null) {
            throw new IllegalArgumentException("rotationEncoding and scaleEncoding must not be null");
        }
        if (strideBytes <= 0 || strideBytes % 4 != 0) {
            throw new IllegalArgumentException("strideBytes must be a positive multiple of 4: " + strideBytes);
        }
        checkField("translation", translationOffset, TRANSLATION_BYTES, 4, strideBytes);
        checkField("rotation", rotationOffset, rotationEncoding.bytes(), rotationEncoding.alignment(), strideBytes);
        checkField("scale", scaleOffset, scaleEncoding.bytes(), scaleEncoding.alignment(), strideBytes);
        if (overlap(translationOffset, TRANSLATION_BYTES, rotationOffset, rotationEncoding.bytes())
                || overlap(translationOffset, TRANSLATION_BYTES, scaleOffset, scaleEncoding.bytes())
                || overlap(rotationOffset, rotationEncoding.bytes(), scaleOffset, scaleEncoding.bytes())) {
            throw new IllegalArgumentException("the translation, rotation and scale fields overlap");
        }
        this.strideBytes = strideBytes;
        this.translationOffset = translationOffset;
        this.rotationOffset = rotationOffset;
        this.scaleOffset = scaleOffset;
        this.rotationEncoding = rotationEncoding;
        this.scaleEncoding = scaleEncoding;
    }

    private static void checkField(String name, int offset, int bytes, int alignment, int strideBytes) {
        if (offset < 0 || offset % alignment != 0 || (long) offset + bytes > strideBytes) {
            throw new IllegalArgumentException(name + " (" + bytes + " bytes at offset " + offset
                    + ") must be aligned to " + alignment + " bytes and end within the stride " + strideBytes);
        }
    }

    private static boolean overlap(int a, int aBytes, int b, int bBytes) {
        return a < b + bBytes && b < a + aBytes;
    }

    /**
     * The 40-byte float layout: translation at 0, rotation ({@code FLOAT4}) at 12, scale ({@code FLOAT3}) at 28. It
     * is the GLSL struct {@code layout(scalar) struct { vec3 t; vec4 q; vec3 s; }}, or ten floats; it is not
     * expressible under std430, where the {@code vec4} would need 16-byte alignment.
     *
     * @return a new layout
     */
    public static GpuTransformLayout floatTRS() {
        return new GpuTransformLayout(40, 0, 12, 28, RotationEncoding.FLOAT4, ScaleEncoding.FLOAT3);
    }

    /**
     * The 28-byte compact layout: float translation at 0, smallest-three rotation at 12, half-float scale at 20,
     * and two bytes of padding at the end. It is the GLSL struct
     * {@code layout(scalar) struct { vec3 t; uvec2 q; uint16_t s[3]; }} (16-bit storage), or seven 32-bit words with
     * the halves unpacked by {@code unpackHalf2x16}.
     *
     * @return a new layout
     */
    public static GpuTransformLayout compactTRS() {
        return new GpuTransformLayout(28, 0, 12, 20, RotationEncoding.QUAT_SMALLEST3_16, ScaleEncoding.HALF3);
    }

    /**
     * Bytes needed for {@code count} transforms: {@code count * strideBytes}.
     *
     * @param count the number of transforms, at least 0
     * @return the buffer size in bytes
     * @throws IllegalArgumentException if {@code count} is negative
     * @throws ArithmeticException if the size does not fit in an {@code int} (a {@link ByteBuffer} limit)
     */
    public int requiredBytes(int count) {
        if (count < 0) {
            throw new IllegalArgumentException("count must not be negative: " + count);
        }
        return java.lang.Math.multiplyExact(count, strideBytes);
    }

    /**
     * Writes {@code transform} as record {@code index}, each field in the encoding the layout names: the translation
     * as three floats, the rotation as four floats ({@code FLOAT4}) or the smallest-three code
     * ({@code QUAT_SMALLEST3_16}, the rotation normalized first, see {@link QuatCompression#packSmallest3}), and the
     * scale as three floats ({@code FLOAT3}) or three half floats ({@code HALF3}). The bytes are always
     * little-endian, the order GPUs read, whatever {@code dst.order()} is. Bytes of the record that no field covers
     * are left unchanged.
     *
     * @param dst the buffer; written with absolute puts, its position and order are left unchanged
     * @param index the instance index, at least 0; the record starts at byte {@code index * strideBytes}
     * @param transform the transform to write
     * @throws IllegalArgumentException if {@code dst} or {@code transform} is null, {@code index} is negative, or the
     *         record does not end within {@code dst.limit()}; nothing is written then
     */
    public void write(ByteBuffer dst, int index, Transformf transform) {
        if (dst == null || transform == null) {
            throw new IllegalArgumentException("dst and transform must not be null");
        }
        if (index < 0 || ((long) index + 1) * strideBytes > dst.limit()) {
            throw new IllegalArgumentException("record " + index + " (stride " + strideBytes
                    + ") does not fit in a buffer of limit " + dst.limit());
        }
        writeRecord(dst, index * strideBytes, transform);
    }

    /** Writes one record at byte {@code base}; the caller has checked that it fits. */
    void writeRecord(ByteBuffer dst, int base, Transformf transform) {
        boolean swap = dst.order() != ByteOrder.LITTLE_ENDIAN;
        putFloat(dst, base + translationOffset, transform.translation.x, swap);
        putFloat(dst, base + translationOffset + 4, transform.translation.y, swap);
        putFloat(dst, base + translationOffset + 8, transform.translation.z, swap);
        if (rotationEncoding == RotationEncoding.FLOAT4) {
            putFloat(dst, base + rotationOffset, transform.rotation.x, swap);
            putFloat(dst, base + rotationOffset + 4, transform.rotation.y, swap);
            putFloat(dst, base + rotationOffset + 8, transform.rotation.z, swap);
            putFloat(dst, base + rotationOffset + 12, transform.rotation.w, swap);
        } else {
            long q = QuatCompression.packSmallest3(transform.rotation);
            dst.putLong(base + rotationOffset, swap ? Long.reverseBytes(q) : q);
        }
        if (scaleEncoding == ScaleEncoding.FLOAT3) {
            putFloat(dst, base + scaleOffset, transform.scale.x, swap);
            putFloat(dst, base + scaleOffset + 4, transform.scale.y, swap);
            putFloat(dst, base + scaleOffset + 8, transform.scale.z, swap);
        } else {
            putShort(dst, base + scaleOffset, Half.pack(transform.scale.x), swap);
            putShort(dst, base + scaleOffset + 2, Half.pack(transform.scale.y), swap);
            putShort(dst, base + scaleOffset + 4, Half.pack(transform.scale.z), swap);
        }
    }

    private static void putFloat(ByteBuffer dst, int at, float v, boolean swap) {
        int bits = Float.floatToRawIntBits(v);
        dst.putInt(at, swap ? Integer.reverseBytes(bits) : bits);
    }

    private static void putShort(ByteBuffer dst, int at, short v, boolean swap) {
        dst.putShort(at, swap ? Short.reverseBytes(v) : v);
    }
}
