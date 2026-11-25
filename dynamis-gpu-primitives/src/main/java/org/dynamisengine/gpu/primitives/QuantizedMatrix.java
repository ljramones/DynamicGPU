package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.kit.Checks;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * A matrix of weights quantized by blocks along its rows (1.9.0): {@code rows x cols}, each row split into blocks of
 * {@link #block} elements, each block stored as small integer codes and FP16 parameters. It is how
 * {@link QuantizedGemm} reads the weights of a linear layer ({@code n x k}, B transposed), at a quarter to a half of
 * FP16's bytes.
 * <p>
 * <b>Formats</b> (compatible in meaning with the common Q8_0, Q4_0 and Q4_1 schemes):
 * <ul>
 * <li>{@link Format#Q8}: {@code x = q d}, codes in [-127, 127], {@code d = fp16(max|x| * (1/127))};</li>
 * <li>{@link Format#Q4}: {@code x = q d}, codes in [-8, 7], {@code d = fp16(v * (-1/8))} where {@code v} is the
 * block's value of largest magnitude, so it maps to -8;</li>
 * <li>{@link Format#Q4M}: {@code x = fl(q d + m)}, codes in [0, 15], {@code m = fp16(min)},
 * {@code d = fp16((max - min) * (1/15))}.</li>
 * </ul>
 * Each code is the nearest integer to {@code x / d} (Q4M: {@code fl(x - m) / d}), ties to even, then clamped to its
 * range; a block whose {@code d} is 0 has codes 0. The scale is computed with multiplies, never a division, and each
 * code is found by comparing against exact midpoints {@code (2c + 1) d / 2}, so the GPU quantizer
 * ({@link Quantize}) gives the same bits as {@link #quantize} on every device. Dequantized Q8 and Q4 values are exact
 * in FP32; Q4M rounds once.
 * <p>
 * <b>Layout</b> (little-endian, the bytes a GPU buffer holds):
 * <ul>
 * <li>codes, row-major: Q8 one byte per element (two's complement), Q4 and Q4M two per byte (element {@code 2t} in
 * the low nibble), as unsigned codes (Q4 stores {@code q + 8});</li>
 * <li>then, from the first multiple of 16 bytes, the block parameters, row-major: one FP16 {@code d} per block (Q8,
 * Q4), or a pair {@code d, m} (Q4M), padded with zeros to a multiple of 4 bytes.</li>
 * </ul>
 * Values must be finite, and a block's scale must fit FP16 (|x| up to about 65504 times the code range).
 *
 * @param format the block format
 * @param block elements per block: 32 or 64
 * @param rows the matrix's rows (n, the outputs of a linear layer)
 * @param cols the matrix's columns (k, the inputs): a multiple of {@code block}
 */
public record QuantizedMatrix(Format format, int block, int rows, int cols) {
    /** A block format. */
    public enum Format {
        /** Symmetric 8-bit: {@code x = q d}. */
        Q8(8, 1),
        /** Symmetric 4-bit: {@code x = q d}, q in [-8, 7]. */
        Q4(4, 1),
        /** 4-bit with a minimum: {@code x = q d + m}, q in [0, 15]. */
        Q4M(4, 2);

        private final int bits, parameters;
        Format(int bits, int parameters) { this.bits = bits; this.parameters = parameters; }
        /** Bits per code. */
        public int bits() { return bits; }
        /** FP16 parameters per block: d, or d and m. */
        public int parameters() { return parameters; }
    }

    /** The reciprocal constants the scales are computed with (exact in their use: one correctly rounded multiply). */
    static final float INV127 = 1f / 127f, INV15 = 1f / 15f;

    public QuantizedMatrix {
        Checks.notNull(format, "format");
        if (block != 32 && block != 64) throw new IllegalArgumentException("block must be 32 or 64: " + block);
        if (rows <= 0 || cols <= 0) throw new IllegalArgumentException("dimensions must be positive: " + rows + "x" + cols);
        if (cols % block != 0) throw new IllegalArgumentException("cols must be a multiple of the block: " + cols + " % " + block);
        long bytes = (long) rows * cols * format.bits / 8 + 2L * format.parameters * rows * (cols / block) + 16;
        if (bytes > Checks.MAX_INDEX) throw new IllegalArgumentException("matrix too large: " + rows + "x" + cols);
    }

    /** Blocks per row. */
    public int blocksPerRow() { return cols / block; }
    /** Bytes of one row's codes. */
    public long rowCodeBytes() { return (long) cols * format.bits / 8; }
    /** Bytes of all codes. */
    public long codeBytes() { return rows * rowCodeBytes(); }
    /** Where the block parameters start: the codes rounded up to 16 bytes. */
    public long parameterOffset() { return (codeBytes() + 15) & ~15L; }
    /** Bytes of all block parameters. */
    public long parameterBytes() { return 2L * format.parameters * rows * blocksPerRow(); }
    /** The bytes the packed matrix takes: the parameters end rounded up to 4 (the GPU writes whole words). */
    public long bytes() { return (parameterOffset() + parameterBytes() + 3) & ~3L; }
    /** Bytes of the same matrix in FP16. */
    public long fp16Bytes() { return 2L * rows * cols; }

    // --- the CPU encoder and decoder -------------------------------------------------------------------------------

    /**
     * Quantizes {@code values} ({@code rows x cols}, row-major) into this layout, bit for bit as the GPU quantizer.
     *
     * @throws IllegalArgumentException if a value is NaN or infinite, or a block's scale overflows FP16
     */
    public byte[] quantize(float[] values) {
        if (values.length != (long) rows * cols) throw new IllegalArgumentException("values must hold " + rows + "x" + cols + ": " + values.length);
        byte[] out = new byte[Math.toIntExact(bytes())];
        var params = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
        int[] codes = new int[block];
        for (int r = 0; r < rows; r++) {
            for (int bl = 0; bl < blocksPerRow(); bl++) {
                int base = r * cols + bl * block;
                for (int i = 0; i < block; i++) {
                    float v = values[base + i];
                    if (v - v != 0f) throw new IllegalArgumentException("value " + (base + i) + " = " + v + " is not finite");
                }
                short[] p = encodeBlock(values, base, codes);
                long blockIndex = (long) r * blocksPerRow() + bl;
                for (int t = 0; t < format.parameters; t++) params.putShort((int) (parameterOffset() + 2 * (blockIndex * format.parameters + t)), p[t]);
                for (int i = 0; i < block; i++) putCode(out, r, bl * block + i, codes[i]);
            }
        }
        return out;
    }

    /** Encodes one block: its parameters (d, or d and m) as FP16 bits, and its codes (signed for Q8 and Q4). */
    short[] encodeBlock(float[] x, int base, int[] codes) {
        float lo = Float.POSITIVE_INFINITY, hi = Float.NEGATIVE_INFINITY, amax = 0f, extreme = 0f;
        for (int i = 0; i < block; i++) {
            float v = x[base + i];
            lo = Math.min(lo, v); hi = Math.max(hi, v);
            if (Math.abs(v) > amax) { amax = Math.abs(v); extreme = v; }   // the first of largest magnitude
        }
        float d, m = 0f;
        short dBits, mBits = 0;
        switch (format) {
            case Q8 -> { dBits = Float.floatToFloat16(amax * INV127); }
            case Q4 -> { dBits = Float.floatToFloat16(extreme * -0.125f); }
            default -> {
                mBits = Float.floatToFloat16(lo);
                m = Float.float16ToFloat(mBits);
                dBits = Float.floatToFloat16((hi - lo) * INV15);
            }
        }
        d = Float.float16ToFloat(dBits);
        if (Float.isInfinite(d) || Float.isInfinite(m)) throw new IllegalArgumentException("a block's parameters overflow FP16 at element " + base);
        int min = format == Format.Q8 ? -127 : format == Format.Q4 ? -8 : 0, max = format == Format.Q8 ? 127 : format == Format.Q4 ? 7 : 15;
        for (int i = 0; i < block; i++) {
            float y = format == Format.Q4M ? x[base + i] - m : x[base + i];
            // Q4's d is negative when the extreme is positive: y / d = (-y) / (-d)
            codes[i] = d == 0f ? 0 : Math.max(min, Math.min(max, d > 0 ? nearest(y, d) : nearest(-y, -d)));
        }
        return format == Format.Q4M ? new short[] {dBits, mBits} : new short[] {dBits};
    }

    /**
     * The integer nearest to {@code y / d} (ties to even) for {@code d > 0}, found as the GPU finds it: an estimate,
     * corrected by comparing {@code y} with the exact midpoints {@code (2c + 1) d / 2} (exact: {@code 2c + 1} has at
     * most 9 bits and d 11). Defined for {@code |y / d| < 2^8}; beyond, the result only needs to clamp correctly.
     */
    static int nearest(float y, float d) {
        float est = (float) Math.rint(y * (1f / d));
        int c = (int) Math.max(-300f, Math.min(300f, est));
        for (int step = 0; step < 2; step++) {
            float up = (2 * c + 1) * d * 0.5f, down = (2 * c - 1) * d * 0.5f;
            if (y > up || y == up && (c & 1) != 0) c++;
            else if (y < down || y == down && (c & 1) != 0) c--;
        }
        return c;
    }

    private void putCode(byte[] out, int r, int col, int code) {
        if (format == Format.Q8) { out[(int) (r * rowCodeBytes() + col)] = (byte) code; return; }
        int u = format == Format.Q4 ? code + 8 : code;
        int idx = (int) (r * rowCodeBytes() + col / 2);
        out[idx] = (byte) ((col & 1) == 0 ? (out[idx] & 0xF0) | u : (out[idx] & 0x0F) | (u << 4));
    }

    /** The code of element {@code (r, col)} of a packed matrix, signed for Q8 and Q4. */
    int code(byte[] packed, int r, int col) {
        if (format == Format.Q8) return packed[(int) (r * rowCodeBytes() + col)];
        int b = packed[(int) (r * rowCodeBytes() + col / 2)] & 0xFF;
        int u = (col & 1) == 0 ? b & 0xF : b >>> 4;
        return format == Format.Q4 ? u - 8 : u;
    }

    /** FP16 parameter {@code t} (0 = d, 1 = m) of block {@code (r, bl)}, as a float. */
    float parameter(byte[] packed, int r, int bl, int t) {
        long at = parameterOffset() + 2 * (((long) r * blocksPerRow() + bl) * format.parameters + t);
        return Float.float16ToFloat((short) ((packed[(int) at] & 0xFF) | (packed[(int) at + 1] << 8)));
    }

    /** Element {@code (r, col)} of a packed matrix, dequantized: {@code q d}, or {@code fl(q d + m)}. */
    public float value(byte[] packed, int r, int col) {
        int bl = col / block;
        float d = parameter(packed, r, bl, 0), q = code(packed, r, col);
        return format == Format.Q4M ? q * d + parameter(packed, r, bl, 1) : q * d;
    }

    /** Dequantizes a packed matrix to FP32 ({@code rows x cols}, row-major), bit for bit as {@link Quantize}. */
    public float[] dequantize(byte[] packed) {
        if (packed.length < bytes()) throw new IllegalArgumentException("packed must hold " + bytes() + " bytes: " + packed.length);
        float[] out = new float[rows * cols];
        for (int r = 0; r < rows; r++) for (int col = 0; col < cols; col++) out[r * cols + col] = value(packed, r, col);
        return out;
    }

    /**
     * The worst-case error of one dequantized value: half a step ({@code d / 2}) for Q8 and Q4, plus Q4M's rounding of
     * {@code m} (half an FP16 ulp of min) and of {@code q d + m}; inside the clamp, a value maps to its nearest code.
     */
    public String errorBound() {
        return switch (format) {
            case Q8, Q4 -> "|x - dequantized| <= d / 2 inside the code range";
            case Q4M -> "|x - dequantized| <= d / 2 + ulp16(min) / 2 + ulp32(x) / 2";
        };
    }
}
