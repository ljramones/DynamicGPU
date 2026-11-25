package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.vectrix.nn.BlockQuant;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * A matrix of weights quantized by blocks along its rows (1.9.0): {@code rows x cols}, each row split into blocks of
 * {@link #block} elements, each block stored as small integer codes and FP16 parameters. It is how
 * {@link QuantizedGemm} reads the weights of a linear layer ({@code n x k}, B transposed), at a quarter to a half of
 * FP16's bytes. {@link GgufBlocks} converts llama.cpp's blocks to and from it without loss.
 * <p>
 * <b>Formats with one scale per block</b> (block 32, 64 or 128; the meanings of llama.cpp's Q8_0, Q4_0 and Q4_1):
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
 * <b>K formats</b> (1.9.1; llama.cpp's Q4_K, Q5_K and Q6_K): super-blocks of 256 ({@code block} is 256) with FP16
 * {@code d} (and {@code dmin}) and small integer scales per sub-block, which give {@code D = d sc} and
 * {@code M = dmin mn}, both exact in FP32:
 * <ul>
 * <li>{@link Format#Q4K}: {@code x = fl(D q - M)}, codes in [0, 15], 8 sub-blocks of 32 with 6-bit {@code sc} and
 * {@code mn};</li>
 * <li>{@link Format#Q5K}: as Q4K with codes in [0, 31];</li>
 * <li>{@link Format#Q6K}: {@code x = fl(D q)}, codes in [-32, 31], 16 sub-blocks of 16 with int8 {@code sc}.</li>
 * </ul>
 * These values are bit for bit llama.cpp's dequantization. {@link #quantize} has a simple encoder for them (scales
 * from each sub-block's range, codes nearest); llama.cpp's tools search the scales and are more accurate, and their
 * output imports through {@link GgufBlocks}. There is no GPU encoder for the K formats.
 * <p>
 * <b>Layout</b> (little-endian, the bytes a GPU buffer holds):
 * <ul>
 * <li>codes, row-major in element order: Q8 one byte per element (two's complement); the others a nibble per
 * element (element {@code 2t} in the low nibble): the unsigned code for Q4 ({@code q + 8}), Q4M and Q4K, its low 4
 * bits for Q5K and for Q6K's {@code q + 32};</li>
 * <li>then from a 16-byte boundary, Q5K and Q6K's high bits, row-major in element order: Q5K one bit per element
 * (element {@code e} of a row at bit {@code e % 32} of word {@code e / 32}), Q6K two ({@code 2 (e % 16)} of word
 * {@code e / 16});</li>
 * <li>then from a 16-byte boundary the block parameters, row-major: one FP16 {@code d} per block (Q8, Q4) or a pair
 * {@code d, m} (Q4M); per super-block 16 bytes {@code d, dmin} and llama.cpp's 12 packed scale bytes (Q4K, Q5K), or
 * 20 bytes {@code d}, 0 and 16 int8 scales (Q6K); padded with zeros to a multiple of 4 bytes.</li>
 * </ul>
 * Values must be finite, and a block's scale must fit FP16 (|x| up to about 65504 times the code range).
 * <p>
 * <b>The math is Vectrix's</b> (1.13.0): the encoders, scale packing and decoding are
 * {@link org.dynamisengine.vectrix.nn.BlockQuant}'s, which holds the same formats under the same names; this record
 * adds the GPU layout.
 *
 * @param format the block format
 * @param block elements per block: 32, 64 or 128; 256 for the K formats
 * @param rows the matrix's rows (n, the outputs of a linear layer)
 * @param cols the matrix's columns (k, the inputs): a multiple of {@code block}
 */
public record QuantizedMatrix(Format format, int block, int rows, int cols) {
    /** A block format. */
    public enum Format {
        /** Symmetric 8-bit: {@code x = q d}. */
        Q8(8, 1, 0),
        /** Symmetric 4-bit: {@code x = q d}, q in [-8, 7]. */
        Q4(4, 1, 0),
        /** 4-bit with a minimum: {@code x = q d + m}, q in [0, 15]. */
        Q4M(4, 2, 0),
        /** K-quant 4-bit (llama.cpp Q4_K): {@code x = D q - M}, q in [0, 15], sub-blocks of 32. @since 1.9.1 */
        Q4K(4, 0, 32),
        /** K-quant 5-bit (llama.cpp Q5_K): {@code x = D q - M}, q in [0, 31], sub-blocks of 32. @since 1.9.1 */
        Q5K(5, 0, 32),
        /** K-quant 6-bit (llama.cpp Q6_K): {@code x = D q}, q in [-32, 31], sub-blocks of 16. @since 1.9.1 */
        Q6K(6, 0, 16);

        private final int bits, parameters, subBlock;
        Format(int bits, int parameters, int subBlock) { this.bits = bits; this.parameters = parameters; this.subBlock = subBlock; }
        /** Bits per code. */
        public int bits() { return bits; }
        /** FP16 parameters per block: d, or d and m (0 for the K formats, whose parameters are per super-block). */
        public int parameters() { return parameters; }
        /** Whether this is a K format: super-blocks of 256 with per-sub-block scales. */
        public boolean kQuant() { return subBlock != 0; }
        /** Elements per sub-block of a K format (0 otherwise). */
        public int subBlock() { return subBlock; }
        /** Bits per element in the low code plane: 8 for Q8, else 4. */
        int lowBits() { return this == Q8 ? 8 : 4; }
        /** Bits per element in the high plane: Q5K 1, Q6K 2, else 0. */
        int highBits() { return kQuant() ? bits - 4 : 0; }
        /** Parameter bytes per block or super-block. */
        int parameterBytes() { return this == Q6K ? 20 : kQuant() ? 16 : 2 * parameters; }
        /** The same format in Vectrix's {@link BlockQuant}, which holds the math. */
        BlockQuant.Format quant() { return BlockQuant.Format.valueOf(name()); }
    }

    /** Elements per super-block of the K formats. */
    public static final int SUPER_BLOCK = BlockQuant.SUPER_BLOCK;

    public QuantizedMatrix {
        Checks.notNull(format, "format");
        if (format.kQuant() ? block != SUPER_BLOCK : block != 32 && block != 64 && block != 128)
            throw new IllegalArgumentException(format + " blocks must be " + (format.kQuant() ? "256" : "32, 64 or 128") + ": " + block);
        if (rows <= 0 || cols <= 0) throw new IllegalArgumentException("dimensions must be positive: " + rows + "x" + cols);
        if (cols % block != 0) throw new IllegalArgumentException("cols must be a multiple of the block: " + cols + " % " + block);
        long elements = (long) rows * cols;
        long bytes = elements * format.bits / 8 + (long) format.parameterBytes() * rows * (cols / block) + 48;
        if (bytes > Checks.MAX_INDEX) throw new IllegalArgumentException("matrix too large: " + rows + "x" + cols);
    }

    /** Blocks (super-blocks for the K formats) per row. */
    public int blocksPerRow() { return cols / block; }
    /** Bytes of one row's low codes. */
    public long rowCodeBytes() { return (long) cols * format.lowBits() / 8; }
    /** Bytes of all low codes. */
    public long codeBytes() { return rows * rowCodeBytes(); }
    /** Where the high bits start (Q5K, Q6K): the codes rounded up to 16 bytes. @since 1.9.1 */
    public long highOffset() { return (codeBytes() + 15) & ~15L; }
    /** Bytes of the high bits. @since 1.9.1 */
    public long highBytes() { return (long) rows * cols * format.highBits() / 8; }
    /** Where the block parameters start: after the codes and high bits, on 16 bytes. */
    public long parameterOffset() { return (highOffset() + highBytes() + 15) & ~15L; }
    /** Bytes of all block parameters. */
    public long parameterBytes() { return (long) format.parameterBytes() * rows * blocksPerRow(); }
    /** The bytes the packed matrix takes: the parameters end rounded up to 4 (the GPU writes whole words). */
    public long bytes() { return (parameterOffset() + parameterBytes() + 3) & ~3L; }
    /** Bytes of the same matrix in FP16. */
    public long fp16Bytes() { return 2L * rows * cols; }

    // --- the CPU encoder and decoder -------------------------------------------------------------------------------

    /**
     * Quantizes {@code values} ({@code rows x cols}, row-major) into this layout: bit for bit as the GPU quantizer for
     * Q8, Q4 and Q4M; a simple encoder for the K formats.
     *
     * @throws IllegalArgumentException if a value is NaN or infinite, or a block's scale overflows FP16
     */
    public byte[] quantize(float[] values) {
        if (values.length != (long) rows * cols) throw new IllegalArgumentException("values must hold " + rows + "x" + cols + ": " + values.length);
        for (int i = 0; i < values.length; i++) {
            float v = values[i];
            if (v - v != 0f) throw new IllegalArgumentException("value " + i + " = " + v + " is not finite");
        }
        byte[] out = new byte[Math.toIntExact(bytes())];
        var params = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
        BlockQuant.Format q = format.quant();
        int[] codes = new int[block];
        short[] p = new short[2];
        int[] sc = new int[SUPER_BLOCK / 16], mn = new int[SUPER_BLOCK / 16];
        for (int r = 0; r < rows; r++) {
            for (int bl = 0; bl < blocksPerRow(); bl++) {
                int base = r * cols + bl * block;
                int at = (int) (parameterOffset() + ((long) r * blocksPerRow() + bl) * format.parameterBytes());
                if (format.kQuant()) {
                    // d and dmin (Q6K: d and 0), then the sub-block scales: llama.cpp's 12 packed bytes, or 16 int8
                    BlockQuant.encodeSuperBlock(q, values, base, codes, 0, p, 0, sc, 0, mn, 0);
                    params.putShort(at, p[0]).putShort(at + 2, p[1]);
                    if (format == Format.Q6K) for (int t = 0; t < q.subBlocks(); t++) out[at + 4 + t] = (byte) sc[t];
                    else BlockQuant.packScales(sc, 0, mn, 0, out, at + 4);
                } else {
                    BlockQuant.encodeBlock(q, values, base, block, codes, 0, p, 0);
                    for (int t = 0; t < format.parameters; t++) params.putShort(at + 2 * t, p[t]);
                }
                for (int i = 0; i < block; i++) putCode(out, r, bl * block + i, codes[i]);
            }
        }
        return out;
    }

    /** Writes the code of element {@code (r, col)}: the low plane and, for Q5K and Q6K, the high bits. */
    void putCode(byte[] out, int r, int col, int code) {
        if (format == Format.Q8) { out[(int) (r * rowCodeBytes() + col)] = (byte) code; return; }
        int u = switch (format) { case Q4 -> code + 8; case Q6K -> code + 32; default -> code; };
        int idx = (int) (r * rowCodeBytes() + col / 2);
        int lo = u & 0xF;
        out[idx] = (byte) ((col & 1) == 0 ? (out[idx] & 0xF0) | lo : (out[idx] & 0x0F) | (lo << 4));
        int hb = format.highBits();
        if (hb > 0) {
            long bit = ((long) r * cols + col) * hb;
            int at = (int) (highOffset() + bit / 8), shift = (int) (bit % 8);
            int mask = ((1 << hb) - 1) << shift;
            out[at] = (byte) ((out[at] & ~mask) | (((u >> 4) << shift) & mask));
        }
    }

    /** The code of element {@code (r, col)}: signed for Q8, Q4 and Q6K. */
    int code(byte[] packed, int r, int col) {
        if (format == Format.Q8) return packed[(int) (r * rowCodeBytes() + col)];
        int b = packed[(int) (r * rowCodeBytes() + col / 2)] & 0xFF;
        int u = (col & 1) == 0 ? b & 0xF : b >>> 4;
        int hb = format.highBits();
        if (hb > 0) {
            long bit = ((long) r * cols + col) * hb;
            u |= ((packed[(int) (highOffset() + bit / 8)] >> (int) (bit % 8)) & ((1 << hb) - 1)) << 4;
        }
        return switch (format) { case Q4 -> u - 8; case Q6K -> u - 32; default -> u; };
    }

    private static float half(byte[] b, long at) { return Float.float16ToFloat((short) ((b[(int) at] & 0xFF) | (b[(int) at + 1] << 8))); }

    /** FP16 parameter {@code t} (0 = d, 1 = m) of block {@code (r, bl)} of a Q8, Q4 or Q4M matrix, as a float. */
    float parameter(byte[] packed, int r, int bl, int t) {
        return half(packed, parameterOffset() + ((long) r * blocksPerRow() + bl) * format.parameterBytes() + 2L * t);
    }

    /**
     * The scale and offset of element {@code (r, col)}'s sub-block of a K-format matrix: {@code D = d sc} and
     * {@code M = dmin mn} (0 for Q6K), exact in FP32.
     */
    float[] subBlockScale(byte[] packed, int r, int col) {
        int at = (int) (parameterOffset() + ((long) r * blocksPerRow() + col / SUPER_BLOCK) * format.parameterBytes());
        int j = (col % SUPER_BLOCK) / format.subBlock;
        float d = half(packed, at);
        if (format == Format.Q6K) return new float[] {d * packed[at + 4 + j], 0f};
        return new float[] {d * BlockQuant.unpackScale(packed, at + 4, j, false),
            half(packed, at + 2) * BlockQuant.unpackScale(packed, at + 4, j, true)};
    }

    /** Element {@code (r, col)} of a packed matrix, dequantized: {@code q d}, {@code fl(q d + m)}, {@code fl(D q - M)}, {@code fl(D q)}. */
    public float value(byte[] packed, int r, int col) {
        int q = code(packed, r, col);
        if (format.kQuant()) {
            int at = (int) (parameterOffset() + ((long) r * blocksPerRow() + col / SUPER_BLOCK) * format.parameterBytes());
            int j = (col % SUPER_BLOCK) / format.subBlock;
            float d = half(packed, at);
            if (format == Format.Q6K) return BlockQuant.decodeK(format.quant(), q, d, 0f, packed[at + 4 + j], 0);
            return BlockQuant.decodeK(format.quant(), q, d, half(packed, at + 2),
                BlockQuant.unpackScale(packed, at + 4, j, false), BlockQuant.unpackScale(packed, at + 4, j, true));
        }
        int bl = col / block;
        return BlockQuant.decode(format.quant(), q, parameter(packed, r, bl, 0),
            format == Format.Q4M ? parameter(packed, r, bl, 1) : 0f);
    }

    /** Dequantizes a packed matrix to FP32 ({@code rows x cols}, row-major), bit for bit as {@link Quantize}. */
    public float[] dequantize(byte[] packed) {
        if (packed.length < bytes()) throw new IllegalArgumentException("packed must hold " + bytes() + " bytes: " + packed.length);
        float[] out = new float[rows * cols];
        for (int r = 0; r < rows; r++) for (int col = 0; col < cols; col++) out[r * cols + col] = value(packed, r, col);
        return out;
    }

    /**
     * The worst-case error of one dequantized value from {@link #quantize}: half a step ({@code d / 2}) for Q8 and Q4,
     * plus Q4M's rounding of {@code m} (half an FP16 ulp of min) and of {@code q d + m}; inside the clamp, a value maps
     * to its nearest code. The K formats add the rounding of each sub-block's scale to a 6- or 8-bit integer.
     */
    public String errorBound() {
        return switch (format) {
            case Q8, Q4 -> "|x - dequantized| <= d / 2 inside the code range";
            case Q4M -> "|x - dequantized| <= d / 2 + ulp16(min) / 2 + ulp32(x) / 2";
            case Q4K, Q5K, Q6K -> "|x - dequantized| <= D / 2 inside the code range, D the sub-block's scale (its rounding to an integer multiple of d widens the range)";
        };
    }
}
