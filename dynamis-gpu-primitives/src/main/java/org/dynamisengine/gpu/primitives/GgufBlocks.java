package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.primitives.QuantizedMatrix.Format;

/**
 * llama.cpp's quantized tensor blocks (GGML types, as GGUF files hold them) to and from {@link QuantizedMatrix}, without
 * loss (1.9.1): the codes and the FP16 and integer scales move unchanged, so the dequantized values are the same bits
 * as llama.cpp's ({@link QuantizedMatrix#value} computes them as {@code ggml-quants.c} does).
 * <p>
 * A GGUF tensor of {@code ne0 x ne1} elements is {@code ne1} rows of {@code ne0}: as the weights of a linear layer,
 * {@code ne1 = n} outputs and {@code ne0 = k} inputs, which are this class's {@code rows} and {@code cols}. Reading the
 * GGUF container (its header, metadata and tensor table) is left to the caller; {@link #bytes} gives a tensor's size.
 *
 * <table>
 * <caption>Supported types</caption>
 * <tr><th>GGML type</th><th>id</th><th>Block bytes</th><th>Format</th></tr>
 * <tr><td>Q4_0</td><td>2</td><td>18 per 32</td><td>{@link Format#Q4}, block 32</td></tr>
 * <tr><td>Q4_1</td><td>3</td><td>20 per 32</td><td>{@link Format#Q4M}, block 32</td></tr>
 * <tr><td>Q8_0</td><td>8</td><td>34 per 32</td><td>{@link Format#Q8}, block 32</td></tr>
 * <tr><td>Q4_K</td><td>12</td><td>144 per 256</td><td>{@link Format#Q4K}</td></tr>
 * <tr><td>Q5_K</td><td>13</td><td>176 per 256</td><td>{@link Format#Q5K}</td></tr>
 * <tr><td>Q6_K</td><td>14</td><td>210 per 256</td><td>{@link Format#Q6K}</td></tr>
 * </table>
 */
public final class GgufBlocks {
    private GgufBlocks() { }

    /** A GGML block type this class converts. */
    public enum Type {
        Q4_0(2, 32, 18, Format.Q4), Q4_1(3, 32, 20, Format.Q4M), Q8_0(8, 32, 34, Format.Q8),
        Q4_K(12, 256, 144, Format.Q4K), Q5_K(13, 256, 176, Format.Q5K), Q6_K(14, 256, 210, Format.Q6K);

        private final int id, elements, blockBytes;
        private final Format format;
        Type(int id, int elements, int blockBytes, Format format) { this.id = id; this.elements = elements; this.blockBytes = blockBytes; this.format = format; }
        /** The GGML type id (as a GGUF tensor table stores it). */
        public int id() { return id; }
        /** Elements per block. */
        public int elements() { return elements; }
        /** Bytes per block. */
        public int blockBytes() { return blockBytes; }
        /** The format it converts to. */
        public Format format() { return format; }

        /** The type of GGML id {@code id}; {@code IllegalArgumentException} for one this class does not convert. */
        public static Type of(int id) {
            for (var t : values()) if (t.id == id) return t;
            throw new IllegalArgumentException("GGML type " + id + " is not supported (Q4_0, Q4_1, Q8_0, Q4_K, Q5_K and Q6_K are)");
        }

        /** The GGML type of {@code m}, if it has one (block 32 for Q8, Q4 and Q4M). */
        public static Type of(QuantizedMatrix m) {
            for (var t : values()) if (t.format == m.format() && t.elements == m.block()) return t;
            throw new IllegalArgumentException(m + " has no GGML type");
        }
    }

    /** A matrix converted from GGML blocks: its description and packed bytes. */
    public record Packed(QuantizedMatrix matrix, byte[] bytes) { }

    /** The bytes a {@code rows x cols} tensor of {@code type} takes in a GGUF file. */
    public static long bytes(Type type, int rows, int cols) {
        if (cols % type.elements != 0) throw new IllegalArgumentException("cols must be a multiple of " + type.elements + ": " + cols);
        return (long) rows * (cols / type.elements) * type.blockBytes;
    }

    private static int u16(byte[] b, int at) { return (b[at] & 0xFF) | ((b[at + 1] & 0xFF) << 8); }
    private static void put16(byte[] b, int at, int v) { b[at] = (byte) v; b[at + 1] = (byte) (v >> 8); }

    /**
     * Converts {@code data} ({@code rows x cols} of {@code type} from offset {@code offset}, as a GGUF file stores it) to
     * a packed {@link QuantizedMatrix}.
     */
    public static Packed fromGgml(Type type, byte[] data, int offset, int rows, int cols) {
        Checks.notNull(type, "type"); Checks.notNull(data, "data");
        long need = bytes(type, rows, cols);
        if (offset < 0 || data.length - (long) offset < need) throw new IllegalArgumentException("data must hold " + need + " bytes from " + offset);
        var m = new QuantizedMatrix(type.format, type.elements, rows, cols);
        byte[] out = new byte[Math.toIntExact(m.bytes())];
        int perRow = cols / type.elements, pb = type.format.parameterBytes();
        for (int r = 0; r < rows; r++) {
            for (int bl = 0; bl < perRow; bl++) {
                int src = offset + (r * perRow + bl) * type.blockBytes, c0 = bl * type.elements;
                int par = (int) (m.parameterOffset() + ((long) r * perRow + bl) * pb);
                switch (type) {
                    case Q8_0 -> {
                        put16(out, par, u16(data, src));
                        for (int i = 0; i < 32; i++) m.putCode(out, r, c0 + i, data[src + 2 + i]);
                    }
                    case Q4_0, Q4_1 -> {
                        int qs = src + (type == Type.Q4_0 ? 2 : 4);
                        put16(out, par, u16(data, src));
                        if (type == Type.Q4_1) put16(out, par + 2, u16(data, src + 2));
                        for (int i = 0; i < 16; i++) {
                            int b = data[qs + i] & 0xFF, lo = b & 0xF, hi = b >> 4;
                            m.putCode(out, r, c0 + i, type == Type.Q4_0 ? lo - 8 : lo);
                            m.putCode(out, r, c0 + 16 + i, type == Type.Q4_0 ? hi - 8 : hi);
                        }
                    }
                    case Q4_K, Q5_K -> {
                        System.arraycopy(data, src, out, par, 16);   // d, dmin, the 12 scale bytes
                        int qh = src + 16, qs = src + (type == Type.Q5_K ? 48 : 16);
                        for (int e = 0; e < 256; e++) {
                            int chunk = e / 64, l = e % 32;
                            int b = data[qs + 32 * chunk + l] & 0xFF;
                            int q = (e % 64) < 32 ? b & 0xF : b >> 4;
                            if (type == Type.Q5_K) q |= ((data[qh + l] >> (e / 32)) & 1) << 4;
                            m.putCode(out, r, c0 + e, q);
                        }
                    }
                    case Q6_K -> {
                        put16(out, par, u16(data, src + 208));
                        System.arraycopy(data, src + 192, out, par + 4, 16);   // the int8 scales
                        for (int e = 0; e < 256; e++) {
                            int half = e / 128, w = e % 128, l = w % 32;
                            int lb = data[src + 64 * half + (w % 64)] & 0xFF;
                            int lo = w < 64 ? lb & 0xF : lb >> 4;
                            int hi = (data[src + 128 + 32 * half + l] >> (2 * (w / 32))) & 3;
                            m.putCode(out, r, c0 + e, (lo | (hi << 4)) - 32);
                        }
                    }
                }
            }
        }
        return new Packed(m, out);
    }

    /** Converts a packed {@link QuantizedMatrix} back to GGML blocks, bit for bit the blocks it came from. */
    public static byte[] toGgml(QuantizedMatrix m, byte[] packed) {
        var type = Type.of(m);
        if (packed.length < m.bytes()) throw new IllegalArgumentException("packed must hold " + m.bytes() + " bytes");
        int rows = m.rows(), cols = m.cols(), perRow = cols / type.elements, pb = type.format.parameterBytes();
        byte[] out = new byte[Math.toIntExact(bytes(type, rows, cols))];
        for (int r = 0; r < rows; r++) {
            for (int bl = 0; bl < perRow; bl++) {
                int dst = (r * perRow + bl) * type.blockBytes, c0 = bl * type.elements;
                int par = (int) (m.parameterOffset() + ((long) r * perRow + bl) * pb);
                switch (type) {
                    case Q8_0 -> {
                        put16(out, dst, u16(packed, par));
                        for (int i = 0; i < 32; i++) out[dst + 2 + i] = (byte) m.code(packed, r, c0 + i);
                    }
                    case Q4_0, Q4_1 -> {
                        int qs = dst + (type == Type.Q4_0 ? 2 : 4), bias = type == Type.Q4_0 ? 8 : 0;
                        put16(out, dst, u16(packed, par));
                        if (type == Type.Q4_1) put16(out, dst + 2, u16(packed, par + 2));
                        for (int i = 0; i < 16; i++)
                            out[qs + i] = (byte) ((m.code(packed, r, c0 + i) + bias) | ((m.code(packed, r, c0 + 16 + i) + bias) << 4));
                    }
                    case Q4_K, Q5_K -> {
                        System.arraycopy(packed, par, out, dst, 16);
                        int qh = dst + 16, qs = dst + (type == Type.Q5_K ? 48 : 16);
                        for (int e = 0; e < 256; e++) {
                            int q = m.code(packed, r, c0 + e), chunk = e / 64, l = e % 32;
                            int at = qs + 32 * chunk + l;
                            out[at] = (byte) ((e % 64) < 32 ? (out[at] & 0xF0) | (q & 0xF) : (out[at] & 0x0F) | ((q & 0xF) << 4));
                            if (type == Type.Q5_K) out[qh + l] = (byte) (out[qh + l] | (((q >> 4) & 1) << (e / 32)));
                        }
                    }
                    case Q6_K -> {
                        put16(out, dst + 208, u16(packed, par));
                        System.arraycopy(packed, par + 4, out, dst + 192, 16);
                        for (int e = 0; e < 256; e++) {
                            int u = m.code(packed, r, c0 + e) + 32, half = e / 128, w = e % 128, l = w % 32;
                            int at = dst + 64 * half + (w % 64);
                            out[at] = (byte) (w < 64 ? (out[at] & 0xF0) | (u & 0xF) : (out[at] & 0x0F) | ((u & 0xF) << 4));
                            int h = dst + 128 + 32 * half + l;
                            out[h] = (byte) (out[h] | (((u >> 4) & 3) << (2 * (w / 32))));
                        }
                    }
                }
            }
        }
        return out;
    }
}
