package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.primitives.QuantizedMatrix.Format;
import org.junit.jupiter.api.Test;
import java.util.SplittableRandom;
import static org.junit.jupiter.api.Assertions.*;

/** The block formats on the CPU: layout, round trips within the bound, nearest codes, ties, and refusals. */
class QuantizedMatrixTest {
    static float[] random(int count, SplittableRandom r, double range) {
        float[] f = new float[count];
        for (int i = 0; i < count; i++) f[i] = (float) ((r.nextDouble() * 2 - 1) * range);
        return f;
    }

    @Test void layout() {
        var q8 = new QuantizedMatrix(Format.Q8, 32, 3, 64);
        assertEquals(64, q8.rowCodeBytes()); assertEquals(192, q8.codeBytes()); assertEquals(192, q8.parameterOffset());
        assertEquals(2 * 3 * 2, q8.parameterBytes()); assertEquals(204, q8.bytes());
        var q4 = new QuantizedMatrix(Format.Q4, 64, 3, 64);
        assertEquals(32, q4.rowCodeBytes()); assertEquals(96, q4.parameterOffset()); assertEquals(96 + 2 * 3 + 2, q4.bytes());
        var q4m = new QuantizedMatrix(Format.Q4M, 32, 1, 96);
        assertEquals(48, q4m.codeBytes()); assertEquals(48, q4m.parameterOffset()); assertEquals(48 + 4 * 3, q4m.bytes());
        assertEquals(4096L * 4096 / 2 + 2L * 4096 * 128, new QuantizedMatrix(Format.Q4, 32, 4096, 4096).bytes(),
                "4-bit weights with block 32 take 4.5 bits per weight");
        assertThrows(IllegalArgumentException.class, () -> new QuantizedMatrix(Format.Q8, 16, 1, 32));
        assertThrows(IllegalArgumentException.class, () -> new QuantizedMatrix(Format.Q8, 32, 1, 48));
    }

    /** Every format and block size: dequantized values are within the format's bound and the largest maps exactly. */
    @Test void roundTripsWithinTheBound() {
        var r = new SplittableRandom(3);
        for (var format : new Format[] {Format.Q8, Format.Q4, Format.Q4M}) {
            for (int block : new int[] {32, 64, 128}) {
                var m = new QuantizedMatrix(format, block, 7, 4 * block);
                for (double range : new double[] {1, 1e-3, 3e4}) {
                    float[] x = random(7 * 4 * block, r, range);
                    byte[] packed = m.quantize(x);
                    float[] y = m.dequantize(packed);
                    for (int row = 0; row < 7; row++) {
                        for (int bl = 0; bl < 4; bl++) {
                            float d = Math.abs(m.parameter(packed, row, bl, 0));
                            for (int i = 0; i < block; i++) {
                                int at = row * 4 * block + bl * block + i;
                                // half a step; Q4: the extreme's opposite may clamp from 8 to 7; Q4M: min rounded to
                                // FP16 (relative 2^-11, at the clamp); one FP32 rounding
                                double bound = d / 2 * 1.0001 + (format == Format.Q4 ? d : 0) + Math.abs(x[at]) * 0x1p-10 + Math.ulp(x[at]) + 0x1p-24;
                                assertTrue(Math.abs(x[at] - y[at]) <= bound,
                                        format + "/" + block + " x=" + x[at] + " y=" + y[at] + " d=" + d);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * The K formats' simple encoder: every value within half a sub-block step, plus what rounding the sub-block's
     * scale and minimum to integer multiples of d and dmin costs at the clamp; the codes survive a round trip.
     */
    @Test void kFormatsRoundTripWithinTheirBound() {
        var r = new SplittableRandom(5);
        for (var format : new Format[] {Format.Q4K, Format.Q5K, Format.Q6K}) {
            var m = new QuantizedMatrix(format, 256, 5, 512);
            for (double range : new double[] {1, 1e-3, 100}) {
                float[] x = random(5 * 512, r, range);
                for (int i = 0; i < x.length; i += 97) x[i] = Math.abs(x[i]);   // sub-blocks of mixed sign and not
                byte[] packed = m.quantize(x);
                float[] y = m.dequantize(packed);
                int qmax = format == Format.Q4K ? 15 : format == Format.Q5K ? 31 : 32;
                for (int row = 0; row < 5; row++) {
                    for (int col = 0; col < 512; col++) {
                        int at = row * 512 + col;
                        float[] dm = m.subBlockScale(packed, row, col);
                        int sb = (int) (m.parameterOffset() + ((long) row * 2 + col / 256) * (format == Format.Q6K ? 20 : 16));
                        float d = Math.abs(Float.float16ToFloat((short) ((packed[sb] & 0xFF) | (packed[sb + 1] << 8))));
                        float dmin = format == Format.Q6K ? 0 : Math.abs(Float.float16ToFloat((short) ((packed[sb + 2] & 0xFF) | (packed[sb + 3] << 8))));
                        // Q6K, like Q4: the extreme maps to -32, so its opposite may clamp from 32 to 31, one step
                        double bound = Math.abs(dm[0]) / 2 + (format == Format.Q6K ? Math.abs(dm[0]) : 0) + qmax * d / 2 * 1.01 + dmin / 2
                                + Math.abs(x[at]) * 0x1p-10 + 0x1p-24;
                        assertTrue(Math.abs(x[at] - y[at]) <= bound, format + " x=" + x[at] + " y=" + y[at] + " D=" + dm[0] + " d=" + d);
                        assertEquals(m.code(packed, row, col), new QuantizedMatrix(format, 256, 5, 512).code(packed, row, col));
                    }
                }
                var back = GgufBlocks.fromGgml(GgufBlocks.Type.of(m), GgufBlocks.toGgml(m, packed), 0, 5, 512);
                assertArrayEquals(packed, back.bytes(), format + ": through GGML and back");
            }
        }
    }

    @Test void kLayout() {
        var q5 = new QuantizedMatrix(Format.Q5K, 256, 3, 512);
        assertEquals(768, q5.codeBytes()); assertEquals(768, q5.highOffset()); assertEquals(192, q5.highBytes());
        assertEquals(960, q5.parameterOffset()); assertEquals(16 * 6, q5.parameterBytes()); assertEquals(1056, q5.bytes());
        var q6 = new QuantizedMatrix(Format.Q6K, 256, 1, 256);
        assertEquals(128 + 64 + 20, q6.bytes(), "the same bytes as llama.cpp's 210, padded");
        assertThrows(IllegalArgumentException.class, () -> new QuantizedMatrix(Format.Q4K, 128, 1, 256));
        assertThrows(IllegalArgumentException.class, () -> new QuantizedMatrix(Format.Q4, 256, 1, 256));
        assertEquals(64 + 4, new QuantizedMatrix(Format.Q4, 128, 1, 128).bytes(), "block 128: 64 code bytes, one d, padded to a word");
    }

    @Test void codesAreNearestWithTiesToEven() {
        assertEquals(2, QuantizedMatrix.nearest(2.5f, 1f));
        assertEquals(4, QuantizedMatrix.nearest(3.5f, 1f));
        assertEquals(-2, QuantizedMatrix.nearest(-2.5f, 1f));
        assertEquals(3, QuantizedMatrix.nearest(2.5000002f, 1f));
        assertEquals(7, QuantizedMatrix.nearest(0.875f, 0.125f));
        var r = new SplittableRandom(9);
        for (int i = 0; i < 100_000; i++) {
            float d = Float.float16ToFloat(Float.floatToFloat16((float) (r.nextDouble() * 4 + 1e-4)));
            float y = (float) ((r.nextDouble() * 2 - 1) * 127 * d);
            assertEquals((int) Math.rint((double) y / d), QuantizedMatrix.nearest(y, d), "y=" + y + " d=" + d);
        }
    }

    /** Q8 maps the largest magnitude to ±127, Q4 its signed extreme to -8, Q4M the minimum to code 0. */
    @Test void extremesMapToTheEndCodes() {
        float[] x = new float[32];
        for (int i = 0; i < 32; i++) x[i] = i - 10f;   // -10 .. 21
        for (var format : new Format[] {Format.Q8, Format.Q4, Format.Q4M}) {
            var m = new QuantizedMatrix(format, 32, 1, 32);
            byte[] p = m.quantize(x);
            switch (format) {
                case Q8 -> { assertEquals(127, m.code(p, 0, 31)); assertEquals(Float.float16ToFloat(Float.floatToFloat16(21f / 127f)), m.parameter(p, 0, 0, 0)); }
                case Q4 -> { assertEquals(-8, m.code(p, 0, 31)); assertTrue(m.parameter(p, 0, 0, 0) < 0, "d = 21 * -1/8"); }
                case Q4M -> { assertEquals(0, m.code(p, 0, 0)); assertEquals(15, m.code(p, 0, 31)); assertEquals(-10f, m.parameter(p, 0, 0, 1)); }
            }
        }
        var zeros = new QuantizedMatrix(Format.Q8, 32, 1, 32);
        byte[] p = zeros.quantize(new float[32]);
        assertArrayEquals(new float[32], zeros.dequantize(p));
    }

    /** The quantized kernels' ABI: the GEMM block plus the parameter word, and their specialization constants. */
    @Test void kernelAbi() {
        for (var alg : QuantizedGemm.Algorithm.values()) {
            for (var in : new GemmF16.Input[] {GemmF16.Input.FP16, GemmF16.Input.BF16}) {
                for (var out : Precision.values()) {
                    String k = QuantizedGemm.kernel(alg, in, out);
                    assertEquals(java.util.List.of("a", "b", "c", "epBias", "epGate", "epResidual"), GemmF16.KERNELS.buffers(k), k);
                    assertEquals(68 + 8, GemmF16.KERNELS.parameterBytes(k), k);   // qparams, qhigh
                    var names = GemmF16.KERNELS.constants(k).keySet();
                    assertTrue(names.containsAll(alg == QuantizedGemm.Algorithm.SKINNY ? java.util.List.of("QFORMAT", "QBLOCK", "QROWS")
                            : java.util.List.of("QFORMAT", "QBLOCK")), k + " " + names);
                }
            }
        }
        assertEquals(java.util.List.of("x", "packed", "errors"), GemmF16.KERNELS.buffers("quant_pack"));
        assertEquals(java.util.List.of("packed", "y"), GemmF16.KERNELS.buffers("quant_unpack"));
        assertEquals(1, QuantizedGemm.rowsTier(1)); assertEquals(4, QuantizedGemm.rowsTier(3)); assertEquals(16, QuantizedGemm.rowsTier(5));
    }

    @Test void refusesNonFiniteValuesAndOverflowingScales() {
        var m = new QuantizedMatrix(Format.Q8, 32, 1, 32);
        float[] x = new float[32];
        x[3] = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> m.quantize(x));
        x[3] = 1e30f;
        assertThrows(IllegalArgumentException.class, () -> m.quantize(x), "the scale overflows FP16");
        assertThrows(IllegalArgumentException.class, () -> m.quantize(new float[31]));
    }
}
