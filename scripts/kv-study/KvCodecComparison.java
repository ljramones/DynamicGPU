/*
 * The affine encoding and decoding arithmetic in Packed adapts the Rust host fixture
 * in magnitudedev/magnitude (inference/engine/kernels/tests/attention_k8v4.rs) to Java.
 * Copyright 2026 Magnitude AI Inc. Licensed under Apache-2.0, see LICENSE.magnitude.
 */
import org.dynamisengine.vectrix.nn.BlockQuant;
import org.dynamisengine.vectrix.nn.BlockQuant.Format;

import java.util.Arrays;
import java.util.Locale;
import java.util.SplittableRandom;

/** Offline, finite FP16 cache samples. Run as a Java 25 source file with vectrix-ml on the class path. */
public final class KvCodecComparison {
    private static final int WIDTH = 128;
    private static final int GROUP = 32;
    private static final int QUERY_HEADS = 4;
    private static final int[] LENGTHS = {4096, 32768};
    private static final long[] SEEDS = {117L, 401L, 947L};

    enum Fixture { CENTERED, SHIFTED_GROUPS, OUTLIERS }
    enum Codec { AFFINE_K8_V4, VECTRIX_Q8_Q4M }

    /** One head vector. Codes are packed in element order, low nibble first for V. */
    static final class Packed {
        final Codec codec;
        final boolean key;
        final byte[] codes;
        final short[] coefficients;
        final int paramsPerGroup;
        final int[] groupCodes = new int[GROUP];

        Packed(Codec codec, boolean key) {
            this.codec = codec;
            this.key = key;
            this.paramsPerGroup = key && codec == Codec.VECTRIX_Q8_Q4M ? 1 : 2;
            this.codes = new byte[key ? WIDTH : WIDTH / 2];
            this.coefficients = new short[WIDTH / GROUP * paramsPerGroup];
        }

        int bytes() {
            return codes.length + Short.BYTES * coefficients.length;
        }

        void encode(float[] source, int offset) {
            for (int start = 0; start < WIDTH; start += GROUP) {
                int p = start / GROUP * paramsPerGroup;
                if (codec == Codec.AFFINE_K8_V4) {
                    affine(source, offset + start, p);
                } else {
                    BlockQuant.encodeBlock(key ? Format.Q8 : Format.Q4M, source, offset + start,
                            GROUP, groupCodes, 0, coefficients, p);
                }
                if (key) {
                    for (int i = 0; i < GROUP; i++) codes[start + i] = (byte) groupCodes[i];
                } else {
                    for (int i = 0; i < GROUP; i += 2) {
                        codes[(start + i) / 2] = (byte) (groupCodes[i] | (groupCodes[i + 1] << 4));
                    }
                }
            }
        }

        // Magnitude's host reference in inference/engine/kernels/tests/attention_k8v4.rs.
        void affine(float[] source, int offset, int p) {
            float low = source[offset], high = low;
            for (int i = 1; i < GROUP; i++) {
                low = Math.min(low, source[offset + i]);
                high = Math.max(high, source[offset + i]);
            }
            int levels = key ? 255 : 15;
            coefficients[p] = Float.floatToFloat16((high - low) / levels);
            coefficients[p + 1] = Float.floatToFloat16(low);
            float scale = Float.float16ToFloat(coefficients[p]);
            float zero = Float.float16ToFloat(coefficients[p + 1]);
            float inverse = scale > 0 ? 1.0f / scale : 0.0f;
            for (int i = 0; i < GROUP; i++) {
                int code = (int) Math.fma(source[offset + i] - zero, inverse, 0.5f);
                groupCodes[i] = Math.clamp(code, 0, levels);
            }
        }

        void decode(float[] destination, int offset) {
            for (int start = 0; start < WIDTH; start += GROUP) {
                int p = start / GROUP * paramsPerGroup;
                float scale = Float.float16ToFloat(coefficients[p]);
                float zero = paramsPerGroup == 2 ? Float.float16ToFloat(coefficients[p + 1]) : 0;
                for (int i = start; i < start + GROUP; i++) {
                    int code;
                    if (key) {
                        code = codec == Codec.VECTRIX_Q8_Q4M ? codes[i] : Byte.toUnsignedInt(codes[i]);
                    } else {
                        code = (codes[i / 2] >>> (4 * (i % 2))) & 15;
                    }
                    destination[offset + i] = codec == Codec.AFFINE_K8_V4
                            ? Math.fma((float) code, scale, zero)
                            : BlockQuant.decode(key ? Format.Q8 : Format.Q4M, code, scale, zero);
                }
            }
        }
    }

    static final class Errors {
        long count;
        double squaredError, squaredReference, maxAbsolute;

        void add(double expected, double actual) {
            if (!Double.isFinite(expected) || !Double.isFinite(actual)) throw new AssertionError("nonfinite sample");
            double error = actual - expected;
            count++;
            squaredError += error * error;
            squaredReference += expected * expected;
            maxAbsolute = Math.max(maxAbsolute, Math.abs(error));
        }

        double rmse() { return Math.sqrt(squaredError / count); }
        double relativeL2() { return Math.sqrt(squaredError / squaredReference); }
    }

    record Measurements(Errors keys, Errors values, Errors attention, Errors keyOnly, Errors valueOnly) {
        Measurements() { this(new Errors(), new Errors(), new Errors(), new Errors(), new Errors()); }
    }

    static float half(double x) { return Float.float16ToFloat(Float.floatToFloat16((float) x)); }

    static float[] samples(Fixture fixture, SplittableRandom rng) {
        float[] samples = new float[LENGTHS[LENGTHS.length - 1] * WIDTH];
        for (int start = 0; start < samples.length; start += GROUP) {
            double shift = fixture == Fixture.SHIFTED_GROUPS ? (rng.nextBoolean() ? 3 : -3) : 0;
            double spread = fixture == Fixture.SHIFTED_GROUPS ? 0.35 : 1.0;
            for (int i = start; i < start + GROUP; i++) {
                double value = shift + spread * rng.nextGaussian();
                if (fixture == Fixture.OUTLIERS && rng.nextInt(128) == 0) value += rng.nextBoolean() ? 12 : -12;
                samples[i] = half(value);
            }
        }
        return samples;
    }

    /** Stable two-pass FP64 attention over widened inputs, one KV head shared by four query heads. */
    static double[] attention(float[] keys, float[] values, float[] queries, int rows) {
        double[] output = new double[QUERY_HEADS * WIDTH];
        double[] logits = new double[rows];
        for (int h = 0; h < QUERY_HEADS; h++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int t = 0; t < rows; t++) {
                double dot = 0;
                for (int d = 0; d < WIDTH; d++) dot += (double) queries[h * WIDTH + d] * keys[t * WIDTH + d];
                logits[t] = dot / StrictMath.sqrt(WIDTH);
                max = Math.max(max, logits[t]);
            }
            double sum = 0;
            for (int t = 0; t < rows; t++) {
                double w = StrictMath.exp(logits[t] - max);
                sum += w;
                for (int d = 0; d < WIDTH; d++) output[h * WIDTH + d] += w * values[t * WIDTH + d];
            }
            for (int d = 0; d < WIDTH; d++) output[h * WIDTH + d] /= sum;
        }
        return output;
    }

    static void run(Fixture fixture, long seed, Measurements[][] results) {
        SplittableRandom rng = new SplittableRandom(seed);
        float[] queries = new float[QUERY_HEADS * WIDTH];
        for (int i = 0; i < queries.length; i++) {
            double scale = i / WIDTH < QUERY_HEADS / 2 ? 1 : 2;
            queries[i] = half(scale * rng.nextGaussian());
        }
        float[] keys = samples(fixture, rng.split()), values = samples(fixture, rng.split());
        float[] restoredKeys = new float[keys.length], restoredValues = new float[values.length];
        double[][] baseline = new double[LENGTHS.length][];
        for (int s = 0; s < LENGTHS.length; s++) baseline[s] = attention(keys, values, queries, LENGTHS[s]);
        for (Codec codec : Codec.values()) {
            Packed k = new Packed(codec, true), v = new Packed(codec, false);
            for (int offset = 0; offset < keys.length; offset += WIDTH) {
                k.encode(keys, offset);
                v.encode(values, offset);
                k.decode(restoredKeys, offset);
                v.decode(restoredValues, offset);
            }
            for (int s = 0; s < LENGTHS.length; s++) {
                Measurements m = results[codec.ordinal()][s];
                for (int i = 0; i < LENGTHS[s] * WIDTH; i++) {
                    m.keys.add(keys[i], restoredKeys[i]);
                    m.values.add(values[i], restoredValues[i]);
                }
                double[] actual = attention(restoredKeys, restoredValues, queries, LENGTHS[s]);
                double[] keyOnly = attention(restoredKeys, values, queries, LENGTHS[s]);
                double[] valueOnly = attention(keys, restoredValues, queries, LENGTHS[s]);
                for (int i = 0; i < actual.length; i++) {
                    m.attention.add(baseline[s][i], actual[i]);
                    m.keyOnly.add(baseline[s][i], keyOnly[i]);
                    m.valueOnly.add(baseline[s][i], valueOnly[i]);
                }
            }
        }
    }

    static void sanity() {
        float[] x = new float[WIDTH], restored = new float[WIDTH];
        for (Codec codec : Codec.values()) {
            Packed k = new Packed(codec, true), v = new Packed(codec, false);
            int expected = codec == Codec.AFFINE_K8_V4 ? 224 : 216;
            if (k.bytes() + v.bytes() != expected) throw new AssertionError("payload bytes");
            for (int i = 0; i < WIDTH; i++) x[i] = i % 16;
            v.encode(x, 0);
            v.decode(restored, 0);
            if ((v.codes[0] & 255) != 0x10 || !Arrays.equals(x, restored)) throw new AssertionError("v4 ramp");
            Arrays.fill(x, 0);
            k.encode(x, 0);
            k.decode(restored, 0);
            if (!Arrays.equals(x, restored)) throw new AssertionError("k8 zero block");
            for (int i = 0; i < WIDTH; i++) x[i] = (i % 2 == 0) ? -127 : 127;
            k.encode(x, 0);
            k.decode(restored, 0);
            if (restored[0] != -127 || Math.abs(restored[1] - 127) > 0.1) throw new AssertionError("k8 endpoints");
        }
        float[] queries = new float[QUERY_HEADS * WIDTH];
        Arrays.fill(x, -2);
        for (double value : attention(x, x, queries, 1)) if (value != -2) throw new AssertionError("one-token attention");
    }

    public static void main(String[] args) {
        sanity();
        System.out.println("fixture,tokens,codec,bytes_per_head_token,k_rmse,k_max,v_rmse,v_max,output_rmse,"
                + "output_relative_l2,output_max,key_only_output_relative_l2,value_only_output_relative_l2");
        for (Fixture fixture : Fixture.values()) {
            Measurements[][] results = new Measurements[Codec.values().length][LENGTHS.length];
            for (var group : results) Arrays.setAll(group, unused -> new Measurements());
            for (long seed : SEEDS) run(fixture, seed, results);
            for (int s = 0; s < LENGTHS.length; s++) {
                for (Codec codec : Codec.values()) {
                    Measurements m = results[codec.ordinal()][s];
                    int bytes = new Packed(codec, true).bytes() + new Packed(codec, false).bytes();
                    System.out.printf(Locale.ROOT, "%s,%d,%s,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g%n",
                            fixture, LENGTHS[s], codec, bytes, m.keys.rmse(), m.keys.maxAbsolute,
                            m.values.rmse(), m.values.maxAbsolute, m.attention.rmse(),
                            m.attention.relativeL2(), m.attention.maxAbsolute, m.keyOnly.relativeL2(),
                            m.valueOnly.relativeL2());
                }
            }
        }
    }
}
