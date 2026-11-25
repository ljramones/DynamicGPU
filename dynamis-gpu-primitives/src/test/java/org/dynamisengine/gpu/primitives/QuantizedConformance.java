package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.primitives.QuantizedMatrix.Format;
import org.junit.jupiter.api.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.EnumSet;
import java.util.List;
import java.util.SplittableRandom;
import static org.junit.jupiter.api.Assertions.*;

/** Quantized weights on a real device (1.9.0): the GPU quantizer and dequantizer bit for bit against the CPU's. */
abstract class QuantizedConformance {
    protected abstract GpuProvider provider();
    protected GpuDevice device;
    static final Duration WAIT = Duration.ofSeconds(60);

    @BeforeEach void open() { device = provider().adapters().getFirst().open(); }
    @AfterEach void close() { device.close(); }

    GpuBuffer buffer(long bytes) {
        return device.allocate(new BufferDescriptor(bytes, EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE));
    }

    static byte[] bytes(float[] f) { var b = ByteBuffer.allocate(4 * f.length).order(ByteOrder.LITTLE_ENDIAN); for (float v : f) b.putFloat(v); return b.array(); }

    /** Weights with awkward blocks: ties at half a step, zero blocks, a lone extreme, equal magnitudes of both signs. */
    static float[] weights(QuantizedMatrix m, SplittableRandom r) {
        int n = m.rows() * m.cols();
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            int block = i / m.block(), kind = block % 6;
            double v = r.nextDouble() * 2 - 1;
            x[i] = switch (kind) {
                case 0 -> (float) v;
                case 1 -> (float) (v * 1e-3);
                case 2 -> 0f;                                            // an all-zero block
                case 3 -> i % m.block() == 5 ? 100f : (float) v;         // a lone extreme
                case 4 -> (i % 2 == 0 ? 1 : -1) * (float) Math.rint(v * 16) / 16f;   // many ties, equal magnitudes
                default -> (float) (v * 3e4);
            };
        }
        return x;
    }

    /** pack gives the CPU encoder's bytes and unpack its values, for every format and block size. */
    @Test void packsAndUnpacksBitForBit() {
        var r = new SplittableRandom(17);
        try (var q = Quantize.create(device)) {
            for (var format : Format.values()) {
                for (int block : format.kQuant() ? new int[] {256} : new int[] {32, 64, 128}) {
                    for (int[] shape : new int[][] {{7, 3 * block}, {64, 4 * block}, {1, block}}) {
                        var m = new QuantizedMatrix(format, block, shape[0], shape[1]);
                        float[] x = weights(m, r);
                        byte[] expected = m.quantize(x);
                        float[] values = m.dequantize(expected);
                        byte[] packed, unpacked;
                        try (var gx = buffer(4L * x.length); var gp = buffer(m.bytes()); var gy = buffer(4L * x.length); var e = device.encoder()) {
                            e.upload(gx, bytes(x)).fill(gp, 0xDEADBEEF);
                            if (format.kQuant()) e.upload(gp, expected);   // no GPU encoder: unpack the CPU's blocks
                            else q.pack(e, gx, m, gp);
                            q.unpack(e, m, gp, gy);
                            var rp = e.download(gp); var ry = e.download(gy);
                            try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
                            packed = rp.bytes(); unpacked = ry.bytes();
                        }
                        String what = format + "/" + block + " " + shape[0] + "x" + shape[1];
                        int differ = 0, first = -1;
                        for (int i = 0; i < expected.length; i++) if (packed[i] != expected[i]) { differ++; if (first < 0) first = i; }
                        assertEquals(0, differ, what + ": bytes differing from QuantizedMatrix.quantize, first at " + first);
                        var got = ByteBuffer.wrap(unpacked).order(ByteOrder.LITTLE_ENDIAN);
                        for (int i = 0; i < values.length; i++)
                            assertEquals(Float.floatToRawIntBits(values[i]), Float.floatToRawIntBits(got.getFloat(4 * i)), what + ": value " + i);
                    }
                }
            }
        }
    }

    static byte[] halfBytes(short[] h) {
        var b = ByteBuffer.allocate(2 * (h.length + (h.length & 1))).order(ByteOrder.LITTLE_ENDIAN);
        for (short v : h) b.putShort(v);
        return b.array();
    }

    /** Activations of {@code input} (FP16 or BF16) as raw bits, random in [-1, 1]. */
    static short[] activations(GemmF16.Input input, int count, SplittableRandom r) {
        short[] h = new short[count];
        for (int i = 0; i < count; i++) {
            float v = (float) (r.nextDouble() * 2 - 1);
            h[i] = input == GemmF16.Input.BF16 ? GemmF16.floatToBf16(v) : Float.floatToFloat16(v);
        }
        return h;
    }

    /** Runs {@code g} on {@code s} and checks every element of C against QuantizedGemm.reference, bit for bit. */
    void checkQuantized(QuantizedGemm g, QuantizedMatrix w, GemmF16.Shape s, GemmF16.Epilogue ep, long seed) {
        var r = new SplittableRandom(seed);
        var in = g.input(); var out = g.output();
        boolean fp32 = out == Precision.FP32;
        short[] a = activations(in, (int) s.aElements() + 1, r);
        var elements = in == GemmF16.Input.BF16 ? GemmF16.Elements.bf16(a) : GemmF16.Elements.fp16(a);
        float[] values = new float[w.rows() * w.cols()];
        for (int i = 0; i < values.length; i++) values[i] = (float) (r.nextDouble() * 2 - 1);
        byte[] packed = w.quantize(values);
        int cLen = (int) s.cElements() + 1, biasLen = ep.bias() == GemmF16.Epilogue.Bias.ROW ? s.m() : s.n();
        float[] c = new float[cLen], bias = new float[biasLen + 1], residual = new float[cLen];
        for (float[] f : new float[][] {c, bias, residual}) for (int i = 0; i < f.length; i++) f[i] = Float.float16ToFloat(Float.floatToFloat16((float) (r.nextDouble() * 2 - 1)));
        byte[] cb = fp32 ? bytes(c) : halfBytes(toHalves(c)), bb = fp32 ? bytes(bias) : halfBytes(toHalves(bias)), rb = fp32 ? bytes(residual) : halfBytes(toHalves(residual));
        byte[] ab = halfBytes(a);
        byte[] got;
        try (var ga = buffer(ab.length); var gw = buffer(packed.length); var gc = buffer(cb.length); var gbias = buffer(bb.length); var gres = buffer(rb.length);
             var e = device.encoder()) {
            e.upload(ga, ab).upload(gw, packed).upload(gc, cb).upload(gbias, bb).upload(gres, rb);
            g.encode(e, ga, w, gw, gc, s, ep, ep.bias() == GemmF16.Epilogue.Bias.NONE ? null : gbias, null, ep.residual() ? gres : null);
            var rc = e.download(gc);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
            got = rc.bytes();
        }
        var alg = g.algorithm(s);
        var bufG = ByteBuffer.wrap(got).order(ByteOrder.LITTLE_ENDIAN);
        String what = alg + " " + w + " " + in + " " + out + " " + ep + " " + s.m() + "x" + s.n() + "x" + s.k() + "x" + s.batch();
        int differ = 0, first = -1;
        if (fp32) {
            float[] expect = c.clone();
            QuantizedGemm.reference(alg, elements, w, packed, expect, s, ep, bias, null, residual);
            for (int i = 0; i < cLen; i++) if (Float.floatToRawIntBits(expect[i]) != Float.floatToRawIntBits(bufG.getFloat(4 * i))) { differ++; if (first < 0) first = i; }
        } else {
            short[] expect = toHalves(c);
            QuantizedGemm.referenceF16(alg, elements, w, packed, expect, s, ep, toHalves(bias), null, toHalves(residual));
            for (int i = 0; i < cLen; i++) if (expect[i] != bufG.getShort(2 * i)) { differ++; if (first < 0) first = i; }
        }
        assertEquals(0, differ, what + ": elements differing from QuantizedGemm.reference, first at " + first);
        // and the reference is near the FP32 product of A and the dequantized weights
        if (ep.equals(GemmF16.Epilogue.NONE) && s.beta() == 0 && fp32) {
            float[] dq = w.dequantize(packed);
            for (int z = 0; z < s.batch(); z++) for (int i = 0; i < s.m(); i++) for (int j = 0; j < s.n(); j++) {
                double exact = 0, abs = 0;
                for (int q = 0; q < s.k(); q++) { double t = (double) elements.at(s.aIndex(z, i, q)) * dq[j * s.k() + q]; exact += t; abs += Math.abs(t); }
                double v = bufG.getFloat(4 * (int) s.cIndex(z, i, j));
                assertTrue(Math.abs(v - s.alpha() * exact) <= (s.k() + 2) * 0x1p-24 * Math.abs(s.alpha()) * abs * 2 + 1e-30, what + " C[" + i + "][" + j + "]");
            }
        }
    }

    static short[] toHalves(float[] f) { short[] h = new short[f.length]; for (int i = 0; i < f.length; i++) h[i] = Float.floatToFloat16(f[i]); return h; }

    /** Every format, block size, activation and output type, both algorithms, with an epilogue, beta and a batch. */
    @Test void quantizedGemmMatchesTheReference() {
        var none = GemmF16.Epilogue.NONE;
        var epilogue = none.withBias(GemmF16.Epilogue.Bias.COLUMN).withActivation(GemmF16.Epilogue.Activation.RELU).withResidual();
        long seed = 1;
        for (var in : new GemmF16.Input[] {GemmF16.Input.FP16, GemmF16.Input.BF16}) {
            for (var out : Precision.values()) {
                for (var alg : QuantizedGemm.Algorithm.values()) {
                    try (var g = QuantizedGemm.create(device, alg, in, out)) {
                        for (var format : Format.values()) {
                            for (int block : format.kQuant() ? new int[] {256} : new int[] {32, 64, 128}) {
                                int k = format.kQuant() ? 512 : 4 * block;
                                var w = new QuantizedMatrix(format, block, 37, k);
                                checkQuantized(g, w, GemmF16.Shape.ofTransposedB(1, 37, k, 1f, 0f), none, seed++);
                                checkQuantized(g, w, GemmF16.Shape.ofTransposedB(3, 37, k, -0.5f, 0.75f), epilogue, seed++);
                                checkQuantized(g, w, GemmF16.Shape.ofTransposedB(2, 37, k, 1f, 0f).withBatch(3, 2L * k + 3, 0, 2 * 37 + 5), none, seed++);
                                if (alg == QuantizedGemm.Algorithm.TILED) {
                                    var big = new QuantizedMatrix(format, block, 65, 2 * k);
                                    checkQuantized(g, big, GemmF16.Shape.ofTransposedB(70, 65, 2 * k, 1f, 0f), none, seed++);
                                    checkQuantized(g, big, new GemmF16.Shape(33, 65, 2 * k, 2 * k + 1, 2 * k, 67, 1f, 0.5f, true, 1, 0, 0, 0), epilogue, seed++);
                                }
                            }
                        }
                    }
                }
            }
        }
        // decode-sized: the 32-lane split over many blocks
        try (var g = QuantizedGemm.create(device, GemmF16.Input.FP16, Precision.FP32)) {
            for (var format : Format.values())
                checkQuantized(g, new QuantizedMatrix(format, format.kQuant() ? 256 : 32, 96, 4096), GemmF16.Shape.ofTransposedB(4, 96, 4096, 1f, 0f), none, seed++);
            assertEquals(QuantizedGemm.Algorithm.SKINNY, g.algorithm(GemmF16.Shape.ofTransposedB(16, 96, 4096, 1f, 0f)));
            assertEquals(QuantizedGemm.Algorithm.TILED, g.algorithm(GemmF16.Shape.ofTransposedB(17, 96, 4096, 1f, 0f)));
        }
    }

    /**
     * -Dquant.decode=true: decode with quantized weights against FP16 weights (GemmF16 SKINNY), as GB/s of the
     * weights' bytes and time per product. The quantized weights rotate through copies over 256 MiB, so the system
     * cache cannot serve them; the FP16 weights (32 MiB and more) are measured as before.
     */
    @Test void decodeBandwidth() {
        Assumptions.assumeTrue(Boolean.getBoolean("quant.decode"), "set -Dquant.decode=true");
        int[][] shapes = Boolean.getBoolean("quant.short") ? new int[][] {{4096, 4096}} : new int[][] {{4096, 4096}, {11008, 4096}, {4096, 11008}};
        try (var q = QuantizedGemm.create(device, GemmF16.Input.FP16, Precision.FP32); var f = GemmF16.create(device, GemmF16.Algorithm.SKINNY, Precision.FP32)) {
            // warm up: the GPU's clocks ramp up under load, and the first shape measured otherwise runs slow
            var warm = GemmF16.Shape.ofTransposedB(1, 11008, 4096, 1f, 0f);
            try (var ga = buffer(2L * 4096); var gb = buffer(warm.bBytes()); var gc = buffer(warm.cBytes())) {
                long end = System.nanoTime() + 2_000_000_000L;
                while (System.nanoTime() < end) nanos(e -> f.encode(e, ga, gb, gc, warm));
            }
            for (int[] nk : shapes) {
                for (int m : java.util.Arrays.stream(System.getProperty("quant.m", "1,4,16,64,512").split(",")).mapToInt(Integer::parseInt).toArray()) {
                    var s = GemmF16.Shape.ofTransposedB(m, nk[0], nk[1], 1f, 0f);
                    double fp16 = -1;
                    if (m <= 16) try (var ga = buffer(2L * m * nk[1]); var gb = buffer(s.bBytes()); var gc = buffer(s.cBytes()); var e0 = device.encoder()) {
                        e0.fill(ga, 0x3C003C00).fill(gb, 0x3C003C00);
                        try (var sub = device.queue().submit(e0)) { assertTrue(sub.await(WAIT)); }
                        fp16 = nanos(e -> f.encode(e, ga, gb, gc, s));
                    }
                    var line = new StringBuilder(String.format("qgemm decode %s m=%d n=%d k=%d: FP16 %s", device.adapter().name(), m, nk[0], nk[1],
                            fp16 < 0 ? "-" : String.format("%.1f us", fp16 / 1e3)));
                    for (var w : new QuantizedMatrix[] {new QuantizedMatrix(Format.Q8, 32, nk[0], nk[1]), new QuantizedMatrix(Format.Q4, 32, nk[0], nk[1]),
                            new QuantizedMatrix(Format.Q4M, 32, nk[0], nk[1]), new QuantizedMatrix(Format.Q4M, 128, nk[0], nk[1]),
                            new QuantizedMatrix(Format.Q4K, 256, nk[0], nk[1]), new QuantizedMatrix(Format.Q5K, 256, nk[0], nk[1]),
                            new QuantizedMatrix(Format.Q6K, 256, nk[0], nk[1])}) {
                        var format = w.format() + (w.block() == 128 ? "/128" : "");
                        if (System.getProperty("quant.formats") != null && !java.util.List.of(System.getProperty("quant.formats").split(",")).contains(format)) continue;
                        // copies of the weights to rotate through, over 256 MiB, so the system cache cannot serve them
                        int copies = (int) Math.max(1, (256L << 20) / w.bytes() + 1);
                        var gws = new java.util.ArrayList<GpuBuffer>();
                        try (var ga = buffer(2L * m * nk[1] + 4); var gc = buffer(s.cBytes())) {
                            for (int i = 0; i < copies; i++) gws.add(buffer(w.bytes()));
                            try (var e = device.encoder()) {
                                e.fill(ga, 0x3C003C00);
                                for (var gw : gws) e.fill(gw, 0);
                                try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
                            }
                            int[] next = {0};
                            double ns = nanos(e -> q.encode(e, ga, w, gws.get(next[0]++ % copies), gc, s));
                            gws.forEach(GpuBuffer::close);
                            line.append(String.format(", %s %.1f us (%.0f GB/s%s)", format, ns / 1e3, w.bytes() / ns,
                                    fp16 > 0 ? String.format(", %.2fx FP16", fp16 / ns) : ""));
                        }
                    }
                    System.out.println(line);
                }
            }
        }
    }

    /**
     * GPU nanoseconds per run of {@code body}: an untimed round, then the best of 4 rounds sized to about 20 ms of GPU
     * work, so the GPU's clocks are up (rounds of 1 ms after the host-side setup between formats ran up to twice slow).
     */
    double nanos(java.util.function.Consumer<GpuCommandEncoder> body) {
        double first = round(body, 20);
        int reps = (int) Math.max(20, Math.min(2000, 20e6 / first));
        double best = Double.MAX_VALUE;
        for (int r = 0; r < 4; r++) best = Math.min(best, round(body, reps));
        return best;
    }

    private double round(java.util.function.Consumer<GpuCommandEncoder> body, int reps) {
        try (var e = device.encoder()) {
            for (int i = 0; i < reps; i++) body.accept(e);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); return sub.gpuTime().orElseThrow().toNanos() / (double) reps; }
        }
    }

    /**
     * -Dgguf.samples=file (written by the GGML oracle script from real models): their blocks, imported, dequantize on
     * the GPU to gguf-py's values bit for bit, and multiply bit for bit as QuantizedGemm.reference.
     */
    @Test void realModelBlocks() throws Exception {
        String path = System.getProperty("gguf.samples");
        Assumptions.assumeTrue(path != null, "set -Dgguf.samples=<file>");
        List<GgufBlocksTest.Sample> samples;
        try (var in = java.nio.file.Files.newInputStream(java.nio.file.Path.of(path))) { samples = GgufBlocksTest.read(in); }
        var r = new SplittableRandom(3);
        try (var q = Quantize.create(device); var g = QuantizedGemm.create(device, GemmF16.Input.FP16, Precision.FP32)) {
            for (var sample : samples) {
                var packed = GgufBlocks.fromGgml(sample.type(), sample.raw(), 0, sample.rows(), sample.cols());
                var m = packed.matrix();
                byte[] got;
                try (var gp = buffer(m.bytes()); var gy = buffer(4L * m.rows() * m.cols()); var e = device.encoder()) {
                    e.upload(gp, packed.bytes());
                    q.unpack(e, m, gp, gy);
                    var ry = e.download(gy);
                    try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
                    got = ry.bytes();
                }
                var values = ByteBuffer.wrap(got).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < sample.values().length; i++)
                    assertEquals(Float.floatToRawIntBits(sample.values()[i]), Float.floatToRawIntBits(values.getFloat(4 * i)), sample.type() + " value " + i);
                // a decode and a prefill product with the real weights
                for (int rows : new int[] {1, 20}) {
                    short[] a = activations(GemmF16.Input.FP16, rows * m.cols() + 1, r);
                    var shape = GemmF16.Shape.ofTransposedB(rows, m.rows(), m.cols(), 1f, 0f);
                    float[] expect = new float[rows * m.rows()];
                    QuantizedGemm.reference(g.algorithm(shape), GemmF16.Elements.fp16(a), m, packed.bytes(), expect, shape, GemmF16.Epilogue.NONE, null, null, null);
                    try (var ga = buffer(halfBytes(a).length); var gw = buffer(m.bytes()); var gc = buffer(4L * expect.length); var e = device.encoder()) {
                        e.upload(ga, halfBytes(a)).upload(gw, packed.bytes());
                        g.encode(e, ga, m, gw, gc, shape);
                        var rc = e.download(gc);
                        try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
                        var cv = ByteBuffer.wrap(rc.bytes()).order(ByteOrder.LITTLE_ENDIAN);
                        for (int i = 0; i < expect.length; i++)
                            assertEquals(Float.floatToRawIntBits(expect[i]), Float.floatToRawIntBits(cv.getFloat(4 * i)), sample.type() + " " + g.algorithm(shape) + " C[" + i + "]");
                    }
                }
            }
            System.out.println("quantized real samples on " + device.adapter().name() + ": " + samples.size());
        }
    }

    /** A non-finite value, or a scale beyond FP16, fails the submission. */
    @Test void refusesNonFiniteValues() {
        var m = new QuantizedMatrix(Format.Q8, 32, 2, 32);
        try (var q = Quantize.create(device)) {
            for (float bad : new float[] {Float.NaN, Float.POSITIVE_INFINITY, 1e30f}) {
                float[] x = new float[64];
                x[40] = bad;
                try (var gx = buffer(256); var gp = buffer(m.bytes()); var e = device.encoder()) {
                    e.upload(gx, bytes(x));
                    q.pack(e, gx, m, gp);
                    var s = device.queue().submit(e);
                    assertEquals(ComputeException.Code.KERNEL_ERROR, assertThrows(ComputeException.class, () -> s.await(WAIT)).code(), "value " + bad);
                    s.close();
                }
            }
        }
    }
}
