package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.junit.jupiter.api.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.EnumSet;
import java.util.List;
import java.util.SplittableRandom;
import static org.junit.jupiter.api.Assertions.*;

/** The inference kernels (1.10.0) on a real device: norms, RoPE, embedding, sampling, activations, log-sum-exp. */
abstract class InferenceConformance {
    protected abstract GpuProvider provider();
    protected GpuDevice device;
    static final Duration WAIT = Duration.ofSeconds(60);

    @BeforeEach void open() { device = provider().adapters().getFirst().open(); }
    @AfterEach void close() { device.close(); }

    GpuBuffer buffer(long bytes) {
        return device.allocate(new BufferDescriptor(Math.max(4, (bytes + 3) & ~3L), EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE,
                BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE));
    }

    /** The bytes of {@code v} as {@code p} (FP16 rounded), padded to whole words. */
    static byte[] encode(float[] v, Precision p) {
        int n = p == Precision.FP16 ? v.length + (v.length & 1) : v.length;
        var b = ByteBuffer.allocate(p.bytes() * n).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : v) { if (p == Precision.FP16) b.putShort(Float.floatToFloat16(f)); else b.putFloat(f); }
        return b.array();
    }
    static float[] decode(byte[] bytes, Precision p, int count) {
        var b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[count];
        for (int i = 0; i < count; i++) out[i] = p == Precision.FP16 ? Float.float16ToFloat(b.getShort(2 * i)) : b.getFloat(4 * i);
        return out;
    }
    static byte[] ints(int[] v) { var b = ByteBuffer.allocate(4 * v.length).order(ByteOrder.LITTLE_ENDIAN); for (int x : v) b.putInt(x); return b.array(); }
    static float[] values(int n, SplittableRandom r, double range, Precision p) {
        float[] f = new float[n];
        for (int i = 0; i < n; i++) { f[i] = (float) ((r.nextDouble() * 2 - 1) * range); if (p == Precision.FP16) f[i] = Float.float16ToFloat(Float.floatToFloat16(f[i])); }
        return f;
    }
    static float round(float v, Precision p) { return p == Precision.FP16 ? Float.float16ToFloat(Float.floatToFloat16(v)) : v; }

    byte[] run(java.util.function.Function<GpuCommandEncoder, GpuReadback> body) {
        try (var e = device.encoder()) {
            var read = body.apply(e);
            try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
            return read.bytes();
        }
    }

    // --- norms -----------------------------------------------------------------------------------------------------

    @Test void normsWithinTheirBound() {
        var r = new SplittableRandom(1);
        for (var in : Precision.values()) for (var out : Precision.values()) {
            try (var norms = Norms.create(device, in, out)) {
                for (var kind : Norms.Kind.values()) for (int n : new int[] {1, 7, 300, 4096}) for (int mode = 0; mode < 3; mode++) {   // plain, residual into h, in place
                    int rows = 3;
                    var s = Norms.Shape.of(rows, n, 1e-5f);
                    float[] x = values(rows * n, r, 2, in), res = values(rows * n, r, 1, in), w = values(n, r, 1.5, Precision.FP32),
                            b = kind == Norms.Kind.LAYER ? values(n, r, 0.5, Precision.FP32) : null;
                    byte[][] got = new byte[2][];
                    try (var gx = buffer(encode(x, in).length); var gr = buffer(encode(res, in).length); var gh = buffer(encode(x, in).length);
                         var gw = buffer(4L * n); var gb = buffer(4L * n); var gy = buffer(encode(x, out).length); var e = device.encoder()) {
                        e.upload(gx, encode(x, in)).upload(gr, encode(res, in)).upload(gw, encode(w, Precision.FP32)).upload(gy, encode(new float[rows * n], out));
                        if (b != null) e.upload(gb, encode(b, Precision.FP32));
                        var h = mode == 0 ? null : mode == 1 ? gh : gx;
                        norms.encode(e, kind, gx, mode == 0 ? null : gr, h, gw, b == null ? null : gb, gy, s);
                        var ry = e.download(gy); var rh = e.download(mode == 2 ? gx : gh);
                        try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
                        got[0] = ry.bytes(); got[1] = rh.bytes();
                    }
                    float[] y = decode(got[0], out, rows * n), h = decode(got[1], in, rows * n);
                    float[] v = x.clone();
                    if (mode != 0) for (int i = 0; i < v.length; i++) v[i] = Norms.residualSum(x[i], res[i], in);
                    if (mode != 0) for (int i = 0; i < v.length; i++) assertEquals(Float.floatToRawIntBits(v[i]), Float.floatToRawIntBits(h[i]), "h is x + res exactly");
                    String what = kind + " " + in + "->" + out + " n=" + n + " mode " + mode;
                    for (int row = 0; row < rows; row++) {
                        double[] exact = Norms.exact(kind, v, row * n, n, w, b, s.eps());
                        double mean = 0, var = 0;
                        if (kind == Norms.Kind.LAYER) { for (int j = 0; j < n; j++) mean += v[row * n + j]; mean /= n; }
                        for (int j = 0; j < n; j++) { double d = v[row * n + j] - mean; var += d * d; }
                        double inv = 1 / Math.sqrt(var / n + s.eps());
                        for (int j = 0; j < n; j++) {
                            double scale = Math.abs(w[j] * (v[row * n + j] - mean) * inv) + (b == null ? 0 : Math.abs(b[j]));
                            double bound = (16 + n / 256.0) * 0x1p-24 * scale + 1e-30 + (out == Precision.FP16 ? Math.abs(exact[j]) * 0x1p-10 + 0x1p-24 : 0);
                            assertTrue(Math.abs(y[row * n + j] - exact[j]) <= bound, what + " y[" + row + "][" + j + "] = " + y[row * n + j] + ", exact " + exact[j]);
                        }
                    }
                }
            }
        }
    }

    // --- RoPE ------------------------------------------------------------------------------------------------------

    @Test void ropeIsBitForBit() {
        var r = new SplittableRandom(2);
        var table = Rope.Table.standard(8192, 96, 10000);
        var partial = Rope.Table.scaled(8192, 32, 500000, 4);
        for (var in : Precision.values()) for (var out : Precision.values()) {
            try (var rope = Rope.create(device, in, out); var gt = buffer(4L * table.cosSin().length); var gp = buffer(4L * partial.cosSin().length)) {
                run(e -> { e.upload(gt, table.bytes()).upload(gp, partial.bytes()); return e.download(gt); });
                for (var layout : Rope.Layout.values()) for (boolean inPlace : in == out ? new boolean[] {false, true} : new boolean[] {false}) {
                    for (var cs : new Object[][] {{Rope.Shape.of(5, 4, 96, layout), table, gt},
                            {new Rope.Shape(3, 2, 64, 32, 2 * 64 + 8, 2 * 64 + 8, 64, layout), partial, gp}}) {
                        var s = (Rope.Shape) cs[0]; var tab = (Rope.Table) cs[1]; var gtab = (GpuBuffer) cs[2];
                        int len = (int) s.elements(s.ldx());
                        float[] x = values(len, r, 3, in);
                        int[] pos = new int[s.tokens()];
                        for (int t = 0; t < pos.length; t++) pos[t] = t == 0 ? 8191 : r.nextInt(8192);
                        byte[] got;
                        try (var gx = buffer(encode(x, in).length); var gpos = buffer(4L * pos.length); var gy = buffer(encode(x, out).length)) {
                            got = run(e -> {
                                e.upload(gx, encode(x, in)).upload(gpos, ints(pos)).upload(gy, encode(x, out));
                                rope.encode(e, gx, gpos, tab, gtab, inPlace ? gx : gy, s);
                                return e.download(inPlace ? gx : gy);
                            });
                        }
                        float[] expect = new float[len];
                        for (int i = 0; i < len; i++) expect[i] = round(x[i], out);   // untouched gaps keep the input
                        Rope.reference(x, pos, tab, expect, s, out);
                        float[] y = decode(got, out, len);
                        for (int i = 0; i < len; i++)
                            assertEquals(Float.floatToRawIntBits(expect[i]), Float.floatToRawIntBits(y[i]), layout + " " + in + "->" + out + " in place " + inPlace + " element " + i);
                    }
                }
                // a position past the table fails the submission
                try (var gx = buffer(4L * 96 * 4); var gy = buffer(4L * 96 * 4); var gpos = buffer(4)) {
                    var ex = assertThrows(ComputeException.class, () -> run(e -> {
                        e.upload(gpos, ints(new int[] {8192}));
                        rope.encode(e, gx, gpos, table, gt, gy, Rope.Shape.of(1, 4, 96, Rope.Layout.HALVES));
                        return e.download(gpos);
                    }));
                    assertEquals(ComputeException.Code.KERNEL_ERROR, ex.code());
                }
            }
        }
    }

    // --- embedding -------------------------------------------------------------------------------------------------

    @Test void embeddingIsBitForBit() {
        var r = new SplittableRandom(3);
        int vocab = 300, dim = 512;
        float[] values = values(vocab * dim, r, 1, Precision.FP32);
        var q8 = new QuantizedMatrix(QuantizedMatrix.Format.Q8, 32, vocab, dim);
        var q4k = new QuantizedMatrix(QuantizedMatrix.Format.Q4K, 256, vocab, dim);
        var q6k = new QuantizedMatrix(QuantizedMatrix.Format.Q6K, 256, vocab, dim);
        int[] tokens = {0, 299, 5, 5, 17, 123};
        for (var out : Precision.values()) {
            try (var emb = Embedding.create(device, out)) {
                for (Object[] c : new Object[][] {{Embedding.Table.of(vocab, dim, Precision.FP32), encode(values, Precision.FP32)},
                        {Embedding.Table.of(vocab, dim, Precision.FP16), encode(values, Precision.FP16)},
                        {Embedding.Table.quantized(q8), q8.quantize(values)}, {Embedding.Table.quantized(q4k), q4k.quantize(values)},
                        {Embedding.Table.quantized(q6k), q6k.quantize(values)}}) {
                    var tab = (Embedding.Table) c[0]; byte[] bytes = (byte[]) c[1];
                    byte[] got;
                    try (var gt = buffer(bytes.length); var gtok = buffer(4L * tokens.length); var gy = buffer((long) out.bytes() * tokens.length * dim)) {
                        byte[] padded = java.util.Arrays.copyOf(bytes, (bytes.length + 3) & ~3);
                        got = run(e -> {
                            e.upload(gt, padded).upload(gtok, ints(tokens));
                            emb.encode(e, gtok, tokens.length, tab, gt, gy, dim);
                            return e.download(gy);
                        });
                    }
                    float[] y = decode(got, out, tokens.length * dim);
                    for (int t = 0; t < tokens.length; t++) for (int j = 0; j < dim; j++) {
                        int row = tokens[t];
                        float v = tab.quantized() != null ? tab.quantized().value(bytes, row, j)
                                : tab.precision() == Precision.FP16 ? round(values[row * dim + j], Precision.FP16) : values[row * dim + j];
                        assertEquals(Float.floatToRawIntBits(round(v, out)), Float.floatToRawIntBits(y[t * dim + j]), tab + " token " + t + " element " + j);
                    }
                }
                // a token past the vocabulary fails the submission
                try (var gt = buffer(4L * vocab * dim); var gtok = buffer(4); var gy = buffer(4L * dim)) {
                    var ex = assertThrows(ComputeException.class, () -> run(e -> {
                        e.upload(gtok, ints(new int[] {vocab}));
                        emb.encode(e, gtok, 1, Embedding.Table.of(vocab, dim, Precision.FP32), gt, gy, dim);
                        return e.download(gtok);
                    }));
                    assertEquals(ComputeException.Code.KERNEL_ERROR, ex.code());
                }
            }
        }
    }

    // --- sampling --------------------------------------------------------------------------------------------------

    int[] sample(Sampler sampler, float[] logits, Sampler.Shape s, Sampler.Settings st, int step) {
        try (var gl = buffer(4L * logits.length); var gt = buffer(4L * s.rows())) {
            byte[] got = run(e -> { e.upload(gl, encode(logits, Precision.FP32)); sampler.encode(e, gl, gt, s, st, step); return e.download(gt); });
            var b = ByteBuffer.wrap(got).order(ByteOrder.LITTLE_ENDIAN);
            int[] out = new int[s.rows()];
            for (int i = 0; i < out.length; i++) out[i] = b.getInt(4 * i);
            return out;
        }
    }

    /** Greedy is exact; each mode agrees with the reference (but at boundaries, rarely) and repeats for the same seed. */
    @Test void samplingMatchesTheReference() {
        var r = new SplittableRandom(4);
        try (var sampler = Sampler.create(device)) {
            int rows = 4;
            // the last case: integer logits, so most of the k largest tie and their order (by index, across chunks and
            // merge rounds) decides the tokens (1.10.1)
            for (int vocab : new int[] {7, 1000, 32000, 128256, -128256}) {
                boolean tied = vocab < 0;
                vocab = Math.abs(vocab);
                var s = Sampler.Shape.of(rows, vocab);
                float[] logits = values(rows * vocab, r, 6, Precision.FP32);
                if (tied) for (int i = 0; i < logits.length; i++) logits[i] = r.nextInt(8);
                logits[3] = logits[vocab + 9] = 50f; logits[vocab + 2] = 50f;        // ties at the max: the first wins
                logits[2 * vocab + 1] = Float.NaN; logits[2 * vocab + 4] = Float.NEGATIVE_INFINITY;
                var settings = new Sampler.Settings[] {Sampler.Settings.greedy(), new Sampler.Settings(0.8f, 0, 1f, 42),
                        new Sampler.Settings(1f, 40, 1f, 7), new Sampler.Settings(0.7f, 0, 0.9f, 9), new Sampler.Settings(1.2f, 50, 0.95f, 11),
                        new Sampler.Settings(1f, 1024, 1f, 13)};
                for (var st : settings) {
                    int mismatches = 0, draws = 0;
                    for (int step = 0; step < 12; step++) {
                        int[] got = sample(sampler, logits, s, st, step);
                        assertArrayEquals(got, sample(sampler, logits, s, st, step), "the same seed and step give the same tokens");
                        for (int row = 0; row < rows; row++) {
                            int expect = Sampler.reference(logits, row * vocab, vocab, st, row, step);
                            draws++;
                            if (got[row] != expect) mismatches++;
                            assertTrue(got[row] >= 0 && got[row] < vocab);
                            if (st.equals(Sampler.Settings.greedy())) assertEquals(expect, got[row], "greedy, vocab " + vocab + " row " + row);
                        }
                    }
                    System.out.println("sampler " + device.adapter().name() + " vocab " + vocab + (tied ? " tied" : "") + " " + st + ": " + mismatches + " of " + draws + " differ from the reference");
                    assertEquals(0, mismatches, "draws differing from the reference: " + st);
                }
            }
        }
    }

    @Test void optionalSamplerPipelinesCanBeUnsupported() {
        GpuDevice fallback = new GpuDevice() {
            @Override public GpuAdapter adapter() { return device.adapter(); }
            @Override public GpuCapabilities capabilities() { return device.capabilities(); }
            @Override public GpuBuffer allocate(BufferDescriptor d) { return device.allocate(d); }
            @Override public GpuCommandEncoder encoder() { return device.encoder(); }
            @Override public GpuQueue queue() { return device.queue(); }
            @Override public void close() { }
            @Override public GpuComputePipeline pipeline(PipelineDescriptor d) {
                if (d.entryPoint().equals("sample_bounds") || d.entryPoint().equals("sample_shuffle"))
                    throw new ComputeException(ComputeException.Code.UNSUPPORTED, "per-kernel thread limit");
                return device.pipeline(d);
            }
        };
        int vocab = 65549;
        float[] logits = values(vocab, new SplittableRandom(791), 4, Precision.FP32);
        try (var sampler = Sampler.create(fallback)) {
            for (var st : List.of(Sampler.Settings.greedy(), new Sampler.Settings(0.8f, 40, 0.95f, 1),
                    new Sampler.Settings(0.8f, 1024, 0.95f, 1))) {
                assertEquals(Sampler.reference(logits, 0, vocab, st, 0, 1),
                        sample(sampler, logits, Sampler.Shape.of(1, vocab), st, 1)[0]);
            }
        }
    }

    @Test void restrictedSamplingHonorsTiesAndPartialChunks() {
        int vocab = 65549, rows = 5;
        var shape = new Sampler.Shape(rows, vocab, vocab + 11);
        var data = new float[rows * shape.ldl()];
        for (int j = 0; j < vocab; j++) {
            data[j] = (j & 1) == 0 ? -0.0f : 0.0f;
            data[shape.ldl() + j] = j % 64 == 0 ? Float.NaN : Float.NEGATIVE_INFINITY;
            data[2 * shape.ldl() + j] = j >= vocab - 13 ? 10 : j % 7;
            data[3 * shape.ldl() + j] = j % 64 == 39 ? -10 : j % 5;
            data[4 * shape.ldl() + j] = j % 7 == 0 ? Float.NEGATIVE_INFINITY : j;
        }
        try (var sampler = Sampler.create(device)) {
            for (int k : new int[] {2, 13, 39, 40, 64, 65, 257, 1024}) {
                var st = new Sampler.Settings(0.8f, k, 0.95f, 901);
                for (int step = 0; step < 4; step++) {
                    var tokens = sample(sampler, data, shape, st, step);
                    for (int row = 0; row < rows; row++) {
                        assertEquals(Sampler.reference(data, row * shape.ldl(), vocab, st, row, step), tokens[row],
                                "top-k " + k + " row " + row + " step " + step);
                    }
                }
            }
        }
    }

    /** Over many steps the draws follow the softmax: each token's frequency within 5 standard deviations. */
    @Test void samplingFollowsTheDistribution() {
        try (var sampler = Sampler.create(device)) {
            float[] logits = {1f, 2f, 0.5f, -1f, 3f, 0f, 2.5f, -3f};
            var st = new Sampler.Settings(1f, 0, 1f, 2024);
            int steps = 3000;   // independent draws: one row each, the same logits
            int[] counts = new int[logits.length];
            float[] rows = new float[steps * logits.length];
            for (int i = 0; i < steps; i++) System.arraycopy(logits, 0, rows, i * logits.length, logits.length);
            try (var gl = buffer(4L * rows.length); var gt = buffer(4L * steps)) {
                byte[] got = run(e -> {
                    e.upload(gl, encode(rows, Precision.FP32));
                    sampler.encode(e, gl, gt, Sampler.Shape.of(steps, logits.length), st, 0);
                    return e.download(gt);
                });
                var b = ByteBuffer.wrap(got).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < steps; i++) counts[b.getInt(4 * i)]++;
            }
            double z = 0;
            for (float l : logits) z += Math.exp(l);
            for (int j = 0; j < logits.length; j++) {
                double pj = Math.exp(logits[j]) / z, sd = Math.sqrt(steps * pj * (1 - pj));
                assertTrue(Math.abs(counts[j] - steps * pj) <= 5 * sd + 1, "token " + j + ": " + counts[j] + " draws, expected " + steps * pj);
            }
        }
    }

    /** The chain stays on the GPU: the sampled token is the next embedding gather's input. */
    @Test void sampledTokensFeedTheEmbedding() {
        int vocab = 64, dim = 32;
        float[] table = new float[vocab * dim];
        for (int i = 0; i < table.length; i++) table[i] = i;
        float[] logits = new float[vocab];
        logits[37] = 10f;
        try (var sampler = Sampler.create(device); var emb = Embedding.create(device, Precision.FP32); var gl = buffer(4L * vocab);
             var gtok = buffer(4); var gtab = buffer(4L * table.length); var gy = buffer(4L * dim)) {
            byte[] got = run(e -> {
                e.upload(gl, encode(logits, Precision.FP32)).upload(gtab, encode(table, Precision.FP32));
                sampler.encode(e, gl, gtok, Sampler.Shape.of(1, vocab), Sampler.Settings.greedy(), 0);
                emb.encode(e, gtok, 1, Embedding.Table.of(vocab, dim, Precision.FP32), gtab, gy, dim);
                return e.download(gy);
            });
            float[] y = decode(got, Precision.FP32, dim);
            for (int j = 0; j < dim; j++) assertEquals(37 * dim + j, y[j]);
        }
    }

    // --- activations and log-sum-exp -------------------------------------------------------------------------------

    /**
     * The activations' backward passes (1.12.0) against their derivatives in double, within bounds carried through
     * each step of the kernels' arithmetic (absolute: the derivatives cross zero).
     */
    @Test void activationGradientsWithinTheirBound() {
        var r = new SplittableRandom(55);
        int n = 20000;
        float[] x = values(n, r, 12, Precision.FP32), dy = values(n, r, 3, Precision.FP32);
        x[0] = -1.2784645f; x[1] = -0.7517916f; x[2] = 0f; x[3] = -0f; x[4] = 40f; x[5] = -40f; x[6] = Float.MIN_NORMAL;
        final double E = 0x1p-24;
        try (var ew = Elementwise.create(device); var gx = buffer(4L * n); var gd = buffer(4L * n); var gy = buffer(4L * n)) {
            var v = StridedView.contiguous(n);
            for (var op : new Elementwise.Binary[] {Elementwise.Binary.SILU_BACKWARD, Elementwise.Binary.GELU_BACKWARD, Elementwise.Binary.RELU_BACKWARD}) {
                float[] y = decode(run(e -> {
                    e.upload(gx, encode(x, Precision.FP32)).upload(gd, encode(dy, Precision.FP32));
                    ew.binary(e, op, gx, v, gd, v, gy, v);
                    return e.download(gy);
                }), Precision.FP32, n);
                double worst = 0;
                for (int i = 0; i < n; i++) {
                    double a = x[i], g = dy[i], exact, bound;
                    if (op == Elementwise.Binary.RELU_BACKWARD) {
                        assertEquals(Float.floatToIntBits(Elementwise.apply(op, x[i], dy[i])), Float.floatToIntBits(y[i]), "relu backward at " + a);
                        continue;
                    }
                    if (op == Elementwise.Binary.SILU_BACKWARD) {
                        double s = 1 / (1 + Math.exp(-a)), q = 1 - s, w = a * q, z = 1 + w, t = s * z;
                        exact = g * t;
                        double es = s * (7 + 2 * Math.abs(a)) * 2 * E + 0x1p-126;   // sigmoid, absolute
                        double eq = es + Math.abs(q) * E;
                        double ew2 = Math.abs(a) * eq + Math.abs(w) * E;
                        double ez = ew2 + Math.abs(z) * E;
                        double et = Math.abs(z) * es + s * ez + Math.abs(t) * E;
                        bound = Math.abs(g) * et * 1.01 + Math.abs(exact) * E + 0x1p-126;
                    } else {
                        double c = Math.sqrt(2 / Math.PI), k = 0.044715;
                        double u = c * (a + k * a * a * a), du = c * (1 + 3 * k * a * a);
                        double t = Math.tanh(u), A = 0.5 * (1 + t), B = 0.5 * a * (1 - t * t) * du;
                        exact = g * (A + B);
                        double eu = c * (Math.abs(a) + k * Math.abs(a * a * a)) * 4 * E;     // u's roundings, absolute
                        double uu = Math.min(Math.abs(u), 30);
                        double e2 = Math.exp(2 * Math.min(u, 40));
                        double relE = (4 + 4 * uu) * 2 * E + 2 * eu;                           // exp(2u), relative
                        double frac = 2 / (e2 + 1);
                        double et = frac * (relE * e2 / (e2 + 1) + 6 * E) + E;                // t, absolute
                        double eA = 0.5 * (et + Math.abs(1 + t) * E);
                        double e1t = 2 * Math.abs(t) * et + 2 * E;
                        double eB = Math.abs(0.5 * a * du) * (e1t + Math.abs(1 - t * t) * 6 * E) + Math.abs(B) * 4 * E;
                        bound = Math.abs(g) * (eA + eB + (Math.abs(A) + Math.abs(B)) * E) * 1.01 + Math.abs(exact) * E + 0x1p-126;
                    }
                    double err = Math.abs(y[i] - exact);
                    assertTrue(err <= bound, op + "(" + a + ", " + g + ") = " + y[i] + ", exact " + exact + ", bound " + bound);
                    worst = Math.max(worst, err / bound);
                }
                if (op != Elementwise.Binary.RELU_BACKWARD)
                    System.out.printf("activation gradient %s on %s: worst error/bound %.3g%n", op, device.adapter().name(), worst);
            }
        }
    }

    @Test void activationsWithinTheirBound() {
        var r = new SplittableRandom(5);
        int n = 20000;
        float[] a = values(n, r, 12, Precision.FP32), b = values(n, r, 3, Precision.FP32);
        try (var ew = Elementwise.create(device); var ga = buffer(4L * n); var gb = buffer(4L * n); var gy = buffer(4L * n)) {
            var v = StridedView.contiguous(n);
            for (Object op : new Object[] {Elementwise.Unary.SILU, Elementwise.Unary.GELU, Elementwise.Binary.SWIGLU}) {
                byte[] got = run(e -> {
                    e.upload(ga, encode(a, Precision.FP32)).upload(gb, encode(b, Precision.FP32));
                    if (op instanceof Elementwise.Unary u) ew.unary(e, u, ga, v, gy, v); else ew.binary(e, (Elementwise.Binary) op, ga, v, gb, v, gy, v);
                    return e.download(gy);
                });
                float[] y = decode(got, Precision.FP32, n);
                double worst = 0;
                for (int i = 0; i < n; i++) {
                    double x = a[i], exact;
                    if (op == Elementwise.Unary.SILU) exact = x / (1 + Math.exp(-x));
                    else if (op == Elementwise.Unary.GELU) exact = 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
                    else exact = x / (1 + Math.exp(-x)) * b[i];
                    double u = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x);
                    double bound = op == Elementwise.Unary.GELU ? (8 + 4 * Math.min(Math.abs(u), 30)) * Math.ulp((float) exact) + Math.abs(x) * 0x1p-22
                            : (8 + 2 * Math.abs(x)) * Math.ulp((float) exact) + 1e-37;
                    assertTrue(Math.abs(y[i] - exact) <= bound, op + "(" + x + ") = " + y[i] + ", exact " + exact);
                    worst = Math.max(worst, Math.abs(y[i] - exact) / Math.ulp((float) exact));
                }
                System.out.println("activation " + op + " on " + device.adapter().name() + ": worst " + worst + " ULP");
            }
        }
    }

    @Test void logSumExpWithinItsBound() {
        // one chunk, one past a chunk of 4096, and many chunks (1.10.1)
        for (int n : new int[] {300, 4097, 50000}) logSumExpWithinItsBound(n);
    }

    void logSumExpWithinItsBound(int n) {
        var r = new SplittableRandom(6 + n);
        int rows = 5;
        float[] x = values(rows * n, r, 20, Precision.FP32);
        for (int j = 0; j < n; j++) x[3 * n + j] = Float.NEGATIVE_INFINITY;   // a row of -infinity
        x[4 * n + 7] = Float.NaN;
        try (var reduce = Reduce.create(device); var gx = buffer(4L * x.length); var gy = buffer(4L * rows)) {
            byte[] got = run(e -> {
                e.upload(gx, encode(x, Precision.FP32));
                reduce.logSumExp(e, gx, StridedView.matrix(rows, n), gy, StridedView.contiguous(rows));
                return e.download(gy);
            });
            float[] y = decode(got, Precision.FP32, rows);
            for (int row = 0; row < rows; row++) {
                double exact = Reduce.logSumExpExact(x, row * n, n);
                if (row == 3) { assertEquals(Float.NEGATIVE_INFINITY, y[row]); continue; }
                if (row == 4) { assertTrue(Float.isNaN(y[row])); continue; }
                double bound = 0x1p-24 * 20 + (n + 10 + 2 * 40) * 0x1p-24 + 4 * Math.ulp((float) Math.log(n));
                assertTrue(Math.abs(y[row] - exact) <= bound, "lse n " + n + " row " + row + " = " + y[row] + ", exact " + exact);
            }
        }
    }

    /** GPU microseconds per run of {@code body}: an untimed round, then the best of 4 rounds of about 20 ms. */
    double micros(java.util.function.Consumer<GpuCommandEncoder> body) {
        return GpuBenchmark.measure(device, body).gpuMicros();
    }

    /** Whole encodes on tied and varied logits. A fast prefilter can depend on their distribution. */
    @Test void topKBenchmark() {
        Assumptions.assumeTrue(Boolean.getBoolean("inference.benchmark"));
        int vocab = Integer.getInteger("sampler.benchVocab", 128256);
        var shape = Sampler.Shape.of(1, vocab);
        var st = new Sampler.Settings(0.8f, Integer.getInteger("sampler.benchK", 40), 0.95f, 1);
        int k = st.topK() == 0 ? Math.min(vocab, Sampler.MAX_K) : st.topK();
        try (var sampler = Sampler.create(device); var gl = buffer(4L * vocab); var gt = buffer(4)) {
            for (String pattern : new String[] {"tied", "varied", "skewed", "concentrated", "weak-bound", "ascending"}) {
                var data = new float[vocab];
                var random = new SplittableRandom(4329);
                for (int i = 0; i < vocab; i++) {
                    data[i] = switch (pattern) {
                        case "tied" -> 0f;
                        case "ascending" -> i;
                        case "concentrated" -> i >= 1024 && i < 2048 ? i % 7 : Float.NEGATIVE_INFINITY;
                        case "weak-bound" -> i == k - 1 || i / 1024 % k == 0
                                ? -12f : (float) random.nextDouble(-6, 6);
                        case "skewed" -> i % 64 == 39 ? -12f : (float) random.nextDouble(-6, 6);
                        default -> (float) random.nextDouble(-6, 6);
                    };
                }
                run(e -> { e.upload(gl, encode(data, Precision.FP32)); return e.download(gl); });
                double best = micros(e -> sampler.encode(e, gl, gt, shape, st, 0));
                System.out.printf("sampler-benchmark %s vocab=%d k=%d %s %.2f us%n", device.adapter().name(), vocab, st.topK(), pattern, best);
            }
        }
    }

    /** -Dinference.benchmark=true: a decode step's kernels at llama-like sizes. */
    @Test void benchmark() {
        Assumptions.assumeTrue(Boolean.getBoolean("inference.benchmark"), "set -Dinference.benchmark=true");
        int dim = 4096, heads = 32, headDim = 128, vocab = 128256;
        try (var norms = Norms.create(device, Precision.FP16, Precision.FP16); var rope = Rope.create(device, Precision.FP16, Precision.FP16);
             var emb = Embedding.create(device, Precision.FP16); var sampler = Sampler.create(device); var reduce = Reduce.create(device);
             var gx = buffer(2L * dim * 512); var gr = buffer(2L * dim * 512); var gw = buffer(4L * dim); var gy = buffer(2L * dim * 512);
             var gq = buffer(2L * heads * headDim * 512); var gpos = buffer(4L * 512); var gtab = buffer(4L * 8192 * headDim);
             var gl = buffer(4L * vocab); var gtok = buffer(4); var gemb = buffer(2L * vocab * dim); var glse = buffer(4)) {
            var table = Rope.Table.standard(8192, headDim, 500000);
            run(e -> { e.fill(gx, 0x3C003C00).fill(gr, 0x3C003C00).fill(gw, Float.floatToRawIntBits(1f)).fill(gq, 0x3C003C00).fill(gpos, 7)
                    .upload(gtab, table.bytes()).fill(gl, 0).fill(gtok, 5).fill(gemb, 0x3C003C00); return e.download(gtok); });
            for (int tokens : new int[] {1, 512}) {
                var ns = Norms.Shape.of(tokens, dim, 1e-5f);
                double rms = micros(e -> norms.encode(e, Norms.Kind.RMS, gx, gr, gx, gw, null, gy, ns));
                var rs = Rope.Shape.of(tokens, heads, headDim, Rope.Layout.HALVES);
                double ro = micros(e -> rope.encode(e, gq, gpos, table, gtab, gq, rs));
                System.out.printf("inference %s tokens=%d: add+RMSNorm(%d) %.1f us, RoPE(%dx%d) %.1f us%n", device.adapter().name(), tokens, dim, rms, heads, headDim, ro);
            }
            double gather = micros(e -> emb.encode(e, gtok, 1, Embedding.Table.of(vocab, dim, Precision.FP16), gemb, gy, dim));
            double lse = micros(e -> reduce.logSumExp(e, gl, StridedView.matrix(1, vocab), glse, StridedView.contiguous(1)));
            var shape = Sampler.Shape.of(1, vocab);
            double greedy = micros(e -> sampler.encode(e, gl, gtok, shape, Sampler.Settings.greedy(), 0));
            double temp = micros(e -> sampler.encode(e, gl, gtok, shape, new Sampler.Settings(0.8f, 0, 1f, 1), 0));
            double topk = micros(e -> sampler.encode(e, gl, gtok, shape, new Sampler.Settings(0.8f, 40, 0.95f, 1), 0));
            System.out.printf("inference %s vocab=%d: embedding row %.1f us, log-sum-exp %.1f us, sample greedy %.1f us, temperature %.1f us, top-k 40 + top-p %.1f us%n",
                    device.adapter().name(), vocab, gather, lse, greedy, temp, topk);
            if (device.capabilities().timestamps()) {
                for (var st : new Sampler.Settings[] {new Sampler.Settings(0.8f, 40, 0.95f, 1), new Sampler.Settings(0.8f, 0, 1f, 1)}) {
                    try (var e = device.encoder()) {
                        e.profile();
                        for (int i = 0; i < 50; i++) sampler.encode(e, gl, gtok, shape, st, 0);
                        try (var sub = device.queue().submit(e)) {
                            assertTrue(sub.await(WAIT));
                            var times = new java.util.TreeMap<String, Double>();
                            for (var d : sub.dispatchTimings()) times.merge(d.label(), d.time().toNanos() / 1e3 / 50, Double::sum);
                            System.out.println("inference stages " + st + ": " + times);
                        }
                    }
                }
            }
        }
    }
}
