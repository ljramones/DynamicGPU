package org.dynamisengine.gpu.primitives;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Duration;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import org.dynamisengine.gpu.api.compute.BufferDescriptor;
import org.dynamisengine.gpu.api.compute.BufferUsage;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.gpu.api.compute.Placement;
import org.dynamisengine.gpu.primitives.GemmF16.Algorithm;
import org.dynamisengine.gpu.primitives.GemmF16.Shape;
import org.dynamisengine.gpu.primitives.QuantizedMatrix.Format;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * -Dperf.probe=true (1.15 plan, Phase 0): the decisive GEMM cases on seeded inputs, with checked outputs and paired
 * timings. Each line is {@code perf|section|variant|m|n|k|batch|layout|input|selected|n|gpuMedian|gpuMin|gpuP10|gpuP90|
 * wallMedian|encodeMedian|check}, times in microseconds per product; {@code check} is the worst sampled error over a
 * loose sanity bound (4 k 2^-24 sum |a b|), which a wrong kernel or garbage input exceeds by orders of magnitude.
 * -Dperf.rounds sets the rounds per case (default 10).
 */
abstract class PerfProbeConformance {
    protected abstract GpuProvider provider();
    protected GpuDevice device;
    static final Duration WAIT = Duration.ofSeconds(120);
    static final int ROUNDS = Integer.getInteger("perf.rounds", 10);

    @BeforeEach void open() { device = provider().adapters().getFirst().open(); }
    @AfterEach void close() { device.close(); }

    GpuBuffer buffer(long bytes) {
        return device.allocate(new BufferDescriptor(Math.max(16, bytes), EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE,
                BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE));
    }

    void run(Consumer<GpuCommandEncoder> body) {
        try (var e = device.encoder()) {
            body.accept(e);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
        }
    }

    float[] download(GpuBuffer b) {
        try (var e = device.encoder()) {
            var r = e.download(b);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
            var bb = java.nio.ByteBuffer.wrap(r.bytes()).order(java.nio.ByteOrder.nativeOrder());
            float[] f = new float[r.bytes().length / 4];
            for (int i = 0; i < f.length; i++) f[i] = bb.getFloat(4 * i);
            return f;
        }
    }

    static void print(String section, String variant, Shape s, String input, String selected, GpuBenchmark.Stats t, double check) {
        System.out.printf(Locale.ROOT, "perf|%s|%s|%d|%d|%d|%d|%s|%s|%s|%d|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.3g%n", section, variant, s.m(), s.n(), s.k(),
                s.batch(), s.transposeB() ? "B^T" : "B", input, selected, t.n(), t.gpuMedian(), t.gpuMin(), t.gpuP10(), t.gpuP90(),
                t.wallMedian(), t.encodeMedian(), check);
    }

    /** Worst |C - exact| / (4 k 2^-24 sum |a b|) over sampled elements of C = A B (or A B^T). */
    static double checkGemm(float[] c, GemmF16.Elements a, GemmF16.Elements b, Shape s, SplittableRandom r) {
        double worst = 0;
        for (int t = 0; t < 64; t++) {
            int i = r.nextInt(s.m()), j = r.nextInt(s.n());
            double exact = 0, mag = 0;
            for (int l = 0; l < s.k(); l++) {
                double x = a.at((long) i * s.lda() + l);
                double y = s.transposeB() ? b.at((long) j * s.ldb() + l) : b.at((long) l * s.ldb() + j);
                exact += x * y; mag += Math.abs(x * y);
            }
            double bound = 4.0 * s.k() * 0x1p-24 * mag + 1e-30;
            worst = Math.max(worst, Math.abs(c[i * s.ldc() + j] - exact) / bound);
        }
        return worst;
    }

    @Test void probe() {
        Assumptions.assumeTrue(Boolean.getBoolean("perf.probe"), "set -Dperf.probe=true");
        System.out.println("perf-device|" + device.adapter().name() + "|" + device.adapter().backend() + "|rounds=" + ROUNDS);
        String only = System.getProperty("perf.only", "gemm,selector,quant");
        if (only.contains("prefill")) prefillDequant();
        if (only.contains("gate")) prefillGate();
        if (only.contains("gemm")) squareGemm();
        if (only.contains("selector")) selectorMisses();
        if (only.contains("quant")) quantized();
    }

    /**
     * Times AUTO and every supported forced algorithm on one shape, on seeded inputs, alternating in the same rounds
     * (so drift lands on all of them alike), after checking each one's C.
     */
    private void gemmCase(String section, Shape s, GemmF16.Input input, List<Algorithm> forced) {
        var r = new SplittableRandom(0x5eed ^ s.m() * 31L ^ s.n() * 17L ^ s.k());
        var a = GemmF16Conformance.data(input, (int) (GemmF16Conformance.pad4(s.aBytes(input)) / input.bytes()), r);
        var b = GemmF16Conformance.data(input, (int) (GemmF16Conformance.pad4(s.bBytes(input)) / input.bytes()), r);
        var kernels = new ArrayList<GemmF16>();
        try (var ga = buffer(a.bytes().length); var gb = buffer(b.bytes().length); var gc = buffer(s.cBytes())) {
            run(e -> e.upload(ga, a.bytes()).upload(gb, b.bytes()));
            var names = new ArrayList<String>();
            var selected = new ArrayList<String>();
            var auto = GemmF16.create(device, input, Precision.FP32);
            kernels.add(auto); names.add("AUTO"); selected.add(auto.algorithm(s).name());
            for (var alg : forced) {
                if (!GemmF16.supports(device, alg, input)) continue;
                if (alg == Algorithm.SKINNY && s.m() > GemmF16.SKINNY_MAX_M) continue;
                kernels.add(GemmF16.create(device, alg, input, Precision.FP32)); names.add(alg.name()); selected.add(alg.name());
            }
            var bodies = new ArrayList<Consumer<GpuCommandEncoder>>();
            double[] checks = new double[kernels.size()];
            for (int i = 0; i < kernels.size(); i++) {
                var g = kernels.get(i);
                Consumer<GpuCommandEncoder> body = e -> g.encode(e, ga, gb, gc, s);
                bodies.add(body);
                run(e -> e.fill(gc, 0x7FC00000));   // NaN, so a kernel that skips elements fails the check
                run(body);
                checks[i] = s.batch() == 1 ? checkGemm(download(gc), a.elements(), b.elements(), s, new SplittableRandom(7)) : Double.NaN;
                if (s.batch() == 1) assertTrue(checks[i] <= 1, section + " " + names.get(i) + " " + s + ": sampled error " + checks[i] + " of the bound");
            }
            var t = GpuBenchmark.alternating(device, bodies, ROUNDS);
            for (int i = 0; i < kernels.size(); i++)
                print(section, names.get(i), s, input.name(), selected.get(i), GpuBenchmark.Stats.of(t.get(i)), checks[i]);
        } finally {
            kernels.forEach(GemmF16::close);
        }
    }

    private void squareGemm() {
        var all = List.of(Algorithm.values());
        for (int n : new int[] {2048, 4096}) {
            for (var input : new GemmF16.Input[] {GemmF16.Input.FP16, GemmF16.Input.FP32, GemmF16.Input.BF16}) {
                gemmCase("gemm-square", Shape.of(n, n, n, 1, 0), input, all);
                gemmCase("gemm-square", Shape.ofTransposedB(n, n, n, 1, 0), input, all);
            }
        }
    }

    /** The two suspected selector misses and their neighbours, FP16. */
    private void selectorMisses() {
        var all = List.of(Algorithm.values());
        var shapes = new ArrayList<Shape>();
        for (int m : new int[] {4, 8, 16})
            for (int n : new int[] {8, 16, 32})
                for (int k : new int[] {2048, 4096, 8192}) {
                    shapes.add(Shape.ofTransposedB(m, n, k, 1, 0));
                    shapes.add(Shape.of(m, n, k, 1, 0));
                }
        for (int n : new int[] {512, 1024, 2048})
            for (int k : new int[] {512, 1024, 2048}) {
                shapes.add(Shape.of(1, n, k, 1, 0));
                shapes.add(Shape.ofTransposedB(1, n, k, 1, 0));
            }
        // the 1.8.0 shape sweep, to catch regressions elsewhere
        int[][] sweep = {{1, 4096, 4096, 1}, {16, 4096, 4096, 1}, {32, 4096, 4096, 1}, {64, 4096, 4096, 1}, {128, 128, 128, 1},
                {512, 512, 512, 1}, {2048, 2048, 128, 1}, {8, 8, 4096, 1}, {16, 16, 16, 4096}, {32, 32, 32, 4096}, {16, 16, 256, 512},
                {32, 32, 128, 1024}, {8, 32, 64, 2048}, {64, 64, 64, 512}, {48, 48, 48, 1024}, {32, 64, 64, 512}, {8, 300, 200, 1},
                {1, 1024, 1024, 1}, {4, 2048, 512, 1}, {16, 1024, 1024, 1}, {256, 256, 256, 1}, {96, 160, 96, 1}};
        if (!Boolean.getBoolean("perf.selectorOnlyNew")) {
            for (int[] m : sweep)
                for (boolean t : new boolean[] {false, true}) {
                    var shape = t ? Shape.ofTransposedB(m[0], m[1], m[2], 1, 0) : Shape.of(m[0], m[1], m[2], 1, 0);
                    gemmCase("sweep", m[3] > 1 ? shape.batched(m[3]) : shape, GemmF16.Input.FP16, all);
                }
        }
        // three- and five-row projections, absent from the decode grid that refitted SKINNY's ordinary-B row penalty
        for (int m : new int[] {3, 5, 6})
            for (int[] nk : new int[][] {{3456, 4096}, {4096, 4096}, {11008, 4096}})
                for (boolean t : new boolean[] {false, true})
                    gemmCase("rows-check", t ? Shape.ofTransposedB(m, nk[0], nk[1], 1, 0) : Shape.of(m, nk[0], nk[1], 1, 0), GemmF16.Input.FP16, all);
        if (Boolean.getBoolean("perf.rowsCheckOnly")) return;
        // the established large-projection decode grid (1.14.1's -Dgemm.decode=true shapes)
        for (int[] nk : new int[][] {{4096, 4096}, {11008, 4096}, {4096, 11008}})
            for (int m : new int[] {1, 2, 4, 8, 16})
                for (boolean t : new boolean[] {true, false})
                    gemmCase("decode-grid", t ? Shape.ofTransposedB(m, nk[0], nk[1], 1, 0) : Shape.of(m, nk[0], nk[1], 1, 0), GemmF16.Input.FP16, all);
        for (var s : shapes) gemmCase("selector", s, GemmF16.Input.FP16, all);
    }

    /**
     * A fair quantized comparison on one 4096 x 4096 projection: representative weights (normal values, quantized once),
     * FP16 and quantized weights both rotating through more than 256 MiB of copies, FP16 AUTO at every m, and
     * SKINNY and TILED forced at the same m for every format.
     */
    private void quantized() {
        int n = Integer.getInteger("perf.quantN", 4096), k = Integer.getInteger("perf.quantK", 4096);
        var r = new SplittableRandom(4096);
        float[] values = new float[n * k];
        for (int i = 0; i < values.length; i++) values[i] = (float) (r.nextGaussian() * 0.02);
        short[] wh = new short[n * k];
        for (int i = 0; i < wh.length; i++) wh[i] = Float.floatToFloat16(values[i]);
        var fp16W = GemmF16Conformance.bytes(wh);
        int[] ms = java.util.Arrays.stream(System.getProperty("perf.quantM", "1,2,4,8,16,32,64,512").split(",")).mapToInt(Integer::parseInt).toArray();
        var formats = List.of(System.getProperty("perf.quantFormats", "FP16,Q8,Q4,Q4M,Q4K,Q5K,Q6K").split(","));
        var acts = new java.util.HashMap<Integer, short[]>();
        for (int m : ms) acts.put(m, GemmF16Conformance.halves(m * k + (m * k & 1), r));
        // FP16 weights: AUTO at every m, over rotating copies
        var copies = formats.contains("FP16") ? rotating(fp16W, fp16W.length) : List.<GpuBuffer>of();
        try (var auto = GemmF16.create(device, GemmF16.Input.FP16, Precision.FP32)) {
            for (int m : formats.contains("FP16") ? ms : new int[0]) {
                var s = Shape.ofTransposedB(m, n, k, 1, 0);
                var aEl = GemmF16.Elements.fp16(acts.get(m));
                try (var ga = buffer(2L * acts.get(m).length); var gc = buffer(s.cBytes())) {
                    run(e -> e.upload(ga, GemmF16Conformance.bytes(acts.get(m))));
                    quantMeasure("quant-fair", "FP16 AUTO", s, auto.algorithm(s).name(), copies,
                            (e, w) -> auto.encode(e, ga, w, gc, s), gc, c -> checkGemm(c, aEl, GemmF16.Elements.fp16(wh), s, new SplittableRandom(3)));
                }
            }
        } finally { copies.forEach(GpuBuffer::close); }
        for (var w : new QuantizedMatrix[] {new QuantizedMatrix(Format.Q8, 32, n, k), new QuantizedMatrix(Format.Q4, 32, n, k),
                new QuantizedMatrix(Format.Q4M, 32, n, k), new QuantizedMatrix(Format.Q4K, 256, n, k),
                new QuantizedMatrix(Format.Q5K, 256, n, k), new QuantizedMatrix(Format.Q6K, 256, n, k)}) {
            if (!formats.contains(w.format().name())) continue;
            byte[] packed = w.quantize(values);
            var qc = rotating(packed, w.bytes());
            try (var skinny = QuantizedGemm.create(device, QuantizedGemm.Algorithm.SKINNY, GemmF16.Input.FP16, Precision.FP32);
                 var tiled = QuantizedGemm.create(device, QuantizedGemm.Algorithm.TILED, GemmF16.Input.FP16, Precision.FP32);
                 var auto = QuantizedGemm.create(device, GemmF16.Input.FP16, Precision.FP32)) {
                for (int m : ms) {
                    var s = Shape.ofTransposedB(m, n, k, 1, 0);
                    var aEl = GemmF16.Elements.fp16(acts.get(m));
                    try (var ga = buffer(2L * acts.get(m).length); var gc = buffer(s.cBytes())) {
                        run(e -> e.upload(ga, GemmF16Conformance.bytes(acts.get(m))));
                        if (m <= QuantizedGemm.SKINNY_MAX_M)
                            quantMeasure("quant-fair", w.format() + " SKINNY", s, "SKINNY", qc, (e, buf) -> skinny.encode(e, ga, w, buf, gc, s), gc,
                                    c -> checkQuant(c, aEl, w, packed, s));
                        quantMeasure("quant-fair", w.format() + " TILED", s, "TILED", qc, (e, buf) -> tiled.encode(e, ga, w, buf, gc, s), gc,
                                c -> checkQuant(c, aEl, w, packed, s));
                        quantMeasure("quant-fair", w.format() + " AUTO", s, auto.algorithm(s, w).name(), qc, (e, buf) -> auto.encode(e, ga, w, buf, gc, s), gc,
                                c -> checkQuant(c, aEl, w, packed, s));
                    }
                }
            } finally { qc.forEach(GpuBuffer::close); }
        }
    }

    /** Copies of {@code data} totalling over 256 MiB, so the system cache cannot serve repeated reads of one copy. */
    private List<GpuBuffer> rotating(byte[] data, long bytes) {
        int count = (int) Math.max(2, (256L << 20) / bytes + 1);
        var out = new ArrayList<GpuBuffer>(count);
        for (int i = 0; i < count; i++) out.add(buffer(bytes));
        byte[] upload = data.length % 4 == 0 ? data : java.util.Arrays.copyOf(data, (data.length + 3) & ~3);
        for (var b : out) run(e -> e.upload(b, upload));
        return out;
    }

    /**
     * Quantized prefill through existing kernels (1.15 plan, Phase 2 experiment): unpack the packed weights to FP32
     * ({@link Quantize#unpack}), cast them to FP16 ({@link Cast}), then the FP16 GEMM with AUTO selection, timed as
     * one sequence against the current quantized TILED kernel and against the FP16 GEMM alone on the unpacked weights.
     * The FP16 rounding of each weight is a different contract from the quantized kernels'; the check is against the
     * exact dequantized weights with the loose bound.
     */
    private void prefillDequant() {
        var r = new SplittableRandom(512);
        int[][] nks = {{4096, 4096}, {14336, 4096}, {4096, 14336}};
        int[] ms = java.util.Arrays.stream(System.getProperty("perf.prefillM", "64,512,2048").split(",")).mapToInt(Integer::parseInt).toArray();
        var formats = List.of(System.getProperty("perf.prefillFormats", "Q4K,Q4,Q8,Q6K").split(","));
        try (var tiled = QuantizedGemm.create(device, QuantizedGemm.Algorithm.TILED, GemmF16.Input.FP16, Precision.FP32);
             var gemm = GemmF16.create(device, GemmF16.Input.FP16, Precision.FP32); var unpack = Quantize.create(device); var cast = Cast.create(device)) {
            for (int[] nk : nks) {
                int n = nk[0], k = nk[1];
                float[] values = new float[n * k];
                for (int i = 0; i < values.length; i++) values[i] = (float) (r.nextGaussian() * 0.02);
                try (var w32 = buffer(4L * n * k); var w16 = buffer(2L * n * k)) {
                    for (String fname : formats) {
                        var w = new QuantizedMatrix(Format.valueOf(fname), Format.valueOf(fname).kQuant() ? 256 : 32, n, k);
                        byte[] packed = w.quantize(values);
                        byte[] up = packed.length % 4 == 0 ? packed : java.util.Arrays.copyOf(packed, (packed.length + 3) & ~3);
                        try (var gw = buffer(w.bytes())) {
                            run(e -> e.upload(gw, up));
                            var view = org.dynamisengine.gpu.api.compute.kit.StridedView.contiguous((long) n * k);
                            Consumer<GpuCommandEncoder> toFp16 = e -> {
                                unpack.unpack(e, w, gw, w32);
                                cast.cast(e, Cast.Type.FP32, Cast.Type.FP16, w32, view, w16, view);
                            };
                            for (int m : ms) {
                                var s = Shape.ofTransposedB(m, n, k, 1, 0);
                                short[] ah = GemmF16Conformance.halves(m * k, r);
                                var aEl = GemmF16.Elements.fp16(ah);
                                try (var ga = buffer(2L * ah.length); var gc = buffer(s.cBytes())) {
                                    run(e -> e.upload(ga, GemmF16Conformance.bytes(ah)));
                                    var one = List.of(gw);
                                    quantMeasure("prefill-dequant", fname + " TILED", s, "TILED", one, (e, b) -> tiled.encode(e, ga, w, b, gc, s), gc,
                                            c -> checkQuant(c, aEl, w, packed, s));
                                    quantMeasure("prefill-dequant", fname + " unpack+cast+GEMM", s, "UNPACK+" + gemm.algorithm(s).name(), one,
                                            (e, b) -> { toFp16.accept(e); gemm.encode(e, ga, w16, gc, s); }, gc, c -> checkQuant(c, aEl, w, packed, s));
                                    quantMeasure("prefill-dequant", fname + " unpackF16+GEMM", s, "UNPACK16+" + gemm.algorithm(s).name(), one,
                                            (e, b) -> { unpack.unpackF16(e, w, gw, w16); gemm.encode(e, ga, w16, gc, s); }, gc, c -> checkQuant(c, aEl, w, packed, s));
                                    run(toFp16);
                                    quantMeasure("prefill-dequant", fname + " GEMM only", s, gemm.algorithm(s).name(), one,
                                            (e, b) -> gemm.encode(e, ga, w16, gc, s), gc, c -> checkQuant(c, aEl, w, packed, s));
                                    if (m == ms[0]) {
                                        quantMeasure("prefill-dequant", fname + " unpack+cast only", s, "UNPACK", one, (e, b) -> toFp16.accept(e), gc, c -> 0);
                                        quantMeasure("prefill-dequant", fname + " unpackF16 only", s, "UNPACK16", one, (e, b) -> unpack.unpackF16(e, w, gw, w16), gc, c -> 0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * The quantized-prefill performance gate (1.15): quantized TILED, the two-pass path (unpack to FP32, cast, FP16
     * GEMM) and the direct path (unpackF16, FP16 GEMM), alternating in the same rounds so drift lands on all three,
     * around the tile boundaries. Each path's output is checked first. Lines: {@code perf|gate|<format> <path>|...}.
     */
    private void prefillGate() {
        var r = new SplittableRandom(64);
        int[][] nks = {{4096, 4096}, {14336, 4096}, {4096, 14336}};
        int[] ms = java.util.Arrays.stream(System.getProperty("perf.gateM", "17,32,48,64,96,128,256,512,2048").split(",")).mapToInt(Integer::parseInt).toArray();
        var formats = List.of(System.getProperty("perf.gateFormats", "Q4K,Q4,Q8,Q6K").split(","));
        try (var tiled = QuantizedGemm.create(device, QuantizedGemm.Algorithm.TILED, GemmF16.Input.FP16, Precision.FP32);
             var gemm = GemmF16.create(device, GemmF16.Input.FP16, Precision.FP32); var unpack = Quantize.create(device); var cast = Cast.create(device)) {
            for (int[] nk : nks) {
                int n = nk[0], k = nk[1];
                float[] values = new float[n * k];
                for (int i = 0; i < values.length; i++) values[i] = (float) (r.nextGaussian() * 0.02);
                try (var w32 = buffer(4L * n * k); var w16 = buffer(2L * n * k)) {
                    for (String fname : formats) {
                        var format = Format.valueOf(fname);
                        var w = new QuantizedMatrix(format, format.kQuant() ? 256 : 32, n, k);
                        byte[] packed = w.quantize(values);
                        byte[] up = packed.length % 4 == 0 ? packed : java.util.Arrays.copyOf(packed, (packed.length + 3) & ~3);
                        try (var gw = buffer(w.bytes())) {
                            run(e -> e.upload(gw, up));
                            var view = org.dynamisengine.gpu.api.compute.kit.StridedView.contiguous((long) n * k);
                            for (int m : ms) {
                                var s = Shape.ofTransposedB(m, n, k, 1, 0);
                                short[] ah = GemmF16Conformance.halves(m * k + (m * k & 1), r);
                                var aEl = GemmF16.Elements.fp16(ah);
                                try (var ga = buffer(2L * ah.length); var gc = buffer(s.cBytes())) {
                                    run(e -> e.upload(ga, GemmF16Conformance.bytes(ah)));
                                    List<Consumer<GpuCommandEncoder>> paths = List.of(
                                            e -> tiled.encode(e, ga, w, gw, gc, s),
                                            e -> { unpack.unpack(e, w, gw, w32); cast.cast(e, Cast.Type.FP32, Cast.Type.FP16, w32, view, w16, view); gemm.encode(e, ga, w16, gc, s); },
                                            e -> { unpack.unpackF16(e, w, gw, w16); gemm.encode(e, ga, w16, gc, s); });
                                    String[] names = {"TILED", "two-pass", "direct"};
                                    double[] checks = new double[3];
                                    for (int i = 0; i < 3; i++) {
                                        var body = paths.get(i);
                                        run(e -> e.fill(gc, 0x7FC00000));
                                        run(body);
                                        checks[i] = checkQuant(download(gc), aEl, w, packed, s);
                                        assertTrue(checks[i] <= 1, "gate " + fname + " " + names[i] + " " + s + ": " + checks[i]);
                                    }
                                    var t = GpuBenchmark.alternating(device, paths, ROUNDS);
                                    String sel = gemm.algorithm(s).name();
                                    for (int i = 0; i < 3; i++)
                                        print("gate", fname + " " + names[i], s, "FP16", i == 0 ? "TILED" : (i == 1 ? "UNPACK+CAST+" : "UNPACK16+") + sel,
                                                GpuBenchmark.Stats.of(t.get(i)), checks[i]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    interface WeightBody { void accept(GpuCommandEncoder e, GpuBuffer w); }

    private void quantMeasure(String section, String variant, Shape s, String selected, List<GpuBuffer> weights, WeightBody body, GpuBuffer gc,
            java.util.function.ToDoubleFunction<float[]> check) {
        run(e -> e.fill(gc, 0x7FC00000));
        run(e -> body.accept(e, weights.get(0)));
        double err = check.applyAsDouble(download(gc));
        assertTrue(err <= 1, section + " " + variant + " " + s + ": sampled error " + err + " of the bound");
        int[] next = {0};
        var t = GpuBenchmark.Stats.of(GpuBenchmark.paired(device, e -> body.accept(e, weights.get(next[0]++ % weights.size())), ROUNDS));
        print(section, variant, s, "FP16", selected, t, err);
    }

    /** Worst sampled error of C = A W^T against the dequantized W, over the loose bound. */
    static double checkQuant(float[] c, GemmF16.Elements a, QuantizedMatrix w, byte[] packed, Shape s) {
        var r = new SplittableRandom(11);
        double worst = 0;
        for (int t = 0; t < 32; t++) {
            int i = r.nextInt(s.m()), j = r.nextInt(s.n());
            double exact = 0, mag = 0;
            for (int l = 0; l < s.k(); l++) {
                double x = a.at((long) i * s.lda() + l), y = w.value(packed, j, l);
                exact += x * y; mag += Math.abs(x * y);
            }
            worst = Math.max(worst, Math.abs(c[i * s.ldc() + j] - exact) / (4.0 * s.k() * 0x1p-24 * mag + 1e-30));
        }
        return worst;
    }
}
