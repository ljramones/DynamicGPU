package org.dynamisengine.gpu.primitives;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.TreeMap;
import java.util.function.Consumer;
import org.dynamisengine.gpu.api.compute.BufferDescriptor;
import org.dynamisengine.gpu.api.compute.BufferUsage;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.gpu.api.compute.Placement;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.dynamisengine.gpu.primitives.GemmF16.Epilogue;
import org.dynamisengine.gpu.primitives.QuantizedMatrix.Format;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * -Dblock.benchmark=true (1.15 plan, Phase 0): a Llama-3-8B-shaped transformer block through the public Java API, the
 * whole forward pass of {@code block.layers} layers (default 32) in one submission. Dimensions: model 4096, 32 query and
 * 8 KV heads of 128, FFN 14336 (SwiGLU), RMSNorm, RoPE (theta 500000), a paged FP16 KV cache with 64-token pages. Every
 * product uses the automatic kernel selection; activations and the cache are FP16.
 * <p>
 * Each layer: RMSNorm, Q/K/V projections, RoPE on Q and K, the KV cache write, fused causal attention over the paged
 * cache (prefill transposes Q to head-major and O back), the output projection with the residual in its epilogue,
 * RMSNorm, the up projection, the gate projection with SiLU and the up result as its gate, and the down projection
 * with the residual. One set of weights serves every layer (436 MB in FP16, beyond the system cache), each layer with
 * its own KV pages.
 * <p>
 * Before timing, one layer runs alone and every stage of it is checked on sampled elements against a double-precision
 * computation from that stage's own GPU inputs (a sanity bound, not the kernels' contracts).
 * <p>
 * Output: {@code block|backend|weights|phase|tokens|context|layers|n|gpuMedianMs|gpuP10Ms|gpuP90Ms|wallMedianMs|
 * encodeMedianMs|tokensPerSecond(wall)|worstCheck}, then the profiled per-stage GPU time of one pass for diagnosis
 * (dispatch timings can overlap; acceptance uses the whole-submission times). Properties: {@code block.weights}
 * (FP16,Q4K), {@code block.prefill} (512,2048), {@code block.decode} (512,4096,32768), {@code block.rounds} (10),
 * {@code block.sustainMinutes} (0: off; otherwise prefill 512 then 32 decode tokens at 4096, repeated, reported per minute).
 */
abstract class TransformerBlockConformance {
    protected abstract GpuProvider provider();
    protected GpuDevice device;
    static final Duration WAIT = Duration.ofSeconds(300);

    static final int D = 4096, H = 32, KVH = 8, HD = 128, F = 14336, QD = H * HD, KD = KVH * HD, PAGE = 64;
    static final float EPS = 1e-5f, THETA = 500000f;
    final int layers = Integer.getInteger("block.layers", 32);
    final int rounds = Integer.getInteger("block.rounds", 10);

    @BeforeEach void open() { device = provider().adapters().getFirst().open(); }
    @AfterEach void close() { device.close(); }

    GpuBuffer buffer(long bytes) {
        return device.allocate(new BufferDescriptor(Math.max(16, (bytes + 3) & ~3L), EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE,
                BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE));
    }

    void run(Consumer<GpuCommandEncoder> body) {
        try (var e = device.encoder()) {
            body.accept(e);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
        }
    }

    short[] halves(GpuBuffer b, int count) {
        try (var e = device.encoder()) {
            var r = e.download(b);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
            var bb = ByteBuffer.wrap(r.bytes()).order(ByteOrder.nativeOrder());
            short[] h = new short[count];
            for (int i = 0; i < count; i++) h[i] = bb.getShort(2 * i);
            return h;
        }
    }

    static byte[] bytes(short[] h) {
        var b = ByteBuffer.allocate((2 * h.length + 3) & ~3).order(ByteOrder.nativeOrder());
        for (short v : h) b.putShort(v);
        return b.array();
    }

    static byte[] ints(int n, java.util.function.IntUnaryOperator f) {
        var b = ByteBuffer.allocate(4 * n).order(ByteOrder.nativeOrder());
        for (int i = 0; i < n; i++) b.putInt(f.applyAsInt(i));
        return b.array();
    }

    static byte[] floats(float[] f) {
        var b = ByteBuffer.allocate(4 * f.length).order(ByteOrder.nativeOrder());
        for (float v : f) b.putFloat(v);
        return b.array();
    }

    static short[] gaussianHalves(int count, double sigma, SplittableRandom r) {
        short[] h = new short[count];
        for (int i = 0; i < count; i++) h[i] = Float.floatToFloat16((float) (r.nextGaussian() * sigma));
        return h;
    }

    static float f(short h) { return Float.float16ToFloat(h); }

    /** One weight matrix W (n x k, applied as x W^T): FP16 or packed. */
    final class Weight implements AutoCloseable {
        final int n, k;
        final short[] fp16;
        final QuantizedMatrix q;
        final byte[] packed;
        final GpuBuffer buffer;

        Weight(int n, int k, Format format, SplittableRandom r) {
            this.n = n; this.k = k;
            short[] h = gaussianHalves(n * k, 0.02, r);
            if (format == null) {
                fp16 = h; q = null; packed = null;
                buffer = buffer(2L * n * k);
                run(e -> e.upload(buffer, bytes(h)));
            } else {
                fp16 = null;
                float[] v = new float[n * k];
                for (int i = 0; i < v.length; i++) v[i] = f(h[i]);
                q = new QuantizedMatrix(format, format.kQuant() ? 256 : 32, n, k);
                packed = q.quantize(v);
                buffer = buffer(q.bytes());
                byte[] up = packed.length % 4 == 0 ? packed : java.util.Arrays.copyOf(packed, (packed.length + 3) & ~3);
                run(e -> e.upload(buffer, up));
            }
        }

        double at(int row, int col) { return fp16 != null ? f(fp16[row * k + col]) : q.value(packed, row, col); }
        @Override public void close() { buffer.close(); }
    }

    /** The products: FP16 weights through GemmF16, packed ones through QuantizedGemm, both AUTO, FP16 output. */
    final class Products implements AutoCloseable {
        final GemmF16 gemm = GemmF16.create(device, Precision.FP16);
        final QuantizedGemm qgemm = QuantizedGemm.create(device, GemmF16.Input.FP16, Precision.FP16);
        /** -Dblock.quantPrefill=dequant: quantized products beyond SKINNY's rows unpack W to FP16 and run the FP16 GEMM. */
        final boolean dequant = "dequant".equals(System.getProperty("block.quantPrefill"));
        final Quantize unpack = Quantize.create(device);
        final Cast cast = Cast.create(device);
        GpuBuffer w32, w16;

        boolean unpacks(Weight w, int m) { return w.q != null && dequant && m > QuantizedGemm.skinnyMaxM(w.q.format()); }

        void encode(GpuCommandEncoder e, GpuBuffer a, Weight w, GpuBuffer c, int m, Epilogue epi, GpuBuffer gate, GpuBuffer residual) {
            var s = GemmF16.Shape.ofTransposedB(m, w.n, w.k, 1f, 0f);
            if (w.q == null) gemm.encode(e, a, w.buffer, c, s, epi, null, gate, residual);
            else if (unpacks(w, m)) {
                long nk = (long) w.n * w.k;
                if (w32 == null) { w32 = buffer(4L * F * D); w16 = buffer(2L * F * D); }
                var view = StridedView.contiguous(nk);
                unpack.unpack(e, w.q, w.buffer, w32);
                cast.cast(e, Cast.Type.FP32, Cast.Type.FP16, w32, view, w16, view);
                gemm.encode(e, a, w16, c, s, epi, null, gate, residual);
            } else qgemm.encode(e, a, w.q, w.buffer, c, s, epi, null, gate, residual);
        }

        String selected(Weight w, int m) {
            var s = GemmF16.Shape.ofTransposedB(m, w.n, w.k, 1f, 0f);
            if (unpacks(w, m)) return "UNPACK+" + gemm.algorithm(s).name();
            return w.q == null ? gemm.algorithm(s).name() : qgemm.algorithm(s, w.q).name();
        }

        @Override public void close() {
            gemm.close(); qgemm.close(); unpack.close(); cast.close();
            if (w32 != null) { w32.close(); w16.close(); }
        }
    }

    /** Activations for up to {@code maxTokens} tokens. */
    final class Activations implements AutoCloseable {
        final GpuBuffer x0, x1, xn, q, k, v, qr, kr, qh, o, ot, x2, xn2, up, g, positions;
        Activations(int t) {
            positions = buffer(4L * t);
            x0 = buffer(2L * t * D); x1 = buffer(2L * t * D); xn = buffer(2L * t * D);
            q = buffer(2L * t * QD); k = buffer(2L * t * KD); v = buffer(2L * t * KD);
            qr = buffer(2L * t * QD); kr = buffer(2L * t * KD); qh = buffer(2L * t * QD);
            o = buffer(2L * t * QD); ot = buffer(2L * t * QD); x2 = buffer(2L * t * D); xn2 = buffer(2L * t * D);
            up = buffer(2L * t * F); g = buffer(2L * t * F);
        }
        @Override public void close() {
            for (var b : List.of(x0, x1, xn, q, k, v, qr, kr, qh, o, ot, x2, xn2, up, g, positions)) b.close();
        }
    }

    /** Everything one configuration needs. */
    final class Model implements AutoCloseable {
        final Weight wq, wk, wv, wo, wup, wgate, wdown;
        final GpuBuffer norm1, norm2, ropeTable;
        final Rope.Table table;
        final Products products = new Products();
        final Norms norms = Norms.create(device, Precision.FP16, Precision.FP16);
        final Rope rope = Rope.create(device, Precision.FP16, Precision.FP16);
        final AttentionF16 attention = AttentionF16.create(device, Precision.FP16);
        final Cast cast = Cast.create(device);
        final PagedKvCache cache;

        Model(Format format, int maxTokens, int maxContext) {
            var r = new SplittableRandom(0x8b0c);
            wq = new Weight(QD, D, format, r); wk = new Weight(KD, D, format, r); wv = new Weight(KD, D, format, r);
            wo = new Weight(D, QD, format, r); wup = new Weight(F, D, format, r); wgate = new Weight(F, D, format, r);
            wdown = new Weight(D, F, format, r);
            float[] n1 = new float[D], n2 = new float[D];
            for (int i = 0; i < D; i++) { n1[i] = (float) (1 + 0.1 * r.nextGaussian()); n2[i] = (float) (1 + 0.1 * r.nextGaussian()); }
            norm1 = buffer(4L * D); norm2 = buffer(4L * D);
            run(e -> e.upload(norm1, floats(n1)).upload(norm2, floats(n2)));
            table = Rope.Table.standard(maxContext + 1, HD, THETA);
            ropeTable = buffer(table.bytes().length);
            run(e -> e.upload(ropeTable, table.bytes()));
            int pages = Math.ceilDiv(maxContext + 1, PAGE) + 8;
            cache = PagedKvCache.create(device, new PagedKvCache.Config(layers, KVH, HD, PAGE, pages));
        }

        float[] weights(GpuBuffer b) {
            try (var e = device.encoder()) {
                var rb = e.download(b);
                try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
                var bb = ByteBuffer.wrap(rb.bytes()).order(ByteOrder.nativeOrder());
                float[] out = new float[D];
                for (int i = 0; i < D; i++) out[i] = bb.getFloat(4 * i);
                return out;
            }
        }

        /**
         * Records one layer on {@code t} tokens at positions {@code start ..}: x in {@code in}, the result in {@code out}.
         * The sequence must hold {@code start + t} reserved tokens.
         */
        void layer(GpuCommandEncoder e, int layer, Activations a, GpuBuffer in, GpuBuffer out, int t, int start, PagedKvCache.Sequence seq) {
            norms.encode(e, Norms.Kind.RMS, in, null, null, norm1, null, a.xn, Norms.Shape.of(t, D, EPS));
            products.encode(e, a.xn, wq, a.q, t, Epilogue.NONE, null, null);
            products.encode(e, a.xn, wk, a.k, t, Epilogue.NONE, null, null);
            products.encode(e, a.xn, wv, a.v, t, Epilogue.NONE, null, null);
            rope.encode(e, a.q, a.positions, table, ropeTable, a.qr, Rope.Shape.of(t, H, HD, Rope.Layout.HALVES));
            rope.encode(e, a.k, a.positions, table, ropeTable, a.kr, Rope.Shape.of(t, KVH, HD, Rope.Layout.HALVES));
            cache.write(e, layer, seq, start, a.kr, a.v, t, HD, KD);
            var s = new AttentionF16.Shape(H, t, start + t, HD, (float) (1 / Math.sqrt(HD)), true, KVH, start + t);
            if (t == 1) {
                attention.encode(e, a.qr, cache, layer, seq, a.ot, s);   // one token: token-major is head-major
            } else {
                cast.cast(e, Cast.Type.FP16, Cast.Type.FP16, a.qr, StridedView.of(0, new long[] {H, t, HD}, new long[] {HD, QD, 1}),
                        a.qh, StridedView.of(0, new long[] {H, t, HD}, new long[] {(long) t * HD, HD, 1}));
                attention.encode(e, a.qh, cache, layer, seq, a.o, s);
                // Cast writes 16-bit outputs contiguously: read O head-major, write token-major
                cast.cast(e, Cast.Type.FP16, Cast.Type.FP16, a.o, StridedView.of(0, new long[] {t, H, HD}, new long[] {HD, (long) t * HD, 1}),
                        a.ot, StridedView.of(0, new long[] {t, H, HD}, new long[] {QD, HD, 1}));
            }
            products.encode(e, a.ot, wo, a.x2, t, Epilogue.NONE.withResidual(), null, in);
            norms.encode(e, Norms.Kind.RMS, a.x2, null, null, norm2, null, a.xn2, Norms.Shape.of(t, D, EPS));
            products.encode(e, a.xn2, wup, a.up, t, Epilogue.NONE, null, null);
            products.encode(e, a.xn2, wgate, a.g, t, Epilogue.NONE.withActivation(Epilogue.Activation.SILU).withGate(), a.up, null);
            products.encode(e, a.g, wdown, out, t, Epilogue.NONE.withResidual(), null, a.x2);
        }

        /** The whole pass: every layer, x ping-ponging between two buffers. */
        void forward(GpuCommandEncoder e, Activations a, int t, int start, PagedKvCache.Sequence seq, int count) {
            for (int l = 0; l < count; l++) layer(e, l, a, l % 2 == 0 ? a.x0 : a.x1, l % 2 == 0 ? a.x1 : a.x0, t, start, seq);
        }

        @Override public void close() {
            for (var w : List.of(wq, wk, wv, wo, wup, wgate, wdown)) w.close();
            for (var b : List.of(norm1, norm2, ropeTable)) b.close();
            products.close(); norms.close(); rope.close(); attention.close(); cast.close(); cache.close();
        }
    }

    @Test void benchmark() {
        Assumptions.assumeTrue(Boolean.getBoolean("block.benchmark"), "set -Dblock.benchmark=true");
        int[] prefill = java.util.Arrays.stream(System.getProperty("block.prefill", "512,2048").split(",")).mapToInt(Integer::parseInt).toArray();
        int[] decode = java.util.Arrays.stream(System.getProperty("block.decode", "512,4096,32768").split(",")).mapToInt(Integer::parseInt).toArray();
        int sustain = Integer.getInteger("block.sustainMinutes", 0);
        int maxTokens = java.util.Arrays.stream(prefill).max().orElse(1);
        int maxContext = Math.max(java.util.Arrays.stream(decode).max().orElse(1), maxTokens);
        System.out.println("block-device|" + device.adapter().name() + "|" + device.adapter().backend() + "|layers=" + layers + "|rounds=" + rounds);
        for (String wname : System.getProperty("block.weights", "FP16,Q4K").split(",")) {
            Format format = wname.equals("FP16") ? null : Format.valueOf(wname);
            try (var model = new Model(format, maxTokens, maxContext)) {
                for (int t : prefill) try (var a = new Activations(t)) { phase(model, a, wname, "prefill", t, 0); }
                for (int ctx : decode) try (var a = new Activations(1)) { phase(model, a, wname, "decode", 1, ctx - 1); }
                if (sustain > 0) sustained(model, wname, sustain);
            }
        }
    }

    private PagedKvCache.Sequence sequence(Model model, Activations a, int t, int start, SplittableRandom r) {
        var seq = model.cache.newSequence();
        run(e -> model.cache.reserve(e, seq, start + t));
        if (start > 0) {
            // history: random rotated keys and values for positions 0 .. start - 1, the same in every layer
            short[] hk = gaussianHalves(start * KD, 1, r), hv = gaussianHalves(start * KD, 1, r);
            try (var gk = buffer(2L * hk.length); var gv = buffer(2L * hv.length)) {
                run(e -> e.upload(gk, bytes(hk)).upload(gv, bytes(hv)));
                run(e -> { for (int l = 0; l < layers; l++) model.cache.write(e, l, seq, 0, gk, gv, start, HD, KD); });
            }
            history = new short[][] {hk, hv};
        } else history = null;
        // buffers hold at least 16 bytes; pad the positions to the whole buffer
        run(e -> e.upload(a.positions, java.util.Arrays.copyOf(ints(t, i -> start + i), (int) a.positions.bytes())));
        return seq;
    }

    private short[][] history;

    private void phase(Model model, Activations a, String weights, String phase, int t, int start) {
        var r = new SplittableRandom(31L * t + start);
        short[] x = gaussianHalves(t * D, 1, r);
        run(e -> e.upload(a.x0, bytes(x)));
        try (var seq = sequence(model, a, t, start, r)) {
            // check one layer, stage by stage
            run(e -> model.layer(e, 0, a, a.x0, a.x1, t, start, seq));
            double check = checkLayer(model, a, x, t, start);
            assertTrue(check <= 1, weights + " " + phase + " t=" + t + " start=" + start + ": a stage of layer 0 is off by " + check + " of its bound");
            // time the whole pass; every round starts from the same input
            byte[] xb = bytes(x);
            Consumer<GpuCommandEncoder> pass = e -> { e.upload(a.x0, xb); model.forward(e, a, t, start, seq, layers); };
            var timings = new ArrayList<GpuBenchmark.Timing>();
            for (int i = 0; i < 2; i++) GpuBenchmark.once(device, pass, 1);   // warm
            for (int i = 0; i < rounds; i++) timings.add(GpuBenchmark.once(device, pass, 1));
            var st = GpuBenchmark.Stats.of(timings);
            System.out.printf(Locale.ROOT, "block|%s|%s|%s|%d|%d|%d|%d|%.3f|%.3f|%.3f|%.3f|%.3f|%.1f|%.3g%n", device.adapter().backend(), weights, phase,
                    t, start + t, layers, st.n(), st.gpuMedian() / 1e3, st.gpuP10() / 1e3, st.gpuP90() / 1e3, st.wallMedian() / 1e3,
                    st.encodeMedian() / 1e3, t / (st.wallMedian() / 1e6), check);
            System.out.println("block-selected|" + weights + "|" + phase + "|" + t + "|q=" + model.products.selected(model.wq, t) + " o="
                    + model.products.selected(model.wo, t) + " up=" + model.products.selected(model.wup, t) + " down=" + model.products.selected(model.wdown, t));
            profile(model, a, weights, phase, t, start, seq);
        }
    }

    private void profile(Model model, Activations a, String weights, String phase, int t, int start, PagedKvCache.Sequence seq) {
        if (!device.capabilities().timestamps()) return;
        try (var e = device.encoder()) {
            e.profile();
            model.forward(e, a, t, start, seq, layers);
            try (var sub = device.queue().submit(e)) {
                assertTrue(sub.await(WAIT));
                var times = new TreeMap<String, Double>();
                var counts = new TreeMap<String, Integer>();
                for (var d : sub.dispatchTimings()) { times.merge(d.label(), d.time().toNanos() / 1e6, Double::sum); counts.merge(d.label(), 1, Integer::sum); }
                var line = new StringBuilder("block-stages|" + weights + "|" + phase + "|" + t + "|" + (start + t));
                times.forEach((k, v) -> line.append(String.format(Locale.ROOT, "|%s=%.3fms/%d", k, v, counts.get(k))));
                System.out.println(line);
            }
        }
    }

    private void sustained(Model model, String weights, int minutes) {
        int ctx = 4096, t = 512;
        var r = new SplittableRandom(99);
        short[] xp = gaussianHalves(t * D, 1, r), xd = gaussianHalves(D, 1, r);
        try (var ap = new Activations(t); var ad = new Activations(1); var seqP = sequence(model, ap, t, 0, r)) {
            var seqD = sequence(model, ad, 1, ctx - 1, r);
            byte[] xpb = bytes(xp), xdb = bytes(xd);
            try (seqD) {
                long end = System.nanoTime() + minutes * 60_000_000_000L, t0 = System.nanoTime();
                int minute = 0;
                var pre = new ArrayList<GpuBenchmark.Timing>();
                var dec = new ArrayList<GpuBenchmark.Timing>();
                while (System.nanoTime() < end) {
                    pre.add(GpuBenchmark.once(device, e -> { e.upload(ap.x0, xpb); model.forward(e, ap, t, 0, seqP, layers); }, 1));
                    for (int i = 0; i < 32; i++)
                        dec.add(GpuBenchmark.once(device, e -> { e.upload(ad.x0, xdb); model.forward(e, ad, 1, ctx - 1, seqD, layers); }, 1));
                    int now = (int) ((System.nanoTime() - t0) / 60_000_000_000L);
                    if (now > minute || System.nanoTime() >= end) {
                        var ps = GpuBenchmark.Stats.of(pre); var ds = GpuBenchmark.Stats.of(dec);
                        System.out.printf(Locale.ROOT, "block-sustained|%s|%s|minute=%d|prefill512 n=%d wall=%.3fms gpu=%.3fms tok/s=%.0f|decode@4096 n=%d wall=%.3fms gpu=%.3fms tok/s=%.1f%n",
                                device.adapter().backend(), weights, minute + 1, ps.n(), ps.wallMedian() / 1e3, ps.gpuMedian() / 1e3, t / (ps.wallMedian() / 1e6),
                                ds.n(), ds.wallMedian() / 1e3, ds.gpuMedian() / 1e3, 1 / (ds.wallMedian() / 1e6));
                        pre.clear(); dec.clear(); minute = now;
                    }
                }
            }
        }
    }

    // ---- checks: each stage of layer 0 against a double-precision computation from that stage's GPU inputs ----

    private double worst;

    private void expect(double got, double want, double scale, String where) {
        double err = Math.abs(got - want) / (scale + 1e-6);
        if (!(err <= worst)) worst = Double.isNaN(err) ? Double.POSITIVE_INFINITY : Math.max(worst, err);
        if (!(err <= 1)) System.out.printf(Locale.ROOT, "block-check-fail|%s|got=%g want=%g scale=%g%n", where, got, want, scale);
    }

    private double checkLayer(Model m, Activations a, short[] x, int t, int start) {
        worst = 0;
        var r = new SplittableRandom(5);
        short[] xn = halves(a.xn, t * D), q = halves(a.q, t * QD), k = halves(a.k, t * KD), v = halves(a.v, t * KD);
        short[] qr = halves(a.qr, t * QD), kr = halves(a.kr, t * KD), ot = halves(a.ot, t * QD), x2 = halves(a.x2, t * D);
        short[] xn2 = halves(a.xn2, t * D), up = halves(a.up, t * F), g = halves(a.g, t * F), out = halves(a.x1, t * D);
        float[] n1 = m.weights(m.norm1), n2 = m.weights(m.norm2);
        int samples = 24;
        for (int s = 0; s < samples; s++) {
            int row = r.nextInt(t);
            checkNorm(x, xn, n1, row, "norm1");
            checkNorm(x2, xn2, n2, row, "norm2");
            checkProduct(xn, m.wq, q, row, r.nextInt(QD), null, null, false, "q");
            checkProduct(xn, m.wk, k, row, r.nextInt(KD), null, null, false, "k");
            checkProduct(xn, m.wv, v, row, r.nextInt(KD), null, null, false, "v");
            checkRope(m, q, qr, H, row, start + row, r.nextInt(HD / 2), "rope q");
            checkRope(m, k, kr, KVH, row, start + row, r.nextInt(HD / 2), "rope k");
            checkAttention(qr, kr, v, ot, row, r.nextInt(H), start, "attention");
            checkProduct(ot, m.wo, x2, row, r.nextInt(D), null, x, false, "o + residual");
            checkProduct(xn2, m.wup, up, row, r.nextInt(F), null, null, false, "up");
            checkProduct(xn2, m.wgate, g, row, r.nextInt(F), up, null, true, "silu(gate) * up");
            checkProduct(g, m.wdown, out, row, r.nextInt(D), null, x2, false, "down + residual");
        }
        return worst;
    }

    private void checkNorm(short[] in, short[] out, float[] w, int row, String where) {
        double ss = 0;
        for (int c = 0; c < D; c++) { double v = f(in[row * D + c]); ss += v * v; }
        double inv = 1 / Math.sqrt(ss / D + EPS);
        for (int c = 0; c < D; c += 257) {
            double want = f(in[row * D + c]) * inv * w[c];
            expect(f(out[row * D + c]), want, 0x1p-9 * Math.abs(want) + 1e-3, where);
        }
    }

    private void checkProduct(short[] a, Weight w, short[] c, int row, int col, short[] gate, short[] residual, boolean silu, String where) {
        double dot = 0, mag = 0;
        for (int l = 0; l < w.k; l++) { double p = f(a[row * w.k + l]) * w.at(col, l); dot += p; mag += Math.abs(p); }
        double want = dot, scale = 4.0 * w.k * 0x1p-24 * mag;
        if (silu) {
            double sg = 1 / (1 + Math.exp(-dot)), gv = f(gate[row * w.n + col]);
            want = dot * sg * gv;
            scale = scale * Math.abs(gv) * 1.1 + 0x1p-10 * Math.abs(want);
        }
        if (residual != null) want += f(residual[row * w.n + col]);
        expect(f(c[row * w.n + col]), want, scale + 0x1p-9 * Math.abs(want) + 1e-3, where);
    }

    private void checkRope(Model m, short[] in, short[] out, int heads, int row, int pos, int i, String where) {
        int h = (int) (Math.abs((long) row * 7 + i) % heads), width = heads * HD, base = row * width + h * HD;
        double c = m.table.cos(pos, i), s = m.table.sin(pos, i);
        double x0 = f(in[base + i]), x1 = f(in[base + i + HD / 2]);
        double y0 = x0 * c - x1 * s, y1 = x0 * s + x1 * c;
        expect(f(out[base + i]), y0, 0x1p-9 * (Math.abs(x0) + Math.abs(x1)) + 1e-3, where);
        expect(f(out[base + i + HD / 2]), y1, 0x1p-9 * (Math.abs(x0) + Math.abs(x1)) + 1e-3, where);
    }

    /** Query {@code row} of head {@code h} over keys {@code 0 .. start + row}: history first, then this pass's keys. */
    private void checkAttention(short[] qr, short[] kr, short[] v, short[] ot, int row, int h, int start, String where) {
        int kvh = h / (H / KVH), keys = start + row + 1;
        double[] score = new double[keys];
        double max = Double.NEGATIVE_INFINITY, scale = 1 / Math.sqrt(HD), vmax = 0;
        for (int j = 0; j < keys; j++) {
            double dot = 0;
            for (int d = 0; d < HD; d++) dot += f(qr[row * QD + h * HD + d]) * key(kr, j, start, kvh, d);
            score[j] = dot * scale; max = Math.max(max, score[j]);
        }
        double sum = 0;
        for (int j = 0; j < keys; j++) { score[j] = Math.exp(score[j] - max); sum += score[j]; }
        for (int d = 0; d < HD; d += 17) {
            double o = 0;
            for (int j = 0; j < keys; j++) { double vv = value(v, j, start, kvh, d); o += score[j] * vv; vmax = Math.max(vmax, Math.abs(vv)); }
            expect(f(ot[row * QD + h * HD + d]), o / sum, 0x1p-7 * vmax + 1e-3, where);
        }
    }

    private double key(short[] kr, int j, int start, int kvh, int d) {
        return j < start ? f(history[0][j * KD + kvh * HD + d]) : f(kr[(j - start) * KD + kvh * HD + d]);
    }

    private double value(short[] v, int j, int start, int kvh, int d) {
        return j < start ? f(history[1][j * KD + kvh * HD + d]) : f(v[(j - start) * KD + kvh * HD + d]);
    }
}
