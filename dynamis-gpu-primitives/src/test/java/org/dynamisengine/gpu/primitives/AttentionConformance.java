package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.primitives.AttentionF16.Shape;
import org.junit.jupiter.api.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import java.util.function.ObjIntConsumer;
import static org.junit.jupiter.api.Assertions.*;

/**
 * AttentionF16 on a real device: every element of O within the composition of its steps' bounds (GemmF16, Softmax,
 * the FP16 rounding of P, the value product), computed in double precision from the same half inputs.
 */
abstract class AttentionConformance {
    protected abstract GpuProvider provider();
    protected GpuDevice device;
    static final Duration WAIT = Duration.ofSeconds(120);
    static final double U = 0x1p-24;

    @BeforeEach void open() { device = provider().adapters().getFirst().open(); }
    @AfterEach void close() { device.close(); }

    GpuBuffer buffer(long bytes) {
        return device.allocate(new BufferDescriptor((bytes + 3) & ~3L, EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE));
    }
    static short[] halves(long count, double spread, SplittableRandom r) {
        short[] h = new short[(int) count];
        for (int i = 0; i < h.length; i++) h[i] = Float.floatToFloat16((float) ((r.nextDouble() * 2 - 1) * spread));
        return h;
    }
    static byte[] bytes(short[] h) {
        var b = ByteBuffer.allocate((2 * h.length + 3) & ~3).order(ByteOrder.nativeOrder()); for (short v : h) b.putShort(v); return b.array();
    }
    static double f(short h) { return Float.float16ToFloat(h); }

    private record Mask(int group, int... lengths) {
        static final Mask NONE = new Mask(1);

        int visible(Shape s, int head, int query) {
            int causalLimit = s.causal() ? s.keys() - s.queries() + query + 1 : s.keys();
            return lengths.length == 0 ? causalLimit
                    : (int) Math.min(causalLimit, Integer.toUnsignedLong(lengths[head / group]));
        }
    }

    private GpuBuffer uploadMask(GpuCommandEncoder e, Mask mask) {
        if (mask.lengths.length == 0) return null;
        var data = ByteBuffer.allocate(Integer.BYTES * mask.lengths.length).order(ByteOrder.nativeOrder());
        for (int length : mask.lengths) data.putInt(length);
        var gpu = buffer(data.capacity());
        e.upload(gpu, data.array());
        return gpu;
    }

    /** Guard the cache tail that no query head sharing this KV head sees. */
    private static void poisonPadding(short[] k, short[] v, Shape s, Mask mask) {
        int headsPerKv = s.heads() / s.kvHeads();
        for (int kv = 0; kv < s.kvHeads(); kv++) {
            int visible = 0;
            for (int i = 0; i < headsPerKv; i++) {
                visible = Math.max(visible, mask.visible(s, kv * headsPerKv + i, s.queries() - 1));
            }
            int start = (kv * s.cacheKeys() + visible) * s.headDim();
            int end = (kv + 1) * s.cacheKeys() * s.headDim();
            java.util.Arrays.fill(k, start, end, (short) 0x7e00);
            java.util.Arrays.fill(v, start, end, (short) 0x7e00);
        }
    }

    /** The public 1.14.0 composition for padded dense heads. Pipelines live across all measured submissions. */
    private record MaskedComposition(GemmF16 qk, Softmax softmax, GemmF16 pv) implements AutoCloseable {
        static MaskedComposition create(GpuDevice device, Precision output) {
            var qk = GemmF16.create(device, Precision.FP32);
            Softmax softmax = null;
            try {
                softmax = Softmax.create(device, Precision.FP16);
                return new MaskedComposition(qk, softmax, GemmF16.create(device, qk.algorithm(), output));
            } catch (RuntimeException | Error failure) {
                if (softmax != null) softmax.close();
                qk.close();
                throw failure;
            }
        }

        void encode(GpuCommandEncoder e, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer o, GpuBuffer scores,
                GpuBuffer probs, Shape s, GpuBuffer lengths, int group) {
            qk.encode(e, q, k, scores, s.scoresShape());
            softmax.encode(e, scores, probs, s.softmaxShape(), lengths, group);
            pv.encode(e, probs, v, o, s.outputShape());
        }

        @Override public void close() {
            try (qk; softmax) { pv.close(); }
        }
    }

    /** Runs the attention and checks every element of O; returns the worst error / bound. */
    double check(AttentionF16 attention, Shape s, double spread, long seed) {
        return check(attention, s, spread, seed, Mask.NONE);
    }

    double check(AttentionF16 attention, Shape s, double spread, long seed, Mask mask) {
        var r = new SplittableRandom(seed);
        int d = s.headDim(), nq = s.queries(), nk = s.keys();
        short[] q = halves(s.qBytes() / 2, spread, r), k = halves(s.kBytes() / 2, spread, r), v = halves(s.vBytes() / 2, 1, r);
        var out = attention.output(); boolean fp32 = out == Precision.FP32;
        ByteBuffer got;
        try (var gq = buffer(s.qBytes()); var gk = buffer(s.kBytes()); var gv = buffer(s.vBytes()); var go = buffer(s.oBytes(out));
             var gs = buffer(s.scoresBytes()); var gp = buffer(s.probsBytes());
             var composed = mask == Mask.NONE ? null : MaskedComposition.create(device, out);
             var e = device.encoder(); var lens = uploadMask(e, mask)) {
            e.upload(gq, bytes(q)).upload(gk, bytes(k)).upload(gv, bytes(v));
            if (lens == null) attention.encode(e, gq, gk, gv, go, gs, gp, s);
            else composed.encode(e, gq, gk, gv, go, gs, gp, s, lens, mask.group);
            var read = e.download(go);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            got = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
        }
        double worst = 0, scale = s.scale();
        double[] sc = new double[nk], dS = new double[nk], p = new double[nk], dP = new double[nk];
        for (int h = 0; h < s.heads(); h++) {
            for (int i = 0; i < nq; i++) {
                int qb = (h * nq + i) * d, limit = mask.visible(s, h, i);
                double mx = Double.NEGATIVE_INFINITY, maxdS = 0;
                for (int j = 0; j < limit; j++) {
                    int kb = (h * nk + j) * d; double sum = 0, abs = 0;
                    for (int t = 0; t < d; t++) { double pr = f(q[qb + t]) * f(k[kb + t]); sum += pr; abs += Math.abs(pr); }
                    sc[j] = scale * sum; dS[j] = (d + 2) * U * Math.abs(scale) * abs; // GemmF16's bound
                    mx = Math.max(mx, sc[j]); maxdS = Math.max(maxdS, dS[j]);
                }
                double total = 0, weighted = 0;
                for (int j = 0; j < limit; j++) { p[j] = Math.exp(sc[j] - mx); total += p[j]; weighted += p[j] * (mx - sc[j]); }
                double xbar = limit == 0 ? 0 : weighted / total;
                for (int j = 0; j < limit; j++) {
                    p[j] /= total;
                    double x = mx - sc[j] + 2 * maxdS;
                    dP[j] = 2 * maxdS * p[j] * 1.01                                       // scores' error through the softmax
                            + (nk + 10 + 2 * x + 2 * (xbar + 2 * maxdS)) * U * p[j] + 0x1p-126 // Softmax's bound
                            + p[j] * 0x1p-11 * 1.01 + 0x1p-25;                               // P rounded to FP16
                }
                for (int c = 0; c < d; c++) {
                    double o = 0, err = 0, mag = 0;
                    for (int j = 0; j < limit; j++) {
                        double vv = f(v[(h * nk + j) * d + c]);
                        o += p[j] * vv; err += dP[j] * Math.abs(vv); mag += (p[j] + dP[j]) * Math.abs(vv);
                    }
                    double bound = err + (nk + 2) * U * mag + 0x1p-126;
                    if (!fp32) bound += (Math.abs(o) + bound) * 0x1p-11 + 0x1p-25;
                    int idx = (h * nq + i) * d + c;
                    double gv = fp32 ? got.getFloat(4 * idx) : Float.float16ToFloat(got.getShort(2 * idx));
                    if (limit == 0) assertEquals(0.0, gv, "empty composed row");
                    double e = Math.abs(gv - o);
                    assertTrue(e <= bound, "O[" + h + "][" + i + "][" + c + "] = " + gv + ", reference " + o + ", bound " + bound + " for " + s + " " + out);
                    worst = Math.max(worst, e / bound);
                }
            }
        }
        return worst;
    }

    /**
     * Runs the fused attention and checks every element of O against its bound ({@link AttentionF16#fusedContract});
     * returns the worst error / bound. K and V past {@code keys} of each head's cache are NaN: they must not be read.
     */
    double checkFused(AttentionF16 attention, Shape s, double spread, long seed) {
        return checkFused(attention, s, spread, seed, Mask.NONE);
    }

    double checkFused(AttentionF16 attention, Shape s, double spread, long seed, Mask mask) {
        var r = new SplittableRandom(seed);
        short[] q = halves(s.qBytes() / 2, spread, r), k = halves(s.kBytes() / 2, spread, r), v = halves(s.vBytes() / 2, 1, r);
        poisonPadding(k, v, s, mask);
        var out = attention.output(); boolean fp32 = out == Precision.FP32;
        ByteBuffer got;
        try (var gq = buffer(s.qBytes()); var gk = buffer(s.kBytes()); var gv = buffer(s.vBytes()); var go = buffer(s.oBytes(out)); var e = device.encoder(); var lens = uploadMask(e, mask)) {
            e.upload(gq, bytes(q)).upload(gk, bytes(k)).upload(gv, bytes(v));
            if (lens == null) attention.encode(e, gq, gk, gv, go, s);
            else attention.encode(e, gq, gk, gv, go, s, lens, mask.group);
            var read = e.download(go);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            got = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
        }
        return verifyFused(q, k, v, s, got, 0, attention.fusedKernel(s).equals("flash.simdgroup"), fp32, "fused", mask);
    }

    /**
     * Checks O for shape {@code s} (its elements from {@code outBase} of {@code got}) against the fused bound computed in
     * double precision from Q, and K and V as {@code [kvHeads][cacheKeys][headDim]}; returns the worst error / bound.
     */
    double verifyFused(short[] q, short[] k, short[] v, Shape s, ByteBuffer got, int outBase, boolean halfWeights, boolean fp32, String label, Mask mask) {
        int d = s.headDim(), nq = s.queries(), nk = s.keys(), cache = s.cacheKeys(), hpk = s.heads() / s.kvHeads();
        double worst = 0, scale = s.scale();
        int blocks = Math.ceilDiv(nk, AttentionF16.keyBlock(AttentionF16.variant(d))) + 2;
        double expErr = 8 * U;   // 4 ulp
        double[] sc = new double[nk], p = new double[nk], dP = new double[nk], eps = new double[nk];
        for (int h = 0; h < s.heads(); h++) {
            int kvh = h / hpk;
            for (int i = 0; i < nq; i++) {
                int qb = (h * nq + i) * d, limit = mask.visible(s, outBase / (nq * d) + h, i);
                double mx = Double.NEGATIVE_INFINITY, maxdS = 0;
                for (int j = 0; j < limit; j++) {
                    int kb = (kvh * cache + j) * d; double sum = 0, abs = 0;
                    for (int t = 0; t < d; t++) { double pr = f(q[qb + t]) * f(k[kb + t]); sum += pr; abs += Math.abs(pr); }
                    sc[j] = scale * sum;
                    maxdS = Math.max(maxdS, (d + 2) * U * Math.abs(scale) * abs);   // the scores' bound
                    mx = Math.max(mx, sc[j]);
                }
                double total = 0, epsMax = 0;
                for (int j = 0; j < limit; j++) {
                    p[j] = Math.exp(sc[j] - mx); total += p[j];
                    eps[j] = 2 * (mx - sc[j] + 2 * maxdS) * U + (blocks + 2) * expErr + blocks * U;
                    epsMax = Math.max(epsMax, eps[j]);
                }
                for (int j = 0; j < limit; j++) {
                    p[j] /= total;
                    dP[j] = (2 * maxdS + eps[j] + epsMax) * p[j] * 1.01 + 0x1p-126;
                    if (halfWeights) dP[j] += p[j] * 0x1p-11 * 1.01 + 0x1p-25;   // P rounded for its matrix product
                }
                for (int c = 0; c < d; c++) {
                    double o = 0, err = 0, mag = 0;
                    for (int j = 0; j < limit; j++) {
                        double vv = f(v[(kvh * cache + j) * d + c]);
                        o += p[j] * vv; err += dP[j] * Math.abs(vv); mag += (p[j] + dP[j]) * Math.abs(vv);
                    }
                    double bound = err + 2 * (nk + 2 + 2 * blocks) * U * mag + 0x1p-126;
                    bound += 2 * U * (Math.abs(o) + bound);   // acc / l
                    if (!fp32) bound += (Math.abs(o) + bound) * 0x1p-11 + 0x1p-25;
                    int idx = outBase + (h * nq + i) * d + c;
                    double gv = fp32 ? got.getFloat(4 * idx) : Float.float16ToFloat(got.getShort(2 * idx));
                    if (limit == 0) assertEquals(0.0, gv, label + " empty row");
                    double err2 = Math.abs(gv - o);
                    assertTrue(err2 <= bound, label + " O[" + h + "][" + i + "][" + c + "] = " + gv + ", reference " + o + ", bound " + bound + " for " + s + (fp32 ? " FP32" : " FP16"));
                    worst = Math.max(worst, err2 / bound);
                }
            }
        }
        return worst;
    }

    private static String dispatchDescription(AttentionF16 attention, Shape shape) {
        String kernel = attention.fusedKernel(shape);
        int splits = kernel.equals("decode.short") ? 1 : AttentionF16.split(shape)[0];
        return kernel + ", splits " + splits;
    }

    @Test void fusedMatchesTheReference() {
        eachOutput(a -> {
            for (var s : new Shape[] {
                    Shape.of(4, 64, 64, 64), Shape.of(3, 37, 50, 40), Shape.of(8, 128, 128, 128), Shape.of(2, 33, 70, 256),
                    Shape.of(4, 17, 17, 96).asCausal(), Shape.of(4, 64, 64, 64).asCausal(), Shape.of(2, 300, 300, 64).asCausal(),
                    Shape.of(2, 16, 300, 64).asCausal(),                                    // new queries on a cache
                    Shape.of(2, 1, 300, 64), Shape.of(4, 1, 3000, 128).asCausal(),         // decoding: the keys split
                    Shape.of(8, 5, 300, 128).withKvHeads(2).asCausal().withCacheKeys(512),  // grouped-query, a longer cache
                    Shape.of(32, 1, 2000, 128).withKvHeads(8).withCacheKeys(2048),           // a llama-like decode step
                    Shape.of(6, 7, 7, 2).withKvHeads(3),
                    Shape.of(16, 2, 9000, 64).withKvHeads(2).asCausal(),                     // flash, 16 rows per KV head
                    Shape.of(8, 2, 513, 64).withKvHeads(2).asCausal(),                       // decode, 8 rows and a final single key
                    Shape.of(16, 2, 512, 128).withKvHeads(4).withCacheKeys(513).asCausal(),  // 8 rows, short decode at capacity
                    Shape.of(8, 1, 513, 128).withKvHeads(2),                                // first longer context
                    Shape.of(6, 1, 129, 128).withKvHeads(2),                                // decode, 3 rows per KV head
                    Shape.of(2, 3, 5000, 256).withKvHeads(1).asCausal().withCacheKeys(6000), // decode at DMAX 256
                    Shape.of(4, 9, 40, 64).withKvHeads(4)}) {
                assertTrue(a.fusedSupports(s), s.toString());
                double worst = checkFused(a, s, 1, 47);
                System.out.printf("attention fused %s %s worst error/bound %.3g (%s)%n", a.output(), s, worst, dispatchDescription(a, s));
            }
            checkFused(a, Shape.of(2, 64, 96, 64), 4, 48);   // sharp softmax
        });
    }

    @Test void validLengthsInFusedAndComposedAttention() {
        eachOutput(a -> {
            // Unequal causal dimensions preserve the original row alignment. Heads 6 and 7 form a short last group.
            var composed = Shape.of(8, 13, 17, 40).asCausal();
            var groups = new Mask(3, 0, 8, -1);  // the uint32 maximum admits every key
            check(a, composed, 1, 110, groups);
            checkFused(a, composed, 1, 110, groups);

            // A mask group straddles a KV head. Some rows in the same work group see zero, a prefix or all keys.
            var mixed = new Mask(2, 0, 33, -1);
            for (var shape : new Shape[] {
                    Shape.of(6, 2, 97, 128).withKvHeads(2).asCausal(),             // short decode
                    Shape.of(6, 2, 641, 64).withKvHeads(2).asCausal(),             // split decode + combine
                    Shape.of(6, 17, 519, 128).withKvHeads(2).asCausal(),           // split matrix/portable flash + combine
                    Shape.of(6, 43, 19, 256).withKvHeads(2)}) {                   // single-range portable flash
                double worst = checkFused(a, shape, 1, 112, mixed);
                System.out.printf("attention masked %s %s worst error/bound %.3g (%s)%n", a.output(), shape, worst, dispatchDescription(a, shape));
            }
            try (var fixed = AttentionF16.create(device, a.output())) {
                fixed.useFixedTreeDecode();
                checkFused(fixed, Shape.of(6, 2, 641, 64).withKvHeads(2).asCausal(), 1, 113, mixed);
            }
        });
    }

    @Test void unsignedFullLengthsPreserveUnmaskedResults() {
        eachOutput(a -> {
            var random = new SplittableRandom(115);
            for (var shape : new Shape[] {Shape.of(4, 16, 35, 64), Shape.of(4, 1, 129, 128), Shape.of(4, 1, 517, 64)}) {
                try (var q = buffer(shape.qBytes()); var k = buffer(shape.kBytes()); var v = buffer(shape.vBytes());
                     var o = buffer(shape.oBytes(a.output())); var e = device.encoder(); var lengths = uploadMask(e, new Mask(4, -1))) {
                    e.upload(q, bytes(halves(q.bytes() / 2, 1, random))).upload(k, bytes(halves(k.bytes() / 2, 1, random)))
                            .upload(v, bytes(halves(v.bytes() / 2, 1, random)));
                    assertThrows(IllegalArgumentException.class, () -> a.encode(e, q, k, v, o, shape, lengths, 0));
                    assertThrows(IllegalArgumentException.class, () -> a.encode(e, q, k, v, o, shape, lengths, 3));
                    a.encode(e, q, k, v, o, shape);
                    var plain = e.download(o);
                    a.encode(e, q, k, v, o, shape, lengths, 4);
                    var masked = e.download(o);
                    try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
                    assertArrayEquals(plain.bytes(), masked.bytes(), "all keys visible: " + shape);
                }
            }
        });
    }

    byte[] runFused(AttentionF16 a, java.util.function.Consumer<GpuCommandEncoder> body, GpuBuffer go) {
        try (var e = device.encoder()) {
            body.accept(e);
            var read = e.download(go);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
            return read.bytes();
        }
    }

    /**
     * A paged KV cache built by reserve and write (head-major and token-major sources, two layers), with a fork that
     * shares a partly filled page and copies it on write: attention through each sequence's page table gives the same
     * bits as the contiguous attention over the same keys and values, and the host-side facts track every step.
     */
    @Test void pagedMatchesContiguous() {
        checkPagedMatchesContiguous(64);
        checkPagedMatchesContiguous(128);
    }

    private void checkPagedMatchesContiguous(int d) {
        int layers = 2, kvh = 2, ps = 32;
        var r = new SplittableRandom(50);
        try (var cache = PagedKvCache.create(device, new PagedKvCache.Config(layers, kvh, d, ps, 16)); var a = AttentionF16.create(device, Precision.FP32)) {
            // expected[layer][seq] K and V, [kvHeads][tokens][d]
            short[][][] ek = new short[layers][2][], ev = new short[layers][2][];
            var seqA = cache.newSequence();
            // A: 70 tokens, head-major sources
            int n0 = 70;
            try (var e = device.encoder()) {
                cache.reserve(e, seqA, n0);
                for (int l = 0; l < layers; l++) {
                    ek[l][0] = halves((long) kvh * n0 * d, 1, r); ev[l][0] = halves((long) kvh * n0 * d, 1, r);
                    try (var gk = buffer(2L * ek[l][0].length); var gv = buffer(2L * ev[l][0].length)) {
                        e.upload(gk, bytes(ek[l][0])).upload(gv, bytes(ev[l][0]));
                        cache.write(e, l, seqA, 0, gk, gv, n0, n0 * d, d);
                    }
                }
                try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
            }
            assertEquals(new PagedKvCache.Stats(16, 13, 3, 0, 0, 1), cache.stats());
            // B: a fork of A (3 shared pages, the last holding 6 tokens), then 5 tokens, token-major: B copies that page
            var seqB = cache.fork(seqA);
            assertEquals(new PagedKvCache.Stats(16, 13, 3, 0, 3, 2), cache.stats());
            int nb = 5, na = 40;
            short[][] bk = new short[layers][], bv = new short[layers][], ak = new short[layers][], av = new short[layers][];
            try (var e = device.encoder()) {
                cache.reserve(e, seqB, nb);
                assertEquals(new PagedKvCache.Stats(16, 12, 4, 0, 2, 2), cache.stats());
                for (int l = 0; l < layers; l++) {
                    bk[l] = halves((long) kvh * nb * d, 1, r); bv[l] = halves((long) kvh * nb * d, 1, r);   // [tokens][kvHeads][d]
                    try (var gk = buffer(2L * bk[l].length); var gv = buffer(2L * bv[l].length)) {
                        e.upload(gk, bytes(bk[l])).upload(gv, bytes(bv[l]));
                        cache.write(e, l, seqB, n0, gk, gv, nb, d, kvh * d);
                    }
                }
                // A grows into its own last page (no longer shared) and one more
                cache.reserve(e, seqA, na);
                for (int l = 0; l < layers; l++) {
                    ak[l] = halves((long) kvh * na * d, 1, r); av[l] = halves((long) kvh * na * d, 1, r);
                    try (var gk = buffer(2L * ak[l].length); var gv = buffer(2L * av[l].length)) {
                        e.upload(gk, bytes(ak[l])).upload(gv, bytes(av[l]));
                        cache.write(e, l, seqA, n0, gk, gv, na, na * d, d);
                    }
                }
                try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
            }
            assertEquals(110, seqA.tokens()); assertEquals(75, seqB.tokens());
            assertEquals(new PagedKvCache.Stats(16, 11, 5, 0, 2, 2), cache.stats());
            // the expected contiguous K and V of each sequence and layer
            for (int l = 0; l < layers; l++) {
                short[] ka = new short[kvh * 110 * d], va = new short[kvh * 110 * d], kb = new short[kvh * 75 * d], vb = new short[kvh * 75 * d];
                for (int h = 0; h < kvh; h++) for (int t = 0; t < 110; t++) for (int c = 0; c < d; c++) {
                    int at = (h * 110 + t) * d + c;
                    ka[at] = t < n0 ? ek[l][0][(h * n0 + t) * d + c] : ak[l][(h * na + t - n0) * d + c];
                    va[at] = t < n0 ? ev[l][0][(h * n0 + t) * d + c] : av[l][(h * na + t - n0) * d + c];
                }
                for (int h = 0; h < kvh; h++) for (int t = 0; t < 75; t++) for (int c = 0; c < d; c++) {
                    int at = (h * 75 + t) * d + c;
                    kb[at] = t < n0 ? ek[l][0][(h * n0 + t) * d + c] : bk[l][((t - n0) * kvh + h) * d + c];
                    vb[at] = t < n0 ? ev[l][0][(h * n0 + t) * d + c] : bv[l][((t - n0) * kvh + h) * d + c];
                }
                ek[l][0] = ka; ev[l][0] = va; ek[l][1] = kb; ev[l][1] = vb;
            }
            var seqs = new PagedKvCache.Sequence[] {seqA, seqB};
            for (int l = 0; l < layers; l++) for (int which = 0; which < 2; which++) {
                int keys = seqs[which].tokens();
                for (var shape : new Shape[] {new Shape(4, 1, keys, d, 0.125f, true, kvh, keys),         // decode
                        new Shape(4, 40, keys, d, 0.125f, true, kvh, keys),                              // a chunk of prefill
                        new Shape(2, 3, keys - 7, d, 0.125f, false, kvh, keys - 7)}) {                  // fewer keys than the sequence
                    short[] qh = halves(shape.qBytes() / 2, 1, r);
                    short[] kk = new short[kvh * shape.keys() * d], vv = new short[kvh * shape.keys() * d];
                    for (int h = 0; h < kvh; h++) for (int t = 0; t < shape.keys(); t++) for (int c = 0; c < d; c++) {
                        kk[(h * shape.keys() + t) * d + c] = ek[l][which][(h * keys + t) * d + c];
                        vv[(h * shape.keys() + t) * d + c] = ev[l][which][(h * keys + t) * d + c];
                    }
                    int layer = l, w = which;
                    try (var gq = buffer(shape.qBytes()); var gk = buffer(2L * kk.length); var gv = buffer(2L * vv.length); var go = buffer(shape.oBytes(Precision.FP32))) {
                        byte[] contiguous = runFused(a, e -> { e.upload(gq, bytes(qh)).upload(gk, bytes(kk)).upload(gv, bytes(vv)); a.encode(e, gq, gk, gv, go, shape); }, go);
                        byte[] paged = runFused(a, e -> { e.upload(gq, bytes(qh)); a.encode(e, gq, cache, layer, seqs[w], go, shape); }, go);
                        assertArrayEquals(contiguous, paged, "paged attention, layer " + layer + ", sequence " + w + ", " + shape + " (" + a.fusedKernel(shape) + ")");
                        if (layer == 0 && w == 1) {
                            var mask = new Mask(2, 17, 0);
                            byte[] masked = runFused(a, e -> {
                                try (var lens = uploadMask(e, mask)) { a.encode(e, gq, cache, layer, seqs[w], go, shape, lens, mask.group); }
                            }, go);
                            verifyFused(qh, kk, vv, shape, ByteBuffer.wrap(masked).order(ByteOrder.nativeOrder()), 0,
                                    a.fusedKernel(shape).equals("flash.simdgroup"), true, "masked paged", mask);
                        }
                    }
                }
            }
            // truncation and release
            cache.truncate(seqA, 50);
            assertEquals(new PagedKvCache.Stats(16, 13, 3, 0, 2, 2), cache.stats());
            seqB.pin(false);
            assertEquals(new PagedKvCache.Stats(16, 13, 2, 1, 2, 2), cache.stats());   // B's own page is evictable
            seqA.close();
            assertEquals(new PagedKvCache.Stats(16, 13, 0, 3, 0, 1), cache.stats());
            seqB.close();
            assertEquals(new PagedKvCache.Stats(16, 16, 0, 0, 0, 0), cache.stats());
            var big = cache.newSequence();
            try (var e = device.encoder()) {
                var ex = assertThrows(ComputeException.class, () -> cache.reserve(e, big, 17 * ps));
                assertEquals(ComputeException.Code.RESOURCE_EXHAUSTED, ex.code());
            }
            assertEquals(0, big.tokens());
        }
    }

    /**
     * Batched decode (1.11.1): sequences of different lengths in one paged cache, some sharing pages, decoded in one
     * dispatch; each sequence's rows of O within the fused bound of its own keys and values.
     */
    @Test void batchedDecodeMatchesEachSequence() {
        eachOutput(a -> {
            // mixed lengths, the keys split into ranges and combined
            checkBatch(a, AttentionF16.DecodeBatch.of(8, 2, 64), new int[] {1, 37, 300, 1000, 4100}, false, 60);
            // a large batch: one range per sequence, taken in chunks (no combine)
            var random = new SplittableRandom(61);
            int[] lengths = new int[33];
            for (int i = 0; i < lengths.length; i++) lengths[i] = 1 + random.nextInt(700);
            checkBatch(a, AttentionF16.DecodeBatch.of(32, 8, 128), lengths, false, 62);
            // two queries per sequence (8 rows a KV head), and a fork that shares its parent's pages
            checkBatch(a, new AttentionF16.DecodeBatch(8, 2, 2, 128, 0.0884f, true), new int[] {2, 70, 513, 200}, true, 63);
        });
    }

    /**
     * The fixed-tree decode kernel, which devices without subgroup arithmetic run: every device here has subgroups,
     * so it is forced, through single and batched decode (chunked ranges included).
     */
    @Test void fixedTreeDecodeMatchesTheReference() {
        for (var out : Precision.values()) {
            try (var a = AttentionF16.create(device, out)) {
                a.useFixedTreeDecode();
                for (var s : new Shape[] {Shape.of(2, 1, 300, 64), Shape.of(32, 1, 2000, 128).withKvHeads(8).withCacheKeys(2048),
                        Shape.of(8, 2, 513, 64).withKvHeads(2).asCausal(), Shape.of(2, 3, 5000, 256).withKvHeads(1).asCausal()}) {
                    double worst = checkFused(a, s, 1, 64);
                    System.out.printf("attention fixed-tree decode %s %s worst error/bound %.3g (%s)%n", out, s, worst, dispatchDescription(a, s));
                }
                var random = new SplittableRandom(65);
                int[] lengths = new int[33];
                for (int i = 0; i < lengths.length; i++) lengths[i] = 1 + random.nextInt(700);
                checkBatch(a, AttentionF16.DecodeBatch.of(32, 8, 128), lengths, false, 66);
                checkBatch(a, AttentionF16.DecodeBatch.of(8, 2, 64), new int[] {1, 37, 300, 1000, 4100}, false, 67);
            }
        }
    }

    @Test void validLengthsInBatchedPagedDecode() {
        eachOutput(a -> {
            var shape = new AttentionF16.DecodeBatch(4, 2, 2, 64, 0.125f, true);
            // The middle group crosses the sequence boundary: masks index [sequence][query head].
            var mask = new Mask(3, 0, 7, -1);
            int[] tokens = {41, 513};
            checkBatch(a, shape, tokens, false, 120, mask);
            try (var fixed = AttentionF16.create(device, a.output())) {
                fixed.useFixedTreeDecode();
                checkBatch(fixed, shape, tokens, false, 120, mask);
            }
        });
    }

    void checkBatch(AttentionF16 a, AttentionF16.DecodeBatch b, int[] lengths, boolean fork, long seed) {
        checkBatch(a, b, lengths, fork, seed, Mask.NONE);
    }

    void checkBatch(AttentionF16 a, AttentionF16.DecodeBatch b, int[] lengths, boolean fork, long seed, Mask mask) {
        var r = new SplittableRandom(seed);
        int kvh = b.kvHeads(), d = b.headDim(), ps = 32, batch = lengths.length + (fork ? 1 : 0);
        int pages = 8;
        for (int n : lengths) pages += Math.ceilDiv(n + 40, ps);
        try (var cache = PagedKvCache.create(device, new PagedKvCache.Config(1, kvh, d, ps, pages))) {
            var seqs = new java.util.ArrayList<PagedKvCache.Sequence>();
            var ks = new java.util.ArrayList<short[]>();
            var vs = new java.util.ArrayList<short[]>();
            try (var e = device.encoder()) {
                for (int n : lengths) {
                    var seq = cache.newSequence();
                    short[] k = halves((long) kvh * n * d, 1, r), v = halves((long) kvh * n * d, 1, r);
                    cache.reserve(e, seq, n);
                    try (var gk = buffer(2L * k.length); var gv = buffer(2L * v.length)) {
                        e.upload(gk, bytes(k)).upload(gv, bytes(v));
                        cache.write(e, 0, seq, 0, gk, gv, n, n * d, d);
                    }
                    seqs.add(seq); ks.add(k); vs.add(v);
                }
                if (fork) {
                    // a fork of the third sequence that grows by 40 tokens past a partly filled shared page
                    int parent = 2, n0 = lengths[parent], extra = 40, n = n0 + extra;
                    var seq = cache.fork(seqs.get(parent));
                    cache.reserve(e, seq, extra);
                    short[] k = halves((long) kvh * extra * d, 1, r), v = halves((long) kvh * extra * d, 1, r);
                    try (var gk = buffer(2L * k.length); var gv = buffer(2L * v.length)) {
                        e.upload(gk, bytes(k)).upload(gv, bytes(v));
                        cache.write(e, 0, seq, n0, gk, gv, extra, extra * d, d);
                    }
                    short[] fk = new short[kvh * n * d], fv = new short[kvh * n * d];
                    for (int h = 0; h < kvh; h++) for (int t = 0; t < n; t++) for (int c = 0; c < d; c++) {
                        fk[(h * n + t) * d + c] = t < n0 ? ks.get(parent)[(h * n0 + t) * d + c] : k[(h * extra + t - n0) * d + c];
                        fv[(h * n + t) * d + c] = t < n0 ? vs.get(parent)[(h * n0 + t) * d + c] : v[(h * extra + t - n0) * d + c];
                    }
                    seqs.add(seq); ks.add(fk); vs.add(fv);
                }
                try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
            }
            short[] q = halves(b.qBytes(batch) / 2, 1, r);
            var out = a.output(); boolean fp32 = out == Precision.FP32;
            ByteBuffer got;
            try (var gq = buffer(b.qBytes(batch)); var go = buffer(b.oBytes(out, batch)); var e = device.encoder(); var lens = uploadMask(e, mask)) {
                e.upload(gq, bytes(q));
                if (lens == null) a.encodeBatch(e, gq, cache, 0, seqs, go, b);
                else a.encodeBatch(e, gq, cache, 0, seqs, go, b, lens, mask.group);
                var read = e.download(go);
                try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
                got = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
            }
            int maxKeys = 0;
            for (var seq : seqs) maxKeys = Math.max(maxKeys, seq.tokens());
            int perSeq = b.heads() * b.queries() * d;
            double worst = 0;
            for (int i = 0; i < batch; i++) {
                var s = b.shape(seqs.get(i).tokens());
                short[] qi = java.util.Arrays.copyOfRange(q, i * perSeq, (i + 1) * perSeq);
                worst = Math.max(worst, verifyFused(qi, ks.get(i), vs.get(i), s, got, i * perSeq, false, fp32, "batched decode, sequence " + i, mask));
            }
            int[] sp = AttentionF16.batchSplit(b, batch, maxKeys);
            System.out.printf("attention batched decode %s %s, %d sequences up to %d keys: worst error/bound %.3g (splits %d of %d keys)%n",
                    out, b, batch, maxKeys, worst, sp[0], sp[1]);
            for (var seq : seqs) seq.close();
        }
    }

    /** -Dgemm.benchmark=true: a batched decode step against one encode per sequence (host, GPU and wall time). */
    @Test void benchmarkBatchedDecode() {
        Assumptions.assumeTrue(Boolean.getBoolean("gemm.benchmark"), "set -Dgemm.benchmark=true");
        var b = AttentionF16.DecodeBatch.of(32, 8, 128);
        for (int[] cfg : new int[][] {{8, 4096}, {32, 1024}, {32, 4096}, {64, 512}}) {
            int batch = cfg[0], keys = cfg[1], ps = 64;
            var config = new PagedKvCache.Config(1, 8, 128, ps, batch * Math.ceilDiv(keys, ps));
            try (var a = AttentionF16.create(device, Precision.FP16); var cache = PagedKvCache.create(device, config);
                 var q = buffer(b.qBytes(batch)); var o = buffer(b.oBytes(Precision.FP16, batch)); var init = device.encoder()) {
                var seqs = new java.util.ArrayList<PagedKvCache.Sequence>();
                for (int i = 0; i < batch; i++) { var seq = cache.newSequence(); cache.reserve(init, seq, keys); seqs.add(seq); }
                init.fill(q, 0x3800b800).fill(cache.keys(0), 0x34003800).fill(cache.values(0), 0xbc003c00);
                try (var sub = device.queue().submit(init)) { assertTrue(sub.await(WAIT)); }
                var batched = GpuBenchmark.measure(device, e -> a.encodeBatch(e, q, cache, 0, seqs, o, b));
                var s = b.shape(keys);
                var single = GpuBenchmark.measure(device, e -> {
                    for (var seq : seqs) a.encode(e, q, cache, 0, seq, o, s);   // the same work, one sequence at a time
                });
                double bytes = 2.0 * batch * keys * 8 * 128 * 2;
                System.out.printf("attention %s batched decode %d x %d keys: batched GPU %.1f us (%.0f GB/s), host %.1f us; one encode each: GPU %.1f us, host %.1f us%n",
                        device.adapter().name(), batch, keys, batched.gpuMicros(), bytes / batched.gpuMicros() / 1e3, batched.encodeMicros(),
                        single.gpuMicros(), single.encodeMicros());
                for (var seq : seqs) seq.close();
            }
        }
    }

    /**
     * The training forward pass (1.12.0): each row's log-sum-exp of the scaled scores within its bound of the exact
     * value from the same half inputs, and, for shapes that already take a flash kernel, the same O bits as without it.
     */
    @Test void forwardWritesTheLogSumExp() {
        eachOutput(a -> {
            for (var s : new Shape[] {Shape.of(4, 64, 64, 64).asCausal(), Shape.of(3, 37, 50, 40), Shape.of(8, 128, 128, 128).asCausal(),
                    Shape.of(2, 16, 300, 64).asCausal(), Shape.of(8, 5, 300, 128).withKvHeads(2).asCausal(), Shape.of(2, 33, 70, 256),
                    Shape.of(4, 1, 3000, 128).asCausal()}) {
                var r = new SplittableRandom(91);
                int d = s.headDim(), nq = s.queries(), nk = s.keys(), hpk = s.heads() / s.kvHeads();
                short[] q = halves(s.qBytes() / 2, 1, r), k = halves(s.kBytes() / 2, 1, r), v = halves(s.vBytes() / 2, 1, r);
                byte[] withLse, plain, lseBytes;
                try (var gq = buffer(s.qBytes()); var gk = buffer(s.kBytes()); var gv = buffer(s.vBytes()); var go = buffer(s.oBytes(a.output()));
                     var gl = buffer(4L * s.heads() * nq)) {
                    try (var e = device.encoder()) {
                        e.upload(gq, bytes(q)).upload(gk, bytes(k)).upload(gv, bytes(v));
                        a.encode(e, gq, gk, gv, go, gl, s);
                        var ro = e.download(go); var rl = e.download(gl);
                        try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
                        withLse = ro.bytes(); lseBytes = rl.bytes();
                    }
                    plain = runFused(a, e -> a.encode(e, gq, gk, gv, go, s), go);
                }
                if (!AttentionF16.decodes(s)) assertArrayEquals(plain, withLse, "O with and without the log-sum-exp: " + s);
                var lb = ByteBuffer.wrap(lseBytes).order(ByteOrder.nativeOrder());
                int blocks = Math.ceilDiv(nk, AttentionF16.keyBlock(AttentionF16.variant(d))) + 2;
                double worst = 0;
                for (int h = 0; h < s.heads(); h++) for (int i = 0; i < nq; i++) {
                    int kvh = h / hpk, limit = s.causal() ? i + nk - nq + 1 : nk, qb = (h * nq + i) * d;
                    double mx = Double.NEGATIVE_INFINITY, maxdS = 0;
                    double[] sc = new double[limit];
                    for (int j = 0; j < limit; j++) {
                        double sum = 0, abs = 0;
                        for (int t = 0; t < d; t++) { double pr = f(q[qb + t]) * f(k[(kvh * nk + j) * d + t]); sum += pr; abs += Math.abs(pr); }
                        sc[j] = s.scale() * sum; mx = Math.max(mx, sc[j]); maxdS = Math.max(maxdS, (d + 2) * U * Math.abs(s.scale()) * abs);
                    }
                    double sum = 0, dev = 0;
                    for (int j = 0; j < limit; j++) { sum += Math.exp(sc[j] - mx); dev = Math.max(dev, mx - sc[j]); }
                    double exact = mx + Math.log(sum);
                    double bound = 2 * maxdS + U * (Math.abs(exact) + Math.abs(mx)) + (nk + 2 + 2 * blocks) * U + (blocks + 2) * 8 * U
                            + 2 * (dev + 2 * maxdS) * U + 4 * Math.ulp((float) Math.log(sum));
                    double got = lb.getFloat(4 * (h * nq + i)), err = Math.abs(got - exact);
                    assertTrue(err <= bound, "lse[" + h + "][" + i + "] = " + got + ", exact " + exact + ", bound " + bound + " for " + s);
                    worst = Math.max(worst, err / bound);
                }
                System.out.printf("attention log-sum-exp %s %s worst error/bound %.3g (%s)%n", a.output(), s, worst, a.fusedKernel(s));
            }
        });
    }

    /**
     * The backward pass (1.12.0) after the training forward pass on the GPU: dQ, dK and dV against the exact gradients in
     * double from the same half inputs, each within a bound that carries the forward pass's errors (O and L) and the
     * backward's own through every step.
     */
    @Test void backwardMatchesTheReference() {
        for (boolean scalar : new boolean[] {false, true}) for (var out : Precision.values()) try (var a = AttentionF16.create(device, out)) {
            if (scalar) a.useScalarBackward();   // the portable kernels, where the matrix ones would serve
            for (var s : new Shape[] {Shape.of(4, 64, 64, 64).asCausal(), Shape.of(3, 37, 50, 40), Shape.of(8, 20, 45, 128).withKvHeads(2).asCausal(),
                    Shape.of(2, 9, 33, 256), Shape.of(2, 1, 300, 64), Shape.of(2, 30, 30, 2).asCausal(), Shape.of(4, 17, 40, 128).withKvHeads(4).withCacheKeys(48)}) {
                assertTrue(a.backwardSupports(s), s.toString());
                if (scalar && !a.matrixPath(s.headDim())) continue;   // already the portable kernels
                double worst = checkBackward(a, s, 95);
                System.out.printf("attention backward %s O %s worst error/bound %.3g (%s)%n", a.output(), s, worst, a.matrixBackward(s) ? "matrix" : "portable");
            }
        }
    }

    @Test void validLengthsInTrainingForwardAndBackward() {
        eachOutput(a -> {
            var mask = new Mask(2, 0, 9, 23);
            for (var s : new Shape[] {
                    Shape.of(6, 19, 37, 128).withKvHeads(2).asCausal().withCacheKeys(42),
                    Shape.of(6, 13, 19, 40).withKvHeads(2).asCausal(),
                    Shape.of(6, 9, 519, 64).withKvHeads(2)}) {
                double worst = checkBackward(a, s, 117, mask);
                System.out.printf("attention masked backward %s %s worst error/bound %.3g (%s)%n",
                        a.output(), s, worst, a.matrixBackward(s) ? "matrix" : "portable");
            }
            if (a.matrixPath(128)) {
                a.useScalarBackward();
                checkBackward(a, Shape.of(6, 19, 37, 128).withKvHeads(2).asCausal(), 117, mask);
            }
        });
    }

    double checkBackward(AttentionF16 a, Shape s, long seed) {
        return checkBackward(a, s, seed, Mask.NONE);
    }

    double checkBackward(AttentionF16 a, Shape s, long seed, Mask mask) {
        var r = new SplittableRandom(seed);
        int d = s.headDim(), nq = s.queries(), nk = s.keys(), cache = s.cacheKeys(), hpk = s.heads() / s.kvHeads(), R = hpk * nq;
        short[] q = halves(s.qBytes() / 2, 1, r), k = halves(s.kBytes() / 2, 1, r), v = halves(s.vBytes() / 2, 1, r);
        if (mask != Mask.NONE) poisonPadding(k, v, s, mask);
        int qn = s.heads() * nq * d, kn = s.kvHeads() * cache * d;
        float[] dO = new float[qn];
        for (int i = 0; i < qn; i++) dO[i] = (float) r.nextDouble(-1, 1);
        var out = a.output(); boolean fp32 = out == Precision.FP32;
        ByteBuffer gq2, gk2, gv2, forwardO = null, savedLse = null;
        try (var gq = buffer(s.qBytes()); var gk = buffer(s.kBytes()); var gv = buffer(s.vBytes()); var go = buffer(s.oBytes(out));
             var gl = buffer(4L * s.heads() * nq); var gdo = buffer(4L * qn); var gdq = buffer(4L * qn); var gdk = buffer(4L * kn); var gdv = buffer(4L * kn);
             var e = device.encoder(); var lens = uploadMask(e, mask)) {
            var db = ByteBuffer.allocate(4 * qn).order(ByteOrder.nativeOrder()); for (float x : dO) db.putFloat(x);
            e.upload(gq, bytes(q)).upload(gk, bytes(k)).upload(gv, bytes(v)).upload(gdo, db.array()).fill(gdk, 0).fill(gdv, 0);
            if (lens == null) {
                a.encode(e, gq, gk, gv, go, gl, s);
                a.backward(e, gq, gk, gv, go, gl, gdo, gdq, gdk, gdv, s);
            } else {
                // A missed output store should fail even for a fully masked row.
                e.fill(go, 0x3c003c00).fill(gl, 0x3f800000).fill(gdq, 0x3f800000);
                e.fill(gdk, 0x3f800000).fill(gdv, 0x3f800000);
                a.encode(e, gq, gk, gv, go, gl, s, lens, mask.group);
                a.backward(e, gq, gk, gv, go, gl, gdo, gdq, gdk, gdv, s, lens, mask.group);
            }
            var ro = lens == null ? null : e.download(go);
            var rl = lens == null ? null : e.download(gl);
            var rq = e.download(gdq); var rk = e.download(gdk); var rv = e.download(gdv);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            if (ro != null) {
                forwardO = ByteBuffer.wrap(ro.bytes()).order(ByteOrder.nativeOrder());
                savedLse = ByteBuffer.wrap(rl.bytes()).order(ByteOrder.nativeOrder());
            }
            gq2 = ByteBuffer.wrap(rq.bytes()).order(ByteOrder.nativeOrder());
            gk2 = ByteBuffer.wrap(rk.bytes()).order(ByteOrder.nativeOrder());
            gv2 = ByteBuffer.wrap(rv.bytes()).order(ByteOrder.nativeOrder());
        }
        boolean halfWeights = a.matrixPath(d), halfOps = a.matrixBackward(s);
        if (forwardO != null) verifyFused(q, k, v, s, forwardO, 0, halfWeights, fp32, "training", mask);
        int blocks = Math.ceilDiv(nk, AttentionF16.keyBlock(AttentionF16.variant(d))) + 2;
        double E = 8 * U, scale = s.scale();
        double[] eK = new double[kn], bK = new double[kn], mK = new double[kn], eV = new double[kn], bV = new double[kn], mV = new double[kn];
        double worst = 0;
        double[] sc = new double[nk], sig = new double[nk], pj = new double[nk], dPj = new double[nk], dSj = new double[nk], ddS = new double[nk], dPf = new double[nk];
        for (int kvh = 0; kvh < s.kvHeads(); kvh++) for (int row = 0; row < R; row++) {
            int h = kvh * hpk + row / nq, i = row % nq, qb = (h * nq + i) * d, limit = mask.visible(s, h, i);
            if (limit == 0) {
                assertEquals(Float.NEGATIVE_INFINITY, savedLse.getFloat(4 * (h * nq + i)), "empty row LSE");
                for (int c = 0; c < d; c++) assertEquals(0f, gq2.getFloat(4 * (qb + c)), "empty row dQ");
                continue;
            }
            double mx = Double.NEGATIVE_INFINITY, smax = 0;
            for (int j = 0; j < limit; j++) {
                double sum = 0, abs = 0;
                for (int t = 0; t < d; t++) { double pr = f(q[qb + t]) * f(k[(kvh * cache + j) * d + t]); sum += pr; abs += Math.abs(pr); }
                sc[j] = scale * sum; sig[j] = (d + 2) * U * Math.abs(scale) * abs; mx = Math.max(mx, sc[j]); smax = Math.max(smax, sig[j]);
            }
            double total = 0, dev = 0;
            for (int j = 0; j < limit; j++) { total += Math.exp(sc[j] - mx); dev = Math.max(dev, mx - sc[j]); }
            double L = mx + Math.log(total);
            double dL = 2 * smax + U * (Math.abs(L) + Math.abs(mx)) + (nk + 2 + 2 * blocks) * U + (blocks + 2) * E + 2 * (dev + 2 * smax) * U
                    + 4 * Math.ulp((float) Math.log(total));
            if (savedLse != null) {
                double got = savedLse.getFloat(4 * (h * nq + i));
                assertTrue(Math.abs(got - L) <= dL, "masked LSE " + got + ", exact " + L + ", bound " + dL);
            }
            // the forward pass's O and its error, the exact probabilities
            double epsMax = 0;
            double[] eps = new double[limit];
            for (int j = 0; j < limit; j++) {
                pj[j] = Math.exp(sc[j] - L);
                eps[j] = 2 * (mx - sc[j] + 2 * smax) * U + (blocks + 2) * E + blocks * U; epsMax = Math.max(epsMax, eps[j]);
            }
            for (int j = 0; j < limit; j++) {
                dPf[j] = (2 * smax + eps[j] + epsMax) * pj[j] * 1.01 + 0x1p-126;
                if (halfWeights) dPf[j] += pj[j] * 0x1p-11 * 1.01 + 0x1p-25;
            }
            double D = 0, dD = 0, magD = 0;
            for (int c = 0; c < d; c++) {
                double o = 0, err = 0, mag = 0;
                for (int j = 0; j < limit; j++) {
                    double vv = f(v[(kvh * cache + j) * d + c]);
                    o += pj[j] * vv; err += dPf[j] * Math.abs(vv); mag += (pj[j] + dPf[j]) * Math.abs(vv);
                }
                double dO_ = err + 2 * (nk + 2 + 2 * blocks) * U * mag + 0x1p-126;
                dO_ += 2 * U * (Math.abs(o) + dO_);
                if (!fp32) dO_ += (Math.abs(o) + dO_) * 0x1p-11 + 0x1p-25;
                double g = dO[qb + c];
                D += g * o; dD += Math.abs(g) * dO_; magD += Math.abs(g * o);
            }
            dD += (d + 2) * U * magD;
            for (int j = 0; j < limit; j++) {
                double dp = 0, magP = 0;
                for (int c = 0; c < d; c++) { double t = dO[qb + c] * f(v[(kvh * cache + j) * d + c]); dp += t; magP += Math.abs(t); }
                double x = sc[j] - L;
                double eP = sig[j] + dL + U * (Math.abs(sc[j]) + Math.abs(x)) + (8 + 4 * Math.abs(x)) * U;
                double dPv = pj[j] * eP * 1.01 + 0x1p-126;
                dPj[j] = dPv;
                dSj[j] = pj[j] * (dp - D);
                ddS[j] = dPv * Math.abs(dp - D) + pj[j] * ((d + 2) * U * magP + dD + U * Math.abs(dp - D)) + U * Math.abs(dSj[j]) + 0x1p-126;
                // the matrix path: dO rounded to FP16 for dP, and dS rounded to FP16 for its products
                if (halfOps) ddS[j] += pj[j] * 0x1p-11 * magP * 1.01 + (Math.abs(dSj[j]) + ddS[j]) * 0x1p-11 * 1.01 + 0x1p-25;
            }
            // dQ for this row
            for (int c = 0; c < d; c++) {
                double exact = 0, err = 0, mag = 0;
                for (int j = 0; j < limit; j++) {
                    double kv = f(k[(kvh * cache + j) * d + c]);
                    exact += dSj[j] * kv; err += ddS[j] * Math.abs(kv); mag += Math.abs(dSj[j] * kv);
                }
                exact *= scale;
                double bound = Math.abs(scale) * (err + (nk + 2) * U * mag) * 1.01 + U * Math.abs(exact) + 0x1p-126;
                double got = gq2.getFloat(4 * (qb + c)), e = Math.abs(got - exact);
                assertTrue(e <= bound, "dQ[" + h + "][" + i + "][" + c + "] = " + got + ", exact " + exact + ", bound " + bound + " for " + s);
                worst = Math.max(worst, e / bound);
            }
            // contributions to dK and dV
            for (int j = 0; j < limit; j++) for (int c = 0; c < d; c++) {
                int at = (kvh * cache + j) * d + c;
                double g = dO[qb + c], qq = f(q[qb + c]);
                eV[at] += pj[j] * g; bV[at] += dPj[j] * Math.abs(g); mV[at] += pj[j] * Math.abs(g);
                if (halfOps) bV[at] += (pj[j] + dPj[j]) * Math.abs(g) * 2 * 0x1p-11 * 1.01 + 0x1p-25 * Math.abs(g);   // P and dO rounded to FP16
                eK[at] += dSj[j] * qq; bK[at] += ddS[j] * Math.abs(qq); mK[at] += Math.abs(dSj[j] * qq);
            }
        }
        for (int kvh = 0; kvh < s.kvHeads(); kvh++) for (int j = 0; j < nk; j++) for (int c = 0; c < d; c++) {
            int at = (kvh * cache + j) * d + c;
            double exK = scale * eK[at], boundK = Math.abs(scale) * (bK[at] + (R + 2) * U * mK[at]) * 1.01 + U * Math.abs(exK) + 0x1p-126;
            double boundV = (bV[at] + (R + 2) * U * mV[at]) * 1.01 + 0x1p-126;
            double gotK = gk2.getFloat(4 * at), gotV = gv2.getFloat(4 * at);
            if (mask != Mask.NONE && mV[at] == 0) {
                assertEquals(0f, gotK, "masked column dK");
                assertEquals(0f, gotV, "masked column dV");
            }
            assertTrue(Math.abs(gotK - exK) <= boundK, "dK[" + kvh + "][" + j + "][" + c + "] = " + gotK + ", exact " + exK + ", bound " + boundK + " for " + s);
            assertTrue(Math.abs(gotV - eV[at]) <= boundV, "dV[" + kvh + "][" + j + "][" + c + "] = " + gotV + ", exact " + eV[at] + ", bound " + boundV + " for " + s);
            worst = Math.max(worst, Math.max(Math.abs(gotK - exK) / boundK, Math.abs(gotV - eV[at]) / boundV));
        }
        for (int kvh = 0; kvh < s.kvHeads(); kvh++) for (int j = nk; j < cache; j++) for (int c = 0; c < d; c++) {
            int offset = 4 * ((kvh * cache + j) * d + c);
            float initial = mask == Mask.NONE ? 0f : 1f;
            assertEquals(initial, gk2.getFloat(offset), "dK past the keys is not written");
            assertEquals(initial, gv2.getFloat(offset), "dV past the keys is not written");
        }
        return worst;
    }

    @Test void fusedRefusesWhatItCannotTake() {
        eachOutput(a -> {
            assertFalse(a.fusedSupports(Shape.of(1, 1, 1, 7)));
            assertFalse(a.fusedSupports(Shape.of(1, 1, 1, 258)));
            assertThrows(ComputeException.class, () -> a.encode(null, null, null, null, null, Shape.of(1, 1, 1, 7)));
            assertThrows(IllegalArgumentException.class, () -> a.encode(null, null, null, null, null, null, null, Shape.of(4, 1, 1, 8).withKvHeads(2)));
        });
        assertThrows(IllegalArgumentException.class, () -> Shape.of(4, 1, 1, 8).withKvHeads(3));
        assertThrows(IllegalArgumentException.class, () -> Shape.of(4, 1, 8, 8).withCacheKeys(7));
        assertThrows(IllegalArgumentException.class, () -> AttentionF16.DecodeBatch.of(32, 2, 128));   // 16 rows per KV head
        assertThrows(IllegalArgumentException.class, () -> new AttentionF16.DecodeBatch(8, 3, 1, 64, 0.1f, true));
    }

    void eachOutput(java.util.function.Consumer<AttentionF16> body) {
        for (var out : Precision.values()) try (var a = AttentionF16.create(device, out)) { body.accept(a); }
    }

    @Test void matchesTheReferenceWithinTheComposedBound() {
        eachOutput(a -> {
            for (var s : new Shape[] {Shape.of(4, 64, 64, 64), Shape.of(3, 37, 50, 40), Shape.of(2, 1, 300, 64), Shape.of(8, 128, 128, 128)}) {
                double worst = check(a, s, 1, 41);
                System.out.printf("attention %s %s %dx%dx%dx%d worst error/bound %.3g%n", a.algorithm(), a.output(), s.heads(), s.queries(), s.keys(), s.headDim(), worst);
            }
        });
    }

    @Test void causalMask() {
        eachOutput(a -> {
            check(a, Shape.of(4, 64, 64, 64).asCausal(), 1, 42);
            check(a, Shape.of(3, 37, 37, 40).asCausal(), 1, 43);
            check(a, Shape.of(2, 16, 300, 64).asCausal(), 1, 44);   // 16 new queries on a cache of 300 keys
            check(a, Shape.of(2, 1, 300, 64).asCausal(), 1, 45);    // one decode step sees everything
        });
    }

    @Test void recordedAttentionRetainsItsPageTable() {
        var shape = Shape.of(1, 1, 1, 2);
        try (var q = buffer(shape.qBytes()); var first = buffer(8); var second = buffer(8);
             var a = AttentionF16.create(device, Precision.FP32); var e = device.encoder()) {
            try (var cache = PagedKvCache.create(device, new PagedKvCache.Config(1, 1, 2, 32, 2));
                 var seq = cache.newSequence()) {
                e.fill(q, 0).fill(cache.keys(0), 0);
                short[] values = new short[128];
                java.util.Arrays.fill(values, 0, 64, Float.floatToFloat16(1f));
                java.util.Arrays.fill(values, 64, 128, Float.floatToFloat16(2f));
                e.upload(cache.values(0), bytes(values));
                cache.reserve(e, seq, 1);
                a.encode(e, q, cache, 0, seq, first, shape);
                cache.truncate(seq, 0);
                cache.reserve(e, seq, 1);
                a.encode(e, q, cache, 0, seq, second, shape);
            }
            var old = e.download(first);
            var current = e.download(second);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
            for (int i = 0; i < 2; i++) {
                assertEquals(1f, ByteBuffer.wrap(old.bytes()).order(ByteOrder.nativeOrder()).getFloat(4 * i));
                assertEquals(2f, ByteBuffer.wrap(current.bytes()).order(ByteOrder.nativeOrder()).getFloat(4 * i));
            }
        }
    }

    @Test void sharpSoftmax() { eachOutput(a -> check(a, Shape.of(2, 64, 96, 64), 4, 46)); }

    @Test void rejectsBadShapes() {
        assertThrows(IllegalArgumentException.class, () -> Shape.of(0, 1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> Shape.of(1, 10, 5, 8).asCausal());
        assertThrows(IllegalArgumentException.class, () -> new Shape(1, 1, 1, 1, Float.NaN, false));
    }

    /** -Dgemm.benchmark=true: attention throughput (the two products' FLOPs) by GPU time. */
    @Test void benchmark() {
        Assumptions.assumeTrue(Boolean.getBoolean("gemm.benchmark"), "set -Dgemm.benchmark=true");
        try (var a = AttentionF16.create(device, Precision.FP16)) {
            for (var s : new Shape[] {Shape.of(32, 512, 512, 128).asCausal(), Shape.of(32, 1024, 1024, 128).asCausal(), Shape.of(32, 2048, 2048, 128).asCausal()}) {
                try (var gq = buffer(s.qBytes()); var gk = buffer(s.kBytes()); var gv = buffer(s.vBytes()); var go = buffer(s.oBytes(Precision.FP16));
                     var gs = buffer(s.scoresBytes()); var gp = buffer(s.probsBytes())) {
                    try (var init = device.encoder()) {
                        init.fill(gq, 0x3800b800).fill(gk, 0x34003800).fill(gv, 0xbc003c00);
                        try (var sub = device.queue().submit(init)) { assertTrue(sub.await(WAIT)); }
                    }
                    double best = time(e -> a.encode(e, gq, gk, gv, go, gs, gp, s));
                    double fusedBest = time(e -> a.encode(e, gq, gk, gv, go, s));
                    System.out.printf("attention %s %s heads=%d seq=%d dim=%d causal: %.2f ms, %.0f GFLOP/s (dense-equivalent); fused %.2f ms, %.0f GFLOP/s%n", device.adapter().name(),
                            a.algorithm(), s.heads(), s.queries(), s.headDim(), best * 1e3, s.flops() / best / 1e9, fusedBest * 1e3, s.flops() / fusedBest / 1e9);
                }
            }
            // training: the forward pass with its log-sum-exp, and the backward pass (1.12.0)
            for (var s : new Shape[] {Shape.of(32, 512, 512, 128).asCausal(), Shape.of(32, 1024, 1024, 128).asCausal(), Shape.of(32, 2048, 2048, 128).asCausal()}) {
                long qn = (long) s.heads() * s.queries() * s.headDim();
                try (var gq = buffer(s.qBytes()); var gk = buffer(s.kBytes()); var gv = buffer(s.vBytes()); var go = buffer(s.oBytes(Precision.FP16));
                     var gl = buffer(4L * s.heads() * s.queries()); var gdo = buffer(4 * qn); var gdq = buffer(4 * qn); var gdk = buffer(4 * qn); var gdv = buffer(4 * qn)) {
                    try (var init = device.encoder()) {
                        init.fill(gq, 0x3800b800).fill(gk, 0x34003800).fill(gv, 0xbc003c00).fill(gdo, 0x3c23d70a);
                        try (var sub = device.queue().submit(init)) { assertTrue(sub.await(WAIT)); }
                    }
                    double fwd = time(e -> a.encode(e, gq, gk, gv, go, gl, s));
                    double bwd = time(e -> a.backward(e, gq, gk, gv, go, gl, gdo, gdq, gdk, gdv, s));
                    System.out.printf("attention %s training heads=%d seq=%d dim=%d causal: forward with lse %.2f ms, backward %.2f ms (%.0f GFLOP/s dense-equivalent at 2.5x the forward's)%n",
                            device.adapter().name(), s.heads(), s.queries(), s.headDim(), fwd * 1e3, bwd * 1e3, 2.5 * s.flops() / bwd / 1e9);
                    var stages = new java.util.TreeMap<String, Double>();
                    try (var e = device.encoder()) {
                        e.profile();
                        a.backward(e, gq, gk, gv, go, gl, gdo, gdq, gdk, gdv, s);
                        try (var sub = device.queue().submit(e)) {
                            assertTrue(sub.await(WAIT));
                            for (var d : sub.dispatchTimings()) stages.merge(d.label(), d.time().toNanos() / 1e6, Double::sum);
                        }
                    }
                    System.out.println("attention backward stages (ms) seq=" + s.queries() + ": " + stages);
                }
            }
            // decoding: one query per head against a long cache, grouped-query (32 heads, 8 KV heads)
            for (int keys : new int[] {512, 4096, 32768}) {
                var s = new Shape(32, 1, keys, 128, (float) (1 / Math.sqrt(128)), true, 8, keys);
                try (var gq = buffer(s.qBytes()); var gk = buffer(s.kBytes()); var gv = buffer(s.vBytes()); var go = buffer(s.oBytes(Precision.FP16))) {
                    try (var init = device.encoder()) {
                        init.fill(gq, 0x3800b800).fill(gk, 0x34003800).fill(gv, 0xbc003c00);
                        try (var sub = device.queue().submit(init)) { assertTrue(sub.await(WAIT)); }
                    }
                    double t = time(e -> a.encode(e, gq, gk, gv, go, s));
                    System.out.printf("attention %s decode heads=32/8 keys=%d dim=128: fused %.1f us, %.0f GB/s of K and V (%s)%n", device.adapter().name(),
                            keys, t * 1e6, (s.kBytes() + s.vBytes()) / t / 1e9, dispatchDescription(a, s));
                    var stages = new java.util.TreeMap<String, Double>();
                    try (var e = device.encoder()) {
                        e.profile();
                        for (int i = 0; i < 20; i++) a.encode(e, gq, gk, gv, go, s);
                        try (var sub = device.queue().submit(e)) {
                            assertTrue(sub.await(WAIT));
                            for (var d : sub.dispatchTimings()) stages.merge(d.label(), d.time().toNanos() / 1e3 / 20, Double::sum);
                        }
                    }
                    System.out.println("attention decode stages keys=" + keys + ": " + stages);
                }
            }
        }
    }

    /** Padded causal prefill. Compare the two public paths on the same inputs and GPU-resident lengths. */
    @Test void benchmarkMaskedPrefill() {
        Assumptions.assumeTrue(Boolean.getBoolean("attentionMask.benchmark"), "set -DattentionMask.benchmark=true");
        var s = Shape.of(16, 512, 512, 128).asCausal();
        var r = new SplittableRandom(1141);
        try (var a = AttentionF16.create(device, Precision.FP16); var composed = MaskedComposition.create(device, Precision.FP16);
             var q = buffer(s.qBytes()); var k = buffer(s.kBytes());
             var v = buffer(s.vBytes()); var o = buffer(s.oBytes(Precision.FP16)); var lengths = buffer(16);
             var scores = buffer(s.scoresBytes()); var probs = buffer(s.probsBytes()); var init = device.encoder()) {
            init.upload(q, bytes(halves(q.bytes() / 2, 1, r)));
            init.upload(k, bytes(halves(k.bytes() / 2, 1, r)));
            init.upload(v, bytes(halves(v.bytes() / 2, 1, r)));
            init.upload(lengths, ByteBuffer.allocate(16).order(ByteOrder.nativeOrder())
                    .putInt(512).putInt(384).putInt(128).putInt(0).array());
            try (var sub = device.queue().submit(init)) { assertTrue(sub.await(WAIT)); }
            var timings = GpuBenchmark.rounds(device, List.of(
                    e -> a.encode(e, q, k, v, o, s, lengths, 4),
                    e -> composed.encode(e, q, k, v, o, scores, probs, s, lengths, 4)));
            for (int mode = 0; mode < timings.length; mode++) {
                for (var t : timings[mode]) {
                    System.out.printf(Locale.ROOT, "masked-prefill %s %s gpuUs=%.2f hostEncodeUs=%.2f wallUs=%.2f%n",
                            device.adapter().name(), mode == 0 ? "fused" : "composed", t.gpuMicros(), t.encodeMicros(), t.wallMicros());
                }
            }
        }
    }

    /** Whole submissions containing four layers of paged decode for the same sequence. */
    @Test void benchmarkPagedDecode() {
        Assumptions.assumeTrue(Boolean.getBoolean("gemm.benchmark"), "set -Dgemm.benchmark=true");
        var shape = Shape.of(32, 1, 512, 128).withKvHeads(8);
        var config = new PagedKvCache.Config(4, 8, 128, 32, 16);
        try (var a = AttentionF16.create(device, Precision.FP16); var cache = PagedKvCache.create(device, config);
             var seq = cache.newSequence(); var q = buffer(shape.qBytes()); var o = buffer(shape.oBytes(Precision.FP16));
             var init = device.encoder()) {
            init.fill(q, 0x3800b800);
            cache.reserve(init, seq, shape.keys());
            for (int layer = 0; layer < config.layers(); layer++)
                init.fill(cache.keys(layer), 0x34003800).fill(cache.values(layer), 0xbc003c00);
            try (var sub = device.queue().submit(init)) { assertTrue(sub.await(WAIT)); }
            var timing = GpuBenchmark.measure(device, e -> {
                for (int layer = 0; layer < config.layers(); layer++) a.encode(e, q, cache, layer, seq, o, shape);
            });
            System.out.printf("attention %s four paged decodes, keys=512: encode %.1f us, GPU %.1f us, wall %.1f us%n",
                    device.adapter().name(), timing.encodeMicros(), timing.gpuMicros(), timing.wallMicros());
        }
    }

    /** -DpagedDecode.benchmark=true: compare repeated and rotating K/V pages with identical decode calls. */
    @Test void benchmarkPagedDecodeWorkingSet() {
        Assumptions.assumeTrue(Boolean.getBoolean("pagedDecode.benchmark"), "set -DpagedDecode.benchmark=true");
        var b = AttentionF16.DecodeBatch.of(32, 8, 128);
        try (var attention = AttentionF16.create(device, Precision.FP16)) {
            for (int keys : new int[] {4096, 32768}) {
                for (int batch : new int[] {1, 2}) {
                    long bytesPerLayer = 4L * batch * keys * b.kvHeads() * b.headDim();
                    int layers = Integer.getInteger("pagedDecode.layers",
                            Math.max(2, Math.toIntExact(Math.ceilDiv(320L << 20, bytesPerLayer))));
                    assertTrue(layers >= 2, "pagedDecode.layers must be at least 2");
                    comparePagedDecode(attention, b, batch, keys, layers);
                }
            }
        }
    }

    private void comparePagedDecode(AttentionF16 attention, AttentionF16.DecodeBatch b, int batch, int keys, int layers) {
        int pageSize = 64;
        var config = new PagedKvCache.Config(layers, b.kvHeads(), b.headDim(), pageSize, batch * Math.ceilDiv(keys, pageSize));
        var shape = b.shape(keys);
        var random = new SplittableRandom(20261125);
        try (var cache = PagedKvCache.create(device, config); var q = buffer(b.qBytes(batch));
             var o = buffer(b.oBytes(Precision.FP16, batch))) {
            var seqs = new ArrayList<PagedKvCache.Sequence>();
            try (var init = device.encoder()) {
                for (int i = 0; i < batch; i++) {
                    var seq = cache.newSequence();
                    cache.reserve(init, seq, keys);
                    seqs.add(seq);
                }
                init.upload(q, bytes(halves(q.bytes() / 2, 1, random)));
                for (int layer = 0; layer < layers; layer++) {
                    fillDecodePool(init, cache.keys(layer), random, b.headDim(), 0);
                    fillDecodePool(init, cache.values(layer), random, b.headDim(), layer + 1);
                }
                try (var sub = device.queue().submit(init)) { assertTrue(sub.await(WAIT)); }
            }
            ObjIntConsumer<GpuCommandEncoder> decode = batch == 1
                    ? (e, layer) -> attention.encode(e, q, cache, layer, seqs.getFirst(), o, shape)
                    : (e, layer) -> attention.encodeBatch(e, q, cache, layer, seqs, o, b);

            // Column zero of every value row holds the layer marker. Its weighted mean identifies the bound pool.
            for (int layer = 0; layer < layers; layer++) {
                try (var e = device.encoder()) {
                    decode.accept(e, layer);
                    var read = e.download(o);
                    try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
                    var result = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder()).asShortBuffer();
                    for (int row = 0; row < batch * b.heads(); row++) {
                        double marker = layer + 1;
                        assertEquals(marker, f(result.get(row * b.headDim())), marker * 0.001, "KV layer marker");
                    }
                }
            }

            Consumer<GpuCommandEncoder> reused = e -> {
                for (int layer = 0; layer < layers; layer++) decode.accept(e, 0);
            };
            Consumer<GpuCommandEncoder> rotating = e -> {
                for (int layer = 0; layer < layers; layer++) decode.accept(e, layer);
            };
            var timings = GpuBenchmark.rounds(device, List.of(reused, rotating));
            long bytesPerLayer = 2 * config.poolBytes();
            for (int mode = 0; mode < timings.length; mode++) {
                long footprint = mode == 0 ? bytesPerLayer : bytesPerLayer * layers;
                for (int round = 0; round < timings[mode].length; round++) {
                    var t = timings[mode][round];
                    System.out.printf(Locale.ROOT,
                            "paged-decode %s keys=%d batch=%d calls=%d mode=%s round=%d kvWorkingSetBytes=%d "
                                    + "gpuUs=%.2f hostEncodeUs=%.2f wallUs=%.2f%n",
                            device.adapter().name(), keys, batch, layers, mode == 0 ? "reused" : "rotating", round + 1,
                            footprint, t.gpuMicros(), t.encodeMicros(), t.wallMicros());
                }
            }
            for (var seq : seqs) seq.close();
        }
    }

    /** Tile randomized FP16 values across physically separate pages. Keep setup transfers and readbacks outside timing. */
    private void fillDecodePool(GpuCommandEncoder init, GpuBuffer pool, SplittableRandom random, int dim, float marker) {
        int tileBytes = (int) Math.min(1 << 20, pool.bytes());
        short[] data = halves(tileBytes / 2, 1, random);
        for (int i = 0; i < data.length; i += dim) data[i] = Float.floatToFloat16(marker);
        try (var tile = buffer(tileBytes)) {
            init.upload(tile, bytes(data));
            for (long offset = 0; offset < pool.bytes(); offset += tileBytes) {
                long count = Math.min(tileBytes, pool.bytes() - offset);
                try (var from = tile.slice(0, count); var to = pool.slice(offset, count)) {
                    init.copy(from, to);
                }
            }
        }
    }

    /** Seconds per run of {@code body}: an untimed round, then the best of 4 rounds of at least 20 ms. */
    double time(java.util.function.Consumer<GpuCommandEncoder> body) {
        return GpuBenchmark.measure(device, body).gpuMicros() / 1e6;
    }
}
