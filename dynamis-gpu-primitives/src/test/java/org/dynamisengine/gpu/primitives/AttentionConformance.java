package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.primitives.AttentionF16.Shape;
import org.junit.jupiter.api.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.EnumSet;
import java.util.SplittableRandom;
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

    /** Runs the attention and checks every element of O; returns the worst error / bound. */
    double check(AttentionF16 attention, Shape s, double spread, long seed) {
        var r = new SplittableRandom(seed);
        int d = s.headDim(), nq = s.queries(), nk = s.keys();
        short[] q = halves(s.qBytes() / 2, spread, r), k = halves(s.kBytes() / 2, spread, r), v = halves(s.vBytes() / 2, 1, r);
        var out = attention.output(); boolean fp32 = out == Precision.FP32;
        ByteBuffer got;
        try (var gq = buffer(s.qBytes()); var gk = buffer(s.kBytes()); var gv = buffer(s.vBytes()); var go = buffer(s.oBytes(out));
             var gs = buffer(s.scoresBytes()); var gp = buffer(s.probsBytes()); var e = device.encoder()) {
            e.upload(gq, bytes(q)).upload(gk, bytes(k)).upload(gv, bytes(v));
            attention.encode(e, gq, gk, gv, go, gs, gp, s); var read = e.download(go);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            got = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
        }
        double worst = 0, scale = s.scale();
        double[] sc = new double[nk], dS = new double[nk], p = new double[nk], dP = new double[nk];
        for (int h = 0; h < s.heads(); h++) {
            for (int i = 0; i < nq; i++) {
                int qb = (h * nq + i) * d, limit = s.causal() ? i + nk - nq + 1 : nk;
                double mx = Double.NEGATIVE_INFINITY, maxdS = 0;
                for (int j = 0; j < limit; j++) {
                    int kb = (h * nk + j) * d; double sum = 0, abs = 0;
                    for (int t = 0; t < d; t++) { double pr = f(q[qb + t]) * f(k[kb + t]); sum += pr; abs += Math.abs(pr); }
                    sc[j] = scale * sum; dS[j] = (d + 2) * U * Math.abs(scale) * abs; // GemmF16's bound
                    mx = Math.max(mx, sc[j]); maxdS = Math.max(maxdS, dS[j]);
                }
                double total = 0, weighted = 0;
                for (int j = 0; j < limit; j++) { p[j] = Math.exp(sc[j] - mx); total += p[j]; weighted += p[j] * (mx - sc[j]); }
                double xbar = weighted / total;
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
        var r = new SplittableRandom(seed);
        int d = s.headDim(), nq = s.queries(), nk = s.keys(), cache = s.cacheKeys(), hpk = s.heads() / s.kvHeads();
        short[] q = halves(s.qBytes() / 2, spread, r), k = halves(s.kBytes() / 2, spread, r), v = halves(s.vBytes() / 2, 1, r);
        for (int h = 0; h < s.kvHeads(); h++) for (int j = nk; j < cache; j++) for (int t = 0; t < d; t++) {
            k[(h * cache + j) * d + t] = (short) 0x7E00; v[(h * cache + j) * d + t] = (short) 0x7E00;
        }
        var out = attention.output(); boolean fp32 = out == Precision.FP32;
        ByteBuffer got;
        try (var gq = buffer(s.qBytes()); var gk = buffer(s.kBytes()); var gv = buffer(s.vBytes()); var go = buffer(s.oBytes(out)); var e = device.encoder()) {
            e.upload(gq, bytes(q)).upload(gk, bytes(k)).upload(gv, bytes(v));
            attention.encode(e, gq, gk, gv, go, s); var read = e.download(go);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            got = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
        }
        boolean halfWeights = attention.fusedKernel(s).equals("flash.simdgroup");
        double worst = 0, scale = s.scale();
        int blocks = Math.ceilDiv(nk, AttentionF16.keyBlock(AttentionF16.variant(d))) + 2;
        double expErr = 8 * U;   // 4 ulp
        double[] sc = new double[nk], p = new double[nk], dP = new double[nk], eps = new double[nk];
        for (int h = 0; h < s.heads(); h++) {
            int kvh = h / hpk;
            for (int i = 0; i < nq; i++) {
                int qb = (h * nq + i) * d, limit = s.causal() ? i + nk - nq + 1 : nk;
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
                    int idx = (h * nq + i) * d + c;
                    double gv = fp32 ? got.getFloat(4 * idx) : Float.float16ToFloat(got.getShort(2 * idx));
                    double err2 = Math.abs(gv - o);
                    assertTrue(err2 <= bound, "fused O[" + h + "][" + i + "][" + c + "] = " + gv + ", reference " + o + ", bound " + bound + " for " + s + " " + out);
                    worst = Math.max(worst, err2 / bound);
                }
            }
        }
        return worst;
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
                    Shape.of(6, 1, 129, 128).withKvHeads(2),                                // decode, 3 rows per KV head
                    Shape.of(2, 3, 5000, 256).withKvHeads(1).asCausal().withCacheKeys(6000), // decode at DMAX 256
                    Shape.of(4, 9, 40, 64).withKvHeads(4)}) {
                assertTrue(a.fusedSupports(s), s.toString());
                double worst = checkFused(a, s, 1, 47);
                System.out.printf("attention fused %s %s worst error/bound %.3g (%s, splits %d)%n", a.output(), s, worst, a.fusedKernel(s), AttentionF16.split(s)[0]);
            }
            checkFused(a, Shape.of(2, 64, 96, 64), 4, 48);   // sharp softmax
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
        int layers = 2, kvh = 2, d = 64, ps = 32;
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

    @Test void fusedRefusesWhatItCannotTake() {
        eachOutput(a -> {
            assertFalse(a.fusedSupports(Shape.of(1, 1, 1, 7)));
            assertFalse(a.fusedSupports(Shape.of(1, 1, 1, 258)));
            assertThrows(ComputeException.class, () -> a.encode(null, null, null, null, null, Shape.of(1, 1, 1, 7)));
            assertThrows(IllegalArgumentException.class, () -> a.encode(null, null, null, null, null, null, null, Shape.of(4, 1, 1, 8).withKvHeads(2)));
        });
        assertThrows(IllegalArgumentException.class, () -> Shape.of(4, 1, 1, 8).withKvHeads(3));
        assertThrows(IllegalArgumentException.class, () -> Shape.of(4, 1, 8, 8).withCacheKeys(7));
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
            // decoding: one query per head against a long cache, grouped-query (32 heads, 8 KV heads)
            for (int keys : new int[] {512, 4096, 32768}) {
                var s = new Shape(32, 1, keys, 128, (float) (1 / Math.sqrt(128)), true, 8, keys);
                try (var gq = buffer(s.qBytes()); var gk = buffer(s.kBytes()); var gv = buffer(s.vBytes()); var go = buffer(s.oBytes(Precision.FP16))) {
                    try (var init = device.encoder()) {
                        init.fill(gq, 0x3800b800).fill(gk, 0x34003800).fill(gv, 0xbc003c00);
                        try (var sub = device.queue().submit(init)) { assertTrue(sub.await(WAIT)); }
                    }
                    double t = time(e -> a.encode(e, gq, gk, gv, go, s));
                    System.out.printf("attention %s decode heads=32/8 keys=%d dim=128: fused %.1f us, %.0f GB/s of K and V (splits %d)%n", device.adapter().name(),
                            keys, t * 1e6, (s.kBytes() + s.vBytes()) / t / 1e9, AttentionF16.split(s)[0]);
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

    /** Seconds per run of {@code body}: an untimed round, then the best of 4 rounds of at least 20 ms. */
    double time(java.util.function.Consumer<GpuCommandEncoder> body) {
        return GpuBenchmark.measure(device, body).gpuMicros() / 1e6;
    }
}
