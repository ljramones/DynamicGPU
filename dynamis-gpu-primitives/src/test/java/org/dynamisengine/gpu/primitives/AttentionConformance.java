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
                    double best = Double.MAX_VALUE;
                    for (int round = 0; round < 4; round++) {
                        try (var e = device.encoder()) {
                            for (int i = 0; i < 4; i++) a.encode(e, gq, gk, gv, go, gs, gp, s);
                            try (var sub = device.queue().submit(e)) {
                                assertTrue(sub.await(WAIT));
                                var t = sub.gpuTime(); if (t.isPresent()) best = Math.min(best, t.get().toNanos() / 4e9);
                            }
                        }
                    }
                    System.out.printf("attention %s %s heads=%d seq=%d dim=%d causal: %.2f ms, %.0f GFLOP/s (dense-equivalent)%n", device.adapter().name(),
                            a.algorithm(), s.heads(), s.queries(), s.headDim(), best * 1e3, s.flops() / best / 1e9);
                }
            }
        }
    }
}
