package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.junit.jupiter.api.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.EnumSet;
import java.util.SplittableRandom;
import static org.junit.jupiter.api.Assertions.*;

/** GemmF16 on a real device: bit for bit against GemmF16.reference, and within the documented bound of the exact product. */
abstract class GemmF16Conformance {
    protected abstract GpuProvider provider();
    protected GpuDevice device;
    /** The algorithm under test; every test runs once per algorithm the device supports. */
    protected GemmF16 gemm;
    static final Duration WAIT = Duration.ofSeconds(60);

    @BeforeEach void open() {
        device = provider().adapters().getFirst().open();
        System.out.println("gemm device=" + device.adapter().name() + " backend=" + device.adapter().backend());
    }
    @AfterEach void close() { device.close(); }

    interface Body { void run() throws Exception; }
    /** Runs {@code body} with {@link #gemm} set to each algorithm the device supports. */
    void eachAlgorithm(Body body) throws Exception {
        for (var algorithm : GemmF16.Algorithm.values()) {
            if (!GemmF16.supports(device, algorithm)) {
                assertThrows(ComputeException.class, () -> GemmF16.create(device, algorithm), algorithm + " refused");
                continue;
            }
            try (var g = GemmF16.create(device, algorithm)) { gemm = g; body.run(); } finally { gemm = null; }
        }
    }

    GpuBuffer buffer(long bytes) {
        return device.allocate(new BufferDescriptor(bytes, EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE));
    }

    static long pad4(long bytes) { return (bytes + 3) & ~3L; }
    static short[] halves(int count, SplittableRandom r) {
        short[] h = new short[count];
        for (int i = 0; i < count; i++) h[i] = Float.floatToFloat16((float) (r.nextDouble() * 2 - 1));
        return h;
    }
    static byte[] bytes(short[] h) { var b = ByteBuffer.allocate(2 * h.length).order(ByteOrder.nativeOrder()); for (short v : h) b.putShort(v); return b.array(); }
    static byte[] bytes(float[] f) { var b = ByteBuffer.allocate(4 * f.length).order(ByteOrder.nativeOrder()); for (float v : f) b.putFloat(v); return b.array(); }

    /** Multiplies on the GPU and checks every element of C against the bound; returns the largest error / bound. */
    double check(GemmF16.Shape s, long seed, boolean nanC) {
        var r = new SplittableRandom(seed);
        // transfers move whole 32-bit words, so the half arrays are padded to an even length
        short[] a = halves((int) (pad4(s.aBytes()) / 2), r), b = halves((int) (pad4(s.bBytes()) / 2), r);
        float[] c0 = new float[(int) (s.cBytes() / 4)];
        for (int i = 0; i < c0.length; i++) c0[i] = nanC ? Float.NaN : (float) (r.nextDouble() * 2 - 1);
        float[] out;
        try (var ga = buffer(pad4(s.aBytes())); var gb = buffer(pad4(s.bBytes())); var gc = buffer(s.cBytes()); var e = device.encoder()) {
            e.upload(ga, bytes(a)).upload(gb, bytes(b)).upload(gc, bytes(c0));
            gemm.encode(e, ga, gb, gc, s); var read = e.download(gc);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            var bb = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder()); out = new float[c0.length];
            for (int i = 0; i < out.length; i++) out[i] = bb.getFloat(4 * i);
        }
        float[] exact = c0.clone(); GemmF16.reference(a, b, exact, s);
        if (gemm.algorithm() == GemmF16.Algorithm.TILED) {
            for (int i = 0; i < out.length; i++)
                assertEquals(Float.floatToRawIntBits(exact[i]), Float.floatToRawIntBits(out[i]), "C[" + i + "] differs from GemmF16.reference for " + s);
        } else {
            int differ = 0; for (int i = 0; i < out.length; i++) if (Float.floatToRawIntBits(exact[i]) != Float.floatToRawIntBits(out[i])) differ++;
            System.out.println("gemm " + gemm.algorithm() + " " + s.m() + "x" + s.n() + "x" + s.k() + " differs from reference in " + differ + " of " + out.length);
        }
        double worst = 0;
        for (int i = 0; i < s.m(); i++) {
            for (int j = 0; j < s.n(); j++) {
                double sum = 0, abs = 0;
                for (int q = 0; q < s.k(); q++) {
                    double p = (double) Float.float16ToFloat(a[i * s.lda() + q]) * Float.float16ToFloat(b[q * s.ldb() + j]);
                    sum += p; abs += Math.abs(p);
                }
                int idx = i * s.ldc() + j; double cOld = s.beta() == 0 ? 0 : c0[idx];
                double ref = s.alpha() * sum + s.beta() * cOld;
                double bound = (s.k() + 2.0) * 0x1p-24 * (Math.abs(s.alpha()) * abs + Math.abs(s.beta() * cOld)) + Float.MIN_NORMAL;
                double err = Math.abs(out[idx] - ref);
                assertTrue(err <= bound, "C[" + i + "][" + j + "] = " + out[idx] + ", reference " + ref + ", bound " + bound + " for " + s);
                worst = Math.max(worst, err / bound);
            }
        }
        // elements between rows (padding) are untouched
        for (int i = 0; i < s.m(); i++) for (int j = s.n(); j < s.ldc() && i * s.ldc() + j < out.length; j++)
            assertEquals(Float.floatToRawIntBits(c0[i * s.ldc() + j]), Float.floatToRawIntBits(out[i * s.ldc() + j]), "padding written");
        return worst;
    }

    @Test void matchesTheReferenceWithinTheBound() throws Exception {
        int[][] shapes = {{1, 1, 1}, {64, 64, 16}, {64, 64, 64}, {65, 63, 17}, {1, 100, 300}, {130, 1, 77}, {100, 37, 513}, {128, 256, 1024}};
        eachAlgorithm(() -> {
            long seed = 1;
            for (int[] d : shapes) {
                double worst = check(GemmF16.Shape.of(d[0], d[1], d[2], 1f, 0f), seed++, false);
                System.out.printf("gemm %s %dx%dx%d worst error/bound %.3g%n", gemm.algorithm(), d[0], d[1], d[2], worst);
            }
        });
    }

    @Test void appliesAlphaAndBetaAndStrides() throws Exception {
        eachAlgorithm(() -> {
            check(new GemmF16.Shape(70, 50, 90, 96, 64, 72, -0.75f, 0.5f), 11, false);
            check(new GemmF16.Shape(33, 129, 40, 40, 130, 131, 2f, 1f), 12, false);
        });
    }

    @Test void betaZeroIgnoresC() throws Exception { eachAlgorithm(() -> check(GemmF16.Shape.of(65, 66, 67, 1.5f, 0f), 13, true)); }

    @Test void rejectsUndersizedBuffers() {
        var s = GemmF16.Shape.of(8, 8, 8, 1, 0);
        try (var gemm = GemmF16.create(device); var small = buffer(64); var ok = buffer(256); var e = device.encoder()) {
            assertThrows(IllegalArgumentException.class, () -> gemm.encode(e, small, ok, ok, s));
            assertThrows(IllegalArgumentException.class, () -> gemm.encode(e, ok, small, ok, s));
            assertThrows(IllegalArgumentException.class, () -> gemm.encode(e, ok, ok, small, s));
        }
        assertThrows(IllegalArgumentException.class, () -> GemmF16.Shape.of(0, 1, 1, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new GemmF16.Shape(2, 2, 4, 3, 2, 2, 1, 0));
    }

    /** -Dgemm.benchmark=true: GFLOP/s for square multiplies over back-to-back dispatches, by GPU time and by wall time. */
    @Test void benchmark() throws Exception {
        Assumptions.assumeTrue(Boolean.getBoolean("gemm.benchmark"), "set -Dgemm.benchmark=true");
        eachAlgorithm(this::benchmarkOne);
    }

    private void benchmarkOne() {
        for (int size : new int[] {512, 1024, 2048, 4096}) {
            var s = GemmF16.Shape.of(size, size, size, 1, 0);
            try (var ga = buffer(s.aBytes()); var gb = buffer(s.bBytes()); var gc = buffer(s.cBytes())) {
                int reps = Math.max(2, (int) (2e11 / s.flops()));
                double best = Double.MAX_VALUE, bestGpu = Double.MAX_VALUE;
                for (int round = 0; round < 4; round++) {
                    long t0 = System.nanoTime();
                    try (var e = device.encoder()) {
                        for (int i = 0; i < reps; i++) gemm.encode(e, ga, gb, gc, s);
                        try (var sub = device.queue().submit(e)) {
                            assertTrue(sub.await(WAIT));
                            var gpu = sub.gpuTime(); if (gpu.isPresent()) bestGpu = Math.min(bestGpu, gpu.get().toNanos() / 1e9);
                        }
                    }
                    best = Math.min(best, (System.nanoTime() - t0) / 1e9);
                }
                System.out.printf("gemm %s %s %d^3: %.0f GFLOP/s by GPU time, %.0f by wall time (%d per submission)%n", device.adapter().name(),
                        gemm.algorithm(), size, reps * s.flops() / bestGpu / 1e9, reps * s.flops() / best / 1e9, reps);
            }
        }
    }
}
