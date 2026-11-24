package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.primitives.GemmF16.Algorithm;
import org.dynamisengine.gpu.primitives.GemmF16.Output;
import org.dynamisengine.gpu.primitives.GemmF16.Shape;
import org.junit.jupiter.api.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.EnumSet;
import java.util.SplittableRandom;
import static org.junit.jupiter.api.Assertions.*;

/**
 * GemmF16 on a real device, for every algorithm the device supports and both output types: the tiled kernel bit for
 * bit against GemmF16.reference (every element of C, so padding and gaps between batch matrices must be untouched),
 * and every kernel within the documented bound of the exact product.
 */
abstract class GemmF16Conformance {
    protected abstract GpuProvider provider();
    protected GpuDevice device;
    /** The kernel under test; every test runs once per algorithm the device supports and output type. */
    protected GemmF16 gemm;
    static final Duration WAIT = Duration.ofSeconds(60);

    @BeforeEach void open() {
        device = provider().adapters().getFirst().open();
        System.out.println("gemm device=" + device.adapter().name() + " backend=" + device.adapter().backend());
    }
    @AfterEach void close() { device.close(); }

    interface Body { void run() throws Exception; }
    /** Runs {@code body} with {@link #gemm} set to each supported algorithm and each output in {@code outputs}. */
    void eachKernel(Body body, Output... outputs) throws Exception {
        for (var algorithm : Algorithm.values()) {
            for (var output : outputs) {
                if (!GemmF16.supports(device, algorithm)) {
                    assertThrows(ComputeException.class, () -> GemmF16.create(device, algorithm, output), algorithm + " refused");
                    continue;
                }
                try (var g = GemmF16.create(device, algorithm, output)) { gemm = g; body.run(); } finally { gemm = null; }
            }
        }
    }
    void eachKernel(Body body) throws Exception { eachKernel(body, Output.values()); }

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

    /** Multiplies on the GPU and checks every element of C; returns the largest error / bound. */
    double check(Shape s, long seed, boolean nanC) {
        var r = new SplittableRandom(seed);
        var output = gemm.output();
        // transfers move whole 32-bit words, so the half arrays are padded to an even length
        short[] a = halves((int) (pad4(s.aBytes()) / 2), r), b = halves((int) (pad4(s.bBytes()) / 2), r);
        int cLength = (int) (pad4(s.cBytes(output)) / output.bytes());
        float[] cF = new float[cLength]; short[] cH = new short[cLength];
        for (int i = 0; i < cLength; i++) {
            float v = nanC ? Float.NaN : (float) (r.nextDouble() * 2 - 1);
            cF[i] = v; cH[i] = Float.floatToFloat16(v);
        }
        boolean fp32 = output == Output.FP32;
        ByteBuffer got;
        try (var ga = buffer(pad4(s.aBytes())); var gb = buffer(pad4(s.bBytes())); var gc = buffer(pad4(s.cBytes(output))); var e = device.encoder()) {
            e.upload(ga, bytes(a)).upload(gb, bytes(b)).upload(gc, fp32 ? bytes(cF) : bytes(cH));
            gemm.encode(e, ga, gb, gc, s); var read = e.download(gc);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            got = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
        }
        // bit for bit against the reference, every element (untouched elements must keep their initial bits)
        int differ = 0;
        if (fp32) {
            float[] exact = cF.clone(); GemmF16.reference(a, b, exact, s);
            for (int i = 0; i < cLength; i++) if (Float.floatToRawIntBits(exact[i]) != Float.floatToRawIntBits(got.getFloat(4 * i))) differ++;
        } else {
            short[] exact = cH.clone(); GemmF16.referenceF16(a, b, exact, s);
            for (int i = 0; i < cLength; i++) if (exact[i] != got.getShort(2 * i)) differ++;
        }
        String what = gemm.algorithm() + " " + output + " " + s;
        if (gemm.algorithm() == Algorithm.TILED) assertEquals(0, differ, "elements differing from the reference for " + what);
        else System.out.println("gemm " + gemm.algorithm() + " " + output + " " + s.m() + "x" + s.n() + "x" + s.k() + "x" + s.batch()
                + (s.transposeB() ? " B^T" : "") + " differs from reference in " + differ + " of " + cLength);
        double worst = 0;
        for (int z = 0; z < s.batch(); z++) {
            for (int i = 0; i < s.m(); i++) {
                for (int j = 0; j < s.n(); j++) {
                    double sum = 0, abs = 0;
                    for (int q = 0; q < s.k(); q++) {
                        double p = (double) Float.float16ToFloat(a[(int) (z * s.strideA() + (long) i * s.lda() + q)])
                                * Float.float16ToFloat(b[(int) s.bIndex(z, q, j)]);
                        sum += p; abs += Math.abs(p);
                    }
                    int idx = (int) (z * s.strideC() + (long) i * s.ldc() + j);
                    double cOld = s.beta() == 0 ? 0 : fp32 ? cF[idx] : Float.float16ToFloat(cH[idx]);
                    double ref = s.alpha() * sum + s.beta() * cOld;
                    double bound = (s.k() + 2.0) * 0x1p-24 * (Math.abs(s.alpha()) * abs + Math.abs(s.beta() * cOld)) + Float.MIN_NORMAL;
                    if (!fp32) bound += Math.abs(ref) * 0x1p-11 + 0x1p-25; // one rounding to half, subnormals included
                    double v = fp32 ? got.getFloat(4 * idx) : Float.float16ToFloat(got.getShort(2 * idx));
                    double err = Math.abs(v - ref);
                    assertTrue(err <= bound, "C[" + z + "][" + i + "][" + j + "] = " + v + ", reference " + ref + ", bound " + bound + " for " + what);
                    worst = Math.max(worst, err / bound);
                }
            }
        }
        return worst;
    }

    @Test void matchesTheReferenceWithinTheBound() throws Exception {
        int[][] shapes = {{1, 1, 1}, {64, 64, 16}, {64, 64, 64}, {65, 63, 17}, {1, 100, 300}, {130, 1, 77}, {100, 37, 513}, {128, 256, 1024}};
        eachKernel(() -> {
            long seed = 1;
            for (int[] d : shapes) {
                double worst = check(Shape.of(d[0], d[1], d[2], 1f, 0f), seed++, false);
                System.out.printf("gemm %s %s %dx%dx%d worst error/bound %.3g%n", gemm.algorithm(), gemm.output(), d[0], d[1], d[2], worst);
            }
        });
    }

    @Test void appliesAlphaAndBetaAndStrides() throws Exception {
        eachKernel(() -> {
            check(new Shape(70, 50, 90, 96, 64, 72, -0.75f, 0.5f), 11, false);
            check(new Shape(33, 129, 40, 40, 130, 131, 2f, 1f), 12, false);
        });
    }

    @Test void betaZeroIgnoresC() throws Exception { eachKernel(() -> check(Shape.of(65, 66, 67, 1.5f, 0f), 13, true)); }

    /** B stored n x k (the weights of a linear layer): interior tiles, edges, and a padded ldb. */
    @Test void transposedB() throws Exception {
        eachKernel(() -> {
            check(Shape.ofTransposedB(128, 128, 64, 1f, 0f), 21, false);
            check(Shape.ofTransposedB(65, 63, 17, 1f, 0f), 22, false);
            check(Shape.ofTransposedB(64, 192, 520, 0.5f, 0f), 23, false);
            check(new Shape(70, 50, 90, 96, 100, 72, -0.75f, 0.5f, true, 1, 0, 0, 0), 24, false);
        });
    }

    /** A batch: dense, with B shared across the batch (stride 0), and with gaps between C matrices. */
    @Test void batches() throws Exception {
        eachKernel(() -> {
            check(Shape.of(64, 64, 32, 1f, 0f).batched(5), 31, false);
            check(Shape.ofTransposedB(65, 40, 48, 1f, 0f).withBatch(3, 65 * 48, 0, 65 * 40 + 7), 32, false);
            check(new Shape(33, 70, 24, 24, 72, 70, 2f, 0.5f).withBatch(4, 33 * 24 + 5, 24 * 72, 33 * 70 + 3), 33, false);
        });
    }

    @Test void rejectsUndersizedBuffersAndBadShapes() {
        var s = Shape.of(8, 8, 8, 1, 0);
        try (var gemm = GemmF16.create(device); var gemm16 = GemmF16.create(device, Output.FP16);
             var small = buffer(64); var a = buffer(256); var b = buffer(256); var c = buffer(256); var e = device.encoder()) {
            assertThrows(IllegalArgumentException.class, () -> gemm.encode(e, small, b, c, s));
            assertThrows(IllegalArgumentException.class, () -> gemm.encode(e, a, small, c, s));
            assertThrows(IllegalArgumentException.class, () -> gemm.encode(e, a, b, small, s));
            gemm16.encode(e, a, b, c, Shape.of(8, 8, 8, 1, 0).batched(2)); // FP16 C: 2 x 64 halves fit in 256 bytes
            assertThrows(IllegalArgumentException.class, () -> gemm.encode(e, a, b, c, Shape.of(8, 8, 8, 1, 0).batched(2)));
        }
        assertThrows(IllegalArgumentException.class, () -> Shape.of(0, 1, 1, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new Shape(2, 2, 4, 3, 2, 2, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new Shape(2, 2, 4, 4, 3, 2, 1, 0, true, 1, 0, 0, 0)); // ldb < k
        assertThrows(IllegalArgumentException.class, () -> Shape.of(4, 4, 4, 1, 0).withBatch(2, 16, 16, 8)); // C overlaps
        assertThrows(IllegalArgumentException.class, () -> Shape.of(4, 4, 4, 1, 0).withBatch(0, 16, 16, 16));
    }

    /** -Dgemm.benchmark=true: GFLOP/s for square multiplies over back-to-back dispatches, by GPU time and by wall time. */
    @Test void benchmark() throws Exception {
        Assumptions.assumeTrue(Boolean.getBoolean("gemm.benchmark"), "set -Dgemm.benchmark=true");
        eachKernel(this::benchmarkOne, Output.FP32);
    }

    private void benchmarkOne() {
        for (int size : new int[] {512, 1024, 2048, 4096}) {
            for (var s : new Shape[] {Shape.of(size, size, size, 1, 0), Shape.ofTransposedB(size, size, size, 1, 0)}) {
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
                    System.out.printf("gemm %s %s %s %d^3: %.0f GFLOP/s by GPU time, %.0f by wall time (%d per submission)%n", device.adapter().name(),
                            gemm.algorithm(), s.transposeB() ? "B^T" : "B", size, reps * s.flops() / bestGpu / 1e9, reps * s.flops() / best / 1e9, reps);
                }
            }
        }
    }
}
