package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.primitives.GemmF16.Algorithm;
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
    void eachKernel(Body body, Precision... outputs) throws Exception {
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
    void eachKernel(Body body) throws Exception { eachKernel(body, Precision.values()); }

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

    /** Whether the kernel under test refuses {@code s}: SKINNY with m > 16 or A transposed. */
    boolean refuses(Shape s) {
        return gemm.algorithm(s) == Algorithm.SKINNY && (s.m() > GemmF16.SKINNY_MAX_M || s.transposeA());
    }

    /** Multiplies on the GPU and checks every element of C; returns the largest error / bound. */
    double check(Shape s, long seed, boolean nanC) {
        if (refuses(s)) {
            try (var ga = buffer(pad4(s.aBytes())); var gb = buffer(pad4(s.bBytes())); var gc = buffer(pad4(s.cBytes(gemm.output()))); var e = device.encoder()) {
                assertThrows(IllegalArgumentException.class, () -> gemm.encode(e, ga, gb, gc, s), "SKINNY refuses m > 16 and A^T");
            }
            return 0;
        }
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
        boolean fp32 = output == Precision.FP32;
        ByteBuffer got;
        try (var ga = buffer(pad4(s.aBytes())); var gb = buffer(pad4(s.bBytes())); var gc = buffer(pad4(s.cBytes(output))); var e = device.encoder()) {
            e.upload(ga, bytes(a)).upload(gb, bytes(b)).upload(gc, fp32 ? bytes(cF) : bytes(cH));
            gemm.encode(e, ga, gb, gc, s); var read = e.download(gc);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            got = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
        }
        // bit for bit against the reference, every element (untouched elements must keep their initial bits)
        int differ = 0;
        boolean skinny = gemm.algorithm(s) == Algorithm.SKINNY;
        if (fp32) {
            float[] exact = cF.clone();
            if (skinny) GemmF16.referenceSkinny(a, b, exact, s); else GemmF16.reference(a, b, exact, s);
            for (int i = 0; i < cLength; i++) if (Float.floatToRawIntBits(exact[i]) != Float.floatToRawIntBits(got.getFloat(4 * i))) differ++;
        } else {
            short[] exact = cH.clone();
            if (skinny) GemmF16.referenceSkinnyF16(a, b, exact, s); else GemmF16.referenceF16(a, b, exact, s);
            for (int i = 0; i < cLength; i++) if (exact[i] != got.getShort(2 * i)) differ++;
        }
        String what = gemm.algorithm(s) + " " + output + " " + s;
        if (gemm.contract().reproducible() || gemm.algorithm(s) != Algorithm.SIMDGROUP_MATRIX)
            assertEquals(0, differ, "elements differing from the reference for " + what);
        else System.out.println("gemm " + gemm.algorithm() + " " + output + " " + s.m() + "x" + s.n() + "x" + s.k() + "x" + s.batch()
                + (s.transposeB() ? " B^T" : "") + " differs from reference in " + differ + " of " + cLength);
        double worst = 0;
        for (int z = 0; z < s.batch(); z++) {
            for (int i = 0; i < s.m(); i++) {
                for (int j = 0; j < s.n(); j++) {
                    double sum = 0, abs = 0;
                    for (int q = 0; q < s.k(); q++) {
                        double p = (double) Float.float16ToFloat(a[(int) s.aIndex(z, i, q)])
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

    /**
     * The skinny shapes (m up to 16) every kernel computes: B either way round, k odd (the tail lane), rows that start
     * on an odd element (read a half at a time), n odd (a half column pair), batches, and beta.
     */
    @Test void skinnyShapes() throws Exception {
        eachKernel(() -> {
            long seed = 100;
            for (int m : new int[] {1, 2, 3, 8, 16}) {
                check(Shape.ofTransposedB(m, 77, 301, 1f, 0f), seed++, false);
                check(Shape.of(m, 131, 257, 1f, 0f), seed++, false);
            }
            check(Shape.ofTransposedB(1, 512, 4096, 1f, 0f), seed++, false);              // the vector build
            check(Shape.ofTransposedB(5, 77, 304, -1f, 0.5f), seed++, false);
            check(Shape.ofTransposedB(3, 40, 48, 1f, 0f).withBatch(3, 3 * 48, 0, 3 * 40 + 7), seed++, false);
            check(Shape.of(1, 512, 1024, 1f, 0f), seed++, false);
            check(new Shape(5, 33, 63, 65, 67, 35, -0.5f, 0.75f, true, 1, 0, 0, 0), seed++, false);   // odd lda and ldb
            check(new Shape(4, 33, 63, 65, 35, 34, 2f, 1f), seed++, false);                         // odd lda, ldb, n
            check(Shape.ofTransposedB(3, 40, 48, 1f, 0f).withBatch(3, 3 * 48 + 1, 0, 3 * 40 + 7), seed++, false);
            check(Shape.of(2, 70, 24, 1f, 0.5f).withBatch(4, 2 * 24 + 1, 24 * 70 + 1, 2 * 70 + 3), seed++, false);
        });
    }

    /** A stored k x m (A^T B, the weight gradient): interior tiles, edges, a padded lda, B either way round, batches. */
    @Test void transposedA() throws Exception {
        eachKernel(() -> {
            check(Shape.ofTransposedA(128, 128, 64, 1f, 0f), 41, false);
            check(Shape.ofTransposedA(65, 63, 17, 1f, 0f), 42, false);
            check(new Shape(70, 50, 90, 72, 50, 50, -0.75f, 0.5f, false, 1, 0, 0, 0, true), 43, false);
            check(new Shape(64, 128, 96, 64, 96, 128, 1f, 0f, true, 1, 0, 0, 0, true), 44, false);   // A^T B^T
            check(Shape.ofTransposedA(40, 33, 24, 1f, 0f).batched(3), 45, false);
        });
    }

    static float[] floats(int count, SplittableRandom r) {
        float[] f = new float[count];
        for (int i = 0; i < count; i++) f[i] = (float) (r.nextDouble() * 2 - 1);
        return f;
    }

    /**
     * Multiplies with {@code epilogue} on the GPU and checks every element of C against the reference: bit for bit for
     * a fixed-order kernel and an exact epilogue, else within the GEMM bound carried through the epilogue plus the
     * device's exp error.
     */
    void checkEpilogue(Shape s, GemmF16.Epilogue ep, long seed) {
        if (refuses(s)) return;
        var r = new SplittableRandom(seed);
        var output = gemm.output();
        boolean fp32 = output == Precision.FP32;
        short[] a = halves((int) (pad4(s.aBytes()) / 2), r), b = halves((int) (pad4(s.bBytes()) / 2), r);
        int cLength = (int) (pad4(s.cBytes(output)) / output.bytes());
        int biasLength = ep.bias() == GemmF16.Epilogue.Bias.ROW ? s.m() : s.n();
        float[] c = floats(cLength, r), bias = floats(biasLength + 1, r), gate = floats(cLength, r), residual = floats(cLength, r);
        if (!fp32) for (float[] f : new float[][] {c, bias, gate, residual}) for (int i = 0; i < f.length; i++) f[i] = Float.float16ToFloat(Float.floatToFloat16(f[i]));
        ByteBuffer got;
        try (var ga = buffer(pad4(s.aBytes())); var gb = buffer(pad4(s.bBytes())); var gc = buffer(pad4(s.cBytes(output)));
             var gbias = buffer(pad4((long) output.bytes() * bias.length)); var ggate = buffer(pad4(s.cBytes(output)));
             var gres = buffer(pad4(s.cBytes(output))); var e = device.encoder()) {
            e.upload(ga, bytes(a)).upload(gb, bytes(b)).upload(gc, encode(c, fp32)).upload(gbias, encode(bias, fp32))
                    .upload(ggate, encode(gate, fp32)).upload(gres, encode(residual, fp32));
            gemm.encode(e, ga, gb, gc, s, ep, ep.bias() == GemmF16.Epilogue.Bias.NONE ? null : gbias, ep.gated() ? ggate : null,
                    ep.residual() ? gres : null);
            var read = e.download(gc);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            got = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
        }
        Algorithm run = gemm.algorithm(s);
        Algorithm order = run == Algorithm.SIMDGROUP_MATRIX ? Algorithm.TILED : run;
        float[] ref = c.clone();
        GemmF16.reference(order, a, b, ref, s, ep, bias, gate, residual);   // in FP32; rounded to half below for FP16 C
        boolean bitwise = run != Algorithm.SIMDGROUP_MATRIX && ep.exact();
        String what = run + " " + output + " " + ep + " " + s;
        int differ = 0;
        for (int i = 0; i < cLength; i++) {
            float v = fp32 ? got.getFloat(4 * i) : Float.float16ToFloat(got.getShort(2 * i));
            float expect = fp32 ? ref[i] : Float.float16ToFloat(Float.floatToFloat16(ref[i]));
            if (Float.floatToRawIntBits(v) != Float.floatToRawIntBits(expect)) differ++;
        }
        if (bitwise) { assertEquals(0, differ, "elements differing from the reference for " + what); return; }
        for (int z = 0; z < s.batch(); z++) {
            for (int i = 0; i < s.m(); i++) {
                for (int j = 0; j < s.n(); j++) {
                    double abs = 0;
                    for (int q = 0; q < s.k(); q++)
                        abs += Math.abs((double) Float.float16ToFloat(a[(int) s.aIndex(z, i, q)]) * Float.float16ToFloat(b[(int) s.bIndex(z, q, j)]));
                    int idx = (int) s.cIndex(z, i, j);
                    // the GEMM bound, carried through bias (1), the activation (slope <= 1.13) and the gate (|gate| <= 1);
                    // plus the device's exp and the epilogue's roundings, relative to the result
                    double gemmBound = (s.k() + 2.0) * 0x1p-24 * (Math.abs(s.alpha()) * abs + Math.abs(s.beta() * c[idx]));
                    double bound = 2 * gemmBound + 0x1p-17 * (Math.abs(ref[idx]) + 1);
                    if (!fp32) bound += Math.abs(ref[idx]) * 0x1p-10 + 0x1p-24;
                    double v = fp32 ? got.getFloat(4 * idx) : Float.float16ToFloat(got.getShort(2 * idx));
                    assertTrue(Math.abs(v - ref[idx]) <= bound, "C[" + z + "][" + i + "][" + j + "] = " + v + ", reference " + ref[idx]
                            + ", bound " + bound + " for " + what);
                }
            }
        }
    }

    static byte[] encode(float[] f, boolean fp32) {
        if (fp32) return bytes(f);
        short[] h = new short[f.length + (f.length & 1)];
        for (int i = 0; i < f.length; i++) h[i] = Float.floatToFloat16(f[i]);
        return bytes(h);
    }

    /** Every epilogue step, alone and together, on the skinny and tiled shapes, with beta, a batch and A^T. */
    @Test void epilogues() throws Exception {
        var none = GemmF16.Epilogue.NONE;
        var epilogues = new GemmF16.Epilogue[] {
            none.withBias(GemmF16.Epilogue.Bias.COLUMN).withActivation(GemmF16.Epilogue.Activation.RELU),
            none.withBias(GemmF16.Epilogue.Bias.ROW).withResidual(),
            none.withActivation(GemmF16.Epilogue.Activation.GELU),
            GemmF16.Epilogue.swiGlu(),
            none.withBias(GemmF16.Epilogue.Bias.COLUMN).withActivation(GemmF16.Epilogue.Activation.RELU).withGate().withResidual(),
            none.withBias(GemmF16.Epilogue.Bias.COLUMN).withActivation(GemmF16.Epilogue.Activation.GELU).withGate().withResidual(),
        };
        var shapes = new Shape[] {Shape.ofTransposedB(8, 70, 64, 1f, 0f), Shape.ofTransposedB(65, 70, 64, 1f, 0f),
                new Shape(3, 66, 40, 40, 66, 70, 0.5f, 0.75f), Shape.ofTransposedB(128, 128, 64, 1f, 0f),
                Shape.ofTransposedB(4, 40, 48, 1f, 0f).withBatch(2, 4 * 48, 0, 4 * 40 + 3), Shape.ofTransposedA(64, 64, 32, 1f, 0f)};
        eachKernel(() -> {
            long seed = 200;
            for (var ep : epilogues) for (var s : shapes) checkEpilogue(s, ep, seed++);
        });
    }

    /** The epilogue's operands are checked: given exactly when used, and large enough. */
    @Test void rejectsMissingOrUnusedEpilogueOperands() {
        var s = Shape.of(8, 8, 8, 1, 0);
        var bias = GemmF16.Epilogue.NONE.withBias(GemmF16.Epilogue.Bias.COLUMN);
        try (var g = GemmF16.create(device); var a = buffer(256); var b = buffer(256); var c = buffer(256); var small = buffer(16); var e = device.encoder()) {
            assertThrows(IllegalArgumentException.class, () -> g.encode(e, a, b, c, s, bias, null, null, null), "bias missing");
            assertThrows(IllegalArgumentException.class, () -> g.encode(e, a, b, c, s, bias, small, null, null), "bias too small");
            assertThrows(IllegalArgumentException.class, () -> g.encode(e, a, b, c, s, GemmF16.Epilogue.NONE, a, null, null), "bias unused");
            assertThrows(IllegalArgumentException.class, () -> g.encode(e, a, b, c, s, GemmF16.Epilogue.swiGlu(), null, small, null), "gate too small");
            g.encode(e, a, b, c, s, bias, a, null, null);
        }
    }

    /** A batch: dense, with B shared across the batch (stride 0), and with gaps between C matrices. */
    @Test void batches() throws Exception {
        eachKernel(() -> {
            check(Shape.of(64, 64, 32, 1f, 0f).batched(5), 31, false);
            check(Shape.ofTransposedB(65, 40, 48, 1f, 0f).withBatch(3, 65 * 48, 0, 65 * 40 + 7), 32, false);
            check(new Shape(33, 70, 24, 24, 72, 70, 2f, 0.5f).withBatch(4, 33 * 24 + 5, 24 * 72, 33 * 70 + 3), 33, false);
        });
    }

    /** GemmF16 labels its dispatches with its algorithm, joined to the caller's label. */
    @Test void labelsItsDispatches() throws Exception {
        Assumptions.assumeTrue(device.capabilities().timestamps(), "needs timestamps");
        eachKernel(() -> {
            var s = Shape.of(16, 64, 64, 1, 0);
            try (var ga = buffer(s.aBytes()); var gb = buffer(s.bBytes()); var gc = buffer(s.cBytes(gemm.output())); var e = device.encoder()) {
                e.profile().fill(ga, 0).fill(gb, 0).label("decode");
                gemm.encode(e, ga, gb, gc, s);
                var sub = device.queue().submit(e); assertTrue(sub.await(WAIT));
                String expected = "decode/" + GemmF16.label(gemm.algorithm());
                assertEquals(expected, sub.dispatchTimings().getFirst().label());
            }
        }, Precision.FP32);
    }

    @Test void rejectsUndersizedBuffersAndBadShapes() {
        var s = Shape.of(8, 8, 8, 1, 0);
        try (var gemm = GemmF16.create(device); var gemm16 = GemmF16.create(device, Precision.FP16);
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

    /**
     * Shape dispatch: the default instance runs the registry's choice per shape, and every registered variant the
     * device supports fits the compiled pipeline's and the device's limits.
     */
    @Test void dispatchesByShapeWithinLimits() {
        if (!GemmF16.supports(device, Algorithm.TILED)) return;
        var shapes = new Shape[] {Shape.of(16, 300, 200, 1, 0), Shape.ofTransposedB(8, 300, 200, 1, 0), Shape.ofTransposedB(40, 300, 200, 1, 0),
                Shape.of(128, 160, 96, 1, 0), Shape.ofTransposedB(96, 64, 72, 1, 0), Shape.of(2, 300, 200, 1, 0)};
        for (var s : shapes) assertEquals(java.util.List.of(), GemmF16.REGISTRY.checkLimits(device, s), "limits for " + s);
        try (var g = GemmF16.create(device)) {
            gemm = g;
            for (var s : shapes) {
                assertEquals(GemmF16.REGISTRY.select(device, s), g.algorithm(s), s.toString());
                check(s, s.m() * 31L + s.n(), false);   // bit for bit where the registry picks a fixed-order kernel
            }
            assertEquals(Algorithm.TILED, g.algorithm(shapes[0]), "m = 16, B not transposed: tiled");
            assertEquals(Algorithm.SKINNY, g.algorithm(shapes[1]), "m <= 16, B transposed: skinny");
            assertEquals(Algorithm.TILED, g.algorithm(shapes[2]), "m = 40, B transposed: tiled");
            assertEquals(Algorithm.SKINNY, g.algorithm(shapes[5]), "m = 2, B not transposed: skinny");
            assertEquals(GemmF16.supports(device, Algorithm.SIMDGROUP_MATRIX) ? Algorithm.SIMDGROUP_MATRIX : Algorithm.TILED,
                    g.algorithm(shapes[3]), "m >= 64: matrix where available");
        } finally { gemm = null; }
    }

    /** GPU nanoseconds per multiply of {@code s} with {@code g}: the best of 4 rounds of back-to-back dispatches. */
    double nanosPer(GemmF16 g, Shape s) {
        try (var ga = buffer(s.aBytes()); var gb = buffer(s.bBytes()); var gc = buffer(s.cBytes())) {
            int reps = (int) Math.max(4, Math.min(2000, 4e10 / s.flops()));
            double best = Double.MAX_VALUE;
            for (int round = 0; round < 4; round++) {
                try (var e = device.encoder()) {
                    for (int i = 0; i < reps; i++) g.encode(e, ga, gb, gc, s);
                    try (var sub = device.queue().submit(e)) {
                        assertTrue(sub.await(WAIT));
                        best = Math.min(best, sub.gpuTime().orElseThrow().toNanos() / (double) reps);
                    }
                }
            }
            return best;
        }
    }

    /** -Dgemm.shapes=true: both algorithms over decode, small and attention shapes, for the shape registry's heuristic. */
    @Test void shapeSweep() {
        Assumptions.assumeTrue(Boolean.getBoolean("gemm.shapes"), "set -Dgemm.shapes=true");
        Assumptions.assumeTrue(GemmF16.supports(device, GemmF16.Algorithm.SIMDGROUP_MATRIX), "needs both algorithms");
        try (var tiled = GemmF16.create(device, GemmF16.Algorithm.TILED); var matrix = GemmF16.create(device, GemmF16.Algorithm.SIMDGROUP_MATRIX)) {
            int[][] shapes = {{1, 4096, 4096}, {4, 4096, 4096}, {16, 4096, 4096}, {32, 4096, 4096}, {64, 4096, 4096},
                    {64, 64, 64}, {128, 128, 128}, {256, 256, 256}, {512, 512, 512}, {32, 2048, 128}, {2048, 2048, 128}, {8, 8, 4096}};
            for (int[] m : shapes) {
                for (var shape : new Shape[] {Shape.of(m[0], m[1], m[2], 1, 0), Shape.ofTransposedB(m[0], m[1], m[2], 1, 0)}) {
                    double t = nanosPer(tiled, shape), x = nanosPer(matrix, shape);
                    System.out.printf("gemm shape %d x %d x %d %s: TILED %.1f us, SIMDGROUP_MATRIX %.1f us (%.2fx)%n", m[0], m[1], m[2],
                            shape.transposeB() ? "B^T" : "B", t / 1e3, x / 1e3, t / x);
                }
            }
        }
    }

    /** GPU nanoseconds per run of {@code body}, the best of 4 rounds of {@code reps} back-to-back runs. */
    double nanosPer(int reps, java.util.function.Consumer<GpuCommandEncoder> body) {
        double best = Double.MAX_VALUE;
        for (int round = 0; round < 4; round++) {
            try (var e = device.encoder()) {
                for (int i = 0; i < reps; i++) body.accept(e);
                try (var sub = device.queue().submit(e)) {
                    assertTrue(sub.await(WAIT));
                    best = Math.min(best, sub.gpuTime().orElseThrow().toNanos() / (double) reps);
                }
            }
        }
        return best;
    }

    /**
     * -Dgemm.decode=true: the decode products (m from 1 to 16) by algorithm, as GB/s of B and as a share of the
     * device's streaming-read bandwidth, measured as a row sum (Reduce) over 256 MiB.
     */
    @Test void decodeBandwidth() {
        Assumptions.assumeTrue(Boolean.getBoolean("gemm.decode"), "set -Dgemm.decode=true");
        double stream = 0, copy;
        long bytes = 256L << 20;
        try (var reduce = Reduce.create(device); var x = buffer(bytes); var x2 = buffer(bytes); var y = buffer(1 << 16)) {
            try (var e = device.encoder()) { e.fill(x, 0); try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); } }
            for (int rows : new int[] {1 << 12, 1 << 14}) {
                int cols = (int) (bytes / 4 / rows);
                var xv = org.dynamisengine.gpu.api.compute.kit.StridedView.matrix(rows, cols);
                var yv = org.dynamisengine.gpu.api.compute.kit.StridedView.contiguous(rows);
                stream = Math.max(stream, bytes / nanosPer(8, e -> reduce.rows(e, Reduce.Op.SUM, x, xv, y, yv)));
            }
            copy = 2.0 * bytes / nanosPer(8, e -> e.copy(x, x2));
        }
        System.out.printf("gemm decode %s: streaming read %.0f GB/s (row sums), copy %.0f GB/s read + write%n", device.adapter().name(), stream, copy);
        int[][] shapes = {{4096, 4096}, {11008, 4096}, {4096, 11008}};
        for (var algorithm : Algorithm.values()) {
            if (!GemmF16.supports(device, algorithm)) continue;
            try (var g = GemmF16.create(device, algorithm)) {
                for (int[] nk : shapes) {
                    for (int m : new int[] {1, 2, 4, 8, 16}) {
                        for (var shape : new Shape[] {Shape.ofTransposedB(m, nk[0], nk[1], 1, 0), Shape.of(m, nk[0], nk[1], 1, 0)}) {
                            try (var ga = buffer(pad4(shape.aBytes())); var gb = buffer(pad4(shape.bBytes())); var gc = buffer(shape.cBytes()); var e0 = device.encoder()) {
                                e0.fill(ga, 0).fill(gb, 0);
                                try (var sub = device.queue().submit(e0)) { assertTrue(sub.await(WAIT)); }
                                double ns = nanosPer(20, e -> g.encode(e, ga, gb, gc, shape));
                                double gbs = shape.bBytes() / ns;
                                System.out.printf("gemm decode %s m=%d n=%d k=%d %s: %.1f us, %.0f GB/s of B, %.0f%% of streaming%n", algorithm, m,
                                        nk[0], nk[1], shape.transposeB() ? "B^T" : "B", ns / 1e3, gbs, 100 * gbs / stream);
                            }
                        }
                    }
                }
            }
        }
    }

    /** -Dgemm.benchmark=true: GFLOP/s for square multiplies over back-to-back dispatches, by GPU time and by wall time. */
    @Test void benchmark() throws Exception {
        Assumptions.assumeTrue(Boolean.getBoolean("gemm.benchmark"), "set -Dgemm.benchmark=true");
        eachKernel(this::benchmarkOne, Precision.FP32);
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
