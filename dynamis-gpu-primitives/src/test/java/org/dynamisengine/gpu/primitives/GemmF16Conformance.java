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
    /**
     * Runs {@code body} with {@link #gemm} set to each supported algorithm, for each input type and each output in
     * {@code outputs}.
     */
    void eachKernel(Body body, Precision... outputs) throws Exception {
        for (var algorithm : Algorithm.values()) {
            for (var input : GemmF16.Input.values()) {
                for (var output : outputs) {
                    if (!GemmF16.supports(device, algorithm, input)) {
                        assertThrows(ComputeException.class, () -> GemmF16.create(device, algorithm, input, output), algorithm + " " + input + " refused");
                        continue;
                    }
                    try (var g = GemmF16.create(device, algorithm, input, output)) { gemm = g; body.run(); } finally { gemm = null; }
                }
            }
        }
    }

    /** Values of A or B for the kernel under test's input type: the bytes to upload and how the references read them. */
    record Data(byte[] bytes, GemmF16.Elements elements) { float at(long i) { return elements.at(i); } }

    /** {@code count} random elements in [-1, 1] of {@code input} (count even for 2-byte types, so whole words move). */
    static Data data(GemmF16.Input input, int count, SplittableRandom r) {
        switch (input) {
            case FP16 -> { short[] h = halves(count, r); return new Data(bytes(h), GemmF16.Elements.fp16(h)); }
            case BF16 -> {
                short[] h = new short[count];
                for (int i = 0; i < count; i++) h[i] = GemmF16.floatToBf16((float) (r.nextDouble() * 2 - 1));
                return new Data(bytes(h), GemmF16.Elements.bf16(h));
            }
            default -> { float[] f = floats(count, r); return new Data(bytes(f), GemmF16.Elements.fp32(f)); }
        }
    }

    /** Whether the kernel under test must match its reference bit for bit on this device for {@code s}. */
    boolean bitwise(Shape s) {
        return gemm.algorithm(s) != Algorithm.SIMDGROUP_MATRIX
                && (gemm.input().exactProducts() || device.capabilities().shaderFeatures().contains(GpuCapabilities.FUSED_FMA));
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
            try (var ga = buffer(pad4(s.aBytes(gemm.input()))); var gb = buffer(pad4(s.bBytes(gemm.input()))); var gc = buffer(pad4(s.cBytes(gemm.output())));
                 var e = device.encoder()) {
                assertThrows(IllegalArgumentException.class, () -> gemm.encode(e, ga, gb, gc, s), "SKINNY refuses m > 16 and A^T");
            }
            return 0;
        }
        var r = new SplittableRandom(seed);
        var output = gemm.output();
        var in = gemm.input();
        // transfers move whole 32-bit words, so 2-byte arrays are padded to an even length
        Data a = data(in, (int) (pad4(s.aBytes(in)) / in.bytes()), r), b = data(in, (int) (pad4(s.bBytes(in)) / in.bytes()), r);
        int cLength = (int) (pad4(s.cBytes(output)) / output.bytes());
        float[] cF = new float[cLength]; short[] cH = new short[cLength];
        for (int i = 0; i < cLength; i++) {
            float v = nanC ? Float.NaN : (float) (r.nextDouble() * 2 - 1);
            cF[i] = v; cH[i] = Float.floatToFloat16(v);
        }
        boolean fp32 = output == Precision.FP32;
        ByteBuffer got;
        try (var ga = buffer(a.bytes().length); var gb = buffer(b.bytes().length); var gc = buffer(pad4(s.cBytes(output))); var e = device.encoder()) {
            e.upload(ga, a.bytes()).upload(gb, b.bytes()).upload(gc, fp32 ? bytes(cF) : bytes(cH));
            gemm.encode(e, ga, gb, gc, s); var read = e.download(gc);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            got = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
        }
        // bit for bit against the reference, every element (untouched elements must keep their initial bits)
        int differ = 0;
        Algorithm order = gemm.algorithm(s) == Algorithm.SIMDGROUP_MATRIX ? Algorithm.TILED : gemm.algorithm(s);
        var none = GemmF16.Epilogue.NONE;
        if (fp32) {
            float[] exact = cF.clone();
            GemmF16.reference(order, a.elements(), b.elements(), exact, s, none, null, null, null);
            for (int i = 0; i < cLength; i++) if (Float.floatToRawIntBits(exact[i]) != Float.floatToRawIntBits(got.getFloat(4 * i))) differ++;
        } else {
            short[] exact = cH.clone();
            GemmF16.referenceF16(order, a.elements(), b.elements(), exact, s, none, null, null, null);
            for (int i = 0; i < cLength; i++) if (exact[i] != got.getShort(2 * i)) differ++;
        }
        String what = gemm.algorithm(s) + " " + in + " " + output + " " + s;
        if (bitwise(s)) assertEquals(0, differ, "elements differing from the reference for " + what);
        else System.out.println("gemm " + gemm.algorithm(s) + " " + in + " " + output + " " + s.m() + "x" + s.n() + "x" + s.k() + "x" + s.batch()
                + (s.transposeB() ? " B^T" : "") + " differs from reference in " + differ + " of " + cLength);
        double worst = 0;
        for (int z = 0; z < s.batch(); z++) {
            for (int i = 0; i < s.m(); i++) {
                for (int j = 0; j < s.n(); j++) {
                    double sum = 0, abs = 0;
                    for (int q = 0; q < s.k(); q++) {
                        double p = (double) a.at(s.aIndex(z, i, q)) * b.at(s.bIndex(z, q, j));
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

    /**
     * The matrix kernel's row-only edge (1.15.0): where only rows pass m (complete column tiles, k a multiple of 8),
     * the m logical rows of C are the same bits as the same product with A padded by zero rows to whole 64-row tiles,
     * which the interior path computes, and nothing outside them is written: the padding of a wider ldc and the words
     * after C keep their sentinel. Both layouts of B, FP16 and FP32 inputs, FP32 and FP16 outputs, a batch of two, and
     * the epilogues the transformer uses (column bias, residual, SiLU gated); unaligned k takes the general edge path
     * in both runs.
     */
    @Test void rowEdgeMatchesPaddedRows() throws Exception {
        if (!GemmF16.supports(device, Algorithm.SIMDGROUP_MATRIX)) return;
        var epilogues = new GemmF16.Epilogue[] {GemmF16.Epilogue.NONE, GemmF16.Epilogue.NONE.withBias(GemmF16.Epilogue.Bias.COLUMN),
                GemmF16.Epilogue.NONE.withResidual(), GemmF16.Epilogue.NONE.withActivation(GemmF16.Epilogue.Activation.SILU).withGate()};
        var r = new SplittableRandom(6413);
        int cases = 0;
        for (var input : new GemmF16.Input[] {GemmF16.Input.FP16, GemmF16.Input.FP32}) {
            if (!GemmF16.supports(device, Algorithm.SIMDGROUP_MATRIX, input)) continue;
            for (var output : Precision.values()) {
                try (var g = GemmF16.create(device, Algorithm.SIMDGROUP_MATRIX, input, output)) {
                    for (int m : new int[] {1, 7, 8, 9, 17, 33, 63, 65, 100, 127})
                        for (int[] nk : new int[][] {{64, 72}, {192, 136}, {128, 100}})
                            for (boolean t : new boolean[] {false, true})
                                for (var ep : epilogues) {
                                    rowEdgeCase(g, input, output, m, nk[0], nk[1], t, ep, r);
                                    cases++;
                                }
                }
            }
        }
        System.out.println("gemm row edge on " + device.adapter().name() + ": " + cases + " cases bit for bit against padded rows");
    }

    private void rowEdgeCase(GemmF16 g, GemmF16.Input input, Precision output, int m, int n, int k, boolean transB, GemmF16.Epilogue ep,
            SplittableRandom r) {
        int mp = Math.ceilDiv(m, 64) * 64, ldc = n + 8, batch = 2, guard = 32, cb = output.bytes();
        int lda = k, ldb = transB ? k : n;
        long aStride = (long) m * lda, apStride = (long) mp * lda, bStride = (long) (transB ? n : k) * ldb;
        long cStride = (long) m * ldc, cpStride = (long) mp * ldc;
        var s = new Shape(m, n, k, lda, ldb, ldc, 0.75f, 0f, transB, batch, aStride, bStride, cStride, false);
        var sp = new Shape(mp, n, k, lda, ldb, ldc, 0.75f, 0f, transB, batch, apStride, bStride, cpStride, false);
        var a = data(input, (int) (batch * aStride + (batch * aStride & 1)), r);
        var b = data(input, (int) (batch * bStride + (batch * bStride & 1)), r);
        // A padded to mp rows per matrix: the added rows are zero
        var ap = ByteBuffer.allocate((int) pad4(batch * apStride * input.bytes())).order(ByteOrder.nativeOrder());
        for (int z = 0; z < batch; z++)
            ap.put((int) (z * apStride * input.bytes()), a.bytes(), (int) (z * aStride * input.bytes()), (int) (aStride * input.bytes()));
        // epilogue operands, laid out as C (residual, gate) or n per column (bias), in C's type
        int cElems = (int) (batch * cStride), cpElems = (int) (batch * cpStride);
        float[] side = new float[cpElems];
        for (int i = 0; i < side.length; i++) side[i] = (float) (r.nextDouble() * 2 - 1);
        float[] sideM = new float[cElems];
        for (int z = 0; z < batch; z++) System.arraycopy(side, (int) (z * cpStride), sideM, (int) (z * cStride), (int) cStride);
        java.util.function.Function<float[], byte[]> asC = v -> {
            var bb = ByteBuffer.allocate((int) pad4((long) v.length * cb)).order(ByteOrder.nativeOrder());
            for (float x : v) { if (cb == 4) bb.putFloat(x); else bb.putShort(Float.floatToFloat16(x)); }
            return bb.array();
        };
        boolean usesSide = ep.residual() || ep.gated();
        int sentinel = 0x7FC0_7FC0;
        try (var ga = buffer(a.bytes().length); var gap = buffer(ap.capacity()); var gb = buffer(b.bytes().length);
             var gc = buffer(pad4((long) (cElems + guard) * cb)); var gcp = buffer(pad4((long) cpElems * cb));
             var gs = buffer(pad4((long) cElems * cb)); var gsp = buffer(pad4((long) cpElems * cb)); var gbias = buffer(pad4((long) n * cb));
             var e = device.encoder()) {
            float[] bias = new float[n];
            for (int i = 0; i < n; i++) bias[i] = (float) (r.nextDouble() - 0.5);
            e.upload(ga, a.bytes()).upload(gap, ap.array()).upload(gb, b.bytes()).upload(gs, asC.apply(sideM)).upload(gsp, asC.apply(side))
                    .upload(gbias, asC.apply(bias)).fill(gc, sentinel).fill(gcp, sentinel);
            var biasArg = ep.bias() == GemmF16.Epilogue.Bias.NONE ? null : gbias;
            g.encode(e, ga, gb, gc, s, ep, biasArg, ep.gated() ? gs : null, ep.residual() ? gs : null);
            g.encode(e, gap, gb, gcp, sp, ep, biasArg, ep.gated() ? gsp : null, ep.residual() ? gsp : null);
            var rc = e.download(gc); var rcp = e.download(gcp);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT)); }
            var got = ByteBuffer.wrap(rc.bytes()).order(ByteOrder.nativeOrder());
            var want = ByteBuffer.wrap(rcp.bytes()).order(ByteOrder.nativeOrder());
            String what = input + " -> " + output + " " + m + "x" + n + "x" + k + (transB ? " B^T " : " B ") + ep;
            int sentinelC = cb == 4 ? sentinel : 0x7FC0;
            for (int z = 0; z < batch; z++)
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < ldc; j++) {
                        long at = z * cStride + (long) i * ldc + j, atp = z * cpStride + (long) i * ldc + j;
                        int g1 = cb == 4 ? got.getInt((int) (4 * at)) : got.getShort((int) (2 * at)) & 0xFFFF;
                        int g2 = cb == 4 ? want.getInt((int) (4 * atp)) : want.getShort((int) (2 * atp)) & 0xFFFF;
                        if (j < n) assertEquals(g2, g1, what + ": C[" + z + "][" + i + "][" + j + "] differs from the padded rows");
                        else assertEquals(sentinelC, g1, what + ": ldc padding C[" + z + "][" + i + "][" + j + "] written");
                    }
            for (int w = 0; w < guard; w++) {
                long at = cElems + w;
                int g1 = cb == 4 ? got.getInt((int) (4 * at)) : got.getShort((int) (2 * at)) & 0xFFFF;
                assertEquals(sentinelC, g1, what + ": the word " + w + " after C written");
            }
        }
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
        var in = gemm.input();
        Data a = data(in, (int) (pad4(s.aBytes(in)) / in.bytes()), r), b = data(in, (int) (pad4(s.bBytes(in)) / in.bytes()), r);
        int cLength = (int) (pad4(s.cBytes(output)) / output.bytes());
        int biasLength = ep.bias() == GemmF16.Epilogue.Bias.ROW ? s.m() : s.n();
        float[] c = floats(cLength, r), bias = floats(biasLength + 1, r), gate = floats(cLength, r), residual = floats(cLength, r);
        if (!fp32) for (float[] f : new float[][] {c, bias, gate, residual}) for (int i = 0; i < f.length; i++) f[i] = Float.float16ToFloat(Float.floatToFloat16(f[i]));
        ByteBuffer got;
        try (var ga = buffer(a.bytes().length); var gb = buffer(b.bytes().length); var gc = buffer(pad4(s.cBytes(output)));
             var gbias = buffer(pad4((long) output.bytes() * bias.length)); var ggate = buffer(pad4(s.cBytes(output)));
             var gres = buffer(pad4(s.cBytes(output))); var e = device.encoder()) {
            e.upload(ga, a.bytes()).upload(gb, b.bytes()).upload(gc, encode(c, fp32)).upload(gbias, encode(bias, fp32))
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
        GemmF16.reference(order, a.elements(), b.elements(), ref, s, ep, bias, gate, residual);   // in FP32; rounded to half below for FP16 C
        boolean bitwise = bitwise(s) && ep.exact();
        String what = run + " " + in + " " + output + " " + ep + " " + s;
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
                        abs += Math.abs((double) a.at(s.aIndex(z, i, q)) * b.at(s.bIndex(z, q, j)));
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

    /**
     * Runs a grouped GEMM over {@code problems} in shared buffers of {@code aLen}, {@code bLen} and {@code cLen}
     * elements and returns C, or throws the submission's failure.
     */
    float[] grouped(GemmF16 g, java.util.List<GemmF16.Problem> problems, GemmF16.Grouped gr, GemmF16.Epilogue ep, int cLen,
            Data a, Data b, float[] c, float[] bias, float[] residual) {
        boolean fp32 = g.output() == Precision.FP32;
        byte[] ab = a.bytes(), bb = b.bytes();
        byte[] cb = encode(c, fp32), pb = GemmF16.Problem.encode(problems), biasb = encode(bias, fp32), rb = encode(residual, fp32);
        try (var ga = buffer(ab.length); var gb = buffer(bb.length); var gc = buffer(cb.length); var gp = buffer(pb.length);
             var gbias = buffer(biasb.length); var gres = buffer(rb.length); var e = device.encoder()) {
            e.upload(ga, ab).upload(gb, bb).upload(gc, cb).upload(gp, pb).upload(gbias, biasb).upload(gres, rb);
            g.encodeGrouped(e, ga, gb, gc, gp, gr, ep, ep.bias() == GemmF16.Epilogue.Bias.NONE ? null : gbias, null, ep.residual() ? gres : null);
            var read = e.download(gc);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            var got = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
            float[] r = new float[cLen];
            for (int i = 0; i < cLen; i++) r[i] = fp32 ? got.getFloat(4 * i) : Float.float16ToFloat(got.getShort(2 * i));
            return r;
        }
    }

    /**
     * A grouped GEMM: mixture-of-experts problems (tokens routed to experts, one of them empty, B^T weights per
     * expert) and problems of unrelated shapes, with a per-problem bias and an exact epilogue, bit for bit against
     * referenceGrouped for both output types. A problem out of range, or beyond maxM, fails the submission.
     */
    @Test void groupedProblems() {
        if (!GemmF16.supports(device, Algorithm.TILED)) return;
        var r = new SplittableRandom(77);
        for (var input : GemmF16.Input.values()) for (var output : Precision.values()) {
            if (!GemmF16.supports(device, Algorithm.TILED, input)) continue;
            boolean fp32 = output == Precision.FP32;
            boolean bitwise = input.exactProducts() || device.capabilities().shaderFeatures().contains(GpuCapabilities.FUSED_FMA);
            try (var g = GemmF16.create(device, Algorithm.TILED, input, output)) {
                // experts: n = 48, k = 40, tokens 5, 0, 70, 33 (rows of one token matrix)
                int n = 48, k = 40;
                int[] tokens = {5, 0, 70, 33};
                var moe = new java.util.ArrayList<GemmF16.Problem>();
                int row = 0;
                for (int ex = 0; ex < tokens.length; ex++) {
                    moe.add(GemmF16.Problem.dense(tokens[ex], n, k, false, true, (long) row * k, (long) ex * n * k, (long) row * n, (long) ex * n));
                    row += tokens[ex];
                }
                // unrelated shapes, B not transposed, packed one after another
                var mixed = java.util.List.of(new GemmF16.Problem(65, 17, 9, 9, 17, 20, 0, 0, 0, 0),
                        new GemmF16.Problem(3, 130, 70, 72, 131, 130, 65 * 9 + 5, 9 * 17 + 3, 65 * 20 + 1, 17),
                        new GemmF16.Problem(1, 1, 1, 1, 1, 1, 65 * 9 + 5 + 3 * 72, 9 * 17 + 3 + 70 * 131, 65 * 20 + 1 + 3 * 130, 17 + 130));
                record Case(java.util.List<GemmF16.Problem> problems, GemmF16.Grouped grouped, GemmF16.Epilogue epilogue) { }
                var cases = java.util.List.of(
                        new Case(moe, new GemmF16.Grouped(moe.size(), 70, n, 1f, 0f, false, true),
                                GemmF16.Epilogue.NONE.withBias(GemmF16.Epilogue.Bias.COLUMN).withActivation(GemmF16.Epilogue.Activation.RELU)),
                        new Case(mixed, new GemmF16.Grouped(mixed.size(), 65, 130, 0.5f, 0.75f, false, false),
                                GemmF16.Epilogue.NONE.withBias(GemmF16.Epilogue.Bias.ROW).withResidual()));
                for (var cs : cases) {
                    int aLen = 0, bLen = 0, cLen = 0, biasLen = 0;
                    for (var q : cs.problems()) {
                        if (q.m() == 0) continue;
                        aLen = (int) Math.max(aLen, q.offsetA() + (long) (q.m() - 1) * q.lda() + q.k());
                        bLen = (int) Math.max(bLen, q.offsetB() + (long) ((cs.grouped().transposeB() ? q.n() : q.k()) - 1) * q.ldb()
                                + (cs.grouped().transposeB() ? q.k() : q.n()));
                        cLen = (int) Math.max(cLen, q.offsetC() + (long) (q.m() - 1) * q.ldc() + q.n());
                        biasLen = (int) Math.max(biasLen, q.offsetBias() + (cs.epilogue().bias() == GemmF16.Epilogue.Bias.ROW ? q.m() : q.n()));
                    }
                    bLen = Math.max(bLen, tokens.length * n * k);
                    Data a = data(input, aLen + 2 - (aLen & 1), r), b = data(input, bLen + 2 - (bLen & 1), r);
                    float[] c = floats(cLen, r), bias = floats(biasLen, r), residual = floats(cLen, r);
                    if (!fp32) for (float[] f : new float[][] {c, bias, residual}) for (int i = 0; i < f.length; i++) f[i] = Float.float16ToFloat(Float.floatToFloat16(f[i]));
                    float[] got = grouped(g, cs.problems(), cs.grouped(), cs.epilogue(), cLen, a, b, c, bias, residual);
                    float[] expect;
                    if (fp32) {
                        expect = c.clone();
                        GemmF16.referenceGrouped(a.elements(), b.elements(), expect, cs.problems(), cs.grouped(), cs.epilogue(), bias, null, residual);
                    } else {
                        short[] h = new short[cLen];
                        for (int i = 0; i < cLen; i++) h[i] = Float.floatToFloat16(c[i]);
                        short[] hb = new short[bias.length], hr = new short[cLen];
                        for (int i = 0; i < bias.length; i++) hb[i] = Float.floatToFloat16(bias[i]);
                        for (int i = 0; i < cLen; i++) hr[i] = Float.floatToFloat16(residual[i]);
                        GemmF16.referenceGroupedF16(a.elements(), b.elements(), h, cs.problems(), cs.grouped(), cs.epilogue(), hb, null, hr);
                        expect = new float[cLen];
                        for (int i = 0; i < cLen; i++) expect[i] = Float.float16ToFloat(h[i]);
                    }
                    int differ = 0;
                    double worst = 0;
                    for (int i = 0; i < cLen; i++) {
                        if (Float.floatToRawIntBits(got[i]) != Float.floatToRawIntBits(expect[i])) differ++;
                        worst = Math.max(worst, Math.abs(got[i] - expect[i]) / (Math.abs(expect[i]) + 1));
                    }
                    String what = "grouped " + input + " " + output + " " + cs.problems().size() + " problems";
                    if (bitwise) assertEquals(0, differ, what + ": elements differing from referenceGrouped");
                    else assertTrue(worst <= 0x1p-14, what + ": relative error " + worst);   // k <= 70 FP32 fma steps rounded twice
                }
                // a problem reaching past A, and one beyond maxM: the submission fails
                var bad = java.util.List.of(GemmF16.Problem.dense(4, 8, 8, false, false, 0, 0, 0, 0), GemmF16.Problem.dense(4, 8, 8, false, false, 1000, 0, 32, 0));
                var tooBig = java.util.List.of(GemmF16.Problem.dense(9, 8, 8, false, false, 0, 0, 0, 0));
                Data a = data(input, 64, r), b = data(input, 64, r);
                float[] c = new float[64], none = new float[2];
                for (var problems : java.util.List.of(bad, tooBig)) {
                    var gr = new GemmF16.Grouped(problems.size(), 8, 8, 1f, 0f, false, false);
                    var ex = assertThrows(ComputeException.class, () -> grouped(g, problems, gr, GemmF16.Epilogue.NONE, 64, a, b, c, none, c));
                    assertEquals(ComputeException.Code.KERNEL_ERROR, ex.code(), "an out-of-range problem fails its submission");
                }
            }
        }
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
            try (var ga = buffer(s.aBytes(gemm.input())); var gb = buffer(s.bBytes(gemm.input())); var gc = buffer(s.cBytes(gemm.output())); var e = device.encoder()) {
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
        var shapes = new Shape[] {Shape.ofTransposedB(1, 1024, 1024, 1, 0), Shape.of(16, 16, 16, 1, 0).batched(64), Shape.of(512, 512, 64, 1, 0),
                Shape.ofTransposedB(8, 300, 200, 1, 0), Shape.of(128, 160, 96, 1, 0), Shape.ofTransposedA(96, 64, 72, 1, 0), Shape.of(2, 300, 200, 1, 0)};
        for (var s : shapes) assertEquals(java.util.List.of(), GemmF16.REGISTRY.checkLimits(device, s), "limits for " + s);
        try (var g = GemmF16.create(device)) {
            gemm = g;
            for (var s : shapes) {
                assertEquals(GemmF16.REGISTRY.select(device, s), g.algorithm(s), s.toString());
                check(s, s.m() * 31L + s.n(), false);   // bit for bit where the registry picks a fixed-order kernel
            }
            assertEquals(Algorithm.SKINNY, g.algorithm(shapes[0]), "decode: skinny");
            assertEquals(Algorithm.SMALL, g.algorithm(shapes[1]), "a batch of 16 x 16: small");
            assertEquals(GemmF16.supports(device, Algorithm.SIMDGROUP_MATRIX) ? Algorithm.SIMDGROUP_MATRIX : Algorithm.TILED,
                    g.algorithm(shapes[2]), "512 x 512: matrix where available, else tiled");
            assertEquals(Algorithm.SMALL, g.algorithm(shapes[3]), "too small for skinny: small");
        } finally { gemm = null; }
        // BF16 has no matrix kernel, and dispatch leaves out FP32's, which is slower than tiled
        for (var in : new GemmF16.Input[] {GemmF16.Input.BF16, GemmF16.Input.FP32}) {
            try (var g = GemmF16.create(device, in, Precision.FP32)) {
                assertEquals(Algorithm.TILED, g.algorithm(shapes[2]), in + ", 512 x 512: tiled");
                assertEquals(Algorithm.SKINNY, g.algorithm(shapes[0]), in + ", decode: skinny");
                assertEquals(Algorithm.TILED, g.algorithm(), in + ": no matrix kernel by default");
            }
        }
    }

    /** GPU nanoseconds per multiply of {@code s} with {@code g}: the best of 4 rounds of back-to-back dispatches. */
    double nanosPer(GemmF16 g, Shape s) {
        try (var ga = buffer(s.aBytes(g.input())); var gb = buffer(s.bBytes(g.input())); var gc = buffer(s.cBytes())) {
            int reps = (int) Math.max(4, Math.min(Integer.getInteger("gemm.reps", 2000), 4e10 / s.flops()));
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

    /**
     * -Dgemm.shapes=true: every algorithm over decode, small (batched) and attention shapes, for the shape registry's
     * costs; prints each algorithm's time relative to TILED.
     */
    @Test void shapeSweep() {
        Assumptions.assumeTrue(Boolean.getBoolean("gemm.shapes"), "set -Dgemm.shapes=true");
        var kernels = new java.util.EnumMap<Algorithm, GemmF16>(Algorithm.class);
        try {
            for (var a : Algorithm.values()) if (GemmF16.supports(device, a)) kernels.put(a, GemmF16.create(device, a));
            int[][] shapes = {{1, 4096, 4096, 1}, {16, 4096, 4096, 1}, {32, 4096, 4096, 1}, {64, 4096, 4096, 1}, {128, 128, 128, 1},
                    {512, 512, 512, 1}, {2048, 2048, 128, 1}, {8, 8, 4096, 1}, {16, 16, 16, 4096}, {32, 32, 32, 4096}, {16, 16, 256, 512},
                    {32, 32, 128, 1024}, {8, 32, 64, 2048}, {64, 64, 64, 512}, {48, 48, 48, 1024}, {32, 64, 64, 512}, {8, 300, 200, 1},
                    {1, 1024, 1024, 1}, {4, 2048, 512, 1}, {16, 1024, 1024, 1}, {256, 256, 256, 1}, {96, 160, 96, 1}};
            for (int[] m : shapes) {
                for (boolean t : new boolean[] {false, true}) {
                    var shape = (t ? Shape.ofTransposedB(m[0], m[1], m[2], 1, 0) : Shape.of(m[0], m[1], m[2], 1, 0));
                    if (m[3] > 1) shape = shape.batched(m[3]);
                    double tiled = nanosPer(kernels.get(Algorithm.TILED), shape);
                    var line = new StringBuilder(String.format("gemm shape %d x %d x %d x %d %s: TILED %.1f us", m[0], m[1], m[2], m[3], t ? "B^T" : "B", tiled / 1e3));
                    for (var k : kernels.entrySet()) {
                        if (k.getKey() == Algorithm.TILED || k.getKey() == Algorithm.SKINNY && m[0] > GemmF16.SKINNY_MAX_M) continue;
                        line.append(String.format(", %s %.2f", k.getKey(), nanosPer(k.getValue(), shape) / tiled));
                    }
                    System.out.println(line);
                }
            }
        } finally {
            kernels.values().forEach(GemmF16::close);
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
            if (!GemmF16.supports(device, algorithm, benchInput())) continue;
            try (var g = GemmF16.create(device, algorithm, benchInput(), Precision.FP32)) {
                for (int[] nk : shapes) {
                    for (int m : new int[] {1, 2, 4, 8, 16}) {
                        for (var shape : new Shape[] {Shape.ofTransposedB(m, nk[0], nk[1], 1, 0), Shape.of(m, nk[0], nk[1], 1, 0)}) {
                            try (var ga = buffer(pad4(shape.aBytes(g.input()))); var gb = buffer(pad4(shape.bBytes(g.input()))); var gc = buffer(shape.cBytes()); var e0 = device.encoder()) {
                                e0.fill(ga, 0).fill(gb, 0);
                                try (var sub = device.queue().submit(e0)) { assertTrue(sub.await(WAIT)); }
                                double ns = nanosPer(20, e -> g.encode(e, ga, gb, gc, shape));
                                double gbs = shape.bBytes(g.input()) / ns;
                                System.out.printf("gemm decode %s m=%d n=%d k=%d %s: %.1f us, %.0f GB/s of B, %.0f%% of streaming%n", algorithm, m,
                                        nk[0], nk[1], shape.transposeB() ? "B^T" : "B", ns / 1e3, gbs, 100 * gbs / stream);
                            }
                        }
                    }
                }
            }
        }
    }

    /** The input type of the benchmarks: -Dgemm.input=FP16 (default), BF16 or FP32. */
    static GemmF16.Input benchInput() { return GemmF16.Input.valueOf(System.getProperty("gemm.input", "FP16")); }

    /** -Dgemm.benchmark=true: GFLOP/s for square multiplies over back-to-back dispatches, by GPU time and by wall time. */
    @Test void benchmark() throws Exception {
        Assumptions.assumeTrue(Boolean.getBoolean("gemm.benchmark"), "set -Dgemm.benchmark=true");
        eachKernel(this::benchmarkOne, Precision.FP32);
    }

    private void benchmarkOne() {
        if (gemm.algorithm() == Algorithm.SKINNY) return;   // decode shapes only: -Dgemm.decode=true
        for (int size : new int[] {512, 1024, 2048, 4096}) {
            for (var s : new Shape[] {Shape.of(size, size, size, 1, 0), Shape.ofTransposedB(size, size, size, 1, 0)}) {
                try (var ga = buffer(s.aBytes(gemm.input())); var gb = buffer(s.bBytes(gemm.input())); var gc = buffer(s.cBytes())) {
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
                    System.out.printf("gemm %s %s %s %s %d^3: %.0f GFLOP/s by GPU time, %.0f by wall time (%d per submission)%n", device.adapter().name(),
                            gemm.algorithm(), gemm.input(), s.transposeB() ? "B^T" : "B", size, reps * s.flops() / bestGpu / 1e9, reps * s.flops() / best / 1e9, reps);
                }
            }
        }
    }
}
