package org.dynamisengine.gpu.primitives;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.EnumSet;
import java.util.SplittableRandom;
import org.dynamisengine.gpu.api.compute.BufferDescriptor;
import org.dynamisengine.gpu.api.compute.BufferUsage;
import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.gpu.api.compute.GpuReadback;
import org.dynamisengine.gpu.api.compute.KernelErrors;
import org.dynamisengine.gpu.api.compute.Placement;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * The kit primitives on a real device: exact operations bit for bit against their CPU references, the others within
 * their accuracy contracts, over dense, strided, transposed, broadcast and batched views.
 */
abstract class KitPrimitivesConformance {
    protected abstract GpuProvider provider();
    protected GpuDevice device;
    static final Duration WAIT = Duration.ofSeconds(60);

    @BeforeEach void open() { device = provider().adapters().getFirst().open(); }
    @AfterEach void close() { device.close(); }

    GpuBuffer buffer(long bytes) {
        return device.allocate(new BufferDescriptor(Math.max(4, bytes), EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE,
                BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE));
    }

    static byte[] bytes(float[] f) {
        var b = ByteBuffer.allocate(4 * f.length).order(ByteOrder.nativeOrder());
        for (float v : f) b.putFloat(v);
        return b.array();
    }

    static float[] floats(GpuReadback r) {
        var b = ByteBuffer.wrap(r.bytes()).order(ByteOrder.nativeOrder());
        float[] f = new float[r.bytes().length / 4];
        for (int i = 0; i < f.length; i++) f[i] = b.getFloat(4 * i);
        return f;
    }

    static int[] ints(GpuReadback r) {
        var b = ByteBuffer.wrap(r.bytes()).order(ByteOrder.nativeOrder());
        int[] f = new int[r.bytes().length / 4];
        for (int i = 0; i < f.length; i++) f[i] = b.getInt(4 * i);
        return f;
    }

    /** Random normal floats in [-range, range], with special values sprinkled in when {@code specials}. */
    static float[] random(int n, long seed, double range, boolean specials) {
        var r = new SplittableRandom(seed);
        float[] special = {0f, -0f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 1f, -1f, Float.MAX_VALUE};
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            f[i] = specials && r.nextInt(16) == 0 ? special[r.nextInt(special.length)] : (float) ((r.nextDouble() * 2 - 1) * range);
        }
        return f;
    }

    static void assertSameBits(float want, float got, String where) {
        if (Float.isNaN(want) && Float.isNaN(got)) return;
        assertEquals(Float.floatToRawIntBits(want), Float.floatToRawIntBits(got), where + ": want " + want + ", got " + got);
    }

    /** Runs a unary op from x (whole buffer) into y (whole buffer) and returns y. */
    float[] unary(Elementwise ew, Elementwise.Unary op, float[] x, StridedView xv, int yLength, StridedView yv) {
        try (var gx = buffer(4L * x.length); var gy = buffer(4L * yLength); var e = device.encoder()) {
            float[] init = new float[yLength];
            java.util.Arrays.fill(init, -7f);
            e.upload(gx, bytes(x)).upload(gy, bytes(init));
            ew.unary(e, op, gx, xv, gy, yv);
            var r = e.download(gy);
            try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
            return floats(r);
        }
    }

    @Test void unaryOperationsMeetTheirContracts() {
        int rows = 37, cols = 53, ld = 64;
        float[] x = random(rows * ld, 1, 50, true);
        var xv = StridedView.matrix(0, rows, cols, ld);
        var yv = StridedView.matrix(rows, cols);   // dense output
        try (var ew = Elementwise.create(device)) {
            for (var op : Elementwise.Unary.values()) {
                float[] in = x.clone();
                if (op == Elementwise.Unary.EXP) for (int i = 0; i < in.length; i++) in[i] = Float.isFinite(in[i]) ? in[i] / 8 : in[i];   // stay in range
                float[] y = unary(ew, op, in, xv, rows * cols, yv);
                var contract = Elementwise.contract(op);
                assertEquals(op.exact(), contract.reproducible(), op.toString());
                long worst = 0;
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        float a = in[i * ld + j], got = y[i * cols + j];
                        String where = op + " of " + a + " at (" + i + ", " + j + ")";
                        if (op.exact()) { assertSameBits(Elementwise.apply(op, a), got, where); continue; }
                        double exact = switch (op) {
                            case SQRT -> Math.sqrt(a);
                            case EXP -> Math.exp(a);
                            case LOG -> Math.log(a);
                            case SILU -> a / (1 + Math.exp(-(double) a));
                            case GELU -> 0.5 * a * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (a + 0.044715 * a * a * (double) a)));
                            default -> 1.0 / a;
                        };
                        float want = (float) exact;
                        if (Float.isNaN(want) || Float.isInfinite(want) || want == 0 || Math.abs(want) < 0x1p-126) {
                            // special cases: NaN stays NaN, overflow to infinity, exact zero; subnormals may flush
                            if (Float.isNaN(want)) assertTrue(Float.isNaN(got), where);
                            else if (Math.abs(want) >= 0x1p-126 || want == 0) assertSameBits(want, got, where);
                            continue;
                        }
                        if (op == Elementwise.Unary.GELU) {
                            // an absolute term near zero, where a distance in ULP counts every float down to 0
                            double u = Math.sqrt(2 / Math.PI) * (a + 0.044715 * a * a * (double) a);
                            double bound = (8 + 4 * Math.min(30, Math.abs(u))) * Math.ulp(want) + Math.abs(a) * 0x1p-22;
                            assertTrue(Math.abs(got - exact) <= bound, where + ": " + got + ", exact " + exact);
                            continue;
                        }
                        long ulps = AccuracyContract.ulpDistance(want, got);
                        worst = Math.max(worst, ulps);
                        double allowed = switch (op) {
                            case EXP -> Math.max(4, 3 + 2 * Math.abs(a));
                            case LOG -> (a < 0.5 || a > 2) ? 4 : Math.max(4, 0x1p-21 / Math.ulp(want));
                            case SILU -> 7 + 2 * Math.abs(a);
                            default -> contract.bound().ulps();
                        };
                        assertTrue(ulps <= allowed, where + ": " + ulps + " ULP, contract " + contract.bound().formula());
                    }
                }
                if (!op.exact()) System.out.println("elementwise " + op + " on " + device.adapter().name() + ": worst " + worst + " ULP");
            }
        }
    }

    @Test void binaryOperationsBroadcastAndMeetTheirContracts() {
        int rows = 29, cols = 70;
        float[] a = random(cols * rows, 2, 100, true), b = random(cols, 3, 100, true);
        // a is stored transposed (cols x rows) and read through a transposed view; b is a row broadcast down the rows
        var av = StridedView.matrix(cols, rows).transpose(0, 1);
        var bv = StridedView.contiguous(cols);
        var yv = StridedView.matrix(rows, cols);
        try (var ew = Elementwise.create(device)) {
            for (var op : Elementwise.Binary.values()) {
                float[] y;
                try (var ga = buffer(4L * a.length); var gb = buffer(4L * b.length); var gy = buffer(4L * rows * cols); var e = device.encoder()) {
                    e.upload(ga, bytes(a)).upload(gb, bytes(b));
                    ew.binary(e, op, ga, av, gb, bv, gy, yv);
                    var r = e.download(gy);
                    try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
                    y = floats(r);
                }
                long worst = 0;
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        float x1 = a[j * rows + i], x2 = b[j], got = y[i * cols + j];
                        float want = Elementwise.apply(op, x1, x2);
                        String where = op + "(" + x1 + ", " + x2 + ") at (" + i + ", " + j + ")";
                        if (Math.abs(want) < 0x1p-126 && want != 0) continue;   // subnormal results may flush
                        if (op.exact() || !Float.isFinite(want) || want == 0) { assertSameBits(want, got, where); continue; }
                        if (Math.abs(x2) < 0x1p-126 || Math.abs(x2) > 0x1p126) continue;
                        long ulps = AccuracyContract.ulpDistance(want, got);
                        worst = Math.max(worst, ulps);
                        assertTrue(ulps <= (op == Elementwise.Binary.SWIGLU ? 8 + 2 * Math.abs(x1) : 2.5), where + ": " + ulps + " ULP");
                    }
                }
                if (!op.exact()) System.out.println("elementwise " + op + " on " + device.adapter().name() + ": worst " + worst + " ULP");
            }
            // shapes that do not broadcast, and outputs that alias themselves, are refused
            try (var g = buffer(4L * rows * cols); var e = device.encoder()) {
                assertThrows(IllegalArgumentException.class, () -> ew.binary(e, Elementwise.Binary.ADD, g, StridedView.contiguous(3), g,
                        StridedView.contiguous(1), g, StridedView.contiguous(5)));
                assertThrows(IllegalArgumentException.class, () -> ew.unary(e, Elementwise.Unary.COPY, g, StridedView.contiguous(4), g,
                        StridedView.contiguous(1).batch(4, 0)));
            }
        }
    }

    float[] reduce(Reduce reduce, Reduce.Op op, float[] x, StridedView xv, int yLength, StridedView yv) {
        try (var gx = buffer(4L * x.length); var gy = buffer(4L * yLength); var e = device.encoder()) {
            e.upload(gx, bytes(x));
            e.fill(gy, Float.floatToRawIntBits(-7f));
            reduce.rows(e, op, gx, xv, gy, yv);
            var r = e.download(gy);
            try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
            return floats(r);
        }
    }

    @Test void reductionsMatchTheirReferenceBitForBit() {
        try (var reduce = Reduce.create(device)) {
            // row lengths around the group size, long rows, a batched rank-3 view and a transposed view
            int[] lengths = {1, 7, 255, 256, 257, 1000, 4099};
            for (int n : lengths) {
                int rows = 13;
                float[] x = random(rows * n, n, 10, false);
                var xv = StridedView.matrix(rows, n);
                var yv = StridedView.contiguous(rows);
                for (var op : Reduce.Op.values()) {
                    float[] y = reduce(reduce, op, x, xv, rows, yv), want = new float[rows];
                    Reduce.reference(op, x, xv, want, yv);
                    for (int r = 0; r < rows; r++) assertSameBits(want[r], y[r], op + " of row " + r + " of length " + n);
                    if (op == Reduce.Op.SUM) {
                        for (int r = 0; r < rows; r++) {
                            double exact = 0, abs = 0;
                            for (int j = 0; j < n; j++) { exact += x[r * n + j]; abs += Math.abs(x[r * n + j]); }
                            double bound = (Math.ceil(n / 256.0) + 8) * 0x1p-24 * abs * 1.01 + 0x1p-126;
                            assertTrue(Math.abs(y[r] - exact) <= bound, "sum of row " + r + " of length " + n + " within its contract");
                        }
                    }
                }
            }
            // batched: [4][6][300] reduced to [4][6], with the input transposed in its outer dimensions
            float[] x = random(4 * 6 * 300, 9, 10, true);
            var xv = StridedView.of(0, new long[] {6, 4, 300}, new long[] {300, 6 * 300, 1});
            var yv = StridedView.of(0, new long[] {6, 4}, new long[] {1, 6});
            for (var op : Reduce.Op.values()) {
                float[] y = reduce(reduce, op, x, xv, 24, yv), want = new float[24];
                Reduce.reference(op, x, xv, want, yv);
                for (int r = 0; r < 24; r++) assertSameBits(want[r], y[r], op + " of batched row " + r);
            }
            // NaN and signed zeros follow Java's Math.max and Math.min; an empty row gives the identity
            float[] specials = {-0f, 0f, -0f, Float.NaN, 3f, -0f, -0f, -0f};
            var sv = StridedView.matrix(2, 4);
            for (var op : new Reduce.Op[] {Reduce.Op.MAX, Reduce.Op.MIN}) {
                float[] y = reduce(reduce, op, specials, sv, 2, StridedView.contiguous(2)), want = new float[2];
                Reduce.reference(op, specials, sv, want, StridedView.contiguous(2));
                for (int r = 0; r < 2; r++) assertSameBits(want[r], y[r], op + " of special row " + r);
            }
            float[] empty = reduce(reduce, Reduce.Op.MAX, new float[1], StridedView.matrix(0, 3, 0, 0), 3, StridedView.contiguous(3));
            for (int r = 0; r < 3; r++) assertSameBits(Float.NEGATIVE_INFINITY, empty[r], "max of an empty row");
        }
    }

    @Test void argmaxFindsTheFirstLargest() {
        try (var reduce = Reduce.create(device)) {
            int rows = 9, n = 777;
            float[] x = random(rows * n, 11, 4, false);
            for (int r = 0; r < rows; r++) for (int j = 0; j < n; j++) x[r * n + j] = Math.round(x[r * n + j]);   // many ties
            x[3 * n + 500] = Float.NaN; x[3 * n + 600] = Float.NaN;   // the first NaN wins
            x[5 * n + 776] = 100f;
            var xv = StridedView.matrix(rows, n);
            int[] got;
            try (var gx = buffer(4L * x.length); var gi = buffer(4L * rows); var e = device.encoder()) {
                e.upload(gx, bytes(x));
                reduce.argmax(e, gx, xv, gi, StridedView.contiguous(rows));
                var r = e.download(gi);
                try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
                got = ints(r);
            }
            int[] want = new int[rows];
            Reduce.referenceArgmax(x, xv, want, StridedView.contiguous(rows));
            for (int r = 0; r < rows; r++) assertEquals(want[r], got[r], "argmax of row " + r);
            assertEquals(500, got[3]); assertEquals(776, got[5]);
        }
    }

    @Test void scansMatchTheirReferenceBitForBit() {
        try (var scan = Scan.create(device)) {
            for (int n : new int[] {1, 100, 256, 257, 1000, 3000}) {
                int rows = 5, ld = n + 3;
                float[] x = random(rows * ld, 100 + n, 10, false);
                var xv = StridedView.matrix(0, rows, n, ld);
                var yv = StridedView.matrix(rows, n);
                for (var op : Reduce.Op.values()) {
                    float[] y;
                    try (var gx = buffer(4L * x.length); var gy = buffer(4L * rows * n); var e = device.encoder()) {
                        e.upload(gx, bytes(x));
                        scan.rows(e, op, gx, xv, gy, yv);
                        var r = e.download(gy);
                        try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
                        y = floats(r);
                    }
                    float[] want = new float[rows * n];
                    Scan.reference(op, x, xv, want, yv);
                    for (int i = 0; i < want.length; i++) assertSameBits(want[i], y[i], op + " scan element " + i + " of rows of " + n);
                    if (op == Reduce.Op.SUM) {
                        for (int r = 0; r < rows; r++) {
                            double exact = 0, abs = 0;
                            for (int j = 0; j < n; j++) {
                                exact += x[r * ld + j]; abs += Math.abs(x[r * ld + j]);
                                double bound = (18 + Math.ceil((j + 1) / 256.0)) * 0x1p-24 * abs * 1.01 + 0x1p-126;
                                assertTrue(Math.abs(y[r * n + j] - exact) <= bound, "prefix " + j + " of row " + r + " within its contract");
                            }
                        }
                    }
                }
            }
        }
    }

    @Test void nanChecksReportTheFirstNonFiniteElement() {
        try (var check = NanCheck.create(device); var errors = device.allocate(KernelErrors.descriptor())) {
            int rows = 40, cols = 100, ld = 128;
            float[] x = random(rows * ld, 5, 10, false);
            x[7 * ld + 120] = Float.NaN;            // outside the view's columns: ignored
            x[9 * ld + 3] = Float.POSITIVE_INFINITY;
            x[30 * ld + 50] = Float.NaN;
            var xv = StridedView.matrix(0, rows, cols, ld);
            try (var gx = buffer(4L * x.length)) {
                try (var e = device.encoder()) {
                    e.upload(gx, bytes(x)).resetErrors(errors);
                    check.encode(e, gx, xv, errors);
                    var report = e.download(errors);
                    try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
                    assertEquals(new KernelErrors.Report(NanCheck.NONFINITE, 9L * cols + 3, 2), KernelErrors.parse(report.bytes()));
                }
                // a tripwire fails its submission; clean data passes
                try (var e = device.encoder()) {
                    check.tripwire(e, gx, xv, errors);
                    var s = device.queue().submit(e);
                    assertEquals(ComputeException.Code.KERNEL_ERROR, assertThrows(ComputeException.class, () -> s.await(WAIT)).code());
                    s.close();   // a failed submission closes quietly: its failure was reported
                }
                try (var e = device.encoder()) {
                    check.tripwire(e, gx, StridedView.matrix(0, 9, cols, ld), errors);   // rows 0 .. 8 are finite
                    try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
                }
            }
        }
    }

    /** Debug builds check every access yet give the same bits and report nothing for valid views. */
    @Test void debugBuildsMatchNormalBuilds() {
        var debugOptions = new org.dynamisengine.gpu.api.compute.kit.KitOptions(false, true);
        int rows = 33, cols = 300;
        float[] a = random(rows * cols, 21, 10, false), b = random(cols, 22, 10, false);
        var mv = StridedView.matrix(rows, cols);
        float[][] results = new float[2][];
        for (int pass = 0; pass < 2; pass++) {
            boolean debug = pass == 1;
            try (var ew = debug ? debugOptions.call(() -> Elementwise.create(device)) : Elementwise.create(device);
                 var reduce = debug ? debugOptions.call(() -> Reduce.create(device)) : Reduce.create(device);
                 var scan = debug ? debugOptions.call(() -> Scan.create(device)) : Scan.create(device);
                 var ga = buffer(4L * a.length); var gb = buffer(4L * b.length); var gy = buffer(4L * a.length);
                 var gs = buffer(4L * a.length); var gr = buffer(4L * rows); var e = device.encoder()) {
                e.upload(ga, bytes(a)).upload(gb, bytes(b));
                ew.binary(e, Elementwise.Binary.MUL, ga, mv, gb, StridedView.contiguous(cols), gy, mv);
                scan.rows(e, Reduce.Op.SUM, gy, mv, gs, mv);
                reduce.rows(e, Reduce.Op.SUM, gs, mv, gr, StridedView.contiguous(rows));
                var r1 = e.download(gs); var r2 = e.download(gr);
                try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT), "debug " + debug); }
                float[] all = new float[a.length + rows];
                System.arraycopy(floats(r1), 0, all, 0, a.length);
                System.arraycopy(floats(r2), 0, all, a.length, rows);
                results[pass] = all;
            }
        }
        for (int i = 0; i < results[0].length; i++) assertSameBits(results[0][i], results[1][i], "debug build, element " + i);
    }

    /** Reproducible mode makes the GEMM-based primitives choose the fixed-order algorithm. */
    @Test void reproducibleModeChoosesFixedOrderAlgorithms() {
        if (!device.capabilities().dataTypes().contains(org.dynamisengine.gpu.api.compute.GpuCapabilities.FP16)) return;
        var reproducible = new org.dynamisengine.gpu.api.compute.kit.KitOptions(true, false);
        try (var gemm = reproducible.call(() -> GemmF16.create(device));
             var attention = reproducible.call(() -> AttentionF16.create(device, Precision.FP16))) {
            assertEquals(GemmF16.Algorithm.TILED, gemm.algorithm());
            assertTrue(gemm.contract().reproducible());
            assertEquals("GemmF16.reference", gemm.contract().reference());
            assertEquals(GemmF16.Algorithm.TILED, attention.algorithm());
        }
        try (var gemm = GemmF16.create(device)) {
            assertEquals(gemm.algorithm() == GemmF16.Algorithm.TILED, gemm.contract().reproducible());
        }
    }
}
