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
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.gpu.api.compute.Placement;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * The 1.14.0 tensor-program primitives on a real device: Logic and Cast bit for bit against their references, over
 * dense, strided, transposed and broadcast views, with the special values (NaN, infinities, signed zeros, subnormals,
 * the int32 extremes) every case includes.
 */
abstract class TensorOpsConformance {
    protected abstract GpuProvider provider();
    protected GpuDevice device;
    static final Duration WAIT = Duration.ofSeconds(60);

    @BeforeEach void open() { device = provider().adapters().getFirst().open(); }
    @AfterEach void close() { device.close(); }

    GpuBuffer buffer(long bytes) {
        return device.allocate(new BufferDescriptor(Math.max(4, bytes), EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE,
                BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE));
    }

    static byte[] words(int[] w) {
        var b = ByteBuffer.allocate(4 * w.length).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : w) b.putInt(v);
        return b.array();
    }

    static int[] words(byte[] bytes) {
        var b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int[] w = new int[bytes.length / 4];
        for (int i = 0; i < w.length; i++) w[i] = b.getInt(4 * i);
        return w;
    }

    static final int[] SPECIAL_FLOATS = {0, 0x80000000, 0x7FC00000, 0xFFC00001, 0x7F800000, 0xFF800000, 0x3F800000, 0xBF800000,
        0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x4F000000, 0xCF000000, 0x4EFFFFFF, 0x47800000,
        0x477FF000, 0x477FF001, 0x33800000, 0x33000000, 0x33000001, 0x387FC000, 0x3F808000, 0x3F818000};

    /** FP32 words: random values over many binades, with the special values mixed in. */
    static int[] floatWords(int n, long seed) {
        var r = new SplittableRandom(seed);
        int[] w = new int[n];
        for (int i = 0; i < n; i++) {
            int k = r.nextInt(8);
            w[i] = k == 0 ? SPECIAL_FLOATS[r.nextInt(SPECIAL_FLOATS.length)]
                    : k == 1 ? r.nextInt()                                                      // any bits
                    : Float.floatToRawIntBits((float) ((r.nextDouble() * 2 - 1) * Math.pow(2, r.nextInt(-30, 40))));
        }
        return w;
    }

    /** INT32 words: small and extreme values. */
    static int[] intWords(int n, long seed) {
        var r = new SplittableRandom(seed);
        int[] special = {0, 1, -1, Integer.MAX_VALUE, Integer.MIN_VALUE, 16777217, -16777217, 1 << 30};
        int[] w = new int[n];
        for (int i = 0; i < n; i++) w[i] = r.nextInt(6) == 0 ? special[r.nextInt(special.length)] : r.nextInt(5) == 0 ? r.nextInt() : r.nextInt(-8, 8);
        return w;
    }

    int[] run(java.util.function.Consumer<org.dynamisengine.gpu.api.compute.GpuCommandEncoder> record, GpuBuffer out) {
        try (var e = device.encoder()) {
            record.accept(e);
            var r = e.download(out);
            try (var s = device.queue().submit(e)) { assertTrue(s.await(WAIT)); }
            return words(r.bytes());
        }
    }

    // a: transposed (cols x rows storage) through a transposed view; b: a row broadcast down the rows
    static final int ROWS = 23, COLS = 41;

    @Test void comparisonsLogicalAndIntegerOperationsAreExact() {
        try (var logic = Logic.create(device); var ga = buffer(4L * ROWS * COLS); var gb = buffer(4L * COLS); var gy = buffer(4L * ROWS * COLS)) {
            var av = StridedView.matrix(COLS, ROWS).transpose(0, 1);
            var bv = StridedView.contiguous(COLS);
            var yv = StridedView.matrix(ROWS, COLS);
            for (var type : Logic.Operand.values()) {
                int[] a = type == Logic.Operand.FP32 ? floatWords(ROWS * COLS, 1) : intWords(ROWS * COLS, 1);
                int[] b = type == Logic.Operand.FP32 ? floatWords(COLS, 2) : intWords(COLS, 2);
                for (int j = 0; j < COLS; j += 3) b[j] = a[j * ROWS + j % ROWS];   // some equal pairs
                for (var op : Logic.Compare.values()) {
                    int[] y = run(e -> { e.upload(ga, words(a)).upload(gb, words(b)); logic.compare(e, op, type, ga, av, gb, bv, gy, yv); }, gy);
                    for (int i = 0; i < ROWS; i++) for (int j = 0; j < COLS; j++) {
                        int x1 = a[j * ROWS + i], x2 = b[j];
                        assertEquals(Logic.reference(op, type, x1, x2), y[i * COLS + j], op + " " + type + " of " + Integer.toHexString(x1) + ", " + Integer.toHexString(x2));
                    }
                }
            }
            int[] a = intWords(ROWS * COLS, 3), b = intWords(COLS, 4);
            for (var op : Logic.Logical.values()) {
                int[] y = run(e -> { e.upload(ga, words(a)).upload(gb, words(b)); logic.logical(e, op, ga, av, gb, bv, gy, yv); }, gy);
                for (int i = 0; i < ROWS; i++) for (int j = 0; j < COLS; j++) assertEquals(Logic.reference(op, a[j * ROWS + i], b[j]), y[i * COLS + j], op.toString());
            }
            for (var op : Logic.IntOp.values()) {
                int[] y = run(e -> { e.upload(ga, words(a)).upload(gb, words(b)); logic.integer(e, op, ga, av, gb, bv, gy, yv); }, gy);
                for (int i = 0; i < ROWS; i++) for (int j = 0; j < COLS; j++) assertEquals(Logic.reference(op, a[j * ROWS + i], b[j]), y[i * COLS + j], op.toString());
            }
        }
    }

    @Test void classifyNotAndSelectAreExact() {
        int n = ROWS * COLS;
        try (var logic = Logic.create(device); var gx = buffer(4L * n); var gm = buffer(4L * n); var gb = buffer(4L * COLS); var gy = buffer(4L * n)) {
            int[] x = floatWords(n, 5);
            var v = StridedView.matrix(ROWS, COLS);
            for (var op : Logic.Classify.values()) {
                int[] y = run(e -> { e.upload(gx, words(x)); logic.classify(e, op, gx, v, gy, v); }, gy);
                for (int i = 0; i < n; i++) assertEquals(Logic.reference(op, x[i]), y[i], op + " of " + Integer.toHexString(x[i]));
            }
            int[] m = intWords(n, 6);
            int[] notY = run(e -> { e.upload(gm, words(m)); logic.not(e, gm, v, gy, v); }, gy);
            for (int i = 0; i < n; i++) assertEquals(m[i] == 0 ? 1 : 0, notY[i]);
            // select: mask strided (every other column of a wider buffer), a dense, b a broadcast row; NaNs on both sides
            int[] b = floatWords(COLS, 7);
            var mv = StridedView.matrix(0, ROWS, COLS / 2, COLS);
            var yv = StridedView.matrix(ROWS, COLS / 2);
            var av = StridedView.matrix(0, ROWS, COLS / 2, COLS);
            var bv = StridedView.contiguous(COLS / 2);
            int[] y = run(e -> { e.upload(gm, words(m)).upload(gx, words(x)).upload(gb, words(b)); logic.select(e, gm, mv, gx, av, gb, bv, gy, yv); }, gy);
            for (int i = 0; i < ROWS; i++) for (int j = 0; j < COLS / 2; j++) {
                int want = m[i * COLS + j] != 0 ? x[i * COLS + j] : b[j];
                assertEquals(want, y[i * (COLS / 2) + j], "select at (" + i + ", " + j + ")");
            }
            try (var e = device.encoder()) {
                assertThrows(IllegalArgumentException.class, () -> logic.select(e, gm, StridedView.contiguous(3), gx, v, gb, bv, gy, v));
            }
        }
    }

    /** Packs 16-bit values into words, little-endian, from element {@code offset}. */
    static int[] pack16(int[] halves, int offset, int words) {
        int[] w = new int[words];
        for (int i = 0; i < halves.length; i++) {
            int e = offset + i;
            w[e >> 1] |= (halves[i] & 0xFFFF) << (16 * (e & 1));
        }
        return w;
    }

    static int half(int[] words, int e) { return words[e >> 1] >>> (16 * (e & 1)) & 0xFFFF; }

    static int[] inputs(Cast.Type type, int n, long seed) {
        var r = new SplittableRandom(seed);
        return switch (type) {
            case FP32 -> floatWords(n, seed);
            case INT32 -> intWords(n, seed);
            case BOOL -> { int[] w = new int[n]; for (int i = 0; i < n; i++) w[i] = r.nextInt(3) == 0 ? 0 : r.nextInt(3) == 0 ? r.nextInt() : 1; yield w; }
            default -> {   // 16-bit: every exponent, subnormals, infinities and NaNs
                int[] h = new int[n];
                int[] special = {0, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE01, 0x7F80, 0x7FC0, 0xFF81, 0x3C00};
                for (int i = 0; i < n; i++) h[i] = r.nextInt(5) == 0 ? special[r.nextInt(special.length)] : r.nextInt(0x10000);
                yield h;
            }
        };
    }

    @Test void castsAreExactBetweenEveryPairOfTypes() {
        int n = ROWS * COLS;
        try (var cast = Cast.create(device); var gx = buffer(4L * n + 8); var gy = buffer(4L * n + 8)) {
            for (var from : Cast.Type.values()) {
                for (var to : Cast.Type.values()) {
                    int[] x = inputs(from, n, from.ordinal() * 7 + to.ordinal());
                    // the input through a transposed view; a 16-bit output contiguous from an odd element offset, whose
                    // neighbouring halves must survive
                    var xv = StridedView.matrix(COLS, ROWS).transpose(0, 1);
                    int yOff = to.bytes() == 2 ? 3 : 0;
                    var yv = StridedView.of(yOff, new long[] {ROWS, COLS}, new long[] {COLS, 1});
                    int[] xw = from.bytes() == 2 ? pack16(x, 0, n + 2) : java.util.Arrays.copyOf(x, n + 2);
                    int[] fill = new int[n + 2];
                    java.util.Arrays.fill(fill, 0x5A5A5A5A);
                    int[] y = run(e -> { e.upload(gx, words(xw)).upload(gy, words(fill)); cast.cast(e, from, to, gx, xv, gy, yv); }, gy);
                    for (int i = 0; i < ROWS; i++) for (int j = 0; j < COLS; j++) {
                        int want = Cast.reference(from, to, x[j * ROWS + i]);
                        int e = yOff + i * COLS + j;
                        int got = to.bytes() == 2 ? half(y, e) : y[e];
                        assertEquals(want, got, from + " to " + to + " of " + Integer.toHexString(x[j * ROWS + i]) + ": want "
                                + Integer.toHexString(want) + ", got " + Integer.toHexString(got));
                    }
                    if (to.bytes() == 2) {
                        assertEquals(0x5A5A, half(y, yOff - 1), "the half before the view is kept");
                        assertEquals(0x5A5A, half(y, yOff + n), "the half after the view is kept");
                    } else {
                        assertEquals(0x5A5A5A5A, y[n], "nothing past the view is written");
                    }
                }
            }
            try (var e = device.encoder()) {
                assertThrows(IllegalArgumentException.class, () -> cast.cast(e, Cast.Type.FP32, Cast.Type.FP16, gx, StridedView.contiguous(4), gy,
                        StridedView.matrix(0, 2, 2, 4)));   // a strided 16-bit output
            }
        }
    }

    @Test void castReferenceMatchesJava() {
        // the reference itself against Java's conversions, where Java defines them (NaN payloads aside)
        var r = new SplittableRandom(11);
        for (int i = 0; i < 200_000; i++) {
            int bits = i < 1000 ? SPECIAL_FLOATS[i % SPECIAL_FLOATS.length] : r.nextInt();
            float f = Float.intBitsToFloat(bits);
            if (!Float.isNaN(f)) {
                assertEquals(Float.floatToFloat16(f) & 0xFFFF, Cast.reference(Cast.Type.FP32, Cast.Type.FP16, bits));
                assertEquals(GemmF16.floatToBf16(f) & 0xFFFF, Cast.reference(Cast.Type.FP32, Cast.Type.BF16, bits));
            }
            assertEquals((int) f, Cast.reference(Cast.Type.FP32, Cast.Type.INT32, bits));
            assertEquals(Float.floatToRawIntBits((float) bits), Cast.reference(Cast.Type.INT32, Cast.Type.FP32, bits));
        }
    }

    static float[] floats(int[] w) { float[] f = new float[w.length]; for (int i = 0; i < w.length; i++) f[i] = Float.intBitsToFloat(w[i]); return f; }
    static int[] bits(float[] f) { int[] w = new int[f.length]; for (int i = 0; i < f.length; i++) w[i] = Float.floatToRawIntBits(f[i]); return w; }

    /** Values with ties (a coarse grid) and occasional NaNs and infinities: argmax and the max's gradient meet them. */
    static float[] tied(int n, long seed) {
        var r = new SplittableRandom(seed);
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            int k = r.nextInt(40);
            f[i] = k == 0 ? Float.NaN : k == 1 ? Float.POSITIVE_INFINITY : k == 2 ? Float.NEGATIVE_INFINITY : k == 3 ? -0f : r.nextInt(-6, 7) * 0.5f;
        }
        return f;
    }

    @Test void reductionsAlongEveryAxisAreFixedOrder() {
        // x: 5 x 37 x 300, stored with a padded middle stride; reduced along each axis in turn
        int d0 = 5, d1 = 37, d2 = 300, ld = 304;
        var xv = StridedView.of(7, new long[] {d0, d1, d2}, new long[] {(long) d1 * ld, ld, 1});
        int xLen = 7 + d0 * d1 * ld;
        var rng = new SplittableRandom(21);
        float[] x = new float[xLen];
        for (int i = 0; i < xLen; i++) x[i] = (float) ((rng.nextDouble() * 2 - 1) * Math.pow(2, rng.nextInt(-8, 9)));
        float[] xt = tied(xLen, 22);
        try (var reduce = Reduce.create(device); var gx = buffer(4L * xLen); var gy = buffer(4L * d1 * d2); var gi = buffer(4L * d1 * d2);
                var gdx = buffer(4L * d0 * d1 * d2)) {
            for (int axis = 0; axis < 3; axis++) {
                long[] ye = new long[2];
                for (int d = 0, k = 0; d < 3; d++) if (d != axis) ye[k++] = xv.extent(d);
                var yv = StridedView.matrix(ye[0], ye[1]);
                int yn = (int) (ye[0] * ye[1]);
                for (var op : Reduce.Op.values()) {
                    float[] in = op == Reduce.Op.SUM || op == Reduce.Op.MEAN ? x : xt;
                    final int ax = axis;
                    int[] got = run(e -> { e.upload(gx, words(bits(in))); reduce.axis(e, op, gx, xv, ax, gy, yv); }, gy);
                    float[] want = new float[d1 * d2];
                    Reduce.reference(op, in, Views.moveToLast(xv, axis), want, yv);
                    for (int i = 0; i < yn; i++) {
                        float g = Float.intBitsToFloat(got[i]);
                        String where = op + " along axis " + axis + " at " + i;
                        if (op == Reduce.Op.MEAN) {
                            assertTrue(org.dynamisengine.gpu.api.compute.kit.AccuracyContract.ulpDistance(want[i], g) <= 2, where + ": want " + want[i] + ", got " + g);
                        } else if (!(Float.isNaN(want[i]) && Float.isNaN(g))) {
                            assertEquals(Float.floatToRawIntBits(want[i]), got[i], where + ": want " + want[i] + ", got " + g);
                        }
                    }
                }
                // argmax and the max's gradient along the axis, with ties and NaNs
                final int ax = axis;
                int[] idx = run(e -> { e.upload(gx, words(bits(xt))); reduce.argmaxAxis(e, gx, xv, ax, gi, yv); }, gi);
                int[] wantIdx = new int[d1 * d2];
                Reduce.referenceArgmax(xt, Views.moveToLast(xv, axis), wantIdx, yv);
                for (int i = 0; i < yn; i++) assertEquals(wantIdx[i], idx[i], "argmax along axis " + axis + " at " + i);
                float[] dy = new float[yn];
                for (int i = 0; i < yn; i++) dy[i] = (float) rng.nextDouble() - 0.5f;
                var dxv = StridedView.of(0, new long[] {d0, d1, d2}, new long[] {(long) d1 * d2, d2, 1});
                int[] dx = run(e -> {
                    e.upload(gi, words(idx)).upload(gy, words(java.util.Arrays.copyOf(bits(dy), d1 * d2)));
                    reduce.maxBackward(e, gi, yv, gy, yv, ax, gdx, dxv);
                }, gdx);
                float[] wantDx = new float[d0 * d1 * d2];
                Reduce.referenceMaxBackward(idx, yv, dy, yv, axis, wantDx, dxv);
                for (int i = 0; i < wantDx.length; i++) assertEquals(Float.floatToRawIntBits(wantDx[i]), dx[i], "max backward along axis " + axis + " at " + i);
            }
            try (var e = device.encoder()) {
                assertThrows(IllegalArgumentException.class, () -> reduce.axis(e, Reduce.Op.SUM, gx, xv, 3, gy, StridedView.matrix(5, 37)));
            }
        }
    }

    @Test void gatherAlongEveryAxisCopiesExactly() {
        int d0 = 4, d1 = 9, d2 = 70;
        var xv = StridedView.of(3, new long[] {d0, d1, d2}, new long[] {(long) d1 * 72, 72, 1});
        int xLen = 3 + d0 * d1 * 72;
        int[] x = floatWords(xLen, 31);
        var rng = new SplittableRandom(32);
        try (var g = Gather.create(device); var gx = buffer(4L * xLen); var gi = buffer(4L * 300); var gy = buffer(4L * d0 * d1 * 200)) {
            for (int axis = 0; axis < 3; axis++) {
                int extent = (int) xv.extent(axis), n = 2 * extent + 3;
                int[] idx = new int[300];
                for (int j = 0; j < n; j++) idx[2 * j] = rng.nextInt(extent);   // strided index view, repeats included
                var iv = StridedView.of(0, new long[] {n}, new long[] {2});
                long[] ye = xv.extents();
                ye[axis] = n;
                long[] ys = new long[3];
                ys[2] = 1; ys[1] = ye[2]; ys[0] = ye[1] * ye[2];
                var yv = StridedView.of(0, ye, ys);
                final int ax = axis;
                int[] got = run(e -> { e.upload(gx, words(x)).upload(gi, words(idx)); g.gather(e, gx, xv, ax, gi, iv, gy, yv); }, gy);
                int[] want = new int[got.length];
                Gather.referenceGather(x, xv, axis, idx, iv, want, yv);
                long count = yv.elements();
                for (int i = 0; i < count; i++) assertEquals(want[i], got[i], "gather along axis " + axis + " at " + i);
            }
            // an index past the dimension fails the submission
            var ex = assertThrows(org.dynamisengine.gpu.api.compute.ComputeException.class, () -> run(e -> {
                e.fill(gi, d1);
                g.gather(e, gx, xv, 1, gi, StridedView.contiguous(5), gy, StridedView.of(0, new long[] {d0, 5, d2}, new long[] {5L * d2, d2, 1}));
            }, gy));
            assertEquals(org.dynamisengine.gpu.api.compute.ComputeException.Code.KERNEL_ERROR, ex.code());
        }
    }

    @Test void scatterAddSumsRepeatedIndicesInOrder() {
        var rng = new SplittableRandom(41);
        // the embedding's backward pass: rows of a 1000 x 48 table, 700 tokens, one of them 300 times (more than a chunk)
        int vocab = 1000, dim = 48, n = 700;
        int[] idx = new int[n];
        for (int j = 0; j < n; j++) idx[j] = j % 7 == 0 || j > 400 ? 17 : rng.nextInt(vocab);
        idx[n - 1] = vocab - 1;
        float[] dy = new float[n * dim];
        for (int i = 0; i < dy.length; i++) dy[i] = (float) ((rng.nextDouble() * 2 - 1) * Math.pow(2, rng.nextInt(-6, 7)));
        float[] base = new float[vocab * dim];
        for (int i = 0; i < base.length; i++) base[i] = (float) rng.nextDouble() - 0.5f;
        try (var g = Gather.create(device); var gdy = buffer(4L * dy.length); var gi = buffer(4L * n); var gdx = buffer(4L * base.length)) {
            for (boolean accumulate : new boolean[] {false, true}) {
                var dyv = StridedView.matrix(n, dim);
                var dxv = StridedView.matrix(vocab, dim);
                int[] got = run(e -> {
                    e.upload(gdy, words(bits(dy))).upload(gi, words(idx)).upload(gdx, words(bits(base)));
                    g.scatterAdd(e, gdy, dyv, 0, gi, StridedView.contiguous(n), gdx, dxv, accumulate);
                }, gdx);
                float[] want = base.clone();
                Gather.referenceScatterAdd(dy, dyv, 0, idx, StridedView.contiguous(n), want, dxv, accumulate);
                for (int i = 0; i < want.length; i++) assertEquals(Float.floatToRawIntBits(want[i]), got[i], "scatterAdd (accumulate " + accumulate + ") at " + i);
            }
            // along the middle dimension of a rank-3 view, through a transposed dy
            int d0 = 3, v = 11, d2 = 5, m = 40;
            int[] idx2 = new int[m];
            for (int j = 0; j < m; j++) idx2[j] = rng.nextInt(v);
            float[] dy2 = new float[d0 * m * d2];
            for (int i = 0; i < dy2.length; i++) dy2[i] = (float) rng.nextDouble();
            var dyv2 = StridedView.of(0, new long[] {d0, d2, m}, new long[] {(long) m * d2, 1, d2}).transpose(1, 2);   // logical d0 x m x d2
            var dxv2 = StridedView.of(0, new long[] {d0, v, d2}, new long[] {(long) v * d2, d2, 1});
            int[] got = run(e -> {
                e.upload(gdy, words(java.util.Arrays.copyOf(bits(dy2), dy.length))).upload(gi, words(java.util.Arrays.copyOf(idx2, n)));
                g.scatterAdd(e, gdy, dyv2, 1, gi, StridedView.contiguous(m), gdx, dxv2, false);
            }, gdx);
            float[] want = new float[d0 * v * d2];
            Gather.referenceScatterAdd(dy2, dyv2, 1, idx2, StridedView.contiguous(m), want, dxv2, false);
            for (int i = 0; i < want.length; i++) assertEquals(Float.floatToRawIntBits(want[i]), got[i], "scatterAdd along axis 1 at " + i);
            // an index past the table fails the submission
            var ex = assertThrows(org.dynamisengine.gpu.api.compute.ComputeException.class, () -> run(e -> {
                e.fill(gi, vocab);
                g.scatterAdd(e, gdy, StridedView.matrix(n, dim), 0, gi, StridedView.contiguous(n), gdx, StridedView.matrix(vocab, dim), false);
            }, gdx));
            assertEquals(org.dynamisengine.gpu.api.compute.ComputeException.Code.KERNEL_ERROR, ex.code());
        }
    }

    @Test void sigmoidAndTanhMeetTheirBounds() {
        // a sweep of x over [-90, 90] (dense near 0 and around the polynomial's edge), forward and backward, each element
        // within its contract against the double result; dy random
        var rng = new SplittableRandom(51);
        int n = 1 << 16;
        float[] x = new float[n], dy = new float[n];
        for (int i = 0; i < n; i++) {
            int k = rng.nextInt(4);
            x[i] = k == 0 ? (float) ((rng.nextDouble() * 2 - 1) * 90) : k == 1 ? (float) ((rng.nextDouble() * 2 - 1) * Math.pow(2, rng.nextInt(-30, 0)))
                    : k == 2 ? (float) (0.625 + (rng.nextDouble() - 0.5) * 0.01) * (rng.nextBoolean() ? 1 : -1) : (float) ((rng.nextDouble() * 2 - 1) * 8);
            dy[i] = (float) ((rng.nextDouble() * 2 - 1) * 4);
        }
        x[0] = Float.NaN; x[1] = Float.POSITIVE_INFINITY; x[2] = Float.NEGATIVE_INFINITY; x[3] = 0f; x[4] = -0f; x[5] = 0.625f; x[6] = -0.625f;
        var v = StridedView.contiguous(n);
        try (var ew = Elementwise.create(device); var gx = buffer(4L * n); var gd = buffer(4L * n); var gy = buffer(4L * n)) {
            for (int pass = 0; pass < 4; pass++) {
                final int ps = pass;
                int[] got = run(e -> {
                    e.upload(gx, words(bits(x))).upload(gd, words(bits(dy)));
                    switch (ps) {
                        case 0 -> ew.unary(e, Elementwise.Unary.SIGMOID, gx, v, gy, v);
                        case 1 -> ew.unary(e, Elementwise.Unary.TANH, gx, v, gy, v);
                        case 2 -> ew.binary(e, Elementwise.Binary.SIGMOID_BACKWARD, gx, v, gd, v, gy, v);
                        default -> ew.binary(e, Elementwise.Binary.TANH_BACKWARD, gx, v, gd, v, gy, v);
                    }
                }, gy);
                String name = new String[] {"sigmoid", "tanh", "sigmoid backward", "tanh backward"}[pass];
                long worst = 0;
                for (int i = 0; i < n; i++) {
                    double a = x[i];
                    double exact = switch (pass) {
                        case 0 -> 1 / (1 + Math.exp(-a));
                        case 1 -> Math.tanh(a);
                        case 2 -> { double s = 1 / (1 + Math.exp(-Math.abs(a))); yield dy[i] * s * Math.exp(-Math.abs(a)) * s; }
                        default -> { double c = Math.cosh(a); yield dy[i] / (c * c); }
                    };
                    float want = (float) exact, g = Float.intBitsToFloat(got[i]);
                    String where = name + " of " + x[i] + (pass >= 2 ? " (dy " + dy[i] + ")" : "") + ": exact " + exact + ", got " + g;
                    if (Double.isNaN(exact)) { assertTrue(Float.isNaN(g), where); continue; }
                    if (want == 0 || Math.abs(want) < 0x1p-126) { assertTrue(Math.abs(g) < 0x1p-125, where); continue; }   // flushes allowed
                    if (Math.abs(a) < 0x1p-126) continue;   // a subnormal input may flush to 0
                    long ulps = org.dynamisengine.gpu.api.compute.kit.AccuracyContract.ulpDistance(want, g);
                    double allowed = switch (pass) {
                        case 0 -> 6 + 2 * Math.abs(a);
                        case 1 -> 8 + 4 * Math.abs(a);
                        case 2 -> 9 + 2 * Math.abs(a);
                        default -> 9 + 4 * Math.abs(a);
                    };
                    boolean flushable = pass >= 2 && Math.exp(-(pass == 2 ? 1 : 2) * Math.abs(a)) < 0x1p-126;   // a subnormal exponential
                    assertTrue(ulps <= allowed || flushable && g == 0, where + ": " + ulps + " ULP, allowed " + allowed);
                    if (ulps <= allowed) worst = Math.max(worst, ulps);
                }
                System.out.println(name + " on " + device.adapter().name() + ": worst " + worst + " ULP");
            }
        }
    }
}
