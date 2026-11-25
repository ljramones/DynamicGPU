package org.dynamisengine.gpu.primitives;

import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.dynamisengine.gpu.api.compute.Access;
import org.dynamisengine.gpu.api.compute.BufferDescriptor;
import org.dynamisengine.gpu.api.compute.BufferUsage;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuComputePipeline;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.Placement;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract.Bound;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract.Order;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.Grid;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import org.dynamisengine.gpu.api.compute.kit.PipelineSet;
import org.dynamisengine.gpu.api.compute.kit.StridedView;

/**
 * Fixed-order reductions along the last dimension of an FP32 strided view of rank up to 4: sum, max, min and argmax.
 * The output has the input's shape without its last dimension.
 * <p>
 * <b>Order.</b> One group of 256 threads per row: thread {@code t} folds columns {@code t, t + 256, ...} in order from
 * the identity, then the 256 partials combine in a fixed tree (pairs {@code (t, t + w)} for {@code w = 128 .. 1}). So
 * a sum is the same bits on every device, and {@link #reference} reproduces it on the CPU. Max and min follow Java's
 * {@code Math.max} and {@code Math.min}. An empty row gives the identity: 0, -infinity or +infinity.
 * <p>
 * <b>Any axis</b> (1.14.0): {@link #axis} and {@link #argmaxAxis} reduce along any dimension by moving it last in the
 * view (no copy), and {@link #maxBackward} routes a max's gradient to the index argmax chose. MEAN is the sum divided by
 * the row length.
 *
 * @since 1.7.0
 */
public final class Reduce implements AutoCloseable {
    /** Reductions. */
    public enum Op {
        SUM, MAX, MIN,
        /** The fixed-order sum divided by the row length (1.14.0); an empty row gives NaN. */
        MEAN
    }

    /** The threads of a row's group. */
    public static final int THREADS = 256;
    private final GpuDevice device;
    private final Map<Op, PipelineSet> rows = new EnumMap<>(Op.class);
    private PipelineSet argmax, lse, maxBackward;
    private final ParameterLayout maxBackwardLayout = ParameterLayout.of(GemmF16.KERNELS, "reduce_max_backward");
    private final ParameterLayout layout = ParameterLayout.of(GemmF16.KERNELS, "reduce_rows");
    private final ParameterLayout argmaxLayout = ParameterLayout.of(GemmF16.KERNELS, "reduce_argmax");
    private final ParameterLayout lseLayout = ParameterLayout.of(GemmF16.KERNELS, "reduce_lse");
    private final ParameterLayout lseCombineLayout = ParameterLayout.of(GemmF16.KERNELS, "reduce_lse_combine");
    /** Elements of a row one log-sum-exp group takes (1.10.1). */
    static final int LSE_CHUNK = 4096;
    private final DebugChecks debug;
    private boolean closed;

    private Reduce(GpuDevice device) { this.device = device; this.debug = new DebugChecks(device); }

    /** Reductions on {@code device}; pipelines compile on first use. */
    public static Reduce create(GpuDevice device) { return new Reduce(Checks.notNull(device, "device")); }

    private GpuComputePipeline pipeline(Op op) {
        if (closed) throw new IllegalStateException("closed");
        return rows.computeIfAbsent(op, o -> PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("reduce_rows",
                List.of(Access.READ, Access.WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of(),
                debug.constants(Map.of("OP", o.ordinal())))))).get(0);
    }

    /** Records {@code y[..] = op over the last dimension of x[..]}. */
    public GpuCommandEncoder rows(GpuCommandEncoder encoder, Op op, GpuBuffer x, StridedView xv, GpuBuffer y, StridedView yv) {
        Checks.notNull(op, "op");
        var x4 = xv.withRank(4);
        var y4 = Views.rows(xv, x4, yv, "y");
        xv.check(x, 4, "x");
        yv.checkWritable(y, 4, "y");
        long rowCount = Views.rowCount(x4);
        if (rowCount == 0) return encoder;
        var grid = Grid.linear(device.capabilities(), rowCount);
        byte[] params = layout.block().u32("rows", rowCount).u32("gridX", grid.gridX()).u32("xLength", DebugChecks.length(x))
                .u32("yLength", DebugChecks.length(y)).view("x", x4).view("y", y4).bytes();
        var pipeline = pipeline(op);
        debug.before(encoder);
        encoder.dispatch(pipeline, Map.of(0, x, 1, y, 2, debug.buffer()), params, grid.groups(), "reduce." + op.name().toLowerCase());
        debug.after(encoder);
        return encoder;
    }

    /**
     * Records {@code y = op over dimension axis of x} (1.14.0): y has x's shape without that dimension, in x's order.
     * The reduction runs as {@link #rows} on x's view with the axis moved last, so its order and results are those of
     * {@link #reference} on that view.
     */
    public GpuCommandEncoder axis(GpuCommandEncoder encoder, Op op, GpuBuffer x, StridedView xv, int axis, GpuBuffer y, StridedView yv) {
        return rows(encoder, op, x, Views.moveToLast(xv, axis), y, yv);
    }

    /** Records argmax along dimension {@code axis} of x (1.14.0), as {@link #argmax} with the axis moved last. */
    public GpuCommandEncoder argmaxAxis(GpuCommandEncoder encoder, GpuBuffer x, StridedView xv, int axis, GpuBuffer indices, StridedView iv) {
        return argmax(encoder, x, Views.moveToLast(xv, axis), indices, iv);
    }

    /**
     * Records the gradient of a max along dimension {@code axis} (1.14.0): {@code dx = dy} at the index {@code indices}
     * holds for its row ({@link #argmaxAxis}'s output: the first largest element, or the first NaN) and 0 elsewhere.
     * dx has the input's shape; indices (uint32) and dy have it without the axis. Overwrites dx.
     */
    public GpuCommandEncoder maxBackward(GpuCommandEncoder encoder, GpuBuffer indices, StridedView iv, GpuBuffer dy, StridedView dyv,
            int axis, GpuBuffer dx, StridedView dxv) {
        if (closed) throw new IllegalStateException("closed");
        var dxp = Views.moveToLast(dxv, axis);
        var dx4 = dxp.withRank(4);
        var i4 = Views.rows(dxp, dx4, iv, "indices");
        var g4 = Views.rows(dxp, dx4, dyv, "dy");
        iv.check(indices, 4, "indices");
        dyv.check(dy, 4, "dy");
        dxv.checkWritable(dx, 4, "dx");
        long count = Checks.u32(dx4.elements(), "element count");
        if (count == 0) return encoder;
        if (maxBackward == null) {
            maxBackward = PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("reduce_max_backward",
                    List.of(Access.READ, Access.READ, Access.WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of(), debug.constants(Map.of()))));
        }
        var grid = Grid.forThreads(device.capabilities(), count, THREADS);
        var block = maxBackwardLayout.block().u32("count", count).u32("gridX", grid.gridX()).u32("iLength", DebugChecks.length(indices))
                .u32("dyLength", DebugChecks.length(dy)).u32("dxLength", DebugChecks.length(dx));
        Views.broadcastStrides(block, "i", i4);
        Views.broadcastStrides(block, "dy", g4);
        byte[] params = block.view("dx", dx4).bytes();
        debug.before(encoder);
        encoder.dispatch(maxBackward.get(0), Map.of(0, indices, 1, dy, 2, dx, 3, debug.buffer()), params, grid.groups(), "reduce.max.backward");
        debug.after(encoder);
        return encoder;
    }

    /** {@link #maxBackward} on the CPU: dx over x's shape (with the axis moved last), from argmax's indices and dy. */
    public static void referenceMaxBackward(int[] indices, StridedView iv, float[] dy, StridedView dyv, int axis, float[] dx, StridedView dxv) {
        var dxp = Views.moveToLast(dxv, axis);
        var dx4 = dxp.withRank(4);
        var i4 = Views.rows(dxp, dx4, iv, "indices");
        var g4 = Views.rows(dxp, dx4, dyv, "dy");
        long rowCount = Views.rowCount(dx4), n = dx4.extent(3);
        for (long row = 0; row < rowCount; row++) {
            int idx = indices[(int) Views.at(i4, row, 0)];
            float g = dy[(int) Views.at(g4, row, 0)];
            for (long j = 0; j < n; j++) dx[(int) Views.at(dx4, row, j)] = j == idx ? g : 0f;
        }
    }

    /**
     * Records the index of the first largest element of each row into {@code indices} (uint32). NaN counts as the
     * largest. Rows must not be empty.
     */
    public GpuCommandEncoder argmax(GpuCommandEncoder encoder, GpuBuffer x, StridedView xv, GpuBuffer indices, StridedView iv) {
        if (closed) throw new IllegalStateException("closed");
        var x4 = xv.withRank(4);
        if (x4.extent(3) == 0) throw new IllegalArgumentException("argmax of an empty row");
        var i4 = Views.rows(xv, x4, iv, "indices");
        xv.check(x, 4, "x");
        iv.checkWritable(indices, 4, "indices");
        long rowCount = Views.rowCount(x4);
        if (rowCount == 0) return encoder;
        if (argmax == null) {
            argmax = PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("reduce_argmax",
                    List.of(Access.READ, Access.WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of(), debug.constants(Map.of()))));
        }
        var grid = Grid.linear(device.capabilities(), rowCount);
        byte[] params = argmaxLayout.block().u32("rows", rowCount).u32("gridX", grid.gridX()).u32("xLength", DebugChecks.length(x))
                .u32("yLength", DebugChecks.length(indices)).view("x", x4).view("y", i4).bytes();
        debug.before(encoder);
        encoder.dispatch(argmax.get(0), Map.of(0, x, 1, indices, 2, debug.buffer()), params, grid.groups(), "reduce.argmax");
        debug.after(encoder);
        return encoder;
    }

    /**
     * Records {@code y[..] = log(sum exp(x[..]))} over the last dimension of x (1.10.0): the log-sum-exp of each row,
     * computed as {@code max + log(sum exp(x - max))} with the max and the sum in fixed trees. A row of -infinity gives
     * -infinity, a NaN NaN. exp and log are the device's, so it is bounded ({@link #logSumExpContract}). Since 1.10.1 a
     * row longer than 4096 is split into chunks of 4096 across groups, and a second dispatch combines their maxima and
     * sums ({@code M + log(sum s exp(m - M))}, in a fixed tree): a row of 128256 took 124 us in one group on an M4 Max.
     */
    public GpuCommandEncoder logSumExp(GpuCommandEncoder encoder, GpuBuffer x, StridedView xv, GpuBuffer y, StridedView yv) {
        if (closed) throw new IllegalStateException("closed");
        var x4 = xv.withRank(4);
        var y4 = Views.rows(xv, x4, yv, "y");
        xv.check(x, 4, "x");
        yv.checkWritable(y, 4, "y");
        long rowCount = Views.rowCount(x4);
        if (rowCount == 0) return encoder;
        if (lse == null) {
            lse = PipelineSet.compile(device, List.of(
                    GemmF16.KERNELS.pipeline("reduce_lse", List.of(Access.READ, Access.WRITE, Access.READ_WRITE, Access.READ_WRITE),
                            Set.of(GpuCapabilities.FP32), Set.of(), debug.constants(Map.of())),
                    GemmF16.KERNELS.pipeline("reduce_lse_combine", List.of(Access.READ, Access.WRITE, Access.READ_WRITE),
                            Set.of(GpuCapabilities.FP32), Set.of(), debug.constants(Map.of()))));
        }
        long chunks = Math.max(1, Math.ceilDiv(x4.extent(3), LSE_CHUNK));
        var grid = Grid.linear(device.capabilities(), rowCount * chunks);
        byte[] params = lseLayout.block().u32("rows", rowCount).u32("gridX", grid.gridX()).u32("xLength", DebugChecks.length(x))
                .u32("yLength", DebugChecks.length(y)).u32("chunks", chunks).u32("chunkLen", LSE_CHUNK).view("x", x4).view("y", y4).bytes();
        debug.before(encoder);
        // per dispatch: closing the handle keeps it alive for the recorded commands
        try (var partial = device.allocate(new BufferDescriptor(Math.max(16, 8 * rowCount * chunks), EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE)
                .withLabel("lse-partial"))) {
            encoder.dispatch(lse.get(0), Map.of(0, x, 1, y, 2, debug.buffer(), 3, partial), params, grid.groups(), "reduce.lse");
            if (chunks > 1) {
                var rowsGrid = Grid.linear(device.capabilities(), rowCount);
                byte[] combine = lseCombineLayout.block().u32("rows", rowCount).u32("gridX", rowsGrid.gridX()).u32("yLength", DebugChecks.length(y))
                        .u32("chunks", chunks).view("y", y4).bytes();
                encoder.dispatch(lse.get(1), Map.of(0, partial, 1, y, 2, debug.buffer()), combine, rowsGrid.groups(), "reduce.lse.combine");
            }
        }
        debug.after(encoder);
        return encoder;
    }

    /** The log-sum-exp of {@code x[off .. off + n)} in double: the exact value the GPU's result is held to. @since 1.10.0 */
    public static double logSumExpExact(float[] x, int off, int n) {
        double mx = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < n; j++) { if (Float.isNaN(x[off + j])) return Double.NaN; mx = Math.max(mx, x[off + j]); }
        if (mx == Double.NEGATIVE_INFINITY) return mx;
        double s = 0;
        for (int j = 0; j < n; j++) s += Math.exp(x[off + j] - mx);
        return mx + Math.log(s);
    }

    /**
     * The contract of {@link #logSumExp}: with {@code s = sum exp(x - max)} in [1, n],
     * {@code |y - exact| <= |max| 2^-24 + ((n + 10 + 2 max|x - max|) 2^-24) + 4 ulp of log(s)} (first order). @since 1.10.0
     */
    public static AccuracyContract logSumExpContract() {
        return new AccuracyContract("Reduce LOGSUMEXP", List.of("FP32"), "FP32: max, then sum exp(x - max), fixed trees", "FP32", Order.DEVICE,
                Bound.formula("|y - exact| <= 2^-24 |max| + (n + 10 + 2 max|x - max|) 2^-24 + 4 ulp(log s), s = sum exp(x - max) (first order)"),
                "Reduce.logSumExpExact");
    }

    static float identity(Op op) {
        return switch (op) {
            case SUM, MEAN -> 0f;
            case MAX -> Float.NEGATIVE_INFINITY;
            case MIN -> Float.POSITIVE_INFINITY;
        };
    }

    static float combine(Op op, float a, float b) {
        return switch (op) {
            case SUM, MEAN -> a + b;
            case MAX -> Math.max(a, b);
            case MIN -> Math.min(a, b);
        };
    }

    /** The GPU's result on the CPU, bit for bit: {@code x} and {@code y} hold the buffers' FP32 elements. */
    public static void reference(Op op, float[] x, StridedView xv, float[] y, StridedView yv) {
        var x4 = xv.withRank(4);
        var y4 = Views.rows(xv, x4, yv, "y");
        long rowCount = Views.rowCount(x4), n = x4.extent(3);
        float[] red = new float[THREADS];
        for (long row = 0; row < rowCount; row++) {
            for (int t = 0; t < THREADS; t++) {
                float acc = identity(op);
                for (long j = t; j < n; j += THREADS) acc = combine(op, acc, x[(int) Views.at(x4, row, j)]);
                red[t] = acc;
            }
            for (int w = THREADS / 2; w > 0; w /= 2) {
                for (int t = 0; t < w; t++) red[t] = combine(op, red[t], red[t + w]);
            }
            y[(int) Views.at(y4, row, 0)] = op == Op.MEAN ? red[0] / n : red[0];
        }
    }

    /** The GPU's argmax on the CPU: the first index of the largest value, NaN counting as the largest. */
    public static void referenceArgmax(float[] x, StridedView xv, int[] indices, StridedView iv) {
        var x4 = xv.withRank(4);
        var i4 = Views.rows(xv, x4, iv, "indices");
        long rowCount = Views.rowCount(x4), n = x4.extent(3);
        for (long row = 0; row < rowCount; row++) {
            int best = 0;
            for (int j = 1; j < n; j++) {
                float v = x[(int) Views.at(x4, row, j)], b = x[(int) Views.at(x4, row, best)];
                if (!Float.isNaN(b) && (Float.isNaN(v) || v > b)) best = j;
            }
            indices[(int) Views.at(i4, row, 0)] = best;
        }
    }

    /** The accuracy contract of {@code op}. */
    public static AccuracyContract contract(Op op) {
        if (op == Op.MEAN) {
            return new AccuracyContract("Reduce MEAN", List.of("FP32"), "FP32", "FP32", Order.FIXED,
                    Bound.ulps(2.5, "Reduce.reference's fixed-order sum (bit for bit), then the device's division by n: within 2.5 ULP of "
                            + "the correctly rounded quotient on Vulkan, correctly rounded on Metal"), "Reduce.reference");
        }
        String formula = op == Op.SUM
                ? "bit for bit Reduce.reference; against the exact sum, |err| <= (ceil(n / 256) + 8) u sum |x| (first order)"
                : "bit for bit Reduce.reference, which is the exact " + op.name().toLowerCase() + " (Java Math semantics)";
        return new AccuracyContract("Reduce " + op, List.of("FP32"), op == Op.SUM ? "FP32" : "none", "FP32", Order.FIXED,
                new Bound(Bound.Kind.EXACT, 0, formula), "Reduce.reference");
    }

    /** The accuracy contract of argmax: exact. */
    public static AccuracyContract argmaxContract() {
        return new AccuracyContract("Reduce ARGMAX", List.of("FP32"), "none", "UINT32", Order.FIXED,
                new Bound(Bound.Kind.EXACT, 0, "the first index of the largest value; NaN counts as the largest"),
                "Reduce.referenceArgmax");
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        List<AutoCloseable> all = new java.util.ArrayList<>(rows.values());
        all.add(argmax);
        all.add(lse);
        all.add(maxBackward);
        all.add(debug);
        PipelineSet.closeAll(all.toArray(new AutoCloseable[0]));
    }
}
