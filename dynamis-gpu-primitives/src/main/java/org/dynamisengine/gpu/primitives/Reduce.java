package org.dynamisengine.gpu.primitives;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.dynamisengine.gpu.api.compute.Access;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuComputePipeline;
import org.dynamisengine.gpu.api.compute.GpuDevice;
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
 *
 * @since 1.7.0
 */
public final class Reduce implements AutoCloseable {
    /** Reductions. */
    public enum Op { SUM, MAX, MIN }

    /** The threads of a row's group. */
    public static final int THREADS = 256;
    private final GpuDevice device;
    private final Map<Op, PipelineSet> rows = new EnumMap<>(Op.class);
    private PipelineSet argmax;
    private final ParameterLayout layout = ParameterLayout.of(GemmF16.KERNELS, "reduce_rows");
    private final ParameterLayout argmaxLayout = ParameterLayout.of(GemmF16.KERNELS, "reduce_argmax");
    private boolean closed;

    private Reduce(GpuDevice device) { this.device = device; }

    /** Reductions on {@code device}; pipelines compile on first use. */
    public static Reduce create(GpuDevice device) { return new Reduce(Checks.notNull(device, "device")); }

    private GpuComputePipeline pipeline(Op op) {
        if (closed) throw new IllegalStateException("closed");
        return rows.computeIfAbsent(op, o -> PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("reduce_rows",
                List.of(Access.READ, Access.WRITE), Set.of(GpuCapabilities.FP32), Set.of(), Map.of("OP", o.ordinal()))))).get(0);
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
        byte[] params = layout.block().u32("rows", rowCount).u32("gridX", grid.gridX()).view("x", x4).view("y", y4).bytes();
        return encoder.dispatch(pipeline(op), Map.of(0, x, 1, y), params, grid.groups(), "reduce." + op.name().toLowerCase());
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
            argmax = PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("reduce_argmax", List.of(Access.READ, Access.WRITE),
                    Set.of(GpuCapabilities.FP32), Set.of())));
        }
        var grid = Grid.linear(device.capabilities(), rowCount);
        byte[] params = argmaxLayout.block().u32("rows", rowCount).u32("gridX", grid.gridX()).view("x", x4).view("y", i4).bytes();
        return encoder.dispatch(argmax.get(0), Map.of(0, x, 1, indices), params, grid.groups(), "reduce.argmax");
    }

    static float identity(Op op) {
        return switch (op) {
            case SUM -> 0f;
            case MAX -> Float.NEGATIVE_INFINITY;
            case MIN -> Float.POSITIVE_INFINITY;
        };
    }

    static float combine(Op op, float a, float b) {
        return switch (op) {
            case SUM -> a + b;
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
            y[(int) Views.at(y4, row, 0)] = red[0];
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
        PipelineSet.closeAll(all.toArray(new AutoCloseable[0]));
    }
}
