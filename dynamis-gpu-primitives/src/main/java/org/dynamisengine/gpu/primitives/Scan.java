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
 * Fixed-order inclusive scans along the last dimension of an FP32 strided view of rank up to 4: {@code y[.., j] =
 * x[.., 0] op ... op x[.., j]}, for sum, max and min. The output has the input's shape.
 * <p>
 * <b>Order.</b> One group of 256 threads per row; the row goes in blocks of 256. Each block is scanned in a fixed tree
 * (Blelloch: up-sweep, then down-sweep) and combined with the carry of the blocks before it, so the result is the same
 * bits on every device and {@link #reference} reproduces it.
 *
 * @since 1.7.0
 */
public final class Scan implements AutoCloseable {
    /** The threads of a row's group, and the block length. */
    public static final int THREADS = 256;
    private final GpuDevice device;
    private final Map<Reduce.Op, PipelineSet> rows = new EnumMap<>(Reduce.Op.class);
    private final ParameterLayout layout = ParameterLayout.of(GemmF16.KERNELS, "scan_rows");
    private final DebugChecks debug;
    private boolean closed;

    private Scan(GpuDevice device) { this.device = device; this.debug = new DebugChecks(device); }

    /** Scans on {@code device}; pipelines compile on first use. */
    public static Scan create(GpuDevice device) { return new Scan(Checks.notNull(device, "device")); }

    private GpuComputePipeline pipeline(Reduce.Op op) {
        if (closed) throw new IllegalStateException("closed");
        return rows.computeIfAbsent(op, o -> PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("scan_rows",
                List.of(Access.READ, Access.WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of(),
                debug.constants(Map.of("OP", o.ordinal())))))).get(0);
    }

    /** Records the inclusive scan of each row of {@code x} into {@code y} (same shape). */
    public GpuCommandEncoder rows(GpuCommandEncoder encoder, Reduce.Op op, GpuBuffer x, StridedView xv, GpuBuffer y, StridedView yv) {
        Checks.notNull(op, "op");
        if (op == Reduce.Op.MEAN) throw new IllegalArgumentException("a scan takes SUM, MAX or MIN");
        var x4 = xv.withRank(4);
        var y4 = yv.withRank(4);
        for (int d = 0; d < 4; d++) {
            if (x4.extent(d) != y4.extent(d)) throw new IllegalArgumentException("y " + yv + " must have x's shape " + xv);
        }
        xv.check(x, 4, "x");
        yv.checkWritable(y, 4, "y");
        long rowCount = Views.rowCount(x4);
        if (rowCount == 0 || x4.extent(3) == 0) return encoder;
        var grid = Grid.linear(device.capabilities(), rowCount);
        byte[] params = layout.block().u32("rows", rowCount).u32("gridX", grid.gridX()).u32("xLength", DebugChecks.length(x))
                .u32("yLength", DebugChecks.length(y)).view("x", x4).view("y", y4).bytes();
        var pipeline = pipeline(op);
        debug.before(encoder);
        encoder.dispatch(pipeline, Map.of(0, x, 1, y, 2, debug.buffer()), params, grid.groups(), "scan." + op.name().toLowerCase());
        debug.after(encoder);
        return encoder;
    }

    /** The GPU's result on the CPU, bit for bit. */
    public static void reference(Reduce.Op op, float[] x, StridedView xv, float[] y, StridedView yv) {
        var x4 = xv.withRank(4);
        var y4 = yv.withRank(4);
        long rowCount = Views.rowCount(x4), n = x4.extent(3);
        float id = Reduce.identity(op);
        float[] buf = new float[THREADS], v = new float[THREADS];
        for (long row = 0; row < rowCount; row++) {
            float carry = id;
            for (long base = 0; base < n; base += THREADS) {
                for (int t = 0; t < THREADS; t++) {
                    long j = base + t;
                    v[t] = j < n ? x[(int) Views.at(x4, row, j)] : id;
                    buf[t] = v[t];
                }
                // up-sweep
                for (int d = 1; d < THREADS; d *= 2) {
                    for (int t = 0; t < THREADS; t++) {
                        int i = (t + 1) * d * 2 - 1;
                        if (i < THREADS) buf[i] = Reduce.combine(op, buf[i - d], buf[i]);
                    }
                }
                buf[THREADS - 1] = id;
                // down-sweep
                for (int d = THREADS / 2; d > 0; d /= 2) {
                    for (int t = 0; t < THREADS; t++) {
                        int i = (t + 1) * d * 2 - 1;
                        if (i < THREADS) {
                            int l = i - d;
                            float sum = Reduce.combine(op, buf[i], buf[l]);
                            buf[l] = buf[i];
                            buf[i] = sum;
                        }
                    }
                }
                float last = carry;
                for (int t = 0; t < THREADS; t++) {
                    float out = Reduce.combine(op, carry, Reduce.combine(op, buf[t], v[t]));
                    long j = base + t;
                    if (j < n) y[(int) Views.at(y4, row, j)] = out;
                    if (t == THREADS - 1) last = out;
                }
                carry = last;
            }
        }
    }

    /** The accuracy contract of {@code op}. */
    public static AccuracyContract contract(Reduce.Op op) {
        String formula = op == Reduce.Op.SUM
                ? "bit for bit Scan.reference; against the exact prefix sum, |err_j| <= (18 + ceil((j + 1) / 256)) u sum_{i <= j} |x_i| (first order)"
                : "bit for bit Scan.reference, which is the exact running " + op.name().toLowerCase();
        return new AccuracyContract("Scan " + op, List.of("FP32"), op == Reduce.Op.SUM ? "FP32" : "none", "FP32", Order.FIXED,
                new Bound(Bound.Kind.EXACT, 0, formula), "Scan.reference");
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        List<AutoCloseable> all = new java.util.ArrayList<>(rows.values());
        all.add(debug);
        PipelineSet.closeAll(all.toArray(new AutoCloseable[0]));
    }
}
