package org.dynamisengine.gpu.primitives;

import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.dynamisengine.gpu.api.compute.Access;
import org.dynamisengine.gpu.api.compute.BufferDescriptor;
import org.dynamisengine.gpu.api.compute.BufferUsage;
import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.KernelErrors;
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
 * Gather and scatter-add along one dimension of strided views (1.14.0), with the indices (uint32) in a GPU buffer:
 * <ul>
 * <li>{@link #gather}: {@code y[.., j, ..] = x[.., indices[j], ..]}. y has x's shape with that dimension's extent
 * replaced by the number of indices. 32-bit elements of any type, copied exactly.</li>
 * <li>{@link #scatterAdd}: its gradient, {@code dx[.., k, ..] (+)= sum of dy[.., j, ..] over indices[j] == k}, FP32. A
 * repeated index sums its rows in increasing j, with no float atomics, so the result does not depend on scheduling:
 * bit for bit {@link #referenceScatterAdd} on every device. The embedding's backward pass is scatterAdd along the
 * table's rows.</li>
 * </ul>
 * An index outside the dimension fails the submission with {@link ComputeException.Code#KERNEL_ERROR}.
 * <p>
 * <b>Cost.</b> scatterAdd runs two dispatches: integer-atomic counts of each index (exact in any order), then one
 * group per row k of dx, which walks the indices in order until it has found all of k's. A row no index names only
 * writes zeros (or nothing, accumulating).
 *
 * @since 1.14.0
 */
public final class Gather implements AutoCloseable {
    private static final int THREADS = 256;
    private final GpuDevice device;
    private final ParameterLayout gatherLayout = ParameterLayout.of(GemmF16.KERNELS, "gather");
    private final ParameterLayout countLayout = ParameterLayout.of(GemmF16.KERNELS, "scatter_count");
    private final ParameterLayout addLayout = ParameterLayout.of(GemmF16.KERNELS, "scatter_add");
    private final DebugChecks debug;
    private PipelineSet gather, scatter;
    private boolean closed;

    private Gather(GpuDevice device) { this.device = device; this.debug = new DebugChecks(device); }

    /** Gather and scatter-add on {@code device}; pipelines compile on first use. */
    public static Gather create(GpuDevice device) { return new Gather(Checks.notNull(device, "device")); }

    /** Checks that {@code a} and {@code b} (axis moved last) agree on every dimension but the last. */
    private static void sameOuter(StridedView a, StridedView b, String what) {
        if (a.rank() != b.rank()) throw new IllegalArgumentException(what + ": ranks differ, " + a + " and " + b);
        for (int d = 0; d < a.rank() - 1; d++) {
            if (a.extent(d) != b.extent(d)) throw new IllegalArgumentException(what + ": shapes differ outside the axis, " + a + " and " + b);
        }
    }

    private static void indexView(StridedView iv) {
        if (iv.rank() != 1) throw new IllegalArgumentException("indices must be a rank-1 view: " + iv);
    }

    /** Records {@code y[.., j, ..] = x[.., indices[j], ..]} along dimension {@code axis} of x and y. */
    public GpuCommandEncoder gather(GpuCommandEncoder encoder, GpuBuffer x, StridedView xv, int axis, GpuBuffer indices, StridedView iv,
            GpuBuffer y, StridedView yv) {
        if (closed) throw new IllegalStateException("closed");
        indexView(iv);
        var xp = Views.moveToLast(xv, axis);
        var yp = Views.moveToLast(yv, axis);
        sameOuter(xp, yp, "gather");
        if (yp.extent(yp.rank() - 1) != iv.extent(0)) throw new IllegalArgumentException("y's axis must have one element per index: " + yv);
        xv.check(x, 4, "x");
        iv.check(indices, 4, "indices");
        yv.checkWritable(y, 4, "y");
        var x4 = xp.withRank(4);
        var y4 = yp.withRank(4);
        long count = Checks.u32(y4.elements(), "element count");
        if (count == 0) return encoder;
        if (gather == null) {
            gather = PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("gather",
                    List.of(Access.READ, Access.READ, Access.WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of(), debug.constants(Map.of()))));
        }
        var grid = Grid.forThreads(device.capabilities(), count, THREADS);
        byte[] params = gatherLayout.block().u32("count", count).u32("gridX", grid.gridX()).u32("extent", Checks.u32(x4.extent(3), "extent"))
                .u32("xLength", DebugChecks.length(x)).u32("yLength", DebugChecks.length(y)).u32("iLength", DebugChecks.length(indices))
                .u32("iOffset", iv.offset()).u32("iStride", iv.stride(0)).view("x", x4).view("y", y4).bytes();
        try (var errors = device.allocate(KernelErrors.descriptor().withLabel("gather-errors"))) {
            encoder.resetErrors(errors);
            encoder.dispatch(gather.get(0), Map.of(0, x, 1, indices, 2, y, 3, errors), params, grid.groups(), "gather");
            encoder.checkErrors(errors);
        }
        return encoder;
    }

    /**
     * Records {@code dx[.., k, ..] = sum of dy[.., j, ..] over indices[j] == k} along dimension {@code axis}, in
     * increasing j; with {@code accumulate} the sum is added to dx's values (in the same order: {@code ((dx + dy[j1]) +
     * dy[j2]) ...}), otherwise it starts from +0 and a k no index names gets +0. FP32. dy has dx's shape with the axis's
     * extent replaced by the number of indices.
     */
    public GpuCommandEncoder scatterAdd(GpuCommandEncoder encoder, GpuBuffer dy, StridedView dyv, int axis, GpuBuffer indices, StridedView iv,
            GpuBuffer dx, StridedView dxv, boolean accumulate) {
        if (closed) throw new IllegalStateException("closed");
        indexView(iv);
        var gp = Views.moveToLast(dyv, axis);
        var xp = Views.moveToLast(dxv, axis);
        sameOuter(gp, xp, "scatterAdd");
        if (gp.extent(gp.rank() - 1) != iv.extent(0)) throw new IllegalArgumentException("dy's axis must have one element per index: " + dyv);
        dyv.check(dy, 4, "dy");
        iv.check(indices, 4, "indices");
        dxv.checkWritable(dx, 4, "dx");
        var g4 = gp.withRank(4);
        var x4 = xp.withRank(4);
        long extent = Checks.u32(x4.extent(3), "extent"), n = Checks.u32(iv.extent(0), "index count");
        Checks.u32(x4.elements(), "element count");
        if (extent == 0 || x4.elements() == 0 || (accumulate && n == 0)) return encoder;
        if (scatter == null) {
            scatter = PipelineSet.compile(device, List.of(
                    GemmF16.KERNELS.pipeline("scatter_count", List.of(Access.READ, Access.READ_WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32),
                            Set.of(), debug.constants(Map.of())),
                    GemmF16.KERNELS.pipeline("scatter_add", List.of(Access.READ, Access.READ, Access.READ, Access.READ_WRITE, Access.READ_WRITE),
                            Set.of(GpuCapabilities.FP32), Set.of(), debug.constants(Map.of()))));
        }
        var groups = Grid.linear(device.capabilities(), extent);
        byte[] add = addLayout.block().u32("extent", extent).u32("gridX", groups.gridX()).u32("n", n).bool("accumulate", accumulate)
                .u32("dxLength", DebugChecks.length(dx)).u32("dyLength", DebugChecks.length(dy)).u32("iOffset", iv.offset())
                .u32("iStride", iv.stride(0)).view("dy", g4).view("dx", x4).bytes();
        // per dispatch: closing the handles keeps them alive for the recorded commands
        try (var counts = device.allocate(new BufferDescriptor(4 * Math.max(1, extent), EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_DESTINATION),
                Placement.PRIVATE).withLabel("scatter-counts"));
                var errors = device.allocate(KernelErrors.descriptor().withLabel("scatter-errors"))) {
            encoder.resetErrors(errors);
            encoder.fill(counts, 0);
            if (n > 0) {
                var threads = Grid.forThreads(device.capabilities(), n, THREADS);
                byte[] count = countLayout.block().u32("n", n).u32("gridX", threads.gridX()).u32("extent", extent)
                        .u32("iLength", DebugChecks.length(indices)).u32("iOffset", iv.offset()).u32("iStride", iv.stride(0)).bytes();
                encoder.dispatch(scatter.get(0), Map.of(0, indices, 1, counts, 2, errors), count, threads.groups(), "scatter.count");
            }
            encoder.dispatch(scatter.get(1), Map.of(0, dy, 1, indices, 2, counts, 3, dx, 4, errors), add, groups.groups(), "scatter.add");
            encoder.checkErrors(errors);
        }
        return encoder;
    }

    /** {@link #gather} on the CPU, on 32-bit words. */
    public static void referenceGather(int[] x, StridedView xv, int axis, int[] indices, StridedView iv, int[] y, StridedView yv) {
        var x4 = Views.moveToLast(xv, axis).withRank(4);
        var y4 = Views.moveToLast(yv, axis).withRank(4);
        long rows = Views.rowCount(y4), n = y4.extent(3);
        for (long r = 0; r < rows; r++) {
            for (long j = 0; j < n; j++) {
                int k = indices[(int) (iv.offset() + j * iv.stride(0))];
                y[(int) Views.at(y4, r, j)] = x[(int) Views.at(x4, r, k)];
            }
        }
    }

    /** {@link #scatterAdd} on the CPU, bit for bit: the rows of each k added in increasing j. */
    public static void referenceScatterAdd(float[] dy, StridedView dyv, int axis, int[] indices, StridedView iv, float[] dx, StridedView dxv,
            boolean accumulate) {
        var g4 = Views.moveToLast(dyv, axis).withRank(4);
        var x4 = Views.moveToLast(dxv, axis).withRank(4);
        long rows = Views.rowCount(x4), extent = x4.extent(3), n = iv.extent(0);
        for (long r = 0; r < rows; r++) {
            for (long k = 0; k < extent; k++) {
                int xi = (int) Views.at(x4, r, k);
                float acc = accumulate ? dx[xi] : 0f;
                for (long j = 0; j < n; j++) {
                    if (indices[(int) (iv.offset() + j * iv.stride(0))] == k) acc = acc + dy[(int) Views.at(g4, r, j)];
                }
                dx[xi] = acc;
            }
        }
    }

    /** The contract of {@link #gather}: exact. */
    public static AccuracyContract gatherContract() {
        return new AccuracyContract("Gather", List.of("32-bit words", "UINT32 indices"), "none", "the gathered words", Order.FIXED,
                new Bound(Bound.Kind.EXACT, 0, "the input's elements, copied"), "Gather.referenceGather");
    }

    /** The contract of {@link #scatterAdd}: fixed order, bit for bit its reference. */
    public static AccuracyContract scatterAddContract() {
        return new AccuracyContract("Gather SCATTER_ADD", List.of("FP32", "UINT32 indices"), "FP32, in increasing index position", "FP32",
                Order.FIXED, new Bound(Bound.Kind.EXACT, 0, "bit for bit Gather.referenceScatterAdd; against the exact sum, "
                        + "|err| <= (m - 1) u sum |dy| over a k's m rows (first order)"), "Gather.referenceScatterAdd");
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        PipelineSet.closeAll(gather, scatter, debug);
    }
}
