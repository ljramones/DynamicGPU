package org.dynamisengine.gpu.primitives;

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
 * Gradient clipping by global norm (1.14.0), on the GPU with no host round trip:
 * <ul>
 * <li>{@link #squaredNorm}: the sum of the squares of every element of a list of FP32 tensors, in a fixed order, into
 * a one-element buffer: bit for bit {@link #referenceSquaredNorm} on every device.</li>
 * <li>{@link #clip}: scales every tensor in place by {@code maxNorm / sqrt(s)} when {@code sqrt(s) > maxNorm}, s read
 * from that buffer, and leaves them unchanged otherwise (also when s is NaN), as Vectrix's {@code Optim.clipFactor}.</li>
 * </ul>
 * <b>Order.</b> Each tensor is cut into chunks of {@value #CHUNK} elements (row-major); a group sums a chunk's squares
 * (thread t folds elements t, t + 256, ... in order, then a fixed tree) into a partial; one group then sums all the
 * partials, tensor after tensor, the same way.
 *
 * @since 1.14.0
 */
public final class GradientClip implements AutoCloseable {
    /** A tensor: a buffer and its view (FP32 elements). */
    public record Part(GpuBuffer buffer, StridedView view) {
        public Part { Checks.notNull(buffer, "buffer"); Checks.notNull(view, "view"); }
    }

    /** Elements of a tensor one group sums. */
    public static final int CHUNK = 4096;
    private static final int THREADS = 256;
    private final GpuDevice device;
    private final ParameterLayout sumLayout = ParameterLayout.of(GemmF16.KERNELS, "clip_sumsq");
    private final ParameterLayout combineLayout = ParameterLayout.of(GemmF16.KERNELS, "clip_combine");
    private final ParameterLayout scaleLayout = ParameterLayout.of(GemmF16.KERNELS, "clip_scale");
    private final DebugChecks debug;
    private PipelineSet norm, scale;
    private boolean closed;

    private GradientClip(GpuDevice device) { this.device = device; this.debug = new DebugChecks(device); }

    /** Clipping on {@code device}; pipelines compile on first use. */
    public static GradientClip create(GpuDevice device) { return new GradientClip(Checks.notNull(device, "device")); }

    private static long chunks(StridedView v) { return Math.ceilDiv(v.elements(), CHUNK); }

    /** Records {@code out[0] = sum over the parts of sum x^2} (FP32), in the fixed order above. */
    public GpuCommandEncoder squaredNorm(GpuCommandEncoder encoder, List<Part> parts, GpuBuffer out) {
        if (closed) throw new IllegalStateException("closed");
        Checks.notNull(parts, "parts");
        Checks.range(out, 0, 1, 4, "out");
        long total = 0;
        for (var part : parts) {
            part.view().check(part.buffer(), 4, "part");
            Checks.u32(part.view().elements(), "element count");
            total += chunks(part.view());
        }
        Checks.u32(total, "chunk count");
        if (norm == null) {
            norm = PipelineSet.compile(device, List.of(
                    GemmF16.KERNELS.pipeline("clip_sumsq", List.of(Access.READ, Access.READ_WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32),
                            Set.of(), debug.constants(Map.of())),
                    GemmF16.KERNELS.pipeline("clip_combine", List.of(Access.READ, Access.WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32),
                            Set.of(), debug.constants(Map.of()))));
        }
        debug.before(encoder);
        // per call: closing the handle keeps it alive for the recorded commands
        try (var partial = device.allocate(new BufferDescriptor(4 * Math.max(1, total), EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE)
                .withLabel("clip-partial"))) {
            long base = 0;
            for (var part : parts) {
                long n = chunks(part.view());
                if (n == 0) continue;
                var grid = Grid.linear(device.capabilities(), n);
                byte[] params = sumLayout.block().u32("count", part.view().elements()).u32("gridX", grid.gridX()).u32("chunks", n).u32("base", base)
                        .u32("xLength", DebugChecks.length(part.buffer())).u32("pLength", total).view("x", part.view().withRank(4)).bytes();
                encoder.dispatch(norm.get(0), Map.of(0, part.buffer(), 1, partial, 2, debug.buffer()), params, grid.groups(), "clip.sumsq");
                base += n;
            }
            byte[] params = combineLayout.block().u32("n", total).u32("pLength", total).bytes();
            encoder.dispatch(norm.get(1), Map.of(0, partial, 1, out, 2, debug.buffer()), params, new org.dynamisengine.gpu.api.compute.Dimensions(1, 1, 1),
                    "clip.combine");
        }
        debug.after(encoder);
        return encoder;
    }

    /** Records the in-place scaling of every part by the clip factor of {@code sumSq[0]} (a {@link #squaredNorm} result). */
    public GpuCommandEncoder clip(GpuCommandEncoder encoder, GpuBuffer sumSq, float maxNorm, List<Part> parts) {
        if (closed) throw new IllegalStateException("closed");
        Checks.notNull(parts, "parts");
        Checks.range(sumSq, 0, 1, 4, "sumSq");
        if (!(maxNorm > 0) || Float.isInfinite(maxNorm)) throw new IllegalArgumentException("maxNorm must be positive and finite: " + maxNorm);
        if (scale == null) {
            scale = PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("clip_scale", List.of(Access.READ, Access.READ_WRITE, Access.READ_WRITE),
                    Set.of(GpuCapabilities.FP32), Set.of(), debug.constants(Map.of()))));
        }
        debug.before(encoder);
        for (var part : parts) {
            part.view().checkWritable(part.buffer(), 4, "part");
            long count = Checks.u32(part.view().elements(), "element count");
            if (count == 0) continue;
            var grid = Grid.forThreads(device.capabilities(), count, THREADS);
            byte[] params = scaleLayout.block().u32("count", count).u32("gridX", grid.gridX()).f32("maxNorm", maxNorm)
                    .u32("xLength", DebugChecks.length(part.buffer())).view("x", part.view().withRank(4)).bytes();
            encoder.dispatch(scale.get(0), Map.of(0, sumSq, 1, part.buffer(), 2, debug.buffer()), params, grid.groups(), "clip.scale");
        }
        debug.after(encoder);
        return encoder;
    }

    /** {@link #squaredNorm} on the CPU, bit for bit: {@code values.get(i)} holds part i's buffer as floats. */
    public static float referenceSquaredNorm(List<float[]> values, List<StridedView> views) {
        int parts = values.size();
        java.util.ArrayList<Float> partials = new java.util.ArrayList<>();
        float[] red = new float[THREADS];
        for (int k = 0; k < parts; k++) {
            float[] x = values.get(k);
            var v4 = views.get(k).withRank(4);
            long n = v4.elements();
            for (long first = 0; first < n; first += CHUNK) {
                long end = Math.min(n, first + CHUNK);
                for (int t = 0; t < THREADS; t++) {
                    float acc = 0f;
                    for (long i = first + t; i < end; i += THREADS) {
                        float v = x[(int) linear(v4, i)];
                        acc = acc + v * v;
                    }
                    red[t] = acc;
                }
                partials.add(tree(red));
            }
        }
        for (int t = 0; t < THREADS; t++) {
            float acc = 0f;
            for (int i = t; i < partials.size(); i += THREADS) acc = acc + partials.get(i);
            red[t] = acc;
        }
        return tree(red);
    }

    private static float tree(float[] red) {
        for (int w = THREADS / 2; w > 0; w /= 2) for (int t = 0; t < w; t++) red[t] = red[t] + red[t + w];
        return red[0];
    }

    private static long linear(StridedView v4, long i) {
        long i3 = i % v4.extent(3); i /= v4.extent(3);
        long i2 = i % v4.extent(2); i /= v4.extent(2);
        long i1 = i % v4.extent(1), i0 = i / v4.extent(1);
        return v4.offset() + i0 * v4.stride(0) + i1 * v4.stride(1) + i2 * v4.stride(2) + i3 * v4.stride(3);
    }

    /** The contract of {@link #squaredNorm}: fixed order, bit for bit its reference. */
    public static AccuracyContract squaredNormContract() {
        return new AccuracyContract("GradientClip squaredNorm", List.of("FP32"), "FP32: chunks of 4096, fixed trees", "FP32", Order.FIXED,
                new Bound(Bound.Kind.EXACT, 0, "bit for bit GradientClip.referenceSquaredNorm; against the exact sum of squares, "
                        + "|err| <= (25 + ceil(p / 256) + 8) u s, p the number of partials (first order)"), "GradientClip.referenceSquaredNorm");
    }

    /** The contract of {@link #clip}: the device's sqrt and division, then one rounded product. */
    public static AccuracyContract clipContract() {
        return new AccuracyContract("GradientClip clip", List.of("FP32"), "none", "FP32", Order.DEVICE,
                Bound.ulps(9, "x maxNorm / sqrt(s): sqrt within 5 ULP (Vulkan), the division within 2.5, the product correctly rounded; "
                        + "x unchanged when sqrt(s) <= maxNorm or s is NaN"), "");
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        PipelineSet.closeAll(norm, scale, debug);
    }
}
