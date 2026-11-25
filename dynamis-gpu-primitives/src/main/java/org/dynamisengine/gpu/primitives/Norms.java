package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.Grid;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import org.dynamisengine.gpu.api.compute.kit.PipelineSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * RMSNorm and LayerNorm of each row (1.10.0), optionally after a fused residual add, with FP32 or FP16 activations
 * and output:
 * <pre>
 *     v = x                       or, with a residual,   h = in(x + res), written to h, and v = h
 *     RMSNorm:   y = v * rsqrt(sum v^2 / n + eps) * w
 *     LayerNorm: y = (v - mean) * rsqrt(sum (v - mean)^2 / n + eps) * w [+ b]
 * </pre>
 * computed in FP32 and rounded once to the output type. w and b are FP32 vectors of n. The residual stream h may be x
 * itself (updated in place). Each row is one group of 256 threads, with the sums combined in a fixed tree, so a
 * device's results do not depend on scheduling; rsqrt and the division by n are the device's, so they are bounded
 * ({@link #contract}). h is exact: {@code in(x + res)}, one rounding.
 */
public final class Norms implements AutoCloseable {
    /** Which norm. */
    public enum Kind { RMS, LAYER }

    /**
     * The rows: {@code rows} of {@code n}, leading dimensions in elements for x, the residual, h and y.
     *
     * @param eps added to the mean square (or variance) before the reciprocal square root
     */
    public record Shape(int rows, int n, int ldx, int ldr, int ldh, int ldy, float eps) {
        public Shape {
            if (rows <= 0 || n <= 0) throw new IllegalArgumentException("dimensions must be positive: " + rows + "x" + n);
            if (ldx < n || ldr < n || ldh < n || ldy < n) throw new IllegalArgumentException("leading dimension smaller than a row");
            if (!(eps >= 0) || Float.isInfinite(eps)) throw new IllegalArgumentException("eps must be finite and not negative: " + eps);
        }
        /** Densely packed rows. */
        public static Shape of(int rows, int n, float eps) { return new Shape(rows, n, n, n, n, n, eps); }
        long elements(int ld) { return (long) (rows - 1) * ld + n; }
    }

    /**
     * The backward pass's rows (1.12.0): {@code rows} of {@code n}, leading dimensions in elements for v (the values
     * the forward pass normalized: x, or h after the residual add), dy and dv.
     */
    public record BackwardShape(int rows, int n, int ldv, int lddy, int lddv, float eps) {
        public BackwardShape {
            if (rows <= 0 || n <= 0) throw new IllegalArgumentException("dimensions must be positive: " + rows + "x" + n);
            if (ldv < n || lddy < n || lddv < n) throw new IllegalArgumentException("leading dimension smaller than a row");
            if (!(eps >= 0) || Float.isInfinite(eps)) throw new IllegalArgumentException("eps must be finite and not negative: " + eps);
        }
        /** Densely packed rows. */
        public static BackwardShape of(int rows, int n, float eps) { return new BackwardShape(rows, n, n, n, n, eps); }
        long elements(int ld) { return (long) (rows - 1) * ld + n; }
    }

    /** Rows per group of the backward pass: each group's weight-gradient partials cover this many. */
    static final int BACKWARD_ROWS = 16;

    private final GpuDevice device;
    private final GpuComputePipeline pipeline;
    private PipelineSet backward;
    private final Precision input, output;
    private final ParameterLayout layout;
    private PipelineSet placeholders;
    /** Bound in h's writable slot without a residual: a writable binding may not share the read placeholder. */
    private GpuBuffer scratch;

    private Norms(GpuDevice device, GpuComputePipeline pipeline, Precision input, Precision output) {
        this.device = device; this.pipeline = pipeline; this.input = input; this.output = output;
        this.layout = ParameterLayout.of(GemmF16.KERNELS, kernel(Precision.FP32, Precision.FP32));
    }

    static String kernel(Precision input, Precision output) {
        return "norm_" + (input == Precision.FP32 ? "f32" : "f16") + (output == Precision.FP16 ? "_h" : "");
    }

    /** Compiles the kernel for {@code input} activations and {@code output}; FP16 needs {@link GpuCapabilities#FP16}. */
    public static Norms create(GpuDevice device, Precision input, Precision output) {
        Checks.notNull(device, "device");
        var types = input == Precision.FP16 || output == Precision.FP16 ? Set.of(GpuCapabilities.FP32, GpuCapabilities.FP16) : Set.of(GpuCapabilities.FP32);
        var accesses = List.of(Access.READ, Access.READ, Access.READ_WRITE, Access.READ, Access.READ, Access.READ_WRITE);
        return new Norms(device, device.pipeline(GemmF16.KERNELS.pipeline(kernel(input, output), accesses, types, Set.of())), input, output);
    }

    public Precision input() { return input; }
    public Precision output() { return output; }

    /**
     * Records the norm of each row of {@code x} into {@code y}. With a residual, {@code residual} and {@code h} are
     * both given and h receives {@code x + residual} (h may be x); else both are null. {@code bias} is LayerNorm's,
     * or null.
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, Kind kind, GpuBuffer x, GpuBuffer residual, GpuBuffer h,
            GpuBuffer weight, GpuBuffer bias, GpuBuffer y, Shape s) {
        Checks.notNull(kind, "kind"); Checks.notNull(s, "shape");
        if ((residual == null) != (h == null)) throw new IllegalArgumentException("a residual add needs both the residual and h");
        if (kind == Kind.RMS && bias != null) throw new IllegalArgumentException("RMSNorm has no bias");
        int in = input.bytes();
        Checks.range(x, 0, s.elements(s.ldx()), in, "x");
        Checks.range(weight, 0, s.n(), 4, "weight");
        Checks.range(y, 0, s.elements(s.ldy()), output.bytes(), "y");
        if (residual != null) { Checks.range(residual, 0, s.elements(s.ldr()), in, "residual"); Checks.range(h, 0, s.elements(s.ldh()), in, "h"); }
        if (bias != null) Checks.range(bias, 0, s.n(), 4, "bias");
        if (placeholders == null) placeholders = PipelineSet.compile(device, List.of());
        var none = placeholders.placeholder();
        boolean inPlace = h != null && same(x, h);
        if (inPlace && s.ldx() != s.ldh()) throw new IllegalArgumentException("in place, x and h share their leading dimension");
        var grid = Grid.linear(device.capabilities(), s.rows());
        byte[] params = layout.block().u32("rows", s.rows()).u32("n", s.n()).u32("ldx", s.ldx()).u32("ldr", s.ldr()).u32("ldh", s.ldh())
                .u32("ldy", s.ldy()).f32("eps", s.eps()).bool("layer", kind == Kind.LAYER).bool("residual", residual != null)
                .bool("bias", bias != null).u32("gridX", grid.gridX()).bool("inPlace", inPlace).bytes();
        if (h == null && scratch == null) scratch = device.allocate(new BufferDescriptor(4, java.util.EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE));
        return encoder.dispatch(pipeline, Map.of(0, inPlace ? none : x, 1, residual == null ? none : residual, 2, h == null ? scratch : h, 3, weight,
                4, bias == null ? none : bias, 5, y), params, grid.groups(), kind == Kind.RMS ? "norm.rms" : "norm.layer");
    }

    /**
     * Records the backward pass (1.12.0) of the norm of each row of {@code v} (this instance's input type: x, or h
     * after the residual add) with weight {@code weight} and the FP32 output gradient {@code dy}:
     * <ul>
     * <li>{@code dv} (FP32) receives the gradient with respect to v, or, with {@code accumulate}, has it added (the
     * residual stream's gradient, which then flows to both x and the residual);</li>
     * <li>{@code dWeight} and, for LayerNorm, {@code dBias} (FP32 vectors of n, or null) have the sums over the rows of
     * {@code dy xhat} and {@code dy} added: zero them before the first micro-batch.</li>
     * </ul>
     * The statistics are recomputed from v as the forward pass computed them. Sums are in fixed orders, so a device's
     * result does not depend on scheduling; bounded ({@link #backwardContract}).
     */
    public GpuCommandEncoder backward(GpuCommandEncoder encoder, Kind kind, GpuBuffer v, GpuBuffer weight, GpuBuffer dy, GpuBuffer dv,
            GpuBuffer dWeight, GpuBuffer dBias, BackwardShape s, boolean accumulate) {
        Checks.notNull(kind, "kind"); Checks.notNull(s, "shape");
        if (kind == Kind.RMS && dBias != null) throw new IllegalArgumentException("RMSNorm has no bias");
        Checks.range(v, 0, s.elements(s.ldv()), input.bytes(), "v");
        Checks.range(weight, 0, s.n(), 4, "weight");
        Checks.range(dy, 0, s.elements(s.lddy()), 4, "dy");
        Checks.range(dv, 0, s.elements(s.lddv()), 4, "dv");
        if (dWeight != null) Checks.range(dWeight, 0, s.n(), 4, "dWeight");
        if (dBias != null) Checks.range(dBias, 0, s.n(), 4, "dBias");
        if (dWeight == null && dBias != null) throw new IllegalArgumentException("dBias needs dWeight");
        if (backward == null) {
            var types = input == Precision.FP16 ? Set.of(GpuCapabilities.FP32, GpuCapabilities.FP16) : Set.of(GpuCapabilities.FP32);
            backward = PipelineSet.compile(device, List.of(
                    GemmF16.KERNELS.pipeline(input == Precision.FP32 ? "norm_backward_f32" : "norm_backward_f16",
                            List.of(Access.READ, Access.READ, Access.READ, Access.READ_WRITE, Access.READ_WRITE, Access.READ_WRITE), types, Set.of()),
                    GemmF16.KERNELS.pipeline("norm_backward_columns",
                            List.of(Access.READ, Access.READ, Access.READ_WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of())));
        }
        int blocks = Math.ceilDiv(s.rows(), BACKWARD_ROWS);
        boolean partials = dWeight != null;
        var grid = Grid.linear(device.capabilities(), blocks);
        byte[] params = ParameterLayout.of(GemmF16.KERNELS, "norm_backward_f32").block().u32("rows", s.rows()).u32("n", s.n())
                .u32("ldv", s.ldv()).u32("lddy", s.lddy()).u32("lddv", s.lddv()).f32("eps", s.eps()).bool("layer", kind == Kind.LAYER)
                .bool("accumulate", accumulate).u32("gridX", grid.gridX()).bool("partials", partials).bytes();
        long partBytes = partials ? 4L * blocks * s.n() : 16;
        // per dispatch: closing the handles keeps them alive for the recorded commands
        try (var pw = device.allocate(new BufferDescriptor(partBytes, java.util.EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE).withLabel("norm-dw-partials"));
             // LayerNorm writes bias partials whenever it writes weight partials
             var pb = device.allocate(new BufferDescriptor(partials && kind == Kind.LAYER ? partBytes : 16, java.util.EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE)
                     .withLabel("norm-db-partials"));
             var none = device.allocate(new BufferDescriptor(16, java.util.EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE).withLabel("norm-no-bias"))) {
            encoder.dispatch(backward.get(0), Map.of(0, v, 1, weight, 2, dy, 3, dv, 4, pw, 5, pb), params, grid.groups(),
                    kind == Kind.RMS ? "norm.rms.backward" : "norm.layer.backward");
            if (partials) {
                var cgrid = Grid.forThreads(device.capabilities(), s.n(), 256);
                byte[] cp = ParameterLayout.of(GemmF16.KERNELS, "norm_backward_columns").block().u32("n", s.n()).u32("blocks", blocks)
                        .bool("bias", dBias != null).u32("gridX", cgrid.gridX()).bytes();
                encoder.dispatch(backward.get(1), Map.of(0, pw, 1, pb, 2, dWeight, 3, dBias == null ? none : dBias), cp, cgrid.groups(), "norm.backward.columns");
            }
        }
        return encoder;
    }

    /** The backward pass's contract: bounded (the device's rsqrt and divisions; sums in fixed orders). @since 1.12.0 */
    public static AccuracyContract backwardContract(Kind kind, Precision input) {
        return new AccuracyContract("Norms " + kind + " backward, " + input + " v", List.of(input.name(), "FP32 dy"),
                "FP32, fixed tree over 256 threads per row, rows in order", "FP32", AccuracyContract.Order.DEVICE,
                AccuracyContract.Bound.formula("|dv - exact| <= inv (|g| + sum|g| / n + |xhat| sum|g xhat| / n) (4 e_inv + k 2^-24) "
                        + "(+ LayerNorm: the mean's error through xhat), e_inv = (n / 256 + 16) 2^-24, k = n / 256 + 16; "
                        + "the weight and bias gradients within (rows + n / 256 + 16) 2^-24 of their sums of magnitudes"), "Norms.backwardExact");
    }

    /**
     * The backward pass of one row in double from the given v (n values from {@code off}), w and dy: returns dv and
     * fills {@code dwRow} and {@code dbRow} (may be null) with the row's contributions dy xhat and dy. @since 1.12.0
     */
    public static double[] backwardExact(Kind kind, float[] v, int off, int n, float[] w, float[] dy, int dyOff, float eps, double[] dwRow, double[] dbRow) {
        double mean = 0, var = 0;
        if (kind == Kind.LAYER) { for (int j = 0; j < n; j++) mean += v[off + j]; mean /= n; }
        for (int j = 0; j < n; j++) { double d = v[off + j] - mean; var += d * d; }
        double inv = 1 / Math.sqrt(var / n + eps);
        double sgx = 0, sg = 0;
        for (int j = 0; j < n; j++) { double g = (double) w[j] * dy[dyOff + j], xh = (v[off + j] - mean) * inv; sgx += g * xh; sg += g; }
        sgx /= n; sg = kind == Kind.LAYER ? sg / n : 0;
        double[] dv = new double[n];
        for (int j = 0; j < n; j++) {
            double g = (double) w[j] * dy[dyOff + j], xh = (v[off + j] - mean) * inv;
            dv[j] = inv * (g - sg - xh * sgx);
            if (dwRow != null) dwRow[j] = dy[dyOff + j] * xh;
            if (dbRow != null) dbRow[j] = dy[dyOff + j];
        }
        return dv;
    }

    /** Whether two buffers are the same range of the same memory (one may be bound in place of both). */
    static boolean same(GpuBuffer a, GpuBuffer b) {
        return a == b || a.offset() == b.offset() && a.bytes() == b.bytes() && root(a) == root(b);
    }
    private static Object root(GpuBuffer b) { return b instanceof BufferSlice ? b.descriptor() : b; }

    /**
     * The norm of {@code v} (the values the kernel normalizes: x, or h after the residual add) in double, the exact
     * result the GPU's is held to; {@code b} may be null.
     */
    public static double[] exact(Kind kind, float[] v, int off, int n, float[] w, float[] b, float eps) {
        double mean = 0, var = 0;
        if (kind == Kind.LAYER) {
            for (int j = 0; j < n; j++) mean += v[off + j];
            mean /= n;
        }
        for (int j = 0; j < n; j++) { double d = v[off + j] - mean; var += d * d; }
        double inv = 1 / Math.sqrt(var / n + eps);
        double[] y = new double[n];
        for (int j = 0; j < n; j++) y[j] = (v[off + j] - mean) * inv * w[j] + (b == null ? 0 : b[j]);
        return y;
    }

    /** The residual stream the kernel writes: {@code in(x + res)}, one rounding (to half for FP16 activations). */
    public static float residualSum(float x, float res, Precision input) {
        float s = x + res;
        return input == Precision.FP16 ? Float.float16ToFloat(Float.floatToFloat16(s)) : s;
    }

    /** The accuracy contract: bounded. */
    public static AccuracyContract contract(Kind kind, Precision input, Precision output) {
        return new AccuracyContract("Norms " + kind + ", " + input + " in, " + output + " out", List.of(input.name()),
                "FP32, fixed tree over 256 threads", output == Precision.FP32 ? "FP32" : "FP32 rounded once to half", AccuracyContract.Order.DEVICE,
                AccuracyContract.Bound.formula("|y - exact| <= (8 + n 2^-8) 2^-24 |w| (|v - mean| inv) + 2^-24 |b| (rsqrt and the division by n "
                        + "within a few ULP, the sums within n 2^-24)" + (output == Precision.FP16 ? ", plus one rounding to half" : "")
                        + "; h is in(x + res) exactly"), "Norms.exact");
    }

    @Override public void close() { PipelineSet.closeAll(pipeline, placeholders, scratch, backward); }
}
