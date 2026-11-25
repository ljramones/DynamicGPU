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
 * AdamW and SGD steps over FP32 parameters as throughput kernels (1.12.0), with FP32 or FP16 gradients:
 * <pre>
 *     AdamW: m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g g;  p = p (1 - lr wd) - lr (m / bc1) / (sqrt(v / bc2) + eps)
 *     SGD:   g = g + wd p;  b = mu b + g;  p = p - lr (nesterov ? g + mu b : b)
 * </pre>
 * with {@code bc1 = 1 - b1^t}, {@code bc2 = 1 - b2^t} for step t. Options:
 * <ul>
 * <li><b>A gradient scale on the GPU</b> ({@code scale}, one float): the gradients are multiplied by it first, for
 * gradient clipping or loss-scale unscaling computed on the GPU, without a host sync.</li>
 * <li><b>An FP16 copy</b> ({@code shadow}) of the updated parameters, the weights a mixed-precision forward pass reads.</li>
 * </ul>
 * Bit for bit {@link #adamWReference} and {@link #sgdReference}: every operation is one IEEE rounding (none fused),
 * and division and square root are correctly rounded on native Metal, MoltenVK and lavapipe; a Vulkan device whose
 * division or square root is not would stay within their ULP bounds. vectrix-gpu's {@code GpuOptim} holds twins
 * that fuse two operations, as {@code Optim} does.
 */
public final class Optimizer implements AutoCloseable {
    /** AdamW's hyperparameters. */
    public record AdamW(float lr, float beta1, float beta2, float eps, float weightDecay) {
        /** The common defaults: b1 0.9, b2 0.999, eps 1e-8. */
        public static AdamW of(float lr, float weightDecay) { return new AdamW(lr, 0.9f, 0.999f, 1e-8f, weightDecay); }
    }

    /** SGD's hyperparameters. */
    public record Sgd(float lr, float momentum, float weightDecay, boolean nesterov) { }

    private final GpuDevice device;
    private final PipelineSet pipelines;
    private final ParameterLayout layout = ParameterLayout.of(GemmF16.KERNELS, "optimizer_f32");

    private Optimizer(GpuDevice device) {
        this.device = device;
        var acc = List.of(Access.READ_WRITE, Access.READ, Access.READ_WRITE, Access.READ_WRITE, Access.READ, Access.READ_WRITE);
        var list = new java.util.ArrayList<PipelineDescriptor>();
        list.add(GemmF16.KERNELS.pipeline("optimizer_f32", acc, Set.of(GpuCapabilities.FP32), Set.of()));
        if (device.capabilities().dataTypes().contains(GpuCapabilities.FP16))
            list.add(GemmF16.KERNELS.pipeline("optimizer_f16", acc, Set.of(GpuCapabilities.FP32, GpuCapabilities.FP16), Set.of()));
        this.pipelines = PipelineSet.compile(device, list);
    }

    /** Compiles the kernels for {@code device}. */
    public static Optimizer create(GpuDevice device) { return new Optimizer(Checks.notNull(device, "device")); }

    /**
     * Records an AdamW step {@code step} (1-based) over {@code n} parameters: {@code param}, {@code m} and {@code v}
     * FP32, updated in place; {@code grad} in {@code gradType}. {@code scale} (one float) and {@code shadow}
     * ({@code n} halves) may be null.
     */
    public GpuCommandEncoder adamW(GpuCommandEncoder encoder, GpuBuffer param, GpuBuffer grad, Precision gradType, GpuBuffer m, GpuBuffer v,
            GpuBuffer scale, GpuBuffer shadow, long n, AdamW h, long step) {
        Checks.notNull(h, "hyperparameters");
        if (step < 1) throw new IllegalArgumentException("step must be at least 1: " + step);
        float[] sc = adamWScalars(h, step);
        return dispatch(encoder, 0, param, grad, gradType, m, v, scale, shadow, n, sc[0], h.beta1(), h.beta2(), h.eps(), sc[1], sc[2], sc[3], sc[4], sc[5], false);
    }

    /** Records an SGD step over {@code n} parameters; {@code momentumBuffer} FP32, updated in place (start at 0). */
    public GpuCommandEncoder sgd(GpuCommandEncoder encoder, GpuBuffer param, GpuBuffer grad, Precision gradType, GpuBuffer momentumBuffer,
            GpuBuffer scale, GpuBuffer shadow, long n, Sgd h) {
        Checks.notNull(h, "hyperparameters");
        return dispatch(encoder, 1, param, grad, gradType, momentumBuffer, null, scale, shadow, n, -h.lr(), h.momentum(), 0f, 0f, 0f, 0f, 0f, 0f,
                h.weightDecay(), h.nesterov());
    }

    /** AdamW's host scalars: negLr, bc1, bc2, a1, a2, decay. */
    static float[] adamWScalars(AdamW h, long step) {
        return new float[] {-h.lr(), (float) (1 - Math.pow(h.beta1(), step)), (float) (1 - Math.pow(h.beta2(), step)),
            1f - h.beta1(), 1f - h.beta2(), 1f - h.lr() * h.weightDecay()};
    }

    private GpuCommandEncoder dispatch(GpuCommandEncoder encoder, int kind, GpuBuffer param, GpuBuffer grad, Precision gradType, GpuBuffer m,
            GpuBuffer v, GpuBuffer scale, GpuBuffer shadow, long n, float negLr, float beta1, float beta2, float eps, float bc1, float bc2,
            float a1, float a2, float decay, boolean nesterov) {
        Checks.notNull(gradType, "gradType");
        if (n <= 0) return encoder;
        Checks.range(param, 0, n, 4, "param");
        Checks.range(grad, 0, n, gradType.bytes(), "grad");
        Checks.range(m, 0, n, 4, kind == 0 ? "m" : "momentumBuffer");
        if (v != null) Checks.range(v, 0, n, 4, "v");
        if (scale != null) Checks.range(scale, 0, 1, 4, "scale");
        if (shadow != null) Checks.range(shadow, 0, n, 2, "shadow");
        if (gradType == Precision.FP16 && pipelines.size() < 2) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "FP16 gradients need FP16");
        var grid = Grid.forThreads(device.capabilities(), Math.ceilDiv(n, 2), 256);
        byte[] params = layout.block().u32("n", n).u32("gridX", grid.gridX()).u32("kind", kind).bool("scaled", scale != null)
                .bool("shadow", shadow != null).bool("nesterov", nesterov).f32("negLr", negLr).f32("beta1", beta1).f32("beta2", beta2)
                .f32("eps", eps).f32("bc1", bc1).f32("bc2", bc2).f32("a1", a1).f32("a2", a2).f32("decay", decay).bytes();
        try (var none = device.allocate(new BufferDescriptor(16, java.util.EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE).withLabel("optimizer-none"));
             var none2 = device.allocate(new BufferDescriptor(16, java.util.EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE).withLabel("optimizer-none"))) {
            encoder.dispatch(pipelines.get(gradType == Precision.FP16 ? 1 : 0), Map.of(0, param, 1, grad, 2, m, 3, v == null ? none : v,
                    4, scale == null ? pipelines.placeholder() : scale, 5, shadow == null ? none2 : shadow), params, grid.groups(),
                    kind == 0 ? "optimizer.adamw" : "optimizer.sgd");
        }
        return encoder;
    }

    /** AdamW in Java, operation for operation the kernel's: updates {@code p}, {@code m} and {@code v} in place. */
    public static void adamWReference(float[] p, float[] g, float[] m, float[] v, float scale, AdamW h, long step) {
        float[] sc = adamWScalars(h, step);
        for (int i = 0; i < p.length; i++) {
            float gi = g[i] * scale;
            float mi = h.beta1() * m[i] + sc[3] * gi;
            float vi = h.beta2() * v[i] + (sc[4] * gi) * gi;
            m[i] = mi; v[i] = vi;
            float u = (mi / sc[1]) / ((float) Math.sqrt(vi / sc[2]) + h.eps());
            p[i] = p[i] * sc[5] + sc[0] * u;
        }
    }

    /** SGD in Java, operation for operation the kernel's: updates {@code p} and {@code b} in place. */
    public static void sgdReference(float[] p, float[] g, float[] b, float scale, Sgd h) {
        for (int i = 0; i < p.length; i++) {
            float gi = g[i] * scale;
            if (h.weightDecay() != 0f) gi = gi + h.weightDecay() * p[i];
            float bi = h.momentum() * b[i] + gi;
            b[i] = bi;
            p[i] = p[i] + -h.lr() * (h.nesterov() ? gi + h.momentum() * bi : bi);
        }
    }

    /** The contract: bit for bit the references where division and sqrt are correctly rounded (every device tested). */
    public static AccuracyContract contract() {
        return new AccuracyContract("Optimizer AdamW / SGD", List.of("FP32 parameters", "FP32 or FP16 gradients"), "FP32, each operation rounded once",
                "FP32 (and an FP16 copy, rounded to nearest even)", AccuracyContract.Order.FIXED,
                AccuracyContract.Bound.exact(), "Optimizer.adamWReference, Optimizer.sgdReference");
    }

    @Override public void close() { pipelines.close(); }
}
