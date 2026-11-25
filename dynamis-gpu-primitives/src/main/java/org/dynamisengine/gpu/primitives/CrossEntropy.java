package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.Grid;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import org.dynamisengine.gpu.api.compute.kit.PipelineSet;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Cross-entropy of each row of FP32 logits against a target token (1.12.0), and its gradient:
 * <pre>
 *     loss[r] = log sum_j exp(l[r][j]) - l[r][t_r]
 *     dl[r][j] = scale (softmax(l[r])[j] - [j == t_r])
 * </pre>
 * The targets are uint32 on the GPU; {@link #IGNORE} marks a row with no target (padding), whose loss and gradient
 * are 0. The log-sum-exp is {@link Reduce#logSumExp} (multi-group); the loss and gradient are one dispatch over the
 * rows or the elements. The gradient may overwrite the logits (in place). Average the loss over the counted rows with
 * {@link Reduce}, and pass {@code scale = 1 / count} for the mean's gradient. A target past the vocabulary fails the
 * submission with {@link ComputeException.Code#KERNEL_ERROR}.
 */
public final class CrossEntropy implements AutoCloseable {
    /** The target of a row that counts for nothing. */
    public static final int IGNORE = -1;

    /** The rows: {@code rows} of {@code vocab} logits at leading dimension {@code ld}; the gradient at {@code lddl}. */
    public record Shape(int rows, int vocab, int ld, int lddl) {
        public Shape {
            if (rows <= 0 || vocab <= 0) throw new IllegalArgumentException("dimensions must be positive: " + rows + "x" + vocab);
            if (ld < vocab || lddl < vocab) throw new IllegalArgumentException("leading dimension smaller than a row");
            if ((long) rows * vocab > 0xFFFFFFFFL) throw new IllegalArgumentException("rows x vocab must fit 32 bits");
        }
        /** Densely packed rows. */
        public static Shape of(int rows, int vocab) { return new Shape(rows, vocab, vocab, vocab); }
    }

    private final GpuDevice device;
    private final Reduce reduce;
    private final PipelineSet pipelines;
    private final ParameterLayout layout = ParameterLayout.of(GemmF16.KERNELS, "cross_entropy");

    private CrossEntropy(GpuDevice device) {
        this.device = device;
        this.reduce = Reduce.create(device);
        this.pipelines = PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("cross_entropy",
                List.of(Access.READ, Access.READ, Access.READ, Access.READ_WRITE, Access.READ_WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of())));
    }

    /** Compiles the kernels for {@code device}. */
    public static CrossEntropy create(GpuDevice device) { return new CrossEntropy(Checks.notNull(device, "device")); }

    /**
     * Records the loss of each row into {@code loss} (FP32, {@code rows}) and, if {@code dlogits} is not null, the
     * gradient scaled by {@code scale} into it (FP32; it may be {@code logits} itself at the same leading dimension).
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer logits, GpuBuffer targets, GpuBuffer loss, GpuBuffer dlogits,
            float scale, Shape s) {
        Checks.notNull(s, "shape");
        long rows = s.rows();
        Checks.range(logits, 0, (rows - 1) * s.ld() + s.vocab(), 4, "logits");
        Checks.range(targets, 0, rows, 4, "targets");
        Checks.range(loss, 0, rows, 4, "loss");
        boolean grad = dlogits != null, inPlace = grad && Norms.same(logits, dlogits);
        if (grad) Checks.range(dlogits, 0, (rows - 1) * s.lddl() + s.vocab(), 4, "dlogits");
        if (inPlace && s.ld() != s.lddl()) throw new IllegalArgumentException("in place, the logits and the gradient share their leading dimension");
        if (!Float.isFinite(scale)) throw new IllegalArgumentException("scale must be finite: " + scale);
        var grid = Grid.forThreads(device.capabilities(), grad ? rows * s.vocab() : rows, 256);
        byte[] params = layout.block().u32("rows", s.rows()).u32("vocab", s.vocab()).u32("ld", s.ld()).u32("lddl", s.lddl())
                .f32("scale", scale).u32("gridX", grid.gridX()).bool("inPlace", inPlace).bool("grad", grad).bytes();
        // per dispatch: closing the handles keeps them alive for the recorded commands
        try (var lse = scratch(4L * rows, "ce-lse"); var none = scratch(16, "ce-none");
             var errors = device.allocate(KernelErrors.descriptor().withLabel("ce-errors"))) {
            reduce.logSumExp(encoder, logits, StridedView.matrix(0, s.rows(), s.vocab(), s.ld()), lse, StridedView.contiguous(s.rows()));
            encoder.resetErrors(errors);
            encoder.dispatch(pipelines.get(0), Map.of(0, inPlace ? pipelines.placeholder() : logits, 1, targets, 2, lse, 3, loss,
                    4, grad ? dlogits : none, 5, errors), params, grid.groups(), grad ? "cross-entropy.grad" : "cross-entropy");
            encoder.checkErrors(errors);
        }
        return encoder;
    }

    private GpuBuffer scratch(long bytes, String label) {
        return device.allocate(new BufferDescriptor(Math.max(16, bytes), EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE).withLabel(label));
    }

    /** The loss of one row in double from the given logits. */
    public static double lossExact(float[] logits, int off, int vocab, int target) {
        if (target == IGNORE) return 0;
        return Reduce.logSumExpExact(logits, off, vocab) - logits[off + target];
    }

    /** The contract: bounded by the log-sum-exp's bound and the device's exp. */
    public static AccuracyContract contract() {
        return new AccuracyContract("CrossEntropy", List.of("FP32 logits", "uint32 targets"), "FP32: Reduce.logSumExp, then elementwise", "FP32",
                AccuracyContract.Order.DEVICE, AccuracyContract.Bound.formula("|loss - exact| <= d + 2^-24 |loss|, d the log-sum-exp's bound "
                        + "(Reduce.logSumExpContract); |dl - exact| <= |scale| (p (d + 2^-24 |l - lse| + exp error) + 2^-24 |dl / scale|), "
                        + "p the exact probability; ignored rows exactly 0"), "CrossEntropy.lossExact");
    }

    @Override public void close() { PipelineSet.closeAll(pipelines, reduce); }
}
