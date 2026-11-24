package org.dynamisengine.gpu.primitives;

import java.util.List;
import java.util.Map;
import java.util.Set;
import org.dynamisengine.gpu.api.compute.Access;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.KernelErrors;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.Grid;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import org.dynamisengine.gpu.api.compute.kit.PipelineSet;
import org.dynamisengine.gpu.api.compute.kit.StridedView;

/**
 * Finds NaNs and infinities in an FP32 strided view and reports them through a {@link KernelErrors} buffer: code
 * {@link #NONFINITE} and the smallest linear (row-major) index in the view. A {@link #tripwire} armed after a
 * dispatch fails the submission ({@code KERNEL_ERROR}) when that dispatch produced a non-finite value, without
 * reading the data back.
 *
 * @since 1.7.0
 */
public final class NanCheck implements AutoCloseable {
    /** The error code reported for a NaN or an infinity ({@code DG_ERROR_NONFINITE} in {@code dynamis/errors.slang}). */
    public static final int NONFINITE = 2;
    private final GpuDevice device;
    private final PipelineSet pipelines;
    private final ParameterLayout layout = ParameterLayout.of(GemmF16.KERNELS, "nan_check");

    private NanCheck(GpuDevice device) {
        this.device = device;
        this.pipelines = PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("nan_check",
                List.of(Access.READ, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of(),
                Map.of("DG_DEBUG", org.dynamisengine.gpu.api.compute.kit.KitOptions.current().debug()))));
    }

    /** The check, compiled for {@code device}. */
    public static NanCheck create(GpuDevice device) { return new NanCheck(Checks.notNull(device, "device")); }

    /** Records reporting every non-finite element of {@code x} into {@code errors} (not reset first). */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer x, StridedView xv, GpuBuffer errors) {
        xv.check(x, 4, "x");
        Checks.notNull(errors, "errors");
        var x4 = xv.withRank(4);
        long count = Checks.u32(x4.elements(), "element count");
        if (count == 0) return encoder;
        var grid = Grid.forThreads(device.capabilities(), count, 256);
        byte[] params = layout.block().u32("count", count).u32("gridX", grid.gridX()).u32("xLength", DebugChecks.length(x))
                .view("x", x4).bytes();
        return encoder.dispatch(pipelines.get(0), Map.of(0, x, 1, errors), params, grid.groups(), "nan_check");
    }

    /**
     * Records a tripwire: reset {@code errors}, check {@code x}, then {@link GpuCommandEncoder#checkErrors}. The
     * submission fails with {@code KERNEL_ERROR} if {@code x} holds a NaN or an infinity.
     */
    public GpuCommandEncoder tripwire(GpuCommandEncoder encoder, GpuBuffer x, StridedView xv, GpuBuffer errors) {
        encoder.resetErrors(errors);
        encode(encoder, x, xv, errors);
        return encoder.checkErrors(errors);
    }

    @Override public void close() { pipelines.close(); }
}
