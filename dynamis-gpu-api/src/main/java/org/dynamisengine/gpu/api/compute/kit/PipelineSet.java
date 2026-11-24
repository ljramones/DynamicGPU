package org.dynamisengine.gpu.api.compute.kit;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import org.dynamisengine.gpu.api.compute.BufferDescriptor;
import org.dynamisengine.gpu.api.compute.BufferUsage;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuComputePipeline;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.PipelineDescriptor;
import org.dynamisengine.gpu.api.compute.Placement;

/**
 * Pipelines that are compiled together and closed together: if one fails to compile, those already made are closed
 * and the failure is rethrown. Also holds a placeholder buffer for optional arguments a dispatch omits (kernels never
 * read or write it when the argument is absent).
 *
 * @since 1.7.0
 */
public final class PipelineSet implements AutoCloseable {
    private final GpuDevice device;
    private final List<GpuComputePipeline> pipelines;
    private GpuBuffer placeholder;
    private boolean closed;

    private PipelineSet(GpuDevice device, List<GpuComputePipeline> pipelines) {
        this.device = device;
        this.pipelines = pipelines;
    }

    /** Compiles {@code descriptors} in order on {@code device}. */
    public static PipelineSet compile(GpuDevice device, List<PipelineDescriptor> descriptors) {
        Checks.notNull(device, "device");
        List<GpuComputePipeline> made = new ArrayList<>();
        try {
            for (PipelineDescriptor d : descriptors) made.add(device.pipeline(d));
        } catch (RuntimeException e) {
            try {
                closeAll(made.toArray(new AutoCloseable[0]));
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
        return new PipelineSet(device, List.copyOf(made));
    }

    public GpuDevice device() { return device; }

    /** The {@code i}th pipeline, in descriptor order. */
    public GpuComputePipeline get(int i) {
        if (closed) throw new IllegalStateException("pipeline set is closed");
        return pipelines.get(i);
    }

    public int size() { return pipelines.size(); }

    /** A 4-byte private storage buffer to bind in the slot of an absent optional argument; made on first use. */
    public GpuBuffer placeholder() {
        if (closed) throw new IllegalStateException("pipeline set is closed");
        if (placeholder == null) {
            placeholder = device.allocate(new BufferDescriptor(4, EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE, "placeholder"));
        }
        return placeholder;
    }

    /** Closes the pipelines and the placeholder, rethrowing the first failure after trying them all. */
    @Override public void close() {
        if (closed) return;
        closed = true;
        List<AutoCloseable> all = new ArrayList<>(pipelines);
        all.add(placeholder);
        closeAll(all.toArray(new AutoCloseable[0]));
    }

    /** Closes every resource (nulls skipped), rethrowing the first failure after trying them all. */
    public static void closeAll(AutoCloseable... resources) {
        RuntimeException first = null;
        for (AutoCloseable r : resources) {
            try {
                if (r != null) r.close();
            } catch (Exception e) {
                RuntimeException re = e instanceof RuntimeException x ? x : new IllegalStateException(e);
                if (first == null) first = re; else first.addSuppressed(re);
            }
        }
        if (first != null) throw first;
    }
}
