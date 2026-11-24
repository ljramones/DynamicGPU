package org.dynamisengine.gpu.primitives;

import java.util.HashMap;
import java.util.Map;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.KernelErrors;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.KitOptions;
import org.dynamisengine.gpu.api.compute.kit.PipelineSet;

/**
 * A primitive's debug build ({@link KitOptions#debug()}, read at creation): the DG_DEBUG specialization constant, and
 * the error buffer its kernels report out-of-range accesses to. A debug dispatch is bracketed by resetting and checking
 * the buffer, so its submission fails with KERNEL_ERROR; a normal one binds a placeholder its kernels never touch.
 */
final class DebugChecks implements AutoCloseable {
    final boolean on;
    private final GpuDevice device;
    private GpuBuffer errors;
    private PipelineSet placeholder;

    DebugChecks(GpuDevice device) {
        this.device = device;
        this.on = KitOptions.current().debug();
    }

    /** {@code base} with DG_DEBUG set. */
    Map<String, Object> constants(Map<String, ?> base) {
        Map<String, Object> m = new HashMap<>(base);
        m.put("DG_DEBUG", on);
        return m;
    }

    /** The buffer to bind in the kernels' error slot. */
    GpuBuffer buffer() {
        if (on) {
            if (errors == null) errors = device.allocate(KernelErrors.descriptor().withLabel("kernel-debug"));
            return errors;
        }
        if (placeholder == null) placeholder = PipelineSet.compile(device, java.util.List.of());
        return placeholder.placeholder();
    }

    void before(GpuCommandEncoder encoder) { if (on) encoder.resetErrors(buffer()); }

    void after(GpuCommandEncoder encoder) { if (on) encoder.checkErrors(buffer()); }

    /** A buffer's length in 4-byte elements, as a kernel parameter (clamped to the 32-bit index range). */
    static long length(GpuBuffer b) { return Math.min(b.bytes() / 4, Checks.MAX_INDEX); }

    @Override public void close() { PipelineSet.closeAll(errors, placeholder); }
}
