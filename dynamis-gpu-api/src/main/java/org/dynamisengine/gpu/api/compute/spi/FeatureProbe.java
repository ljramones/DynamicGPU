package org.dynamisengine.gpu.api.compute.spi;

import org.dynamisengine.gpu.api.compute.*;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Measures the shader features a device can only report by running code: today {@link GpuCapabilities#FUSED_FMA}.
 * One dispatch of fma_probe (SPIR-V for Vulkan, Metal source for Metal) computes {@code fma(x, x, c)} with
 * {@code c = -(x*x)} rounded here; a fused fma returns the square's rounding error, two roundings return 0.
 * Backends without a probe executable (the mock) and any failure report no features.
 */
final class FeatureProbe {
    private static final System.Logger LOG = System.getLogger(FeatureProbe.class.getName());
    private static final float X = 1 + 0x1p-12f;
    private static final int FUSED_RESIDUAL = Float.floatToRawIntBits(0x1p-24f);

    private FeatureProbe() { }

    static Set<String> measure(ComputeDevice device) {
        KernelBundle.Executable executable = switch (device.adapter().backend()) {
            case VULKAN -> new KernelBundle.Executable(KernelBundle.Format.SPIRV, resource("fma_probe.spv"));
            case METAL -> new KernelBundle.Executable(KernelBundle.Format.MSL_SOURCE, resource("fma_probe.metal"));
            default -> null;
        };
        if (executable == null) return Set.of();
        var bundle = new KernelBundle("dynamisgpu.fma-probe", "1", Map.of(device.adapter().backend(), executable));
        var layout = new BindingLayout(List.of(new BindingLayout.BufferBinding(0, Access.WRITE, 4)), 8);
        var descriptor = new PipelineDescriptor(bundle, "fma_probe", layout, new Dimensions(1, 1, 1), Map.of(),
                Set.of("FP32"), Set.of());
        float c = -(X * X);
        byte[] parameters = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putFloat(X).putFloat(c).array();
        try (var pipeline = device.pipeline(descriptor);
             var out = device.allocate(new BufferDescriptor(4, EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE), Placement.PRIVATE));
             var encoder = device.encoder()) {
            encoder.dispatch(pipeline, Map.of(0, out), parameters, new Dimensions(1, 1, 1));
            var readback = encoder.download(out);
            try (var submission = device.queue().submit(encoder)) {
                if (!submission.await(Duration.ofSeconds(20))) {
                    LOG.log(System.Logger.Level.WARNING, "fma probe timed out on " + device.adapter().name());
                    return Set.of();
                }
            }
            int residual = ByteBuffer.wrap(readback.bytes()).order(ByteOrder.LITTLE_ENDIAN).getInt();
            return residual == FUSED_RESIDUAL ? Set.of(GpuCapabilities.FUSED_FMA) : Set.of();
        } catch (RuntimeException e) {
            LOG.log(System.Logger.Level.WARNING, "fma probe failed on " + device.adapter().name() + "; reporting no fused fma", e);
            return Set.of();
        }
    }

    private static byte[] resource(String name) {
        try (InputStream in = FeatureProbe.class.getResourceAsStream(name)) {
            if (in == null) throw new IllegalStateException("missing probe resource " + name);
            return in.readAllBytes();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
