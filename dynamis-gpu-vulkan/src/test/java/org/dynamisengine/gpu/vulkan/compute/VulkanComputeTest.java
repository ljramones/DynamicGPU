package org.dynamisengine.gpu.vulkan.compute;
import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.test.compute.ComputeConformance;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import java.util.Map;
@EnabledIfSystemProperty(named = "compute.vulkan.required", matches = "true")
class VulkanComputeTest extends ComputeConformance {
    @org.junit.jupiter.api.Test void malformedShaderRejectedBeforeDriver() {
        PipelineDescriptor good; try (var p = pipeline(64)) { good = p.descriptor(); }
        var bad = new KernelBundle("invalid", "1", Map.of(Backend.VULKAN, new KernelBundle.Executable(KernelBundle.Format.SPIRV, new byte[24])));
        var descriptor = new PipelineDescriptor(bad, "vector_add", good.layout(), good.localSize(), Map.of(), java.util.Set.of(), java.util.Set.of());
        org.junit.jupiter.api.Assertions.assertEquals(ComputeException.Code.INVALID_ARGUMENT,
            org.junit.jupiter.api.Assertions.assertThrows(ComputeException.class, () -> device.pipeline(descriptor)).code());
    }
    @Override protected KernelBundle strictKernel() {
        try (var stream = ComputeConformance.class.getResourceAsStream("/compute/fp_strict.spv")) {
            return new KernelBundle("fp-strict", "1", Map.of(Backend.VULKAN, new KernelBundle.Executable(KernelBundle.Format.SPIRV, stream.readAllBytes())));
        } catch (java.io.IOException e) { throw new AssertionError(e); }
    }
    @Override protected GpuProvider provider() { return new VulkanProvider(); }
    @Override protected KernelBundle kernel() {
        try (var stream = ComputeConformance.class.getResourceAsStream("/compute/vector_add.spv")) {
            return new KernelBundle("vector-add", "1", Map.of(Backend.VULKAN, new KernelBundle.Executable(KernelBundle.Format.SPIRV, stream.readAllBytes())));
        } catch (java.io.IOException e) { throw new AssertionError(e); }
    }
}
