package org.dynamisengine.gpu.vulkan.compute;
import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.test.compute.ComputeConformance;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import java.util.Map;
@EnabledIfSystemProperty(named = "compute.vulkan.required", matches = "true")
class VulkanComputeTest extends org.dynamisengine.gpu.test.compute.CapabilityTierConformance {
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
    /** MoltenVK (Apple's vendor ID, 0x106B) runs with fast math off; other drivers may reassociate unmarked code. */
    @Override protected boolean strictWithoutPreciseMarks() { return device.adapter().id().contains(":" + 0x106B + ":"); }
    @Override protected KernelBundle errorKernel() {
        try (var stream = ComputeConformance.class.getResourceAsStream("/compute/error_report.spv")) {
            return new KernelBundle("error-report", "1", Map.of(Backend.VULKAN, new KernelBundle.Executable(KernelBundle.Format.SPIRV, stream.readAllBytes())));
        } catch (java.io.IOException e) { throw new AssertionError(e); }
    }
    /** MoltenVK on Apple GPUs (vendor 0x106B) fuses fma; Mesa's lavapipe (vendor 0x10005) rounds twice. */
    @Override protected Boolean expectedFusedFma() {
        String id = device.adapter().id();
        if (id.contains(":" + 0x106B + ":")) return true;
        if (id.contains(":" + 0x10005 + ":")) return false;
        return null;
    }
    /** MoltenVK on Apple GPUs and Mesa's lavapipe; lavapipe adds FP64 and 64-bit atomics. */
    @Override protected java.util.Set<String> expectedTiers() {
        String id = device.adapter().id(); var apple = java.util.Set.of(GpuCapabilities.FP16, GpuCapabilities.INT16, GpuCapabilities.INT8, GpuCapabilities.INT64,
                GpuCapabilities.SUBGROUP_BASIC, GpuCapabilities.SUBGROUP_VOTE, GpuCapabilities.SUBGROUP_ARITHMETIC,
                GpuCapabilities.SUBGROUP_BALLOT, GpuCapabilities.SUBGROUP_SHUFFLE, GpuCapabilities.ATOMIC_FLOAT32_ADD);
        if (id.contains(":" + 0x106B + ":")) return apple;
        if (id.contains(":" + 0x10005 + ":")) {
            var all = new java.util.TreeSet<>(apple); all.add(GpuCapabilities.FP64); all.add(GpuCapabilities.ATOMIC_INT64); return all;
        }
        return java.util.Set.of();
    }
    /** A module that uses a capability the device lacks is refused even when its descriptor does not declare it. */
    @org.junit.jupiter.api.Test void undeclaredCapabilitiesAreRefused() {
        for (var t : ALL) {
            if (reports(t)) continue;
            var undeclared = descriptor(t, false);
            org.junit.jupiter.api.Assertions.assertEquals(ComputeException.Code.UNSUPPORTED,
                org.junit.jupiter.api.Assertions.assertThrows(ComputeException.class, () -> device.pipeline(undeclared)).code(), t.kernel());
            System.out.println("undeclared " + t.requirement() + " refused");
        }
    }
    @Override protected GpuProvider provider() { return new VulkanProvider(); }
    @Override protected KernelBundle kernel() {
        try (var stream = ComputeConformance.class.getResourceAsStream("/compute/vector_add.spv")) {
            return new KernelBundle("vector-add", "1", Map.of(Backend.VULKAN, new KernelBundle.Executable(KernelBundle.Format.SPIRV, stream.readAllBytes())));
        } catch (java.io.IOException e) { throw new AssertionError(e); }
    }
}
