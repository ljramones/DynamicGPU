package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.gpu.vulkan.compute.VulkanProvider;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

@EnabledIfSystemProperty(named = "compute.vulkan.required", matches = "true")
class VulkanGemmF16Test extends GemmF16Conformance {
    @Override protected GpuProvider provider() { return new VulkanProvider(); }
}
