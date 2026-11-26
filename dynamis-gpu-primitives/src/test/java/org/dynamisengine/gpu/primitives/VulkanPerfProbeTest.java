package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

@EnabledIfSystemProperty(named = "compute.vulkan.required", matches = "true")
class VulkanPerfProbeTest extends PerfProbeConformance {
    @Override protected GpuProvider provider() { return new org.dynamisengine.gpu.vulkan.compute.VulkanProvider(); }
}
