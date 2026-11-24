package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.gpu.metal.MetalProvider;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

@EnabledIfSystemProperty(named = "compute.metal.required", matches = "true")
class MetalGemmF16Test extends GemmF16Conformance {
    @Override protected GpuProvider provider() { return new MetalProvider(); }
}
