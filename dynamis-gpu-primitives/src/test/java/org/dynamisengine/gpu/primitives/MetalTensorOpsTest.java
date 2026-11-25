package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

@EnabledIfSystemProperty(named = "compute.metal.required", matches = "true")
class MetalTensorOpsTest extends TensorOpsConformance {
    @Override protected GpuProvider provider() { return new org.dynamisengine.gpu.metal.MetalProvider(); }
}
