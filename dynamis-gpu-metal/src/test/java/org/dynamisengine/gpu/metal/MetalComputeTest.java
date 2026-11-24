package org.dynamisengine.gpu.metal;
import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.test.compute.ComputeConformance;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import java.util.Map;
@EnabledIfSystemProperty(named = "compute.metal.required", matches = "true")
class MetalComputeTest extends ComputeConformance {
    @org.junit.jupiter.api.Test void malformedShaderReportsNativeCompilerError() {
        PipelineDescriptor good; try (var p = pipeline(64)) { good = p.descriptor(); }
        var bad = new KernelBundle("invalid", "1", Map.of(Backend.METAL, new KernelBundle.Executable(KernelBundle.Format.MSL_SOURCE, "this is invalid MSL".getBytes(java.nio.charset.StandardCharsets.UTF_8))));
        var descriptor = new PipelineDescriptor(bad, "vector_add", good.layout(), good.localSize(), Map.of(), java.util.Set.of(), java.util.Set.of());
        var error = org.junit.jupiter.api.Assertions.assertThrows(ComputeException.class, () -> device.pipeline(descriptor));
        org.junit.jupiter.api.Assertions.assertEquals(ComputeException.Code.NATIVE_ERROR, error.code());
        org.junit.jupiter.api.Assertions.assertTrue(error.getMessage().contains("Metal"));
    }
    /** Runtime MSL is strict from macOS 15 (MTLMathModeSafe and the contract pragma); earlier systems may fuse. */
    @Override protected KernelBundle strictKernel() {
        if (Integer.parseInt(System.getProperty("os.version").split("\\.")[0]) < 15) return null;
        try (var stream = ComputeConformance.class.getResourceAsStream("/compute/fp_strict.metal")) {
            return new KernelBundle("fp-strict", "1", Map.of(Backend.METAL, new KernelBundle.Executable(KernelBundle.Format.MSL_SOURCE, stream.readAllBytes())));
        } catch (java.io.IOException e) { throw new AssertionError(e); }
    }
    /** Metal's fma is one rounding (MSL), and Apple GPUs implement it so. */
    @Override protected Boolean expectedFusedFma() { return true; }
    @Override protected GpuProvider provider() { return new MetalProvider(); }
    @Override protected KernelBundle kernel() {
        try {
            var metallib = MetalComputeTest.class.getResourceAsStream("/native/macos-" + (System.getProperty("os.arch").equals("aarch64") ? "arm64" : "x86_64") + "/vector_add.metallib");
            if (metallib != null) try (metallib) { return new KernelBundle("vector-add", "1", Map.of(Backend.METAL, new KernelBundle.Executable(KernelBundle.Format.METALLIB, metallib.readAllBytes()))); }
            try (var stream = ComputeConformance.class.getResourceAsStream("/compute/vector_add.metal")) {
                return new KernelBundle("vector-add", "1", Map.of(Backend.METAL, new KernelBundle.Executable(KernelBundle.Format.MSL_SOURCE, stream.readAllBytes())));
            }
        } catch (java.io.IOException e) { throw new AssertionError(e); }
    }
}
