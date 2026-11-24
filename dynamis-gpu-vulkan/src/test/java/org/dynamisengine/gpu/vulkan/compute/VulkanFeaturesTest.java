package org.dynamisengine.gpu.vulkan.compute;

import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.junit.jupiter.api.Test;
import java.io.IOException;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.*;

/** Feature reporting and SPIR-V capability checks, without a device. */
class VulkanFeaturesTest {
    private static final int ALL_SUBGROUP_OPS = 0x1F;
    private static final VulkanFeatures NONE = new VulkanFeatures(false, false, false, false, false, false, false, false, false, 0, 0, 0, true);
    private static final VulkanFeatures FULL = new VulkanFeatures(true, true, true, true, true, true, true, true, true, ALL_SUBGROUP_OPS, 4, 64, true);

    private static byte[] tier(String name) throws IOException {
        try (var in = VulkanFeaturesTest.class.getResourceAsStream("/compute/tiers/" + name + ".spv")) { return in.readAllBytes(); }
    }

    @Test void reportsOnlyWhatIsSupported() {
        assertEquals(Set.of("FP32", "INT32", "BOOL"), NONE.dataTypes());
        assertEquals(Set.of(), NONE.shaderFeatures());
        assertEquals(Set.of("FP32", "INT32", "BOOL", "FP16", "INT16", "INT8", "INT64", "FP64"), FULL.dataTypes());
        assertTrue(FULL.shaderFeatures().containsAll(Set.of(GpuCapabilities.SUBGROUP_SHUFFLE, GpuCapabilities.ATOMIC_INT64, GpuCapabilities.ATOMIC_FLOAT32_ADD)));
        var halfOnly = new VulkanFeatures(false, false, false, false, true, false, false, false, false, 0, 0, 0, true);
        assertFalse(halfOnly.dataTypes().contains(GpuCapabilities.FP16), "FP16 needs 16-bit storage too");
    }

    @Test void readsDeclaredCapabilities() throws IOException {
        assertTrue(VulkanFeatures.declaredCapabilities(tier("fp16_arith")).containsAll(Set.of(1, 9, 4434)), "Shader, Float16, 16-bit access");
        assertTrue(VulkanFeatures.declaredCapabilities(tier("atomic_int64")).containsAll(Set.of(11, 12)));
    }

    @Test void refusesModulesNeedingDisabledFeatures() throws IOException {
        for (String k : new String[] {"fp16_arith", "int8_arith", "int16_arith", "int64_arith", "fp64_arith", "subgroup_arithmetic",
                "subgroup_ballot", "subgroup_shuffle", "subgroup_vote", "atomic_float_add", "atomic_int64"}) {
            byte[] spv = tier(k);
            assertEquals(ComputeException.Code.UNSUPPORTED, assertThrows(ComputeException.class, () -> NONE.checkModule(spv, k)).code(), k);
            FULL.checkModule(spv, k);
        }
    }
}
