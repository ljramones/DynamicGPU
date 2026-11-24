package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.test.compute.MockComputeProvider;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Without a GPU: the kernel's ABI, the shape arithmetic, and refusal on a device without FP16. */
class GemmF16Test {
    @Test void kernelAbiMatchesTheEncoder() {
        assertEquals(java.util.List.of("a", "b", "c"), GemmF16.KERNELS.buffers("gemm_f16"));
        assertEquals(32, GemmF16.KERNELS.parameterBytes("gemm_f16"));
        assertEquals(new org.dynamisengine.gpu.api.compute.Dimensions(16, 16, 1), GemmF16.KERNELS.localSize("gemm_f16"));
        assertEquals(java.util.List.of("m", "n", "k", "lda", "ldb", "ldc", "alpha", "beta"),
                java.util.List.copyOf(GemmF16.KERNELS.parameterOffsets("gemm_f16").keySet()));
    }

    @Test void shapeSizes() {
        var s = new GemmF16.Shape(3, 5, 7, 8, 6, 9, 1, 0);
        assertEquals(2L * (2 * 8 + 7), s.aBytes()); assertEquals(2L * (6 * 6 + 5), s.bBytes()); assertEquals(4L * (2 * 9 + 5), s.cBytes());
        assertEquals(2.0 * 3 * 5 * 7, s.flops());
    }

    @Test void refusesDevicesWithoutFp16() {
        try (var device = new MockComputeProvider().adapters().getFirst().open()) {
            assertEquals(ComputeException.Code.UNSUPPORTED, assertThrows(ComputeException.class, () -> GemmF16.create(device)).code());
            for (var algorithm : GemmF16.Algorithm.values()) {
                assertFalse(GemmF16.supports(device, algorithm));
                assertEquals(ComputeException.Code.UNSUPPORTED, assertThrows(ComputeException.class, () -> GemmF16.create(device, algorithm)).code());
            }
        }
    }
}
