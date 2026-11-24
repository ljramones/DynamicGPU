package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.test.compute.MockComputeProvider;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Without a GPU: the kernel's ABI, the shape arithmetic, and refusal on a device without FP16. */
class GemmF16Test {
    @Test void kernelAbiMatchesTheEncoder() {
        for (String k : new String[] {"gemm_f16", "gemm_f16_h"}) {
            assertEquals(java.util.List.of("a", "b", "c"), GemmF16.KERNELS.buffers(k));
            assertEquals(48, GemmF16.KERNELS.parameterBytes(k));
            assertEquals(new org.dynamisengine.gpu.api.compute.Dimensions(16, 16, 1), GemmF16.KERNELS.localSize(k));
            assertEquals(java.util.List.of("m", "n", "k", "lda", "ldb", "ldc", "alpha", "beta", "transB", "strideA", "strideB", "strideC"),
                    java.util.List.copyOf(GemmF16.KERNELS.parameterOffsets(k).keySet()));
        }
    }

    @Test void shapeSizes() {
        var s = new GemmF16.Shape(3, 5, 7, 8, 6, 9, 1, 0);
        assertEquals(2L * (2 * 8 + 7), s.aBytes()); assertEquals(2L * (6 * 6 + 5), s.bBytes()); assertEquals(4L * (2 * 9 + 5), s.cBytes());
        assertEquals(2L * (2 * 9 + 5), s.cBytes(Precision.FP16));
        assertEquals(2.0 * 3 * 5 * 7, s.flops());
        var t = GemmF16.Shape.ofTransposedB(3, 5, 7, 1, 0); // B is 5 x 7
        assertEquals(7, t.ldb()); assertEquals(2L * (4 * 7 + 7), t.bBytes()); assertEquals(2 * 7 + 4, t.bIndex(0, 4, 2));
        var batch = GemmF16.Shape.of(3, 5, 7, 1, 0).batched(4);
        assertEquals(3 * 7, batch.strideA()); assertEquals(7 * 5, batch.strideB()); assertEquals(3 * 5, batch.strideC());
        assertEquals(4L * 3 * 7, batch.aElements()); assertEquals(4L * 3 * 5, batch.cElements()); assertEquals(4 * 2.0 * 3 * 5 * 7, batch.flops());
        var shared = GemmF16.Shape.of(3, 5, 7, 1, 0).withBatch(4, 21, 0, 15);
        assertEquals(7L * 5, shared.bElements(), "a stride of 0 shares B");
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
