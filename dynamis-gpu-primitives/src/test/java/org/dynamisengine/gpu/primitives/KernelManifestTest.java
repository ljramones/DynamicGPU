package org.dynamisengine.gpu.primitives;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.junit.jupiter.api.Test;

/** The group-shared memory the primitives' kernels declare, read from their SPIR-V, within every device's minimum. */
class KernelManifestTest {
    @Test void groupSharedMemoryIsReadFromTheSpirv() {
        // softmax: red[256] floats, plus rowMax and rowSum
        assertEquals(256 * 4 + 4 + 4, GemmF16.KERNELS.groupSharedBytes("softmax_f32"));
        assertEquals(256 * 4 + 4 + 4, GemmF16.KERNELS.groupSharedBytes("softmax_f16"));
        // gemm: tileA and tileB, 16 x 64 floats each
        assertEquals(2 * 16 * 64 * 4, GemmF16.KERNELS.groupSharedBytes("gemm_f16"));
        for (String k : GemmF16.KERNELS.kernels()) {
            long bytes = GemmF16.KERNELS.groupSharedBytes(k);
            assertEquals(true, bytes >= 0 && bytes <= GpuCapabilities.MIN_GROUP_SHARED_BYTES, k + " declares " + bytes + " bytes");
        }
    }
}
