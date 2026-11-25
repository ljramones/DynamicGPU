package org.dynamisengine.gpu.primitives;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.junit.jupiter.api.Test;

/**
 * The group-shared memory the primitives' kernels declare, read from their SPIR-V: within every device's minimum, but
 * for the fused attention's, which AttentionF16.fusedSupports checks against the device (at most 32 KiB).
 */
class KernelManifestTest {
    @Test void groupSharedMemoryIsReadFromTheSpirv() {
        // softmax: red[256] floats, plus rowMax and rowSum
        assertEquals(256 * 4 + 4 + 4, GemmF16.KERNELS.groupSharedBytes("softmax_f32"));
        assertEquals(256 * 4 + 4 + 4, GemmF16.KERNELS.groupSharedBytes("softmax_f16"));
        // gemm: tileA and tileB, 16 x 64 floats each
        assertEquals(2 * 16 * 64 * 4, GemmF16.KERNELS.groupSharedBytes("gemm_f16"));
        for (String k : GemmF16.KERNELS.kernels()) {
            long bytes = GemmF16.KERNELS.groupSharedBytes(k);
            boolean attention = k.startsWith("attn_flash_") || k.startsWith("attn_decode_");
            long limit = attention ? 32768 : GpuCapabilities.MIN_GROUP_SHARED_BYTES;
            assertEquals(true, bytes >= 0 && bytes <= limit, k + " declares " + bytes + " bytes");
        }
        for (int variant : new int[] {64, 128, 256}) assertEquals(true, AttentionF16.groupBytes(variant) > 0);
    }
}
