package org.dynamisengine.gpu.primitives;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.junit.jupiter.api.Test;

/**
 * The group-shared memory the primitives' kernels declare, read from their SPIR-V: within every device's minimum, but
 * for the fused attention's, which AttentionF16.fusedSupports and backwardSupports check against the device (at most
 * 32 KiB).
 */
class KernelManifestTest {
    @Test void wideFlashUsesTheSameBindingsAndGroupMemory() {
        var kernels = GemmF16.KERNELS;
        for (String suffix : new String[] {"", "_f32"}) {
            String base = "attn_flash_128" + suffix, wide = "attn_flash_128_wide" + suffix;
            assertEquals(256, kernels.localSize(wide).x());
            assertEquals(kernels.buffers(base), kernels.buffers(wide));
            assertEquals(kernels.parameterOffsets(base), kernels.parameterOffsets(wide));
            assertEquals(kernels.groupSharedBytes(base), kernels.groupSharedBytes(wide));
        }
    }

    @Test void groupSharedMemoryIsReadFromTheSpirv() {
        // softmax: red[256] floats, plus rowMax and rowSum
        assertEquals(256 * 4 + 4 + 4, GemmF16.KERNELS.groupSharedBytes("softmax_f32"));
        assertEquals(256 * 4 + 4 + 4, GemmF16.KERNELS.groupSharedBytes("softmax_f16"));
        // gemm: tileA and tileB, 16 x 64 floats each
        assertEquals(2 * 16 * 64 * 4, GemmF16.KERNELS.groupSharedBytes("gemm_f16"));
        for (String k : GemmF16.KERNELS.kernels()) {
            long bytes = GemmF16.KERNELS.groupSharedBytes(k);
            // the fused attention's kernels, forward and backward (1.12.0), which fusedSupports and backwardSupports check
            boolean attention = k.startsWith("attn_flash_") || k.startsWith("attn_decode_") || k.startsWith("attn_bwd_");
            long limit = attention ? 32768 : GpuCapabilities.MIN_GROUP_SHARED_BYTES;
            assertEquals(true, bytes >= 0 && bytes <= limit, k + " declares " + bytes + " bytes");
        }
        for (int variant : new int[] {64, 128, 256}) assertEquals(true, AttentionF16.groupBytes(variant) > 0);
    }
}
