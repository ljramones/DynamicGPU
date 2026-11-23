package org.dynamisengine.gpu.vulkan.compute;

import org.dynamisengine.gpu.api.compute.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;

/** Minimal structural and fixed LocalSize validation before passing executables to a driver. */
final class SpirvWorkgroup {
    static void validate(byte[] bytes, PipelineDescriptor descriptor) {
        if (bytes.length < 20 || bytes.length % 4 != 0) fail("SPIR-V must contain a complete word-aligned header");
        var b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        if (b.getInt(0) != 0x07230203) fail("Invalid SPIR-V magic");
        int entryId = -1;
        for (int offset = 20; offset < bytes.length;) {
            int op = b.getInt(offset), count = op >>> 16, code = op & 65535;
            if (count == 0 || count > (bytes.length - offset) / 4) fail("Invalid SPIR-V instruction length");
            if (code == 15 && count >= 4 && b.getInt(offset + 4) == 5) {
                int end = offset + 12; while (end < offset + count * 4 && bytes[end] != 0) end++;
                String name = new String(bytes, offset + 12, end - offset - 12, StandardCharsets.UTF_8);
                if (name.equals(descriptor.entryPoint())) entryId = b.getInt(offset + 8);
            }
            offset += count * 4;
        }
        if (entryId < 0) fail("SPIR-V compute entry point not found: " + descriptor.entryPoint());
        for (int offset = 20; offset < bytes.length;) {
            int op = b.getInt(offset), count = op >>> 16, code = op & 65535;
            if (code == 16 && count == 6 && b.getInt(offset + 4) == entryId && b.getInt(offset + 8) == 17) {
                var local = new Dimensions(b.getInt(offset + 12), b.getInt(offset + 16), b.getInt(offset + 20));
                if (!local.equals(descriptor.localSize())) fail("SPIR-V LocalSize " + local + " differs from pipeline " + descriptor.localSize()); return;
            }
            offset += count * 4;
        }
        throw new ComputeException(ComputeException.Code.UNSUPPORTED, "SPIR-V requires fixed LocalSize; LocalSizeId is not supported by this runtime version");
    }
    private static void fail(String text) { throw new ComputeException(ComputeException.Code.INVALID_ARGUMENT, text); }
}
