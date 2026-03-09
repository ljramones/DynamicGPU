package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class GpuMeshletIndirectDrawPayloadTest {
  @Test
  void storesPackedCommandBytes() {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(12).putInt(1).putInt(24).putInt(4).putInt(0).flip();

    GpuMeshletIndirectDrawPayload payload =
        GpuMeshletIndirectDrawPayload.fromLittleEndianBytes(1, 1, bytes);

    assertEquals(1, payload.sourceVisibleMeshletCount());
    assertEquals(1, payload.commandCount());
    assertEquals(20, payload.commandByteSize());
    ByteBuffer view = payload.commandBytes().order(ByteOrder.LITTLE_ENDIAN);
    assertEquals(12, view.getInt(0));
    assertEquals(24, view.getInt(8));
    assertEquals(4, view.getInt(12));
  }

  @Test
  void rejectsCommandCountLargerThanVisibleCount() {
    ByteBuffer bytes = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuMeshletIndirectDrawPayload.fromLittleEndianBytes(1, 2, bytes));
  }
}

