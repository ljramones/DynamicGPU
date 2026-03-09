package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import org.junit.jupiter.api.Test;

class GpuMeshletVisibilityFlagsPayloadTest {
  @Test
  void storesOneBytePerMeshlet() {
    ByteBuffer bytes = ByteBuffer.wrap(new byte[] {1, 0, 1});
    GpuMeshletVisibilityFlagsPayload payload = GpuMeshletVisibilityFlagsPayload.fromBytes(3, bytes);

    assertEquals(3, payload.meshletCount());
    assertEquals(3, payload.flagsByteSize());
    ByteBuffer view = payload.flagsBytes();
    assertEquals((byte) 1, view.get(0));
    assertEquals((byte) 0, view.get(1));
    assertEquals((byte) 1, view.get(2));
  }

  @Test
  void rejectsByteCountMismatch() {
    ByteBuffer bytes = ByteBuffer.wrap(new byte[] {1, 0});
    assertThrows(IllegalArgumentException.class, () -> GpuMeshletVisibilityFlagsPayload.fromBytes(3, bytes));
  }
}

