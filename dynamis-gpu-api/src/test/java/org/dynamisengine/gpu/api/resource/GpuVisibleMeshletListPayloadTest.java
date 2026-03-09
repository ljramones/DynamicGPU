package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class GpuVisibleMeshletListPayloadTest {
  @Test
  void storesLittleEndianVisibleIndices() {
    ByteBuffer bytes = ByteBuffer.allocate(3 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(1).putInt(3).putInt(4).flip();

    GpuVisibleMeshletListPayload payload =
        GpuVisibleMeshletListPayload.fromLittleEndianBytes(6, 3, bytes);

    assertEquals(6, payload.sourceMeshletCount());
    assertEquals(3, payload.visibleMeshletCount());
    assertEquals(12, payload.visibleIndicesByteSize());
    ByteBuffer view = payload.visibleIndicesBytes();
    assertEquals(1, view.getInt(0));
    assertEquals(3, view.getInt(4));
    assertEquals(4, view.getInt(8));
  }

  @Test
  void rejectsVisibleCountGreaterThanSourceCount() {
    ByteBuffer bytes = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuVisibleMeshletListPayload.fromLittleEndianBytes(2, 3, bytes));
  }

  @Test
  void rejectsByteCountMismatch() {
    ByteBuffer bytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(9).flip();
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuVisibleMeshletListPayload.fromLittleEndianBytes(4, 2, bytes));
  }
}

