package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class GpuMeshletStreamingPayloadTest {
  @Test
  void acceptsEmptyPayload() {
    ByteBuffer empty = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    GpuMeshletStreamingPayload payload =
        GpuMeshletStreamingPayload.fromLittleEndianBytes(0, 0, 5, empty);

    assertEquals(0, payload.streamUnitCount());
    assertEquals(0, payload.streamUnitsByteSize());
    assertEquals(0, payload.streamUnitsIntCount());
    assertEquals(20, payload.streamUnitsStrideBytes());
  }

  @Test
  void preservesLittleEndianLayoutAndCounts() {
    ByteBuffer bytes = ByteBuffer.allocate(40).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0)
        .putInt(0)
        .putInt(32)
        .putInt(0)
        .putInt(4096)
        .putInt(1)
        .putInt(32)
        .putInt(16)
        .putInt(4096)
        .putInt(2048)
        .flip();

    GpuMeshletStreamingPayload payload =
        GpuMeshletStreamingPayload.fromLittleEndianBytes(2, 0, 5, bytes);

    assertEquals(2, payload.streamUnitCount());
    assertEquals(40, payload.streamUnitsByteSize());
    assertEquals(10, payload.streamUnitsIntCount());
    assertEquals(10, payload.expectedStreamUnitsIntCount());
    assertEquals(ByteOrder.LITTLE_ENDIAN, payload.streamUnitsBytes().order());
  }

  @Test
  void rejectsInconsistentCountStrideByteSize() {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuMeshletStreamingPayload.fromLittleEndianBytes(2, 0, 5, bytes));
  }

  @Test
  void rejectsNonIntAlignedBytePayload() {
    ByteBuffer bytes = ByteBuffer.allocate(10).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuMeshletStreamingPayload.fromLittleEndianBytes(0, 0, 5, bytes));
  }
}
