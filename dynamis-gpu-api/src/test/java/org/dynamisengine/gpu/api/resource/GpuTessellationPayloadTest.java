package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class GpuTessellationPayloadTest {
  @Test
  void acceptsEmptyPayload() {
    ByteBuffer empty = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    GpuTessellationPayload payload = GpuTessellationPayload.fromLittleEndianBytes(0, 0, 6, empty);

    assertEquals(0, payload.regionCount());
    assertEquals(0, payload.regionsByteSize());
    assertEquals(0, payload.regionsIntCount());
    assertEquals(24, payload.regionsStrideBytes());
  }

  @Test
  void preservesLittleEndianLayoutAndCounts() {
    ByteBuffer bytes = ByteBuffer.allocate(48).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0)
        .putInt(0)
        .putInt(12)
        .putInt(3)
        .putInt(Float.floatToRawIntBits(1.0f))
        .putInt(0)
        .putInt(1)
        .putInt(12)
        .putInt(9)
        .putInt(4)
        .putInt(Float.floatToRawIntBits(2.0f))
        .putInt(2)
        .flip();

    GpuTessellationPayload payload = GpuTessellationPayload.fromLittleEndianBytes(2, 0, 6, bytes);

    assertEquals(2, payload.regionCount());
    assertEquals(48, payload.regionsByteSize());
    assertEquals(12, payload.regionsIntCount());
    assertEquals(12, payload.expectedRegionsIntCount());
    assertEquals(ByteOrder.LITTLE_ENDIAN, payload.regionsBytes().order());
  }

  @Test
  void rejectsInconsistentCountStrideByteSize() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuTessellationPayload.fromLittleEndianBytes(2, 0, 6, bytes));
  }

  @Test
  void rejectsNonIntAlignedBytePayload() {
    ByteBuffer bytes = ByteBuffer.allocate(10).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuTessellationPayload.fromLittleEndianBytes(0, 0, 6, bytes));
  }
}
