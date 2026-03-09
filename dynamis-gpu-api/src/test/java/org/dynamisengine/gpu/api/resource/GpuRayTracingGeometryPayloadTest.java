package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class GpuRayTracingGeometryPayloadTest {
  @Test
  void acceptsEmptyPayload() {
    ByteBuffer empty = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(0, 0, 5, empty);

    assertEquals(0, payload.regionCount());
    assertEquals(0, payload.regionsByteSize());
    assertEquals(0, payload.regionsIntCount());
    assertEquals(20, payload.regionsStrideBytes());
  }

  @Test
  void preservesLittleEndianLayoutAndCounts() {
    ByteBuffer bytes = ByteBuffer.allocate(40).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0)
        .putInt(0)
        .putInt(12)
        .putInt(0)
        .putInt(1)
        .putInt(1)
        .putInt(12)
        .putInt(9)
        .putInt(2)
        .putInt(0)
        .flip();

    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(2, 0, 5, bytes);

    assertEquals(2, payload.regionCount());
    assertEquals(40, payload.regionsByteSize());
    assertEquals(10, payload.regionsIntCount());
    assertEquals(10, payload.expectedRegionsIntCount());
    assertEquals(ByteOrder.LITTLE_ENDIAN, payload.regionsBytes().order());
  }

  @Test
  void rejectsInconsistentCountStrideByteSize() {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuRayTracingGeometryPayload.fromLittleEndianBytes(2, 0, 5, bytes));
  }

  @Test
  void rejectsNonIntAlignedBytePayload() {
    ByteBuffer bytes = ByteBuffer.allocate(10).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuRayTracingGeometryPayload.fromLittleEndianBytes(0, 0, 5, bytes));
  }
}

