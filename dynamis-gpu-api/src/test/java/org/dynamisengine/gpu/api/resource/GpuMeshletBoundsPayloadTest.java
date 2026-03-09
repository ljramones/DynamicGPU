package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class GpuMeshletBoundsPayloadTest {
  @Test
  void acceptsEmptyPayload() {
    ByteBuffer empty = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    GpuMeshletBoundsPayload payload = GpuMeshletBoundsPayload.fromLittleEndianBytes(0, 0, 6, empty);

    assertEquals(0, payload.meshletCount());
    assertEquals(0, payload.boundsByteSize());
    assertEquals(0, payload.boundsFloatCount());
    assertEquals(24, payload.boundsStrideBytes());
  }

  @Test
  void preservesLittleEndianLayoutAndCounts() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putFloat(1f).putFloat(2f).putFloat(3f).putFloat(4f).putFloat(5f).putFloat(6f).flip();

    GpuMeshletBoundsPayload payload = GpuMeshletBoundsPayload.fromLittleEndianBytes(1, 0, 6, bytes);

    assertEquals(1, payload.meshletCount());
    assertEquals(24, payload.boundsByteSize());
    assertEquals(6, payload.boundsFloatCount());
    assertEquals(6, payload.expectedBoundsFloatCount());
    assertEquals(ByteOrder.LITTLE_ENDIAN, payload.boundsBytes().order());
  }

  @Test
  void rejectsInconsistentCountStrideByteSize() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class, () -> GpuMeshletBoundsPayload.fromLittleEndianBytes(2, 0, 6, bytes));
  }

  @Test
  void rejectsNonFloatAlignedBytePayload() {
    ByteBuffer bytes = ByteBuffer.allocate(10).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class, () -> GpuMeshletBoundsPayload.fromLittleEndianBytes(0, 0, 6, bytes));
  }
}

