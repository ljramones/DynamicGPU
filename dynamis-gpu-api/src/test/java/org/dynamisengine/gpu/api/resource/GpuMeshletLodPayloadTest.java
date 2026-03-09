package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class GpuMeshletLodPayloadTest {
  @Test
  void acceptsEmptyPayload() {
    ByteBuffer empty = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    GpuMeshletLodPayload payload = GpuMeshletLodPayload.fromLittleEndianBytes(0, 0, 4, empty);

    assertEquals(0, payload.levelCount());
    assertEquals(0, payload.levelsByteSize());
    assertEquals(0, payload.levelsIntCount());
    assertEquals(16, payload.levelsStrideBytes());
  }

  @Test
  void preservesLittleEndianLayoutAndCounts() {
    ByteBuffer bytes = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0)
        .putInt(0)
        .putInt(64)
        .putInt(Float.floatToRawIntBits(0.0f))
        .putInt(1)
        .putInt(64)
        .putInt(32)
        .putInt(Float.floatToRawIntBits(0.75f))
        .flip();

    GpuMeshletLodPayload payload = GpuMeshletLodPayload.fromLittleEndianBytes(2, 0, 4, bytes);

    assertEquals(2, payload.levelCount());
    assertEquals(32, payload.levelsByteSize());
    assertEquals(8, payload.levelsIntCount());
    assertEquals(8, payload.expectedLevelsIntCount());
    assertEquals(ByteOrder.LITTLE_ENDIAN, payload.levelsBytes().order());
  }

  @Test
  void rejectsInconsistentCountStrideByteSize() {
    ByteBuffer bytes = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class, () -> GpuMeshletLodPayload.fromLittleEndianBytes(2, 0, 4, bytes));
  }

  @Test
  void rejectsNonIntAlignedBytePayload() {
    ByteBuffer bytes = ByteBuffer.allocate(10).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class, () -> GpuMeshletLodPayload.fromLittleEndianBytes(0, 0, 4, bytes));
  }
}
