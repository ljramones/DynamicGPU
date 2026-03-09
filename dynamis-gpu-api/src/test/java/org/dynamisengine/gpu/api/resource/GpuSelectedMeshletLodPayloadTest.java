package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class GpuSelectedMeshletLodPayloadTest {
  @Test
  void createsPayloadWithExpectedMetadata() {
    int errorBits = Float.floatToRawIntBits(0.75f);
    GpuSelectedMeshletLodPayload payload = GpuSelectedMeshletLodPayload.of(1, 64, 32, errorBits);

    assertEquals(1, payload.selectedLodLevel());
    assertEquals(64, payload.meshletStart());
    assertEquals(32, payload.meshletCount());
    assertEquals(errorBits, payload.geometricErrorBits());
    assertEquals(0.75f, payload.geometricError());
    assertEquals(16, payload.byteSize());
    assertEquals(ByteOrder.LITTLE_ENDIAN, payload.selectedBytes().order());
  }

  @Test
  void decodesLittleEndianBytes() {
    int errorBits = Float.floatToRawIntBits(1.25f);
    ByteBuffer bytes = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(2).putInt(96).putInt(8).putInt(errorBits).flip();

    GpuSelectedMeshletLodPayload payload = GpuSelectedMeshletLodPayload.fromLittleEndianBytes(bytes);

    assertEquals(2, payload.selectedLodLevel());
    assertEquals(96, payload.meshletStart());
    assertEquals(8, payload.meshletCount());
    assertEquals(errorBits, payload.geometricErrorBits());
  }

  @Test
  void rejectsInvalidMeshletCount() {
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuSelectedMeshletLodPayload.of(0, 0, 0, Float.floatToRawIntBits(0.0f)));
  }

  @Test
  void rejectsWrongByteSizeOnDecode() {
    ByteBuffer bad = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(IllegalArgumentException.class, () -> GpuSelectedMeshletLodPayload.fromLittleEndianBytes(bad));
  }
}
