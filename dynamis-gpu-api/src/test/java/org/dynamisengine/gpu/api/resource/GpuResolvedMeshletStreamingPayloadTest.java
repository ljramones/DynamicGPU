package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class GpuResolvedMeshletStreamingPayloadTest {
  @Test
  void createsPayloadWithExpectedMetadata() {
    GpuResolvedMeshletStreamingPayload payload =
        GpuResolvedMeshletStreamingPayload.of(5, 64, 32, 8192, 4096);

    assertEquals(5, payload.streamUnitId());
    assertEquals(64, payload.meshletStart());
    assertEquals(32, payload.meshletCount());
    assertEquals(8192, payload.payloadByteOffset());
    assertEquals(4096, payload.payloadByteSize());
    assertEquals(20, payload.byteSize());
    assertEquals(ByteOrder.LITTLE_ENDIAN, payload.resolvedBytes().order());
  }

  @Test
  void decodesLittleEndianBytes() {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(9).putInt(128).putInt(16).putInt(12288).putInt(2048).flip();

    GpuResolvedMeshletStreamingPayload payload =
        GpuResolvedMeshletStreamingPayload.fromLittleEndianBytes(bytes);

    assertEquals(9, payload.streamUnitId());
    assertEquals(128, payload.meshletStart());
    assertEquals(16, payload.meshletCount());
    assertEquals(12288, payload.payloadByteOffset());
    assertEquals(2048, payload.payloadByteSize());
  }

  @Test
  void rejectsInvalidMeshletCount() {
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuResolvedMeshletStreamingPayload.of(0, 0, 0, 0, 1024));
  }

  @Test
  void rejectsInvalidPayloadByteSize() {
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuResolvedMeshletStreamingPayload.of(0, 0, 1, 0, 0));
  }

  @Test
  void rejectsWrongByteSizeOnDecode() {
    ByteBuffer bad = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class, () -> GpuResolvedMeshletStreamingPayload.fromLittleEndianBytes(bad));
  }
}

