package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.junit.jupiter.api.Test;

class MeshletBoundsPayloadIngestionTest {
  @Test
  void ingestsValidPayloadMetadataAndBytes() {
    ByteBuffer bytes = ByteBuffer.allocate(48).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 12; i++) {
      bytes.putFloat(i + 1f);
    }
    bytes.flip();

    GpuMeshletBoundsPayload payload =
        MeshletBoundsPayloadIngestion.ingest(2, 0, 6, 12, 12, bytes);
    GpuMeshletBoundsResource resource =
        MeshletBoundsPayloadIngestion.toResource(new GpuBufferHandle(99L), payload);

    assertEquals(2, payload.meshletCount());
    assertEquals(48, payload.boundsByteSize());
    assertEquals(2, resource.meshletCount());
    assertEquals(99L, resource.bufferHandle().value());
  }

  @Test
  void ingestsEmptyPayloadWhenCountsAreConsistent() {
    ByteBuffer bytes = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    GpuMeshletBoundsPayload payload =
        MeshletBoundsPayloadIngestion.ingest(0, 0, 6, 0, 0, bytes);
    assertEquals(0, payload.meshletCount());
    assertEquals(0, payload.boundsByteSize());
  }

  @Test
  void rejectsMalformedUpstreamCountMismatch() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> MeshletBoundsPayloadIngestion.ingest(1, 0, 6, 6, 12, bytes));
  }

  @Test
  void rejectsPayloadByteCountMismatchAfterValidation() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> MeshletBoundsPayloadIngestion.ingest(2, 0, 6, 6, 6, bytes));
  }
}

