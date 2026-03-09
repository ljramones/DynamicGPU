package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
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
        MeshletBoundsPayloadIngestion.toResource(new TestBuffer(99L, payload.boundsByteSize()), payload);

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

  @Test
  void rejectsResourceBufferSizeMismatch() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    GpuMeshletBoundsPayload payload =
        MeshletBoundsPayloadIngestion.ingest(1, 0, 6, 6, 6, bytes);
    assertThrows(
        IllegalArgumentException.class,
        () -> MeshletBoundsPayloadIngestion.toResource(new TestBuffer(77L, 16), payload));
  }

  private static final class TestBuffer implements GpuBuffer {
    private final GpuBufferHandle handle;
    private final long sizeBytes;
    private final AtomicInteger closeCount = new AtomicInteger();

    private TestBuffer(long handle, long sizeBytes) {
      this.handle = new GpuBufferHandle(handle);
      this.sizeBytes = sizeBytes;
    }

    @Override
    public GpuBufferHandle handle() {
      return handle;
    }

    @Override
    public long sizeBytes() {
      return sizeBytes;
    }

    @Override
    public GpuBufferUsage usage() {
      return GpuBufferUsage.TRANSFER_DST;
    }

    @Override
    public GpuMemoryLocation memoryLocation() {
      return GpuMemoryLocation.DEVICE_LOCAL;
    }

    @Override
    public void close() {
      closeCount.incrementAndGet();
    }
  }
}
