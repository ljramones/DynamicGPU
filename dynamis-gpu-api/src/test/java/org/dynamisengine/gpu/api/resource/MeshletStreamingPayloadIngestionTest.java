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

class MeshletStreamingPayloadIngestionTest {
  @Test
  void ingestsValidPayloadMetadataAndBytes() {
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
        MeshletStreamingPayloadIngestion.ingest(2, 0, 5, 10, 10, bytes);
    GpuMeshletStreamingResource resource =
        MeshletStreamingPayloadIngestion.toResource(new TestBuffer(299L, payload.streamUnitsByteSize()), payload);

    assertEquals(2, payload.streamUnitCount());
    assertEquals(40, payload.streamUnitsByteSize());
    assertEquals(2, resource.streamUnitCount());
    assertEquals(299L, resource.bufferHandle().value());
  }

  @Test
  void ingestsEmptyPayloadWhenCountsAreConsistent() {
    ByteBuffer bytes = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    GpuMeshletStreamingPayload payload =
        MeshletStreamingPayloadIngestion.ingest(0, 0, 5, 0, 0, bytes);
    assertEquals(0, payload.streamUnitCount());
    assertEquals(0, payload.streamUnitsByteSize());
  }

  @Test
  void rejectsMalformedUpstreamCountMismatch() {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> MeshletStreamingPayloadIngestion.ingest(1, 0, 5, 5, 10, bytes));
  }

  @Test
  void rejectsPayloadByteCountMismatchAfterValidation() {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> MeshletStreamingPayloadIngestion.ingest(2, 0, 5, 5, 5, bytes));
  }

  @Test
  void rejectsResourceBufferSizeMismatch() {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    GpuMeshletStreamingPayload payload =
        MeshletStreamingPayloadIngestion.ingest(1, 0, 5, 5, 5, bytes);
    assertThrows(
        IllegalArgumentException.class,
        () -> MeshletStreamingPayloadIngestion.toResource(new TestBuffer(277L, 8), payload));
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
