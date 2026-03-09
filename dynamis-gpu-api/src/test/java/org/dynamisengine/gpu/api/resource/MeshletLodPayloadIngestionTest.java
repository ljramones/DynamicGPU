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

class MeshletLodPayloadIngestionTest {
  @Test
  void ingestsValidPayloadMetadataAndBytes() {
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

    GpuMeshletLodPayload payload =
        MeshletLodPayloadIngestion.ingest(2, 0, 4, 8, 8, bytes);
    GpuMeshletLodResource resource =
        MeshletLodPayloadIngestion.toResource(new TestBuffer(199L, payload.levelsByteSize()), payload);

    assertEquals(2, payload.levelCount());
    assertEquals(32, payload.levelsByteSize());
    assertEquals(2, resource.levelCount());
    assertEquals(199L, resource.bufferHandle().value());
  }

  @Test
  void ingestsEmptyPayloadWhenCountsAreConsistent() {
    ByteBuffer bytes = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    GpuMeshletLodPayload payload = MeshletLodPayloadIngestion.ingest(0, 0, 4, 0, 0, bytes);
    assertEquals(0, payload.levelCount());
    assertEquals(0, payload.levelsByteSize());
  }

  @Test
  void rejectsMalformedUpstreamCountMismatch() {
    ByteBuffer bytes = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> MeshletLodPayloadIngestion.ingest(1, 0, 4, 4, 8, bytes));
  }

  @Test
  void rejectsPayloadByteCountMismatchAfterValidation() {
    ByteBuffer bytes = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> MeshletLodPayloadIngestion.ingest(2, 0, 4, 4, 4, bytes));
  }

  @Test
  void rejectsResourceBufferSizeMismatch() {
    ByteBuffer bytes = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    GpuMeshletLodPayload payload = MeshletLodPayloadIngestion.ingest(1, 0, 4, 4, 4, bytes);
    assertThrows(
        IllegalArgumentException.class,
        () -> MeshletLodPayloadIngestion.toResource(new TestBuffer(177L, 8), payload));
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
