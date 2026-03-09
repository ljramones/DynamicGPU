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

class RayTracingGeometryPayloadIngestionTest {
  @Test
  void ingestsValidPayloadMetadataAndBytes() {
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
        RayTracingGeometryPayloadIngestion.ingest(2, 0, 5, 10, 10, bytes);
    GpuRayTracingGeometryResource resource =
        RayTracingGeometryPayloadIngestion.toResource(new TestBuffer(499L, payload.regionsByteSize()), payload);

    assertEquals(2, payload.regionCount());
    assertEquals(40, payload.regionsByteSize());
    assertEquals(2, resource.regionCount());
    assertEquals(499L, resource.bufferHandle().value());
  }

  @Test
  void ingestsEmptyPayloadWhenCountsAreConsistent() {
    ByteBuffer bytes = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    GpuRayTracingGeometryPayload payload =
        RayTracingGeometryPayloadIngestion.ingest(0, 0, 5, 0, 0, bytes);
    assertEquals(0, payload.regionCount());
    assertEquals(0, payload.regionsByteSize());
  }

  @Test
  void rejectsMalformedUpstreamCountMismatch() {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> RayTracingGeometryPayloadIngestion.ingest(1, 0, 5, 5, 10, bytes));
  }

  @Test
  void rejectsPayloadByteCountMismatchAfterValidation() {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> RayTracingGeometryPayloadIngestion.ingest(2, 0, 5, 5, 5, bytes));
  }

  @Test
  void rejectsResourceBufferSizeMismatch() {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    GpuRayTracingGeometryPayload payload =
        RayTracingGeometryPayloadIngestion.ingest(1, 0, 5, 5, 5, bytes);
    assertThrows(
        IllegalArgumentException.class,
        () -> RayTracingGeometryPayloadIngestion.toResource(new TestBuffer(477L, 8), payload));
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

