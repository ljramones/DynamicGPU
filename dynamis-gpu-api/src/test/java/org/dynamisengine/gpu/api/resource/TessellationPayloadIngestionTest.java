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

class TessellationPayloadIngestionTest {
  @Test
  void ingestsValidPayloadMetadataAndBytes() {
    ByteBuffer bytes = ByteBuffer.allocate(48).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0)
        .putInt(0)
        .putInt(12)
        .putInt(3)
        .putInt(Float.floatToRawIntBits(1.0f))
        .putInt(0)
        .putInt(1)
        .putInt(12)
        .putInt(9)
        .putInt(4)
        .putInt(Float.floatToRawIntBits(2.0f))
        .putInt(2)
        .flip();

    GpuTessellationPayload payload = TessellationPayloadIngestion.ingest(2, 0, 6, 12, 12, bytes);
    GpuTessellationResource resource =
        TessellationPayloadIngestion.toResource(new TestBuffer(599L, payload.regionsByteSize()), payload);

    assertEquals(2, payload.regionCount());
    assertEquals(48, payload.regionsByteSize());
    assertEquals(2, resource.regionCount());
    assertEquals(599L, resource.bufferHandle().value());
  }

  @Test
  void ingestsEmptyPayloadWhenCountsAreConsistent() {
    ByteBuffer bytes = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
    GpuTessellationPayload payload = TessellationPayloadIngestion.ingest(0, 0, 6, 0, 0, bytes);
    assertEquals(0, payload.regionCount());
    assertEquals(0, payload.regionsByteSize());
  }

  @Test
  void rejectsMalformedUpstreamCountMismatch() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> TessellationPayloadIngestion.ingest(1, 0, 6, 6, 12, bytes));
  }

  @Test
  void rejectsPayloadByteCountMismatchAfterValidation() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    assertThrows(
        IllegalArgumentException.class,
        () -> TessellationPayloadIngestion.ingest(2, 0, 6, 6, 6, bytes));
  }

  @Test
  void rejectsResourceBufferSizeMismatch() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    GpuTessellationPayload payload = TessellationPayloadIngestion.ingest(1, 0, 6, 6, 6, bytes);
    assertThrows(
        IllegalArgumentException.class,
        () -> TessellationPayloadIngestion.toResource(new TestBuffer(577L, 8), payload));
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
