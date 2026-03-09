package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.junit.jupiter.api.Test;

class GpuTessellationResourceTest {
  @Test
  void exposesResourceMetadataFromPayload() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0)
        .putInt(0)
        .putInt(12)
        .putInt(3)
        .putInt(Float.floatToRawIntBits(1.0f))
        .putInt(0)
        .flip();
    GpuTessellationPayload payload = GpuTessellationPayload.fromLittleEndianBytes(1, 0, 6, bytes);

    TestBuffer buffer = new TestBuffer(742L, payload.regionsByteSize());
    GpuTessellationResource resource = new GpuTessellationResource(buffer, payload);

    assertEquals(1, resource.regionCount());
    assertEquals(24, resource.regionsByteSize());
    assertEquals(24, resource.regionsStrideBytes());
    assertEquals(742L, resource.bufferHandle().value());
    assertFalse(resource.isClosed());

    resource.close();
    assertTrue(resource.isClosed());
    assertEquals(1, buffer.closeCount.get());
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
