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

class GpuMeshletBoundsResourceTest {
  @Test
  void exposesResourceMetadataFromPayload() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putFloat(1f).putFloat(2f).putFloat(3f).putFloat(4f).putFloat(5f).putFloat(6f).flip();
    GpuMeshletBoundsPayload payload = GpuMeshletBoundsPayload.fromLittleEndianBytes(1, 0, 6, bytes);

    TestBuffer buffer = new TestBuffer(42L, payload.boundsByteSize());
    GpuMeshletBoundsResource resource = new GpuMeshletBoundsResource(buffer, payload);

    assertEquals(1, resource.meshletCount());
    assertEquals(24, resource.boundsByteSize());
    assertEquals(24, resource.boundsStrideBytes());
    assertEquals(42L, resource.bufferHandle().value());
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
