package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.junit.jupiter.api.Test;

class GpuRayTracingBlasResourceTest {
  @Test
  void exposesPayloadAndGeometryMetadataAndClosesOwnedBuffer() {
    GpuRayTracingGeometryResource geometryResource = createGeometryResource();
    GpuRayTracingBlasPayload payload =
        GpuRayTracingBlasPayload.forGeometryResource(geometryResource);
    TestBuffer buffer = new TestBuffer(8200L, payload.byteSize());
    GpuRayTracingBlasResource resource =
        new GpuRayTracingBlasResource(buffer, payload, geometryResource);

    assertEquals(8200L, resource.bufferHandle().value());
    assertEquals(geometryResource, resource.sourceGeometryResource());
    assertEquals(2, resource.regionCount());
    assertEquals(5 * Integer.BYTES, resource.regionsStrideBytes());
    assertEquals(2 * 5 * Integer.BYTES, resource.regionsByteSize());

    resource.close();
    resource.close();

    assertTrue(resource.isClosed());
    assertEquals(1, buffer.closeCount.get());
  }

  private static GpuRayTracingGeometryResource createGeometryResource() {
    ByteBuffer bytes = ByteBuffer.allocate(2 * 5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0).putInt(0).putInt(36).putInt(0).putInt(0)
        .putInt(1).putInt(36).putInt(42).putInt(0).putInt(1)
        .flip();
    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(2, 0, 5, bytes);
    return new GpuRayTracingGeometryResource(new TestBuffer(8201L, payload.regionsByteSize()), payload);
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
      return GpuBufferUsage.STORAGE;
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

