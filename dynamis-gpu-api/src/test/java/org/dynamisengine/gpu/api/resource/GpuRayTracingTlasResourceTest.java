package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.junit.jupiter.api.Test;

class GpuRayTracingTlasResourceTest {
  @Test
  void exposesPayloadAndHandleAndClosesOnce() {
    GpuRayTracingTlasPayload payload = GpuRayTracingTlasPayload.of(2, 128);
    TestBuffer buffer = new TestBuffer(1500L, 512L);
    GpuRayTracingTlasResource resource =
        new GpuRayTracingTlasResource(buffer, payload, 7777L, null);

    assertEquals(1500L, resource.bufferHandle().value());
    assertEquals(2, resource.instanceCount());
    assertEquals(128, resource.instanceByteSize());
    assertEquals(7777L, resource.accelerationStructureHandle());

    resource.close();
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

