package org.dynamisengine.gpu.api.resource;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.layout.SubmeshRange;
import org.dynamisengine.gpu.api.layout.VertexAttribute;
import org.dynamisengine.gpu.api.layout.VertexFormat;
import org.dynamisengine.gpu.api.layout.VertexLayout;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GpuMeshResourceTest {

  @Test
  void closesOwnedBuffersOnce() {
    TestBuffer vertex = new TestBuffer(1L, GpuBufferUsage.VERTEX);
    TestBuffer index = new TestBuffer(2L, GpuBufferUsage.INDEX);
    VertexLayout layout =
        new VertexLayout(12, List.of(new VertexAttribute(0, 0, VertexFormat.FLOAT3)));
    GpuMeshResource resource =
        new GpuMeshResource(
            vertex, index, layout, org.dynamisengine.gpu.api.layout.IndexType.UINT32, List.of(new SubmeshRange(0, 3, 0)));

    assertTrue(resource.isIndexed());
    assertFalse(resource.isClosed());

    resource.close();
    resource.close();

    assertTrue(resource.isClosed());
    assertEquals(1, vertex.closeCount.get());
    assertEquals(1, index.closeCount.get());
  }

  private static final class TestBuffer implements GpuBuffer {
    private final GpuBufferHandle handle;
    private final GpuBufferUsage usage;
    private final AtomicInteger closeCount = new AtomicInteger();

    private TestBuffer(long handle, GpuBufferUsage usage) {
      this.handle = new GpuBufferHandle(handle);
      this.usage = usage;
    }

    @Override
    public GpuBufferHandle handle() {
      return handle;
    }

    @Override
    public long sizeBytes() {
      return 64L;
    }

    @Override
    public GpuBufferUsage usage() {
      return usage;
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
