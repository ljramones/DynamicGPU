package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.junit.jupiter.api.Test;

class GpuRayTracingBuildInputResourceTest {
  @Test
  void exposesResolvedDeviceAddressesAndMetadata() {
    GpuRayTracingBuildInputPayload payload = createBuildInputPayload();
    GpuRayTracingBuildInputResource resource =
        new GpuRayTracingBuildInputResource(payload, 9001L, 9002L);

    assertEquals(9001L, resource.vertexBufferDeviceAddress());
    assertEquals(9002L, resource.indexBufferDeviceAddress());
    assertEquals(1, resource.regionCount());
    assertEquals(32, resource.vertexStrideBytes());
    assertEquals(1024, resource.maxVertexIndex());

    resource.close();
    assertTrue(resource.isClosed());
  }

  @Test
  void rejectsInvalidDeviceAddresses() {
    GpuRayTracingBuildInputPayload payload = createBuildInputPayload();
    assertThrows(IllegalArgumentException.class, () -> new GpuRayTracingBuildInputResource(payload, 0L, 1L));
    assertThrows(IllegalArgumentException.class, () -> new GpuRayTracingBuildInputResource(payload, 1L, 0L));
  }

  private static GpuRayTracingBuildInputPayload createBuildInputPayload() {
    ByteBuffer bytes = ByteBuffer.allocate(5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(36).putInt(0).putInt(0).flip();
    GpuRayTracingGeometryPayload geometryPayload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(1, 0, 5, bytes);
    GpuRayTracingGeometryResource geometryResource =
        new GpuRayTracingGeometryResource(new TestBuffer(7100L, geometryPayload.regionsByteSize()), geometryPayload);
    return GpuRayTracingBuildInputPayload.of(
        geometryResource, new GpuBufferHandle(7101L), new GpuBufferHandle(7102L), 32, 1024, 0L, 0L);
  }

  private static final class TestBuffer implements GpuBuffer {
    private final GpuBufferHandle handle;
    private final long sizeBytes;

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
    public void close() {}
  }
}

