package org.dynamisengine.gpu.vulkan.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBuildInputPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBuildInputResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryResource;
import org.junit.jupiter.api.Test;

class VulkanRayTracingBuildInputResolverTest {
  @Test
  void resolvesDeviceAddressesIntoBuildInputResource() throws Exception {
    GpuRayTracingBuildInputPayload payload = createBuildInputPayload();
    VulkanRayTracingBuildInputResolver resolver =
        new VulkanRayTracingBuildInputResolver(handle -> handle + 1000L);

    GpuRayTracingBuildInputResource resource = resolver.resolve(payload);

    assertEquals(8001L, resource.vertexBufferDeviceAddress());
    assertEquals(8002L, resource.indexBufferDeviceAddress());
    assertEquals(1, resource.regionCount());
  }

  @Test
  void rejectsInvalidResolvedAddress() {
    GpuRayTracingBuildInputPayload payload = createBuildInputPayload();
    VulkanRayTracingBuildInputResolver resolver =
        new VulkanRayTracingBuildInputResolver(handle -> 0L);

    assertThrows(GpuException.class, () -> resolver.resolve(payload));
  }

  private static GpuRayTracingBuildInputPayload createBuildInputPayload() {
    ByteBuffer bytes = ByteBuffer.allocate(5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(36).putInt(0).putInt(0).flip();
    GpuRayTracingGeometryPayload geometryPayload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(1, 0, 5, bytes);
    GpuRayTracingGeometryResource geometryResource =
        new GpuRayTracingGeometryResource(new TestBuffer(7300L, geometryPayload.regionsByteSize()), geometryPayload);
    return GpuRayTracingBuildInputPayload.of(
        geometryResource, new GpuBufferHandle(7001L), new GpuBufferHandle(7002L), 32, 1024, 0L, 0L);
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

