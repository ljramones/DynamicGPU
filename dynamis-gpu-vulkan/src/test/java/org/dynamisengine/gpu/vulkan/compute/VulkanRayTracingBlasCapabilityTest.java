package org.dynamisengine.gpu.vulkan.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.gpu.RayTracingBlasWork;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryResource;
import org.junit.jupiter.api.Test;

class VulkanRayTracingBlasCapabilityTest {
  @Test
  void producesBlasBuildPrepResourceFromGeometryInput() throws Exception {
    GpuRayTracingGeometryResource geometryResource = createGeometryResource();
    VulkanRayTracingBlasCapability capability =
        new VulkanRayTracingBlasCapability(src -> new TestBuffer(9100L, src.remaining()));

    GpuRayTracingBlasResource result =
        capability.execute(RayTracingBlasWork.fromGeometryResource(geometryResource));

    assertEquals(9100L, result.bufferHandle().value());
    assertEquals(geometryResource, result.sourceGeometryResource());
    assertEquals(2, result.regionCount());
    assertEquals(5 * Integer.BYTES, result.regionsStrideBytes());
    assertEquals(2 * 5 * Integer.BYTES, result.regionsByteSize());
  }

  @Test
  void rejectsClosedGeometryResource() {
    GpuRayTracingGeometryResource geometryResource = createGeometryResource();
    geometryResource.close();
    VulkanRayTracingBlasCapability capability =
        new VulkanRayTracingBlasCapability(src -> new TestBuffer(9101L, src.remaining()));

    assertThrows(
        IllegalStateException.class,
        () -> capability.execute(RayTracingBlasWork.fromGeometryResource(geometryResource)));
  }

  @Test
  void rejectsUploadBufferSizeMismatch() {
    GpuRayTracingGeometryResource geometryResource = createGeometryResource();
    VulkanRayTracingBlasCapability capability =
        new VulkanRayTracingBlasCapability(src -> new TestBuffer(9102L, 8L));

    assertThrows(
        IllegalStateException.class,
        () -> capability.execute(RayTracingBlasWork.fromGeometryResource(geometryResource)));
  }

  private static GpuRayTracingGeometryResource createGeometryResource() {
    ByteBuffer bytes = ByteBuffer.allocate(2 * 5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0).putInt(0).putInt(36).putInt(0).putInt(0)
        .putInt(1).putInt(36).putInt(42).putInt(0).putInt(1)
        .flip();
    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(2, 0, 5, bytes);
    return new GpuRayTracingGeometryResource(new TestBuffer(9103L, payload.regionsByteSize()), payload);
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

