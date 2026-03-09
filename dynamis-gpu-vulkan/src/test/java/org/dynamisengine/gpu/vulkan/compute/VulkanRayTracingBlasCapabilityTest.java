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
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.gpu.RayTracingBlasWork;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBuildInputPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBuildInputResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryResource;
import org.junit.jupiter.api.Test;

class VulkanRayTracingBlasCapabilityTest {
  @Test
  void executesBlasAndReturnsBuiltResourceFromBuildInput() throws Exception {
    GpuRayTracingBuildInputResource buildInputResource = createBuildInputResource();
    GpuRayTracingBlasPayload payload =
        GpuRayTracingBlasPayload.forGeometryResource(buildInputResource.payload().geometryResource());
    VulkanRayTracingBlasCapability capability =
        new VulkanRayTracingBlasCapability(
            work ->
                new GpuRayTracingBlasResource(
                    new TestBuffer(9100L, payload.byteSize()),
                    payload,
                    buildInputResource.payload().geometryResource(),
                    4444L,
                    null));

    GpuRayTracingBlasResource result =
        capability.execute(RayTracingBlasWork.fromBuildInputResource(buildInputResource));

    assertEquals(9100L, result.bufferHandle().value());
    assertEquals(buildInputResource.payload().geometryResource(), result.sourceGeometryResource());
    assertEquals(2, result.regionCount());
    assertEquals(5 * Integer.BYTES, result.regionsStrideBytes());
    assertEquals(2 * 5 * Integer.BYTES, result.regionsByteSize());
    assertEquals(4444L, result.accelerationStructureHandle());
  }

  @Test
  void rejectsClosedBuildInputResource() {
    GpuRayTracingBuildInputResource buildInputResource = createBuildInputResource();
    buildInputResource.close();
    VulkanRayTracingBlasCapability capability =
        new VulkanRayTracingBlasCapability(
            work -> {
              if (work.buildInputResource().isClosed()) {
                throw new IllegalStateException("buildInputResource is already closed");
              }
              throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "unexpected path", false);
            });

    assertThrows(
        IllegalStateException.class,
        () -> capability.execute(RayTracingBlasWork.fromBuildInputResource(buildInputResource)));
  }

  @Test
  void propagatesExecutorFailures() {
    GpuRayTracingBuildInputResource buildInputResource = createBuildInputResource();
    VulkanRayTracingBlasCapability capability =
        new VulkanRayTracingBlasCapability(
            work -> {
              throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "expected", false);
            });

    assertThrows(GpuException.class, () -> capability.execute(RayTracingBlasWork.fromBuildInputResource(buildInputResource)));
  }

  private static GpuRayTracingBuildInputResource createBuildInputResource() {
    ByteBuffer bytes = ByteBuffer.allocate(2 * 5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0).putInt(0).putInt(36).putInt(0).putInt(0)
        .putInt(1).putInt(36).putInt(42).putInt(0).putInt(1)
        .flip();
    GpuRayTracingGeometryPayload geometryPayload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(2, 0, 5, bytes);
    GpuRayTracingGeometryResource geometryResource =
        new GpuRayTracingGeometryResource(new TestBuffer(9103L, geometryPayload.regionsByteSize()), geometryPayload);
    GpuRayTracingBuildInputPayload buildInputPayload =
        GpuRayTracingBuildInputPayload.of(
            geometryResource, new GpuBufferHandle(9104L), new GpuBufferHandle(9105L), 32, 2048, 0L, 0L);
    return new GpuRayTracingBuildInputResource(buildInputPayload, 10001L, 10002L);
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
