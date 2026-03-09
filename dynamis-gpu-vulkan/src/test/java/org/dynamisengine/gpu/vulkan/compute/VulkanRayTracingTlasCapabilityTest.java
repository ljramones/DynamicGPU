package org.dynamisengine.gpu.vulkan.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.gpu.RayTracingTlasWork;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingTlasInstanceMetadata;
import org.dynamisengine.gpu.api.resource.GpuRayTracingTlasPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingTlasResource;
import org.junit.jupiter.api.Test;

class VulkanRayTracingTlasCapabilityTest {
  @Test
  void executesAndReturnsTlasResource() throws Exception {
    RayTracingTlasWork work = createWork();
    VulkanRayTracingTlasCapability capability =
        new VulkanRayTracingTlasCapability(
            tlasWork ->
                new GpuRayTracingTlasResource(
                    new TestBuffer(2000L, 256L),
                    GpuRayTracingTlasPayload.of(tlasWork.instances().size(), 64),
                    9000L,
                    null));

    GpuRayTracingTlasResource result = capability.execute(work);

    assertEquals(2000L, result.bufferHandle().value());
    assertEquals(1, result.instanceCount());
    assertEquals(64, result.instanceByteSize());
    assertEquals(9000L, result.accelerationStructureHandle());
  }

  @Test
  void propagatesBackendFailure() {
    RayTracingTlasWork work = createWork();
    VulkanRayTracingTlasCapability capability =
        new VulkanRayTracingTlasCapability(
            tlasWork -> {
              throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "expected", false);
            });

    assertThrows(GpuException.class, () -> capability.execute(work));
  }

  private static RayTracingTlasWork createWork() {
    ByteBuffer bytes = ByteBuffer.allocate(5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(36).putInt(0).putInt(0).flip();
    GpuRayTracingGeometryPayload geometryPayload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(1, 0, 5, bytes);
    GpuRayTracingGeometryResource geometryResource =
        new GpuRayTracingGeometryResource(new TestBuffer(2100L, geometryPayload.regionsByteSize()), geometryPayload);
    GpuRayTracingBlasPayload blasPayload = GpuRayTracingBlasPayload.forGeometryResource(geometryResource);
    GpuRayTracingBlasResource blasResource =
        new GpuRayTracingBlasResource(
            new TestBuffer(2101L, blasPayload.byteSize()), blasPayload, geometryResource, 9100L, null);
    GpuRayTracingTlasInstanceMetadata instance =
        GpuRayTracingTlasInstanceMetadata.of(
            blasResource, new float[] {1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f}, 0, 0xFF, 0, 0);
    return RayTracingTlasWork.fromInstances(List.of(instance));
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

