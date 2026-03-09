package org.dynamisengine.gpu.api.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingTlasInstanceMetadata;
import org.junit.jupiter.api.Test;

class RayTracingTlasWorkTest {
  @Test
  void capturesInstances() {
    GpuRayTracingTlasInstanceMetadata instance = createInstance();

    RayTracingTlasWork work = RayTracingTlasWork.fromInstances(List.of(instance));

    assertEquals(1, work.instances().size());
    assertEquals(instance, work.instances().get(0));
  }

  @Test
  void rejectsEmptyInstances() {
    assertThrows(IllegalArgumentException.class, () -> RayTracingTlasWork.fromInstances(List.of()));
  }

  private static GpuRayTracingTlasInstanceMetadata createInstance() {
    ByteBuffer bytes = ByteBuffer.allocate(5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(36).putInt(0).putInt(0).flip();
    GpuRayTracingGeometryPayload geometryPayload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(1, 0, 5, bytes);
    GpuRayTracingGeometryResource geometryResource =
        new GpuRayTracingGeometryResource(new TestBuffer(120L, geometryPayload.regionsByteSize()), geometryPayload);
    GpuRayTracingBlasPayload blasPayload = GpuRayTracingBlasPayload.forGeometryResource(geometryResource);
    GpuRayTracingBlasResource blasResource =
        new GpuRayTracingBlasResource(
            new TestBuffer(121L, blasPayload.byteSize()), blasPayload, geometryResource, 5000L, null);
    return GpuRayTracingTlasInstanceMetadata.of(
        blasResource, new float[] {1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f}, 0, 0xFF, 0, 0);
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

