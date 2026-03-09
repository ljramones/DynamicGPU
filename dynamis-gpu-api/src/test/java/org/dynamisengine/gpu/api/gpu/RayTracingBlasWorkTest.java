package org.dynamisengine.gpu.api.gpu;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBuildInputPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBuildInputResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryResource;
import org.junit.jupiter.api.Test;

class RayTracingBlasWorkTest {
  @Test
  void capturesBuildInputResource() {
    GpuRayTracingBuildInputResource buildInputResource = createBuildInputResource();

    RayTracingBlasWork work = RayTracingBlasWork.fromBuildInputResource(buildInputResource);

    assertSame(buildInputResource, work.buildInputResource());
  }

  @Test
  void rejectsNullBuildInputResource() {
    assertThrows(NullPointerException.class, () -> RayTracingBlasWork.fromBuildInputResource(null));
  }

  private static GpuRayTracingBuildInputResource createBuildInputResource() {
    ByteBuffer bytes = ByteBuffer.allocate(5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(36).putInt(0).putInt(0).flip();
    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(1, 0, 5, bytes);
    GpuRayTracingGeometryResource geometryResource =
        new GpuRayTracingGeometryResource(new TestBuffer(8000L, payload.regionsByteSize()), payload);
    GpuRayTracingBuildInputPayload buildInputPayload =
        GpuRayTracingBuildInputPayload.of(
            geometryResource, new GpuBufferHandle(8001L), new GpuBufferHandle(8002L), 32, 1024, 0L, 0L);
    return new GpuRayTracingBuildInputResource(buildInputPayload, 9001L, 9002L);
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
