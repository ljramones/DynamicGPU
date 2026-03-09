package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.junit.jupiter.api.Test;

class RayTracingBuildInputIngestionTest {
  @Test
  void ingestsAndResolvesBuildInputPayload() {
    GpuRayTracingGeometryResource geometryResource = createGeometryResource(0, 36);

    GpuRayTracingBuildInputPayload payload =
        RayTracingBuildInputIngestion.ingest(
            geometryResource, new GpuBufferHandle(7201L), new GpuBufferHandle(7202L), 32, 2048, 64L, 32L);
    GpuRayTracingBuildInputResource resolved =
        RayTracingBuildInputIngestion.toResolvedResource(payload, 8001L, 8002L);

    assertEquals(1, payload.regionCount());
    assertEquals(32, payload.vertexStrideBytes());
    assertEquals(2048, payload.maxVertexIndex());
    assertEquals(8001L, resolved.vertexBufferDeviceAddress());
    assertEquals(8002L, resolved.indexBufferDeviceAddress());
  }

  @Test
  void rejectsInvalidRegionMetadataForBuildInput() {
    GpuRayTracingGeometryResource badFirstIndex = createGeometryResource(-1, 36);
    assertThrows(
        IllegalArgumentException.class,
        () ->
            RayTracingBuildInputIngestion.ingest(
                badFirstIndex, new GpuBufferHandle(1L), new GpuBufferHandle(2L), 16, 10, 0L, 0L));

    GpuRayTracingGeometryResource badIndexCount = createGeometryResource(0, 0);
    assertThrows(
        IllegalArgumentException.class,
        () ->
            RayTracingBuildInputIngestion.ingest(
                badIndexCount, new GpuBufferHandle(1L), new GpuBufferHandle(2L), 16, 10, 0L, 0L));
  }

  private static GpuRayTracingGeometryResource createGeometryResource(int firstIndex, int indexCount) {
    ByteBuffer bytes = ByteBuffer.allocate(5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(firstIndex).putInt(indexCount).putInt(0).putInt(0).flip();
    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(1, 0, 5, bytes);
    return new GpuRayTracingGeometryResource(new TestBuffer(7200L, payload.regionsByteSize()), payload);
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

