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

class GpuRayTracingBuildInputPayloadTest {
  @Test
  void createsValidBuildInputPayload() {
    GpuRayTracingGeometryResource geometryResource = createGeometryResource();

    GpuRayTracingBuildInputPayload payload =
        GpuRayTracingBuildInputPayload.of(
            geometryResource, new GpuBufferHandle(7001L), new GpuBufferHandle(7002L), 32, 1024, 0L, 0L);

    assertEquals(1, payload.regionCount());
    assertEquals(32, payload.vertexStrideBytes());
    assertEquals(1024, payload.maxVertexIndex());
    assertEquals(7001L, payload.vertexBufferHandle().value());
    assertEquals(7002L, payload.indexBufferHandle().value());
  }

  @Test
  void rejectsInvalidBuildInputPayloadFields() {
    GpuRayTracingGeometryResource geometryResource = createGeometryResource();
    assertThrows(
        IllegalArgumentException.class,
        () ->
            GpuRayTracingBuildInputPayload.of(
                geometryResource, new GpuBufferHandle(1L), new GpuBufferHandle(2L), 0, 1, 0L, 0L));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            GpuRayTracingBuildInputPayload.of(
                geometryResource, new GpuBufferHandle(1L), new GpuBufferHandle(2L), 18, 1, 0L, 0L));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            GpuRayTracingBuildInputPayload.of(
                geometryResource, new GpuBufferHandle(1L), new GpuBufferHandle(2L), 16, -1, 0L, 0L));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            GpuRayTracingBuildInputPayload.of(
                geometryResource, new GpuBufferHandle(1L), new GpuBufferHandle(2L), 16, 1, -1L, 0L));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            GpuRayTracingBuildInputPayload.of(
                geometryResource, new GpuBufferHandle(1L), new GpuBufferHandle(2L), 16, 1, 0L, -1L));
  }

  private static GpuRayTracingGeometryResource createGeometryResource() {
    ByteBuffer bytes = ByteBuffer.allocate(5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(36).putInt(0).putInt(0).flip();
    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(1, 0, 5, bytes);
    return new GpuRayTracingGeometryResource(new TestBuffer(7000L, payload.regionsByteSize()), payload);
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

