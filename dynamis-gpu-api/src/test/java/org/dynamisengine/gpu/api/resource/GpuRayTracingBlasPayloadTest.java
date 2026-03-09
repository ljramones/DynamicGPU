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

class GpuRayTracingBlasPayloadTest {
  @Test
  void createsBuildPrepPayloadFromGeometryResource() {
    GpuRayTracingGeometryResource geometryResource = createGeometryResource();

    GpuRayTracingBlasPayload payload =
        GpuRayTracingBlasPayload.forGeometryResource(geometryResource);

    assertEquals(2, payload.regionCount());
    assertEquals(5 * Integer.BYTES, payload.regionsStrideBytes());
    assertEquals(2 * 5 * Integer.BYTES, payload.regionsByteSize());
    assertEquals(0, payload.reservedFlags());
    assertEquals(GpuRayTracingBlasPayload.BYTE_SIZE, payload.byteSize());
  }

  @Test
  void rejectsInvalidPayloadFields() {
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuRayTracingBlasPayload.of(0, 20, 20, 0));
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuRayTracingBlasPayload.of(1, 0, 20, 0));
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuRayTracingBlasPayload.of(1, 20, 0, 0));
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuRayTracingBlasPayload.of(1, 18, 20, 0));
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuRayTracingBlasPayload.of(1, 20, 18, 0));
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuRayTracingBlasPayload.of(1, 20, 20, -1));
  }

  private static GpuRayTracingGeometryResource createGeometryResource() {
    ByteBuffer bytes = ByteBuffer.allocate(2 * 5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0).putInt(0).putInt(36).putInt(0).putInt(0)
        .putInt(1).putInt(36).putInt(42).putInt(0).putInt(1)
        .flip();
    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(2, 0, 5, bytes);
    return new GpuRayTracingGeometryResource(new TestBuffer(8100L, payload.regionsByteSize()), payload);
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

