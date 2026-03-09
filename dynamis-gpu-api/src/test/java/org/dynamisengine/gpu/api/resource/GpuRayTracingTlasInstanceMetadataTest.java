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

class GpuRayTracingTlasInstanceMetadataTest {
  @Test
  void createsValidInstanceMetadata() {
    GpuRayTracingBlasResource blas = createBlasResource();
    float[] transform = identity3x4();

    GpuRayTracingTlasInstanceMetadata instance =
        GpuRayTracingTlasInstanceMetadata.of(blas, transform, 12, 0xFF, 0, 0);

    assertEquals(12, instance.instanceCustomIndex());
    assertEquals(0xFF, instance.mask());
    assertEquals(0, instance.shaderBindingTableRecordOffset());
    assertEquals(0, instance.flags());
    assertEquals(12, instance.transform3x4RowMajor().length);
  }

  @Test
  void rejectsInvalidFields() {
    GpuRayTracingBlasResource blas = createBlasResource();
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuRayTracingTlasInstanceMetadata.of(blas, new float[11], 0, 1, 0, 0));
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuRayTracingTlasInstanceMetadata.of(blas, identity3x4(), -1, 1, 0, 0));
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuRayTracingTlasInstanceMetadata.of(blas, identity3x4(), 0, 256, 0, 0));
  }

  private static GpuRayTracingBlasResource createBlasResource() {
    ByteBuffer bytes = ByteBuffer.allocate(5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(36).putInt(0).putInt(0).flip();
    GpuRayTracingGeometryPayload geometryPayload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(1, 0, 5, bytes);
    GpuRayTracingGeometryResource geometryResource =
        new GpuRayTracingGeometryResource(new TestBuffer(100L, geometryPayload.regionsByteSize()), geometryPayload);
    GpuRayTracingBlasPayload blasPayload = GpuRayTracingBlasPayload.forGeometryResource(geometryResource);
    return new GpuRayTracingBlasResource(
        new TestBuffer(101L, blasPayload.byteSize()), blasPayload, geometryResource, 9999L, null);
  }

  private static float[] identity3x4() {
    return new float[] {1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f};
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

