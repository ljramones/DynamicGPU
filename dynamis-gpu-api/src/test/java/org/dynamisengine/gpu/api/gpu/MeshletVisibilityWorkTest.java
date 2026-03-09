package org.dynamisengine.gpu.api.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsResource;
import org.junit.jupiter.api.Test;

class MeshletVisibilityWorkTest {
  @Test
  void forAllMeshletsUsesResourceCount() {
    GpuMeshletBoundsResource resource = createBoundsResource(2);
    MeshletVisibilityFrustum frustum = MeshletVisibilityFrustum.of(new float[24]);

    MeshletVisibilityWork work = MeshletVisibilityWork.forAllMeshlets(resource, frustum);
    assertEquals(2, work.meshletCount());
  }

  @Test
  void rejectsRequestedCountLargerThanResource() {
    GpuMeshletBoundsResource resource = createBoundsResource(1);
    MeshletVisibilityFrustum frustum = MeshletVisibilityFrustum.of(new float[24]);

    assertThrows(IllegalArgumentException.class, () -> new MeshletVisibilityWork(resource, 2, frustum));
  }

  private static GpuMeshletBoundsResource createBoundsResource(int meshletCount) {
    int floatCount = meshletCount * 6;
    ByteBuffer bytes = ByteBuffer.allocate(floatCount * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < floatCount; i++) {
      bytes.putFloat(0f);
    }
    bytes.flip();
    GpuMeshletBoundsPayload payload =
        GpuMeshletBoundsPayload.fromLittleEndianBytes(meshletCount, 0, 6, bytes);
    return new GpuMeshletBoundsResource(new TestBuffer(payload.boundsByteSize()), payload);
  }

  private static final class TestBuffer implements GpuBuffer {
    private final long sizeBytes;

    private TestBuffer(long sizeBytes) {
      this.sizeBytes = sizeBytes;
    }

    @Override
    public GpuBufferHandle handle() {
      return new GpuBufferHandle(10L);
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

