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
import org.dynamisengine.gpu.api.resource.GpuMeshletDrawMetadataPayload;
import org.junit.jupiter.api.Test;

class GpuDrivenClusterVisibilityValidationWorkTest {
  @Test
  void forAllMeshletsUsesBoundsCount() {
    GpuMeshletBoundsResource bounds = createBoundsResource(2);
    MeshletVisibilityFrustum frustum = MeshletVisibilityFrustum.of(new float[24]);
    GpuMeshletDrawMetadataPayload metadata =
        GpuMeshletDrawMetadataPayload.of(new int[] {1, 1}, new int[] {0, 1}, new int[] {0, 0});

    GpuDrivenClusterVisibilityValidationWork work =
        GpuDrivenClusterVisibilityValidationWork.forAllMeshlets(bounds, frustum, metadata);
    assertEquals(2, work.meshletCount());
  }

  @Test
  void rejectsInsufficientDrawMetadata() {
    GpuMeshletBoundsResource bounds = createBoundsResource(2);
    MeshletVisibilityFrustum frustum = MeshletVisibilityFrustum.of(new float[24]);
    GpuMeshletDrawMetadataPayload metadata =
        GpuMeshletDrawMetadataPayload.of(new int[] {1}, new int[] {0}, new int[] {0});

    assertThrows(
        IllegalArgumentException.class,
        () -> new GpuDrivenClusterVisibilityValidationWork(bounds, frustum, metadata, 2));
  }

  private static GpuMeshletBoundsResource createBoundsResource(int meshletCount) {
    ByteBuffer bytes =
        ByteBuffer.allocate(meshletCount * 6 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < meshletCount * 6; i++) {
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
      return new GpuBufferHandle(450L);
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

