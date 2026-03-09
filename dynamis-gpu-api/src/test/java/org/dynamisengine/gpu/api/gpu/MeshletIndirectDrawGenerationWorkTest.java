package org.dynamisengine.gpu.api.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.resource.GpuMeshletDrawMetadataPayload;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListPayload;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListResource;
import org.junit.jupiter.api.Test;

class MeshletIndirectDrawGenerationWorkTest {
  @Test
  void forAllVisibleUsesVisibleCount() {
    GpuVisibleMeshletListResource visible = createVisibleResource(8, 1, 3, 5);
    GpuMeshletDrawMetadataPayload metadata =
        GpuMeshletDrawMetadataPayload.of(new int[8], new int[8], new int[8]);

    MeshletIndirectDrawGenerationWork work =
        MeshletIndirectDrawGenerationWork.forAllVisibleMeshlets(visible, metadata);
    assertEquals(3, work.commandCount());
  }

  @Test
  void rejectsCommandCountGreaterThanVisibleCount() {
    GpuVisibleMeshletListResource visible = createVisibleResource(4, 0, 1);
    GpuMeshletDrawMetadataPayload metadata =
        GpuMeshletDrawMetadataPayload.of(new int[4], new int[4], new int[4]);
    assertThrows(
        IllegalArgumentException.class,
        () -> new MeshletIndirectDrawGenerationWork(visible, metadata, 3));
  }

  private static GpuVisibleMeshletListResource createVisibleResource(int sourceCount, int... indices) {
    ByteBuffer bytes = ByteBuffer.allocate(indices.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int index : indices) {
      bytes.putInt(index);
    }
    bytes.flip();
    GpuVisibleMeshletListPayload payload =
        GpuVisibleMeshletListPayload.fromLittleEndianBytes(sourceCount, indices.length, bytes);
    return new GpuVisibleMeshletListResource(new TestBuffer(payload.visibleIndicesByteSize()), payload);
  }

  private static final class TestBuffer implements GpuBuffer {
    private final long sizeBytes;

    private TestBuffer(long sizeBytes) {
      this.sizeBytes = sizeBytes;
    }

    @Override
    public GpuBufferHandle handle() {
      return new GpuBufferHandle(333L);
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

