package org.dynamisengine.gpu.api.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsResource;
import org.junit.jupiter.api.Test;

class MeshletVisibilityCompactionWorkTest {
  @Test
  void forAllMeshletsUsesFlagsCount() {
    GpuMeshletVisibilityFlagsResource flags = createFlagsResource(new byte[] {0, 1, 0, 1});
    MeshletVisibilityCompactionWork work = MeshletVisibilityCompactionWork.forAllMeshlets(flags);
    assertEquals(4, work.meshletCount());
  }

  @Test
  void rejectsRequestedCountLargerThanFlagsResource() {
    GpuMeshletVisibilityFlagsResource flags = createFlagsResource(new byte[] {1});
    assertThrows(IllegalArgumentException.class, () -> new MeshletVisibilityCompactionWork(flags, 2));
  }

  private static GpuMeshletVisibilityFlagsResource createFlagsResource(byte[] values) {
    ByteBuffer bytes = ByteBuffer.wrap(values);
    GpuMeshletVisibilityFlagsPayload payload = GpuMeshletVisibilityFlagsPayload.fromBytes(values.length, bytes);
    return new GpuMeshletVisibilityFlagsResource(new TestBuffer(payload.flagsByteSize()), payload);
  }

  private static final class TestBuffer implements GpuBuffer {
    private final long sizeBytes;

    private TestBuffer(long sizeBytes) {
      this.sizeBytes = sizeBytes;
    }

    @Override
    public GpuBufferHandle handle() {
      return new GpuBufferHandle(200L);
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

