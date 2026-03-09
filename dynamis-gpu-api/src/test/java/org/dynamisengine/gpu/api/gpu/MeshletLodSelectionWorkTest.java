package org.dynamisengine.gpu.api.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.resource.GpuMeshletLodPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletLodResource;
import org.junit.jupiter.api.Test;

class MeshletLodSelectionWorkTest {
  @Test
  void forTargetLevelUsesProvidedTarget() {
    GpuMeshletLodResource resource = createLodResource();

    MeshletLodSelectionWork work = MeshletLodSelectionWork.forTargetLevel(resource, 1);
    assertEquals(1, work.targetLodLevel());
    assertEquals(resource, work.lodResource());
  }

  @Test
  void rejectsNegativeTarget() {
    GpuMeshletLodResource resource = createLodResource();
    assertThrows(IllegalArgumentException.class, () -> new MeshletLodSelectionWork(resource, -1));
  }

  private static GpuMeshletLodResource createLodResource() {
    ByteBuffer bytes = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(64).putInt(Float.floatToRawIntBits(0.0f)).flip();
    GpuMeshletLodPayload payload = GpuMeshletLodPayload.fromLittleEndianBytes(1, 0, 4, bytes);
    return new GpuMeshletLodResource(new TestBuffer(payload.levelsByteSize()), payload);
  }

  private static final class TestBuffer implements GpuBuffer {
    private final long sizeBytes;

    private TestBuffer(long sizeBytes) {
      this.sizeBytes = sizeBytes;
    }

    @Override
    public GpuBufferHandle handle() {
      return new GpuBufferHandle(888L);
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
