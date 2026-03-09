package org.dynamisengine.gpu.api.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.resource.GpuMeshletStreamingPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletStreamingResource;
import org.junit.jupiter.api.Test;

class MeshletStreamingResidencyWorkTest {
  @Test
  void forTargetStreamUnitUsesProvidedTarget() {
    GpuMeshletStreamingResource resource = createStreamingResource();

    MeshletStreamingResidencyWork work = MeshletStreamingResidencyWork.forTargetStreamUnit(resource, 7);
    assertEquals(7, work.targetStreamUnitId());
    assertEquals(resource, work.streamingResource());
  }

  @Test
  void rejectsNegativeTarget() {
    GpuMeshletStreamingResource resource = createStreamingResource();
    assertThrows(IllegalArgumentException.class, () -> new MeshletStreamingResidencyWork(resource, -1));
  }

  private static GpuMeshletStreamingResource createStreamingResource() {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(7).putInt(0).putInt(32).putInt(0).putInt(4096).flip();
    GpuMeshletStreamingPayload payload =
        GpuMeshletStreamingPayload.fromLittleEndianBytes(1, 0, 5, bytes);
    return new GpuMeshletStreamingResource(new TestBuffer(payload.streamUnitsByteSize()), payload);
  }

  private static final class TestBuffer implements GpuBuffer {
    private final long sizeBytes;

    private TestBuffer(long sizeBytes) {
      this.sizeBytes = sizeBytes;
    }

    @Override
    public GpuBufferHandle handle() {
      return new GpuBufferHandle(1888L);
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

