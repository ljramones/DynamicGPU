package org.dynamisengine.gpu.vulkan.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.gpu.MeshletVisibilityCompactionWork;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsResource;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListResource;
import org.junit.jupiter.api.Test;

class VulkanMeshletVisibilityCompactionCapabilityTest {
  @Test
  void compactsMixedFlagsInStableOrder() throws Exception {
    GpuMeshletVisibilityFlagsResource flags = createFlagsResource(new byte[] {0, 1, 0, 1, 1});
    VulkanMeshletVisibilityCompactionCapability capability =
        new VulkanMeshletVisibilityCompactionCapability(src -> new TestBuffer(500L, src.remaining()));

    GpuVisibleMeshletListResource out =
        capability.execute(MeshletVisibilityCompactionWork.forAllMeshlets(flags));

    assertEquals(5, out.sourceMeshletCount());
    assertEquals(3, out.visibleMeshletCount());
    assertEquals(3 * Integer.BYTES, out.visibleIndicesByteSize());
    ByteBuffer indices = out.payload().visibleIndicesBytes().order(ByteOrder.LITTLE_ENDIAN);
    assertEquals(1, indices.getInt(0));
    assertEquals(3, indices.getInt(4));
    assertEquals(4, indices.getInt(8));
  }

  @Test
  void supportsAllCulledAndAllVisibleCases() throws Exception {
    VulkanMeshletVisibilityCompactionCapability capability =
        new VulkanMeshletVisibilityCompactionCapability(src -> new TestBuffer(501L, src.remaining()));

    GpuVisibleMeshletListResource culledOut =
        capability.execute(MeshletVisibilityCompactionWork.forAllMeshlets(createFlagsResource(new byte[] {0, 0, 0})));
    assertEquals(0, culledOut.visibleMeshletCount());
    assertEquals(0, culledOut.visibleIndicesByteSize());

    GpuVisibleMeshletListResource visibleOut =
        capability.execute(MeshletVisibilityCompactionWork.forAllMeshlets(createFlagsResource(new byte[] {1, 1, 1})));
    assertEquals(3, visibleOut.visibleMeshletCount());
    ByteBuffer indices = visibleOut.payload().visibleIndicesBytes().order(ByteOrder.LITTLE_ENDIAN);
    assertEquals(0, indices.getInt(0));
    assertEquals(1, indices.getInt(4));
    assertEquals(2, indices.getInt(8));
  }

  @Test
  void rejectsExecutionAfterClose() throws Exception {
    VulkanMeshletVisibilityCompactionCapability capability =
        new VulkanMeshletVisibilityCompactionCapability(src -> new TestBuffer(502L, src.remaining()));
    capability.close();
    assertThrows(
        IllegalStateException.class,
        () -> capability.execute(MeshletVisibilityCompactionWork.forAllMeshlets(createFlagsResource(new byte[] {1}))));
  }

  private static GpuMeshletVisibilityFlagsResource createFlagsResource(byte[] flags) {
    GpuMeshletVisibilityFlagsPayload payload =
        GpuMeshletVisibilityFlagsPayload.fromBytes(flags.length, ByteBuffer.wrap(flags));
    return new GpuMeshletVisibilityFlagsResource(new TestBuffer(900L, payload.flagsByteSize()), payload);
  }

  private static final class TestBuffer implements GpuBuffer {
    private final GpuBufferHandle handle;
    private final long sizeBytes;

    private TestBuffer(long handleValue, long sizeBytes) {
      this.handle = new GpuBufferHandle(handleValue);
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

