package org.dynamisengine.gpu.vulkan.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.gpu.MeshletIndirectDrawGenerationWork;
import org.dynamisengine.gpu.api.resource.GpuMeshletDrawMetadataPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletIndirectDrawResource;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListPayload;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListResource;
import org.junit.jupiter.api.Test;

class VulkanMeshletIndirectDrawGenerationCapabilityTest {
  @Test
  void generatesCommandsForMixedVisibleListInStableOrder() throws Exception {
    GpuVisibleMeshletListResource visible = createVisibleResource(6, 1, 3, 4);
    GpuMeshletDrawMetadataPayload metadata =
        GpuMeshletDrawMetadataPayload.of(
            new int[] {9, 12, 15, 18, 21, 24},
            new int[] {0, 9, 21, 36, 54, 75},
            new int[] {0, 2, 4, 6, 8, 10});
    VulkanMeshletIndirectDrawGenerationCapability capability =
        new VulkanMeshletIndirectDrawGenerationCapability(src -> new TestBuffer(700L, src.remaining()));

    GpuMeshletIndirectDrawResource out =
        capability.execute(MeshletIndirectDrawGenerationWork.forAllVisibleMeshlets(visible, metadata));

    assertEquals(3, out.commandCount());
    assertEquals(3 * 20, out.commandByteSize());
    ByteBuffer cmd = out.payload().commandBytes().order(ByteOrder.LITTLE_ENDIAN);

    assertEquals(12, cmd.getInt(0));
    assertEquals(1, cmd.getInt(4));
    assertEquals(9, cmd.getInt(8));
    assertEquals(2, cmd.getInt(12));
    assertEquals(0, cmd.getInt(16));

    assertEquals(18, cmd.getInt(20));
    assertEquals(36, cmd.getInt(28));
    assertEquals(6, cmd.getInt(32));

    assertEquals(21, cmd.getInt(40));
    assertEquals(54, cmd.getInt(48));
    assertEquals(8, cmd.getInt(52));
  }

  @Test
  void supportsEmptyVisibleList() throws Exception {
    GpuVisibleMeshletListResource visible = createVisibleResource(4);
    GpuMeshletDrawMetadataPayload metadata =
        GpuMeshletDrawMetadataPayload.of(new int[] {1, 2, 3, 4}, new int[] {0, 1, 3, 6}, new int[] {0, 0, 0, 0});
    VulkanMeshletIndirectDrawGenerationCapability capability =
        new VulkanMeshletIndirectDrawGenerationCapability(src -> new TestBuffer(701L, src.remaining()));

    GpuMeshletIndirectDrawResource out =
        capability.execute(MeshletIndirectDrawGenerationWork.forAllVisibleMeshlets(visible, metadata));

    assertEquals(0, out.commandCount());
    assertEquals(0, out.commandByteSize());
  }

  @Test
  void rejectsExecutionAfterClose() throws Exception {
    GpuVisibleMeshletListResource visible = createVisibleResource(1, 0);
    GpuMeshletDrawMetadataPayload metadata =
        GpuMeshletDrawMetadataPayload.of(new int[] {3}, new int[] {0}, new int[] {0});
    VulkanMeshletIndirectDrawGenerationCapability capability =
        new VulkanMeshletIndirectDrawGenerationCapability(src -> new TestBuffer(702L, src.remaining()));
    capability.close();

    assertThrows(
        IllegalStateException.class,
        () -> capability.execute(MeshletIndirectDrawGenerationWork.forAllVisibleMeshlets(visible, metadata)));
  }

  private static GpuVisibleMeshletListResource createVisibleResource(int sourceMeshletCount, int... visibleIndices) {
    ByteBuffer bytes =
        ByteBuffer.allocate(visibleIndices.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int visibleIndex : visibleIndices) {
      bytes.putInt(visibleIndex);
    }
    bytes.flip();
    GpuVisibleMeshletListPayload payload =
        GpuVisibleMeshletListPayload.fromLittleEndianBytes(
            sourceMeshletCount, visibleIndices.length, bytes);
    return new GpuVisibleMeshletListResource(new TestBuffer(800L, payload.visibleIndicesByteSize()), payload);
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
      return GpuBufferUsage.INDIRECT;
    }

    @Override
    public GpuMemoryLocation memoryLocation() {
      return GpuMemoryLocation.DEVICE_LOCAL;
    }

    @Override
    public void close() {}
  }
}

