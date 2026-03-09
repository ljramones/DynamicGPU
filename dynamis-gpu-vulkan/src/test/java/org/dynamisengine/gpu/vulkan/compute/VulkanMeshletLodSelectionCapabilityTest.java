package org.dynamisengine.gpu.vulkan.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.gpu.MeshletLodSelectionWork;
import org.dynamisengine.gpu.api.resource.GpuMeshletLodPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletLodResource;
import org.dynamisengine.gpu.api.resource.GpuSelectedMeshletLodResource;
import org.junit.jupiter.api.Test;

class VulkanMeshletLodSelectionCapabilityTest {
  @Test
  void selectsLevelZeroWhenRequested() throws Exception {
    GpuMeshletLodResource lodResource = createLodResource();
    VulkanMeshletLodSelectionCapability capability =
        new VulkanMeshletLodSelectionCapability(src -> new TestBuffer(3000L, src.remaining()));

    GpuSelectedMeshletLodResource selected =
        capability.execute(MeshletLodSelectionWork.forTargetLevel(lodResource, 0));

    assertEquals(0, selected.selectedLodLevel());
    assertEquals(0, selected.meshletStart());
    assertEquals(64, selected.meshletCount());
    assertEquals(3000L, selected.bufferHandle().value());
  }

  @Test
  void selectsHigherLevelWhenRequested() throws Exception {
    GpuMeshletLodResource lodResource = createLodResource();
    VulkanMeshletLodSelectionCapability capability =
        new VulkanMeshletLodSelectionCapability(src -> new TestBuffer(3001L, src.remaining()));

    GpuSelectedMeshletLodResource selected =
        capability.execute(MeshletLodSelectionWork.forTargetLevel(lodResource, 2));

    assertEquals(2, selected.selectedLodLevel());
    assertEquals(96, selected.meshletStart());
    assertEquals(16, selected.meshletCount());
  }

  @Test
  void rejectsTargetLevelNotPresent() throws Exception {
    GpuMeshletLodResource lodResource = createLodResource();
    VulkanMeshletLodSelectionCapability capability =
        new VulkanMeshletLodSelectionCapability(src -> new TestBuffer(3002L, src.remaining()));

    assertThrows(
        IllegalArgumentException.class,
        () -> capability.execute(MeshletLodSelectionWork.forTargetLevel(lodResource, 5)));
  }

  @Test
  void rejectsUploadBufferSizeMismatch() throws Exception {
    GpuMeshletLodResource lodResource = createLodResource();
    VulkanMeshletLodSelectionCapability capability =
        new VulkanMeshletLodSelectionCapability(src -> new TestBuffer(3003L, 8L));

    assertThrows(
        IllegalStateException.class,
        () -> capability.execute(MeshletLodSelectionWork.forTargetLevel(lodResource, 1)));
  }

  private static GpuMeshletLodResource createLodResource() {
    ByteBuffer bytes = ByteBuffer.allocate(3 * 4 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0)
        .putInt(0)
        .putInt(64)
        .putInt(Float.floatToRawIntBits(0.0f))
        .putInt(1)
        .putInt(64)
        .putInt(32)
        .putInt(Float.floatToRawIntBits(0.6f))
        .putInt(2)
        .putInt(96)
        .putInt(16)
        .putInt(Float.floatToRawIntBits(1.2f))
        .flip();
    GpuMeshletLodPayload payload = GpuMeshletLodPayload.fromLittleEndianBytes(3, 0, 4, bytes);
    return new GpuMeshletLodResource(new TestBuffer(2222L, payload.levelsByteSize()), payload);
  }

  private static final class TestBuffer implements GpuBuffer {
    private final GpuBufferHandle handle;
    private final long sizeBytes;
    private final AtomicInteger closeCount = new AtomicInteger();

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
    public void close() {
      closeCount.incrementAndGet();
    }
  }
}
