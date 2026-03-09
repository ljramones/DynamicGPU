package org.dynamisengine.gpu.vulkan.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.gpu.MeshletVisibilityFrustum;
import org.dynamisengine.gpu.api.gpu.MeshletVisibilityWork;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsResource;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsResource;
import org.junit.jupiter.api.Test;

class VulkanMeshletVisibilityCapabilityTest {
  private static final MeshletVisibilityFrustum ALWAYS_VISIBLE_FRUSTUM =
      MeshletVisibilityFrustum.of(
          new float[] {
            0f, 0f, 0f, 1f,
            0f, 0f, 0f, 1f,
            0f, 0f, 0f, 1f,
            0f, 0f, 0f, 1f,
            0f, 0f, 0f, 1f,
            0f, 0f, 0f, 1f
          });

  private static final MeshletVisibilityFrustum ALWAYS_CULLED_FRUSTUM =
      MeshletVisibilityFrustum.of(
          new float[] {
            1f, 0f, 0f, -10f,
            0f, 0f, 0f, 1f,
            0f, 0f, 0f, 1f,
            0f, 0f, 0f, 1f,
            0f, 0f, 0f, 1f,
            0f, 0f, 0f, 1f
          });

  @Test
  void computesMixedVisibilityFlags() throws Exception {
    GpuMeshletBoundsResource bounds =
        createBoundsResource(
            new float[][] {
              {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f},
              {2f, 2f, 2f, 3f, 3f, 3f}
            });
    VulkanMeshletVisibilityCapability capability =
        new VulkanMeshletVisibilityCapability(src -> new TestBuffer(100L, src.remaining()));

    GpuMeshletVisibilityFlagsResource visibleOutput =
        capability.execute(MeshletVisibilityWork.forAllMeshlets(bounds, ALWAYS_VISIBLE_FRUSTUM));
    GpuMeshletVisibilityFlagsResource culledOutput =
        capability.execute(MeshletVisibilityWork.forAllMeshlets(bounds, ALWAYS_CULLED_FRUSTUM));

    assertEquals(2, visibleOutput.meshletCount());
    assertEquals(2, visibleOutput.flagsByteSize());
    assertEquals((byte) 1, visibleOutput.payload().flagsBytes().get(0));
    assertEquals((byte) 1, visibleOutput.payload().flagsBytes().get(1));
    assertEquals((byte) 0, culledOutput.payload().flagsBytes().get(0));
    assertEquals((byte) 0, culledOutput.payload().flagsBytes().get(1));
  }

  @Test
  void handlesAllVisibleAndAllCulledCases() throws Exception {
    VulkanMeshletVisibilityCapability capability =
        new VulkanMeshletVisibilityCapability(src -> new TestBuffer(101L, src.remaining()));
    GpuMeshletBoundsResource allVisible =
        createBoundsResource(
            new float[][] {
              {-0.2f, -0.2f, -0.2f, 0.2f, 0.2f, 0.2f}, {-0.8f, -0.8f, -0.8f, 0.8f, 0.8f, 0.8f}
            });
    GpuMeshletBoundsResource allCulled =
        createBoundsResource(new float[][] {{2f, 0f, 0f, 3f, 1f, 1f}, {3f, -3f, -3f, 4f, -2f, -2f}});

    GpuMeshletVisibilityFlagsResource visibleOut =
        capability.execute(MeshletVisibilityWork.forAllMeshlets(allVisible, ALWAYS_VISIBLE_FRUSTUM));
    GpuMeshletVisibilityFlagsResource culledOut =
        capability.execute(MeshletVisibilityWork.forAllMeshlets(allCulled, ALWAYS_CULLED_FRUSTUM));

    assertEquals((byte) 1, visibleOut.payload().flagsBytes().get(0));
    assertEquals((byte) 1, visibleOut.payload().flagsBytes().get(1));
    assertEquals((byte) 0, culledOut.payload().flagsBytes().get(0));
    assertEquals((byte) 0, culledOut.payload().flagsBytes().get(1));
  }

  @Test
  void rejectsExecutionAfterClose() throws Exception {
    GpuMeshletBoundsResource bounds =
        createBoundsResource(new float[][] {{-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f}});
    VulkanMeshletVisibilityCapability capability =
        new VulkanMeshletVisibilityCapability(src -> new TestBuffer(102L, src.remaining()));
    capability.close();
    assertThrows(
        IllegalStateException.class,
        () ->
            capability.execute(
                MeshletVisibilityWork.forAllMeshlets(bounds, ALWAYS_CULLED_FRUSTUM)));
  }

  private static GpuMeshletBoundsResource createBoundsResource(float[][] meshletBounds) {
    ByteBuffer bytes =
        ByteBuffer.allocate(meshletBounds.length * 6 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float[] bounds : meshletBounds) {
      if (bounds.length != 6) {
        throw new IllegalArgumentException("bounds must have 6 floats");
      }
      for (float value : bounds) {
        bytes.putFloat(value);
      }
    }
    bytes.flip();

    GpuMeshletBoundsPayload payload =
        GpuMeshletBoundsPayload.fromLittleEndianBytes(meshletBounds.length, 0, 6, bytes);
    return new GpuMeshletBoundsResource(new TestBuffer(900L, payload.boundsByteSize()), payload);
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
