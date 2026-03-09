package org.dynamisengine.gpu.vulkan.compute;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.gpu.GpuDrivenClusterVisibilityValidationHarness;
import org.dynamisengine.gpu.api.gpu.GpuDrivenClusterVisibilityValidationResult;
import org.dynamisengine.gpu.api.gpu.GpuDrivenClusterVisibilityValidationWork;
import org.dynamisengine.gpu.api.gpu.MeshletVisibilityFrustum;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsResource;
import org.dynamisengine.gpu.api.resource.GpuMeshletDrawMetadataPayload;
import org.junit.jupiter.api.Test;

class GpuDrivenClusterVisibilityValidationHarnessTest {
  private static final MeshletVisibilityFrustum TEST_FRUSTUM =
      MeshletVisibilityFrustum.of(
          new float[] {
            1f, 0f, 0f, 1f,
            -1f, 0f, 0f, 1f,
            0f, 1f, 0f, 1f,
            0f, -1f, 0f, 1f,
            0f, 0f, 1f, 1f,
            0f, 0f, -1f, 1f
          });

  @Test
  void validatesEndToEndChainWithMixedVisibilityAndTiming() throws Exception {
    GpuMeshletBoundsResource bounds =
        createBoundsResource(
            new float[][] {
              {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f},
              {2f, 2f, 2f, 3f, 3f, 3f},
              {-0.25f, -0.25f, -0.25f, 0.25f, 0.25f, 0.25f}
            });
    GpuMeshletDrawMetadataPayload drawMetadata =
        GpuMeshletDrawMetadataPayload.of(
            new int[] {9, 12, 15},
            new int[] {0, 9, 21},
            new int[] {0, 2, 4});

    VulkanMeshletVisibilityCapability visibility =
        new VulkanMeshletVisibilityCapability(src -> new TestBuffer(100L, src.remaining()));
    VulkanMeshletVisibilityCompactionCapability compaction =
        new VulkanMeshletVisibilityCompactionCapability(src -> new TestBuffer(101L, src.remaining()));
    VulkanMeshletIndirectDrawGenerationCapability indirect =
        new VulkanMeshletIndirectDrawGenerationCapability(src -> new TestBuffer(102L, src.remaining()));
    GpuDrivenClusterVisibilityValidationHarness harness =
        new GpuDrivenClusterVisibilityValidationHarness(visibility, compaction, indirect);

    GpuDrivenClusterVisibilityValidationResult result =
        harness.run(
            GpuDrivenClusterVisibilityValidationWork.forAllMeshlets(
                bounds, TEST_FRUSTUM, drawMetadata));

    byte[] expectedFlags = cpuExpectedVisibilityFlags(bounds, TEST_FRUSTUM);
    byte[] actualFlags = new byte[result.visibilityFlags().meshletCount()];
    result.visibilityFlags().payload().flagsBytes().get(actualFlags);
    assertArrayEquals(expectedFlags, actualFlags);

    int[] expectedVisibleIndices = expectedVisibleIndices(expectedFlags);
    assertEquals(expectedVisibleIndices.length, result.visibleMeshlets().visibleMeshletCount());
    ByteBuffer visible = result.visibleMeshlets().payload().visibleIndicesBytes().order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < expectedVisibleIndices.length; i++) {
      assertEquals(expectedVisibleIndices[i], visible.getInt(i * Integer.BYTES));
    }

    assertEquals(expectedVisibleIndices.length, result.indirectDraws().commandCount());
    ByteBuffer commands = result.indirectDraws().payload().commandBytes().order(ByteOrder.LITTLE_ENDIAN);
    for (int commandIndex = 0; commandIndex < expectedVisibleIndices.length; commandIndex++) {
      int meshletIndex = expectedVisibleIndices[commandIndex];
      int base = commandIndex * 20;
      assertEquals(drawMetadata.indexCount(meshletIndex), commands.getInt(base));
      assertEquals(1, commands.getInt(base + 4));
      assertEquals(drawMetadata.firstIndex(meshletIndex), commands.getInt(base + 8));
      assertEquals(drawMetadata.vertexOffset(meshletIndex), commands.getInt(base + 12));
      assertEquals(0, commands.getInt(base + 16));
    }

    assertTrue(result.visibilityNanos() >= 0);
    assertTrue(result.compactionNanos() >= 0);
    assertTrue(result.indirectGenerationNanos() >= 0);
    assertTrue(result.totalNanos() >= 0);
    assertEquals(result.visibleMeshlets().visibleMeshletCount(), result.indirectDraws().commandCount());
  }

  private static byte[] cpuExpectedVisibilityFlags(
      GpuMeshletBoundsResource bounds, MeshletVisibilityFrustum frustum) {
    ByteBuffer bytes = bounds.payload().boundsBytes().order(ByteOrder.LITTLE_ENDIAN);
    float[] planes = frustum.planes24();
    byte[] flags = new byte[bounds.meshletCount()];
    for (int i = 0; i < bounds.meshletCount(); i++) {
      int base = i * 6 * Float.BYTES;
      float minX = bytes.getFloat(base);
      float minY = bytes.getFloat(base + Float.BYTES);
      float minZ = bytes.getFloat(base + (2 * Float.BYTES));
      float maxX = bytes.getFloat(base + (3 * Float.BYTES));
      float maxY = bytes.getFloat(base + (4 * Float.BYTES));
      float maxZ = bytes.getFloat(base + (5 * Float.BYTES));
      flags[i] = (byte) (isVisible(minX, minY, minZ, maxX, maxY, maxZ, planes) ? 1 : 0);
    }
    return flags;
  }

  private static int[] expectedVisibleIndices(byte[] flags) {
    int visibleCount = 0;
    for (byte flag : flags) {
      if (flag != 0) {
        visibleCount++;
      }
    }
    int[] indices = new int[visibleCount];
    int out = 0;
    for (int i = 0; i < flags.length; i++) {
      if (flags[i] != 0) {
        indices[out++] = i;
      }
    }
    return indices;
  }

  private static boolean isVisible(
      float minX,
      float minY,
      float minZ,
      float maxX,
      float maxY,
      float maxZ,
      float[] planes24) {
    for (int i = 0; i < planes24.length; i += 4) {
      float a = planes24[i];
      float b = planes24[i + 1];
      float c = planes24[i + 2];
      float d = planes24[i + 3];
      float px = a >= 0f ? maxX : minX;
      float py = b >= 0f ? maxY : minY;
      float pz = c >= 0f ? maxZ : minZ;
      if ((a * px) + (b * py) + (c * pz) + d < 0f) {
        return false;
      }
    }
    return true;
  }

  private static GpuMeshletBoundsResource createBoundsResource(float[][] meshletBounds) {
    ByteBuffer bytes =
        ByteBuffer.allocate(meshletBounds.length * 6 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float[] bounds : meshletBounds) {
      for (float value : bounds) {
        bytes.putFloat(value);
      }
    }
    bytes.flip();
    GpuMeshletBoundsPayload payload =
        GpuMeshletBoundsPayload.fromLittleEndianBytes(meshletBounds.length, 0, 6, bytes);
    return new GpuMeshletBoundsResource(new TestBuffer(200L, payload.boundsByteSize()), payload);
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
