package org.dynamisengine.gpu.vulkan.upload;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsResource;
import org.junit.jupiter.api.Test;

class VulkanMeshletBoundsUploaderTest {
  @Test
  void uploadsNonEmptyPayloadAndPreservesMetadata() throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putFloat(1f).putFloat(2f).putFloat(3f).putFloat(4f).putFloat(5f).putFloat(6f).flip();
    GpuMeshletBoundsPayload payload = GpuMeshletBoundsPayload.fromLittleEndianBytes(1, 0, 6, bytes);

    VulkanMeshletBoundsUploader uploader =
        new VulkanMeshletBoundsUploader(src -> new TestBuffer(900L, src.remaining()));

    GpuMeshletBoundsResource uploaded = uploader.upload(payload);
    assertEquals(1, uploaded.meshletCount());
    assertEquals(24, uploaded.boundsByteSize());
    assertEquals(24, uploaded.boundsStrideBytes());
    assertEquals(900L, uploaded.bufferHandle().value());
  }

  @Test
  void rejectsEmptyPayloadUpload() throws Exception {
    GpuMeshletBoundsPayload payload =
        GpuMeshletBoundsPayload.fromLittleEndianBytes(
            0, 0, 6, ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN));
    VulkanMeshletBoundsUploader uploader =
        new VulkanMeshletBoundsUploader(src -> new TestBuffer(901L, src.remaining()));

    assertThrows(IllegalArgumentException.class, () -> uploader.upload(payload));
  }

  @Test
  void rejectsBufferSizeMismatchFromFactory() throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putFloat(1f).putFloat(2f).putFloat(3f).putFloat(4f).putFloat(5f).putFloat(6f).flip();
    GpuMeshletBoundsPayload payload = GpuMeshletBoundsPayload.fromLittleEndianBytes(1, 0, 6, bytes);

    VulkanMeshletBoundsUploader uploader =
        new VulkanMeshletBoundsUploader(src -> new TestBuffer(902L, 8L));

    assertThrows(IllegalStateException.class, () -> uploader.upload(payload));
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
      return GpuBufferUsage.TRANSFER_DST;
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

