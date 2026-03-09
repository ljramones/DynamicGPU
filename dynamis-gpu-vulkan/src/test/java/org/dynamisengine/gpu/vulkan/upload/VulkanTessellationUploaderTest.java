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
import org.dynamisengine.gpu.api.resource.GpuTessellationPayload;
import org.dynamisengine.gpu.api.resource.GpuTessellationResource;
import org.junit.jupiter.api.Test;

class VulkanTessellationUploaderTest {
  @Test
  void uploadsNonEmptyPayloadAndPreservesMetadata() throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0)
        .putInt(0)
        .putInt(12)
        .putInt(3)
        .putInt(Float.floatToRawIntBits(1.0f))
        .putInt(0)
        .flip();
    GpuTessellationPayload payload = GpuTessellationPayload.fromLittleEndianBytes(1, 0, 6, bytes);

    VulkanTessellationUploader uploader =
        new VulkanTessellationUploader(src -> new TestBuffer(1300L, src.remaining()));

    GpuTessellationResource uploaded = uploader.upload(payload);
    assertEquals(1, uploaded.regionCount());
    assertEquals(24, uploaded.regionsByteSize());
    assertEquals(24, uploaded.regionsStrideBytes());
    assertEquals(1300L, uploaded.bufferHandle().value());
  }

  @Test
  void rejectsEmptyPayloadUpload() throws Exception {
    GpuTessellationPayload payload =
        GpuTessellationPayload.fromLittleEndianBytes(
            0, 0, 6, ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN));
    VulkanTessellationUploader uploader =
        new VulkanTessellationUploader(src -> new TestBuffer(1301L, src.remaining()));

    assertThrows(IllegalArgumentException.class, () -> uploader.upload(payload));
  }

  @Test
  void rejectsBufferSizeMismatchFromFactory() throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(0)
        .putInt(0)
        .putInt(12)
        .putInt(3)
        .putInt(Float.floatToRawIntBits(1.0f))
        .putInt(0)
        .flip();
    GpuTessellationPayload payload = GpuTessellationPayload.fromLittleEndianBytes(1, 0, 6, bytes);

    VulkanTessellationUploader uploader =
        new VulkanTessellationUploader(src -> new TestBuffer(1302L, 8L));

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
