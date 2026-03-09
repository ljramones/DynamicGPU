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
import org.dynamisengine.gpu.api.resource.GpuMeshletStreamingPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletStreamingResource;
import org.junit.jupiter.api.Test;

class VulkanMeshletStreamingUploaderTest {
  @Test
  void uploadsNonEmptyPayloadAndPreservesMetadata() throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(3).putInt(64).putInt(32).putInt(4096).putInt(2048).flip();
    GpuMeshletStreamingPayload payload =
        GpuMeshletStreamingPayload.fromLittleEndianBytes(1, 0, 5, bytes);

    VulkanMeshletStreamingUploader uploader =
        new VulkanMeshletStreamingUploader(src -> new TestBuffer(1100L, src.remaining()));

    GpuMeshletStreamingResource uploaded = uploader.upload(payload);
    assertEquals(1, uploaded.streamUnitCount());
    assertEquals(20, uploaded.streamUnitsByteSize());
    assertEquals(20, uploaded.streamUnitsStrideBytes());
    assertEquals(1100L, uploaded.bufferHandle().value());
  }

  @Test
  void rejectsEmptyPayloadUpload() throws Exception {
    GpuMeshletStreamingPayload payload =
        GpuMeshletStreamingPayload.fromLittleEndianBytes(
            0, 0, 5, ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN));
    VulkanMeshletStreamingUploader uploader =
        new VulkanMeshletStreamingUploader(src -> new TestBuffer(1101L, src.remaining()));

    assertThrows(IllegalArgumentException.class, () -> uploader.upload(payload));
  }

  @Test
  void rejectsBufferSizeMismatchFromFactory() throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(3).putInt(64).putInt(32).putInt(4096).putInt(2048).flip();
    GpuMeshletStreamingPayload payload =
        GpuMeshletStreamingPayload.fromLittleEndianBytes(1, 0, 5, bytes);

    VulkanMeshletStreamingUploader uploader =
        new VulkanMeshletStreamingUploader(src -> new TestBuffer(1102L, 8L));

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
