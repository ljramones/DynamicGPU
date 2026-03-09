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
import org.dynamisengine.gpu.api.resource.GpuMeshletLodPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletLodResource;
import org.junit.jupiter.api.Test;

class VulkanMeshletLodUploaderTest {
  @Test
  void uploadsNonEmptyPayloadAndPreservesMetadata() throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(16).putInt(Float.floatToRawIntBits(0.5f)).flip();
    GpuMeshletLodPayload payload = GpuMeshletLodPayload.fromLittleEndianBytes(1, 0, 4, bytes);

    VulkanMeshletLodUploader uploader =
        new VulkanMeshletLodUploader(src -> new TestBuffer(1000L, src.remaining()));

    GpuMeshletLodResource uploaded = uploader.upload(payload);
    assertEquals(1, uploaded.levelCount());
    assertEquals(16, uploaded.levelsByteSize());
    assertEquals(16, uploaded.levelsStrideBytes());
    assertEquals(1000L, uploaded.bufferHandle().value());
  }

  @Test
  void rejectsEmptyPayloadUpload() throws Exception {
    GpuMeshletLodPayload payload =
        GpuMeshletLodPayload.fromLittleEndianBytes(
            0, 0, 4, ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN));
    VulkanMeshletLodUploader uploader =
        new VulkanMeshletLodUploader(src -> new TestBuffer(1001L, src.remaining()));

    assertThrows(IllegalArgumentException.class, () -> uploader.upload(payload));
  }

  @Test
  void rejectsBufferSizeMismatchFromFactory() throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(16).putInt(Float.floatToRawIntBits(0.5f)).flip();
    GpuMeshletLodPayload payload = GpuMeshletLodPayload.fromLittleEndianBytes(1, 0, 4, bytes);

    VulkanMeshletLodUploader uploader =
        new VulkanMeshletLodUploader(src -> new TestBuffer(1002L, 8L));

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
