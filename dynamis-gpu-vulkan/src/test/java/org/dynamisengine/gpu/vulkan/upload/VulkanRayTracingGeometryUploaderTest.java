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
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryResource;
import org.junit.jupiter.api.Test;

class VulkanRayTracingGeometryUploaderTest {
  @Test
  void uploadsNonEmptyPayloadAndPreservesMetadata() throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(12).putInt(0).putInt(1).flip();
    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(1, 0, 5, bytes);

    VulkanRayTracingGeometryUploader uploader =
        new VulkanRayTracingGeometryUploader(src -> new TestBuffer(1200L, src.remaining()));

    GpuRayTracingGeometryResource uploaded = uploader.upload(payload);
    assertEquals(1, uploaded.regionCount());
    assertEquals(20, uploaded.regionsByteSize());
    assertEquals(20, uploaded.regionsStrideBytes());
    assertEquals(1200L, uploaded.bufferHandle().value());
  }

  @Test
  void rejectsEmptyPayloadUpload() throws Exception {
    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(
            0, 0, 5, ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN));
    VulkanRayTracingGeometryUploader uploader =
        new VulkanRayTracingGeometryUploader(src -> new TestBuffer(1201L, src.remaining()));

    assertThrows(IllegalArgumentException.class, () -> uploader.upload(payload));
  }

  @Test
  void rejectsBufferSizeMismatchFromFactory() throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(0).putInt(0).putInt(12).putInt(0).putInt(1).flip();
    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(1, 0, 5, bytes);

    VulkanRayTracingGeometryUploader uploader =
        new VulkanRayTracingGeometryUploader(src -> new TestBuffer(1202L, 8L));

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

