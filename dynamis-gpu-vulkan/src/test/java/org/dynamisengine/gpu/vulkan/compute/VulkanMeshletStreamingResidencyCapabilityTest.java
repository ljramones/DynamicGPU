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
import org.dynamisengine.gpu.api.gpu.MeshletStreamingResidencyWork;
import org.dynamisengine.gpu.api.resource.GpuMeshletStreamingPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletStreamingResource;
import org.dynamisengine.gpu.api.resource.GpuResolvedMeshletStreamingResource;
import org.junit.jupiter.api.Test;

class VulkanMeshletStreamingResidencyCapabilityTest {
  @Test
  void resolvesSingleRequestedStreamUnit() throws Exception {
    GpuMeshletStreamingResource streamingResource = createStreamingResource();
    VulkanMeshletStreamingResidencyCapability capability =
        new VulkanMeshletStreamingResidencyCapability(src -> new TestBuffer(4000L, src.remaining()));

    GpuResolvedMeshletStreamingResource resolved =
        capability.execute(MeshletStreamingResidencyWork.forTargetStreamUnit(streamingResource, 20));

    assertEquals(20, resolved.streamUnitId());
    assertEquals(64, resolved.meshletStart());
    assertEquals(32, resolved.meshletCount());
    assertEquals(4096, resolved.payloadByteOffset());
    assertEquals(2048, resolved.payloadByteSize());
    assertEquals(4000L, resolved.bufferHandle().value());
  }

  @Test
  void resolvesDifferentRequestedStreamUnit() throws Exception {
    GpuMeshletStreamingResource streamingResource = createStreamingResource();
    VulkanMeshletStreamingResidencyCapability capability =
        new VulkanMeshletStreamingResidencyCapability(src -> new TestBuffer(4001L, src.remaining()));

    GpuResolvedMeshletStreamingResource resolved =
        capability.execute(MeshletStreamingResidencyWork.forTargetStreamUnit(streamingResource, 30));

    assertEquals(30, resolved.streamUnitId());
    assertEquals(96, resolved.meshletStart());
    assertEquals(16, resolved.meshletCount());
    assertEquals(6144, resolved.payloadByteOffset());
    assertEquals(1024, resolved.payloadByteSize());
  }

  @Test
  void rejectsTargetStreamUnitNotPresent() throws Exception {
    GpuMeshletStreamingResource streamingResource = createStreamingResource();
    VulkanMeshletStreamingResidencyCapability capability =
        new VulkanMeshletStreamingResidencyCapability(src -> new TestBuffer(4002L, src.remaining()));

    assertThrows(
        IllegalArgumentException.class,
        () -> capability.execute(MeshletStreamingResidencyWork.forTargetStreamUnit(streamingResource, 999)));
  }

  @Test
  void rejectsUploadBufferSizeMismatch() throws Exception {
    GpuMeshletStreamingResource streamingResource = createStreamingResource();
    VulkanMeshletStreamingResidencyCapability capability =
        new VulkanMeshletStreamingResidencyCapability(src -> new TestBuffer(4003L, 8L));

    assertThrows(
        IllegalStateException.class,
        () -> capability.execute(MeshletStreamingResidencyWork.forTargetStreamUnit(streamingResource, 20)));
  }

  private static GpuMeshletStreamingResource createStreamingResource() {
    ByteBuffer bytes = ByteBuffer.allocate(3 * 5 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes
        .putInt(10)
        .putInt(0)
        .putInt(64)
        .putInt(0)
        .putInt(4096)
        .putInt(20)
        .putInt(64)
        .putInt(32)
        .putInt(4096)
        .putInt(2048)
        .putInt(30)
        .putInt(96)
        .putInt(16)
        .putInt(6144)
        .putInt(1024)
        .flip();
    GpuMeshletStreamingPayload payload =
        GpuMeshletStreamingPayload.fromLittleEndianBytes(3, 0, 5, bytes);
    return new GpuMeshletStreamingResource(new TestBuffer(3333L, payload.streamUnitsByteSize()), payload);
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

