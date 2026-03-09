package org.dynamisengine.gpu.vulkan.compute;

import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.function.BiFunction;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.gpu.MeshletVisibilityCompactionCapability;
import org.dynamisengine.gpu.api.gpu.MeshletVisibilityCompactionWork;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsPayload;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListPayload;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListResource;
import org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanMemoryOps;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

/**
 * Minimal Vulkan-side compaction capability for meshlet visibility flags.
 *
 * <p>This Phase 2.4 slice performs correctness-first compaction into a compact visible index list.
 */
public final class VulkanMeshletVisibilityCompactionCapability
    implements MeshletVisibilityCompactionCapability {
  @FunctionalInterface
  interface VisibleListBufferFactory {
    GpuBuffer create(ByteBuffer payloadBytes) throws GpuException;
  }

  private final VisibleListBufferFactory visibleListBufferFactory;
  private boolean closed;

  public VulkanMeshletVisibilityCompactionCapability(
      VkDevice device, VkPhysicalDevice physicalDevice, long commandPool, VkQueue graphicsQueue) {
    Objects.requireNonNull(device, "device");
    Objects.requireNonNull(physicalDevice, "physicalDevice");
    Objects.requireNonNull(graphicsQueue, "graphicsQueue");
    if (commandPool == 0L) {
      throw new IllegalArgumentException("commandPool must not be VK_NULL_HANDLE");
    }
    BiFunction<String, Integer, GpuException> vkFailure =
        (op, code) ->
            new GpuException(
                GpuErrorCode.BACKEND_INIT_FAILED, op + " failed with code " + code, false);
    this.visibleListBufferFactory =
        payloadBytes -> {
          try (MemoryStack stack = MemoryStack.stackPush()) {
            VulkanBufferAlloc alloc =
                VulkanMemoryOps.createDeviceLocalBufferWithStaging(
                    device,
                    physicalDevice,
                    commandPool,
                    graphicsQueue,
                    stack,
                    payloadBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    vkFailure);
            return new VulkanGpuBuffer(
                device,
                alloc.buffer(),
                alloc.memory(),
                payloadBytes.remaining(),
                GpuBufferUsage.STORAGE,
                GpuMemoryLocation.DEVICE_LOCAL);
          }
        };
  }

  VulkanMeshletVisibilityCompactionCapability(VisibleListBufferFactory visibleListBufferFactory) {
    this.visibleListBufferFactory = Objects.requireNonNull(visibleListBufferFactory, "visibleListBufferFactory");
  }

  @Override
  public GpuVisibleMeshletListResource execute(MeshletVisibilityCompactionWork work) throws GpuException {
    ensureOpen();
    Objects.requireNonNull(work, "work");
    if (work.flagsResource().isClosed()) {
      throw new IllegalStateException("flagsResource is already closed");
    }

    GpuVisibleMeshletListPayload payload = compact(work);
    GpuBuffer buffer =
        payload.visibleIndicesByteSize() == 0
            ? new EmptyGpuBuffer()
            : visibleListBufferFactory.create(payload.visibleIndicesBytes());
    if (buffer.sizeBytes() != payload.visibleIndicesByteSize()) {
      throw new IllegalStateException(
          "visible list buffer size mismatch: buffer="
              + buffer.sizeBytes()
              + " payload="
              + payload.visibleIndicesByteSize());
    }
    return new GpuVisibleMeshletListResource(buffer, payload);
  }

  @Override
  public void close() {
    closed = true;
  }

  private static GpuVisibleMeshletListPayload compact(MeshletVisibilityCompactionWork work) {
    GpuMeshletVisibilityFlagsPayload flagsPayload = work.flagsResource().payload();
    ByteBuffer flags = flagsPayload.flagsBytes();
    int meshletCount = work.meshletCount();

    int visibleCount = 0;
    for (int i = 0; i < meshletCount; i++) {
      if (flags.get(i) != 0) {
        visibleCount++;
      }
    }

    ByteBuffer visibleIndices = ByteBuffer.allocate(visibleCount * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < meshletCount; i++) {
      if (flags.get(i) != 0) {
        visibleIndices.putInt(i);
      }
    }
    visibleIndices.flip();
    return GpuVisibleMeshletListPayload.fromLittleEndianBytes(meshletCount, visibleCount, visibleIndices);
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("Capability has been closed");
    }
  }

  private static final class EmptyGpuBuffer implements GpuBuffer {
    @Override
    public GpuBufferHandle handle() {
      return new GpuBufferHandle(0L);
    }

    @Override
    public long sizeBytes() {
      return 0L;
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

