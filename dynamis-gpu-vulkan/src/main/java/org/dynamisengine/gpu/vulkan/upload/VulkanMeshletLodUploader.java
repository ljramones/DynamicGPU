package org.dynamisengine.gpu.vulkan.upload;

import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.function.BiFunction;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshletLodPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletLodResource;
import org.dynamisengine.gpu.api.upload.GpuMeshletLodUploader;
import org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferOps;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

/**
 * Vulkan upload/create path for meshlet LOD resources.
 *
 * <p>LOD selection execution is intentionally deferred; this class only creates/uploads
 * the LOD metadata buffer resource.
 */
public final class VulkanMeshletLodUploader implements GpuMeshletLodUploader {
  @FunctionalInterface
  interface BufferFactory {
    GpuBuffer create(ByteBuffer payloadBytes) throws GpuException;
  }

  private final BufferFactory bufferFactory;

  public VulkanMeshletLodUploader(
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
    this.bufferFactory =
        payloadBytes -> {
          try (MemoryStack stack = MemoryStack.stackPush()) {
            VulkanBufferAlloc alloc =
                VulkanBufferOps.createDeviceLocalBufferWithStaging(
                    device,
                    physicalDevice,
                    commandPool,
                    graphicsQueue,
                    stack,
                    payloadBytes,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    vkFailure);
            return new VulkanGpuBuffer(
                device,
                alloc.buffer(),
                alloc.memory(),
                payloadBytes.remaining(),
                GpuBufferUsage.TRANSFER_DST,
                GpuMemoryLocation.DEVICE_LOCAL);
          }
        };
  }

  VulkanMeshletLodUploader(BufferFactory bufferFactory) {
    this.bufferFactory = Objects.requireNonNull(bufferFactory, "bufferFactory");
  }

  @Override
  public GpuMeshletLodResource upload(GpuMeshletLodPayload payload) throws GpuException {
    Objects.requireNonNull(payload, "payload");
    if (payload.levelCount() == 0 || payload.levelsByteSize() == 0) {
      throw new IllegalArgumentException("meshlet LOD upload requires non-empty payload");
    }

    ByteBuffer payloadBytes = payload.levelsBytes();
    GpuBuffer buffer = bufferFactory.create(payloadBytes);
    if (buffer.sizeBytes() != payload.levelsByteSize()) {
      throw new IllegalStateException(
          "uploaded buffer size mismatch: buffer="
              + buffer.sizeBytes()
              + " payload="
              + payload.levelsByteSize());
    }
    return new GpuMeshletLodResource(buffer, payload);
  }
}
