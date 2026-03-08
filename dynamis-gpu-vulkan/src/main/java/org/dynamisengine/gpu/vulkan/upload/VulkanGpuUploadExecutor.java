package org.dynamisengine.gpu.vulkan.upload;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.function.BiFunction;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshResource;
import org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan;
import org.dynamisengine.gpu.api.upload.GpuUploadExecutor;
import org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanMemoryOps;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

/**
 * Vulkan implementation of {@link GpuUploadExecutor} for runtime geometry payloads.
 */
public final class VulkanGpuUploadExecutor implements GpuUploadExecutor {
  private final VkDevice device;
  private final VkPhysicalDevice physicalDevice;
  private final long commandPool;
  private final VkQueue graphicsQueue;
  private final BiFunction<String, Integer, GpuException> vkFailure;

  public VulkanGpuUploadExecutor(
      VkDevice device,
      VkPhysicalDevice physicalDevice,
      long commandPool,
      VkQueue graphicsQueue) {
    this.device = Objects.requireNonNull(device, "device");
    this.physicalDevice = Objects.requireNonNull(physicalDevice, "physicalDevice");
    if (commandPool == 0L) {
      throw new IllegalArgumentException("commandPool must not be VK_NULL_HANDLE");
    }
    this.commandPool = commandPool;
    this.graphicsQueue = Objects.requireNonNull(graphicsQueue, "graphicsQueue");
    this.vkFailure =
        (op, code) ->
            new GpuException(
                GpuErrorCode.BACKEND_INIT_FAILED, op + " failed with code " + code, false);
  }

  @Override
  public GpuMeshResource upload(GpuGeometryUploadPlan plan) throws GpuException {
    Objects.requireNonNull(plan, "plan");

    VulkanGpuBuffer vertexBuffer = null;
    VulkanGpuBuffer indexBuffer = null;
    try (MemoryStack stack = MemoryStack.stackPush()) {
      ByteBuffer vertexBytes = toDirectCopy(plan.vertexData());
      VulkanBufferAlloc vertexAlloc =
          VulkanMemoryOps.createDeviceLocalBufferWithStaging(
              device,
              physicalDevice,
              commandPool,
              graphicsQueue,
              stack,
              vertexBytes,
              toVkBufferUsage(GpuBufferUsage.VERTEX),
              vkFailure);
      vertexBuffer =
          new VulkanGpuBuffer(
              device,
              vertexAlloc.buffer(),
              vertexAlloc.memory(),
              vertexBytes.remaining(),
              GpuBufferUsage.VERTEX,
              GpuMemoryLocation.DEVICE_LOCAL);

      if (plan.indexData() != null) {
        ByteBuffer indexBytes = toDirectCopy(plan.indexData());
        VulkanBufferAlloc indexAlloc =
            VulkanMemoryOps.createDeviceLocalBufferWithStaging(
                device,
                physicalDevice,
                commandPool,
                graphicsQueue,
                stack,
                indexBytes,
                toVkBufferUsage(GpuBufferUsage.INDEX),
                vkFailure);
        indexBuffer =
            new VulkanGpuBuffer(
                device,
                indexAlloc.buffer(),
                indexAlloc.memory(),
                indexBytes.remaining(),
                GpuBufferUsage.INDEX,
                GpuMemoryLocation.DEVICE_LOCAL);
      }

      return new GpuMeshResource(
          vertexBuffer, indexBuffer, plan.vertexLayout(), plan.indexType(), plan.submeshes());
    } catch (Throwable t) {
      closeQuietly(indexBuffer);
      closeQuietly(vertexBuffer);
      if (t instanceof GpuException gpuException) {
        throw gpuException;
      }
      throw new GpuException(
          GpuErrorCode.BACKEND_INIT_FAILED, "Geometry upload failed: " + t.getMessage(), t, false);
    }
  }

  static int toVkBufferUsage(GpuBufferUsage usage) {
    return switch (usage) {
      case VERTEX -> VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      case INDEX -> VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      case TRANSFER_SRC -> VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      case TRANSFER_DST -> VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    };
  }

  static int toVkMemoryProperties(GpuMemoryLocation location) {
    return switch (location) {
      case DEVICE_LOCAL -> VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      case HOST_VISIBLE -> VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    };
  }

  static ByteBuffer toDirectCopy(ByteBuffer source) {
    ByteBuffer duplicate = source.duplicate();
    ByteBuffer direct = ByteBuffer.allocateDirect(duplicate.remaining());
    direct.put(duplicate);
    direct.flip();
    return direct;
  }

  private static void closeQuietly(VulkanGpuBuffer buffer) {
    if (buffer == null) {
      return;
    }
    try {
      buffer.close();
    } catch (RuntimeException ignored) {
      // Preserve original failure in upload path.
    }
  }
}
