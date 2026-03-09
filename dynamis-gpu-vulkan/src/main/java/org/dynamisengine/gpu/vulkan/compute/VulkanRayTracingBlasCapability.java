package org.dynamisengine.gpu.vulkan.compute;

import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.function.BiFunction;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.gpu.RayTracingBlasCapability;
import org.dynamisengine.gpu.api.gpu.RayTracingBlasWork;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryResource;
import org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanMemoryOps;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

/**
 * Minimal Vulkan-side BLAS capability foundation.
 *
 * <p>This phase intentionally provides a BLAS build-preparation seam only. It does not execute
 * final Vulkan acceleration-structure build commands yet.
 */
public final class VulkanRayTracingBlasCapability implements RayTracingBlasCapability {
  @FunctionalInterface
  interface BlasBufferFactory {
    GpuBuffer create(ByteBuffer payloadBytes) throws GpuException;
  }

  private final BlasBufferFactory blasBufferFactory;
  private boolean closed;

  public VulkanRayTracingBlasCapability(
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
    this.blasBufferFactory =
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

  VulkanRayTracingBlasCapability(BlasBufferFactory blasBufferFactory) {
    this.blasBufferFactory = Objects.requireNonNull(blasBufferFactory, "blasBufferFactory");
  }

  @Override
  public GpuRayTracingBlasResource execute(RayTracingBlasWork work) throws GpuException {
    ensureOpen();
    Objects.requireNonNull(work, "work");
    GpuRayTracingGeometryResource geometryResource = work.geometryResource();
    if (geometryResource.isClosed()) {
      throw new IllegalStateException("geometryResource is already closed");
    }

    GpuRayTracingBlasPayload payload = GpuRayTracingBlasPayload.forGeometryResource(geometryResource);
    GpuBuffer buffer = blasBufferFactory.create(payload.buildPrepBytes());
    if (buffer.sizeBytes() != payload.byteSize()) {
      throw new IllegalStateException(
          "BLAS payload buffer size mismatch: buffer="
              + buffer.sizeBytes()
              + " payload="
              + payload.byteSize());
    }
    return new GpuRayTracingBlasResource(buffer, payload, geometryResource);
  }

  @Override
  public void close() {
    closed = true;
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("Capability has been closed");
    }
  }
}

