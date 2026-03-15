package org.dynamisengine.gpu.vulkan.compute;

import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.function.BiFunction;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.gpu.MeshletStreamingResidencyCapability;
import org.dynamisengine.gpu.api.gpu.MeshletStreamingResidencyWork;
import org.dynamisengine.gpu.api.resource.GpuMeshletStreamingPayload;
import org.dynamisengine.gpu.api.resource.GpuResolvedMeshletStreamingPayload;
import org.dynamisengine.gpu.api.resource.GpuResolvedMeshletStreamingResource;
import org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferOps;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

/**
 * Minimal Vulkan-side meshlet streaming residency capability.
 *
 * <p>This phase uses a simple deterministic rule: resolve the exact requested stream unit id from
 * uploaded stream-unit metadata.
 */
public final class VulkanMeshletStreamingResidencyCapability implements MeshletStreamingResidencyCapability {
  @FunctionalInterface
  interface ResolvedStreamingBufferFactory {
    GpuBuffer create(ByteBuffer payloadBytes) throws GpuException;
  }

  private final ResolvedStreamingBufferFactory resolvedStreamingBufferFactory;
  private boolean closed;

  public VulkanMeshletStreamingResidencyCapability(
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
    this.resolvedStreamingBufferFactory =
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

  VulkanMeshletStreamingResidencyCapability(ResolvedStreamingBufferFactory resolvedStreamingBufferFactory) {
    this.resolvedStreamingBufferFactory =
        Objects.requireNonNull(resolvedStreamingBufferFactory, "resolvedStreamingBufferFactory");
  }

  @Override
  public GpuResolvedMeshletStreamingResource execute(MeshletStreamingResidencyWork work) throws GpuException {
    ensureOpen();
    Objects.requireNonNull(work, "work");
    if (work.streamingResource().isClosed()) {
      throw new IllegalStateException("streamingResource is already closed");
    }

    GpuResolvedMeshletStreamingPayload payload = resolve(work);
    GpuBuffer buffer = resolvedStreamingBufferFactory.create(payload.resolvedBytes());
    if (buffer.sizeBytes() != payload.byteSize()) {
      throw new IllegalStateException(
          "resolved streaming buffer size mismatch: buffer="
              + buffer.sizeBytes()
              + " payload="
              + payload.byteSize());
    }
    return new GpuResolvedMeshletStreamingResource(buffer, payload);
  }

  @Override
  public void close() {
    closed = true;
  }

  private static GpuResolvedMeshletStreamingPayload resolve(MeshletStreamingResidencyWork work) {
    GpuMeshletStreamingPayload payload = work.streamingResource().payload();
    int unitCount = payload.streamUnitCount();
    if (unitCount == 0) {
      throw new IllegalArgumentException("streaming resource has no stream units");
    }

    ByteBuffer streamUnits = payload.streamUnitsBytes().order(ByteOrder.LITTLE_ENDIAN);
    int offset = payload.streamUnitsOffsetInts();
    int stride = payload.streamUnitsStrideInts();
    int target = work.targetStreamUnitId();

    for (int i = 0; i < unitCount; i++) {
      int baseInt = offset + (i * stride);
      int baseByte = baseInt * Integer.BYTES;
      int streamUnitId = streamUnits.getInt(baseByte);
      if (streamUnitId != target) {
        continue;
      }
      int meshletStart = streamUnits.getInt(baseByte + Integer.BYTES);
      int meshletCount = streamUnits.getInt(baseByte + (2 * Integer.BYTES));
      int payloadByteOffset = streamUnits.getInt(baseByte + (3 * Integer.BYTES));
      int payloadByteSize = streamUnits.getInt(baseByte + (4 * Integer.BYTES));
      return GpuResolvedMeshletStreamingPayload.of(
          streamUnitId, meshletStart, meshletCount, payloadByteOffset, payloadByteSize);
    }

    throw new IllegalArgumentException("targetStreamUnitId not present in streaming resource: " + target);
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("Capability has been closed");
    }
  }
}

