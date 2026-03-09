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
import org.dynamisengine.gpu.api.gpu.MeshletLodSelectionCapability;
import org.dynamisengine.gpu.api.gpu.MeshletLodSelectionWork;
import org.dynamisengine.gpu.api.resource.GpuMeshletLodPayload;
import org.dynamisengine.gpu.api.resource.GpuSelectedMeshletLodPayload;
import org.dynamisengine.gpu.api.resource.GpuSelectedMeshletLodResource;
import org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanMemoryOps;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

/**
 * Minimal Vulkan-side meshlet LOD selection capability.
 *
 * <p>This phase uses a simple deterministic rule: select the exact requested target LOD level
 * from uploaded LOD metadata.
 */
public final class VulkanMeshletLodSelectionCapability implements MeshletLodSelectionCapability {
  @FunctionalInterface
  interface SelectedLodBufferFactory {
    GpuBuffer create(ByteBuffer payloadBytes) throws GpuException;
  }

  private final SelectedLodBufferFactory selectedLodBufferFactory;
  private boolean closed;

  public VulkanMeshletLodSelectionCapability(
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
    this.selectedLodBufferFactory =
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

  VulkanMeshletLodSelectionCapability(SelectedLodBufferFactory selectedLodBufferFactory) {
    this.selectedLodBufferFactory =
        Objects.requireNonNull(selectedLodBufferFactory, "selectedLodBufferFactory");
  }

  @Override
  public GpuSelectedMeshletLodResource execute(MeshletLodSelectionWork work) throws GpuException {
    ensureOpen();
    Objects.requireNonNull(work, "work");
    if (work.lodResource().isClosed()) {
      throw new IllegalStateException("lodResource is already closed");
    }

    GpuSelectedMeshletLodPayload payload = select(work);
    GpuBuffer buffer = selectedLodBufferFactory.create(payload.selectedBytes());
    if (buffer.sizeBytes() != payload.byteSize()) {
      throw new IllegalStateException(
          "selected LOD buffer size mismatch: buffer="
              + buffer.sizeBytes()
              + " payload="
              + payload.byteSize());
    }
    return new GpuSelectedMeshletLodResource(buffer, payload);
  }

  @Override
  public void close() {
    closed = true;
  }

  private static GpuSelectedMeshletLodPayload select(MeshletLodSelectionWork work) {
    GpuMeshletLodPayload payload = work.lodResource().payload();
    int levelCount = payload.levelCount();
    if (levelCount == 0) {
      throw new IllegalArgumentException("LOD resource has no levels");
    }

    ByteBuffer levels = payload.levelsBytes().order(ByteOrder.LITTLE_ENDIAN);
    int offset = payload.levelsOffsetInts();
    int stride = payload.levelsStrideInts();
    int target = work.targetLodLevel();

    for (int i = 0; i < levelCount; i++) {
      int baseInt = offset + (i * stride);
      int baseByte = baseInt * Integer.BYTES;
      int lodLevel = levels.getInt(baseByte);
      if (lodLevel != target) {
        continue;
      }
      int meshletStart = levels.getInt(baseByte + Integer.BYTES);
      int meshletCount = levels.getInt(baseByte + (2 * Integer.BYTES));
      int geometricErrorBits = levels.getInt(baseByte + (3 * Integer.BYTES));
      return GpuSelectedMeshletLodPayload.of(lodLevel, meshletStart, meshletCount, geometricErrorBits);
    }

    throw new IllegalArgumentException("targetLodLevel not present in LOD resource: " + target);
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("Capability has been closed");
    }
  }
}
