package org.dynamisengine.gpu.vulkan.compute;

import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

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
import org.dynamisengine.gpu.api.gpu.MeshletIndirectDrawGenerationCapability;
import org.dynamisengine.gpu.api.gpu.MeshletIndirectDrawGenerationWork;
import org.dynamisengine.gpu.api.resource.GpuMeshletIndirectDrawPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletIndirectDrawResource;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListPayload;
import org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanMemoryOps;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

/**
 * Minimal Vulkan capability that generates indexed-indirect commands from compact visible meshlets.
 */
public final class VulkanMeshletIndirectDrawGenerationCapability
    implements MeshletIndirectDrawGenerationCapability {
  @FunctionalInterface
  interface IndirectBufferFactory {
    GpuBuffer create(ByteBuffer payloadBytes) throws GpuException;
  }

  private final IndirectBufferFactory indirectBufferFactory;
  private boolean closed;

  public VulkanMeshletIndirectDrawGenerationCapability(
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
    this.indirectBufferFactory =
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
                    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    vkFailure);
            return new VulkanGpuBuffer(
                device,
                alloc.buffer(),
                alloc.memory(),
                payloadBytes.remaining(),
                GpuBufferUsage.INDIRECT,
                GpuMemoryLocation.DEVICE_LOCAL);
          }
        };
  }

  VulkanMeshletIndirectDrawGenerationCapability(IndirectBufferFactory indirectBufferFactory) {
    this.indirectBufferFactory = Objects.requireNonNull(indirectBufferFactory, "indirectBufferFactory");
  }

  @Override
  public GpuMeshletIndirectDrawResource execute(MeshletIndirectDrawGenerationWork work) throws GpuException {
    ensureOpen();
    Objects.requireNonNull(work, "work");
    if (work.visibleMeshlets().isClosed()) {
      throw new IllegalStateException("visibleMeshlets resource is already closed");
    }

    GpuMeshletIndirectDrawPayload payload = generateCommands(work);
    GpuBuffer buffer =
        payload.commandByteSize() == 0
            ? new EmptyGpuBuffer()
            : indirectBufferFactory.create(payload.commandBytes());
    if (buffer.sizeBytes() != payload.commandByteSize()) {
      throw new IllegalStateException(
          "indirect command buffer size mismatch: buffer="
              + buffer.sizeBytes()
              + " payload="
              + payload.commandByteSize());
    }
    return new GpuMeshletIndirectDrawResource(buffer, payload);
  }

  @Override
  public void close() {
    closed = true;
  }

  private static GpuMeshletIndirectDrawPayload generateCommands(MeshletIndirectDrawGenerationWork work) {
    int commandCount = work.commandCount();
    ByteBuffer commandBytes =
        ByteBuffer.allocate(commandCount * GpuMeshletIndirectDrawPayload.COMMAND_STRIDE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    GpuVisibleMeshletListPayload visiblePayload = work.visibleMeshlets().payload();
    ByteBuffer visibleIndicesBytes = visiblePayload.visibleIndicesBytes().order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < commandCount; i++) {
      int meshletIndex = visibleIndicesBytes.getInt(i * Integer.BYTES);
      commandBytes.putInt(work.drawMetadata().indexCount(meshletIndex)); // indexCount
      commandBytes.putInt(1); // instanceCount
      commandBytes.putInt(work.drawMetadata().firstIndex(meshletIndex)); // firstIndex
      commandBytes.putInt(work.drawMetadata().vertexOffset(meshletIndex)); // vertexOffset
      commandBytes.putInt(0); // firstInstance
    }
    commandBytes.flip();
    return GpuMeshletIndirectDrawPayload.fromLittleEndianBytes(
        visiblePayload.visibleMeshletCount(), commandCount, commandBytes);
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
      return GpuBufferUsage.INDIRECT;
    }

    @Override
    public GpuMemoryLocation memoryLocation() {
      return GpuMemoryLocation.DEVICE_LOCAL;
    }

    @Override
    public void close() {}
  }
}

