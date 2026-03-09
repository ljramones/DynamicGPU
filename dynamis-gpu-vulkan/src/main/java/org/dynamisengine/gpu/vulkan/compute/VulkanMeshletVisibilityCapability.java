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
import org.dynamisengine.gpu.api.gpu.MeshletVisibilityCapability;
import org.dynamisengine.gpu.api.gpu.MeshletVisibilityWork;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsResource;
import org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanMemoryOps;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

/**
 * Minimal Vulkan-side meshlet visibility capability.
 *
 * <p>This Phase 2.3 implementation keeps execution correctness-first: visibility flags are
 * produced from the authoritative meshlet bounds + frustum contract and then uploaded into a
 * GPU-managed storage buffer.
 */
public final class VulkanMeshletVisibilityCapability implements MeshletVisibilityCapability {
  @FunctionalInterface
  interface FlagsBufferFactory {
    GpuBuffer create(ByteBuffer payloadBytes) throws GpuException;
  }

  private final FlagsBufferFactory flagsBufferFactory;
  private boolean closed;

  public VulkanMeshletVisibilityCapability(
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
    this.flagsBufferFactory =
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

  VulkanMeshletVisibilityCapability(FlagsBufferFactory flagsBufferFactory) {
    this.flagsBufferFactory = Objects.requireNonNull(flagsBufferFactory, "flagsBufferFactory");
  }

  @Override
  public GpuMeshletVisibilityFlagsResource execute(MeshletVisibilityWork work) throws GpuException {
    ensureOpen();
    Objects.requireNonNull(work, "work");
    if (work.boundsResource().isClosed()) {
      throw new IllegalStateException("boundsResource is already closed");
    }

    ByteBuffer flagsBytes = computeVisibilityFlags(work);
    GpuMeshletVisibilityFlagsPayload payload =
        GpuMeshletVisibilityFlagsPayload.fromBytes(work.meshletCount(), flagsBytes);
    GpuBuffer buffer =
        payload.flagsByteSize() == 0
            ? new EmptyGpuBuffer()
            : flagsBufferFactory.create(payload.flagsBytes());
    if (buffer.sizeBytes() != payload.flagsByteSize()) {
      throw new IllegalStateException(
          "visibility flags buffer size mismatch: buffer="
              + buffer.sizeBytes()
              + " payload="
              + payload.flagsByteSize());
    }
    return new GpuMeshletVisibilityFlagsResource(buffer, payload);
  }

  @Override
  public void close() {
    closed = true;
  }

  private static ByteBuffer computeVisibilityFlags(MeshletVisibilityWork work) {
    int meshletCount = work.meshletCount();
    ByteBuffer flags = ByteBuffer.allocate(meshletCount);
    if (meshletCount == 0) {
      flags.flip();
      return flags;
    }

    GpuMeshletBoundsPayload boundsPayload = work.boundsResource().payload();
    ByteBuffer boundsBytes = boundsPayload.boundsBytes().order(ByteOrder.LITTLE_ENDIAN);
    int boundsOffsetFloats = boundsPayload.boundsOffsetFloats();
    int boundsStrideFloats = boundsPayload.boundsStrideFloats();
    float[] planes = work.frustum().planes24();

    for (int meshletIndex = 0; meshletIndex < meshletCount; meshletIndex++) {
      int byteIndex = (boundsOffsetFloats + (meshletIndex * boundsStrideFloats)) * Float.BYTES;
      float minX = boundsBytes.getFloat(byteIndex);
      float minY = boundsBytes.getFloat(byteIndex + Float.BYTES);
      float minZ = boundsBytes.getFloat(byteIndex + (2 * Float.BYTES));
      float maxX = boundsBytes.getFloat(byteIndex + (3 * Float.BYTES));
      float maxY = boundsBytes.getFloat(byteIndex + (4 * Float.BYTES));
      float maxZ = boundsBytes.getFloat(byteIndex + (5 * Float.BYTES));
      flags.put((byte) (isVisible(minX, minY, minZ, maxX, maxY, maxZ, planes) ? 1 : 0));
    }

    flags.flip();
    return flags;
  }

  private static boolean isVisible(
      float minX,
      float minY,
      float minZ,
      float maxX,
      float maxY,
      float maxZ,
      float[] planes24) {
    for (int i = 0; i < planes24.length; i += 4) {
      float a = planes24[i];
      float b = planes24[i + 1];
      float c = planes24[i + 2];
      float d = planes24[i + 3];

      float px = a >= 0f ? maxX : minX;
      float py = b >= 0f ? maxY : minY;
      float pz = c >= 0f ? maxZ : minZ;
      if ((a * px) + (b * py) + (c * pz) + d < 0f) {
        return false;
      }
    }
    return true;
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
