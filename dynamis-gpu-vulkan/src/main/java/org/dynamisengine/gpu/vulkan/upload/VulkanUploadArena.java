package org.dynamisengine.gpu.vulkan.upload;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanMemoryOps;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;

import static org.lwjgl.system.MemoryUtil.memAddress;
import static org.lwjgl.system.MemoryUtil.memByteBuffer;
import static org.lwjgl.system.MemoryUtil.memCopy;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkDestroyBuffer;
import static org.lwjgl.vulkan.VK10.vkFreeMemory;
import static org.lwjgl.vulkan.VK10.vkMapMemory;
import static org.lwjgl.vulkan.VK10.vkUnmapMemory;

/**
 * Reusable host-visible upload arena with simple linear suballocation and submit-time reset.
 */
final class VulkanUploadArena implements AutoCloseable {
  private static final int DEFAULT_ALIGNMENT = 16;

  private final VkDevice device;
  private final VkPhysicalDevice physicalDevice;
  private VulkanBufferAlloc allocation;
  private ByteBuffer mapped;
  private int capacityBytes;
  private int headBytes;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  VulkanUploadArena(VkDevice device, VkPhysicalDevice physicalDevice, int initialCapacityBytes)
      throws GpuException {
    this.device = Objects.requireNonNull(device, "device");
    this.physicalDevice = Objects.requireNonNull(physicalDevice, "physicalDevice");
    int initial = Math.max(64 * 1024, roundUpToPowerOfTwo(initialCapacityBytes));
    createBackingAllocation(initial);
    reset();
  }

  Slice stage(MemoryStack stack, ByteBuffer source, int alignment) throws GpuException {
    ensureOpen();
    Objects.requireNonNull(stack, "stack");
    Objects.requireNonNull(source, "source");
    int size = source.remaining();
    int effectiveAlignment = Math.max(DEFAULT_ALIGNMENT, alignment);
    int startOffset = alignUp(headBytes, effectiveAlignment);
    ensureCapacity(stack, startOffset + size);
    int endOffset = startOffset + size;
    memCopy(memAddress(source), memAddress(mapped) + startOffset, size);
    headBytes = endOffset;
    return new Slice(allocation.buffer(), startOffset, size);
  }

  void reset() {
    ensureOpen();
    headBytes = 0;
  }

  @Override
  public void close() {
    if (!closed.compareAndSet(false, true)) {
      return;
    }
    destroyBackingAllocation();
  }

  private void ensureCapacity(MemoryStack stack, int requiredBytes) throws GpuException {
    if (requiredBytes <= capacityBytes) {
      return;
    }
    int expanded = roundUpToPowerOfTwo(requiredBytes);
    recreateBackingAllocation(stack, expanded);
    headBytes = 0;
  }

  private void recreateBackingAllocation(MemoryStack stack, int newCapacityBytes) throws GpuException {
    destroyBackingAllocation();
    createBackingAllocation(newCapacityBytes);
  }

  private void createBackingAllocation(int newCapacityBytes) throws GpuException {
    try (MemoryStack stack = MemoryStack.stackPush()) {
      createBackingAllocation(stack, newCapacityBytes);
    }
  }

  private void createBackingAllocation(MemoryStack stack, int newCapacityBytes) throws GpuException {
    allocation =
        VulkanMemoryOps.createBuffer(
            device,
            physicalDevice,
            stack,
            newCapacityBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    PointerBuffer pMapped = stack.mallocPointer(1);
    int mapResult = vkMapMemory(device, allocation.memory(), 0, newCapacityBytes, 0, pMapped);
    if (mapResult != VK_SUCCESS) {
      destroyBufferAndMemory(allocation.buffer(), allocation.memory());
      allocation = null;
      throw new GpuException(
          org.dynamisengine.gpu.api.error.GpuErrorCode.BACKEND_INIT_FAILED,
          "vkMapMemory(upload-arena) failed: " + mapResult,
          false);
    }
    mapped = memByteBuffer(pMapped.get(0), newCapacityBytes);
    capacityBytes = newCapacityBytes;
  }

  private void destroyBackingAllocation() {
    if (allocation == null) {
      return;
    }
    vkUnmapMemory(device, allocation.memory());
    destroyBufferAndMemory(allocation.buffer(), allocation.memory());
    allocation = null;
    mapped = null;
    capacityBytes = 0;
    headBytes = 0;
  }

  private void destroyBufferAndMemory(long buffer, long memory) {
    if (buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, buffer, null);
    }
    if (memory != VK_NULL_HANDLE) {
      vkFreeMemory(device, memory, null);
    }
  }

  private void ensureOpen() {
    if (closed.get()) {
      throw new IllegalStateException("VulkanUploadArena is closed");
    }
  }

  private static int alignUp(int value, int alignment) {
    int mask = alignment - 1;
    return (value + mask) & ~mask;
  }

  private static int roundUpToPowerOfTwo(int value) {
    int v = Math.max(1024, value);
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
  }

  record Slice(long stagingBuffer, long offsetBytes, int sizeBytes) {}
}
