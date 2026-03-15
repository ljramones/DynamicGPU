package org.dynamisengine.gpu.vulkan.upload;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferOps;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;

import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.vkDestroyBuffer;
import static org.lwjgl.vulkan.VK10.vkFreeMemory;

/**
 * Reuses device-local Vulkan buffer allocations across uploads to reduce allocation churn.
 */
final class VulkanDeviceLocalBufferPool implements AutoCloseable {
  private final VkDevice device;
  private final VkPhysicalDevice physicalDevice;
  private final Map<PoolKey, ArrayDeque<PooledAlloc>> freeByKey = new HashMap<>();
  private final Set<PooledAlloc> allAllocations = new HashSet<>();
  private final AtomicBoolean closed = new AtomicBoolean(false);

  VulkanDeviceLocalBufferPool(VkDevice device, VkPhysicalDevice physicalDevice) {
    this.device = Objects.requireNonNull(device, "device");
    this.physicalDevice = Objects.requireNonNull(physicalDevice, "physicalDevice");
  }

  Lease acquire(MemoryStack stack, int requestedSizeBytes, int usageFlags) throws GpuException {
    ensureOpen();
    if (requestedSizeBytes <= 0) {
      throw new IllegalArgumentException("requestedSizeBytes must be > 0");
    }
    int bucketSize = roundUpToPowerOfTwo(requestedSizeBytes);
    PoolKey key = new PoolKey(usageFlags, bucketSize);
    ArrayDeque<PooledAlloc> free = freeByKey.computeIfAbsent(key, unused -> new ArrayDeque<>());
    PooledAlloc alloc = free.pollFirst();
    if (alloc == null) {
      VulkanBufferAlloc created =
          VulkanBufferOps.createBuffer(
              device,
              physicalDevice,
              stack,
              bucketSize,
              usageFlags,
              VulkanGpuUploadExecutor.toVkMemoryProperties(
                  org.dynamisengine.gpu.api.buffer.GpuMemoryLocation.DEVICE_LOCAL));
      alloc = new PooledAlloc(created.buffer(), created.memory(), bucketSize);
      allAllocations.add(alloc);
    }
    if (alloc.inUse) {
      throw new IllegalStateException(
          "Pooled allocation reuse before retirement buffer="
              + alloc.buffer
              + " size="
              + alloc.sizeBytes);
    }
    alloc.inUse = true;
    return new Lease(this, key, alloc, requestedSizeBytes);
  }

  @Override
  public void close() {
    if (!closed.compareAndSet(false, true)) {
      return;
    }
    for (PooledAlloc allocation : allAllocations) {
      if (allocation.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, allocation.buffer, null);
      }
      if (allocation.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, allocation.memory, null);
      }
    }
    allAllocations.clear();
    freeByKey.clear();
  }

  private void release(PoolKey key, PooledAlloc alloc) {
    if (closed.get()) {
      return;
    }
    if (!alloc.inUse) {
      throw new IllegalStateException(
          "Pooled allocation double-release buffer=" + alloc.buffer + " size=" + alloc.sizeBytes);
    }
    alloc.inUse = false;
    freeByKey.computeIfAbsent(key, unused -> new ArrayDeque<>()).offerFirst(alloc);
  }

  private void ensureOpen() {
    if (closed.get()) {
      throw new IllegalStateException("VulkanDeviceLocalBufferPool is closed");
    }
  }

  private static int roundUpToPowerOfTwo(int value) {
    int v = Math.max(256, value);
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
  }

  static final class Lease {
    private final VulkanDeviceLocalBufferPool owner;
    private final PoolKey key;
    private final PooledAlloc allocation;
    private final int requestedSizeBytes;
    private final AtomicBoolean released = new AtomicBoolean(false);

    private Lease(
        VulkanDeviceLocalBufferPool owner, PoolKey key, PooledAlloc allocation, int requestedSizeBytes) {
      this.owner = owner;
      this.key = key;
      this.allocation = allocation;
      this.requestedSizeBytes = requestedSizeBytes;
    }

    long bufferHandle() {
      return allocation.buffer;
    }

    long memoryHandle() {
      return allocation.memory;
    }

    int requestedSizeBytes() {
      return requestedSizeBytes;
    }

    Runnable releaseAction() {
      return this::release;
    }

    private void release() {
      if (!released.compareAndSet(false, true)) {
        return;
      }
      owner.release(key, allocation);
    }
  }

  private record PoolKey(int usageFlags, int bucketSize) {}

  private static final class PooledAlloc {
    private final long buffer;
    private final long memory;
    private final int sizeBytes;
    private boolean inUse;

    private PooledAlloc(long buffer, long memory, int sizeBytes) {
      this.buffer = buffer;
      this.memory = memory;
      this.sizeBytes = sizeBytes;
      this.inUse = false;
    }
  }
}
