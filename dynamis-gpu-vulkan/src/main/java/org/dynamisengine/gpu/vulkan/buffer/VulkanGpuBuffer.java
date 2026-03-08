package org.dynamisengine.gpu.vulkan.buffer;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.lwjgl.vulkan.VkDevice;

import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.vkDestroyBuffer;
import static org.lwjgl.vulkan.VK10.vkFreeMemory;

/**
 * Vulkan-backed {@link GpuBuffer} owning a buffer and bound device memory allocation.
 */
public final class VulkanGpuBuffer implements GpuBuffer {
  private final VkDevice device;
  private final GpuBufferHandle handle;
  private final long memoryHandle;
  private final long sizeBytes;
  private final GpuBufferUsage usage;
  private final GpuMemoryLocation memoryLocation;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public VulkanGpuBuffer(
      VkDevice device,
      long bufferHandle,
      long memoryHandle,
      long sizeBytes,
      GpuBufferUsage usage,
      GpuMemoryLocation memoryLocation) {
    this.device = Objects.requireNonNull(device, "device");
    if (memoryHandle == VK_NULL_HANDLE) {
      throw new IllegalArgumentException("memoryHandle must not be VK_NULL_HANDLE");
    }
    if (sizeBytes <= 0L) {
      throw new IllegalArgumentException("sizeBytes must be > 0");
    }
    this.handle = new GpuBufferHandle(bufferHandle);
    this.memoryHandle = memoryHandle;
    this.sizeBytes = sizeBytes;
    this.usage = Objects.requireNonNull(usage, "usage");
    this.memoryLocation = Objects.requireNonNull(memoryLocation, "memoryLocation");
  }

  @Override
  public GpuBufferHandle handle() {
    return handle;
  }

  @Override
  public long sizeBytes() {
    return sizeBytes;
  }

  @Override
  public GpuBufferUsage usage() {
    return usage;
  }

  @Override
  public GpuMemoryLocation memoryLocation() {
    return memoryLocation;
  }

  @Override
  public void close() {
    if (!closed.compareAndSet(false, true)) {
      return;
    }
    vkDestroyBuffer(device, handle.value(), null);
    vkFreeMemory(device, memoryHandle, null);
  }

  public boolean isClosed() {
    return closed.get();
  }
}
