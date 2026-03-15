package org.dynamisengine.gpu.vulkan.upload;

import java.util.function.BiFunction;
import org.dynamisengine.gpu.api.error.GpuException;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;

import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkDestroyFence;
import static org.lwjgl.vulkan.VK10.vkFreeCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkGetFenceStatus;
import static org.lwjgl.vulkan.VK10.vkWaitForFences;

/**
 * Package-private utility for Vulkan fence and command-buffer synchronisation helpers.
 */
final class VulkanUploadSyncOps {

  private final VkDevice device;
  private final long commandPool;
  private final BiFunction<String, Integer, GpuException> vkFailure;
  private final boolean debugUpload;

  VulkanUploadSyncOps(
      VkDevice device,
      long commandPool,
      BiFunction<String, Integer, GpuException> vkFailure,
      boolean debugUpload) {
    this.device = device;
    this.commandPool = commandPool;
    this.vkFailure = vkFailure;
    this.debugUpload = debugUpload;
  }

  void waitFenceOrThrow(long fenceHandle, long timeoutNanos) throws GpuException {
    try (MemoryStack stack = MemoryStack.stackPush()) {
      if (debugUpload) {
        int status = fenceStatus(fenceHandle);
        System.out.println(
            "[VulkanGpuUploadExecutor] fenceStatusBeforeWait=" + status + " fence=" + fenceHandle);
      }
      int waitResult =
          vkWaitForFences(device, stack.longs(fenceHandle), true, Math.max(1L, timeoutNanos));
      if (waitResult != VK_SUCCESS) {
        throw vkFailure.apply("vkWaitForFences(upload-batch)", waitResult);
      }
      if (debugUpload) {
        int status = fenceStatus(fenceHandle);
        System.out.println(
            "[VulkanGpuUploadExecutor] fenceStatusAfterWait=" + status + " fence=" + fenceHandle);
      }
    }
  }

  int fenceStatus(long fenceHandle) {
    return vkGetFenceStatus(device, fenceHandle);
  }

  void destroyFenceQuietly(long fenceHandle) {
    if (fenceHandle != VK_NULL_HANDLE) {
      vkDestroyFence(device, fenceHandle, null);
    }
  }

  void freeCommandBufferQuietly(long commandBufferHandle) {
    if (commandBufferHandle == VK_NULL_HANDLE) {
      return;
    }
    try (MemoryStack stack = MemoryStack.stackPush()) {
      vkFreeCommandBuffers(device, commandPool, stack.pointers(commandBufferHandle));
    }
  }
}
