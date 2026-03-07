package org.dynamisengine.gpu.vulkan.memory;

/**
 * Pair of Vulkan handles representing a buffer and its bound device memory.
 *
 * @param buffer Vulkan buffer handle
 * @param memory Vulkan device-memory allocation handle
 */
public record VulkanBufferAlloc(long buffer, long memory) {
}
