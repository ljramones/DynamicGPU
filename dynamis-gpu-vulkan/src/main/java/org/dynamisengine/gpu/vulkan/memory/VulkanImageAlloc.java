package org.dynamisengine.gpu.vulkan.memory;

/**
 * Pair of Vulkan handles representing an image and its bound device memory.
 *
 * @param image Vulkan image handle
 * @param memory Vulkan device-memory allocation handle
 */
public record VulkanImageAlloc(long image, long memory) {
}
