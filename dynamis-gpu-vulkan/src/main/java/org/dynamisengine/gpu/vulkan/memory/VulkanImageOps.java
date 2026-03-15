package org.dynamisengine.gpu.vulkan.memory;

import java.util.function.BiFunction;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkBufferImageCopy;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkImageCreateInfo;
import org.lwjgl.vulkan.VkImageMemoryBarrier;
import org.lwjgl.vulkan.VkMemoryAllocateInfo;
import org.lwjgl.vulkan.VkMemoryRequirements;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.VK_IMAGE_ASPECT_COLOR_BIT;
import static org.lwjgl.vulkan.VK10.VK_IMAGE_LAYOUT_UNDEFINED;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_QUEUE_FAMILY_IGNORED;
import static org.lwjgl.vulkan.VK10.VK_SAMPLE_COUNT_1_BIT;
import static org.lwjgl.vulkan.VK10.VK_SHARING_MODE_EXCLUSIVE;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkAllocateMemory;
import static org.lwjgl.vulkan.VK10.vkFreeMemory;

/**
 * Static Vulkan helper methods for image creation and layout transition operations.
 */
public final class VulkanImageOps {
    private VulkanImageOps() {
    }

    /**
     * Convenience overload for image allocation with one mip level.
     */
    public static VulkanImageAlloc createImage(
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            MemoryStack stack,
            int width,
            int height,
            int format,
            int tiling,
            int usage,
            int properties,
            int arrayLayers
    ) throws GpuException {
        return createImage(
                device,
                physicalDevice,
                stack,
                width,
                height,
                format,
                tiling,
                usage,
                properties,
                arrayLayers,
                1
        );
    }

    /**
     * Convenience overload for image allocation with explicit mip level count.
     */
    public static VulkanImageAlloc createImage(
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            MemoryStack stack,
            int width,
            int height,
            int format,
            int tiling,
            int usage,
            int properties,
            int arrayLayers,
            int mipLevels
    ) throws GpuException {
        return createImage(
                device,
                physicalDevice,
                stack,
                width,
                height,
                format,
                tiling,
                usage,
                properties,
                arrayLayers,
                mipLevels,
                0
        );
    }

    /**
     * Creates a Vulkan image and its bound memory allocation.
     */
    public static VulkanImageAlloc createImage(
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            MemoryStack stack,
            int width,
            int height,
            int format,
            int tiling,
            int usage,
            int properties,
            int arrayLayers,
            int mipLevels,
            int createFlags
    ) throws GpuException {
        VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack)
                .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                .flags(createFlags)
                .imageType(VK10.VK_IMAGE_TYPE_2D)
                .extent(e -> e.width(width).height(height).depth(1))
                .mipLevels(Math.max(1, mipLevels))
                .arrayLayers(Math.max(1, arrayLayers))
                .format(format)
                .tiling(tiling)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED)
                .usage(usage)
                .samples(VK_SAMPLE_COUNT_1_BIT)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE);

        var pImage = stack.longs(VK_NULL_HANDLE);
        int createImageResult = VK10.vkCreateImage(device, imageInfo, null, pImage);
        if (createImageResult != VK_SUCCESS || pImage.get(0) == VK_NULL_HANDLE) {
            throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "vkCreateImage failed: " + createImageResult, false);
        }
        long image = pImage.get(0);

        VkMemoryRequirements memReq = VkMemoryRequirements.calloc(stack);
        VK10.vkGetImageMemoryRequirements(device, image, memReq);

        VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(memReq.size())
                .memoryTypeIndex(VulkanMemoryOps.findMemoryType(physicalDevice, memReq.memoryTypeBits(), properties));

        var pMemory = stack.longs(VK_NULL_HANDLE);
        int allocResult = vkAllocateMemory(device, allocInfo, null, pMemory);
        if (allocResult != VK_SUCCESS || pMemory.get(0) == VK_NULL_HANDLE) {
            VK10.vkDestroyImage(device, image, null);
            throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "vkAllocateMemory(image) failed: " + allocResult, false);
        }
        long memory = pMemory.get(0);
        int bindResult = VK10.vkBindImageMemory(device, image, memory, 0);
        if (bindResult != VK_SUCCESS) {
            vkFreeMemory(device, memory, null);
            VK10.vkDestroyImage(device, image, null);
            throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "vkBindImageMemory failed: " + bindResult, false);
        }
        return new VulkanImageAlloc(image, memory);
    }

    /**
     * Convenience overload for image layout transition using one array layer.
     */
    public static void transitionImageLayout(
            VkDevice device,
            long commandPool,
            VkQueue graphicsQueue,
            long image,
            int oldLayout,
            int newLayout,
            BiFunction<String, Integer, GpuException> vkFailure
    ) throws GpuException {
        transitionImageLayout(device, commandPool, graphicsQueue, image, oldLayout, newLayout, 1, 1, vkFailure);
    }

    /**
     * Transitions an image across Vulkan layouts for the specified layer count.
     */
    public static void transitionImageLayout(
            VkDevice device,
            long commandPool,
            VkQueue graphicsQueue,
            long image,
            int oldLayout,
            int newLayout,
            int layerCount,
            int mipLevelCount,
            BiFunction<String, Integer, GpuException> vkFailure
    ) throws GpuException {
        try (MemoryStack stack = stackPush()) {
            VkCommandBuffer cmd = VulkanMemoryOps.beginSingleTimeCommands(device, commandPool, stack, vkFailure);
            VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                    .oldLayout(oldLayout)
                    .newLayout(newLayout)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .image(image);
            barrier.get(0).subresourceRange()
                    .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0)
                    .levelCount(Math.max(1, mipLevelCount))
                    .baseArrayLayer(0)
                    .layerCount(Math.max(1, layerCount));

            int sourceStage;
            int destinationStage;
            if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                barrier.get(0).srcAccessMask(0);
                barrier.get(0).dstAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT);
                sourceStage = VK10.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                destinationStage = VK10.VK_PIPELINE_STAGE_TRANSFER_BIT;
            } else if (oldLayout == VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                    && newLayout == VK10.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                barrier.get(0).srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT);
                barrier.get(0).dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT);
                sourceStage = VK10.VK_PIPELINE_STAGE_TRANSFER_BIT;
                destinationStage = VK10.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            } else {
                throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "Unsupported image layout transition", false);
            }

            VK10.vkCmdPipelineBarrier(
                    cmd,
                    sourceStage,
                    destinationStage,
                    0,
                    null,
                    null,
                    barrier
            );
            VulkanMemoryOps.endSingleTimeCommands(device, commandPool, graphicsQueue, stack, cmd, vkFailure);
        }
    }

    /**
     * Convenience overload to copy a buffer into a one-layer image.
     */
    public static void copyBufferToImage(
            VkDevice device,
            long commandPool,
            VkQueue graphicsQueue,
            long buffer,
            long image,
            int width,
            int height,
            BiFunction<String, Integer, GpuException> vkFailure
    ) throws GpuException {
        copyBufferToImageLayers(device, commandPool, graphicsQueue, buffer, image, width, height, 1, width * height * 4, vkFailure);
    }

    /**
     * Copies a buffer into a Vulkan image across one or more array layers.
     */
    public static void copyBufferToImageLayers(
            VkDevice device,
            long commandPool,
            VkQueue graphicsQueue,
            long buffer,
            long image,
            int width,
            int height,
            int layerCount,
            long bytesPerLayer,
            BiFunction<String, Integer, GpuException> vkFailure
    ) throws GpuException {
        try (MemoryStack stack = stackPush()) {
            VkCommandBuffer cmd = VulkanMemoryOps.beginSingleTimeCommands(device, commandPool, stack, vkFailure);
            int safeLayers = Math.max(1, layerCount);
            long safeLayerBytes = Math.max(1L, bytesPerLayer);
            VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(safeLayers, stack);
            for (int layer = 0; layer < safeLayers; layer++) {
                region.get(layer)
                        .bufferOffset(safeLayerBytes * layer)
                        .bufferRowLength(0)
                        .bufferImageHeight(0);
                region.get(layer).imageSubresource()
                        .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                        .mipLevel(0)
                        .baseArrayLayer(layer)
                        .layerCount(1);
                region.get(layer).imageOffset().set(0, 0, 0);
                region.get(layer).imageExtent().set(width, height, 1);
            }
            VK10.vkCmdCopyBufferToImage(cmd, buffer, image, VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
            VulkanMemoryOps.endSingleTimeCommands(device, commandPool, graphicsQueue, stack, cmd, vkFailure);
        }
    }
}
