package org.dynamisengine.gpu.vulkan.memory;

import java.util.function.BiFunction;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkSubmitInfo;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_SUBMIT_INFO;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkBeginCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkEndCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkFreeCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceMemoryProperties;
import static org.lwjgl.vulkan.VK10.vkQueueSubmit;
import static org.lwjgl.vulkan.VK10.vkQueueWaitIdle;

/**
 * Foundation memory utilities shared by {@link VulkanBufferOps} and {@link VulkanImageOps}.
 */
public final class VulkanMemoryOps {
    private VulkanMemoryOps() {
    }

    static int findMemoryType(VkPhysicalDevice physicalDevice, int typeFilter, int properties) throws GpuException {
        try (MemoryStack stack = stackPush()) {
            VkPhysicalDeviceMemoryProperties memoryProperties = VkPhysicalDeviceMemoryProperties.calloc(stack);
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, memoryProperties);
            for (int i = 0; i < memoryProperties.memoryTypeCount(); i++) {
                boolean typeMatch = (typeFilter & (1 << i)) != 0;
                boolean propsMatch = (memoryProperties.memoryTypes(i).propertyFlags() & properties) == properties;
                if (typeMatch && propsMatch) {
                    return i;
                }
            }
        }
        throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "No suitable Vulkan memory type found", false);
    }

    static VkCommandBuffer beginSingleTimeCommands(
            VkDevice device,
            long commandPool,
            MemoryStack stack,
            BiFunction<String, Integer, GpuException> vkFailure
    ) throws GpuException {
        VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                .commandPool(commandPool)
                .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                .commandBufferCount(1);
        PointerBuffer pCommandBuffer = stack.mallocPointer(1);
        int allocResult = vkAllocateCommandBuffers(device, allocInfo, pCommandBuffer);
        if (allocResult != VK_SUCCESS) {
            throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "vkAllocateCommandBuffers(one-shot) failed: " + allocResult, false);
        }
        VkCommandBuffer cmd = new VkCommandBuffer(pCommandBuffer.get(0), device);
        VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        int beginResult = vkBeginCommandBuffer(cmd, beginInfo);
        if (beginResult != VK_SUCCESS) {
            throw vkFailure.apply("vkBeginCommandBuffer(one-shot)", beginResult);
        }
        return cmd;
    }

    static void endSingleTimeCommands(
            VkDevice device,
            long commandPool,
            VkQueue graphicsQueue,
            MemoryStack stack,
            VkCommandBuffer cmd,
            BiFunction<String, Integer, GpuException> vkFailure
    ) throws GpuException {
        int endResult = vkEndCommandBuffer(cmd);
        if (endResult != VK_SUCCESS) {
            throw vkFailure.apply("vkEndCommandBuffer(one-shot)", endResult);
        }
        VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                .pCommandBuffers(stack.pointers(cmd.address()));
        int submitResult = vkQueueSubmit(graphicsQueue, submitInfo, VK_NULL_HANDLE);
        if (submitResult != VK_SUCCESS) {
            throw vkFailure.apply("vkQueueSubmit(one-shot)", submitResult);
        }
        int waitResult = vkQueueWaitIdle(graphicsQueue);
        if (waitResult != VK_SUCCESS) {
            throw vkFailure.apply("vkQueueWaitIdle(one-shot)", waitResult);
        }
        vkFreeCommandBuffers(device, commandPool, stack.pointers(cmd.address()));
    }
}
