package org.dynamisengine.gpu.vulkan.memory;

import java.nio.ByteBuffer;
import java.util.function.BiFunction;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkBufferCopy;
import org.lwjgl.vulkan.VkBufferCreateInfo;
import org.lwjgl.vulkan.VkBufferDeviceAddressInfo;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkMemoryAllocateInfo;
import org.lwjgl.vulkan.VkMemoryAllocateFlagsInfo;
import org.lwjgl.vulkan.VkMemoryRequirements;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkSubmitInfo;
import org.lwjgl.vulkan.VK12;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.memAddress;
import static org.lwjgl.system.MemoryUtil.memCopy;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_SHARING_MODE_EXCLUSIVE;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_SUBMIT_INFO;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkAllocateMemory;
import static org.lwjgl.vulkan.VK10.vkBeginCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkBindBufferMemory;
import static org.lwjgl.vulkan.VK10.vkCmdCopyBuffer;
import static org.lwjgl.vulkan.VK10.vkCreateBuffer;
import static org.lwjgl.vulkan.VK10.vkDestroyBuffer;
import static org.lwjgl.vulkan.VK10.vkEndCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkFreeCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkFreeMemory;
import static org.lwjgl.vulkan.VK10.vkGetBufferMemoryRequirements;
import static org.lwjgl.vulkan.VK10.vkMapMemory;
import static org.lwjgl.vulkan.VK10.vkQueueSubmit;
import static org.lwjgl.vulkan.VK10.vkQueueWaitIdle;
import static org.lwjgl.vulkan.VK10.vkUnmapMemory;
import static org.lwjgl.vulkan.VK11.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;

/**
 * Static Vulkan helper methods for buffer creation and transfer operations.
 */
public final class VulkanBufferOps {
    private VulkanBufferOps() {
    }

    /**
     * Creates a Vulkan buffer and backing memory allocation, then binds them together.
     */
    public static VulkanBufferAlloc createBuffer(
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            MemoryStack stack,
            int sizeBytes,
            int usage,
            int memoryProperties
    ) throws GpuException {
        return createBuffer(device, physicalDevice, stack, sizeBytes, usage, memoryProperties, 0);
    }

    /**
     * Creates a Vulkan buffer and backing memory allocation, then binds them together.
     *
     * <p>Use allocation flags for features such as buffer device address support.
     */
    public static VulkanBufferAlloc createBuffer(
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            MemoryStack stack,
            int sizeBytes,
            int usage,
            int memoryProperties,
            int memoryAllocateFlags
    ) throws GpuException {
        VkBufferCreateInfo bufferInfo = VkBufferCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                .size(sizeBytes)
                .usage(usage)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
        var pBuffer = stack.longs(VK_NULL_HANDLE);
        int createBufferResult = vkCreateBuffer(device, bufferInfo, null, pBuffer);
        if (createBufferResult != VK_SUCCESS || pBuffer.get(0) == VK_NULL_HANDLE) {
            throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "vkCreateBuffer failed: " + createBufferResult, false);
        }
        long buffer = pBuffer.get(0);

        VkMemoryRequirements memReq = VkMemoryRequirements.calloc(stack);
        vkGetBufferMemoryRequirements(device, buffer, memReq);

        VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(memReq.size())
                .memoryTypeIndex(VulkanMemoryOps.findMemoryType(physicalDevice, memReq.memoryTypeBits(), memoryProperties));
        if (memoryAllocateFlags != 0) {
            VkMemoryAllocateFlagsInfo flagsInfo = VkMemoryAllocateFlagsInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO)
                    .flags(memoryAllocateFlags);
            allocInfo.pNext(flagsInfo.address());
        }

        var pMemory = stack.longs(VK_NULL_HANDLE);
        int allocResult = vkAllocateMemory(device, allocInfo, null, pMemory);
        if (allocResult != VK_SUCCESS || pMemory.get(0) == VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, null);
            throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "vkAllocateMemory failed: " + allocResult, false);
        }
        long memory = pMemory.get(0);
        int bindResult = vkBindBufferMemory(device, buffer, memory, 0);
        if (bindResult != VK_SUCCESS) {
            vkFreeMemory(device, memory, null);
            vkDestroyBuffer(device, buffer, null);
            throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "vkBindBufferMemory failed: " + bindResult, false);
        }
        return new VulkanBufferAlloc(buffer, memory);
    }

    /**
     * Uploads source bytes through a staging buffer into a device-local buffer.
     */
    public static VulkanBufferAlloc createDeviceLocalBufferWithStaging(
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            long commandPool,
            VkQueue graphicsQueue,
            MemoryStack stack,
            ByteBuffer source,
            int usage,
            BiFunction<String, Integer, GpuException> vkFailure
    ) throws GpuException {
        int sizeBytes = source.remaining();
        VulkanBufferAlloc staging = createBuffer(
                device,
                physicalDevice,
                stack,
                sizeBytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        VulkanBufferAlloc deviceLocal = createBuffer(
                device,
                physicalDevice,
                stack,
                sizeBytes,
                usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        try {
            uploadToMemory(device, staging.memory(), source, vkFailure);
            copyBuffer(device, commandPool, graphicsQueue, staging.buffer(), deviceLocal.buffer(), sizeBytes, vkFailure);
            return deviceLocal;
        } finally {
            if (staging.buffer() != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, staging.buffer(), null);
            }
            if (staging.memory() != VK_NULL_HANDLE) {
                vkFreeMemory(device, staging.memory(), null);
            }
        }
    }

    /**
     * Uploads source bytes through a staging buffer into a device-local buffer with
     * shader-device-address-capable memory allocation.
     */
    public static VulkanBufferAlloc createDeviceAddressBufferWithStaging(
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            long commandPool,
            VkQueue graphicsQueue,
            MemoryStack stack,
            ByteBuffer source,
            int usage,
            BiFunction<String, Integer, GpuException> vkFailure
    ) throws GpuException {
        int sizeBytes = source.remaining();
        VulkanBufferAlloc staging = createBuffer(
                device,
                physicalDevice,
                stack,
                sizeBytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        VulkanBufferAlloc deviceLocal = createBuffer(
                device,
                physicalDevice,
                stack,
                sizeBytes,
                usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK12.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK12.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
        );
        try {
            uploadToMemory(device, staging.memory(), source, vkFailure);
            copyBuffer(device, commandPool, graphicsQueue, staging.buffer(), deviceLocal.buffer(), sizeBytes, vkFailure);
            return deviceLocal;
        } finally {
            if (staging.buffer() != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, staging.buffer(), null);
            }
            if (staging.memory() != VK_NULL_HANDLE) {
                vkFreeMemory(device, staging.memory(), null);
            }
        }
    }

    /**
     * Queries the Vulkan device address for a buffer created with device-address-capable usage and
     * allocation flags.
     */
    public static long getBufferDeviceAddress(VkDevice device, long bufferHandle) {
        if (device == null) {
            throw new NullPointerException("device");
        }
        if (bufferHandle == VK_NULL_HANDLE) {
            throw new IllegalArgumentException("bufferHandle must not be VK_NULL_HANDLE");
        }
        try (MemoryStack stack = stackPush()) {
            VkBufferDeviceAddressInfo info = VkBufferDeviceAddressInfo.calloc(stack)
                    .sType(VK12.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO)
                    .buffer(bufferHandle);
            return VK12.vkGetBufferDeviceAddress(device, info);
        }
    }

    /**
     * Records and submits a one-shot buffer-to-buffer copy command.
     */
    public static void copyBuffer(
            VkDevice device,
            long commandPool,
            VkQueue graphicsQueue,
            long srcBuffer,
            long dstBuffer,
            int sizeBytes,
            BiFunction<String, Integer, GpuException> vkFailure
    ) throws GpuException {
        try (MemoryStack stack = stackPush()) {
            VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                    .commandPool(commandPool)
                    .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                    .commandBufferCount(1);
            PointerBuffer pCommandBuffer = stack.mallocPointer(1);
            int allocResult = vkAllocateCommandBuffers(device, allocInfo, pCommandBuffer);
            if (allocResult != VK_SUCCESS) {
                throw new GpuException(GpuErrorCode.BACKEND_INIT_FAILED, "vkAllocateCommandBuffers(copy) failed: " + allocResult, false);
            }
            VkCommandBuffer cmd = new VkCommandBuffer(pCommandBuffer.get(0), device);

            VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            int beginResult = vkBeginCommandBuffer(cmd, beginInfo);
            if (beginResult != VK_SUCCESS) {
                throw vkFailure.apply("vkBeginCommandBuffer(copy)", beginResult);
            }

            VkBufferCopy.Buffer copyRegion = VkBufferCopy.calloc(1, stack);
            copyRegion.get(0).srcOffset(0).dstOffset(0).size(sizeBytes);
            vkCmdCopyBuffer(cmd, srcBuffer, dstBuffer, copyRegion);

            int endResult = vkEndCommandBuffer(cmd);
            if (endResult != VK_SUCCESS) {
                throw vkFailure.apply("vkEndCommandBuffer(copy)", endResult);
            }

            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(stack.pointers(cmd.address()));
            int submitResult = vkQueueSubmit(graphicsQueue, submitInfo, VK_NULL_HANDLE);
            if (submitResult != VK_SUCCESS) {
                throw vkFailure.apply("vkQueueSubmit(copy)", submitResult);
            }
            int waitResult = vkQueueWaitIdle(graphicsQueue);
            if (waitResult != VK_SUCCESS) {
                throw vkFailure.apply("vkQueueWaitIdle(copy)", waitResult);
            }
            vkFreeCommandBuffers(device, commandPool, stack.pointers(cmd.address()));
        }
    }

    /**
     * Maps device memory and copies the given source bytes into it.
     */
    public static void uploadToMemory(
            VkDevice device,
            long memory,
            ByteBuffer source,
            BiFunction<String, Integer, GpuException> vkFailure
    ) throws GpuException {
        try (MemoryStack stack = stackPush()) {
            PointerBuffer pData = stack.mallocPointer(1);
            int mapResult = vkMapMemory(device, memory, 0, source.remaining(), 0, pData);
            if (mapResult != VK_SUCCESS) {
                throw vkFailure.apply("vkMapMemory", mapResult);
            }
            memCopy(memAddress(source), pData.get(0), source.remaining());
            vkUnmapMemory(device, memory);
        }
    }
}
