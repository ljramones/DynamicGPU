package org.dynamisengine.gpu.vulkan.descriptor;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import static org.lwjgl.vulkan.VK10.*;

/**
 * Package-private utility that writes descriptor updates into a bindless heap's descriptor set.
 */
final class VulkanDescriptorUpdater {

    private VulkanDescriptorUpdater() {
    }

    /**
     * Updates descriptor payload for a joint palette handle.
     */
    static boolean updateJointPaletteDescriptor(
            VulkanBindlessDescriptorHeap heap,
            long handle,
            long currentFrame,
            long bufferHandle,
            long rangeBytes
    ) {
        if (!heap.active() || handle == 0L || bufferHandle == VK_NULL_HANDLE || rangeBytes <= 0L) {
            return false;
        }
        int slot = heap.resolveSlot(handle, currentFrame);
        if (slot < 0) {
            return false;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorBufferInfo.Buffer bufferInfo = VkDescriptorBufferInfo.calloc(1, stack);
            bufferInfo.get(0)
                    .buffer(bufferHandle)
                    .offset(0L)
                    .range(rangeBytes);
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack);
            write.get(0)
                    .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                    .dstSet(heap.descriptorSet())
                    .dstBinding(0)
                    .dstArrayElement(slot)
                    .descriptorCount(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .pBufferInfo(bufferInfo);
            vkUpdateDescriptorSets(heap.device(), write, null);
            return true;
        }
    }

    /**
     * Updates descriptor payload for a morph delta handle.
     */
    static boolean updateMorphDeltaDescriptor(
            VulkanBindlessDescriptorHeap heap,
            long handle,
            long currentFrame,
            long bufferHandle,
            long rangeBytes
    ) {
        return updateStorageBufferDescriptor(heap, handle, currentFrame, bufferHandle, rangeBytes, 1);
    }

    /**
     * Updates descriptor payload for a morph weight handle.
     */
    static boolean updateMorphWeightDescriptor(
            VulkanBindlessDescriptorHeap heap,
            long handle,
            long currentFrame,
            long bufferHandle,
            long rangeBytes
    ) {
        return updateUniformBufferDescriptor(heap, handle, currentFrame, bufferHandle, rangeBytes, 2);
    }

    /**
     * Updates descriptor payload for an instance-data handle.
     */
    static boolean updateInstanceDataDescriptor(
            VulkanBindlessDescriptorHeap heap,
            long handle,
            long currentFrame,
            long bufferHandle,
            long rangeBytes
    ) {
        return updateStorageBufferDescriptor(heap, handle, currentFrame, bufferHandle, rangeBytes, 3);
    }

    static boolean updateStorageBufferDescriptor(
            VulkanBindlessDescriptorHeap heap,
            long handle,
            long currentFrame,
            long bufferHandle,
            long rangeBytes,
            int binding
    ) {
        if (!heap.active() || handle == 0L || bufferHandle == VK_NULL_HANDLE || rangeBytes <= 0L) {
            return false;
        }
        int slot = heap.resolveSlot(handle, currentFrame);
        if (slot < 0) {
            return false;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorBufferInfo.Buffer bufferInfo = VkDescriptorBufferInfo.calloc(1, stack);
            bufferInfo.get(0)
                    .buffer(bufferHandle)
                    .offset(0L)
                    .range(rangeBytes);
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack);
            write.get(0)
                    .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                    .dstSet(heap.descriptorSet())
                    .dstBinding(binding)
                    .dstArrayElement(slot)
                    .descriptorCount(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .pBufferInfo(bufferInfo);
            vkUpdateDescriptorSets(heap.device(), write, null);
            return true;
        }
    }

    static boolean updateUniformBufferDescriptor(
            VulkanBindlessDescriptorHeap heap,
            long handle,
            long currentFrame,
            long bufferHandle,
            long rangeBytes,
            int binding
    ) {
        if (!heap.active() || handle == 0L || bufferHandle == VK_NULL_HANDLE || rangeBytes <= 0L) {
            return false;
        }
        int slot = heap.resolveSlot(handle, currentFrame);
        if (slot < 0) {
            return false;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorBufferInfo.Buffer bufferInfo = VkDescriptorBufferInfo.calloc(1, stack);
            bufferInfo.get(0)
                    .buffer(bufferHandle)
                    .offset(0L)
                    .range(rangeBytes);
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack);
            write.get(0)
                    .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                    .dstSet(heap.descriptorSet())
                    .dstBinding(binding)
                    .dstArrayElement(slot)
                    .descriptorCount(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                    .pBufferInfo(bufferInfo);
            vkUpdateDescriptorSets(heap.device(), write, null);
            return true;
        }
    }
}
