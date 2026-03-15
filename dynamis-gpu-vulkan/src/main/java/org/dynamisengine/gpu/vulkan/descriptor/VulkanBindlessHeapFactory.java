package org.dynamisengine.gpu.vulkan.descriptor;

import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.IntBuffer;
import java.util.logging.Logger;

import static org.lwjgl.vulkan.EXTDescriptorIndexing.*;
import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.VK11.vkGetPhysicalDeviceFeatures2;
import static org.lwjgl.vulkan.VK11.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

/**
 * Factory responsible for creating {@link VulkanBindlessDescriptorHeap} instances,
 * including descriptor-indexing gate checks and Vulkan resource allocation.
 */
final class VulkanBindlessHeapFactory {

    private static final Logger LOG = Logger.getLogger(VulkanBindlessHeapFactory.class.getName());

    private VulkanBindlessHeapFactory() {
    }

    /**
     * Creates an active bindless descriptor heap when descriptor indexing is supported.
     */
    static VulkanBindlessDescriptorHeap create(
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            boolean requestedEnabled,
            int framesInFlight
    ) throws GpuException {
        if (!requestedEnabled || device == null || physicalDevice == null) {
            return disabled();
        }
        GateResult gate = checkDescriptorIndexingGate(physicalDevice);
        if (!gate.passed()) {
            LOG.warning("BINDLESS_DESCRIPTOR_INDEXING_UNAVAILABLE " + gate.reason());
            return disabled();
        }

        try (MemoryStack stack = MemoryStack.stackPush()) {
            long layout = createDescriptorSetLayout(device, stack);
            long pool = createDescriptorPool(device, stack);
            long set = allocateDescriptorSet(device, stack, pool, layout);
            return VulkanBindlessDescriptorHeap.createActive(
                    device,
                    layout,
                    pool,
                    set,
                    framesInFlight
            );
        }
    }

    /**
     * Returns a disabled no-op heap implementation.
     */
    static VulkanBindlessDescriptorHeap disabled() {
        return VulkanBindlessDescriptorHeap.createDisabled();
    }

    static GateResult checkDescriptorIndexingGate(VkPhysicalDevice physicalDevice) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexing = VkPhysicalDeviceDescriptorIndexingFeaturesEXT.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT);
            VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
                    .pNext(indexing.address());
            vkGetPhysicalDeviceFeatures2(physicalDevice, features2);

            VkPhysicalDeviceProperties props = VkPhysicalDeviceProperties.calloc(stack);
            vkGetPhysicalDeviceProperties(physicalDevice, props);

            StringBuilder reason = new StringBuilder();
            boolean ok = true;
            ok &= checkFeature(indexing.runtimeDescriptorArray(), "runtimeDescriptorArray", reason);
            ok &= checkFeature(indexing.descriptorBindingPartiallyBound(), "descriptorBindingPartiallyBound", reason);
            ok &= checkFeature(indexing.descriptorBindingVariableDescriptorCount(), "descriptorBindingVariableDescriptorCount", reason);
            ok &= checkFeature(indexing.descriptorBindingStorageBufferUpdateAfterBind(), "descriptorBindingStorageBufferUpdateAfterBind", reason);
            ok &= checkFeature(indexing.shaderStorageBufferArrayNonUniformIndexing(), "shaderStorageBufferArrayNonUniformIndexing", reason);
            ok &= checkFeature(indexing.shaderUniformBufferArrayNonUniformIndexing(), "shaderUniformBufferArrayNonUniformIndexing", reason);

            VkPhysicalDeviceLimits limits = props.limits();
            ok &= checkLimit(limits.maxBoundDescriptorSets(), 4, "maxBoundDescriptorSets", reason);
            ok &= checkLimit(limits.maxPerStageDescriptorStorageBuffers(), 1024, "maxPerStageDescriptorStorageBuffers", reason);
            ok &= checkLimit(limits.maxDescriptorSetStorageBuffers(), 2048, "maxDescriptorSetStorageBuffers", reason);
            ok &= checkLimit(limits.maxPerStageDescriptorUniformBuffers(), 512, "maxPerStageDescriptorUniformBuffers", reason);
            ok &= checkLimit(limits.maxDescriptorSetUniformBuffers(), 512, "maxDescriptorSetUniformBuffers", reason);

            return new GateResult(ok, reason.toString());
        }
    }

    static boolean checkFeature(boolean enabled, String name, StringBuilder reason) {
        if (enabled) {
            return true;
        }
        appendReason(reason, "missingFeature=" + name);
        return false;
    }

    static boolean checkLimit(int actual, int required, String name, StringBuilder reason) {
        if (actual >= required) {
            return true;
        }
        appendReason(reason, "limitShortfall=" + name + "(" + actual + "<" + required + ")");
        return false;
    }

    static void appendReason(StringBuilder reason, String part) {
        if (reason.isEmpty()) {
            reason.append(part);
        } else {
            reason.append(", ").append(part);
        }
    }

    static long createDescriptorSetLayout(VkDevice device, MemoryStack stack) throws GpuException {
        VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(5, stack);
        bindings.get(0)
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(VulkanBindlessDescriptorHeap.JOINT_CAPACITY)
                .stageFlags(VK_SHADER_STAGE_VERTEX_BIT);
        bindings.get(1)
                .binding(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(VulkanBindlessDescriptorHeap.MORPH_DELTA_CAPACITY)
                .stageFlags(VK_SHADER_STAGE_VERTEX_BIT);
        bindings.get(2)
                .binding(2)
                .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(VulkanBindlessDescriptorHeap.MORPH_WEIGHT_CAPACITY)
                .stageFlags(VK_SHADER_STAGE_VERTEX_BIT);
        bindings.get(3)
                .binding(3)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(VulkanBindlessDescriptorHeap.INSTANCE_CAPACITY)
                .stageFlags(VK_SHADER_STAGE_VERTEX_BIT);
        bindings.get(4)
                .binding(4)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(VulkanBindlessDescriptorHeap.DRAW_META_CAPACITY)
                .stageFlags(VK_SHADER_STAGE_VERTEX_BIT);

        IntBuffer bindingFlags = stack.mallocInt(5);
        for (int i = 0; i < 5; i++) {
            bindingFlags.put(i,
                    VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT
                            | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT
            );
        }

        VkDescriptorSetLayoutBindingFlagsCreateInfoEXT flagsInfo = VkDescriptorSetLayoutBindingFlagsCreateInfoEXT.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT)
                .pBindingFlags(bindingFlags);

        VkDescriptorSetLayoutCreateInfo info = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                .flags(VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT)
                .pBindings(bindings)
                .pNext(flagsInfo.address());

        var pLayout = stack.longs(VK_NULL_HANDLE);
        int result = vkCreateDescriptorSetLayout(device, info, null, pLayout);
        if (result != VK_SUCCESS || pLayout.get(0) == VK_NULL_HANDLE) {
            throw new GpuException(
                    GpuErrorCode.BACKEND_INIT_FAILED,
                    "vkCreateDescriptorSetLayout(bindless) failed: " + result,
                    false
            );
        }
        return pLayout.get(0);
    }

    static long createDescriptorPool(VkDevice device, MemoryStack stack) throws GpuException {
        VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(2, stack);
        sizes.get(0)
                .type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(VulkanBindlessDescriptorHeap.JOINT_CAPACITY
                        + VulkanBindlessDescriptorHeap.MORPH_DELTA_CAPACITY
                        + VulkanBindlessDescriptorHeap.INSTANCE_CAPACITY
                        + VulkanBindlessDescriptorHeap.DRAW_META_CAPACITY);
        sizes.get(1)
                .type(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(VulkanBindlessDescriptorHeap.MORPH_WEIGHT_CAPACITY);

        VkDescriptorPoolCreateInfo info = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                .flags(VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT)
                .maxSets(1)
                .pPoolSizes(sizes);
        var pPool = stack.longs(VK_NULL_HANDLE);
        int result = vkCreateDescriptorPool(device, info, null, pPool);
        if (result != VK_SUCCESS || pPool.get(0) == VK_NULL_HANDLE) {
            throw new GpuException(
                    GpuErrorCode.BACKEND_INIT_FAILED,
                    "vkCreateDescriptorPool(bindless) failed: " + result,
                    false
            );
        }
        return pPool.get(0);
    }

    static long allocateDescriptorSet(VkDevice device, MemoryStack stack, long descriptorPool, long descriptorSetLayout)
            throws GpuException {
        VkDescriptorSetAllocateInfo info = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                .descriptorPool(descriptorPool)
                .pSetLayouts(stack.longs(descriptorSetLayout));
        var pSet = stack.longs(VK_NULL_HANDLE);
        int result = vkAllocateDescriptorSets(device, info, pSet);
        if (result != VK_SUCCESS || pSet.get(0) == VK_NULL_HANDLE) {
            throw new GpuException(
                    GpuErrorCode.BACKEND_INIT_FAILED,
                    "vkAllocateDescriptorSets(bindless) failed: " + result,
                    false
            );
        }
        return pSet.get(0);
    }

    record GateResult(boolean passed, String reason) {
    }
}
