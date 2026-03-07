package org.dynamisengine.gpu.vulkan.buffer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.function.BiFunction;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanMemoryOps;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.vkDestroyBuffer;
import static org.lwjgl.vulkan.VK10.vkFreeMemory;

/**
 * Immutable Vulkan morph-target delta buffer allocation wrapper.
 */
public final class VulkanMorphTargetBuffer {
    private final long buffer;
    private final long memory;
    private final int vertexCount;
    private final int targetCount;
    private final int bytes;

    private VulkanMorphTargetBuffer(long buffer, long memory, int vertexCount, int targetCount, int bytes) {
        this.buffer = buffer;
        this.memory = memory;
        this.vertexCount = vertexCount;
        this.targetCount = targetCount;
        this.bytes = bytes;
    }

    /**
     * Creates a device-local morph target buffer and uploads packed delta data through staging.
     *
     * @return created morph target buffer, or {@code null} when input payload is empty
     */
    public static VulkanMorphTargetBuffer create(
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            long commandPool,
            VkQueue graphicsQueue,
            MemoryStack stack,
            float[] packedDeltas,
            int vertexCount,
            int targetCount,
            BiFunction<String, Integer, GpuException> vkFailure
    ) throws GpuException {
        if (packedDeltas == null || packedDeltas.length == 0 || vertexCount <= 0 || targetCount <= 0) {
            return null;
        }
        int expected = vertexCount * targetCount * 6;
        if (packedDeltas.length != expected) {
            throw new GpuException(
                    GpuErrorCode.INVALID_ARGUMENT,
                    "Morph target delta payload length mismatch: expected " + expected + " floats, got " + packedDeltas.length,
                    true
            );
        }
        ByteBuffer source = ByteBuffer.allocateDirect(packedDeltas.length * Float.BYTES).order(ByteOrder.nativeOrder());
        FloatBuffer fb = source.asFloatBuffer();
        fb.put(packedDeltas);
        source.limit(packedDeltas.length * Float.BYTES);
        VulkanBufferAlloc alloc = VulkanMemoryOps.createDeviceLocalBufferWithStaging(
                device,
                physicalDevice,
                commandPool,
                graphicsQueue,
                stack,
                source,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                vkFailure
        );
        return new VulkanMorphTargetBuffer(
                alloc.buffer(),
                alloc.memory(),
                vertexCount,
                targetCount,
                source.remaining()
        );
    }

    /**
     * Destroys Vulkan objects owned by this buffer wrapper.
     *
     * @param device Vulkan device used for destruction calls
     */
    public void destroy(VkDevice device) {
        if (device == null) {
            return;
        }
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, null);
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, null);
        }
    }

    /**
     * @return Vulkan storage buffer handle
     */
    public long bufferHandle() {
        return buffer;
    }

    /**
     * @return vertex count encoded in this payload
     */
    public int vertexCount() {
        return vertexCount;
    }

    /**
     * @return morph target count encoded in this payload
     */
    public int targetCount() {
        return targetCount;
    }

    /**
     * @return payload size in bytes
     */
    public int bytes() {
        return bytes;
    }
}
