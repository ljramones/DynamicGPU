package org.dynamisengine.gpu.vulkan.sync;

import org.dynamisengine.gpu.api.error.GpuException;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDevice;

import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;

/**
 * Coordinates creation of frame-sync state objects used by the Vulkan renderer lifecycle.
 */
public final class VulkanFrameSyncLifecycleCoordinator {
    private VulkanFrameSyncLifecycleCoordinator() {
    }

    /**
     * Creates initialized frame-sync state from a request.
     *
     * @param request frame-sync creation inputs
     * @return initialized frame-sync state
     */
    public static State create(CreateRequest request) throws GpuException {
        VulkanFrameSyncResources.Allocation frameSyncResources = VulkanFrameSyncResources.create(
                request.device(),
                request.stack(),
                request.graphicsQueueFamilyIndex(),
                request.framesInFlight()
        );
        return new State(
                frameSyncResources.commandPool(),
                frameSyncResources.commandBuffers(),
                frameSyncResources.imageAvailableSemaphores(),
                frameSyncResources.renderFinishedSemaphores(),
                frameSyncResources.renderFences(),
                0
        );
    }

    /**
     * Returns an empty state object with null handles.
     *
     * @return empty state
     */
    public static State empty() {
        return new State(
                VK_NULL_HANDLE,
                new VkCommandBuffer[0],
                new long[0],
                new long[0],
                new long[0],
                0
        );
    }

    /**
     * Immutable request object for state creation.
     */
    public record CreateRequest(
            VkDevice device,
            MemoryStack stack,
            int graphicsQueueFamilyIndex,
            int framesInFlight
    ) {
    }

    /**
     * Immutable snapshot of frame-sync runtime state.
     */
    public record State(
            long commandPool,
            VkCommandBuffer[] commandBuffers,
            long[] imageAvailableSemaphores,
            long[] renderFinishedSemaphores,
            long[] renderFences,
            int currentFrame
    ) {
    }
}
