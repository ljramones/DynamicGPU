package org.dynamisengine.gpu.api.compute;

/**
 * Native handles for engine code that drives the same GPU directly, such as LightEngine's renderer: the device,
 * buffers and fences, and (on Vulkan) recording into a command buffer the caller owns.
 * <p>
 * <b>The caller's obligations.</b> DynamisGPU does not see work the caller submits itself.
 * <ul>
 * <li>Keep a buffer open until the caller's own GPU work on it has completed. Closing it returns its range to a
 * pool, where other buffers reuse it.</li>
 * <li>Order the caller's work against DynamisGPU's with fences ({@link #nativeFence}).</li>
 * <li>Submit to the exported queue from the device's owner thread, or synchronize with it: queue submission is not
 * thread-safe.</li>
 * </ul>
 * A pooled buffer is a range of a shared block, so its handle comes with an offset; use only {@code [offset, offset +
 * size)}.
 *
 * @since 1.6.0
 */
public interface GpuInterop {
    /**
     * The device's native handles.
     *
     * @param device VkDevice or {@code id<MTLDevice>}
     * @param physicalDevice VkPhysicalDevice (0 on Metal)
     * @param instance VkInstance (0 on Metal)
     * @param queue the compute queue: VkQueue or {@code id<MTLCommandQueue>}
     * @param queueFamily the compute queue family (-1 on Metal)
     */
    record NativeDevice(Backend backend, long device, long physicalDevice, long instance, long queue, int queueFamily) { }
    /** A buffer: its native buffer (VkBuffer or {@code id<MTLBuffer>}) and the range of it the GpuBuffer covers. */
    record NativeBuffer(Backend backend, long handle, long offset, long size) { }

    NativeDevice nativeDevice();
    NativeBuffer nativeBuffer(GpuBuffer buffer);
    /** A fence's VkSemaphore (timeline) or {@code id<MTLSharedEvent>}. */
    long nativeFence(GpuFence fence);
    /**
     * Records {@code encoder}'s commands into the caller's command buffer (Vulkan: a VkCommandBuffer, being recorded,
     * from a pool of the exported device and queue family), consuming the encoder. The encoder may hold no downloads,
     * error checks or fence operations. The result holds every buffer and pipeline it uses, and its staging. Close it
     * once the caller's command buffer has completed. UNSUPPORTED on Metal.
     */
    AutoCloseable recordInto(GpuCommandEncoder encoder, long commandBuffer);
}
