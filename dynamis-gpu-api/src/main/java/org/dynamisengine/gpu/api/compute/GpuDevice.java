package org.dynamisengine.gpu.api.compute;
/** Device and child operations are confined to the creating thread. Close drains pending work. */
public interface GpuDevice extends AutoCloseable {
    GpuAdapter adapter();
    default GpuCapabilities capabilities() { return adapter().capabilities(); }
    GpuBuffer allocate(BufferDescriptor descriptor);
    GpuComputePipeline pipeline(PipelineDescriptor descriptor);
    GpuCommandEncoder encoder();
    GpuQueue queue();
    /**
     * The device's buffer memory: blocks held, bytes used, live allocations, and driver allocations so far. Buffers
     * are sub-allocated from pooled blocks, so {@code blockAllocations} grows far slower than the number of buffers.
     * A new buffer's contents are undefined (pooled memory is reused); fill or upload before reading.
     *
     * @since 1.5.0
     */
    default MemoryStats memoryStats() { return MemoryStats.NONE; }
    /**
     * The live buffers and their requested bytes per memory class (the {@link BufferDescriptor#label()}; unlabelled
     * buffers under {@code ""}), kept on the host. @since 1.6.0
     */
    default java.util.Map<String, MemoryUse> memoryByLabel() { return java.util.Map.of(); }
    @Override void close();
}
