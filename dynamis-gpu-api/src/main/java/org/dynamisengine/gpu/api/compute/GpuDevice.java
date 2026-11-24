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
     * The queue of {@code kind}. {@link QueueKind#COMPUTE} is {@link #queue()}. {@link QueueKind#TRANSFER} takes only
     * transfers and fills; {@link GpuQueue#dedicated()} says whether it runs concurrently with the compute queue or
     * shares it. Order work across queues with fences. @since 1.6.0
     */
    default GpuQueue queue(QueueKind kind) { return queue(); }
    /** Native handles for engine interop (see {@link GpuInterop}); UNSUPPORTED on the mock. @since 1.6.0 */
    default GpuInterop interop() { throw new ComputeException(ComputeException.Code.UNSUPPORTED, "interop is not supported here"); }
    /**
     * A transient arena over one backing buffer of {@code descriptor} (its bytes are the capacity; its usages and
     * placement apply to every buffer the arena hands out; its label is the memory class). @since 1.6.0
     */
    default GpuArena arena(BufferDescriptor descriptor) { throw new ComputeException(ComputeException.Code.UNSUPPORTED, "arenas are not supported here"); }
    /** A new timeline fence starting at 0 (see {@link GpuFence}); UNSUPPORTED where the backend has none. @since 1.6.0 */
    default GpuFence fence() { throw new ComputeException(ComputeException.Code.UNSUPPORTED, "fences are not supported here"); }
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
