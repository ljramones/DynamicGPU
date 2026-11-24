package org.dynamisengine.gpu.api.compute.spi;

import org.dynamisengine.gpu.api.compute.*;
import java.nio.ByteBuffer;
import java.util.List;

/** Backend extension SPI. Opaque native objects never enter the public compute contracts. */
public interface ComputeDriver extends AutoCloseable {
    record Region(Object allocation, long offset, long bytes) { }
    record Bound(int slot, Region region, Access access) { }
    sealed interface Command permits Upload, Download, Copy, Fill, Dispatch { }
    record Upload(Region destination, byte[] data) implements Command { }
    record Download(Region source, GpuReadback result) implements Command { }
    record Copy(Region source, Region destination) implements Command { }
    record Fill(Region destination, int pattern) implements Command { }
    record Dispatch(Object pipeline, PipelineDescriptor descriptor, List<Bound> bindings,
                    byte[] parameters, Dimensions groups) implements Command { }
    interface Job extends AutoCloseable {
        /** COMPLETE includes host-read visibility and publication of every readback. */
        GpuSubmission.Status poll();
        /**
         * Blocks until the work completes or {@code nanos} pass, without publishing results ({@link #poll} does);
         * true if it completed. The default sleeps briefly and reports false, so callers fall back to polling.
         */
        default boolean await(long nanos) { java.util.concurrent.locks.LockSupport.parkNanos(Math.min(nanos, 100_000)); return false; }
        /** The GPU execution time in nanoseconds, after {@link #poll} returned COMPLETE and before {@link #close}; empty if not measured. */
        default java.util.OptionalLong gpuNanos() { return java.util.OptionalLong.empty(); }
        @Override void close();
    }
    Object allocate(BufferDescriptor descriptor);
    void releaseBuffer(Object buffer);
    ByteBuffer map(Object buffer, long offset, long bytes, Access access);
    void unmap(Object buffer, Access access);
    Object pipeline(PipelineDescriptor descriptor);
    void releasePipeline(Object pipeline);
    Job submit(List<Command> commands);
    /** Commands a driver has prepared for repeated submission (a recording); closed when the recording is. */
    interface Prepared extends AutoCloseable { @Override void close(); }
    /**
     * Prepares {@code commands} (no downloads) for repeated submission. The default keeps nothing, and
     * {@link #submit(Prepared, List)} then submits the list afresh each time.
     */
    default Prepared prepare(List<Command> commands) { return () -> { }; }
    /**
     * Submits a prepared list again. {@code commands} has the prepared list's shape, with the current parameter blocks
     * and bindings; the previous submission of {@code prepared} has completed.
     */
    default Job submit(Prepared prepared, List<Command> commands) { return submit(commands); }
    /** The driver's buffer pools (see {@link MemoryPool}); {@link MemoryStats#NONE} if it has none. */
    default MemoryStats memoryStats() { return MemoryStats.NONE; }
    @Override void close();
}
