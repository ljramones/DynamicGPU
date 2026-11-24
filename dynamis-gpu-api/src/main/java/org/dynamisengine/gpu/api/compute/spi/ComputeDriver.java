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
    /**
     * A dispatch; {@code label} is its effective label (empty for none), for backend debug labels. With
     * {@code indirect} (12 bytes, 4-aligned), the group counts are read from that region and {@code groups} is unused.
     */
    record Dispatch(Object pipeline, PipelineDescriptor descriptor, List<Bound> bindings,
                    byte[] parameters, Dimensions groups, String label, Region indirect) implements Command {
        public Dispatch(Object pipeline, PipelineDescriptor descriptor, List<Bound> bindings, byte[] parameters, Dimensions groups, String label) {
            this(pipeline, descriptor, bindings, parameters, groups, label, null);
        }
        public Dispatch(Object pipeline, PipelineDescriptor descriptor, List<Bound> bindings, byte[] parameters, Dimensions groups) {
            this(pipeline, descriptor, bindings, parameters, groups, "", null);
        }
        public Dispatch withParameters(byte[] p) { return new Dispatch(pipeline, descriptor, bindings, p, groups, label, indirect); }
        public Dispatch withGroups(Dimensions g) { return new Dispatch(pipeline, descriptor, bindings, parameters, g, label, indirect); }
        public Dispatch withBindings(List<Bound> b) { return new Dispatch(pipeline, descriptor, b, parameters, groups, label, indirect); }
    }
    interface Job extends AutoCloseable {
        /** COMPLETE includes host-read visibility and publication of every readback. */
        GpuSubmission.Status poll();
        /**
         * Blocks until the work completes or {@code nanos} pass, without publishing results ({@link #poll} does);
         * true if it completed. The default sleeps briefly and reports false, so callers fall back to polling.
         */
        default boolean await(long nanos) { java.util.concurrent.locks.LockSupport.parkNanos(Math.min(nanos, 100_000)); return false; }
        /** Per-dispatch GPU nanoseconds after {@link #poll} returned COMPLETE, for profiled work; null otherwise. */
        default long[] dispatchNanos() { return null; }
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
    /** Interop: the device's native handles. */
    default GpuInterop.NativeDevice nativeDevice() { throw new ComputeException(ComputeException.Code.UNSUPPORTED, "no interop on this backend"); }
    /** Interop: the native buffer and range of {@code bytes} at {@code offset} of an allocation. */
    default GpuInterop.NativeBuffer nativeBuffer(Object allocation, long offset, long bytes) { throw new ComputeException(ComputeException.Code.UNSUPPORTED, "no interop on this backend"); }
    default long nativeFence(Object fence) { throw new ComputeException(ComputeException.Code.UNSUPPORTED, "no interop on this backend"); }
    /** Records {@code commands} (no downloads) into a caller's command buffer; the result frees its state on close. */
    default Prepared recordInto(long commandBuffer, List<Command> commands) {
        throw new ComputeException(ComputeException.Code.UNSUPPORTED, "recording into external command buffers is not supported by this backend");
    }
    /** Whether the backend runs {@link Dispatch}es with an {@code indirect} region. */
    default boolean indirectDispatch() { return false; }
    /** The hardware queue for transfers (queue 0 is compute), or -1 if transfers share queue 0. */
    default int transferQueue() { return -1; }
    /** Submits on hardware queue {@code queue} (0 compute, or {@link #transferQueue()}: transfers and fills only). */
    default Job submit(int queue, List<Command> commands, boolean profile, List<FenceOp> waits, List<FenceOp> signals) {
        if (queue != 0) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "no transfer queue on this backend");
        return submit(commands, profile, waits, signals);
    }
    /** A fence operation of a submission: wait for, or signal, {@code value} on a driver fence. */
    record FenceOp(Object fence, long value) { }
    /** A new timeline fence at 0, or UNSUPPORTED. */
    default Object fence() { throw new ComputeException(ComputeException.Code.UNSUPPORTED, "fences are not supported by this backend"); }
    default long fenceValue(Object fence) { throw new UnsupportedOperationException(); }
    default void signalFence(Object fence, long value) { throw new UnsupportedOperationException(); }
    /** Blocks until the fence reaches {@code value} or {@code nanos} pass; true if it did. */
    default boolean awaitFence(Object fence, long value, long nanos) { throw new UnsupportedOperationException(); }
    default void releaseFence(Object fence) { }
    /** Submits with fence waits (before the first command) and signals (after the last). */
    default Job submit(List<Command> commands, boolean profile, List<FenceOp> waits, List<FenceOp> signals) {
        if (!waits.isEmpty() || !signals.isEmpty()) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "fences are not supported by this backend");
        return submit(commands, profile);
    }
    /** As {@link #submit(Prepared, List)}, with fence waits and signals. */
    default Job submit(Prepared prepared, List<Command> commands, List<FenceOp> waits, List<FenceOp> signals) {
        if (!waits.isEmpty() || !signals.isEmpty()) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "fences are not supported by this backend");
        return submit(prepared, commands);
    }
    /** Submits {@code commands}, timing each dispatch when {@code profile} (see {@link Job#dispatchNanos()}). */
    default Job submit(List<Command> commands, boolean profile) { return submit(commands); }
    /**
     * Prepares {@code commands} (no downloads) for repeated submission, timing each dispatch of every run when
     * {@code profile}. The default keeps nothing but the flag, and {@link #submit(Prepared, List)} then submits the
     * list afresh each time.
     */
    default Prepared prepare(List<Command> commands, boolean profile) { return new Unprepared(profile); }
    /** What the default {@link #prepare(List, boolean)} keeps. */
    record Unprepared(boolean profile) implements Prepared { @Override public void close() { } }
    /** Commands a driver has prepared for repeated submission (a recording); closed when the recording is. */
    interface Prepared extends AutoCloseable { @Override void close(); }
    /** As {@link #prepare(List, boolean)}, unprofiled. */
    default Prepared prepare(List<Command> commands) { return prepare(commands, false); }
    /**
     * Submits a prepared list again. {@code commands} has the prepared list's shape, with the current parameter blocks
     * and bindings; the previous submission of {@code prepared} has completed.
     */
    default Job submit(Prepared prepared, List<Command> commands) {
        return submit(commands, prepared instanceof Unprepared u && u.profile());
    }
    /** The driver's buffer pools (see {@link MemoryPool}); {@link MemoryStats#NONE} if it has none. */
    default MemoryStats memoryStats() { return MemoryStats.NONE; }
    @Override void close();
}
