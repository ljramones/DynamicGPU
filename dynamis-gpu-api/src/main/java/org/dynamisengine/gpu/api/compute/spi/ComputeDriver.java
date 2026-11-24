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
    @Override void close();
}
