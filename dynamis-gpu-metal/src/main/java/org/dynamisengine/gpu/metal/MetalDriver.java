package org.dynamisengine.gpu.metal;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.spi.ComputeDriver;
import java.lang.foreign.*;
import java.nio.ByteBuffer;
import java.util.*;
import static java.lang.foreign.ValueLayout.*;

final class MetalDriver implements ComputeDriver {
    // Typed Java wrappers prevent accidental buffer/pipeline/device handle substitution.
    private record MetalBuffer(MemorySegment handle, long bytes) { }
    private record MetalPipeline(MemorySegment handle) { }
    private final MetalNative n = MetalNative.instance();
    private final MemorySegment device;
    MetalDriver(int index) { device = n.pointer("dg_device_open", index); }
    @Override public Object allocate(BufferDescriptor d) {
        var buffer = n.address("dg_buffer_create", device, d.bytes(), d.placement() == Placement.PRIVATE ? 1 : 0);
        if (buffer.equals(MemorySegment.NULL)) {
            var error = n.error("buffer allocation bytes=" + d.bytes() + " placement=" + d.placement());
            throw new ComputeException(ComputeException.Code.RESOURCE_EXHAUSTED, error.getMessage(), error);
        }
        return new MetalBuffer(buffer, d.bytes());
    }
    @Override public void releaseBuffer(Object b) { n.release("dg_buffer_release", ((MetalBuffer) b).handle()); }
    @Override public ByteBuffer map(Object b, long offset, long bytes, Access access) {
        var buffer = (MetalBuffer) b; return n.pointer("dg_buffer_contents", buffer.handle()).reinterpret(buffer.bytes()).asSlice(offset, bytes).asByteBuffer();
    }
    @Override public void unmap(Object b, Access access) { /* Shared memory: completion + exclusive host mapping is enforced above. */ }
    @Override public Object pipeline(PipelineDescriptor d) {
        if (!d.specialization().isEmpty()) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "Metal ABI 1 has no function constants; supply a specialized executable");
        var executable = d.kernel().executable(Backend.METAL);
        if (executable.format() != KernelBundle.Format.METALLIB && executable.format() != KernelBundle.Format.MSL_SOURCE)
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "Metal requires METALLIB or MSL_SOURCE");
        try (var arena = Arena.ofConfined()) {
            byte[] bytes = executable.bytes(); var data = arena.allocateFrom(JAVA_BYTE, bytes);
            var p = n.pointer("dg_pipeline_create", device, data, (long) bytes.length, executable.format() == KernelBundle.Format.MSL_SOURCE ? 1 : 0, arena.allocateFrom(d.entryPoint()));
            if (d.localSize().volume() > n.number("dg_pipeline_threads", p)) {
                n.release("dg_pipeline_release", p); throw new ComputeException(ComputeException.Code.UNSUPPORTED, "Workgroup size exceeds Metal pipeline-specific limit");
            }
            return new MetalPipeline(p);
        }
    }
    @Override public void releasePipeline(Object p) { n.release("dg_pipeline_release", ((MetalPipeline) p).handle()); }
    private MemorySegment buffer(Region r) { return ((MetalBuffer) r.allocation()).handle(); }
    @Override public Job submit(List<Command> commands) {
        var batch = n.pointer("dg_batch_create", device);
        List<MetalBuffer> stages = new ArrayList<>(); Map<GpuReadback, MetalBuffer> reads = new LinkedHashMap<>();
        try (var arena = Arena.ofConfined()) {
            for (var command : commands) {
                switch (command) {
                    case Upload u -> n.check("dg_upload", batch, buffer(u.destination()), u.destination().offset(), arena.allocateFrom(JAVA_BYTE, u.data()), (long) u.data().length);
                    case Copy c -> n.check("dg_copy", batch, buffer(c.source()), c.source().offset(), buffer(c.destination()), c.destination().offset(), c.source().bytes());
                    case Fill f -> n.check("dg_fill", batch, buffer(f.destination()), f.destination().offset(), f.destination().bytes(), f.pattern());
                    case Download d -> {
                        var stage = (MetalBuffer) allocate(new BufferDescriptor(d.source().bytes(), Set.of(BufferUsage.TRANSFER_DESTINATION), Placement.SHARED));
                        stages.add(stage); reads.put(d.result(), stage);
                        n.check("dg_copy", batch, buffer(d.source()), d.source().offset(), stage.handle(), 0L, d.source().bytes());
                    }
                    case Dispatch d -> {
                        n.check("dg_compute_begin", batch, ((MetalPipeline) d.pipeline()).handle());
                        for (var binding : d.bindings()) n.check("dg_compute_buffer", batch, buffer(binding.region()), binding.region().offset(), binding.slot());
                        if (d.parameters().length > 0) n.check("dg_compute_parameters", batch, arena.allocateFrom(JAVA_BYTE, d.parameters()), (long) d.parameters().length, d.descriptor().layout().parameterSlot());
                        var g = d.groups(); var l = d.descriptor().localSize(); n.check("dg_compute_dispatch", batch, g.x(), g.y(), g.z(), l.x(), l.y(), l.z());
                    }
                }
            }
            n.check("dg_commit", batch);
        } catch (RuntimeException ex) {
            n.release("dg_batch_release", batch); stages.forEach(this::releaseBuffer); throw ex;
        }
        return new Job() {
            @Override public GpuSubmission.Status poll() {
                int status = n.integer("dg_status", batch); if (status < 0) throw n.error("command buffer completion");
                if (status == 0) return GpuSubmission.Status.PENDING;
                reads.forEach((read, stage) -> {
                    byte[] bytes = new byte[Math.toIntExact(stage.bytes())]; map(stage, 0, stage.bytes(), Access.READ).get(bytes); read.complete(bytes);
                });
                return GpuSubmission.Status.COMPLETE;
            }
            @Override public void close() { n.release("dg_batch_release", batch); stages.forEach(MetalDriver.this::releaseBuffer); }
        };
    }
    @Override public void close() { n.release("dg_device_release", device); }
}
