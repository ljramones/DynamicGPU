package org.dynamisengine.gpu.metal;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.spi.ComputeDriver;
import org.dynamisengine.gpu.api.compute.spi.MemoryPool;
import java.lang.foreign.*;
import java.nio.ByteBuffer;
import java.util.*;
import static java.lang.foreign.ValueLayout.*;

final class MetalDriver implements ComputeDriver {
    // Typed Java wrappers prevent accidental buffer/pipeline/device handle substitution.
    /** A pool block: one MTLBuffer; {@code contents} is its CPU address for shared storage, NULL for private. */
    private record Block(MemorySegment handle, long bytes, MemorySegment contents) { }
    /** A buffer: a range of a block; {@code bytes} is the size asked for. */
    private record MetalBuffer(MemoryPool.Allocation<Block> range, long bytes) {
        MemorySegment handle() { return range.block().handle(); }
        long base() { return range.offset(); }
    }
    /** Blocks of 64 MiB, ranges aligned to 256 bytes; buffers over 16 MiB get their own block. */
    static final long BLOCK_BYTES = 64L << 20, DEDICATED_BYTES = 16L << 20, ALIGNMENT = 256;
    private final Map<Placement, MemoryPool<Block>> pools = new EnumMap<>(Placement.class);
    private record MetalPipeline(MemorySegment handle) { }
    private final MetalNative n = MetalNative.instance();
    private final MemorySegment device;
    MetalDriver(int index) { device = n.pointer("dg_device_open", index); }
    @Override public Object allocate(BufferDescriptor d) { return new MetalBuffer(pool(d.placement()).allocate(d.bytes()), d.bytes()); }
    private MemoryPool<Block> pool(Placement placement) {
        return pools.computeIfAbsent(placement, p -> new MemoryPool<>(new MemoryPool.Blocks<>() {
            @Override public Block create(long bytes) {
                var buffer = n.address("dg_buffer_create", device, bytes, p == Placement.PRIVATE ? 1 : 0);
                if (buffer.equals(MemorySegment.NULL)) {
                    var error = n.error("buffer allocation bytes=" + bytes + " placement=" + p);
                    throw new ComputeException(ComputeException.Code.RESOURCE_EXHAUSTED, error.getMessage(), error);
                }
                var contents = p == Placement.SHARED ? n.pointer("dg_buffer_contents", buffer).reinterpret(bytes) : MemorySegment.NULL;
                return new Block(buffer, bytes, contents);
            }
            @Override public void destroy(Block b) { n.release("dg_buffer_release", b.handle()); }
        }, BLOCK_BYTES, ALIGNMENT, DEDICATED_BYTES));
    }
    @Override public void releaseBuffer(Object b) {
        var buffer = (MetalBuffer) b;
        for (var p : pools.values()) if (p.owns(buffer.range())) { p.free(buffer.range()); return; }
        throw new IllegalStateException("buffer from no pool of this device");
    }
    @Override public org.dynamisengine.gpu.api.compute.MemoryStats memoryStats() {
        var total = new org.dynamisengine.gpu.api.compute.MemoryStats(0, 0, 0, 0, 0);
        for (var p : pools.values()) total = total.plus(p.stats());
        long[] budget = budget();
        return total.withBudget(budget[0], budget[1]);
    }
    /** The recommended working-set size and the bytes allocated now. */
    private long[] budget() {
        try (var arena = Arena.ofConfined()) {
            var values = arena.allocate(JAVA_LONG, 2); n.check("dg_device_memory", device, values);
            return new long[] {values.getAtIndex(JAVA_LONG, 0), values.getAtIndex(JAVA_LONG, 1)};
        }
    }
    /** The CPU view of {@code bytes} at {@code offset} of a shared buffer. */
    private MemorySegment contents(MetalBuffer buffer, long offset, long bytes) {
        var block = buffer.range().block();
        if (block.contents().equals(MemorySegment.NULL)) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "Private Metal buffers cannot be mapped");
        return block.contents().asSlice(buffer.base() + offset, bytes);
    }
    @Override public ByteBuffer map(Object b, long offset, long bytes, Access access) { return contents((MetalBuffer) b, offset, bytes).asByteBuffer(); }
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
    /** A region's offset in its block's MTLBuffer. */
    private static long offset(Region r) { return ((MetalBuffer) r.allocation()).base() + r.offset(); }
    /** A pooled shared staging range holding {@code data}. */
    private MetalBuffer stage(List<MetalBuffer> stages, byte[] data) {
        var stage = (MetalBuffer) allocate(new BufferDescriptor(data.length, Set.of(BufferUsage.TRANSFER_SOURCE), Placement.SHARED)); stages.add(stage);
        MemorySegment.copy(data, 0, contents(stage, 0, data.length), JAVA_BYTE, 0, data.length);
        return stage;
    }
    @Override public Job submit(List<Command> commands) { return submit(commands, false); }
    @Override public Job submit(List<Command> commands, boolean profile) {
        int dispatches = (int) commands.stream().filter(c -> c instanceof Dispatch).count();
        return submitBatch(0, commands, profile ? dispatches : 0, List.of(), List.of());
    }
    @Override public Job submit(List<Command> commands, boolean profile, List<FenceOp> waits, List<FenceOp> signals) {
        int dispatches = (int) commands.stream().filter(c -> c instanceof Dispatch).count();
        return submitBatch(0, commands, profile ? dispatches : 0, waits, signals);
    }
    @Override public int transferQueue() { return 1; }
    @Override public Job submit(int q, List<Command> commands, boolean profile, List<FenceOp> waits, List<FenceOp> signals) {
        int dispatches = (int) commands.stream().filter(c -> c instanceof Dispatch).count();
        return submitBatch(q, commands, profile ? dispatches : 0, waits, signals);
    }
    @Override public Job submit(Prepared prepared, List<Command> commands, List<FenceOp> waits, List<FenceOp> signals) {
        return submit(commands, prepared instanceof Unprepared u && u.profile(), waits, signals);
    }
    // Fences: MTLSharedEvents through the bridge.
    @Override public Object fence() { return n.pointer("dg_event_create", device); }
    @Override public long fenceValue(Object fence) { return n.number("dg_event_value", (MemorySegment) fence); }
    @Override public void signalFence(Object fence, long value) { n.check("dg_event_signal", (MemorySegment) fence, value); }
    @Override public boolean awaitFence(Object fence, long value, long nanos) {
        int r = n.integer("dg_event_wait", (MemorySegment) fence, value, nanos); if (r < 0) throw n.error("event wait"); return r == 1;
    }
    @Override public void releaseFence(Object fence) { n.release("dg_event_release", (MemorySegment) fence); }
    private Job submitBatch(int q, List<Command> commands, int profiled, List<FenceOp> waits, List<FenceOp> signals) {
        var batch = n.pointer("dg_batch_create_on", device, q);
        for (var w : waits) n.check("dg_batch_wait_event", batch, (MemorySegment) w.fence(), w.value());
        // a device without stage-boundary counters runs unprofiled
        final int timed = profiled > 0 && n.integer("dg_batch_profile", batch, profiled) == 0 ? profiled : 0;
        List<MetalBuffer> stages = new ArrayList<>(); Map<GpuReadback, MetalBuffer> reads = new LinkedHashMap<>();
        try (var arena = Arena.ofConfined()) {
            for (var command : commands) {
                switch (command) {
                    case Upload u -> {
                        var stage = stage(stages, u.data());
                        n.check("dg_copy", batch, stage.handle(), stage.base(), buffer(u.destination()), offset(u.destination()), (long) u.data().length);
                    }
                    case Copy c -> n.check("dg_copy", batch, buffer(c.source()), offset(c.source()), buffer(c.destination()), offset(c.destination()), c.source().bytes());
                    case Fill f -> {
                        int words = Math.toIntExact(f.destination().bytes() / 4);
                        var stage = (MetalBuffer) allocate(new BufferDescriptor(4L * words, Set.of(BufferUsage.TRANSFER_SOURCE), Placement.SHARED)); stages.add(stage);
                        var view = contents(stage, 0, 4L * words);
                        for (int i = 0; i < words; i++) view.setAtIndex(JAVA_INT_UNALIGNED, i, f.pattern());
                        n.check("dg_copy", batch, stage.handle(), stage.base(), buffer(f.destination()), offset(f.destination()), 4L * words);
                    }
                    case Download d -> {
                        var stage = (MetalBuffer) allocate(new BufferDescriptor(d.source().bytes(), Set.of(BufferUsage.TRANSFER_DESTINATION), Placement.SHARED));
                        stages.add(stage); reads.put(d.result(), stage);
                        n.check("dg_copy", batch, buffer(d.source()), offset(d.source()), stage.handle(), stage.base(), d.source().bytes());
                    }
                    case Dispatch d -> {
                        n.check("dg_compute_begin", batch, ((MetalPipeline) d.pipeline()).handle());
                        if (!d.label().isEmpty()) n.check("dg_compute_label", batch, arena.allocateFrom(d.label()));
                        for (var binding : d.bindings()) n.check("dg_compute_buffer", batch, buffer(binding.region()), offset(binding.region()), binding.slot());
                        if (d.parameters().length > 0) n.check("dg_compute_parameters", batch, arena.allocateFrom(JAVA_BYTE, d.parameters()), (long) d.parameters().length, d.descriptor().layout().parameterSlot());
                        var g = d.groups(); var l = d.descriptor().localSize(); n.check("dg_compute_dispatch", batch, g.x(), g.y(), g.z(), l.x(), l.y(), l.z());
                    }
                }
            }
            for (var g : signals) n.check("dg_batch_signal_event", batch, (MemorySegment) g.fence(), g.value());
            n.check("dg_commit", batch);
        } catch (RuntimeException ex) {
            n.release("dg_batch_release", batch); stages.forEach(this::releaseBuffer); throw ex;
        }
        return new Job() {
            private java.util.OptionalLong gpuNanos = java.util.OptionalLong.empty();
            @Override public java.util.OptionalLong gpuNanos() { return gpuNanos; }
            private long[] dispatchNanos;
            @Override public long[] dispatchNanos() { return dispatchNanos; }
            @Override public boolean await(long nanos) {
                int r = n.integer("dg_wait", batch, nanos); if (r < 0) throw n.error("command buffer wait"); return r == 1;
            }
            @Override public GpuSubmission.Status poll() {
                int status = n.integer("dg_status", batch); if (status < 0) throw n.error("command buffer completion");
                if (status == 0) return GpuSubmission.Status.PENDING;
                double seconds = n.real("dg_gpu_seconds", batch);
                if (seconds >= 0) gpuNanos = java.util.OptionalLong.of(Math.round(seconds * 1e9));
                if (timed > 0) {
                    try (var arena = Arena.ofConfined()) {
                        var out = arena.allocate(JAVA_LONG, timed);
                        if (n.integer("dg_dispatch_times", batch, out, timed) == 0) dispatchNanos = out.toArray(JAVA_LONG);
                    }
                }
                reads.forEach((read, stage) -> {
                    byte[] bytes = new byte[Math.toIntExact(stage.bytes())]; map(stage, 0, stage.bytes(), Access.READ).get(bytes); read.complete(bytes);
                });
                return GpuSubmission.Status.COMPLETE;
            }
            @Override public void close() { n.release("dg_batch_release", batch); stages.forEach(MetalDriver.this::releaseBuffer); }
        };
    }
    @Override public void close() { pools.values().forEach(MemoryPool::destroyAll); n.release("dg_device_release", device); }
}
