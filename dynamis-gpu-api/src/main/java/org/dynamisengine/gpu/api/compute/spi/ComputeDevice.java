package org.dynamisengine.gpu.api.compute.spi;

import org.dynamisengine.gpu.api.compute.*;
import java.nio.ByteBuffer;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.locks.LockSupport;
import static org.dynamisengine.gpu.api.compute.ComputeException.Code.*;
import static org.dynamisengine.gpu.api.compute.spi.ComputeDriver.*;

/** Shared validation, queue ordering, ownership, and retirement for native backends and mocks. */
public final class ComputeDevice implements GpuDevice {
    private final GpuAdapter adapter;
    private final ComputeDriver driver;
    private final Thread owner = Thread.currentThread();
    private final Set<Resource> resources = Collections.newSetFromMap(new IdentityHashMap<>());
    private final Set<AutoCloseable> handles = Collections.newSetFromMap(new IdentityHashMap<>());
    private final List<Submission> pending = new ArrayList<>();
    private ComputeException queueFailure;
    private boolean closed;
    private Set<String> measuredFeatures; // null until FeatureProbe has run on this device

    public ComputeDevice(GpuAdapter adapter, ComputeDriver driver) {
        this.adapter = Objects.requireNonNull(adapter); this.driver = Objects.requireNonNull(driver);
    }
    private void thread() {
        if (Thread.currentThread() != owner) throw new IllegalStateException("Device operations require owner thread " + owner.getName());
    }
    private void check() { thread(); if (closed) throw new ComputeException(CLOSED, "Device is closed"); }
    private static void require(boolean valid, String detail) { if (!valid) throw new ComputeException(INVALID_ARGUMENT, detail); }
    private static void range(long offset, long length, long size) {
        require(offset >= 0 && length > 0 && offset <= size && length <= size - offset, "Byte range outside allocation");
    }
    @Override public GpuAdapter adapter() { return adapter; }
    /**
     * The adapter's capabilities plus the shader features measured on this device ({@link GpuCapabilities#FUSED_FMA}).
     * The first call on the owner thread runs a one-dispatch probe and waits for it; later calls are free. Called on
     * another thread or after close before that, it returns the adapter's capabilities.
     */
    @Override public GpuCapabilities capabilities() {
        if (measuredFeatures == null) {
            if (Thread.currentThread() != owner || closed) return adapter.capabilities();
            measuredFeatures = Set.of(); // the probe's own calls see the adapter capabilities
            measuredFeatures = FeatureProbe.measure(this);
        }
        return measuredFeatures.isEmpty() ? adapter.capabilities() : adapter.capabilities().withShaderFeatures(measuredFeatures);
    }
    /** Limits for validation; reading them never runs the feature probe. */
    private GpuCapabilities limits() { return adapter.capabilities(); }
    @Override public GpuBuffer allocate(BufferDescriptor descriptor) {
        check(); require(descriptor.bytes() <= limits().maxBufferBytes(), "Buffer exceeds device allocation limit");
        return new Buffer(new Resource(driver.allocate(descriptor), false), descriptor, 0, descriptor.bytes());
    }
    @Override public GpuComputePipeline pipeline(PipelineDescriptor descriptor) {
        check(); var caps = descriptor.requiredFeatures().isEmpty() ? limits() : capabilities(); var layout = descriptor.layout();
        descriptor.kernel().executable(adapter.backend());
        if (!caps.dataTypes().containsAll(descriptor.requiredDataTypes()) || !caps.shaderFeatures().containsAll(descriptor.requiredFeatures()))
            throw new ComputeException(UNSUPPORTED, adapter.name() + " lacks requested data types/features: " + descriptor.requiredDataTypes() + " / " + descriptor.requiredFeatures());
        require(layout.parameterBytes() <= caps.maxParameterBytes(), "Scalar parameter block exceeds device limit");
        require(layout.buffers().size() <= caps.maxBufferBindings()
                && layout.buffers().stream().allMatch(b -> b.slot() < caps.maxBufferBindings()), "Buffer binding slots exceed device limit");
        require(layout.parameterSlot() <= caps.maxBufferBindings(), "Parameter binding slot exceeds device limit");
        caps.validateGroups(descriptor.localSize(), new Dimensions(1, 1, 1));
        return new Pipeline(new Resource(driver.pipeline(descriptor), true), descriptor);
    }
    @Override public GpuCommandEncoder encoder() { check(); return new Encoder(); }
    @Override public GpuQueue queue() { check(); return this::submit; }

    private final class Resource {
        final Object nativeObject;
        final boolean pipeline;
        int references;
        int recorded;
        boolean mapped;
        Resource(Object nativeObject, boolean pipeline) {
            this.nativeObject = nativeObject; this.pipeline = pipeline; resources.add(this);
        }
        void retain() { references++; }
        void release() {
            if (--references == 0) {
                if (pipeline) driver.releasePipeline(nativeObject); else driver.releaseBuffer(nativeObject);
                resources.remove(this);
            }
        }
    }
    private final class Buffer implements BufferSlice {
        final Resource resource;
        final BufferDescriptor descriptor;
        final long offset, bytes;
        boolean released;
        Buffer(Resource resource, BufferDescriptor descriptor, long offset, long bytes) {
            this.resource = resource; this.descriptor = descriptor; this.offset = offset; this.bytes = bytes;
            resource.retain(); handles.add(this);
        }
        void valid() { check(); if (released) throw new ComputeException(CLOSED, "Buffer handle is closed"); }
        void usage(BufferUsage usage) { valid(); require(descriptor.usages().contains(usage), "Buffer requires " + usage); }
        Region region() { valid(); require(!resource.mapped, "Buffer is mapped; close mapping before recording GPU work"); return new Region(resource.nativeObject, offset, bytes); }
        @Override public long bytes() { return bytes; }
        @Override public long offset() { return offset; }
        @Override public GpuDevice device() { return ComputeDevice.this; }
        @Override public BufferDescriptor descriptor() { return descriptor; }
        @Override public BufferSlice slice(long relativeOffset, long length) {
            valid(); range(relativeOffset, length, bytes); return new Buffer(resource, descriptor, Math.addExact(offset, relativeOffset), length);
        }
        @Override public GpuMapping map(Access access) {
            valid(); Objects.requireNonNull(access);
            require(bytes <= Integer.MAX_VALUE, "Mapped view exceeds Java ByteBuffer size; map a smaller slice");
            if (access != Access.WRITE) usage(BufferUsage.MAP_READ);
            if (access != Access.READ) usage(BufferUsage.MAP_WRITE);
            require(descriptor.placement() == Placement.SHARED, "Only shared storage can be mapped");
            progress(); require(resource.recorded == 0 && !resource.mapped, "Buffer has recorded/in-flight GPU work or an active mapping");
            ByteBuffer view = driver.map(resource.nativeObject, offset, bytes, access);
            if (access == Access.READ) view = view.asReadOnlyBuffer();
            final ByteBuffer mapped = view.order(java.nio.ByteOrder.nativeOrder());
            resource.mapped = true; resource.retain();
            GpuMapping mapping = new GpuMapping() {
                boolean done;
                @Override public ByteBuffer bytes() { check(); if (done) throw new ComputeException(CLOSED, "Mapping is closed"); return mapped; }
                @Override public void close() {
                    thread(); if (!done) {
                        done = true;
                        try { driver.unmap(resource.nativeObject, access); }
                        finally { resource.mapped = false; resource.release(); handles.remove(this); }
                    }
                }
            };
            handles.add(mapping); return mapping;
        }
        @Override public void close() { thread(); if (!released) { released = true; handles.remove(this); resource.release(); } }
    }
    private Buffer buffer(GpuBuffer buffer) {
        require(buffer instanceof ComputeDevice.Buffer && buffer.device() == this, "Cross-device or foreign buffer");
        Buffer b = (Buffer) buffer; b.valid(); return b;
    }
    private final class Pipeline implements GpuComputePipeline {
        final Resource resource;
        final PipelineDescriptor descriptor;
        boolean released;
        Pipeline(Resource resource, PipelineDescriptor descriptor) { this.resource = resource; this.descriptor = descriptor; resource.retain(); handles.add(this); }
        @Override public GpuDevice device() { return ComputeDevice.this; }
        @Override public PipelineDescriptor descriptor() { return descriptor; }
        @Override public void close() { thread(); if (!released) { released = true; handles.remove(this); resource.release(); } }
    }
    private final class Encoder implements GpuCommandEncoder {
        final List<Command> commands = new ArrayList<>();
        final Set<Resource> used = Collections.newSetFromMap(new IdentityHashMap<>());
        final List<GpuReadback> errorChecks = new ArrayList<>();
        boolean consumed;
        Encoder() { handles.add(this); }
        private void errorBuffer(GpuBuffer errors) {
            require(errors != null && errors.bytes() >= KernelErrors.BYTES, "A kernel error buffer needs at least " + KernelErrors.BYTES + " bytes");
        }
        @Override public GpuCommandEncoder resetErrors(GpuBuffer errors) { errorBuffer(errors); return fill(errors, 0); }
        @Override public GpuCommandEncoder checkErrors(GpuBuffer errors) { errorBuffer(errors); errorChecks.add(download(errors)); return this; }
        void valid() { check(); if (consumed) throw new ComputeException(CLOSED, "Encoder already submitted/closed"); }
        void retain(Resource r) { if (used.add(r)) { r.retain(); r.recorded++; } }
        void aligned(Buffer b) { require((b.offset & 3) == 0 && (b.bytes & 3) == 0, "Transfers require four-byte aligned ranges"); }
        @Override public GpuCommandEncoder upload(GpuBuffer destination, byte[] data) {
            valid(); Buffer b = buffer(destination); b.usage(BufferUsage.TRANSFER_DESTINATION); aligned(b);
            require(data.length == b.bytes, "Upload byte count must equal destination range");
            Region region = b.region(); byte[] snapshot = data.clone(); retain(b.resource); commands.add(new Upload(region, snapshot)); return this;
        }
        @Override public GpuReadback download(GpuBuffer source) {
            valid(); Buffer b = buffer(source); b.usage(BufferUsage.TRANSFER_SOURCE); aligned(b);
            require(b.bytes <= Integer.MAX_VALUE, "Host readback exceeds Java array size");
            Region region = b.region(); retain(b.resource); var readback = new GpuReadback(); commands.add(new Download(region, readback)); return readback;
        }
        @Override public GpuCommandEncoder copy(GpuBuffer source, GpuBuffer destination) {
            valid(); Buffer a = buffer(source), b = buffer(destination);
            a.usage(BufferUsage.TRANSFER_SOURCE); b.usage(BufferUsage.TRANSFER_DESTINATION); aligned(a); aligned(b);
            require(a.bytes == b.bytes, "Copy ranges must have equal byte sizes");
            require(!overlap(a, b), "Overlapping device copy is unsupported");
            Region ar = a.region(), br = b.region(); retain(a.resource); retain(b.resource); commands.add(new Copy(ar, br)); return this;
        }
        @Override public GpuCommandEncoder fill(GpuBuffer destination, int pattern) {
            valid(); Buffer b = buffer(destination); b.usage(BufferUsage.TRANSFER_DESTINATION); aligned(b);
            Region region = b.region(); retain(b.resource); commands.add(new Fill(region, pattern)); return this;
        }
        @Override public GpuCommandEncoder dispatch(GpuComputePipeline pipeline, Map<Integer, ? extends GpuBuffer> bindings, byte[] parameters, Dimensions groups) {
            valid(); require(pipeline instanceof ComputeDevice.Pipeline && pipeline.device() == ComputeDevice.this, "Cross-device or foreign pipeline");
            Pipeline p = (Pipeline) pipeline;
            if (p.released) throw new ComputeException(CLOSED, "Pipeline handle is closed");
            var d = p.descriptor; limits().validateGroups(d.localSize(), groups);
            require(parameters.length == d.layout().parameterBytes(), "Scalar parameter byte count differs from layout");
            require(bindings.size() == d.layout().buffers().size(), "Binding count differs from layout");
            List<Bound> bound = new ArrayList<>(); List<Buffer> buffers = new ArrayList<>();
            for (var spec : d.layout().buffers()) {
                require(bindings.containsKey(spec.slot()), "Missing buffer slot " + spec.slot());
                Buffer b = buffer(bindings.get(spec.slot())); b.usage(BufferUsage.STORAGE);
                require(b.offset % limits().storageOffsetAlignment() == 0, "Misaligned storage binding at slot " + spec.slot());
                require(b.bytes >= spec.minimumBytes() && b.bytes <= limits().maxStorageRange(), "Invalid storage range at slot " + spec.slot());
                bound.add(new Bound(spec.slot(), b.region(), spec.access())); buffers.add(b);
            }
            for (int i = 0; i < buffers.size(); i++) for (int j = i + 1; j < buffers.size(); j++)
                require(!overlap(buffers.get(i), buffers.get(j)) || (!bound.get(i).access().writes() && !bound.get(j).access().writes()), "Writable dispatch bindings overlap");
            byte[] snapshot = parameters.clone(); retain(p.resource); buffers.forEach(b -> retain(b.resource));
            commands.add(new Dispatch(p.resource.nativeObject, d, List.copyOf(bound), snapshot, groups)); return this;
        }
        void release() { for (var r : used) { r.recorded--; r.release(); } used.clear(); }
        @Override public void close() {
            thread(); if (!consumed) {
                consumed = true; handles.remove(this); release();
                var error = new ComputeException(CLOSED, "Encoder closed without submission");
                for (var c : commands) if (c instanceof Download d) d.result().fail(error);
                commands.clear();
            }
        }
    }
    private boolean overlap(Buffer a, Buffer b) { return a.resource == b.resource && a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes; }
    private GpuSubmission submit(GpuCommandEncoder encoder) {
        check(); progress(); if (queueFailure != null) throw queueFailure;
        require(encoder instanceof ComputeDevice.Encoder && handles.contains(encoder), "Foreign or consumed encoder");
        Encoder e = (Encoder) encoder; e.valid();
        Job job;
        try { job = driver.submit(List.copyOf(e.commands)); }
        catch (RuntimeException ex) {
            var reads = e.commands.stream().filter(c -> c instanceof Download).map(c -> ((Download) c).result()).toList();
            e.close();
            if (ex instanceof ComputeException error) {
                reads.forEach(read -> read.fail(error));
                if (error.code() == DEVICE_LOST) queueFailure = error;
            }
            throw ex;
        }
        e.consumed = true; handles.remove(e);
        Submission submission = new Submission(e, job); pending.add(submission); return submission;
    }
    private final class Submission implements GpuSubmission {
        final Encoder encoder;
        final Job job;
        Status state = Status.PENDING;
        ComputeException error;
        Submission(Encoder encoder, Job job) { this.encoder = encoder; this.job = job; }
        void poll() {
            try {
                var result = job.poll();
                if (result == Status.PENDING) return;
                if (result == Status.FAILED) throw new ComputeException(NATIVE_ERROR, "Backend reported failed submission without diagnostics");
                if (queueFailure != null) throw queueFailure;
                state = Status.COMPLETE;
                // Data-dependent kernel errors fail this submission only: the queue stays usable and the other
                // readbacks keep their results for inspection.
                for (var check : encoder.errorChecks) {
                    var report = KernelErrors.parse(check.bytes());
                    if (report.any()) {
                        state = Status.FAILED;
                        error = new ComputeException(KERNEL_ERROR, "Kernel reported errors: " + report);
                        break;
                    }
                }
            } catch (ComputeException ex) {
                state = Status.FAILED; error = ex; queueFailure = ex;
                for (var c : encoder.commands) if (c instanceof Download d) d.result().fail(ex);
            }
            try { job.close(); } finally { encoder.release(); encoder.commands.clear(); }
        }
        @Override public Status status() { thread(); if (state == Status.PENDING) progress(); return state; }
        @Override public Optional<ComputeException> failure() { status(); return Optional.ofNullable(error); }
        @Override public boolean await(Duration timeout) {
            thread(); if (timeout.isNegative()) throw new IllegalArgumentException("Negative timeout");
            long budget; try { budget = timeout.toNanos(); } catch (ArithmeticException ex) { budget = Long.MAX_VALUE; }
            long start = System.nanoTime();
            while (status() == Status.PENDING) {
                if (System.nanoTime() - start >= budget) return false;
                LockSupport.parkNanos(Math.min(100_000, budget - (System.nanoTime() - start)));
                if (Thread.currentThread().isInterrupted()) throw new ComputeException(TIMEOUT, "Interrupted while waiting; submission remains owned by device");
            }
            if (error != null) throw error; return true;
        }
        @Override public void close() { if (!await(Duration.ofSeconds(30))) throw new ComputeException(TIMEOUT, "Submission still pending after 30 seconds; device retains its resources"); }
    }
    private void progress() {
        while (!pending.isEmpty()) {
            Submission s = pending.getFirst(); s.poll();
            if (s.state == GpuSubmission.Status.PENDING) break;
            pending.removeFirst();
        }
    }
    @Override public void close() {
        thread(); if (closed) return;
        ComputeException failure = queueFailure;
        for (var s : List.copyOf(pending)) {
            try { if (!s.await(Duration.ofSeconds(30))) throw new ComputeException(TIMEOUT, "Device drain timed out; retry close after completion"); }
            catch (ComputeException ex) { if (s.state == GpuSubmission.Status.PENDING) throw ex; failure = ex; }
        }
        // Mappings, handles, and encoders each own independent references; order does not matter.
        for (var handle : List.copyOf(handles)) {
            try { handle.close(); } catch (RuntimeException ex) { throw ex; } catch (Exception ex) { throw new AssertionError(ex); }
        }
        if (!resources.isEmpty()) throw new IllegalStateException("Unreleased compute resources: " + resources.size());
        closed = true; driver.close(); if (failure != null) throw failure;
    }
}
