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
    /** Live buffers and requested bytes per memory class (BufferDescriptor.label). */
    private final Map<String, long[]> memoryClasses = new TreeMap<>();
    private long submittedCount, completedCount, failedCount;
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
    @Override public MemoryStats memoryStats() { check(); return driver.memoryStats(); }
    @Override public Map<String, MemoryUse> memoryByLabel() {
        check(); Map<String, MemoryUse> m = new TreeMap<>();
        memoryClasses.forEach((k, v) -> m.put(k, new MemoryUse(v[0], v[1]))); return Collections.unmodifiableMap(m);
    }
    @Override public GpuBuffer allocate(BufferDescriptor descriptor) {
        check(); require(descriptor.bytes() <= limits().maxBufferBytes(), "Buffer exceeds device allocation limit");
        var resource = new Resource(driver.allocate(descriptor), false);
        resource.memoryClass = descriptor.label(); resource.bytes = descriptor.bytes();
        var use = memoryClasses.computeIfAbsent(descriptor.label(), k -> new long[2]); use[0]++; use[1] += descriptor.bytes();
        return new Buffer(resource, descriptor, 0, descriptor.bytes());
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
    @Override public GpuQueue queue() {
        check();
        return new GpuQueue() {
            @Override public GpuSubmission submit(GpuCommandEncoder encoder) { return ComputeDevice.this.submit(encoder); }
            @Override public GpuSubmission submit(GpuRecording recording) { return ComputeDevice.this.submit(recording); }
            @Override public QueueStats stats() {
                check(); progress();   // non-blocking: polls fences and command buffers, never waits
                long bytes = 0; for (var s : pending) bytes += s.transferBytes;
                return new QueueStats(pending.size(), submittedCount, completedCount, failedCount, bytes);
            }
        };
    }

    private final class Resource {
        final Object nativeObject;
        final boolean pipeline;
        int references;
        int recorded;
        boolean mapped;
        String memoryClass; long bytes;   // buffers only
        Resource(Object nativeObject, boolean pipeline) {
            this.nativeObject = nativeObject; this.pipeline = pipeline; resources.add(this);
        }
        void retain() { references++; }
        void release() {
            if (--references == 0) {
                if (pipeline) driver.releasePipeline(nativeObject); else driver.releaseBuffer(nativeObject);
                resources.remove(this);
                if (memoryClass != null) {
                    var use = memoryClasses.get(memoryClass); use[0]--; use[1] -= bytes;
                    if (use[0] == 0) memoryClasses.remove(memoryClass);
                }
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
        /** The resources each command uses, parallel to {@code commands} (for recordings, which rebind). */
        final List<List<Resource>> commandResources = new ArrayList<>();
        final Set<Resource> used = Collections.newSetFromMap(new IdentityHashMap<>());
        final List<GpuReadback> errorChecks = new ArrayList<>();
        boolean consumed, profile;
        String label = "", name = "";
        /** The effective label of each dispatch, in order. */
        final List<String> dispatchLabels = new ArrayList<>();
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
            Region region = b.region(); byte[] snapshot = data.clone(); retain(b.resource); add(new Upload(region, snapshot), b.resource); return this;
        }
        @Override public GpuReadback download(GpuBuffer source) {
            valid(); Buffer b = buffer(source); b.usage(BufferUsage.TRANSFER_SOURCE); aligned(b);
            require(b.bytes <= Integer.MAX_VALUE, "Host readback exceeds Java array size");
            Region region = b.region(); retain(b.resource); var readback = new GpuReadback(); add(new Download(region, readback), b.resource); return readback;
        }
        @Override public GpuCommandEncoder copy(GpuBuffer source, GpuBuffer destination) {
            valid(); Buffer a = buffer(source), b = buffer(destination);
            a.usage(BufferUsage.TRANSFER_SOURCE); b.usage(BufferUsage.TRANSFER_DESTINATION); aligned(a); aligned(b);
            require(a.bytes == b.bytes, "Copy ranges must have equal byte sizes");
            require(!overlap(a, b), "Overlapping device copy is unsupported");
            Region ar = a.region(), br = b.region(); retain(a.resource); retain(b.resource); add(new Copy(ar, br), a.resource, b.resource); return this;
        }
        @Override public GpuCommandEncoder fill(GpuBuffer destination, int pattern) {
            valid(); Buffer b = buffer(destination); b.usage(BufferUsage.TRANSFER_DESTINATION); aligned(b);
            Region region = b.region(); retain(b.resource); add(new Fill(region, pattern), b.resource); return this;
        }
        @Override public GpuCommandEncoder dispatch(GpuComputePipeline pipeline, Map<Integer, ? extends GpuBuffer> bindings, byte[] parameters, Dimensions groups) {
            return dispatch(pipeline, bindings, parameters, groups, "");
        }
        @Override public GpuCommandEncoder dispatch(GpuComputePipeline pipeline, Map<Integer, ? extends GpuBuffer> bindings, byte[] parameters, Dimensions groups, String dispatchLabel) {
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
            List<Resource> rs = new ArrayList<>(); rs.add(p.resource); buffers.forEach(b -> rs.add(b.resource));
            commands.add(new Dispatch(p.resource.nativeObject, d, List.copyOf(bound), snapshot, groups)); commandResources.add(rs);
            String inner = dispatchLabel == null ? "" : dispatchLabel;
            dispatchLabels.add(label.isEmpty() ? inner : inner.isEmpty() ? label : label + "/" + inner);
            return this;
        }
        void add(Command c, Resource... rs) { commands.add(c); commandResources.add(List.of(rs)); }
        @Override public GpuCommandEncoder profile() { valid(); profile = true; return this; }
        @Override public GpuCommandEncoder label(String label) { valid(); this.label = label == null ? "" : label; return this; }
        @Override public GpuCommandEncoder name(String name) { valid(); this.name = name == null ? "" : name; return this; }
        void release() { for (var r : used) { r.recorded--; r.release(); } used.clear(); }
        @Override public GpuRecording record() {
            valid();
            require(errorChecks.isEmpty(), "A recording cannot check kernel errors (readbacks are single-use)");
            require(commands.stream().noneMatch(c -> c instanceof Download), "A recording cannot contain downloads (readbacks are single-use)");
            consumed = true; handles.remove(this);
            for (var r : used) r.recorded--; // held, not in use, until a run is submitted
            return new Recording(this);
        }
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
        try { job = driver.submit(List.copyOf(e.commands), e.profile); }
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
        Submission submission = new Submission(e.errorChecks, e.commands, () -> { e.release(); e.commands.clear(); }, job);
        submission.name = e.name; submission.labels = List.copyOf(e.dispatchLabels); submission.transferBytes = transferBytes(e.commands);
        submittedCount++; pending.add(submission); return submission;
    }
    private final class Submission implements GpuSubmission {
        final List<GpuReadback> errorChecks;
        final List<Command> commands;
        final Runnable done;
        final Job job;
        String name = "";
        List<String> labels = List.of();
        long transferBytes;
        Status state = Status.PENDING;
        ComputeException error;
        java.util.OptionalLong gpuNanos = java.util.OptionalLong.empty();
        List<Duration> dispatchTimes = List.of();
        Submission(List<GpuReadback> errorChecks, List<Command> commands, Runnable done, Job job) {
            this.errorChecks = errorChecks; this.commands = commands; this.done = done; this.job = job;
        }
        void poll() {
            try {
                var result = job.poll();
                if (result == Status.PENDING) return;
                if (result == Status.FAILED) throw new ComputeException(NATIVE_ERROR, "Backend reported failed submission without diagnostics");
                if (queueFailure != null) throw queueFailure;
                state = Status.COMPLETE;
                gpuNanos = job.gpuNanos();
                long[] times = job.dispatchNanos();
                if (times != null) dispatchTimes = Arrays.stream(times).mapToObj(Duration::ofNanos).toList();
                // Data-dependent kernel errors fail this submission only: the queue stays usable and the other
                // readbacks keep their results for inspection.
                for (var check : errorChecks) {
                    var report = KernelErrors.parse(check.bytes());
                    if (report.any()) {
                        state = Status.FAILED;
                        error = new ComputeException(KERNEL_ERROR, "Kernel reported errors: " + report);
                        break;
                    }
                }
            } catch (ComputeException ex) {
                state = Status.FAILED; error = ex; queueFailure = ex;
                for (var c : commands) if (c instanceof Download d) d.result().fail(ex);
            }
            if (state == Status.COMPLETE) completedCount++; else failedCount++;
            try { job.close(); } finally { done.run(); }
        }
        @Override public Status status() { thread(); if (state == Status.PENDING) progress(); return state; }
        @Override public Optional<ComputeException> failure() { status(); return Optional.ofNullable(error); }
        @Override public List<Duration> dispatchTimes() { status(); return dispatchTimes; }
        @Override public List<DispatchTiming> dispatchTimings() {
            status(); List<DispatchTiming> out = new ArrayList<>();
            for (int i = 0; i < dispatchTimes.size(); i++) out.add(new DispatchTiming(i, i < labels.size() ? labels.get(i) : "", dispatchTimes.get(i)));
            return out;
        }
        @Override public String name() { return name; }
        @Override public Optional<Duration> gpuTime() {
            status(); return gpuNanos.isPresent() ? Optional.of(Duration.ofNanos(gpuNanos.getAsLong())) : Optional.empty();
        }
        @Override public boolean await(Duration timeout) {
            thread(); if (timeout.isNegative()) throw new IllegalArgumentException("Negative timeout");
            long budget; try { budget = timeout.toNanos(); } catch (ArithmeticException ex) { budget = Long.MAX_VALUE; }
            long start = System.nanoTime();
            while (status() == Status.PENDING) {
                long left = budget - (System.nanoTime() - start);
                if (left <= 0) return false;
                // block on the driver's completion signal (a fence, a command-buffer handler), in slices so that
                // interrupts and earlier submissions are noticed; drivers without one sleep briefly
                job.await(Math.min(left, 10_000_000L));
                if (Thread.currentThread().isInterrupted()) throw new ComputeException(TIMEOUT, "Interrupted while waiting; submission remains owned by device");
            }
            if (error != null) throw error; return true;
        }
        @Override public void close() { if (!await(Duration.ofSeconds(30))) throw new ComputeException(TIMEOUT, "Submission still pending after 30 seconds; device retains its resources"); }
    }
    private final class Recording implements GpuRecording {
        final List<Command> commands;
        final List<List<Resource>> commandResources;
        /** How many commands use each resource; each resource is retained once while counted. */
        final Map<Resource, Integer> counts = new IdentityHashMap<>();
        final int[] dispatchIndex;
        final Prepared prepared;
        Submission running;
        boolean closed;
        final String name;
        final List<String> labels;
        Object tag;
        Recording(Encoder e) {
            name = e.name; labels = List.copyOf(e.dispatchLabels);
            commands = new ArrayList<>(e.commands); commandResources = new ArrayList<>(e.commandResources);
            for (var rs : commandResources) for (var r : rs) counts.merge(r, 1, Integer::sum);
            for (var r : e.used) if (!counts.containsKey(r)) r.release(); // never the case; keeps the books exact
            dispatchIndex = java.util.stream.IntStream.range(0, commands.size()).filter(i -> commands.get(i) instanceof Dispatch).toArray();
            Prepared p;
            try { p = driver.prepare(List.copyOf(commands), e.profile); }
            catch (RuntimeException ex) { counts.keySet().forEach(Resource::release); throw ex; }
            prepared = p; handles.add(this);
        }
        void valid() { check(); if (closed) throw new ComputeException(CLOSED, "Recording is closed"); }
        /** Waits for the run in flight, so the recording can change. */
        void idle() {
            if (running != null && running.state == GpuSubmission.Status.PENDING && !running.await(Duration.ofSeconds(30)))
                throw new ComputeException(TIMEOUT, "Recording still running after 30 seconds");
        }
        int index(int dispatch) {
            require(dispatch >= 0 && dispatch < dispatchIndex.length, "No dispatch " + dispatch + " in a recording of " + dispatchIndex.length);
            return dispatchIndex[dispatch];
        }
        @Override public int dispatches() { return dispatchIndex.length; }
        @Override public String name() { return name; }
        @Override public Object tag() { return tag; }
        @Override public GpuRecording tag(Object tag) { valid(); this.tag = tag; return this; }
        @Override public GpuRecording parameters(int dispatch, byte[] parameters) {
            valid(); int i = index(dispatch); var d = (Dispatch) commands.get(i);
            require(parameters.length == d.descriptor().layout().parameterBytes(), "Scalar parameter byte count differs from layout");
            idle(); commands.set(i, new Dispatch(d.pipeline(), d.descriptor(), d.bindings(), parameters.clone(), d.groups())); return this;
        }
        @Override public GpuRecording groups(int dispatch, Dimensions groups) {
            valid(); int i = index(dispatch); var d = (Dispatch) commands.get(i);
            limits().validateGroups(d.descriptor().localSize(), groups);
            idle(); commands.set(i, new Dispatch(d.pipeline(), d.descriptor(), d.bindings(), d.parameters(), groups)); return this;
        }
        @Override public GpuRecording bind(int dispatch, int slot, GpuBuffer buffer) {
            valid(); int i = index(dispatch); var d = (Dispatch) commands.get(i);
            int position = -1;
            for (int k = 0; k < d.bindings().size(); k++) if (d.bindings().get(k).slot() == slot) position = k;
            require(position >= 0, "Dispatch " + dispatch + " has no buffer slot " + slot);
            var spec = d.descriptor().layout().buffers().stream().filter(x -> x.slot() == slot).findFirst().orElseThrow();
            Buffer b = buffer(buffer); b.usage(BufferUsage.STORAGE);
            require(b.offset % limits().storageOffsetAlignment() == 0, "Misaligned storage binding at slot " + slot);
            require(b.bytes >= spec.minimumBytes() && b.bytes <= limits().maxStorageRange(), "Invalid storage range at slot " + slot);
            Region region = b.region();
            // the dispatch's other bindings: writable overlaps are rejected as when recording
            List<Resource> rs = commandResources.get(i);
            for (int k = 0; k < d.bindings().size(); k++) {
                if (k == position) continue;
                var other = d.bindings().get(k).region();
                boolean overlaps = rs.get(k + 1) == b.resource && b.offset < other.offset() + other.bytes() && other.offset() < b.offset + b.bytes;
                require(!overlaps || (!spec.access().writes() && !d.bindings().get(k).access().writes()), "Writable dispatch bindings overlap");
            }
            idle();
            var bound = new ArrayList<>(d.bindings()); bound.set(position, new Bound(slot, region, spec.access()));
            commands.set(i, new Dispatch(d.pipeline(), d.descriptor(), List.copyOf(bound), d.parameters(), d.groups()));
            var newRs = new ArrayList<>(rs); Resource old = newRs.set(position + 1, b.resource); commandResources.set(i, newRs);
            if (counts.merge(b.resource, 1, Integer::sum) == 1) b.resource.retain();
            if (counts.merge(old, -1, Integer::sum) == 0) { counts.remove(old); old.release(); }
            return this;
        }
        @Override public void close() {
            thread(); if (closed) return;
            idle(); closed = true; handles.remove(this);
            try { prepared.close(); } finally { counts.keySet().forEach(Resource::release); counts.clear(); }
        }
    }
    private GpuSubmission submit(GpuRecording recording) {
        check(); progress(); if (queueFailure != null) throw queueFailure;
        require(recording instanceof ComputeDevice.Recording r0 && handles.contains(recording), "Foreign or closed recording");
        Recording r = (Recording) recording; r.valid(); r.idle();
        var inUse = List.copyOf(r.counts.keySet());
        for (var res : inUse) require(!res.mapped, "A buffer of the recording is mapped; close the mapping before submitting");
        for (var res : inUse) res.recorded++;
        Job job;
        try { job = driver.submit(r.prepared, List.copyOf(r.commands)); }
        catch (RuntimeException ex) {
            for (var res : inUse) res.recorded--;
            if (ex instanceof ComputeException error && error.code() == DEVICE_LOST) queueFailure = error;
            throw ex;
        }
        Submission submission = new Submission(List.of(), List.of(), () -> { for (var res : inUse) res.recorded--; }, job);
        submission.name = r.name; submission.labels = r.labels; submission.transferBytes = transferBytes(r.commands);
        submittedCount++; r.running = submission; pending.add(submission); return submission;
    }
    private static long transferBytes(List<Command> commands) {
        long bytes = 0;
        for (var c : commands) {
            if (c instanceof Upload u) bytes += u.data().length;
            else if (c instanceof Download d) bytes += d.source().bytes();
        }
        return bytes;
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
