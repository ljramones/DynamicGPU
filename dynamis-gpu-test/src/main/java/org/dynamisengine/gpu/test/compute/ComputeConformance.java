package org.dynamisengine.gpu.test.compute;

import org.dynamisengine.gpu.api.compute.*;
import org.junit.jupiter.api.*;
import java.nio.*;
import java.time.Duration;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/** Shared JUnit contract tests, inherited unchanged by native hardware profiles. */
public abstract class ComputeConformance {
    protected abstract GpuProvider provider();
    protected abstract KernelBundle kernel();
    protected GpuDevice device;
    protected static final Duration WAIT = Duration.ofSeconds(20);
    @BeforeEach void open() {
        var adapters = provider().adapters(); assertFalse(adapters.isEmpty(), "Required GPU backend unavailable");
        device = adapters.getFirst().open();
        System.out.println("compute backend=" + device.adapter().backend() + " device=" + device.adapter().name() + " id=" + device.adapter().id());
    }
    @AfterEach void close() { if (device != null) device.close(); }
    protected GpuBuffer buffer(int bytes, Placement placement) {
        return device.allocate(new BufferDescriptor(bytes, EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION), placement));
    }
    protected GpuComputePipeline pipeline(int n) {
        var layout = new BindingLayout(List.of(new BindingLayout.BufferBinding(0, Access.READ, n * 4L),
                new BindingLayout.BufferBinding(1, Access.READ, n * 4L), new BindingLayout.BufferBinding(2, Access.WRITE, n * 4L)), 4);
        return device.pipeline(new PipelineDescriptor(kernel(), "vector_add", layout, new Dimensions(64, 1, 1), Map.of(), Set.of("FP32"), Set.of()));
    }
    protected static byte[] scalar(int n) { return ByteBuffer.allocate(4).order(ByteOrder.nativeOrder()).putInt(n).array(); }
    protected void await(GpuSubmission s) { assertTrue(s.await(WAIT), "GPU completion timeout"); }
    @Test void copyFillAndOrderedReadback() {
        try (var a = buffer(256, Placement.PRIVATE); var b = buffer(256, Placement.SHARED); var e = device.encoder()) {
            e.fill(a, 0x12345678); e.copy(a, b); var out = e.download(b); var s = device.queue().submit(e); await(s);
            var bytes = ByteBuffer.wrap(out.bytes()).order(ByteOrder.nativeOrder()); while (bytes.hasRemaining()) assertEquals(0x12345678, bytes.getInt());
            assertEquals(GpuSubmission.Status.COMPLETE, s.status()); s.close(); s.close();
        }
    }
    @Test void vectorAddPartialGroupsAndDependentDispatch() {
        int n = 131;
        try (var a = buffer(n * 4, Placement.SHARED); var b = buffer(n * 4, Placement.PRIVATE);
             var c = buffer(n * 4, Placement.PRIVATE); var d = buffer(n * 4, Placement.SHARED); var p = pipeline(n); var e = device.encoder()) {
            var x = ByteBuffer.allocate(n * 4).order(ByteOrder.nativeOrder()); var y = ByteBuffer.allocate(n * 4).order(ByteOrder.nativeOrder());
            for (int i = 0; i < n; i++) { x.putFloat(i * 0.25f); y.putFloat(2); }
            e.upload(a, x.array()).upload(b, y.array());
            e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(3, 1, 1));
            e.dispatch(p, Map.of(0, c, 1, b, 2, d), scalar(n), new Dimensions(3, 1, 1));
            var output = e.download(d); await(device.queue().submit(e));
            var actual = ByteBuffer.wrap(output.bytes()).order(ByteOrder.nativeOrder());
            for (int i = 0; i < n; i++) assertEquals(i * 0.25f + 4, actual.getFloat(), 1e-6f);
        }
    }
    /** The fp_strict kernel for this backend (resources compute/fp_strict.*), or null when it has none. */
    protected KernelBundle strictKernel() { return null; }
    /**
     * Whether this device rounds even unmarked arithmetic as written (no reassociation without a precise mark). True
     * for native Metal and for MoltenVK, where DynamisGPU disables fast math; a general Vulkan driver may reassociate
     * code that is not marked NoContraction, as GLSL permits.
     */
    protected boolean strictWithoutPreciseMarks() { return true; }
    /**
     * Floating point follows IEEE 754: operations marked precise round as written on every backend (no reassociation,
     * a product rounded before it is added), NaN is preserved, and where {@link #strictWithoutPreciseMarks()} holds,
     * unmarked operations round as written too.
     */
    @Test void floatingPointFollowsIeeeByDefault() {
        var kernel = strictKernel(); Assumptions.assumeTrue(kernel != null, "No strict floating-point kernel for this backend");
        float near = 1 + 0x1p-12f;
        float[] in = {near, near, -(1 + 0x1p-11f), 0.3f, 2f, 1f, 2.5f, 3f, -7.5f, Float.NaN, 1f, 1f,
            Float.POSITIVE_INFINITY, 2f, 3f, -1234.567f, 0.1f, 5f, 1e30f, 1e-30f, -1f};
        int n = in.length / 3;
        var layout = new BindingLayout(List.of(new BindingLayout.BufferBinding(0, Access.READ, n * 12L),
                new BindingLayout.BufferBinding(1, Access.WRITE, n * 16L)), 4);
        try (var p = device.pipeline(new PipelineDescriptor(kernel, "fp_strict", layout, new Dimensions(64, 1, 1), Map.of(), Set.of("FP32"), Set.of()));
             var src = buffer(n * 12, Placement.SHARED); var dst = buffer(n * 16, Placement.SHARED); var e = device.encoder()) {
            var bytes = ByteBuffer.allocate(n * 12).order(ByteOrder.nativeOrder()); for (float f : in) bytes.putFloat(f);
            e.upload(src, bytes.array()).dispatch(p, Map.of(0, src, 1, dst), scalar(n), new Dimensions(1, 1, 1));
            var output = e.download(dst); await(device.queue().submit(e));
            var out = ByteBuffer.wrap(output.bytes()).order(ByteOrder.nativeOrder());
            for (int i = 0; i < n; i++) {
                float x = in[3 * i], y = in[3 * i + 1], z = in[3 * i + 2];
                int rounded = Float.floatToIntBits((x + 12582912f) - 12582912f);
                assertEquals(rounded, Float.floatToIntBits(out.getFloat()), "precise (x + c) - c reassociated for x=" + x);
                int unmarked = Float.floatToIntBits(out.getFloat());
                if (strictWithoutPreciseMarks()) assertEquals(rounded, unmarked, "(x + c) - c reassociated without a precise mark for x=" + x);
                assertEquals(Float.floatToIntBits(x * y + z), Float.floatToIntBits(out.getFloat()), "x*y + z fused for " + x + ", " + y + ", " + z);
                assertEquals(Float.isNaN(x) ? 1f : 0f, out.getFloat(), "isnan(" + x + ")");
            }
        }
    }
    /**
     * Pooled memory: many small buffers share a few driver allocations, keep their own contents, and give their
     * memory back on close; a large buffer gets a dedicated block that is freed with it. Backends without pools
     * report {@link MemoryStats#NONE}.
     */
    @Test void poolsBufferMemory() {
        var probe = buffer(4, Placement.PRIVATE);   // a fresh device holds no blocks yet
        var before = device.memoryStats();
        if (before.equals(MemoryStats.NONE)) { probe.close(); System.out.println("memory pools: none on " + device.adapter().name()); return; }
        int count = 1000, words = 64;
        List<GpuBuffer> buffers = new ArrayList<>();
        try {
            for (int i = 0; i < count; i++) buffers.add(buffer(words * 4, Placement.PRIVATE));
            var during = device.memoryStats();
            assertTrue(during.blockAllocations() - before.blockAllocations() <= 2,
                    count + " buffers made " + (during.blockAllocations() - before.blockAllocations()) + " driver allocations");
            assertTrue(during.allocations() >= before.allocations() + count);
            List<GpuReadback> reads = new ArrayList<>();
            try (var e = device.encoder()) {
                for (int i = 0; i < count; i++) {
                    var data = ByteBuffer.allocate(words * 4).order(ByteOrder.nativeOrder());
                    for (int w = 0; w < words; w++) data.putInt(i * 1000 + w);
                    e.upload(buffers.get(i), data.array());
                }
                for (var b : buffers) reads.add(e.download(b));
                await(device.queue().submit(e));
            }
            for (int i = 0; i < count; i++) {
                var got = ByteBuffer.wrap(reads.get(i).bytes()).order(ByteOrder.nativeOrder());
                for (int w = 0; w < words; w++) assertEquals(i * 1000 + w, got.getInt(4 * w), "buffer " + i + " word " + w);
            }
        } finally { buffers.forEach(GpuBuffer::close); }
        var after = device.memoryStats();
        probe.close();
        assertEquals(before.usedBytes(), after.usedBytes(), "memory returned to the pool");
        after = device.memoryStats();
        long blocks = after.blockAllocations();
        try (var big = buffer(32 << 20, Placement.PRIVATE)) {
            assertEquals(blocks + 1, device.memoryStats().blockAllocations(), "a large buffer gets its own block");
        }
        assertEquals(after.reservedBytes(), device.memoryStats().reservedBytes(), "the dedicated block is freed with its buffer");
        var stats = device.memoryStats();
        assertTrue(stats.budgetBytes() > 0, "a pooled device reports its budget");
        if (stats.deviceUsageBytes() >= 0) assertTrue(stats.deviceUsageBytes() >= stats.reservedBytes() / 2, "usage counts the pools: " + stats);
        System.out.println("memory pools: " + count + " buffers in " + (after.blockAllocations() - before.blockAllocations())
                + " driver allocations; now " + device.memoryStats() + " on " + device.adapter().name());
    }

    /** -Dcompute.benchmark=true: the host cost of allocating and closing a small buffer, and of a small round trip. */
    @Test void allocationBenchmark() {
        Assumptions.assumeTrue(Boolean.getBoolean("compute.benchmark"), "set -Dcompute.benchmark=true");
        for (int round = 0; round < 3; round++) {
            int n = 5000; long t0 = System.nanoTime();
            for (int i = 0; i < n; i++) buffer(1024, Placement.PRIVATE).close();
            long alloc = (System.nanoTime() - t0) / n;
            int trips = 300; byte[] data = new byte[4096];
            try (var b = buffer(4096, Placement.PRIVATE)) {
                long t1 = System.nanoTime();
                for (int i = 0; i < trips; i++) {
                    try (var e = device.encoder()) { e.upload(b, data); var r = e.download(b); await(device.queue().submit(e)); r.bytes(); }
                }
                System.out.printf("allocation benchmark on %s: allocate+close %d ns, 4 KiB upload+download round trip %d us%n",
                        device.adapter().name(), alloc, (System.nanoTime() - t1) / trips / 1000);
            }
        }
    }

    /** -Dcompute.benchmark=true: where a 50-dispatch submission's time goes (record, submit, wait). */
    @Test void dispatchOverheadBenchmark() {
        Assumptions.assumeTrue(Boolean.getBoolean("compute.benchmark"), "set -Dcompute.benchmark=true");
        int n = 64, dispatches = 50, iterations = 200;
        try (var a = buffer(n * 4, Placement.PRIVATE); var b = buffer(n * 4, Placement.PRIVATE); var c = buffer(n * 4, Placement.PRIVATE); var p = pipeline(n)) {
            for (int round = 0; round < 3; round++) {
                long record = 0, submit = 0, wait = 0, gpu = 0;
                for (int it = 0; it < iterations; it++) {
                    long t0 = System.nanoTime();
                    var e = device.encoder();
                    for (int d = 0; d < dispatches; d++) e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(1, 1, 1));
                    long t1 = System.nanoTime();
                    var s = device.queue().submit(e);
                    long t2 = System.nanoTime();
                    await(s); long t3 = System.nanoTime();
                    gpu += s.gpuTime().map(java.time.Duration::toNanos).orElse(0L); s.close();
                    record += t1 - t0; submit += t2 - t1; wait += t3 - t2;
                }
                System.out.printf("dispatch overhead on %s, %d dispatches: record %d us, submit %d us, wait %d us (GPU %d us)%n", device.adapter().name(),
                        dispatches, record / iterations / 1000, submit / iterations / 1000, wait / iterations / 1000, gpu / iterations / 1000);
            }
        }
    }

    private GpuBuffer mappable(int bytes) {
        return device.allocate(new BufferDescriptor(bytes, EnumSet.of(BufferUsage.STORAGE, BufferUsage.MAP_WRITE, BufferUsage.TRANSFER_SOURCE,
                BufferUsage.TRANSFER_DESTINATION), Placement.SHARED));
    }
    private float[] read(GpuBuffer b, int n) {
        try (var e = device.encoder()) {
            var r = e.download(b); await(device.queue().submit(e));
            var bb = ByteBuffer.wrap(r.bytes()).order(ByteOrder.nativeOrder()); float[] f = new float[n];
            for (int i = 0; i < n; i++) f[i] = bb.getFloat(4 * i);
            return f;
        }
    }
    private static byte[] floats(int n, java.util.function.IntToDoubleFunction v) {
        var b = ByteBuffer.allocate(4 * n).order(ByteOrder.nativeOrder()); for (int i = 0; i < n; i++) b.putFloat((float) v.applyAsDouble(i)); return b.array();
    }

    /**
     * A recording replays: the host rewrites an input between runs; parameters, bindings and group counts change; a
     * closed input handle stays alive while the recording holds it.
     */
    @Test void recordingsReplayWithChanges() {
        int n = 256;
        try (var a = buffer(n * 4, Placement.PRIVATE); var b = mappable(n * 4); var c = buffer(n * 4, Placement.PRIVATE);
             var d = buffer(n * 4, Placement.PRIVATE); var d2 = buffer(n * 4, Placement.PRIVATE); var p = pipeline(n)) {
            GpuRecording rec;
            try (var e = device.encoder()) {
                e.upload(a, floats(n, i -> i));                                                    // replays the same bytes
                e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(n / 64, 1, 1)); // c = a + b
                e.dispatch(p, Map.of(0, c, 1, b, 2, d), scalar(n), new Dimensions(n / 64, 1, 1)); // d = c + b = a + 2b
                rec = e.record();
            }
            try (rec) {
                assertEquals(2, rec.dispatches());
                for (int run = 0; run < 5; run++) {
                    final int k = run;
                    try (var m = b.map(Access.WRITE)) { m.bytes().put(floats(n, i -> 10 * k + 0.5 * i)); }
                    await(device.queue().submit(rec));
                    float[] got = read(d, n);
                    for (int i = 0; i < n; i++) assertEquals(i + 2 * (10 * k + 0.5f * i), got[i], "run " + run + " element " + i);
                }
                // parameters: only the first 64 elements of d are recomputed
                try (var e = device.encoder()) { e.fill(d, Float.floatToRawIntBits(-1f)); await(device.queue().submit(e)); }
                rec.parameters(1, scalar(64));
                await(device.queue().submit(rec));
                float[] got = read(d, n);
                for (int i = 0; i < n; i++) assertEquals(i < 64 ? i + 2 * (40 + 0.5f * i) : -1f, got[i], "parameters, element " + i);
                // bindings and group counts: write the second dispatch to d2, over the first 128 elements
                rec.parameters(1, scalar(n)).bind(1, 2, d2).groups(1, new Dimensions(2, 1, 1));
                try (var e = device.encoder()) { e.fill(d2, Float.floatToRawIntBits(-2f)); await(device.queue().submit(e)); }
                await(device.queue().submit(rec));
                got = read(d2, n);
                for (int i = 0; i < n; i++) assertEquals(i < 128 ? i + 2 * (40 + 0.5f * i) : -2f, got[i], "rebind, element " + i);
                // lifetime: closing a handle the recording uses does not free its storage
                a.close();
                await(device.queue().submit(rec));
                assertEquals(0 + 2 * 40f, read(d2, 1)[0]);
                // rejections
                assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> rec.parameters(1, new byte[8])).code());
                assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> rec.bind(1, 7, d)).code());
                assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> rec.parameters(5, scalar(n))).code());
                assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> rec.bind(0, 2, b)).code(), "writable overlap with input b");
                try (var m = b.map(Access.WRITE)) {
                    assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> device.queue().submit(rec)).code(), "input mapped");
                }
            }
            assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> device.queue().submit(rec)).code(), "closed");
            try (var e = device.encoder()) {
                e.download(c);
                assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, e::record).code(), "downloads cannot be recorded");
            }
        }
    }

    /**
     * Profiled work reports one GPU time per dispatch, each positive and together no longer than the submission;
     * unprofiled work reports none. Recordings keep their profiling across runs.
     */
    @Test void timesEachDispatchWhenProfiled() {
        int small = 64, large = 1 << 20;
        try (var a = buffer(large * 4, Placement.PRIVATE); var b = buffer(large * 4, Placement.PRIVATE); var c = buffer(large * 4, Placement.PRIVATE);
             var p = pipeline(large)) {
            GpuSubmission s; GpuRecording rec;
            try (var e = device.encoder()) {
                e.profile().fill(a, 0).fill(b, 0);
                e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(small), new Dimensions(1, 1, 1));
                for (int i = 0; i < 3; i++) e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(large), new Dimensions(large / 64, 1, 1));
                s = device.queue().submit(e);
            }
            await(s);
            var times = s.dispatchTimes();
            System.out.println("dispatch times=" + times.stream().map(java.time.Duration::toNanos).toList() + " gpu=" + s.gpuTime().map(java.time.Duration::toNanos).orElse(-1L)
                    + " ns on " + device.adapter().name());
            if (!device.capabilities().timestamps()) { assertTrue(times.isEmpty()); return; }
            assertEquals(4, times.size());
            long sum = 0;
            for (var t : times) { assertTrue(t.toNanos() > 0, "dispatch time " + t); sum += t.toNanos(); }
            assertTrue(sum <= s.gpuTime().orElseThrow().toNanos() + 100_000, "dispatches " + sum + " ns within the submission " + s.gpuTime().orElseThrow());
            try (var e = device.encoder()) {
                e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(small), new Dimensions(1, 1, 1));
                var plain = device.queue().submit(e); await(plain);
                assertTrue(plain.dispatchTimes().isEmpty(), "unprofiled work reports no dispatch times");
            }
            try (var e = device.encoder()) {
                e.profile().dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(large), new Dimensions(large / 64, 1, 1));
                rec = e.record();
            }
            try (rec) {
                for (int run = 0; run < 2; run++) {
                    var r = device.queue().submit(rec); await(r);
                    assertEquals(1, r.dispatchTimes().size(), "recording run " + run);
                    assertTrue(r.dispatchTimes().getFirst().toNanos() > 0);
                }
            }
        }
    }

    /**
     * Observability hooks, all host-side: dispatch labels joined as outer/inner, submission and recording names,
     * recording tags, memory classes, and queue counters.
     */
    @Test void labelsNamesMemoryClassesAndQueueCounters() {
        int n = 64;
        var before = device.queue().stats();
        try (var a = device.allocate(new BufferDescriptor(n * 4, EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE, "kv"));
             var b = device.allocate(new BufferDescriptor(n * 4, EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE).withLabel("kv"));
             var c = buffer(n * 4, Placement.PRIVATE); var p = pipeline(n)) {
            var classes = device.memoryByLabel();
            assertEquals(new MemoryUse(2, 2L * n * 4), classes.get("kv"), "memory class kv: " + classes);
            GpuSubmission s;
            try (var e = device.encoder()) {
                e.name("step").profile().upload(a, new byte[n * 4]).upload(b, new byte[n * 4]);
                e.label("prefill").dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(1, 1, 1), "add");
                e.label("decode").dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(1, 1, 1));
                e.label(null).dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(1, 1, 1), "add");
                var r = e.download(c);
                s = device.queue().submit(e);
                var during = device.queue().stats();
                assertEquals(before.submitted() + 1, during.submitted());
                if (during.pending() > 0) assertTrue(during.transferBytesInFlight() >= 3L * n * 4, "uploads and the download in flight: " + during);
                await(s); r.bytes();
            }
            assertEquals("step", s.name());
            var after = device.queue().stats();
            assertEquals(before.completed() + 1, after.completed()); assertEquals(0, after.pending()); assertEquals(0, after.transferBytesInFlight());
            var timings = s.dispatchTimings();
            if (device.capabilities().timestamps()) {
                assertEquals(List.of("prefill/add", "decode", "add"), timings.stream().map(DispatchTiming::label).toList());
                assertEquals(List.of(0, 1, 2), timings.stream().map(DispatchTiming::dispatch).toList());
            } else assertTrue(timings.isEmpty());
            GpuRecording rec;
            try (var e = device.encoder()) {
                e.name("decode-step").label("decode").dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(1, 1, 1));
                rec = e.record();
            }
            try (rec) {
                assertEquals("decode-step", rec.name()); assertNull(rec.tag());
                rec.tag("prefix-42"); assertEquals("prefix-42", rec.tag());
                var run = device.queue().submit(rec); await(run); assertEquals("decode-step", run.name());
            }
        }
        assertNull(device.memoryByLabel().get("kv"), "the memory class empties when its buffers close");
    }

    /**
     * Timeline fences: host reads, signals and waits; a submission that waits for a value stays pending until the
     * host signals it; a submission's signal reaches the host; recordings take new values per run.
     */
    @Test void fencesOrderWorkWithoutTheHost() throws InterruptedException {
        GpuFence f;
        try { f = device.fence(); }
        catch (ComputeException e) { assertEquals(ComputeException.Code.UNSUPPORTED, e.code()); System.out.println("fences: none on " + device.adapter().name()); return; }
        try (f; var buf = buffer(256, Placement.PRIVATE)) {
            assertEquals(0, f.value());
            f.signal(5); assertEquals(5, f.value());
            assertTrue(f.await(5, WAIT)); assertFalse(f.await(6, Duration.ofMillis(20)));
            assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> f.signal(3)).code());
            // the GPU waits for the host
            GpuSubmission s; GpuReadback r;
            try (var e = device.encoder()) {
                e.waitFor(f, 6).fill(buf, 7); r = e.download(buf); s = device.queue().submit(e);
            }
            Thread.sleep(50);
            assertEquals(GpuSubmission.Status.PENDING, s.status(), "the submission waits for fence value 6");
            f.signal(6); await(s);
            assertEquals(7, ByteBuffer.wrap(r.bytes()).order(ByteOrder.nativeOrder()).getInt(0));
            // the GPU signals the host
            try (var e = device.encoder()) { e.fill(buf, 8).signal(f, 10); s = device.queue().submit(e); }
            assertTrue(f.await(10, WAIT), "the submission's signal reaches the host"); assertTrue(f.value() >= 10); await(s);
            // recordings take new values per run
            GpuRecording rec;
            try (var e = device.encoder()) { e.fill(buf, 9); rec = e.record(); }
            try (rec) {
                for (long v = 11; v <= 13; v++) {
                    rec.signal(f, v); var run = device.queue().submit(rec);
                    assertTrue(f.await(v, WAIT)); await(run);
                }
                rec.signal(null, 0).waitFor(f, 14);
                var run = device.queue().submit(rec);
                Thread.sleep(30); assertEquals(GpuSubmission.Status.PENDING, run.status(), "the recording waits for 14");
                f.signal(14); await(run);
            }
        }
    }

    /**
     * The transfer queue takes transfers and fills, not dispatches or recordings. With fences, a streaming pipeline
     * runs correctly: each round uploads on the transfer queue and signals; the compute queue waits, then computes.
     */
    @Test void transferQueueStreamsWithFences() {
        var transfer = device.queue(QueueKind.TRANSFER);
        assertEquals(QueueKind.TRANSFER, transfer.kind()); assertEquals(QueueKind.COMPUTE, device.queue().kind());
        int n = 1024;
        try (var a = buffer(n * 4, Placement.PRIVATE); var b = buffer(n * 4, Placement.PRIVATE); var c = buffer(n * 4, Placement.PRIVATE); var p = pipeline(n)) {
            try (var e = device.encoder()) {
                e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(n / 64, 1, 1));
                assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> transfer.submit(e)).code());
            }
            System.out.println("transfer queue dedicated=" + transfer.dedicated() + " on " + device.adapter().name());
            GpuFence f;
            try { f = device.fence(); } catch (ComputeException e) { return; }   // no fences (the mock): stop here
            try (f) {
                for (int round = 1; round <= 6; round++) {
                    final int k = round;
                    try (var up = device.encoder()) {
                        up.upload(a, floats(n, i -> i + k)).upload(b, floats(n, i -> 1000 * k)).signal(f, round);
                        transfer.submit(up);
                    }
                    GpuReadback r; GpuSubmission s;
                    try (var e = device.encoder()) {
                        e.waitFor(f, round).dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(n / 64, 1, 1));
                        r = e.download(c); s = device.queue().submit(e);
                    }
                    await(s);
                    var got = ByteBuffer.wrap(r.bytes()).order(ByteOrder.nativeOrder());
                    for (int i = 0; i < n; i++) assertEquals(i + k + 1000f * k, got.getFloat(4 * i), "round " + round + " element " + i);
                }
                assertEquals(6, transfer.stats().submitted());
            }
        }
    }

    /** -Dcompute.benchmark=true: a 64 MiB upload on the transfer queue alongside long compute, against the two in turn. */
    @Test void transferOverlapBenchmark() {
        Assumptions.assumeTrue(Boolean.getBoolean("compute.benchmark"), "set -Dcompute.benchmark=true");
        int n = 65535 * 64; byte[] data = new byte[64 << 20];   // the largest 1D grid every backend allows
        try (var a = buffer(n * 4, Placement.PRIVATE); var b = buffer(n * 4, Placement.PRIVATE); var c = buffer(n * 4, Placement.PRIVATE);
             var big = buffer(data.length, Placement.PRIVATE); var p = pipeline(n)) {
            for (int round = 0; round < 3; round++) {
                long t0 = System.nanoTime();
                try (var e = device.encoder()) { for (int i = 0; i < 40; i++) e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(n / 64, 1, 1)); e.upload(big, data); await(device.queue().submit(e)); }
                long serial = System.nanoTime() - t0;
                long t1 = System.nanoTime();
                GpuSubmission up, compute;
                try (var u = device.encoder()) { u.upload(big, data); up = device.queue(QueueKind.TRANSFER).submit(u); }
                try (var e = device.encoder()) { for (int i = 0; i < 40; i++) e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(n / 64, 1, 1)); compute = device.queue().submit(e); }
                await(compute); await(up);
                long overlapped = System.nanoTime() - t1;
                System.out.printf("transfer overlap on %s (dedicated=%b): serial %.2f ms, two queues %.2f ms%n", device.adapter().name(),
                        device.queue(QueueKind.TRANSFER).dedicated(), serial / 1e6, overlapped / 1e6);
            }
        }
    }

    private GpuBuffer indirectArgs() {
        return device.allocate(new BufferDescriptor(64, EnumSet.of(BufferUsage.INDIRECT, BufferUsage.STORAGE, BufferUsage.TRANSFER_DESTINATION,
                BufferUsage.TRANSFER_SOURCE), Placement.PRIVATE));
    }
    private static byte[] groups(int x, int y, int z) {
        return ByteBuffer.allocate(16).order(ByteOrder.nativeOrder()).putInt(x).putInt(y).putInt(z).putInt(0).array();
    }

    /**
     * Indirect dispatch: group counts uploaded by the host, written on the GPU by an earlier command, zero (nothing
     * runs) and partial; recordings take new counts between runs; bad arguments are refused.
     */
    @Test void dispatchesIndirectly() {
        int n = 512;
        try (var a = buffer(n * 4, Placement.PRIVATE); var b = buffer(n * 4, Placement.PRIVATE); var c = buffer(n * 4, Placement.PRIVATE);
             var args = indirectArgs(); var source = indirectArgs(); var p = pipeline(n)) {
            try (var e = device.encoder()) { e.upload(a, floats(n, i -> i)).upload(b, floats(n, i -> 2 * i)); await(device.queue().submit(e)); }
            GpuSubmission s; GpuReadback r;
            try (var e = device.encoder()) {
                e.upload(args.slice(0, 16), groups(n / 64, 1, 1));
                try { e.dispatchIndirect(p, Map.of(0, a, 1, b, 2, c), scalar(n), args, 0); }
                catch (ComputeException ex) { assertEquals(ComputeException.Code.UNSUPPORTED, ex.code()); System.out.println("indirect dispatch: none on " + device.adapter().name()); return; }
                r = e.download(c); s = device.queue().submit(e);
            }
            await(s);
            var got = ByteBuffer.wrap(r.bytes()).order(ByteOrder.nativeOrder());
            for (int i = 0; i < n; i++) assertEquals(3f * i, got.getFloat(4 * i), "host-written counts, element " + i);
            // counts written on the GPU by an earlier command; 2 groups = the first 128 elements
            try (var e = device.encoder()) {
                e.upload(source.slice(0, 16), groups(2, 1, 1)).fill(c, Float.floatToRawIntBits(-1f)).copy(source, args);
                e.dispatchIndirect(p, Map.of(0, a, 1, b, 2, c), scalar(n), args, 0);
                r = e.download(c); await(device.queue().submit(e));
            }
            got = ByteBuffer.wrap(r.bytes()).order(ByteOrder.nativeOrder());
            for (int i = 0; i < n; i++) assertEquals(i < 128 ? 3f * i : -1f, got.getFloat(4 * i), "GPU-written counts, element " + i);
            // zero groups: nothing runs
            try (var e = device.encoder()) {
                e.upload(args.slice(0, 16), groups(0, 1, 1)).fill(c, Float.floatToRawIntBits(-2f)).dispatchIndirect(p, Map.of(0, a, 1, b, 2, c), scalar(n), args, 0);
                r = e.download(c); await(device.queue().submit(e));
            }
            got = ByteBuffer.wrap(r.bytes()).order(ByteOrder.nativeOrder());
            for (int i = 0; i < n; i++) assertEquals(-2f, got.getFloat(4 * i), "zero groups, element " + i);
            // a recording reads whatever the counts are at each run
            GpuRecording rec;
            try (var e = device.encoder()) { e.dispatchIndirect(p, Map.of(0, a, 1, b, 2, c), scalar(n), args, 16); rec = e.record(); }
            try (rec) {
                for (int groupsX : new int[] {1, 4, 8}) {
                    try (var e = device.encoder()) {
                        byte[] at16 = new byte[32]; System.arraycopy(groups(groupsX, 1, 1), 0, at16, 16, 16);
                        e.upload(args.slice(0, 32), at16).fill(c, Float.floatToRawIntBits(-3f)); await(device.queue().submit(e));
                    }
                    await(device.queue().submit(rec));
                    float[] f = read(c, n);
                    for (int i = 0; i < n; i++) assertEquals(i < 64 * groupsX ? 3f * i : -3f, f[i], groupsX + " groups, element " + i);
                }
                assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> rec.groups(0, new Dimensions(1, 1, 1))).code(), "groups of an indirect dispatch");
            }
            try (var e = device.encoder()) {
                assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> e.dispatchIndirect(p, Map.of(0, a, 1, b, 2, c), scalar(n), args, 2)).code(), "misaligned");
                assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> e.dispatchIndirect(p, Map.of(0, a, 1, b, 2, c), scalar(n), args, 56)).code(), "past the end");
                assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> e.dispatchIndirect(p, Map.of(0, a, 1, b, 2, c), scalar(n), c, 0)).code(), "no INDIRECT usage");
            }
        }
    }

    /**
     * Transient arenas: a scope's buffers reuse the previous scope's memory, and back-to-back scopes on the compute
     * queue give correct results; nesting, capacity and the compute-queue binding are enforced.
     */
    @Test void arenasReuseMemoryInQueueOrder() {
        int n = 4096, bytes = n * 4;
        var descriptor = new BufferDescriptor(1 << 20, EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE, "scratch");
        try (var arena = device.arena(descriptor); var p = pipeline(n); var out = buffer(bytes, Placement.PRIVATE)) {
            assertEquals(1 << 20, arena.capacity());
            long firstOffset = -1;
            List<GpuSubmission> runs = new ArrayList<>(); List<GpuReadback> reads = new ArrayList<>();
            for (int layer = 0; layer < 5; layer++) {
                final int k = layer;
                try (var scope = arena.scope()) {
                    var a = scope.allocate(bytes); var b = scope.allocate(bytes); var c = scope.allocate(bytes);
                    if (layer == 0) firstOffset = a.offset(); else assertEquals(firstOffset, a.offset(), "each layer reuses the same memory");
                    try (var e = device.encoder()) {
                        e.upload(a, floats(n, i -> i * k)).upload(b, floats(n, i -> k)).dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(n / 64, 1, 1));
                        reads.add(e.download(c)); runs.add(device.queue().submit(e));   // not awaited: layers queue up
                    }
                }
            }
            for (int layer = 0; layer < 5; layer++) {
                await(runs.get(layer));
                var got = ByteBuffer.wrap(reads.get(layer).bytes()).order(ByteOrder.nativeOrder());
                for (int i = 0; i < n; i++) assertEquals((float) i * layer + layer, got.getFloat(4 * i), "layer " + layer + " element " + i);
            }
            assertEquals(0, arena.used()); assertTrue(arena.highWater() >= 3L * bytes);
            assertEquals(new MemoryUse(1, 1 << 20), device.memoryByLabel().get("scratch"), "the arena is one buffer of its memory class");
            try (var outer = arena.scope()) {
                outer.allocate(1024);
                var inner = arena.scope(); inner.allocate(1024);
                assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> outer.allocate(16)).code(), "only the innermost scope allocates");
                assertEquals(ComputeException.Code.RESOURCE_EXHAUSTED, assertThrows(ComputeException.class, () -> inner.allocate(2 << 20)).code(), "capacity");
                var x = inner.allocate(256);
                try (var e = device.encoder()) {
                    e.fill(x, 1);
                    assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> device.queue(QueueKind.TRANSFER).submit(e)).code(), "arena buffers stay on the compute queue");
                }
            } // closes inner first
            assertEquals(0, arena.used());
        }
        assertNull(device.memoryByLabel().get("scratch"));
    }

    /** Interop: native handles are consistent; pooled buffers export their block's handle with their own offsets. */
    @Test void exportsNativeHandles() {
        GpuInterop interop;
        try { interop = device.interop(); } catch (ComputeException e) { assertEquals(ComputeException.Code.UNSUPPORTED, e.code()); return; }
        var nd = interop.nativeDevice();
        assertEquals(device.adapter().backend(), nd.backend()); assertNotEquals(0, nd.device()); assertNotEquals(0, nd.queue());
        try (var a = buffer(1024, Placement.PRIVATE); var b = buffer(1024, Placement.PRIVATE)) {
            var na = interop.nativeBuffer(a); var nb = interop.nativeBuffer(b);
            assertNotEquals(0, na.handle()); assertEquals(1024, na.size());
            assertEquals(na.handle(), nb.handle(), "small buffers share a pool block");
            assertTrue(Math.abs(na.offset() - nb.offset()) >= 1024, "their ranges do not overlap: " + na + " " + nb);
            var slice = interop.nativeBuffer(a.slice(256, 512));
            assertEquals(na.offset() + 256, slice.offset()); assertEquals(512, slice.size());
            try (var f = device.fence()) { assertNotEquals(0, interop.nativeFence(f)); } catch (ComputeException e) { assertEquals(ComputeException.Code.UNSUPPORTED, e.code()); }
        }
    }

    /** Kit: a pipeline set compiles together, fails as a whole, hands out a placeholder and closes everything. */
    @Test void pipelineSetsCompileAndCloseTogether() {
        int n = 256;
        PipelineDescriptor d;
        try (var p = pipeline(n)) { d = p.descriptor(); }
        var unsupported = new PipelineDescriptor(d.kernel(), d.entryPoint(), d.layout(), d.localSize(), d.specialization(),
                d.requiredDataTypes(), Set.of("NO_SUCH_FEATURE"));
        assertThrows(ComputeException.class, () -> org.dynamisengine.gpu.api.compute.kit.PipelineSet.compile(device, List.of(d, d, unsupported)));
        var set = org.dynamisengine.gpu.api.compute.kit.PipelineSet.compile(device, List.of(d, d));
        try (var a = buffer(n * 4, Placement.PRIVATE); var b = buffer(n * 4, Placement.PRIVATE); var c = buffer(n * 4, Placement.PRIVATE)) {
            assertEquals(2, set.size()); assertEquals(4, set.placeholder().bytes()); assertSame(set.placeholder(), set.placeholder());
            try (var e = device.encoder()) {
                e.upload(a, floats(n, i -> i)).upload(b, floats(n, i -> 1));
                e.dispatch(set.get(0), Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(n / 64, 1, 1));
                e.dispatch(set.get(1), Map.of(0, c, 1, b, 2, a), scalar(n), new Dimensions(n / 64, 1, 1));
                var r = e.download(a); await(device.queue().submit(e));
                var got = ByteBuffer.wrap(r.bytes()).order(ByteOrder.nativeOrder());
                for (int i = 0; i < n; i++) assertEquals(i + 2f, got.getFloat(4 * i), "element " + i);
            }
            var pipeline = set.get(0);
            set.close(); set.close();
            assertThrows(IllegalStateException.class, () -> set.get(0));
            try (var e = device.encoder()) {
                assertEquals(ComputeException.Code.CLOSED, assertThrows(ComputeException.class,
                        () -> e.dispatch(pipeline, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(n / 64, 1, 1))).code());
            }
        }
    }

    /** -Dcompute.benchmark=true: a 50-dispatch submission encoded every time against the same work replayed. */
    @Test void replayBenchmark() {
        Assumptions.assumeTrue(Boolean.getBoolean("compute.benchmark"), "set -Dcompute.benchmark=true");
        int n = 64, dispatches = 50, iterations = 200;
        try (var a = buffer(n * 4, Placement.PRIVATE); var b = buffer(n * 4, Placement.PRIVATE); var c = buffer(n * 4, Placement.PRIVATE); var p = pipeline(n)) {
            GpuRecording rec;
            try (var e = device.encoder()) {
                for (int d = 0; d < dispatches; d++) e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(1, 1, 1));
                rec = e.record();
            }
            try (rec) {
                for (int round = 0; round < 3; round++) {
                    long t0 = System.nanoTime();
                    for (int it = 0; it < iterations; it++) {
                        try (var e = device.encoder()) {
                            for (int d = 0; d < dispatches; d++) e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(1, 1, 1));
                            await(device.queue().submit(e));
                        }
                    }
                    long encoded = (System.nanoTime() - t0) / iterations;
                    long t1 = System.nanoTime(), submitTime = 0;
                    for (int it = 0; it < iterations; it++) {
                        long s0 = System.nanoTime(); var s = device.queue().submit(rec); submitTime += System.nanoTime() - s0; await(s);
                    }
                    long replayed = (System.nanoTime() - t1) / iterations;
                    System.out.printf("replay benchmark on %s, %d dispatches: encode+submit+wait %d us, replay+wait %d us (replay submit %d us)%n",
                            device.adapter().name(), dispatches, encoded / 1000, replayed / 1000, submitTime / iterations / 1000);
                }
            }
        }
    }

    /** Where the device reports timestamps, a completed submission has a positive GPU time no longer than its wall time. */
    @Test void measuresGpuTimeWhereTimestampsAreReported() {
        int n = 1 << 20;
        try (var a = buffer(n * 4, Placement.PRIVATE); var b = buffer(n * 4, Placement.PRIVATE); var c = buffer(n * 4, Placement.PRIVATE);
             var p = pipeline(n); var e = device.encoder()) {
            e.fill(a, Float.floatToRawIntBits(1f)).fill(b, Float.floatToRawIntBits(2f));
            for (int i = 0; i < 20; i++) e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(n / 64, 1, 1));
            long start = System.nanoTime();
            try (var s = device.queue().submit(e)) {
                await(s); long wall = System.nanoTime() - start;
                var time = s.gpuTime();
                System.out.println("gpu time=" + time.map(java.time.Duration::toNanos).orElse(-1L) + " ns, wall=" + wall + " ns, timestamps="
                        + device.capabilities().timestamps() + " on " + device.adapter().name());
                if (device.capabilities().timestamps()) {
                    assertTrue(time.isPresent(), "timestamps reported but no GPU time");
                    assertTrue(time.get().toNanos() > 0, "GPU time " + time.get());
                    assertTrue(time.get().toNanos() <= wall + 1_000_000, "GPU time " + time.get().toNanos() + " ns exceeds wall time " + wall + " ns");
                    assertEquals(time, s.gpuTime(), "stable");
                } else {
                    assertTrue(time.isEmpty(), "GPU time without timestamps");
                }
            }
        }
    }
    /** Reported data types and features are consistent: the baseline types, subgroup sizes, and features' prerequisites. */
    @Test void reportsConsistentCapabilityTiers() {
        var c = device.capabilities();
        System.out.println("data types=" + new TreeSet<>(c.dataTypes()) + " features=" + new TreeSet<>(c.shaderFeatures())
                + " subgroup sizes=" + c.minSubgroupSize() + ".." + c.maxSubgroupSize() + " on " + device.adapter().name());
        assertTrue(c.dataTypes().containsAll(Set.of(GpuCapabilities.FP32, GpuCapabilities.INT32, GpuCapabilities.BOOL)));
        assertEquals(device.adapter().capabilities().dataTypes(), c.dataTypes(), "data types come from the adapter");
        boolean subgroups = c.shaderFeatures().contains(GpuCapabilities.SUBGROUP_BASIC);
        for (String f : List.of(GpuCapabilities.SUBGROUP_VOTE, GpuCapabilities.SUBGROUP_ARITHMETIC, GpuCapabilities.SUBGROUP_BALLOT, GpuCapabilities.SUBGROUP_SHUFFLE))
            if (c.shaderFeatures().contains(f)) assertTrue(subgroups, f + " without SUBGROUP_BASIC");
        if (c.shaderFeatures().contains(GpuCapabilities.ATOMIC_INT64)) assertTrue(c.dataTypes().contains(GpuCapabilities.INT64), "64-bit atomics without INT64");
        if (subgroups) {
            assertTrue(c.minSubgroupSize() >= 1 && Integer.bitCount(c.minSubgroupSize()) == 1, "min subgroup size " + c.minSubgroupSize());
            assertTrue(c.maxSubgroupSize() <= 128 && Integer.bitCount(c.maxSubgroupSize()) == 1, "max subgroup size " + c.maxSubgroupSize());
        } else {
            assertEquals(0, c.maxSubgroupSize(), "subgroup size without subgroups");
        }
    }
    /** Whether this backend's devices must report {@link GpuCapabilities#FUSED_FMA}, or null when that is unknown. */
    protected Boolean expectedFusedFma() { return null; }
    /** The device measures FUSED_FMA, and a pipeline requiring it is created only where it is reported. */
    @Test void reportsFusedFmaAsMeasured() {
        boolean reported = device.capabilities().shaderFeatures().contains(GpuCapabilities.FUSED_FMA);
        System.out.println("fused fma=" + reported + " on " + device.adapter().name());
        if (expectedFusedFma() != null) assertEquals(expectedFusedFma(), reported, "FUSED_FMA reported");
        assertEquals(reported, device.capabilities().shaderFeatures().contains(GpuCapabilities.FUSED_FMA), "measured once, stable");
        PipelineDescriptor plain; try (var p = pipeline(64)) { plain = p.descriptor(); }
        var requiring = new PipelineDescriptor(plain.kernel(), plain.entryPoint(), plain.layout(), plain.localSize(), Map.of(),
                plain.requiredDataTypes(), Set.of(GpuCapabilities.FUSED_FMA));
        if (reported) {
            try (var p = device.pipeline(requiring)) { assertTrue(p.descriptor().requiredFeatures().contains(GpuCapabilities.FUSED_FMA)); }
        } else {
            assertEquals(ComputeException.Code.UNSUPPORTED, assertThrows(ComputeException.class, () -> device.pipeline(requiring)).code());
        }
    }
    /** A reported kernel error fails only its submission: the readbacks stay readable and the queue keeps working. */
    @Test void kernelErrorsFailOnlyTheirSubmission() {
        try (var errors = device.allocate(KernelErrors.descriptor()); var data = buffer(64, Placement.PRIVATE)) {
            try (var e = device.encoder()) {
                e.resetErrors(errors).checkErrors(errors); await(device.queue().submit(e));
            }
            GpuReadback kept;
            try (var e = device.encoder()) {
                e.fill(errors, 0x4).fill(data, 7); kept = e.download(data); e.checkErrors(errors);
                var s = device.queue().submit(e);
                var error = assertThrows(ComputeException.class, () -> s.await(WAIT));
                assertEquals(ComputeException.Code.KERNEL_ERROR, error.code());
                assertTrue(error.getMessage().contains("codes 0x00000004"), error.getMessage());
                assertEquals(GpuSubmission.Status.FAILED, s.status());
                assertEquals(7, ByteBuffer.wrap(kept.bytes()).order(ByteOrder.nativeOrder()).getInt(), "readbacks stay readable");
            }
            try (var e = device.encoder()) {
                e.fill(data, 9); var read = e.download(data); await(device.queue().submit(e));
                assertEquals(9, ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder()).getInt(), "the queue keeps working");
            }
        }
    }
    /** The error_report kernel for this backend (resources compute/error_report.*), or null when it has none. */
    protected KernelBundle errorKernel() { return null; }
    /** Kernels report data errors with the shipped helpers; the report does not depend on thread order. */
    @Test void kernelsReportDataErrorsThroughCompletion() {
        var kernel = errorKernel(); Assumptions.assumeTrue(kernel != null, "No error_report kernel for this backend");
        int n = 200;
        var layout = new BindingLayout(List.of(new BindingLayout.BufferBinding(0, Access.READ, n * 4L),
                new BindingLayout.BufferBinding(1, Access.WRITE, n * 4L),
                new BindingLayout.BufferBinding(2, Access.READ_WRITE, KernelErrors.BYTES)), 4);
        try (var p = device.pipeline(new PipelineDescriptor(kernel, "error_report", layout, new Dimensions(64, 1, 1), Map.of(), Set.of("FP32", "INT32"), Set.of()));
             var x = buffer(n * 4, Placement.PRIVATE); var y = buffer(n * 4, Placement.PRIVATE); var errors = device.allocate(KernelErrors.descriptor())) {
            for (boolean bad : new boolean[] {false, true}) {
                var input = ByteBuffer.allocate(n * 4).order(ByteOrder.nativeOrder());
                for (int i = 0; i < n; i++) {
                    float v = i * 0.5f;
                    if (bad && (i == 5 || i == 70 || i == 130)) v = -1f;
                    if (bad && i == 150) v = Float.NaN;
                    input.putFloat(v);
                }
                try (var e = device.encoder()) {
                    e.upload(x, input.array()).resetErrors(errors)
                        .dispatch(p, Map.of(0, x, 1, y, 2, errors), scalar(n), new Dimensions((n + 63) / 64, 1, 1));
                    var report = e.download(errors); var out = e.download(y); e.checkErrors(errors);
                    var s = device.queue().submit(e);
                    if (!bad) { await(s); assertFalse(KernelErrors.parse(report.bytes()).any()); continue; }
                    assertEquals(ComputeException.Code.KERNEL_ERROR, assertThrows(ComputeException.class, () -> s.await(WAIT)).code());
                    var r = KernelErrors.parse(report.bytes());
                    assertEquals(new KernelErrors.Report(4 | 2 | 8, 5, 4), r, "codes OR, smallest index, count");
                    assertEquals(-1f, ByteBuffer.wrap(out.bytes()).order(ByteOrder.nativeOrder()).getFloat(5 * 4), "results stay readable");
                }
            }
        }
    }
    @Test void queueOrdersSeparateSubmissions() {
        try (var b = buffer(256, Placement.PRIVATE); var first = device.encoder(); var second = device.encoder()) {
            first.fill(b, 42); var a = device.queue().submit(first); var out = second.download(b); var s = device.queue().submit(second);
            await(s); assertEquals(GpuSubmission.Status.COMPLETE, a.status()); assertEquals(42, ByteBuffer.wrap(out.bytes()).order(ByteOrder.nativeOrder()).getInt());
        }
    }
    @Test void retainsSlicesAndClosedHandlesThroughCompletion() {
        var b = buffer(512, Placement.PRIVATE); var slice = b.slice(256, 256); b.close(); b.close();
        try (var e = device.encoder()) {
            e.fill(slice, 73); var out = e.download(slice); slice.close(); slice.close(); var s = device.queue().submit(e); await(s);
            assertEquals(73, ByteBuffer.wrap(out.bytes()).order(ByteOrder.nativeOrder()).getInt());
        }
    }
    @Test void retainsClosedPipelineAndUploadSnapshot() {
        int n = 64; var p = pipeline(n);
        try (var a = buffer(n * 4, Placement.PRIVATE); var b = buffer(n * 4, Placement.PRIVATE); var c = buffer(n * 4, Placement.PRIVATE); var e = device.encoder()) {
            byte[] data = new byte[n * 4]; e.upload(a, data); Arrays.fill(data, (byte) 99); e.fill(b, 0);
            e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(1, 1, 1)); p.close();
            var out = e.download(c); var s = device.queue().submit(e); await(s); assertArrayEquals(new byte[n * 4], out.bytes());
        } finally { p.close(); }
    }
    @Test void rejectsInvalidRangesBindingsAndAliases() {
        try (var a = buffer(512, Placement.SHARED); var b = buffer(512, Placement.PRIVATE); var p = pipeline(64); var e = device.encoder()) {
            assertThrows(ComputeException.class, () -> a.slice(Long.MAX_VALUE, 4));
            assertThrows(ComputeException.class, () -> a.slice(4, 512));
            assertThrows(ComputeException.class, () -> e.upload(a, new byte[4]));
            assertThrows(ComputeException.class, () -> e.copy(a, a));
            assertThrows(ComputeException.class, () -> e.dispatch(p, Map.of(0, a, 1, b), scalar(64), new Dimensions(1, 1, 1)));
            assertThrows(ComputeException.class, () -> e.dispatch(p, Map.of(0, a, 1, b, 2, a), scalar(64), new Dimensions(1, 1, 1)));
            try (var shortRange = a.slice(0, 4); var misaligned = a.slice(4, 256)) {
                assertThrows(ComputeException.class, () -> e.dispatch(p, Map.of(0, shortRange, 1, b, 2, a), scalar(64), new Dimensions(1, 1, 1)));
                if (device.capabilities().storageOffsetAlignment() > 4)
                    assertThrows(ComputeException.class, () -> e.dispatch(p, Map.of(0, misaligned, 1, b, 2, a), scalar(64), new Dimensions(1, 1, 1)));
            }
            assertThrows(ComputeException.class, () -> e.dispatch(p, Map.of(0, a, 1, b, 2, a), new byte[0], new Dimensions(1, 1, 1)));
        }
    }
    @Test void rejectsCrossDeviceAndClosedHandles() {
        try (var other = provider().adapters().getFirst().open(); var foreign = other.allocate(new BufferDescriptor(256, Set.of(BufferUsage.TRANSFER_SOURCE), Placement.SHARED));
             var b = buffer(256, Placement.PRIVATE); var e = device.encoder()) {
            assertThrows(ComputeException.class, () -> e.copy(foreign, b)); b.close();
            assertThrows(ComputeException.class, () -> e.fill(b, 0));
        }
    }
    @Test void mappingOwnsStorageAndExcludesGpuUses() {
        var b = device.allocate(new BufferDescriptor(256, EnumSet.allOf(BufferUsage.class), Placement.SHARED));
        try (var e = device.encoder()) {
            var mapping = b.map(Access.WRITE); mapping.bytes().putInt(0, 123);
            assertThrows(ComputeException.class, () -> e.fill(b, 0)); mapping.close(); mapping.close();
            var out = e.download(b); await(device.queue().submit(e)); assertEquals(123, ByteBuffer.wrap(out.bytes()).order(ByteOrder.nativeOrder()).getInt());
            var read = b.map(Access.READ); assertTrue(read.bytes().isReadOnly()); b.close(); read.close();
        } finally { b.close(); }
    }
    @Test void repeatedAllocateReleaseAndDeviceClose() {
        for (int i = 0; i < 24; i++) {
            var b = buffer(256, Placement.PRIVATE); var e = device.encoder(); e.fill(b, i); device.queue().submit(e); b.close();
        }
        device.close(); device.close(); assertThrows(ComputeException.class, () -> buffer(256, Placement.SHARED));
    }
    @Test void rejectsLimitsAndUnsubmittedReadback() {
        try (var a = buffer(256, Placement.PRIVATE); var b = buffer(256, Placement.PRIVATE); var c = buffer(256, Placement.PRIVATE);
             var p = pipeline(64); var e = device.encoder()) {
            var read = e.download(a);
            assertEquals(ComputeException.Code.PENDING, assertThrows(ComputeException.class, read::bytes).code());
            int limit = device.capabilities().maxGroupCount().x();
            if (limit < Integer.MAX_VALUE)
                assertEquals(ComputeException.Code.UNSUPPORTED, assertThrows(ComputeException.class,
                    () -> e.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(64), new Dimensions(limit + 1, 1, 1))).code());
            e.close(); assertEquals(ComputeException.Code.CLOSED, assertThrows(ComputeException.class, read::bytes).code());
            var descriptor = p.descriptor();
            var invalidSlots = new BindingLayout(List.of(new BindingLayout.BufferBinding(Integer.MAX_VALUE, Access.READ, 4)), 4);
            var invalidSlotPipeline = new PipelineDescriptor(kernel(), "vector_add", invalidSlots, descriptor.localSize(), Map.of(), Set.of(), Set.of());
            assertEquals(ComputeException.Code.INVALID_ARGUMENT, assertThrows(ComputeException.class, () -> device.pipeline(invalidSlotPipeline)).code());
            var unsupported = new PipelineDescriptor(kernel(), "vector_add", descriptor.layout(), descriptor.localSize(), Map.of(), Set.of("FP128"), Set.of()); // no device has it
            assertEquals(ComputeException.Code.UNSUPPORTED, assertThrows(ComputeException.class, () -> device.pipeline(unsupported)).code());
        }
    }
    @Test void alignedStorageSlicesUseTheirOwnOffsets() {
        int offset = Math.max(256, device.capabilities().storageOffsetAlignment());
        try (var a = buffer(offset + 256, Placement.PRIVATE); var b = buffer(offset + 256, Placement.SHARED); var c = buffer(offset + 256, Placement.PRIVATE);
             var sa = a.slice(offset, 256); var sb = b.slice(offset, 256); var sc = c.slice(offset, 256); var p = pipeline(64); var e = device.encoder()) {
            e.fill(sa, Float.floatToRawIntBits(1.5f)).fill(sb, Float.floatToRawIntBits(2.25f));
            e.dispatch(p, Map.of(0, sa, 1, sb, 2, sc), scalar(64), new Dimensions(1, 1, 1));
            var read = e.download(sc); await(device.queue().submit(e));
            var result = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
            while (result.hasRemaining()) assertEquals(3.75f, result.getFloat());
        }
    }
    @Test void nativeShaderErrorsKeepDiagnostics() {
        PipelineDescriptor good; try (var p = pipeline(64)) { good = p.descriptor(); }
        var bad = new PipelineDescriptor(kernel(), "missing_entry_point", good.layout(), good.localSize(), Map.of(), Set.of(), Set.of());
        var failure = assertThrows(ComputeException.class, () -> device.pipeline(bad)); assertFalse(failure.getMessage().isBlank());
    }
}
