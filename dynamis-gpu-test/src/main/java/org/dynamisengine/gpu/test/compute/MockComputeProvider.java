package org.dynamisengine.gpu.test.compute;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.spi.*;
import java.nio.*;
import java.util.*;

/** Deterministic asynchronous mock; no native library is loaded. Only the conformance kernel is implemented. */
public final class MockComputeProvider implements GpuProvider {
    private final MockDriver driver;
    public MockComputeProvider() { this(new MockDriver()); }
    public MockComputeProvider(MockDriver driver) { this.driver = driver; }
    @Override public Backend backend() { return Backend.MOCK; }
    @Override public List<GpuAdapter> adapters() {
        return List.of(new GpuAdapter() {
            @Override public Backend backend() { return Backend.MOCK; }
            @Override public String id() { return "mock:0"; }
            @Override public String name() { return "Deterministic compute mock"; }
            @Override public GpuCapabilities capabilities() { return new GpuCapabilities(1L << 28, 1L << 28, 16, 16, 128,
                    new Dimensions(256, 256, 64), 256, new Dimensions(65535, 65535, 65535), Set.of("FP32", "INT32", "BOOL"), Set.of(), false); }
            @Override public GpuDevice open() { return new ComputeDevice(this, driver); }
        });
    }
    public static final class MockDriver implements ComputeDriver {
        public int allocated, released, pipelinesReleased, pollsBeforeCompletion = 2;
        public boolean hold;
        public ComputeException nextFailure;
        @Override public Object allocate(BufferDescriptor d) { allocated++; return ByteBuffer.allocate(Math.toIntExact(d.bytes())).order(ByteOrder.nativeOrder()); }
        @Override public void releaseBuffer(Object b) { released++; }
        private static ByteBuffer region(Region r) { return ((ByteBuffer) r.allocation()).slice(Math.toIntExact(r.offset()), Math.toIntExact(r.bytes())).order(ByteOrder.nativeOrder()); }
        @Override public ByteBuffer map(Object b, long offset, long bytes, Access a) { return region(new Region(b, offset, bytes)); }
        @Override public void unmap(Object b, Access a) { }
        @Override public Object pipeline(PipelineDescriptor d) {
            if (d.kernel().executable(Backend.MOCK).format() != KernelBundle.Format.MOCK || !d.entryPoint().equals("vector_add"))
                throw new ComputeException(ComputeException.Code.NATIVE_ERROR, "Mock supports vector_add only");
            if (!d.specialization().isEmpty()) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "Mock specialization unsupported");
            return d;
        }
        @Override public void releasePipeline(Object p) { pipelinesReleased++; }
        @Override public Job submit(List<Command> commands) {
            ComputeException failure = nextFailure; nextFailure = null;
            return new Job() {
                int polls = pollsBeforeCompletion;
                boolean complete;
                @Override public GpuSubmission.Status poll() {
                    if (hold || polls-- > 0) return GpuSubmission.Status.PENDING;
                    if (failure != null) throw failure;
                    if (!complete) { for (var c : commands) execute(c); complete = true; }
                    return GpuSubmission.Status.COMPLETE;
                }
                @Override public void close() { }
            };
        }
        private void execute(Command command) {
            switch (command) {
                case Upload u -> region(u.destination()).put(u.data());
                case Download d -> { byte[] result = new byte[(int) d.source().bytes()]; region(d.source()).get(result); d.result().complete(result); }
                case Copy c -> region(c.destination()).put(region(c.source()));
                case Fill f -> { var b = region(f.destination()); while (b.hasRemaining()) b.putInt(f.pattern()); }
                case Dispatch d -> {
                    var buffers = new HashMap<Integer, ByteBuffer>(); for (var b : d.bindings()) buffers.put(b.slot(), region(b.region()));
                    int count = ByteBuffer.wrap(d.parameters()).order(ByteOrder.nativeOrder()).getInt();
                    long threads = d.groups().x() * (long) d.descriptor().localSize().x();
                    for (int i = 0; i < Math.min(count, threads); i++) buffers.get(2).putFloat(i * 4, buffers.get(0).getFloat(i * 4) + buffers.get(1).getFloat(i * 4));
                }
            }
        }
        @Override public void close() { }
    }
}
