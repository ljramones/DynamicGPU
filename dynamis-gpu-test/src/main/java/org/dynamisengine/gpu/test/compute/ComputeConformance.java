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
            var unsupported = new PipelineDescriptor(kernel(), "vector_add", descriptor.layout(), descriptor.localSize(), Map.of(), Set.of("FP64"), Set.of());
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
