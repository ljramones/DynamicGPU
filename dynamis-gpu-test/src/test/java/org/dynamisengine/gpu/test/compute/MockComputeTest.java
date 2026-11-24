package org.dynamisengine.gpu.test.compute;
import org.dynamisengine.gpu.api.compute.*;
import org.junit.jupiter.api.Test;
import java.time.Duration;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;
class MockComputeTest extends CapabilityTierConformance {
    @Override protected GpuProvider provider() { return new MockComputeProvider(); }
    /** The mock has no probe executable, so it reports no measured features. */
    @Override protected Boolean expectedFusedFma() { return false; }
    @Override protected KernelBundle kernel() { return new KernelBundle("vector-add", "1", Map.of(Backend.MOCK, new KernelBundle.Executable(KernelBundle.Format.MOCK, new byte[0]))); }
    @Test void pendingMappingAndEncoderCancellationReleaseStorage() {
        var driver = new MockComputeProvider.MockDriver(); driver.hold = true;
        try (var d = new MockComputeProvider(driver).adapters().getFirst().open()) {
            var b = d.allocate(new BufferDescriptor(256, EnumSet.allOf(BufferUsage.class), Placement.SHARED));
            var e = d.encoder(); var read = e.download(b); var s = d.queue().submit(e);
            assertThrows(ComputeException.class, () -> b.map(Access.READ)); b.close();
            assertEquals(0, driver.released); driver.hold = false; assertTrue(s.await(WAIT)); assertEquals(1, driver.released);
            var c = d.allocate(new BufferDescriptor(256, Set.of(BufferUsage.TRANSFER_SOURCE), Placement.SHARED));
            var abandoned = d.encoder(); var cancelled = abandoned.download(c); c.close(); abandoned.close();
            assertEquals(2, driver.released); assertEquals(ComputeException.Code.CLOSED, assertThrows(ComputeException.class, cancelled::bytes).code());
            assertEquals(256, read.bytes().length);
        }
    }
    @Test void timeoutDeferredReleaseAndFailurePoisoning() {
        var driver = new MockComputeProvider.MockDriver(); driver.hold = true;
        var d = new MockComputeProvider(driver).adapters().getFirst().open();
        var b = d.allocate(new BufferDescriptor(256, Set.of(BufferUsage.TRANSFER_DESTINATION, BufferUsage.TRANSFER_SOURCE), Placement.PRIVATE));
        var e = d.encoder(); e.fill(b, 0); var out = e.download(b); var s = d.queue().submit(e); b.close();
        assertFalse(s.await(Duration.ZERO)); assertEquals(0, driver.released);
        assertThrows(ComputeException.class, out::bytes); driver.hold = false; assertTrue(s.await(WAIT)); assertEquals(1, driver.released);
        var c = d.allocate(new BufferDescriptor(256, Set.of(BufferUsage.TRANSFER_DESTINATION), Placement.SHARED));
        var f = d.encoder(); f.fill(c, 1); driver.nextFailure = new ComputeException(ComputeException.Code.DEVICE_LOST, "injected device loss");
        var failed = d.queue().submit(f); c.close();
        assertEquals(ComputeException.Code.DEVICE_LOST, assertThrows(ComputeException.class, () -> failed.await(WAIT)).code());
        assertEquals(2, driver.released); assertThrows(ComputeException.class, d::close); d.close();
    }
}
