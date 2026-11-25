package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.spi.ComputeDevice;
import org.dynamisengine.gpu.api.compute.spi.ComputeDriver;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class DecodePipelineFallbackTest {
    @Test void cachesTheFixedTreeForOnlyTheUnsupportedTier() {
        var driver = new RecordingDriver();
        var tierFour = AttentionF16.Shape.of(4, 1, 32, 64).withKvHeads(1);
        var tierOne = AttentionF16.Shape.of(1, 1, 32, 64);
        var descriptor = new BufferDescriptor(tierFour.kBytes(), Set.of(BufferUsage.STORAGE), Placement.PRIVATE);
        try (var device = new TestAdapter(driver).open(); var attention = AttentionF16.create(device, Precision.FP32);
             var q = device.allocate(descriptor); var k = device.allocate(descriptor); var v = device.allocate(descriptor);
             var o = device.allocate(descriptor); var encoder = device.encoder()) {
            attention.encode(encoder, q, k, v, o, tierFour);
            attention.encode(encoder, q, k, v, o, tierFour);
            attention.encode(encoder, q, k, v, o, tierOne);
            assertEquals(List.of("attn_decode_wave_64_f32/4", "attn_decode_64_f32/4", "attn_decode_wave_64_f32/1"), driver.attempts);

            var tierEight = AttentionF16.Shape.of(8, 1, 32, 64).withKvHeads(1);
            var error = assertThrows(ComputeException.class, () -> attention.encode(encoder, q, k, v, o, tierEight));
            assertEquals(ComputeException.Code.NATIVE_ERROR, error.code());
            assertEquals("attn_decode_wave_64_f32/8", driver.attempts.getLast());
            try (var submission = device.queue().submit(encoder)) {
                assertTrue(submission.await(java.time.Duration.ofSeconds(1)));
            }
            assertEquals(List.of("attn_decode_64_f32", "attn_decode_64_f32", "attn_decode_wave_64_f32"), driver.dispatched);
        }
    }

    private record TestAdapter(ComputeDriver driver) implements GpuAdapter {
        public Backend backend() { return Backend.METAL; }
        public String name() { return "test Metal"; }
        public String id() { return "test:metal"; }
        public GpuDevice open() { return new ComputeDevice(this, driver); }
        public GpuCapabilities capabilities() {
            return new GpuCapabilities(1L << 28, 1L << 28, 16, 16, 128, new Dimensions(256, 256, 64), 256,
                    new Dimensions(65535, 65535, 65535), Set.of("FP16", "FP32", "INT32", "BOOL"),
                    Set.of(GpuCapabilities.SUBGROUP_ARITHMETIC), false, 32, 32, 32768);
        }
    }

    /** Only pipeline creation and command recording are exercised. */
    private static final class RecordingDriver implements ComputeDriver {
        final List<String> attempts = new ArrayList<>();
        final List<String> dispatched = new ArrayList<>();

        public Object pipeline(PipelineDescriptor d) {
            if (d.entryPoint().startsWith("attn_decode_")) {
                int tier = d.specialization().get(0);
                attempts.add(d.entryPoint() + "/" + tier);
                if (d.entryPoint().startsWith("attn_decode_wave_")) {
                    if (tier == 4) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "compiled thread limit");
                    if (tier == 8) throw new ComputeException(ComputeException.Code.NATIVE_ERROR, "compiler error");
                }
            }
            return new Object();
        }
        public Object allocate(BufferDescriptor d) { return new Object(); }
        public void releaseBuffer(Object b) {}
        public void releasePipeline(Object p) {}
        public ByteBuffer map(Object b, long offset, long bytes, Access access) { throw new UnsupportedOperationException(); }
        public void unmap(Object b, Access access) {}
        public Job submit(List<Command> commands) {
            for (var command : commands) {
                if (command instanceof Dispatch d && d.descriptor().entryPoint().startsWith("attn_decode_"))
                    dispatched.add(d.descriptor().entryPoint());
                if (command instanceof Download d) d.result().complete(new byte[Math.toIntExact(d.source().bytes())]);
            }
            return new Job() {
                public GpuSubmission.Status poll() { return GpuSubmission.Status.COMPLETE; }
                public void close() {}
            };
        }
        public void close() {}
    }
}
