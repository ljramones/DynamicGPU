package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.primitives.Softmax.Shape;
import org.junit.jupiter.api.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.EnumSet;
import java.util.SplittableRandom;
import static org.junit.jupiter.api.Assertions.*;

/** Softmax on a real device against a double-precision reference and its documented bound, for both outputs. */
abstract class SoftmaxConformance {
    protected abstract GpuProvider provider();
    protected GpuDevice device;
    static final Duration WAIT = Duration.ofSeconds(60);

    @BeforeEach void open() { device = provider().adapters().getFirst().open(); }
    @AfterEach void close() { device.close(); }

    GpuBuffer buffer(long bytes) {
        return device.allocate(new BufferDescriptor(bytes, EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE));
    }
    static long pad4(long bytes) { return (bytes + 3) & ~3L; }

    /** Runs the softmax of random scores of spread {@code spread} and checks every element; returns the worst error / bound. */
    double check(Softmax softmax, Shape s, double spread, long seed) {
        var r = new SplittableRandom(seed); var out = softmax.output(); boolean fp32 = out == Precision.FP32;
        float[] scores = new float[(int) s.sElements()];
        for (int i = 0; i < scores.length; i++) scores[i] = (float) ((r.nextDouble() * 2 - 1) * spread);
        int yLength = (int) (pad4(s.yElements() * out.bytes()) / out.bytes());
        var init = ByteBuffer.allocate(yLength * out.bytes()).order(ByteOrder.nativeOrder());
        for (int i = 0; i < yLength; i++) { if (fp32) init.putFloat(-7f); else init.putShort(Float.floatToFloat16(-7f)); }
        ByteBuffer got;
        try (var gs = buffer(4L * scores.length); var gy = buffer((long) yLength * out.bytes()); var e = device.encoder()) {
            var sb = ByteBuffer.allocate(4 * scores.length).order(ByteOrder.nativeOrder()); for (float v : scores) sb.putFloat(v);
            e.upload(gs, sb.array()).upload(gy, init.array());
            softmax.encode(e, gs, gy, s); var read = e.download(gy);
            try (var sub = device.queue().submit(e)) { assertTrue(sub.await(WAIT), "timeout"); }
            got = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
        }
        boolean[] written = new boolean[yLength];
        double worst = 0;
        for (int z = 0; z < s.batch(); z++) {
            for (int row = 0; row < s.m(); row++) {
                int base = (int) (z * s.strideS() + (long) row * s.lds());
                double mx = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < s.n(); j++) if (s.visible(row, j)) mx = Math.max(mx, scores[base + j]);
                double sum = 0, weighted = 0;
                for (int j = 0; j < s.n(); j++) if (s.visible(row, j)) { double w = Math.exp(scores[base + j] - mx); sum += w; weighted += w * (mx - scores[base + j]); }
                double xbar = weighted / sum;
                for (int j = 0; j < s.n(); j++) {
                    int idx = (int) (z * s.strideY() + (long) row * s.ldy() + j); written[idx] = true;
                    double v = fp32 ? got.getFloat(4 * idx) : Float.float16ToFloat(got.getShort(2 * idx));
                    String where = "y[" + z + "][" + row + "][" + j + "] of " + s + " " + out;
                    if (!s.visible(row, j)) { assertEquals(0.0, v, where + " masked"); continue; }
                    double x = mx - scores[base + j], ref = Math.exp(-x) / sum;
                    double bound = (s.n() + 10 + 2 * x + 2 * xbar) * 0x1p-24 * ref + 0x1p-126; // 2^-126: devices may flush subnormals
                    if (!fp32) bound += ref * 0x1p-11 + 0x1p-25;
                    double err = Math.abs(v - ref);
                    assertTrue(err <= bound, where + " = " + v + ", reference " + ref + ", bound " + bound);
                    worst = Math.max(worst, err / bound);
                }
            }
        }
        for (int i = 0; i < yLength; i++)
            if (!written[i]) assertEquals(-7.0, fp32 ? got.getFloat(4 * i) : Float.float16ToFloat(got.getShort(2 * i)), "element " + i + " outside the rows written");
        return worst;
    }

    void eachOutput(java.util.function.Consumer<Softmax> body) {
        for (var out : Precision.values()) {
            if (out == Precision.FP16 && !device.capabilities().dataTypes().contains(GpuCapabilities.FP16)) continue;
            try (var softmax = Softmax.create(device, out)) { body.accept(softmax); }
        }
    }

    @Test void matchesTheReferenceWithinTheBound() {
        int[][] shapes = {{1, 1}, {3, 7}, {64, 257}, {33, 1000}, {8, 4096}, {2, 20000}};
        eachOutput(sm -> {
            long seed = 1;
            for (int[] d : shapes) {
                double worst = check(sm, Shape.of(d[0], d[1]), 8, seed++);
                System.out.printf("softmax %s %dx%d worst error/bound %.3g%n", sm.output(), d[0], d[1], worst);
            }
        });
    }

    @Test void largeAndTinyScores() {
        eachOutput(sm -> { check(sm, Shape.of(16, 300), 3e4, 2); check(sm, Shape.of(16, 300), 1e-3, 3); });
    }

    @Test void causalMaskAndKvCacheOffset() {
        eachOutput(sm -> {
            check(sm, Shape.of(70, 70).causal(0), 6, 4);
            check(sm, Shape.of(16, 300).causal(300 - 16), 6, 5);
            check(sm, Shape.of(5, 600).causal(3), 6, 6);
        });
    }

    @Test void batchesAndStrides() {
        eachOutput(sm -> {
            check(sm, Shape.of(32, 100).batched(6), 6, 7);
            check(sm, new Shape(10, 50, 64, 52, true, 20, 3, 10 * 64 + 9, 10 * 52 + 4), 6, 8);
        });
    }

    @Test void rejectsBadShapes() {
        assertThrows(IllegalArgumentException.class, () -> Shape.of(0, 4));
        assertThrows(IllegalArgumentException.class, () -> new Shape(2, 4, 3, 4, false, 0, 1, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> Shape.of(2, 4).causal(-1));
        assertThrows(IllegalArgumentException.class, () -> Shape.of(4, 4).withBatch(2, 16, 8));
        try (var sm = Softmax.create(device, Precision.FP32); var small = buffer(16); var ok = buffer(64); var e = device.encoder()) {
            assertThrows(IllegalArgumentException.class, () -> sm.encode(e, small, ok, Shape.of(4, 4)));
            assertThrows(IllegalArgumentException.class, () -> sm.encode(e, ok, small, Shape.of(4, 4)));
        }
    }
}
