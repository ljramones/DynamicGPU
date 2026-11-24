package org.dynamisengine.gpu.primitives;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.EnumSet;
import java.util.SplittableRandom;
import java.util.function.Function;
import org.dynamisengine.gpu.api.compute.BufferDescriptor;
import org.dynamisengine.gpu.api.compute.BufferUsage;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.Placement;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.dynamisengine.gpu.test.compute.CrossBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** The cross-backend harness on native Metal against MoltenVK: the fixed-order primitives agree bit for bit. */
@EnabledIfSystemProperty(named = "compute.metal.required", matches = "true")
@EnabledIfSystemProperty(named = "compute.vulkan.required", matches = "true")
class CrossBackendPrimitivesTest {
    GpuDevice metal, vulkan;

    @BeforeEach void open() {
        metal = new org.dynamisengine.gpu.metal.MetalProvider().adapters().getFirst().open();
        vulkan = new org.dynamisengine.gpu.vulkan.compute.VulkanProvider().adapters().getFirst().open();
    }

    @AfterEach void close() { metal.close(); vulkan.close(); }

    static float[] input(int n) {
        var r = new SplittableRandom(77);
        float[] f = new float[n];
        for (int i = 0; i < n; i++) f[i] = (float) (r.nextDouble() * 20 - 10);
        return f;
    }

    static float[] run(GpuDevice device, int outLength, Function<GpuDevice, java.util.function.BiConsumer<org.dynamisengine.gpu.api.compute.GpuCommandEncoder,
            org.dynamisengine.gpu.api.compute.GpuBuffer[]>> op, float[] x) {
        var usage = EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION);
        try (var gx = device.allocate(new BufferDescriptor(4L * x.length, usage, Placement.PRIVATE));
             var gy = device.allocate(new BufferDescriptor(4L * outLength, usage, Placement.PRIVATE)); var e = device.encoder()) {
            var b = ByteBuffer.allocate(4 * x.length).order(ByteOrder.nativeOrder());
            for (float v : x) b.putFloat(v);
            e.upload(gx, b.array());
            op.apply(device).accept(e, new org.dynamisengine.gpu.api.compute.GpuBuffer[] {gx, gy});
            var r = e.download(gy);
            try (var s = device.queue().submit(e)) { assertTrue(s.await(Duration.ofSeconds(60))); }
            var got = ByteBuffer.wrap(r.bytes()).order(ByteOrder.nativeOrder());
            float[] f = new float[outLength];
            for (int i = 0; i < outLength; i++) f[i] = got.getFloat(4 * i);
            return f;
        }
    }

    @Test void fixedOrderPrimitivesAgreeAcrossBackends() {
        int rows = 64, n = 3000;
        float[] x = input(rows * n);
        var xv = StridedView.matrix(rows, n);
        var sum = CrossBackend.run(metal, vulkan, Reduce.contract(Reduce.Op.SUM), d -> run(d, rows, dev -> (e, b) -> {
            try (var reduce = Reduce.create(dev)) { reduce.rows(e, Reduce.Op.SUM, b[0], xv, b[1], StridedView.contiguous(rows)); }
        }, x));
        var scan = CrossBackend.run(metal, vulkan, Scan.contract(Reduce.Op.SUM), d -> run(d, rows * n, dev -> (e, b) -> {
            try (var s = Scan.create(dev)) { s.rows(e, Reduce.Op.SUM, b[0], xv, b[1], xv); }
        }, x));
        var exp = CrossBackend.run(metal, vulkan, Elementwise.contract(Elementwise.Unary.EXP), d -> run(d, rows * n, dev -> (e, b) -> {
            try (var ew = Elementwise.create(dev)) { ew.unary(e, Elementwise.Unary.EXP, b[0], xv, b[1], xv); }
        }, x), i -> 2 * Math.max(4, 3 + 2 * Math.abs(x[i])) * Math.ulp((float) Math.exp(x[i])));
        for (var r : java.util.List.of(sum, scan, exp)) {
            System.out.println("cross-backend " + r);
            assertTrue(r.withinContract(), r.toString());
        }
    }
}
