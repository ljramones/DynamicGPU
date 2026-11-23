package org.dynamisengine.gpu.metal;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.spi.ComputeDevice;
import java.lang.foreign.*;
import java.util.*;
import static java.lang.foreign.ValueLayout.*;

/** Native Metal discovery, independent of Vulkan/MoltenVK and presentation. */
public final class MetalProvider implements GpuProvider {
    @Override public Backend backend() { return Backend.METAL; }
    @Override public List<GpuAdapter> adapters() {
        MetalNative n = MetalNative.instance();
        int count = n.integer("dg_device_count"); if (count < 0) throw n.error("device discovery");
        List<GpuAdapter> result = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            var device = n.pointer("dg_device_open", i);
            try (var arena = Arena.ofConfined()) {
                String name = n.pointer("dg_device_name", device).reinterpret(1024).getString(0);
                var values = arena.allocate(JAVA_LONG, 4); n.check("dg_device_caps", device, values);
                var size = new Dimensions((int) values.getAtIndex(JAVA_LONG, 1), (int) values.getAtIndex(JAVA_LONG, 2), (int) values.getAtIndex(JAVA_LONG, 3));
                var caps = new GpuCapabilities(values.getAtIndex(JAVA_LONG, 0), values.getAtIndex(JAVA_LONG, 0), 16, 30, 128,
                        size, size.x(), new Dimensions(Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE), Set.of("FP32", "INT32", "BOOL"), Set.of(), false);
                result.add(new Adapter(i, name, caps));
            } finally { n.release("dg_device_release", device); }
        }
        return List.copyOf(result);
    }
    private record Adapter(int index, String name, GpuCapabilities capabilities) implements GpuAdapter {
        @Override public Backend backend() { return Backend.METAL; }
        @Override public String id() { return "metal:" + index + ":" + name; }
        @Override public GpuDevice open() { return new ComputeDevice(this, new MetalDriver(index)); }
    }
}
