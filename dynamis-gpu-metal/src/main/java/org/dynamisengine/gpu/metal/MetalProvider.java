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
                var families = arena.allocate(JAVA_LONG, 1); n.check("dg_device_features", device, families);
                var tiers = Tiers.of(families.getAtIndex(JAVA_LONG, 0));
                var caps = new GpuCapabilities(values.getAtIndex(JAVA_LONG, 0), values.getAtIndex(JAVA_LONG, 0), 16, 30, 128,
                        size, size.x(), new Dimensions(Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE), tiers.dataTypes(),
                        tiers.shaderFeatures(), false, tiers.minSubgroupSize(), tiers.maxSubgroupSize());
                result.add(new Adapter(i, name, caps));
            } finally { n.release("dg_device_release", device); }
        }
        return List.copyOf(result);
    }
    /**
     * Data types and features by GPU family. Every Metal GPU computes with half, short and char. Apple7 (M1, A14) and
     * later and Mac2 add long and the SIMD-group functions; Metal 3 adds atomic float add in device memory. Apple GPUs
     * have 32-wide SIMD-groups. Metal has no double, and its 64-bit atomics (Apple8 and later) are min and max only,
     * so neither FP64 nor ATOMIC_INT64 is reported.
     */
    record Tiers(Set<String> dataTypes, Set<String> shaderFeatures, int minSubgroupSize, int maxSubgroupSize) {
        static final long APPLE7 = 1, APPLE8 = 2, APPLE9 = 4, MAC2 = 8, METAL3 = 16;
        static Tiers of(long families) {
            boolean apple = (families & APPLE7) != 0, simd = apple || (families & MAC2) != 0;
            Set<String> types = new TreeSet<>(List.of(GpuCapabilities.FP32, GpuCapabilities.INT32, GpuCapabilities.BOOL,
                    GpuCapabilities.FP16, GpuCapabilities.INT16, GpuCapabilities.INT8));
            Set<String> features = new TreeSet<>();
            if (simd) {
                types.add(GpuCapabilities.INT64);
                features.addAll(List.of(GpuCapabilities.SUBGROUP_BASIC, GpuCapabilities.SUBGROUP_VOTE, GpuCapabilities.SUBGROUP_ARITHMETIC,
                        GpuCapabilities.SUBGROUP_BALLOT, GpuCapabilities.SUBGROUP_SHUFFLE));
            }
            if ((families & METAL3) != 0) features.add(GpuCapabilities.ATOMIC_FLOAT32_ADD);
            int min = !simd ? 0 : apple ? 32 : 8, max = !simd ? 0 : apple ? 32 : 64;
            return new Tiers(types, features, min, max);
        }
    }
    private record Adapter(int index, String name, GpuCapabilities capabilities) implements GpuAdapter {
        @Override public Backend backend() { return Backend.METAL; }
        @Override public String id() { return "metal:" + index + ":" + name; }
        @Override public GpuDevice open() { return new ComputeDevice(this, new MetalDriver(index)); }
    }
}
