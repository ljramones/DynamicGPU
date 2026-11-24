package org.dynamisengine.gpu.test.compute;

import org.dynamisengine.gpu.api.compute.*;
import org.junit.jupiter.api.*;
import java.nio.*;
import java.util.*;
import java.util.function.IntFunction;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Every optional data type and shader feature a device reports works, and every one it does not report is refused.
 * Each tier has a Slang kernel ({@code src/main/slang}, built into {@code compute/tiers} by
 * {@code scripts/build-slang-kernels.sh}) whose results are checked exactly. Where the device lacks the tier, a
 * pipeline that requires it must fail with {@link ComputeException.Code#UNSUPPORTED}.
 */
public abstract class CapabilityTierConformance extends ComputeConformance {
    /** The tier kernels; every bundle carries SPIR-V and (except FP64) Metal source. */
    protected static final KernelLibrary TIERS = KernelLibrary.load(CapabilityTierConformance.class, "/compute/tiers");
    /** Kernel name, the data type or feature it needs, and whether that is a data type. */
    protected record Tier(String kernel, String requirement, boolean dataType) { }
    protected static final List<Tier> ALL = List.of(
            new Tier("fp16_arith", GpuCapabilities.FP16, true), new Tier("int16_arith", GpuCapabilities.INT16, true),
            new Tier("int8_arith", GpuCapabilities.INT8, true), new Tier("int64_arith", GpuCapabilities.INT64, true),
            new Tier("fp64_arith", GpuCapabilities.FP64, true),
            new Tier("subgroup_arithmetic", GpuCapabilities.SUBGROUP_ARITHMETIC, false),
            new Tier("subgroup_shuffle", GpuCapabilities.SUBGROUP_SHUFFLE, false),
            new Tier("subgroup_ballot", GpuCapabilities.SUBGROUP_BALLOT, false),
            new Tier("subgroup_vote", GpuCapabilities.SUBGROUP_VOTE, false),
            new Tier("atomic_float_add", GpuCapabilities.ATOMIC_FLOAT32_ADD, false),
            new Tier("atomic_int64", GpuCapabilities.ATOMIC_INT64, false));

    /** The tiers this backend's test device must report, or empty when that is unknown. */
    protected Set<String> expectedTiers() { return Set.of(); }

    protected boolean reports(Tier t) {
        var c = device.capabilities();
        return t.dataType() ? c.dataTypes().contains(t.requirement()) : c.shaderFeatures().contains(t.requirement());
    }

    protected PipelineDescriptor descriptor(Tier t, boolean declare) {
        int buffers = TIERS.buffers(t.kernel()).size(); List<Access> accesses = new ArrayList<>();
        for (int i = 0; i < buffers; i++) accesses.add(i == buffers - 1 ? Access.READ_WRITE : Access.READ);
        Set<String> types = new TreeSet<>(Set.of(GpuCapabilities.FP32)), features = new TreeSet<>();
        if (declare) (t.dataType() ? types : features).add(t.requirement());
        return TIERS.pipeline(t.kernel(), accesses, types, features);
    }

    private static Tier tier(String kernel) { return ALL.stream().filter(t -> t.kernel().equals(kernel)).findFirst().orElseThrow(); }

    /**
     * Runs a tier kernel with {@code inputs} in its first slots and a zeroed {@code outBytes} output in its last, over
     * {@code threads} threads, and returns the output in native order; or, where the device lacks the tier, checks
     * that the pipeline is refused and returns null.
     */
    protected ByteBuffer run(String kernel, int threads, int outBytes, byte[]... inputs) {
        Tier t = tier(kernel);
        if (!reports(t)) {
            assertEquals(ComputeException.Code.UNSUPPORTED,
                    assertThrows(ComputeException.class, () -> device.pipeline(descriptor(t, true))).code(), kernel);
            System.out.println("tier " + t.requirement() + ": not reported, refused");
            return null;
        }
        int local = TIERS.localSize(kernel).x();
        List<GpuBuffer> buffers = new ArrayList<>();
        try (var p = device.pipeline(descriptor(t, true)); var e = device.encoder()) {
            Map<Integer, GpuBuffer> bindings = new HashMap<>();
            for (int i = 0; i < inputs.length; i++) {
                var b = buffer(Math.max(4, inputs[i].length), Placement.PRIVATE); buffers.add(b); bindings.put(i, b);
                e.upload(b, inputs[i]);
            }
            var out = buffer(outBytes, Placement.PRIVATE); buffers.add(out); bindings.put(inputs.length, out);
            e.fill(out, 0).dispatch(p, bindings, scalar(threads), new Dimensions((threads + local - 1) / local, 1, 1));
            var read = e.download(out); await(device.queue().submit(e));
            System.out.println("tier " + t.requirement() + ": ran " + kernel);
            return ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
        } finally { buffers.forEach(GpuBuffer::close); }
    }

    private static byte[] bytes(int count, int width, IntFunction<Long> value) {
        var b = ByteBuffer.allocate(count * width).order(ByteOrder.nativeOrder());
        for (int i = 0; i < count; i++) {
            long v = value.apply(i);
            switch (width) { case 1 -> b.put((byte) v); case 2 -> b.putShort((short) v); case 4 -> b.putInt((int) v); default -> b.putLong(v); }
        }
        return b.array();
    }

    @Test void reportsTheExpectedTiers() {
        var c = device.capabilities(); Set<String> all = new TreeSet<>(c.dataTypes()); all.addAll(c.shaderFeatures());
        assertTrue(all.containsAll(expectedTiers()), "expected " + expectedTiers() + ", reported " + all);
    }

    /** Normal halves with exponents in [-6, 6]: sums and products are exact in float, so rounding them is exact. */
    @Test void fp16ArithmeticIsCorrectlyRounded() {
        int n = 256; var r = new SplittableRandom(16);
        IntFunction<Long> half = i -> (long) ((r.nextInt(2) << 15) | ((r.nextInt(13) + 9) << 10) | r.nextInt(1024));
        byte[] a = bytes(n, 2, half), b = bytes(n, 2, half);
        var out = run("fp16_arith", n, 4 * n, a, b); if (out == null) return;
        var x = ByteBuffer.wrap(a).order(ByteOrder.nativeOrder()); var y = ByteBuffer.wrap(b).order(ByteOrder.nativeOrder());
        for (int i = 0; i < n; i++) {
            float fx = Float.float16ToFloat(x.getShort(2 * i)), fy = Float.float16ToFloat(y.getShort(2 * i));
            assertEquals(Float.floatToFloat16(fx + fy), out.getShort(4 * i), fx + " + " + fy);
            assertEquals(Float.floatToFloat16(fx * fy), out.getShort(4 * i + 2), fx + " * " + fy);
        }
    }

    @Test void int16ArithmeticWraps() {
        int n = 256; var r = new SplittableRandom(17);
        byte[] a = bytes(n, 2, i -> (long) r.nextInt()), b = bytes(n, 2, i -> (long) r.nextInt());
        var out = run("int16_arith", n, 2 * n, a, b); if (out == null) return;
        var x = ByteBuffer.wrap(a).order(ByteOrder.nativeOrder()); var y = ByteBuffer.wrap(b).order(ByteOrder.nativeOrder());
        for (int i = 0; i < n; i++) {
            short p = x.getShort(2 * i), q = y.getShort(2 * i);
            assertEquals((short) (p * q + (p ^ q)), out.getShort(2 * i), p + ", " + q);
        }
    }

    @Test void int8ArithmeticWraps() {
        int n = 256; var r = new SplittableRandom(18);
        byte[] a = bytes(n, 1, i -> (long) r.nextInt()), b = bytes(n, 1, i -> (long) r.nextInt());
        var out = run("int8_arith", n, n, a, b); if (out == null) return;
        for (int i = 0; i < n; i++) assertEquals((byte) (a[i] * b[i] + (a[i] ^ b[i])), out.get(i), a[i] + ", " + b[i]);
    }

    @Test void int64ArithmeticWraps() {
        int n = 256; var r = new SplittableRandom(19);
        byte[] a = bytes(n, 8, i -> r.nextLong()), b = bytes(n, 8, i -> r.nextLong());
        var out = run("int64_arith", n, 8 * n, a, b); if (out == null) return;
        var x = ByteBuffer.wrap(a).order(ByteOrder.nativeOrder()); var y = ByteBuffer.wrap(b).order(ByteOrder.nativeOrder());
        for (int i = 0; i < n; i++) {
            long p = x.getLong(8 * i), q = y.getLong(8 * i);
            assertEquals((p * q + (p >>> 7)) ^ q, out.getLong(8 * i), p + ", " + q);
        }
    }

    @Test void fp64ArithmeticIsCorrectlyRounded() {
        int n = 256; var r = new SplittableRandom(20);
        IntFunction<Long> dbl = i -> Double.doubleToRawLongBits((r.nextBoolean() ? 1 : -1) * (1 + r.nextDouble()) * Math.scalb(1.0, r.nextInt(-100, 100)));
        byte[] a = bytes(n, 8, dbl), b = bytes(n, 8, dbl);
        var out = run("fp64_arith", n, 16 * n, a, b); if (out == null) return;
        var x = ByteBuffer.wrap(a).order(ByteOrder.nativeOrder()); var y = ByteBuffer.wrap(b).order(ByteOrder.nativeOrder());
        for (int i = 0; i < n; i++) {
            double p = x.getDouble(8 * i), q = y.getDouble(8 * i);
            assertEquals(Double.doubleToRawLongBits(p + q), Double.doubleToRawLongBits(out.getDouble(16 * i)), p + " + " + q);
            assertEquals(Double.doubleToRawLongBits(p * q), Double.doubleToRawLongBits(out.getDouble(16 * i + 8)), p + " * " + q);
        }
    }

    /** Groups threads by a per-thread subgroup key; every subgroup lies inside one 128-thread workgroup. */
    private static Map<Integer, List<Integer>> subgroups(int n, IntFunction<Integer> key) {
        Map<Integer, List<Integer>> m = new TreeMap<>();
        for (int t = 0; t < n; t++) m.computeIfAbsent(key.apply(t), k -> new ArrayList<>()).add(t);
        m.forEach((k, members) -> assertEquals(1, members.stream().map(t -> t / 128).distinct().count(), "subgroup " + k + " spans workgroups"));
        return m;
    }

    private void checkLaneCount(int count) {
        var c = device.capabilities();
        assertTrue(count >= c.minSubgroupSize() && count <= c.maxSubgroupSize(), "lane count " + count + " outside "
                + c.minSubgroupSize() + ".." + c.maxSubgroupSize());
    }

    @Test void subgroupArithmeticReducesAndScansEachSubgroup() {
        int n = 512; var out = run("subgroup_arithmetic", n, 20 * n); if (out == null) return;
        java.util.function.IntUnaryOperator word = i -> out.getInt(4 * i);
        for (var members : subgroups(n, t -> word.applyAsInt(5 * t + 3)).values()) {
            int count = word.applyAsInt(5 * members.getFirst()); checkLaneCount(count);
            assertEquals(count, members.size(), "full subgroups in 128-thread groups");
            int sum = members.stream().mapToInt(t -> t + 1).sum();
            Set<Integer> lanes = new TreeSet<>();
            for (int t : members) {
                assertEquals(count, word.applyAsInt(5 * t)); int lane = word.applyAsInt(5 * t + 1); assertTrue(lane < count); lanes.add(lane);
                assertEquals(sum, word.applyAsInt(5 * t + 2), "sum"); assertEquals(members.getFirst(), word.applyAsInt(5 * t + 3), "min");
                assertEquals(lane, word.applyAsInt(5 * t + 4), "exclusive prefix count is the lane index in a full subgroup");
            }
            assertEquals(count, lanes.size(), "distinct lanes");
        }
    }

    @Test void subgroupShuffleReadsTheNamedLane() {
        int n = 512; var out = run("subgroup_shuffle", n, 12 * n); if (out == null) return;
        java.util.function.IntUnaryOperator word = i -> out.getInt(4 * i);
        for (int t = 0; t < n; t++) {
            int count = word.applyAsInt(3 * t), lane = word.applyAsInt(3 * t + 1), next = word.applyAsInt(3 * t + 2); checkLaneCount(count);
            assertTrue(next >= 0 && next < n && next / 128 == t / 128, "thread " + t + " read " + next);
            assertEquals(count, word.applyAsInt(3 * next)); assertEquals((lane + 1) % count, word.applyAsInt(3 * next + 1), "lane of " + next);
        }
    }

    @Test void subgroupBallotCountsAndBroadcasts() {
        int n = 512; var out = run("subgroup_ballot", n, 12 * n); if (out == null) return;
        java.util.function.IntUnaryOperator word = i -> out.getInt(4 * i);
        for (var members : subgroups(n, t -> word.applyAsInt(3 * t + 2)).values()) {
            long multiples = members.stream().filter(t -> word.applyAsInt(3 * t) % 3 == 0).count();
            int lane0 = members.stream().filter(t -> word.applyAsInt(3 * t) == 0).findFirst().orElseThrow();
            for (int t : members) {
                assertEquals((int) multiples, word.applyAsInt(3 * t + 1), "ballot count"); assertEquals(lane0, word.applyAsInt(3 * t + 2), "first lane");
            }
        }
    }

    @Test void subgroupVotesAgree() {
        int n = 512; var out = run("subgroup_vote", n, 4 * n); if (out == null) return;
        boolean single = device.capabilities().minSubgroupSize() == 1;
        for (int t = 0; t < n; t++) {
            int bits = out.getInt(4 * t);
            assertTrue(bits == 3 || single && bits == 7, "thread " + t + " votes " + Integer.toBinaryString(bits));
        }
    }

    @Test void atomicFloatAddSumsExactly() {
        int n = 1000; var out = run("atomic_float_add", n, 32); if (out == null) return;
        assertEquals(n, out.getFloat(0));
        for (int k = 0; k < 7; k++) { int kk = k; assertEquals(0.25f * java.util.stream.IntStream.range(0, n).filter(t -> t % 7 == kk).count(), out.getFloat(4 + 4 * k)); }
    }

    @Test void atomicInt64AddsAndMaxes() {
        int n = 1000; var out = run("atomic_int64", n, 16); if (out == null) return;
        assertEquals((long) n * (n - 1) / 2 + ((long) n << 40), out.getLong(0)); assertEquals((long) (n - 1) << 33, out.getLong(8));
    }
}
