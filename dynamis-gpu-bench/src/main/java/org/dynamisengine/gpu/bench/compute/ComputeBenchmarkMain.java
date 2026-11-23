package org.dynamisengine.gpu.bench.compute;

import org.dynamisengine.gpu.api.compute.*;
import java.util.*;
import java.util.function.LongSupplier;
import static org.dynamisengine.gpu.bench.compute.ComputeExampleMain.*;

/** Wall-clock phase probe. Completion latency includes driver and polling, not GPU timestamp duration. */
public final class ComputeBenchmarkMain {
    private ComputeBenchmarkMain() { }
    private static void measure(String phase, int iterations, LongSupplier operation) {
        for (int i = 0; i < 10; i++) operation.getAsLong();
        long[] times = new long[iterations]; for (int i = 0; i < iterations; i++) times[i] = operation.getAsLong(); Arrays.sort(times);
        System.out.printf(Locale.ROOT, "phase=%s median_us=%.3f p95_us=%.3f iterations=%d%n", phase, times[iterations / 2] / 1000.0, times[Math.min(iterations - 1, (int) (iterations * .95))] / 1000.0, iterations);
    }
    public static void main(String[] args) {
        if (args.length == 0) throw new IllegalArgumentException("Arguments: metal|vulkan [elements=262144] [iterations=100]");
        int n = args.length > 1 ? Integer.parseInt(args[1]) : 262144, iterations = args.length > 2 ? Integer.parseInt(args[2]) : 100;
        if (n <= 0 || iterations <= 0) throw new IllegalArgumentException("Positive elements and iterations required");
        try (var device = device(args[0], 0); var a = buffer(device, n, Placement.PRIVATE); var b = buffer(device, n, Placement.PRIVATE);
             var c = buffer(device, n, Placement.PRIVATE); var p = pipeline(device, n)) {
            diagnostics(device, n); System.out.println("clock=System.nanoTime timing=host-wall gpu-timestamps=false warmup=10");
            var groups = new Dimensions(Math.toIntExact((n + 63L) / 64), 1, 1); byte[] params = scalar(n), host = new byte[Math.multiplyExact(n, 4)];
            var hostFloats = java.nio.ByteBuffer.wrap(host).order(java.nio.ByteOrder.nativeOrder()); while (hostFloats.hasRemaining()) hostFloats.putFloat(1.5f);
            try (var e = device.encoder()) { e.fill(a, Float.floatToRawIntBits(1.5f)).fill(b, Float.floatToRawIntBits(2.25f)); await(device.queue().submit(e)); }
            measure("allocation-and-release", iterations, () -> {
                long start = System.nanoTime(); try (var ignored = buffer(device, n, Placement.PRIVATE)) { } return System.nanoTime() - start;
            });
            measure("empty-submit-call", iterations, () -> {
                try (var e = device.encoder()) { long start = System.nanoTime(); var s = device.queue().submit(e); long elapsed = System.nanoTime() - start; await(s); return elapsed; }
            });
            measure("upload-download-complete", iterations, () -> {
                try (var e = device.encoder()) { long start = System.nanoTime(); e.upload(a, host); var read = e.download(a); await(device.queue().submit(e));
                    if (read.bytes().length != host.length) throw new AssertionError(); return System.nanoTime() - start; }
            });
            measure("resident-dispatch-submit-call", iterations, () -> {
                try (var e = device.encoder()) { e.dispatch(p, Map.of(0, a, 1, b, 2, c), params, groups); long start = System.nanoTime(); var s = device.queue().submit(e); long elapsed = System.nanoTime() - start; await(s); return elapsed; }
            });
            measure("resident-dispatch-complete", iterations, () -> {
                try (var e = device.encoder()) { e.dispatch(p, Map.of(0, a, 1, b, 2, c), params, groups); long start = System.nanoTime(); await(device.queue().submit(e)); return System.nanoTime() - start; }
            });
            try (var e = device.encoder()) {
                var read = e.download(c); await(device.queue().submit(e)); var result = java.nio.ByteBuffer.wrap(read.bytes()).order(java.nio.ByteOrder.nativeOrder());
                while (result.hasRemaining()) if (result.getFloat() != 3.75f) throw new AssertionError("Vector-add validation failed");
            }
        }
    }
}
