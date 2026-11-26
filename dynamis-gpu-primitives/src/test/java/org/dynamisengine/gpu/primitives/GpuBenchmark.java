package org.dynamisengine.gpu.primitives;

import java.time.Duration;
import java.util.List;
import java.util.function.Consumer;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Whole submissions on a warmed GPU. Host encoding and end-to-end wall time are also measured. */
final class GpuBenchmark {
    record Timing(double gpuMicros, double encodeMicros, double wallMicros) { }

    static Timing measure(GpuDevice device, Consumer<GpuCommandEncoder> body) {
        double bestGpu = Double.POSITIVE_INFINITY, bestEncode = bestGpu, bestWall = bestGpu;
        for (var t : rounds(device, List.of(body))[0]) {
            bestGpu = Math.min(bestGpu, t.gpuMicros());
            bestEncode = Math.min(bestEncode, t.encodeMicros());
            bestWall = Math.min(bestWall, t.wallMicros());
        }
        return new Timing(bestGpu, bestEncode, bestWall);
    }

    /** Four warmed rounds per workload, each normalized per body. Within a round all workloads record the same number of bodies. */
    static Timing[][] rounds(GpuDevice device, List<Consumer<GpuCommandEncoder>> workloads) {
        Timing[][] results = new Timing[workloads.size()][4];
        int reps = 1;
        for (int accepted = 0, round = 0; accepted < 4; round++) {
            long fastest = Long.MAX_VALUE;
            for (int offset = 0; offset < workloads.size(); offset++) {
                int index = (accepted + offset) % workloads.size(); // alternate the first workload in accepted pairs
                var body = workloads.get(index);
                long start = System.nanoTime();
                try (var encoder = device.encoder()) {
                    for (int i = 0; i < reps; i++) body.accept(encoder);
                    long encoded = System.nanoTime();
                    try (var sub = device.queue().submit(encoder)) {
                        assertTrue(sub.await(Duration.ofSeconds(120)));
                        long wall = System.nanoTime() - start;
                        long gpu = sub.gpuTime().orElseThrow().toNanos();
                        double divisor = 1000.0 * reps;
                        results[index][accepted] = new Timing(gpu / divisor, (encoded - start) / divisor, wall / divisor);
                        fastest = Math.min(fastest, gpu);
                    }
                }
            }
            if (fastest < 20_000_000L) {
                // Allow for clock ramp-up, then check every submission again.
                assertTrue(reps < 8192, "GPU round too short at 8192 repetitions");
                reps = (int) Math.min(8192, Math.ceil(reps * Math.min(8.0, 24_000_000.0 / Math.max(1, fastest))));
            } else if (round > 0) {
                accepted++;
            }
        }
        return results;
    }

    /** Statistics over paired rounds: each round's GPU, encode and wall times come from the same submission. */
    record Stats(int n, double gpuMedian, double gpuMin, double gpuP10, double gpuP90, double wallMedian, double wallMin, double encodeMedian) {
        static Stats of(List<Timing> rounds) {
            double[] g = rounds.stream().mapToDouble(Timing::gpuMicros).sorted().toArray();
            double[] w = rounds.stream().mapToDouble(Timing::wallMicros).sorted().toArray();
            double[] e = rounds.stream().mapToDouble(Timing::encodeMicros).sorted().toArray();
            return new Stats(g.length, quantile(g, 0.5), g[0], quantile(g, 0.1), quantile(g, 0.9), quantile(w, 0.5), w[0], quantile(e, 0.5));
        }
        private static double quantile(double[] sorted, double q) {
            double at = q * (sorted.length - 1);
            int lo = (int) Math.floor(at), hi = (int) Math.ceil(at);
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (at - lo);
        }
    }

    /**
     * {@code n} paired rounds of {@code body} after the clocks ramp up. Each round records enough bodies for at least
     * 20 ms of GPU time and keeps that one submission's GPU, encode and wall time, normalized per body.
     */
    static List<Timing> paired(GpuDevice device, Consumer<GpuCommandEncoder> body, int n) {
        int reps = 1;
        for (int ramp = 0; ; ramp++) {   // size the rounds; these submissions are discarded
            double gpuNanos = once(device, body, reps).gpuMicros() * 1000.0 * reps;
            if (gpuNanos >= 20e6 || reps >= 8192) break;
            reps = (int) Math.min(8192, Math.ceil(reps * Math.min(8.0, 24e6 / Math.max(1, gpuNanos))));
            assertTrue(ramp < 32, "could not size a 20 ms round");
        }
        once(device, body, reps);   // one full-size round at the final size, also discarded
        var out = new java.util.ArrayList<Timing>(n);
        for (int i = 0; i < n; i++) out.add(once(device, body, reps));
        return out;
    }

    /**
     * {@code n} paired rounds of each workload, alternating: round i runs every workload once, starting from workload
     * i mod size, all with the same repetitions (sized so the fastest takes at least 20 ms). Slow drift (clocks,
     * temperature) then lands on every workload alike. Result: per workload, its n timings.
     */
    static List<List<Timing>> alternating(GpuDevice device, List<Consumer<GpuCommandEncoder>> workloads, int n) {
        int reps = 1;
        for (int ramp = 0; ; ramp++) {
            double fastest = Double.MAX_VALUE;
            for (var w : workloads) fastest = Math.min(fastest, once(device, w, reps).gpuMicros() * 1000.0 * reps);
            if (fastest >= 20e6 || reps >= 8192) break;
            reps = (int) Math.min(8192, Math.ceil(reps * Math.min(8.0, 24e6 / Math.max(1, fastest))));
            assertTrue(ramp < 32, "could not size a 20 ms round");
        }
        var out = new java.util.ArrayList<List<Timing>>();
        for (int i = 0; i < workloads.size(); i++) out.add(new java.util.ArrayList<>());
        for (int round = 0; round < n; round++) {
            for (int j = 0; j < workloads.size(); j++) {
                int w = (round + j) % workloads.size();
                out.get(w).add(once(device, workloads.get(w), reps));
            }
        }
        return out;
    }

    /** One submission of {@code reps} bodies: its GPU, encode and wall time per body. */
    static Timing once(GpuDevice device, Consumer<GpuCommandEncoder> body, int reps) {
        long start = System.nanoTime();
        try (var encoder = device.encoder()) {
            for (int i = 0; i < reps; i++) body.accept(encoder);
            long encoded = System.nanoTime();
            try (var sub = device.queue().submit(encoder)) {
                assertTrue(sub.await(Duration.ofSeconds(120)));
                long wall = System.nanoTime() - start;
                long gpu = sub.gpuTime().orElseThrow().toNanos();
                double divisor = 1000.0 * reps;
                return new Timing(gpu / divisor, (encoded - start) / divisor, wall / divisor);
            }
        }
    }

    private GpuBenchmark() { }
}
