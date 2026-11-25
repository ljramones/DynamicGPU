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

    private GpuBenchmark() { }
}
