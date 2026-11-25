package org.dynamisengine.gpu.primitives;

import java.time.Duration;
import java.util.function.Consumer;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Whole submissions on a warmed GPU. Host encoding and end-to-end wall time are also measured. */
final class GpuBenchmark {
    record Timing(double gpuMicros, double encodeMicros, double wallMicros) { }

    static Timing measure(GpuDevice device, Consumer<GpuCommandEncoder> body) {
        int reps = 1;
        double bestGpu = Double.POSITIVE_INFINITY, bestEncode = bestGpu, bestWall = bestGpu;
        for (int round = 0, accepted = 0; accepted < 4; round++) {
            long start = System.nanoTime();
            try (var encoder = device.encoder()) {
                for (int i = 0; i < reps; i++) body.accept(encoder);
                long encoded = System.nanoTime();
                try (var sub = device.queue().submit(encoder)) {
                    assertTrue(sub.await(Duration.ofSeconds(120)));
                    long wall = System.nanoTime() - start;
                    long gpu = sub.gpuTime().orElseThrow().toNanos();
                    if (gpu < 20_000_000L) {
                        // Allow for clock ramp-up, then check the actual duration again.
                        assertTrue(reps < 8192, "GPU round too short at 8192 repetitions");
                        reps = (int) Math.min(8192, Math.ceil(reps * Math.min(8.0, 24_000_000.0 / Math.max(1, gpu))));
                    } else if (round > 0) {
                        double divisor = 1000.0 * reps;
                        bestGpu = Math.min(bestGpu, gpu / divisor);
                        bestEncode = Math.min(bestEncode, (encoded - start) / divisor);
                        bestWall = Math.min(bestWall, wall / divisor);
                        accepted++;
                    }
                }
            }
        }
        return new Timing(bestGpu, bestEncode, bestWall);
    }

    private GpuBenchmark() { }
}
