package org.dynamisengine.gpu.api.compute;
import java.time.Duration;
import java.util.Optional;
public interface GpuSubmission extends AutoCloseable {
    enum Status { PENDING, COMPLETE, FAILED }
    Status status();
    Optional<ComputeException> failure();
    /** Returns false at the deadline; failed execution throws its structured failure. */
    boolean await(Duration timeout);
    /**
     * How long the GPU took to execute this submission's commands, from the start of the first to the end of the
     * last, measured with GPU timestamps. Empty while the submission is pending, when it failed before its commands
     * ran, and on devices without {@link GpuCapabilities#timestamps()}. It excludes queueing and host work, so it is
     * the time to use for kernel throughput.
     *
     * @since 1.3.0
     */
    default Optional<Duration> gpuTime() { return Optional.empty(); }
    @Override void close();
}
