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
    /**
     * How long each dispatch ran on the GPU, in recording order, for an encoder (or recording) that asked with
     * {@link GpuCommandEncoder#profile()}. Empty otherwise, while pending, and on devices without
     * {@link GpuCapabilities#timestamps()}. Each time spans the dispatch from start to end; Metal measures at its
     * encoder boundaries, which profiling places around every dispatch.
     *
     * @since 1.5.0
     */
    default java.util.List<Duration> dispatchTimes() { return java.util.List.of(); }
    /** {@link #dispatchTimes()} with each dispatch's number and label. @since 1.6.0 */
    default java.util.List<DispatchTiming> dispatchTimings() { return java.util.List.of(); }
    /** The name its encoder was given ({@link GpuCommandEncoder#name(String)}), or empty. @since 1.6.0 */
    default String name() { return ""; }
    @Override void close();
}
