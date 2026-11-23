package org.dynamisengine.gpu.api.compute;
import java.time.Duration;
import java.util.Optional;
public interface GpuSubmission extends AutoCloseable {
    enum Status { PENDING, COMPLETE, FAILED }
    Status status();
    Optional<ComputeException> failure();
    /** Returns false at the deadline; failed execution throws its structured failure. */
    boolean await(Duration timeout);
    @Override void close();
}
