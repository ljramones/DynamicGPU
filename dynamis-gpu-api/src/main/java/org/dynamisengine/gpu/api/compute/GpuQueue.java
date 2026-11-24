package org.dynamisengine.gpu.api.compute;
/** One ordered compute queue. A submission failure poisons later work and is reported on drain. */
public interface GpuQueue {
    GpuSubmission submit(GpuCommandEncoder encoder);
    /** Runs a recording (see {@link GpuRecording}); waits for its previous run first. @since 1.5.0 */
    /** The queue's counters, read on the host without waiting for the device. @since 1.6.0 */
    default QueueStats stats() { return new QueueStats(0, 0, 0, 0, 0); }
    default GpuSubmission submit(GpuRecording recording) {
        throw new ComputeException(ComputeException.Code.UNSUPPORTED, "recordings are not supported here");
    }
}
