package org.dynamisengine.gpu.api.compute;
import java.util.Map;
/** Single-use encoder. All operations declare accesses; backends insert visibility barriers. */
public interface GpuCommandEncoder extends AutoCloseable {
    GpuCommandEncoder upload(GpuBuffer destination, byte[] data);
    GpuReadback download(GpuBuffer source);
    GpuCommandEncoder copy(GpuBuffer source, GpuBuffer destination);
    GpuCommandEncoder fill(GpuBuffer destination, int pattern);
    GpuCommandEncoder dispatch(GpuComputePipeline pipeline, Map<Integer, ? extends GpuBuffer> bindings,
            byte[] parameters, Dimensions groups);
    /**
     * Records clearing a {@link KernelErrors} buffer to "no error". Record it before the dispatches that report into
     * it.
     */
    default GpuCommandEncoder resetErrors(GpuBuffer errors) {
        throw new ComputeException(ComputeException.Code.UNSUPPORTED, "kernel error buffers are not supported here");
    }
    /**
     * Records reading a {@link KernelErrors} buffer back after the preceding commands. If any kernel reported an
     * error, the submission fails with {@link ComputeException.Code#KERNEL_ERROR} and the report in its message. The
     * queue is not poisoned, and the submission's other readbacks stay readable, so results can be inspected.
     */
    default GpuCommandEncoder checkErrors(GpuBuffer errors) {
        throw new ComputeException(ComputeException.Code.UNSUPPORTED, "kernel error buffers are not supported here");
    }
    /**
     * Asks for per-dispatch GPU times ({@link GpuSubmission#dispatchTimes()}) for this encoder's submission, or every
     * run of its recording. It costs a little GPU time per dispatch: Vulkan writes a timestamp pair around each
     * dispatch; Metal gives each dispatch its own encoder. @since 1.5.0
     */
    default GpuCommandEncoder profile() { return this; }
    /**
     * Ends the encoder as a {@link GpuRecording} to submit many times, instead of submitting it once. The encoder must
     * hold no downloads or error checks. @since 1.5.0
     */
    default GpuRecording record() {
        throw new ComputeException(ComputeException.Code.UNSUPPORTED, "recordings are not supported here");
    }
    @Override void close();
}
