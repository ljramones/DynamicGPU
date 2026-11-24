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
     * As {@link #dispatch(GpuComputePipeline, Map, byte[], Dimensions)}, with this dispatch's own label. It is joined
     * to the encoder's current label as {@code outer/inner}; primitives label their dispatches this way
     * ({@code gemm.tiled}, {@code softmax}). @since 1.6.0
     */
    default GpuCommandEncoder dispatch(GpuComputePipeline pipeline, Map<Integer, ? extends GpuBuffer> bindings,
            byte[] parameters, Dimensions groups, String label) {
        return dispatch(pipeline, bindings, parameters, groups);
    }
    /**
     * A dispatch whose group counts the GPU reads, when the dispatch runs, from {@code args} at byte {@code offset}:
     * three unsigned 32-bit values (x, y, z). Earlier commands of the submission may write them, so work sized on
     * the GPU (culling, compaction, routing) needs no readback. The counts cannot be checked on the host: they must
     * be within the device's group-count limits, and 0 runs nothing. {@code args} needs {@link BufferUsage#INDIRECT};
     * {@code offset} must be a multiple of 4. UNSUPPORTED where the backend has no indirect dispatch. @since 1.6.0
     */
    default GpuCommandEncoder dispatchIndirect(GpuComputePipeline pipeline, Map<Integer, ? extends GpuBuffer> bindings,
            byte[] parameters, GpuBuffer args, long offset) {
        throw new ComputeException(ComputeException.Code.UNSUPPORTED, "indirect dispatch is not supported here");
    }
    /**
     * Labels the dispatches recorded after this call (until the next call; null or empty clears it), for attribution
     * in {@link GpuSubmission#dispatchTimings()}, for example {@code "prefill"} or {@code "decode"}. @since 1.6.0
     */
    default GpuCommandEncoder label(String label) { return this; }
    /** Makes the submission wait, before its first command, until {@code fence} reaches {@code value}. @since 1.6.0 */
    default GpuCommandEncoder waitFor(GpuFence fence, long value) {
        throw new ComputeException(ComputeException.Code.UNSUPPORTED, "fences are not supported here");
    }
    /** Sets {@code fence} to {@code value} once the submission's commands complete. @since 1.6.0 */
    default GpuCommandEncoder signal(GpuFence fence, long value) {
        throw new ComputeException(ComputeException.Code.UNSUPPORTED, "fences are not supported here");
    }
    /** Names the submission or recording this encoder becomes ({@link GpuSubmission#name()}). @since 1.6.0 */
    default GpuCommandEncoder name(String name) { return this; }
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
