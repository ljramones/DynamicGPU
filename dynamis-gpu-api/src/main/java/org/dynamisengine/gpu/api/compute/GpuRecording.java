package org.dynamisengine.gpu.api.compute;

/**
 * A command list recorded once and submitted many times ({@link GpuQueue#submit(GpuRecording)}), for work that
 * repeats with small changes, such as an LLM decode step or a per-frame compute pass. It is made by
 * {@link GpuCommandEncoder#record()}.
 * <p>
 * <b>What changes between runs:</b> a dispatch's parameter block ({@link #parameters}), a buffer binding
 * ({@link #bind}) and group counts ({@link #groups}). Each change is validated like the original dispatch. Dispatches
 * are numbered in recording order from 0.
 * <p>
 * <b>Rules.</b>
 * <ul>
 * <li>A recording holds no readbacks, so it can't contain downloads or {@code checkErrors}. Read results with a
 * separate encoder or by mapping. Uploads replay the bytes they recorded.</li>
 * <li>A recording runs once at a time: submitting it, changing it, or closing it waits for a run in flight.</li>
 * <li>It holds the buffers and pipelines it uses until it is closed, even if their handles are closed. They count
 * as in use, blocking host mapping, only while a run is in flight. Between runs the host may map an input buffer,
 * write it, and submit again.</li>
 * </ul>
 * Backends can keep their native state across runs: Vulkan keeps the command buffer, descriptor sets and staging,
 * and re-records only after a change.
 *
 * @since 1.5.0
 */
public interface GpuRecording extends AutoCloseable {
    /** The number of dispatches recorded. */
    int dispatches();
    /** Replaces dispatch {@code dispatch}'s parameter block for later runs. */
    GpuRecording parameters(int dispatch, byte[] parameters);
    /** Binds {@code buffer} at {@code slot} of dispatch {@code dispatch} for later runs. */
    GpuRecording bind(int dispatch, int slot, GpuBuffer buffer);
    /** Replaces dispatch {@code dispatch}'s group counts for later runs. */
    GpuRecording groups(int dispatch, Dimensions groups);
    @Override void close();
}
