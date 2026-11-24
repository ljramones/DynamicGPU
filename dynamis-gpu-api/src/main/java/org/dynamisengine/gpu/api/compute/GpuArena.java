package org.dynamisengine.gpu.api.compute;

/**
 * Transient memory: one backing buffer whose space is handed out in stack-like scopes and reused deliberately. Each
 * scope allocates buffers by bumping a pointer; closing the scope closes its buffers, and the next scope reuses the
 * same memory. Examples are one layer's scratch reused by the next, or one frame's temporaries.
 * <p>
 * <b>Why reuse is safe.</b> An arena is bound to the compute queue: its buffers are refused on any other queue, and
 * work on one queue is ordered. Each command waits for the writes of everything submitted before it (Vulkan pipeline
 * barriers cover earlier submissions; Metal tracks hazards across command buffers of a queue), so a scope's work
 * never overlaps the previous scope's. The host cannot map arena memory while any work on it is in flight.
 * <p>
 * Scopes nest: only the innermost open scope allocates, and closing a scope closes its inner scopes first. Closing
 * the arena frees the memory once submitted work using it has completed.
 *
 * @since 1.6.0
 */
public interface GpuArena extends AutoCloseable {
    /** A scope over the space after every open scope. */
    Scope scope();
    /** The backing buffer's size. */
    long capacity();
    /** Bytes allocated by open scopes. */
    long used();
    /** The most bytes ever allocated at once: what the arena needed. */
    long highWater();
    @Override void close();

    /** A run of allocations, released together. */
    interface Scope extends AutoCloseable {
        /**
         * A buffer of {@code bytes} (aligned to the device's storage offset alignment); RESOURCE_EXHAUSTED when the
         * arena is full.
         */
        GpuBuffer allocate(long bytes);
        /** Closes this scope's buffers; their memory is reused by the next scope. */
        @Override void close();
    }
}
