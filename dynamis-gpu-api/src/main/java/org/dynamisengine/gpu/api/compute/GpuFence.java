package org.dynamisengine.gpu.api.compute;

import java.time.Duration;

/**
 * A timeline fence: a 64-bit value that only grows, set by submissions or the host and waited on by either. It orders
 * work without the host. For example, a transfer-queue upload signals a value that a compute submission waits for.
 * <ul>
 * <li>{@link GpuCommandEncoder#waitFor} makes a submission wait, before any of its commands, until the value is at
 * least {@code v}; {@link GpuCommandEncoder#signal} sets it to {@code v} once the submission's commands complete.</li>
 * <li>On the host, {@link #value()} reads it without blocking, {@link #signal(long)} sets it, and {@link #await}
 * blocks until it reaches a value or a timeout passes.</li>
 * </ul>
 * Values must increase. A submission that waits for a value nobody signals never runs, and closing the device then
 * times out, so pair every wait with a signal. Vulkan uses timeline semaphores; Metal uses {@code MTLSharedEvent}.
 *
 * @since 1.6.0
 */
public interface GpuFence extends AutoCloseable {
    /** The current value, read without waiting. */
    long value();
    /** Sets the value from the host; it must not be below the current value. */
    void signal(long value);
    /** Waits until the value is at least {@code value}; false if {@code timeout} passed first. */
    boolean await(long value, Duration timeout);
    @Override void close();
}
