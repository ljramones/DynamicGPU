package org.dynamisengine.gpu.api.compute.kit;

/**
 * Modes kernel families honour when they compile:
 * <ul>
 * <li><b>reproducible</b> ({@code -Ddynamisgpu.reproducible=true}): choose fixed-order algorithms everywhere, such as
 * {@code GemmF16.Algorithm.TILED}, so results are the same bits on every device and match the CPU references;</li>
 * <li><b>debug</b> ({@code -Ddynamisgpu.kernels.debug=true}): compile kernels with their bounds checks, which report
 * out-of-range accesses through a {@code KernelErrors} buffer and fail the submission ({@code KERNEL_ERROR}).</li>
 * </ul>
 * The modes come from the system properties unless {@link #run} or {@link #call} binds others for a task (a
 * {@link ScopedValue}). They are read when a primitive is created, not per dispatch.
 *
 * @since 1.7.0
 */
public record KitOptions(boolean reproducible, boolean debug) {
    /** The system property for reproducible mode. */
    public static final String REPRODUCIBLE_PROPERTY = "dynamisgpu.reproducible";
    /** The system property for debug kernels. */
    public static final String DEBUG_PROPERTY = "dynamisgpu.kernels.debug";

    private static final ScopedValue<KitOptions> SCOPED = ScopedValue.newInstance();

    /** The options in force: those bound by {@link #run} or {@link #call}, else the system properties. */
    public static KitOptions current() {
        return SCOPED.isBound() ? SCOPED.get() : fromProperties();
    }

    /** The options the system properties ask for. */
    public static KitOptions fromProperties() {
        return new KitOptions(Boolean.getBoolean(REPRODUCIBLE_PROPERTY), Boolean.getBoolean(DEBUG_PROPERTY));
    }

    /** Runs {@code task} with these options in force. */
    public void run(Runnable task) { ScopedValue.where(SCOPED, this).run(task); }

    /** Calls {@code task} with these options in force. */
    public <T> T call(java.util.function.Supplier<T> task) {
        return ScopedValue.where(SCOPED, this).call(task::get);
    }
}
