package org.dynamisengine.gpu.api.compute;

import java.time.Duration;

/**
 * One profiled dispatch: its number in recording order, its label (see {@link GpuCommandEncoder#label(String)}),
 * and its GPU time.
 *
 * @since 1.6.0
 */
public record DispatchTiming(int dispatch, String label, Duration time) { }
