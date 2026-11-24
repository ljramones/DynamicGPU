package org.dynamisengine.gpu.api.compute.kit;

import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.api.compute.Dimensions;
import org.dynamisengine.gpu.api.compute.GpuCapabilities;

/**
 * A linear range of groups folded into the device's group-count limits: {@code groups} is {@code (x, y, 1)} with
 * {@code x} at most the device's limit. The kernel takes {@link #gridX()} as a parameter, computes its linear group as
 * {@code group.y * gridX + group.x} ({@code dg_linear_group} in {@code dynamis/kit.slang}) and skips indices at or
 * past {@link #count()}.
 *
 * @param groups the dispatch's group counts
 * @param gridX the x extent, the kernel's stride between rows of groups
 * @param count the linear groups wanted; {@code groups} may cover up to {@code gridX - 1} more
 * @since 1.7.0
 */
public record Grid(Dimensions groups, int gridX, long count) {
    /** The grid for {@code count} groups (at least one group is dispatched; 0 dispatches one idle group). */
    public static Grid linear(GpuCapabilities caps, long count) {
        if (count < 0) throw new IllegalArgumentException("negative group count " + count);
        long maxX = caps.maxGroupCount().x();
        long x = Math.max(1, Math.min(count, maxX));
        long y = Math.max(1, (count + x - 1) / x);
        if (y > caps.maxGroupCount().y()) {
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, count + " groups exceed the device's group-count limits");
        }
        return new Grid(new Dimensions((int) x, (int) y, 1), (int) x, count);
    }

    /** The grid for {@code threads} threads in groups of {@code groupThreads}. */
    public static Grid forThreads(GpuCapabilities caps, long threads, int groupThreads) {
        if (groupThreads <= 0) throw new IllegalArgumentException("group size must be positive: " + groupThreads);
        return linear(caps, (threads + groupThreads - 1) / groupThreads);
    }
}
