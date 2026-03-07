package org.dynamisengine.gpu.vulkan.upload;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import org.dynamisengine.gpu.api.gpu.StagingScheduler;
import org.dynamisengine.gpu.api.gpu.VkCommandBuffer;

/**
 * Tracks frame-local dirty ranges and clears them once flushed.
 */
public final class VulkanFrameUniformCoordinator implements StagingScheduler {
  private final List<DirtyRange> pendingRanges = new ArrayList<>();
  private boolean destroyed;

  @Override
  public void markDirty(long bufferHandle, long offset, long size) {
    ensureAlive();
    if (size < 0) {
      throw new IllegalArgumentException("size must be >= 0");
    }
    pendingRanges.add(new DirtyRange(bufferHandle, offset, size));
  }

  @Override
  public void flush(VkCommandBuffer cmd) {
    ensureAlive();
    Objects.requireNonNull(cmd, "cmd");
    pendingRanges.clear();
  }

  @Override
  public void reset() {
    ensureAlive();
    pendingRanges.clear();
  }

  @Override
  public void destroy() {
    destroyed = true;
    pendingRanges.clear();
  }

  /**
   * Returns a read-only view of currently pending dirty ranges.
   *
   * @return immutable pending-range list
   */
  public List<DirtyRange> pendingRanges() {
    return Collections.unmodifiableList(pendingRanges);
  }

  private void ensureAlive() {
    if (destroyed) {
      throw new IllegalStateException("Scheduler has been destroyed");
    }
  }

  /**
   * Immutable dirty-range descriptor.
   *
   * @param bufferHandle target buffer handle
   * @param offset byte offset
   * @param size byte size
   */
  public record DirtyRange(long bufferHandle, long offset, long size) {}
}
