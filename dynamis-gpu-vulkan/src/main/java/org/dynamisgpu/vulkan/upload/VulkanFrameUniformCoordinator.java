package org.dynamisgpu.vulkan.upload;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import org.dynamisgpu.api.gpu.StagingScheduler;
import org.dynamisgpu.api.gpu.VkCommandBuffer;

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

  public List<DirtyRange> pendingRanges() {
    return Collections.unmodifiableList(pendingRanges);
  }

  private void ensureAlive() {
    if (destroyed) {
      throw new IllegalStateException("Scheduler has been destroyed");
    }
  }

  public record DirtyRange(long bufferHandle, long offset, long size) {}
}
