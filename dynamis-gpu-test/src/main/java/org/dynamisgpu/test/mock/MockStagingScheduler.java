package org.dynamisgpu.test.mock;

import java.util.ArrayList;
import java.util.List;
import org.dynamisgpu.api.gpu.StagingScheduler;
import org.dynamisgpu.api.gpu.VkCommandBuffer;

public final class MockStagingScheduler implements StagingScheduler {
  private final List<DirtyRange> dirtyRanges = new ArrayList<>();
  private int flushCount;
  private int resetCount;
  private boolean destroyed;

  @Override
  public void markDirty(long bufferHandle, long offset, long size) {
    dirtyRanges.add(new DirtyRange(bufferHandle, offset, size));
  }

  @Override
  public void flush(VkCommandBuffer cmd) {
    flushCount++;
    dirtyRanges.clear();
  }

  @Override
  public void reset() {
    resetCount++;
    dirtyRanges.clear();
  }

  @Override
  public void destroy() {
    destroyed = true;
  }

  public List<DirtyRange> dirtyRanges() {
    return List.copyOf(dirtyRanges);
  }

  public int flushCount() {
    return flushCount;
  }

  public int resetCount() {
    return resetCount;
  }

  public boolean destroyed() {
    return destroyed;
  }

  public record DirtyRange(long bufferHandle, long offset, long size) {}
}
