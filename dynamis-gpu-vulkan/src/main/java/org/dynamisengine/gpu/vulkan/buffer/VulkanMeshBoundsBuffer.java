package org.dynamisengine.gpu.vulkan.buffer;

import java.util.concurrent.atomic.AtomicLong;
import org.dynamisengine.gpu.api.gpu.BoundsBuffer;

/**
 * In-memory Vulkan-side bounds buffer implementation used for culling inputs.
 */
public final class VulkanMeshBoundsBuffer implements BoundsBuffer {
  private static final AtomicLong NEXT_HANDLE = new AtomicLong(2000);

  private final float[][] bounds;
  private final long bufferHandle;
  private boolean dirty;
  private boolean destroyed;

  /**
   * Creates a bounds buffer with a fixed number of writable slots.
   *
   * @param capacity number of bounds entries
   */
  public VulkanMeshBoundsBuffer(int capacity) {
    if (capacity <= 0) {
      throw new IllegalArgumentException("capacity must be > 0");
    }
    this.bounds = new float[capacity][4];
    this.bufferHandle = NEXT_HANDLE.getAndIncrement();
  }

  @Override
  public void writeBounds(int slot, float cx, float cy, float cz, float radius) {
    ensureAlive();
    if (slot < 0 || slot >= bounds.length) {
      throw new IndexOutOfBoundsException("slot out of range: " + slot);
    }
    bounds[slot][0] = cx;
    bounds[slot][1] = cy;
    bounds[slot][2] = cz;
    bounds[slot][3] = radius;
    dirty = true;
  }

  @Override
  public void flush() {
    ensureAlive();
    dirty = false;
  }

  @Override
  public long bufferHandle() {
    return bufferHandle;
  }

  @Override
  public void destroy() {
    destroyed = true;
  }

  /**
   * Indicates whether unflushed bounds writes are present.
   *
   * @return true when writes are pending
   */
  public boolean isDirty() {
    return dirty;
  }

  private void ensureAlive() {
    if (destroyed) {
      throw new IllegalStateException("Buffer has been destroyed");
    }
  }
}
