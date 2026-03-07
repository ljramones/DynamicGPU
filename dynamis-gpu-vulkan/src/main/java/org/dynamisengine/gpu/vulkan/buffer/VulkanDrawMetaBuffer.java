package org.dynamisengine.gpu.vulkan.buffer;

import java.util.Arrays;
import org.dynamisengine.gpu.api.gpu.DrawMetaWriter;

/**
 * Vulkan-oriented draw metadata staging buffer.
 */
public final class VulkanDrawMetaBuffer implements DrawMetaWriter {
  private static final int DRAW_META_FIELDS = 8;

  private final int[][] entries;
  private boolean dirty;
  private boolean destroyed;

  /**
   * Creates a draw metadata buffer with fixed capacity.
   *
   * @param capacity max draw entries
   */
  public VulkanDrawMetaBuffer(int capacity) {
    if (capacity <= 0) {
      throw new IllegalArgumentException("capacity must be > 0");
    }
    this.entries = new int[capacity][DRAW_META_FIELDS];
  }

  @Override
  public void write(
      int drawIndex,
      int jointPaletteIndex,
      int morphDeltaIndex,
      int morphWeightIndex,
      int instanceDataIndex,
      int materialIndex,
      int drawFlags,
      int meshIndex) {
    ensureAlive();
    validateIndex(drawIndex);
    entries[drawIndex][0] = jointPaletteIndex;
    entries[drawIndex][1] = morphDeltaIndex;
    entries[drawIndex][2] = morphWeightIndex;
    entries[drawIndex][3] = instanceDataIndex;
    entries[drawIndex][4] = materialIndex;
    entries[drawIndex][5] = drawFlags;
    entries[drawIndex][6] = meshIndex;
    entries[drawIndex][7] = drawIndex;
    dirty = true;
  }

  @Override
  public void flush() {
    ensureAlive();
    dirty = false;
  }

  @Override
  public void destroy() {
    destroyed = true;
  }

  @Override
  public int capacity() {
    return entries.length;
  }

  /**
   * Indicates whether unflushed metadata writes are present.
   *
   * @return true when writes are pending
   */
  public boolean isDirty() {
    return dirty;
  }

  /**
   * Returns a copy of the raw metadata fields for one draw.
   *
   * @param drawIndex target draw slot
   * @return copied draw metadata payload
   */
  public int[] read(int drawIndex) {
    validateIndex(drawIndex);
    return Arrays.copyOf(entries[drawIndex], DRAW_META_FIELDS);
  }

  private void validateIndex(int drawIndex) {
    if (drawIndex < 0 || drawIndex >= entries.length) {
      throw new IndexOutOfBoundsException("drawIndex out of range: " + drawIndex);
    }
  }

  private void ensureAlive() {
    if (destroyed) {
      throw new IllegalStateException("Buffer has been destroyed");
    }
  }
}
