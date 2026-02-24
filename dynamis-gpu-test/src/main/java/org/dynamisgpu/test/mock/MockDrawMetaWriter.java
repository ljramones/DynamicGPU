package org.dynamisgpu.test.mock;

import java.util.ArrayList;
import java.util.List;
import org.dynamisgpu.api.gpu.DrawMetaWriter;

public final class MockDrawMetaWriter implements DrawMetaWriter {
  private final int capacity;
  private final List<WriteCall> writes = new ArrayList<>();
  private int flushCount;
  private boolean destroyed;

  public MockDrawMetaWriter(int capacity) {
    this.capacity = capacity;
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
    writes.add(
        new WriteCall(
            drawIndex,
            jointPaletteIndex,
            morphDeltaIndex,
            morphWeightIndex,
            instanceDataIndex,
            materialIndex,
            drawFlags,
            meshIndex));
  }

  @Override
  public void flush() {
    flushCount++;
  }

  @Override
  public void destroy() {
    destroyed = true;
  }

  @Override
  public int capacity() {
    return capacity;
  }

  public List<WriteCall> writes() {
    return List.copyOf(writes);
  }

  public int flushCount() {
    return flushCount;
  }

  public boolean destroyed() {
    return destroyed;
  }

  public record WriteCall(
      int drawIndex,
      int jointPaletteIndex,
      int morphDeltaIndex,
      int morphWeightIndex,
      int instanceDataIndex,
      int materialIndex,
      int drawFlags,
      int meshIndex) {}
}
