package org.dynamisgpu.test.mock;

import java.util.ArrayList;
import java.util.List;
import org.dynamisgpu.api.gpu.BoundsBuffer;

public final class MockBoundsBuffer implements BoundsBuffer {
  private final long bufferHandle;
  private final List<BoundsWrite> writes = new ArrayList<>();
  private int flushCount;
  private boolean destroyed;

  public MockBoundsBuffer(long bufferHandle) {
    this.bufferHandle = bufferHandle;
  }

  @Override
  public void writeBounds(int slot, float cx, float cy, float cz, float radius) {
    writes.add(new BoundsWrite(slot, cx, cy, cz, radius));
  }

  @Override
  public void flush() {
    flushCount++;
  }

  @Override
  public long bufferHandle() {
    return bufferHandle;
  }

  @Override
  public void destroy() {
    destroyed = true;
  }

  public List<BoundsWrite> writes() {
    return List.copyOf(writes);
  }

  public int flushCount() {
    return flushCount;
  }

  public boolean destroyed() {
    return destroyed;
  }

  public record BoundsWrite(int slot, float cx, float cy, float cz, float radius) {}
}
