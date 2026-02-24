package org.dynamisgpu.test.mock;

import java.util.ArrayList;
import java.util.List;
import org.dynamisgpu.api.gpu.WeightBuffer;

public final class MockWeightBuffer implements WeightBuffer {
  private final long bufferHandle;
  private final List<float[]> uploads = new ArrayList<>();
  private boolean destroyed;

  public MockWeightBuffer(long bufferHandle) {
    this.bufferHandle = bufferHandle;
  }

  @Override
  public void upload(float[] weights) {
    uploads.add(weights.clone());
  }

  @Override
  public long bufferHandle() {
    return bufferHandle;
  }

  @Override
  public void destroy() {
    destroyed = true;
  }

  public List<float[]> uploads() {
    return List.copyOf(uploads);
  }

  public boolean destroyed() {
    return destroyed;
  }
}
