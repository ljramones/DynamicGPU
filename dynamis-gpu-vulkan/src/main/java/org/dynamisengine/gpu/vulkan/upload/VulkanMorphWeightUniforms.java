package org.dynamisengine.gpu.vulkan.upload;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;
import org.dynamisengine.gpu.api.gpu.WeightBuffer;

public final class VulkanMorphWeightUniforms implements WeightBuffer {
  private static final AtomicLong NEXT_HANDLE = new AtomicLong(4000);

  private final long bufferHandle;
  private float[] weights = new float[0];
  private boolean destroyed;

  public VulkanMorphWeightUniforms() {
    this.bufferHandle = NEXT_HANDLE.getAndIncrement();
  }

  @Override
  public void upload(float[] weights) {
    ensureAlive();
    if (weights == null) {
      throw new IllegalArgumentException("weights must not be null");
    }
    this.weights = Arrays.copyOf(weights, weights.length);
  }

  @Override
  public long bufferHandle() {
    return bufferHandle;
  }

  @Override
  public void destroy() {
    destroyed = true;
    weights = new float[0];
  }

  private void ensureAlive() {
    if (destroyed) {
      throw new IllegalStateException("Buffer has been destroyed");
    }
  }
}
