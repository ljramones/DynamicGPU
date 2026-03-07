package org.dynamisengine.gpu.vulkan.upload;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;
import org.dynamisengine.gpu.api.gpu.JointPaletteBuffer;

public final class VulkanSkinnedMeshUniforms implements JointPaletteBuffer {
  private static final AtomicLong NEXT_HANDLE = new AtomicLong(5000);

  private final long bufferHandle;
  private float[] jointMatrices = new float[0];
  private int jointCount;
  private boolean destroyed;

  public VulkanSkinnedMeshUniforms() {
    this.bufferHandle = NEXT_HANDLE.getAndIncrement();
  }

  @Override
  public void upload(float[] jointMatrices) {
    ensureAlive();
    if (jointMatrices == null) {
      throw new IllegalArgumentException("jointMatrices must not be null");
    }
    this.jointMatrices = Arrays.copyOf(jointMatrices, jointMatrices.length);
    this.jointCount = jointMatrices.length / 16;
  }

  @Override
  public long bufferHandle() {
    return bufferHandle;
  }

  @Override
  public int jointCount() {
    return jointCount;
  }

  @Override
  public void destroy() {
    destroyed = true;
    jointMatrices = new float[0];
    jointCount = 0;
  }

  private void ensureAlive() {
    if (destroyed) {
      throw new IllegalStateException("Buffer has been destroyed");
    }
  }
}
