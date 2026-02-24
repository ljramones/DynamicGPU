package org.dynamisgpu.vulkan.buffer;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;
import org.dynamisgpu.api.gpu.InstanceDataBuffer;

public final class VulkanInstanceBatchBuffer implements InstanceDataBuffer {
  private static final AtomicLong NEXT_HANDLE = new AtomicLong(3000);

  private final long bufferHandle;
  private float[][] modelMatrices = new float[0][];
  private int[] materialIndices = new int[0];
  private int[] flags = new int[0];
  private int instanceCount;
  private boolean destroyed;

  public VulkanInstanceBatchBuffer() {
    this.bufferHandle = NEXT_HANDLE.getAndIncrement();
  }

  @Override
  public void upload(float[][] modelMatrices, int[] materialIndices, int[] flags) {
    ensureAlive();
    if (modelMatrices == null || materialIndices == null || flags == null) {
      throw new IllegalArgumentException("Upload arrays must not be null");
    }
    if (modelMatrices.length != materialIndices.length || modelMatrices.length != flags.length) {
      throw new IllegalArgumentException("All upload arrays must have the same length");
    }

    this.modelMatrices = new float[modelMatrices.length][];
    for (int i = 0; i < modelMatrices.length; i++) {
      this.modelMatrices[i] = Arrays.copyOf(modelMatrices[i], modelMatrices[i].length);
    }
    this.materialIndices = Arrays.copyOf(materialIndices, materialIndices.length);
    this.flags = Arrays.copyOf(flags, flags.length);
    this.instanceCount = modelMatrices.length;
  }

  @Override
  public long bufferHandle() {
    return bufferHandle;
  }

  @Override
  public int instanceCount() {
    return instanceCount;
  }

  @Override
  public void destroy() {
    destroyed = true;
    modelMatrices = new float[0][];
    materialIndices = new int[0];
    flags = new int[0];
    instanceCount = 0;
  }

  private void ensureAlive() {
    if (destroyed) {
      throw new IllegalStateException("Buffer has been destroyed");
    }
  }
}
