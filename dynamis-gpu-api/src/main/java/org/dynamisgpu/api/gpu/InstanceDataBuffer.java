package org.dynamisgpu.api.gpu;

public interface InstanceDataBuffer {
  void upload(float[][] modelMatrices, int[] materialIndices, int[] flags);

  long bufferHandle();

  int instanceCount();

  void destroy();
}
