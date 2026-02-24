package org.dynamisgpu.api.gpu;

public interface WeightBuffer {
  void upload(float[] weights);

  long bufferHandle();

  void destroy();
}
