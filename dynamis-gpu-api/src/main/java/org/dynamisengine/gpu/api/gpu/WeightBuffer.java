package org.dynamisengine.gpu.api.gpu;

public interface WeightBuffer {
  void upload(float[] weights);

  long bufferHandle();

  void destroy();
}
