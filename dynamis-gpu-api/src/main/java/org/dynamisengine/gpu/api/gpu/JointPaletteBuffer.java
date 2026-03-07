package org.dynamisengine.gpu.api.gpu;

public interface JointPaletteBuffer {
  void upload(float[] jointMatrices);

  long bufferHandle();

  int jointCount();

  void destroy();
}
