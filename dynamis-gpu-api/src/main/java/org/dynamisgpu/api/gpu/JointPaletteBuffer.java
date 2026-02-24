package org.dynamisgpu.api.gpu;

public interface JointPaletteBuffer {
  void upload(float[] jointMatrices);

  long bufferHandle();

  int jointCount();

  void destroy();
}
