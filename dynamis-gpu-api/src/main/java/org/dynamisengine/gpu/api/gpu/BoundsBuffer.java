package org.dynamisengine.gpu.api.gpu;

public interface BoundsBuffer {
  void writeBounds(int slot, float cx, float cy, float cz, float radius);

  void flush();

  long bufferHandle();

  void destroy();
}
