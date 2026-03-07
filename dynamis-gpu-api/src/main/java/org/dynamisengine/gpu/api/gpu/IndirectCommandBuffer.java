package org.dynamisengine.gpu.api.gpu;

public interface IndirectCommandBuffer {
  void writeCommand(
      int slot,
      int indexCount,
      int instanceCount,
      int firstIndex,
      int vertexOffset,
      int firstInstance);

  long bufferHandle();

  long countBufferHandle();

  int variantOffset(int variantIndex);

  int variantCapacity(int variantIndex);

  void destroy();
}
