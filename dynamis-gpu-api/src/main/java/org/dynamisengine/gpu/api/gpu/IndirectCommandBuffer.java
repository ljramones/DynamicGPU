package org.dynamisengine.gpu.api.gpu;

/**
 * Writes and exposes indirect draw command data.
 */
public interface IndirectCommandBuffer {
  /**
   * Writes one indirect draw command into the target slot.
   *
   * @param slot destination command slot
   * @param indexCount number of indices for the draw
   * @param instanceCount number of instances for the draw
   * @param firstIndex first index offset
   * @param vertexOffset base vertex offset
   * @param firstInstance first instance offset
   */
  void writeCommand(
      int slot,
      int indexCount,
      int instanceCount,
      int firstIndex,
      int vertexOffset,
      int firstInstance);

  /**
   * Returns the indirect command buffer handle.
   *
   * @return opaque backend buffer handle
   */
  long bufferHandle();

  /**
   * Returns the counter buffer handle used for draw-count style submission.
   *
   * @return opaque backend buffer handle
   */
  long countBufferHandle();

  /**
   * Returns the starting slot offset for a variant bucket.
   *
   * @param variantIndex variant bucket index
   * @return slot offset in the command buffer
   */
  int variantOffset(int variantIndex);

  /**
   * Returns the slot capacity for a variant bucket.
   *
   * @param variantIndex variant bucket index
   * @return number of writable commands in the bucket
   */
  int variantCapacity(int variantIndex);

  /**
   * Releases backend resources owned by this buffer.
   */
  void destroy();
}
