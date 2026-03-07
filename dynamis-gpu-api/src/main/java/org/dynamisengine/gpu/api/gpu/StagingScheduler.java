package org.dynamisengine.gpu.api.gpu;

/**
 * Tracks dirty ranges and emits required transfer commands.
 */
public interface StagingScheduler {
  /**
   * Marks a buffer range as dirty and requiring staging/upload.
   *
   * @param bufferHandle target buffer handle
   * @param offset byte offset
   * @param size byte size
   */
  void markDirty(long bufferHandle, long offset, long size);

  /**
   * Emits pending transfers into the provided command buffer.
   *
   * @param cmd command buffer receiving transfer commands
   */
  void flush(VkCommandBuffer cmd);

  /**
   * Clears scheduler state for the next frame/update.
   */
  void reset();

  /**
   * Releases backend resources owned by the scheduler.
   */
  void destroy();
}
