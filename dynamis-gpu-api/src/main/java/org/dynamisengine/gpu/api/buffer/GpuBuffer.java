package org.dynamisengine.gpu.api.buffer;

/**
 * Backend-neutral GPU buffer resource contract.
 */
public interface GpuBuffer extends AutoCloseable {
  /**
   * @return opaque backend handle
   */
  GpuBufferHandle handle();

  /**
   * @return size in bytes
   */
  long sizeBytes();

  /**
   * @return declared usage
   */
  GpuBufferUsage usage();

  /**
   * @return declared memory location intent
   */
  GpuMemoryLocation memoryLocation();

  /**
   * Releases backend resources owned by this buffer.
   */
  @Override
  void close();
}
