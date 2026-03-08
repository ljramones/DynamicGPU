package org.dynamisengine.gpu.api.buffer;

/**
 * Opaque backend buffer handle value.
 *
 * @param value backend-defined handle value
 */
public record GpuBufferHandle(long value) {
  public GpuBufferHandle {
    if (value <= 0L) {
      throw new IllegalArgumentException("value must be > 0");
    }
  }
}
