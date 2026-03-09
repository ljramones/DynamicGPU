package org.dynamisengine.gpu.api.buffer;

/**
 * Intended usage for a GPU buffer.
 */
public enum GpuBufferUsage {
  VERTEX,
  INDEX,
  INDIRECT,
  STORAGE,
  TRANSFER_SRC,
  TRANSFER_DST
}
