package org.dynamisengine.gpu.api.gpu;

/**
 * Uploads morph/animation weights to GPU-visible memory.
 */
public interface WeightBuffer {
  /**
   * Uploads a contiguous array of weights.
   *
   * @param weights weight values to upload
   */
  void upload(float[] weights);

  /**
   * Returns the backend buffer handle used for descriptors/dispatch.
   *
   * @return opaque backend buffer handle
   */
  long bufferHandle();

  /**
   * Releases backend resources owned by this buffer.
   */
  void destroy();
}
