package org.dynamisengine.gpu.api.gpu;

/**
 * Uploads and exposes per-instance render data.
 */
public interface InstanceDataBuffer {
  /**
   * Uploads instance transforms and metadata for the current frame/update.
   *
   * @param modelMatrices per-instance model matrices
   * @param materialIndices material index per instance
   * @param flags packed per-instance flags
   */
  void upload(float[][] modelMatrices, int[] materialIndices, int[] flags);

  /**
   * Returns the backend buffer handle used for descriptors/dispatch.
   *
   * @return opaque backend buffer handle
   */
  long bufferHandle();

  /**
   * Returns the number of uploaded instances.
   *
   * @return active instance count
   */
  int instanceCount();

  /**
   * Releases backend resources owned by this buffer.
   */
  void destroy();
}
