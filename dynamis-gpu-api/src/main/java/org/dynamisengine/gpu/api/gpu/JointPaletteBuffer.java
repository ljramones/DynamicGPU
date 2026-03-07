package org.dynamisengine.gpu.api.gpu;

/**
 * Uploads joint matrices used by skinning.
 */
public interface JointPaletteBuffer {
  /**
   * Uploads packed joint matrices.
   *
   * @param jointMatrices matrix payload in implementation-defined layout
   */
  void upload(float[] jointMatrices);

  /**
   * Returns the backend buffer handle used for descriptors/dispatch.
   *
   * @return opaque backend buffer handle
   */
  long bufferHandle();

  /**
   * Returns number of joints currently stored.
   *
   * @return joint count
   */
  int jointCount();

  /**
   * Releases backend resources owned by this buffer.
   */
  void destroy();
}
