package org.dynamisengine.gpu.api.gpu;

/**
 * Writes mesh/object bounding volumes for culling.
 */
public interface BoundsBuffer {
  /**
   * Writes one sphere bound.
   *
   * @param slot bound slot index
   * @param cx center x
   * @param cy center y
   * @param cz center z
   * @param radius sphere radius
   */
  void writeBounds(int slot, float cx, float cy, float cz, float radius);

  /**
   * Flushes staged writes to the backing GPU resource, if required by implementation.
   */
  void flush();

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
