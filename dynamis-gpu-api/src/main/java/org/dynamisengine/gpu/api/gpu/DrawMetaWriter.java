package org.dynamisengine.gpu.api.gpu;

/**
 * Writes per-draw metadata consumed by culling and draw submission paths.
 */
public interface DrawMetaWriter {
  /**
   * Writes metadata for a single draw entry.
   *
   * @param drawIndex target draw slot
   * @param jointPaletteIndex joint palette handle/index
   * @param morphDeltaIndex morph delta handle/index
   * @param morphWeightIndex morph weight handle/index
   * @param instanceDataIndex instance data handle/index
   * @param materialIndex material index
   * @param drawFlags packed draw flags
   * @param meshIndex mesh index/id
   */
  void write(
      int drawIndex,
      int jointPaletteIndex,
      int morphDeltaIndex,
      int morphWeightIndex,
      int instanceDataIndex,
      int materialIndex,
      int drawFlags,
      int meshIndex);

  /**
   * Flushes staged writes to the backing GPU resource, if required by implementation.
   */
  void flush();

  /**
   * Releases backend resources owned by this writer.
   */
  void destroy();

  /**
   * Returns maximum number of draw entries this writer can hold.
   *
   * @return entry capacity
   */
  int capacity();
}
