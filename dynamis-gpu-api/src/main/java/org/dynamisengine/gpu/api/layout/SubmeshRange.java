package org.dynamisengine.gpu.api.layout;

/**
 * Draw range metadata for one submesh.
 *
 * @param firstIndex index offset within the index buffer
 * @param indexCount number of indices in the range
 * @param baseVertex base vertex offset for indexed draw calls
 */
public record SubmeshRange(int firstIndex, int indexCount, int baseVertex) {
  public SubmeshRange {
    if (firstIndex < 0) {
      throw new IllegalArgumentException("firstIndex must be >= 0");
    }
    if (indexCount <= 0) {
      throw new IllegalArgumentException("indexCount must be > 0");
    }
  }
}
