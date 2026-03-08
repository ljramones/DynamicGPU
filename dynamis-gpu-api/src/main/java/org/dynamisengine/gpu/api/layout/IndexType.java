package org.dynamisengine.gpu.api.layout;

/**
 * Index element type for indexed mesh draws.
 */
public enum IndexType {
  UINT16(2),
  UINT32(4);

  private final int byteSize;

  IndexType(int byteSize) {
    this.byteSize = byteSize;
  }

  /**
   * @return encoded byte width of one index element
   */
  public int byteSize() {
    return byteSize;
  }
}
