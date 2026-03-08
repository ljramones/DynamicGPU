package org.dynamisengine.gpu.api.layout;

/**
 * Supported vertex attribute storage formats for Phase 1.
 */
public enum VertexFormat {
  FLOAT1(4),
  FLOAT2(8),
  FLOAT3(12),
  FLOAT4(16),
  INT32X1(4),
  INT32X2(8),
  INT32X3(12),
  INT32X4(16),
  FLOAT16X2(4),
  INT16X2(4),
  INT16X4(8),
  UINT8X4_NORM(4),
  UINT8X4(4),
  INT8X4(4),
  UNORM8X4(4),
  SNORM8X4(4),
  UINT16X2(4),
  UINT16X4(8),
  SNORM16X4(8),
  OCTA_SNORM16X2(4);

  private final int byteSize;

  VertexFormat(int byteSize) {
    this.byteSize = byteSize;
  }

  /**
   * @return encoded byte width of this attribute format
   */
  public int byteSize() {
    return byteSize;
  }
}
