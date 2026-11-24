package org.dynamisengine.gpu.api.layout;

/**
 * Vertex attribute storage formats: the GPU binding side of a vertex layout. {@code INT} formats are signed,
 * {@code UINT} unsigned; {@code NORM} formats are normalized to [0, 1] ({@code UNORM}) or [-1, 1] ({@code SNORM}).
 * Mesh-level layouts (attribute semantics and packing choices) belong to MeshForge.
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
  /** @deprecated the same format as {@link #UNORM8X4}; use that. */
  @Deprecated(since = "1.2.0")
  UINT8X4_NORM(4),
  UINT8X4(4),
  INT8X4(4),
  UNORM8X4(4),
  SNORM8X4(4),
  UINT16X2(4),
  UINT16X4(8),
  SNORM16X4(8),
  OCTA_SNORM16X2(4),
  // 1.2.0: the remaining formats Vectrix's vertex layout described
  FLOAT16X4(8),
  UNORM16X2(4),
  UNORM16X4(8),
  SNORM16X2(4),
  UINT32X1(4),
  UINT32X2(8),
  UINT32X3(12),
  UINT32X4(16),
  UINT8X2(2),
  INT8X2(2),
  UNORM8X2(2),
  SNORM8X2(2);

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
