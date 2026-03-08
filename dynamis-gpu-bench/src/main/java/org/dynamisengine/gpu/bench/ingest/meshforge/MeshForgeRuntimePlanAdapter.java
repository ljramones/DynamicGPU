package org.dynamisengine.gpu.bench.ingest.meshforge;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.dynamisengine.gpu.api.layout.IndexType;
import org.dynamisengine.gpu.api.layout.SubmeshRange;
import org.dynamisengine.gpu.api.layout.VertexAttribute;
import org.dynamisengine.gpu.api.layout.VertexFormat;
import org.dynamisengine.gpu.api.layout.VertexLayout;
import org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan;
import org.dynamisengine.meshforge.core.attr.AttributeKey;
import org.dynamisengine.meshforge.gpu.RuntimeGeometryPayload;
import org.dynamisengine.meshforge.pack.buffer.PackedMesh;

/**
 * Converts MeshForge runtime payloads into DynamisGPU Phase 1 upload plans.
 */
public final class MeshForgeRuntimePlanAdapter {
  private MeshForgeRuntimePlanAdapter() {}

  public static GpuGeometryUploadPlan toApiPlan(RuntimeGeometryPayload payload) {
    Objects.requireNonNull(payload, "payload");

    VertexLayout vertexLayout = mapLayout(payload.layout());
    ByteBuffer vertexBytes = asReadOnlyCopy(payload.vertexBytes());

    ByteBuffer indexBytes = null;
    IndexType indexType = null;
    if (payload.indexType() != null && payload.indexCount() > 0 && payload.indexBytes() != null) {
      indexType = mapIndexType(payload.indexType());
      indexBytes = asReadOnlyCopy(payload.indexBytes());
    }

    List<SubmeshRange> submeshes = new ArrayList<>(payload.submeshes().size());
    for (PackedMesh.SubmeshRange submesh : payload.submeshes()) {
      // MeshForge submeshes are index ranges; Phase 1 baseVertex is currently 0.
      submeshes.add(new SubmeshRange(submesh.firstIndex(), submesh.indexCount(), 0));
    }
    if (submeshes.isEmpty()) {
      throw new IllegalArgumentException("MeshForge payload contains no submeshes");
    }

    return new GpuGeometryUploadPlan(vertexBytes, indexBytes, vertexLayout, indexType, submeshes);
  }

  static VertexLayout mapLayout(org.dynamisengine.meshforge.pack.layout.VertexLayout source) {
    Objects.requireNonNull(source, "source");
    List<VertexAttribute> attributes = new ArrayList<>(source.entries().size());
    int location = 0;
    for (Map.Entry<AttributeKey, org.dynamisengine.meshforge.pack.layout.VertexLayout.Entry> entry :
        source.entries().entrySet()) {
      var layoutEntry = entry.getValue();
      attributes.add(
          new VertexAttribute(
              location++,
              layoutEntry.offsetBytes(),
              mapVertexFormat(layoutEntry.format())));
    }
    return new VertexLayout(source.strideBytes(), attributes);
  }

  static VertexFormat mapVertexFormat(org.dynamisengine.meshforge.core.attr.VertexFormat source) {
    return switch (source) {
      case F32x1 -> VertexFormat.FLOAT1;
      case F32x2 -> VertexFormat.FLOAT2;
      case F32x3 -> VertexFormat.FLOAT3;
      case F32x4 -> VertexFormat.FLOAT4;
      case I32x1 -> VertexFormat.INT32X1;
      case I32x2 -> VertexFormat.INT32X2;
      case I32x3 -> VertexFormat.INT32X3;
      case I32x4 -> VertexFormat.INT32X4;
      case F16x2 -> VertexFormat.FLOAT16X2;
      case I16x2 -> VertexFormat.INT16X2;
      case I16x4 -> VertexFormat.INT16X4;
      case U16x4 -> VertexFormat.UINT16X4;
      case SNORM16x4 -> VertexFormat.SNORM16X4;
      case I8x4 -> VertexFormat.INT8X4;
      case U8x4 -> VertexFormat.UINT8X4;
      case UNORM8x4 -> VertexFormat.UNORM8X4;
      case SNORM8x4 -> VertexFormat.SNORM8X4;
      case OCTA_SNORM16x2 -> VertexFormat.OCTA_SNORM16X2;
    };
  }

  static IndexType mapIndexType(PackedMesh.IndexType source) {
    return switch (source) {
      case UINT16 -> IndexType.UINT16;
      case UINT32 -> IndexType.UINT32;
    };
  }

  static ByteBuffer asReadOnlyCopy(ByteBuffer source) {
    ByteBuffer copy = ByteBuffer.allocate(source.remaining());
    ByteBuffer duplicate = source.duplicate();
    copy.put(duplicate);
    copy.flip();
    return copy.asReadOnlyBuffer();
  }
}
