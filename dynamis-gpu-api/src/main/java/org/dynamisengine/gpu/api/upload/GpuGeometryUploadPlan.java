package org.dynamisengine.gpu.api.upload;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.Objects;
import org.dynamisengine.gpu.api.layout.IndexType;
import org.dynamisengine.gpu.api.layout.SubmeshRange;
import org.dynamisengine.gpu.api.layout.VertexLayout;

/**
 * Opaque packed geometry payload plus explicit layout metadata.
 *
 * @param vertexData packed vertex bytes
 * @param indexData optional packed index bytes for indexed meshes
 * @param vertexLayout vertex attribute layout metadata
 * @param indexType optional index element type; required when indexData is present
 * @param submeshes draw ranges encoded against the packed geometry buffers
 */
public record GpuGeometryUploadPlan(
    ByteBuffer vertexData,
    ByteBuffer indexData,
    VertexLayout vertexLayout,
    IndexType indexType,
    List<SubmeshRange> submeshes) {

  public GpuGeometryUploadPlan {
    Objects.requireNonNull(vertexData, "vertexData");
    Objects.requireNonNull(vertexLayout, "vertexLayout");
    Objects.requireNonNull(submeshes, "submeshes");
    if (!vertexData.hasRemaining()) {
      throw new IllegalArgumentException("vertexData must not be empty");
    }
    if (submeshes.isEmpty()) {
      throw new IllegalArgumentException("submeshes must not be empty");
    }
    if (indexData != null && !indexData.hasRemaining()) {
      throw new IllegalArgumentException("indexData must be null or non-empty");
    }
    if ((indexData == null) != (indexType == null)) {
      throw new IllegalArgumentException("indexData and indexType must both be present or both be absent");
    }

    vertexData = vertexData.asReadOnlyBuffer();
    if (indexData != null) {
      indexData = indexData.asReadOnlyBuffer();
    }
    submeshes = List.copyOf(submeshes);
  }
}
