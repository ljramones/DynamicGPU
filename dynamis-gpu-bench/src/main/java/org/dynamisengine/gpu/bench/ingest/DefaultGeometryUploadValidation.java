package org.dynamisengine.gpu.bench.ingest;

import java.util.Objects;
import org.dynamisengine.gpu.api.layout.SubmeshRange;
import org.dynamisengine.gpu.api.resource.GpuMeshResource;
import org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan;

/**
 * Default Phase 1 validation checks for upload plans and uploaded resources.
 */
public final class DefaultGeometryUploadValidation implements GeometryUploadValidation {
  @Override
  public ValidationSummary validatePlan(GpuGeometryUploadPlan plan) {
    Objects.requireNonNull(plan, "plan");

    int strideBytes = plan.vertexLayout().strideBytes();
    int vertexBytes = plan.vertexData().remaining();
    if (vertexBytes % strideBytes != 0) {
      throw new IllegalArgumentException(
          "vertexData bytes must be a multiple of vertex stride: bytes="
              + vertexBytes
              + ", stride="
              + strideBytes);
    }
    int vertexCount = vertexBytes / strideBytes;
    int indexCount = 0;

    if (plan.indexData() != null) {
      int indexBytes = plan.indexData().remaining();
      int elementSize = plan.indexType().byteSize();
      if (indexBytes % elementSize != 0) {
        throw new IllegalArgumentException(
            "indexData bytes must be a multiple of index element size: bytes="
                + indexBytes
                + ", elementSize="
                + elementSize);
      }
      indexCount = indexBytes / elementSize;
    }

    for (SubmeshRange submesh : plan.submeshes()) {
      int submeshEnd = submesh.firstIndex() + submesh.indexCount();
      if (plan.indexData() != null && submeshEnd > indexCount) {
        throw new IllegalArgumentException(
            "submesh exceeds index count: firstIndex="
                + submesh.firstIndex()
                + ", indexCount="
                + submesh.indexCount()
                + ", total="
                + indexCount);
      }
      if (submesh.baseVertex() < 0 || submesh.baseVertex() >= vertexCount) {
        throw new IllegalArgumentException(
            "submesh baseVertex out of range: baseVertex="
                + submesh.baseVertex()
                + ", vertexCount="
                + vertexCount);
      }
    }

    return new ValidationSummary(
        vertexCount, indexCount, strideBytes, plan.indexType(), plan.submeshes().size());
  }

  @Override
  public void validateUploadedResource(
      GpuGeometryUploadPlan plan, GpuMeshResource resource, ValidationSummary summary) {
    Objects.requireNonNull(plan, "plan");
    Objects.requireNonNull(resource, "resource");
    Objects.requireNonNull(summary, "summary");

    if (resource.vertexLayout().strideBytes() != summary.strideBytes()) {
      throw new IllegalStateException("uploaded vertex layout stride differs from plan");
    }
    if (!resource.vertexLayout().equals(plan.vertexLayout())) {
      throw new IllegalStateException("uploaded vertex layout differs from plan");
    }
    if (!resource.submeshes().equals(plan.submeshes())) {
      throw new IllegalStateException("uploaded submesh ranges differ from plan");
    }
    if (resource.vertexBuffer().sizeBytes() != plan.vertexData().remaining()) {
      throw new IllegalStateException("uploaded vertex buffer size differs from plan payload");
    }

    if (plan.indexData() == null) {
      if (resource.isIndexed()) {
        throw new IllegalStateException("resource unexpectedly has index buffer");
      }
      return;
    }

    if (!resource.isIndexed()) {
      throw new IllegalStateException("resource is missing index buffer");
    }
    if (resource.indexType() != summary.indexType()) {
      throw new IllegalStateException("uploaded index type differs from plan");
    }
    if (resource.indexBuffer().sizeBytes() != plan.indexData().remaining()) {
      throw new IllegalStateException("uploaded index buffer size differs from plan payload");
    }
  }
}
