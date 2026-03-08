package org.dynamisengine.gpu.bench.ingest;

import org.dynamisengine.gpu.api.resource.GpuMeshResource;
import org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan;

/**
 * Validation hook contract for plan and upload-result integrity checks.
 */
public interface GeometryUploadValidation {
  /**
   * Validates a plan before upload and extracts fixture-independent metadata.
   */
  ValidationSummary validatePlan(GpuGeometryUploadPlan plan);

  /**
   * Validates uploaded resource against plan and summary captured from pre-upload validation.
   */
  void validateUploadedResource(
      GpuGeometryUploadPlan plan, GpuMeshResource resource, ValidationSummary summary);
}
