package org.dynamisengine.gpu.bench.ingest;

import org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan;

/**
 * Fixture/source seam for supplying upload plans to the ingestion harness.
 */
@FunctionalInterface
public interface GpuGeometryUploadPlanSupplier {
  /**
   * @return next upload plan instance
   */
  GpuGeometryUploadPlan get();
}
