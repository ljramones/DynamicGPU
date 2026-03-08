package org.dynamisengine.gpu.bench.ingest;

import java.util.Objects;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshResource;
import org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan;
import org.dynamisengine.gpu.api.upload.GpuUploadExecutor;

/**
 * Generic ingestion runner that measures upload-plan intake and upload execution timings.
 */
public final class GpuIngestionHarnessRunner {
  private final GpuUploadExecutor uploadExecutor;
  private final GeometryUploadValidation validation;

  public GpuIngestionHarnessRunner(
      GpuUploadExecutor uploadExecutor, GeometryUploadValidation validation) {
    this.uploadExecutor = Objects.requireNonNull(uploadExecutor, "uploadExecutor");
    this.validation = Objects.requireNonNull(validation, "validation");
  }

  public IngestionRunReport run(String fixtureName, GpuGeometryUploadPlanSupplier planSupplier)
      throws GpuException {
    Objects.requireNonNull(fixtureName, "fixtureName");
    Objects.requireNonNull(planSupplier, "planSupplier");

    long totalStart = System.nanoTime();

    long intakeStart = System.nanoTime();
    GpuGeometryUploadPlan plan = Objects.requireNonNull(planSupplier.get(), "planSupplier returned null plan");
    ValidationSummary summary = validation.validatePlan(plan);
    long intakeEnd = System.nanoTime();

    long uploadStart = System.nanoTime();
    try (GpuMeshResource resource = uploadExecutor.upload(plan)) {
      validation.validateUploadedResource(plan, resource, summary);
    }
    long uploadEnd = System.nanoTime();
    long totalEnd = System.nanoTime();

    return new IngestionRunReport(
        fixtureName,
        new IngestionTiming(
            intakeEnd - intakeStart,
            uploadEnd - uploadStart,
            totalEnd - totalStart),
        summary);
  }
}
