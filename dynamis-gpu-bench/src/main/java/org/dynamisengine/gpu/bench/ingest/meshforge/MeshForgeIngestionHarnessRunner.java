package org.dynamisengine.gpu.bench.ingest.meshforge;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshResource;
import org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan;
import org.dynamisengine.gpu.api.upload.GpuUploadExecutor;
import org.dynamisengine.gpu.bench.ingest.GeometryUploadValidation;
import org.dynamisengine.gpu.bench.ingest.ValidationSummary;
import org.dynamisengine.meshforge.gpu.cache.RuntimeGeometryLoader;

/**
 * Real MeshForge -> DynamisGPU ingestion runner with segmented timing.
 */
public final class MeshForgeIngestionHarnessRunner {
  private final RuntimeGeometryLoader loader;
  private final GpuUploadExecutor uploadExecutor;
  private final GeometryUploadValidation validation;
  private final Path cacheDirectory;

  public MeshForgeIngestionHarnessRunner(
      RuntimeGeometryLoader loader, GpuUploadExecutor uploadExecutor, GeometryUploadValidation validation) {
    this(loader, uploadExecutor, validation, null);
  }

  public MeshForgeIngestionHarnessRunner(
      RuntimeGeometryLoader loader,
      GpuUploadExecutor uploadExecutor,
      GeometryUploadValidation validation,
      Path cacheDirectory) {
    this.loader = Objects.requireNonNull(loader, "loader");
    this.uploadExecutor = Objects.requireNonNull(uploadExecutor, "uploadExecutor");
    this.validation = Objects.requireNonNull(validation, "validation");
    this.cacheDirectory = cacheDirectory;
  }

  public MeshForgeIngestionRunReport run(String fixtureName, Path sourceMesh)
      throws IOException, GpuException {
    Objects.requireNonNull(fixtureName, "fixtureName");
    Objects.requireNonNull(sourceMesh, "sourceMesh");

    // Warm cache so measured load step is intended to be cache-hit.
    loadWithConfiguredCache(sourceMesh);

    long totalStart = System.nanoTime();
    long loadStart = System.nanoTime();
    RuntimeGeometryLoader.Result loaded = loadWithConfiguredCache(sourceMesh);
    long loadEnd = System.nanoTime();

    long bridgeStart = System.nanoTime();
    GpuGeometryUploadPlan plan = MeshForgeRuntimePlanAdapter.toApiPlan(loaded.payload());
    ValidationSummary summary = validation.validatePlan(plan);
    long bridgeEnd = System.nanoTime();

    long uploadStart = System.nanoTime();
    try (GpuMeshResource resource = uploadExecutor.upload(plan)) {
      validation.validateUploadedResource(plan, resource, summary);
    }
    long uploadEnd = System.nanoTime();
    long totalEnd = System.nanoTime();

    return new MeshForgeIngestionRunReport(
        fixtureName,
        MeshForgeIngestionStatus.FULL_SUCCESS,
        loaded.source(),
        new MeshForgeIngestionTiming(
            loadEnd - loadStart,
            bridgeEnd - bridgeStart,
            uploadEnd - uploadStart,
            totalEnd - totalStart),
        summary,
        null);
  }

  public MeshForgeIngestionRunReport runPreuploadOnly(
      String fixtureName, Path sourceMesh, String blockedReason) throws IOException {
    Objects.requireNonNull(fixtureName, "fixtureName");
    Objects.requireNonNull(sourceMesh, "sourceMesh");
    Objects.requireNonNull(blockedReason, "blockedReason");

    // Warm cache so measured load step is intended to be cache-hit.
    loadWithConfiguredCache(sourceMesh);

    long totalStart = System.nanoTime();
    long loadStart = System.nanoTime();
    RuntimeGeometryLoader.Result loaded = loadWithConfiguredCache(sourceMesh);
    long loadEnd = System.nanoTime();

    long bridgeStart = System.nanoTime();
    GpuGeometryUploadPlan plan = MeshForgeRuntimePlanAdapter.toApiPlan(loaded.payload());
    ValidationSummary summary = validation.validatePlan(plan);
    long bridgeEnd = System.nanoTime();
    long totalEnd = System.nanoTime();

    return new MeshForgeIngestionRunReport(
        fixtureName,
        MeshForgeIngestionStatus.PREUPLOAD_SUCCESS_UPLOAD_BLOCKED,
        loaded.source(),
        new MeshForgeIngestionTiming(
            loadEnd - loadStart,
            bridgeEnd - bridgeStart,
            -1L,
            totalEnd - totalStart),
        summary,
        blockedReason);
  }

  private RuntimeGeometryLoader.Result loadWithConfiguredCache(Path sourceMesh) throws IOException {
    if (cacheDirectory == null) {
      return loader.load(sourceMesh);
    }
    Path cacheFile = cacheDirectory.resolve(sourceMesh.getFileName().toString() + ".mfgc");
    return loader.load(sourceMesh, cacheFile, false);
  }
}
