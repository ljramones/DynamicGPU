package org.dynamisengine.gpu.bench.ingest.meshforge;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan;
import org.dynamisengine.gpu.bench.ingest.GpuGeometryUploadPlanSupplier;
import org.dynamisengine.meshforge.gpu.cache.RuntimeGeometryLoader;

/**
 * MeshForge-backed plan supplier for one fixture path.
 */
public final class MeshForgeFixturePlanSupplier implements GpuGeometryUploadPlanSupplier {
  private final RuntimeGeometryLoader loader;
  private final Path sourceMesh;

  public MeshForgeFixturePlanSupplier(RuntimeGeometryLoader loader, Path sourceMesh) {
    this.loader = Objects.requireNonNull(loader, "loader");
    this.sourceMesh = Objects.requireNonNull(sourceMesh, "sourceMesh");
  }

  @Override
  public GpuGeometryUploadPlan get() {
    try {
      RuntimeGeometryLoader.Result loaded = loader.load(sourceMesh);
      return MeshForgeRuntimePlanAdapter.toApiPlan(loaded.payload());
    } catch (IOException e) {
      throw new IllegalStateException("Failed loading fixture: " + sourceMesh, e);
    }
  }
}
