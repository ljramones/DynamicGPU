package org.dynamisengine.gpu.api.gpu;

import java.util.Objects;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryResource;

/**
 * Describes one BLAS build-preparation work unit.
 */
public final class RayTracingBlasWork {
  private final GpuRayTracingGeometryResource geometryResource;

  public RayTracingBlasWork(GpuRayTracingGeometryResource geometryResource) {
    this.geometryResource = Objects.requireNonNull(geometryResource, "geometryResource");
  }

  public static RayTracingBlasWork fromGeometryResource(GpuRayTracingGeometryResource geometryResource) {
    return new RayTracingBlasWork(geometryResource);
  }

  public GpuRayTracingGeometryResource geometryResource() {
    return geometryResource;
  }
}

