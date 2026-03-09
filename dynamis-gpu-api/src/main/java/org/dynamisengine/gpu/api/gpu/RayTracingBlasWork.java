package org.dynamisengine.gpu.api.gpu;

import java.util.Objects;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBuildInputResource;

/**
 * Describes one BLAS execution work unit.
 */
public final class RayTracingBlasWork {
  private final GpuRayTracingBuildInputResource buildInputResource;

  public RayTracingBlasWork(GpuRayTracingBuildInputResource buildInputResource) {
    this.buildInputResource = Objects.requireNonNull(buildInputResource, "buildInputResource");
  }

  public static RayTracingBlasWork fromBuildInputResource(GpuRayTracingBuildInputResource buildInputResource) {
    return new RayTracingBlasWork(buildInputResource);
  }

  public GpuRayTracingBuildInputResource buildInputResource() {
    return buildInputResource;
  }
}
