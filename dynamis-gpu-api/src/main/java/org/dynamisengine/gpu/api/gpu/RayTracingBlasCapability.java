package org.dynamisengine.gpu.api.gpu;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasResource;

/**
 * Schedulable GPU capability foundation for BLAS build preparation.
 */
public interface RayTracingBlasCapability extends AutoCloseable {
  /**
   * Executes one BLAS work item and produces a backend-ready BLAS build input resource seam.
   */
  GpuRayTracingBlasResource execute(RayTracingBlasWork work) throws GpuException;

  /**
   * Releases backend resources owned by this capability.
   */
  @Override
  void close();
}

