package org.dynamisengine.gpu.api.gpu;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuRayTracingTlasResource;

/**
 * Schedulable GPU capability for TLAS construction from built BLAS instances.
 */
public interface RayTracingTlasCapability extends AutoCloseable {
  /**
   * Executes one TLAS build work item.
   */
  GpuRayTracingTlasResource execute(RayTracingTlasWork work) throws GpuException;

  /**
   * Releases backend resources owned by this capability.
   */
  @Override
  void close();
}

