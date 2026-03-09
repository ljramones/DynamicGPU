package org.dynamisengine.gpu.api.gpu;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuResolvedMeshletStreamingResource;

/**
 * Schedulable GPU capability for meshlet stream-unit residency resolution.
 */
public interface MeshletStreamingResidencyCapability extends AutoCloseable {
  /**
   * Executes one meshlet streaming residency work item.
   */
  GpuResolvedMeshletStreamingResource execute(MeshletStreamingResidencyWork work) throws GpuException;

  /**
   * Releases backend resources owned by this capability.
   */
  @Override
  void close();
}

