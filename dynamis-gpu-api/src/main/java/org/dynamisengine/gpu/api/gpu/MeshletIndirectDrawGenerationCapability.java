package org.dynamisengine.gpu.api.gpu;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshletIndirectDrawResource;

/**
 * Schedulable GPU capability that generates indexed-indirect draw commands for visible meshlets.
 */
public interface MeshletIndirectDrawGenerationCapability extends AutoCloseable {
  /**
   * Executes one indirect draw generation work item.
   */
  GpuMeshletIndirectDrawResource execute(MeshletIndirectDrawGenerationWork work) throws GpuException;

  /**
   * Releases backend resources owned by this capability.
   */
  @Override
  void close();
}

