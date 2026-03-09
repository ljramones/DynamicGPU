package org.dynamisengine.gpu.api.gpu;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuSelectedMeshletLodResource;

/**
 * Schedulable GPU capability for meshlet LOD level selection.
 */
public interface MeshletLodSelectionCapability extends AutoCloseable {
  /**
   * Executes one meshlet LOD selection work item.
   */
  GpuSelectedMeshletLodResource execute(MeshletLodSelectionWork work) throws GpuException;

  /**
   * Releases backend resources owned by this capability.
   */
  @Override
  void close();
}
