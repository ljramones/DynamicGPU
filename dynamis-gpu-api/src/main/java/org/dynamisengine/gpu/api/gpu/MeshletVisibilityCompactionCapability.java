package org.dynamisengine.gpu.api.gpu;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListResource;

/**
 * Schedulable GPU capability that compacts meshlet visibility flags into a visible index list.
 */
public interface MeshletVisibilityCompactionCapability extends AutoCloseable {
  /**
   * Executes one compaction work item and returns compacted visible meshlet indices.
   */
  GpuVisibleMeshletListResource execute(MeshletVisibilityCompactionWork work) throws GpuException;

  /**
   * Releases backend resources owned by this capability.
   */
  @Override
  void close();
}

