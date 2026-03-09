package org.dynamisengine.gpu.api.gpu;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsResource;

/**
 * Schedulable GPU capability for meshlet frustum visibility.
 *
 * <p>This contract represents a GPU work unit execution seam. Higher layers describe
 * {@link MeshletVisibilityWork}, and backend internals execute it.
 */
public interface MeshletVisibilityCapability extends AutoCloseable {
  /**
   * Executes one meshlet visibility work item and returns the GPU visibility flags output.
   */
  GpuMeshletVisibilityFlagsResource execute(MeshletVisibilityWork work) throws GpuException;

  /**
   * Releases backend resources owned by this capability.
   */
  @Override
  void close();
}

