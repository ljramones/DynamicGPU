package org.dynamisengine.gpu.api.upload;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshResource;

/**
 * Executes runtime geometry upload plans to produce GPU-owned mesh resources.
 */
public interface GpuUploadExecutor {
  /**
   * Uploads geometry payload to backend GPU resources.
   *
   * @param plan geometry upload plan
   * @return uploaded mesh resource
   * @throws GpuException when upload fails
   */
  GpuMeshResource upload(GpuGeometryUploadPlan plan) throws GpuException;
}
