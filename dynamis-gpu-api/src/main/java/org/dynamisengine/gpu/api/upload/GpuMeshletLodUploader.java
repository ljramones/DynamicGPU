package org.dynamisengine.gpu.api.upload;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshletLodPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletLodResource;

/**
 * Uploads validated meshlet-LOD payloads into GPU-managed resources.
 */
public interface GpuMeshletLodUploader {
  /**
   * Creates/uploads a GPU meshlet-LOD resource from a payload contract.
   *
   * @param payload validated meshlet-LOD payload
   * @return uploaded resource
   * @throws GpuException on backend upload/create failures
   */
  GpuMeshletLodResource upload(GpuMeshletLodPayload payload) throws GpuException;
}
