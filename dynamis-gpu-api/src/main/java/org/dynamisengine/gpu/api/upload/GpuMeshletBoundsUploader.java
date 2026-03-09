package org.dynamisengine.gpu.api.upload;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsResource;

/**
 * Uploads validated meshlet-bounds payloads into GPU-managed resources.
 */
public interface GpuMeshletBoundsUploader {
  /**
   * Creates/uploads a GPU meshlet-bounds resource from a payload contract.
   *
   * @param payload validated meshlet-bounds payload
   * @return uploaded resource
   * @throws GpuException on backend upload/create failures
   */
  GpuMeshletBoundsResource upload(GpuMeshletBoundsPayload payload) throws GpuException;
}

