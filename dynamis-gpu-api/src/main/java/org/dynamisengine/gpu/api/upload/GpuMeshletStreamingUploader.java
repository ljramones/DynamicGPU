package org.dynamisengine.gpu.api.upload;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshletStreamingPayload;
import org.dynamisengine.gpu.api.resource.GpuMeshletStreamingResource;

/**
 * Uploads validated meshlet-streaming payloads into GPU-managed resources.
 */
public interface GpuMeshletStreamingUploader {
  /**
   * Creates/uploads a GPU meshlet-streaming resource from a payload contract.
   *
   * @param payload validated meshlet-streaming payload
   * @return uploaded resource
   * @throws GpuException on backend upload/create failures
   */
  GpuMeshletStreamingResource upload(GpuMeshletStreamingPayload payload) throws GpuException;
}
