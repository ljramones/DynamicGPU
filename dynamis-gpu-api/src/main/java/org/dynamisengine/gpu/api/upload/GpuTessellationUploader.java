package org.dynamisengine.gpu.api.upload;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuTessellationPayload;
import org.dynamisengine.gpu.api.resource.GpuTessellationResource;

/**
 * Uploads validated tessellation payloads into GPU-managed resources.
 */
public interface GpuTessellationUploader {
  /**
   * Creates/uploads a GPU tessellation resource from a payload contract.
   *
   * @param payload validated tessellation payload
   * @return uploaded resource
   * @throws GpuException on backend upload/create failures
   */
  GpuTessellationResource upload(GpuTessellationPayload payload) throws GpuException;
}
