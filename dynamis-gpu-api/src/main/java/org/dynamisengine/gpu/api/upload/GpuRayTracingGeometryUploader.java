package org.dynamisengine.gpu.api.upload;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryResource;

/**
 * Uploads validated RT geometry payloads into GPU-managed resources.
 */
public interface GpuRayTracingGeometryUploader {
  /**
   * Creates/uploads a GPU RT geometry resource from a payload contract.
   *
   * @param payload validated RT geometry payload
   * @return uploaded resource
   * @throws GpuException on backend upload/create failures
   */
  GpuRayTracingGeometryResource upload(GpuRayTracingGeometryPayload payload) throws GpuException;
}

