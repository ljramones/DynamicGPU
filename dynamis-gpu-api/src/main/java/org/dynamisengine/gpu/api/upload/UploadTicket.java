package org.dynamisengine.gpu.api.upload;

import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshResource;

/**
 * Handle to a submitted upload request.
 */
public interface UploadTicket {
  /**
   * @return stable ticket id assigned by the upload manager
   */
  long id();

  /**
   * @return true when upload has completed (success or failure)
   */
  boolean isComplete();

  /**
   * Waits for upload completion and returns the produced GPU resource.
   *
   * @return uploaded mesh resource
   * @throws InterruptedException if interrupted while waiting
   * @throws GpuException when upload fails
   */
  GpuMeshResource await() throws InterruptedException, GpuException;
}

