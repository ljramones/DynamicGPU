package org.dynamisengine.gpu.api.error;

/**
 * Stable error categories surfaced by the GPU abstraction layer.
 */
public enum GpuErrorCode {
  /** Backend initialization or driver-level setup failed. */
  BACKEND_INIT_FAILED,
  /** Caller supplied invalid data or arguments to an API operation. */
  INVALID_ARGUMENT
}
