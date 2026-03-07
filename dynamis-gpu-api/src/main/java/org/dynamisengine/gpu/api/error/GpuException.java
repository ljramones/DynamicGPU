package org.dynamisengine.gpu.api.error;

/**
 * Checked exception for GPU API failures with a stable error code and recoverability signal.
 */
public class GpuException extends Exception {
  private final GpuErrorCode errorCode;
  private final boolean recoverable;

  /**
   * Creates a GPU exception without a nested cause.
   *
   * @param errorCode stable error category
   * @param message human-readable failure detail
   * @param recoverable true when callers may retry or continue safely
   */
  public GpuException(GpuErrorCode errorCode, String message, boolean recoverable) {
    super(message);
    this.errorCode = errorCode;
    this.recoverable = recoverable;
  }

  /**
   * Creates a GPU exception with a nested cause.
   *
   * @param errorCode stable error category
   * @param message human-readable failure detail
   * @param cause underlying exception or backend failure
   * @param recoverable true when callers may retry or continue safely
   */
  public GpuException(GpuErrorCode errorCode, String message, Throwable cause, boolean recoverable) {
    super(message, cause);
    this.errorCode = errorCode;
    this.recoverable = recoverable;
  }

  /**
   * Returns the stable error category.
   *
   * @return error classification enum
   */
  public GpuErrorCode errorCode() {
    return errorCode;
  }

  /**
   * Returns whether callers may treat this as recoverable.
   *
   * @return true when retry/continue is expected to be safe
   */
  public boolean recoverable() {
    return recoverable;
  }
}
