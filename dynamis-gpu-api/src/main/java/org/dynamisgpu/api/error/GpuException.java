package org.dynamisgpu.api.error;

public class GpuException extends Exception {
  private final GpuErrorCode errorCode;
  private final boolean recoverable;

  public GpuException(GpuErrorCode errorCode, String message, boolean recoverable) {
    super(message);
    this.errorCode = errorCode;
    this.recoverable = recoverable;
  }

  public GpuException(GpuErrorCode errorCode, String message, Throwable cause, boolean recoverable) {
    super(message, cause);
    this.errorCode = errorCode;
    this.recoverable = recoverable;
  }

  public GpuErrorCode errorCode() {
    return errorCode;
  }

  public boolean recoverable() {
    return recoverable;
  }
}
