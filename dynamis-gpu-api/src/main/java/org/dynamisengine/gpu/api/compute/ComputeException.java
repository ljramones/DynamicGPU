package org.dynamisengine.gpu.api.compute;

/** Structured compute failure; native diagnostics are preserved in the message/cause. */
public final class ComputeException extends RuntimeException {
    public enum Code { UNAVAILABLE, INVALID_ARGUMENT, UNSUPPORTED, CLOSED, RESOURCE_EXHAUSTED,
        NATIVE_ERROR, DEVICE_LOST, TIMEOUT, PENDING }
    private final Code code;
    public ComputeException(Code code, String message) { super(message); this.code = code; }
    public ComputeException(Code code, String message, Throwable cause) { super(message, cause); this.code = code; }
    public Code code() { return code; }
}
