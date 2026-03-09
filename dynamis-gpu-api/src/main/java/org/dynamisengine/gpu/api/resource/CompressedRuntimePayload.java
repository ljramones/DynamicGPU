package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Optional compressed payload form accepted at DynamisGPU ingestion boundaries.
 */
public record CompressedRuntimePayload(
    RuntimePayloadCompressionMode mode,
    int uncompressedByteSize,
    byte[] payloadBytes
) {
  public CompressedRuntimePayload {
    if (mode == null) {
      throw new NullPointerException("mode");
    }
    if (uncompressedByteSize < 0) {
      throw new IllegalArgumentException("uncompressedByteSize must be >= 0");
    }
    if (payloadBytes == null) {
      throw new NullPointerException("payloadBytes");
    }
    payloadBytes = payloadBytes.clone();
  }

  @Override
  public byte[] payloadBytes() {
    return payloadBytes.clone();
  }

  /**
   * Restores canonical uncompressed bytes using this payload's mode and declared byte size.
   */
  public byte[] toUncompressedBytes() {
    return RuntimePayloadCompression.decompress(payloadBytes, uncompressedByteSize, mode);
  }

  /**
   * Restores canonical uncompressed bytes as a little-endian direct buffer.
   */
  public ByteBuffer toUncompressedByteBuffer() {
    byte[] bytes = toUncompressedBytes();
    ByteBuffer out = ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.LITTLE_ENDIAN);
    out.put(bytes);
    out.flip();
    return out;
  }
}

