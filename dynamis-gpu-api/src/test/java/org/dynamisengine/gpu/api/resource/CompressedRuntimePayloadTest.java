package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class CompressedRuntimePayloadTest {
  @Test
  void noneModeRoundTripPreservesBytes() {
    byte[] raw = new byte[] {9, 8, 7, 6, 5};
    CompressedRuntimePayload payload =
        new CompressedRuntimePayload(RuntimePayloadCompressionMode.NONE, raw.length, raw);

    assertEquals(RuntimePayloadCompressionMode.NONE, payload.mode());
    assertArrayEquals(raw, payload.toUncompressedBytes());
    assertEquals(raw.length, payload.toUncompressedByteBuffer().remaining());
  }

  @Test
  void rejectsDecompressionSizeMismatch() {
    byte[] raw = new byte[] {10, 11, 12, 13, 14, 15};
    byte[] compressed = RuntimePayloadCompression.compress(raw, RuntimePayloadCompressionMode.DEFLATE);
    CompressedRuntimePayload payload =
        new CompressedRuntimePayload(RuntimePayloadCompressionMode.DEFLATE, raw.length + 2, compressed);

    assertThrows(IllegalArgumentException.class, payload::toUncompressedBytes);
  }
}

