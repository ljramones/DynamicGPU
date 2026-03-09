package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class RuntimePayloadCompressionTest {
  @Test
  void roundTripsDeflate() {
    byte[] raw = new byte[32 * 1024];
    for (int i = 0; i < raw.length; i++) {
      raw[i] = (byte) ((i * 11) & 0xFF);
    }

    byte[] compressed = RuntimePayloadCompression.compress(raw, RuntimePayloadCompressionMode.DEFLATE);
    byte[] restored =
        RuntimePayloadCompression.decompress(compressed, raw.length, RuntimePayloadCompressionMode.DEFLATE);

    assertArrayEquals(raw, restored);
  }

  @Test
  void rejectsDecompressionSizeMismatch() {
    byte[] raw = new byte[] {1, 2, 3, 4, 5, 6};
    byte[] compressed = RuntimePayloadCompression.compress(raw, RuntimePayloadCompressionMode.DEFLATE);
    assertThrows(
        IllegalArgumentException.class,
        () -> RuntimePayloadCompression.decompress(compressed, raw.length + 1, RuntimePayloadCompressionMode.DEFLATE));
  }
}

