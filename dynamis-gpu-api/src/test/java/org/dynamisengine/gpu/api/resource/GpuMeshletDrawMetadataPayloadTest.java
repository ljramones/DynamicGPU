package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class GpuMeshletDrawMetadataPayloadTest {
  @Test
  void storesPerMeshletDrawFields() {
    GpuMeshletDrawMetadataPayload payload =
        GpuMeshletDrawMetadataPayload.of(new int[] {12, 18}, new int[] {0, 12}, new int[] {3, 7});

    assertEquals(2, payload.meshletCount());
    assertEquals(12, payload.indexCount(0));
    assertEquals(18, payload.indexCount(1));
    assertEquals(12, payload.firstIndex(1));
    assertEquals(7, payload.vertexOffset(1));
  }

  @Test
  void rejectsMismatchedLengths() {
    assertThrows(
        IllegalArgumentException.class,
        () -> GpuMeshletDrawMetadataPayload.of(new int[] {1}, new int[] {0, 1}, new int[] {0}));
  }
}

