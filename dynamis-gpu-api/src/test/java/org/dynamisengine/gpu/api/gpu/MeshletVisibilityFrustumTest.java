package org.dynamisengine.gpu.api.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class MeshletVisibilityFrustumTest {
  @Test
  void acceptsExactlyTwentyFourFloats() {
    float[] planes = new float[24];
    planes[0] = 1f;

    MeshletVisibilityFrustum frustum = MeshletVisibilityFrustum.of(planes);
    float[] copy = frustum.planes24();
    assertEquals(24, copy.length);
    assertEquals(1f, copy[0]);
    assertNotSame(planes, copy);
  }

  @Test
  void rejectsInvalidLength() {
    assertThrows(IllegalArgumentException.class, () -> MeshletVisibilityFrustum.of(new float[23]));
  }
}

