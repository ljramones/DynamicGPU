package org.dynamisengine.gpu.vulkan.compute;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.dynamisengine.gpu.api.gpu.ComputeCullingDispatch;
import org.dynamisengine.gpu.test.mock.MockVkCommandBuffer;
import org.junit.jupiter.api.Test;

/** The frustum argument is six planes of (a, b, c, d), as the interface documents. */
class VulkanComputeCullingDispatchTest {
  @Test
  void acceptsSixPlanesOfFourFloats() {
    var dispatch = new VulkanComputeCullingDispatch();
    float[] planes = new float[ComputeCullingDispatch.FRUSTUM_PLANE_FLOATS];
    for (int i = 0; i < planes.length; i++) {
      planes[i] = i * 0.5f;
    }
    dispatch.dispatch(new MockVkCommandBuffer(7L), 12, planes);
    assertEquals(24, ComputeCullingDispatch.FRUSTUM_PLANE_FLOATS);
    assertEquals(12, dispatch.lastMeshCount());
    assertArrayEquals(planes, dispatch.lastFrustumPlanes());
  }

  @Test
  void rejectsAnyOtherLength() {
    var dispatch = new VulkanComputeCullingDispatch();
    for (int length : new int[] {0, 6, 16, 23, 25}) {
      assertThrows(IllegalArgumentException.class,
          () -> dispatch.dispatch(new MockVkCommandBuffer(7L), 1, new float[length]), "length " + length);
    }
    assertThrows(IllegalArgumentException.class,
        () -> dispatch.dispatch(new MockVkCommandBuffer(7L), -1, new float[24]));
  }
}
