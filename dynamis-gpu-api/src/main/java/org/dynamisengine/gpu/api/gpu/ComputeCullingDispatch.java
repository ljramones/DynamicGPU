package org.dynamisengine.gpu.api.gpu;

/**
 * Dispatches compute-driven culling work.
 */
public interface ComputeCullingDispatch {
  /**
   * The length of the frustum argument: six planes of four floats {@code (a, b, c, d)}, the plane
   * {@code a·x + b·y + c·z + d = 0} with its normal pointing into the frustum, in the order left, right, bottom, top,
   * near, far. This is the layout of Vectrix {@code FrustumPlanes.getVec4Array}.
   */
  int FRUSTUM_PLANE_FLOATS = 24;

  /**
   * Enqueues a culling compute dispatch.
   *
   * @param cmd command buffer receiving dispatch commands
   * @param meshCount number of meshes/entries to process
   * @param frustumPlanes {@link #FRUSTUM_PLANE_FLOATS} floats: six planes of {@code (a, b, c, d)}
   * @throws IllegalArgumentException if {@code frustumPlanes} does not hold exactly {@link #FRUSTUM_PLANE_FLOATS}
   *     values or {@code meshCount} is negative
   */
  void dispatch(VkCommandBuffer cmd, int meshCount, float[] frustumPlanes);

  /**
   * Releases backend resources owned by this dispatcher.
   */
  void destroy();
}
