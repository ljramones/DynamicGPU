package org.dynamisengine.gpu.api.gpu;

/**
 * Dispatches compute-driven culling work.
 */
public interface ComputeCullingDispatch {
  /**
   * Enqueues a culling compute dispatch.
   *
   * @param cmd command buffer receiving dispatch commands
   * @param meshCount number of meshes/entries to process
   * @param frustumPlanes6 six clipping planes, typically 6 x vec4 values packed in float array
   */
  void dispatch(VkCommandBuffer cmd, int meshCount, float[] frustumPlanes6);

  /**
   * Releases backend resources owned by this dispatcher.
   */
  void destroy();
}
