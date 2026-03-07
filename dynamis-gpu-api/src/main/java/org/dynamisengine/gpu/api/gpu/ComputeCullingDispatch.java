package org.dynamisengine.gpu.api.gpu;

public interface ComputeCullingDispatch {
  void dispatch(VkCommandBuffer cmd, int meshCount, float[] frustumPlanes6);

  void destroy();
}
