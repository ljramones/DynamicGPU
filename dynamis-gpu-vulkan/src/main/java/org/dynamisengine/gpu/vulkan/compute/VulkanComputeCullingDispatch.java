package org.dynamisengine.gpu.vulkan.compute;

import java.util.Arrays;
import java.util.Objects;
import org.dynamisengine.gpu.api.gpu.ComputeCullingDispatch;
import org.dynamisengine.gpu.api.gpu.VkCommandBuffer;

/**
 * Simple Vulkan compute-dispatch adapter that captures last submitted values.
 */
public final class VulkanComputeCullingDispatch implements ComputeCullingDispatch {
  private boolean destroyed;
  private long lastCommandBufferHandle;
  private int lastMeshCount;
  private float[] lastFrustumPlanes;

  @Override
  public void dispatch(VkCommandBuffer cmd, int meshCount, float[] frustumPlanes) {
    ensureAlive();
    Objects.requireNonNull(cmd, "cmd");
    Objects.requireNonNull(frustumPlanes, "frustumPlanes");
    if (frustumPlanes.length != FRUSTUM_PLANE_FLOATS) {
      throw new IllegalArgumentException("frustumPlanes must contain exactly " + FRUSTUM_PLANE_FLOATS
          + " values (6 planes of a, b, c, d), got " + frustumPlanes.length);
    }
    if (meshCount < 0) {
      throw new IllegalArgumentException("meshCount must be >= 0");
    }
    lastCommandBufferHandle = cmd.handle();
    lastMeshCount = meshCount;
    lastFrustumPlanes = Arrays.copyOf(frustumPlanes, frustumPlanes.length);
  }

  @Override
  public void destroy() {
    destroyed = true;
    lastFrustumPlanes = null;
  }

  /**
   * @return handle of the last command buffer used for dispatch
   */
  public long lastCommandBufferHandle() {
    return lastCommandBufferHandle;
  }

  /**
   * @return mesh count from the most recent dispatch call
   */
  public int lastMeshCount() {
    return lastMeshCount;
  }

  /**
   * @return defensive copy of the most recent frustum payload
   */
  public float[] lastFrustumPlanes() {
    return lastFrustumPlanes == null ? null : Arrays.copyOf(lastFrustumPlanes, lastFrustumPlanes.length);
  }

  private void ensureAlive() {
    if (destroyed) {
      throw new IllegalStateException("Dispatch has been destroyed");
    }
  }
}
