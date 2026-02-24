package org.dynamisgpu.vulkan.compute;

import java.util.Arrays;
import java.util.Objects;
import org.dynamisgpu.api.gpu.ComputeCullingDispatch;
import org.dynamisgpu.api.gpu.VkCommandBuffer;

public final class VulkanComputeCullingDispatch implements ComputeCullingDispatch {
  private boolean destroyed;
  private long lastCommandBufferHandle;
  private int lastMeshCount;
  private float[] lastFrustumPlanes;

  @Override
  public void dispatch(VkCommandBuffer cmd, int meshCount, float[] frustumPlanes6) {
    ensureAlive();
    Objects.requireNonNull(cmd, "cmd");
    Objects.requireNonNull(frustumPlanes6, "frustumPlanes6");
    if (frustumPlanes6.length != 6) {
      throw new IllegalArgumentException("frustumPlanes6 must contain exactly 6 values");
    }
    if (meshCount < 0) {
      throw new IllegalArgumentException("meshCount must be >= 0");
    }
    lastCommandBufferHandle = cmd.handle();
    lastMeshCount = meshCount;
    lastFrustumPlanes = Arrays.copyOf(frustumPlanes6, frustumPlanes6.length);
  }

  @Override
  public void destroy() {
    destroyed = true;
    lastFrustumPlanes = null;
  }

  public long lastCommandBufferHandle() {
    return lastCommandBufferHandle;
  }

  public int lastMeshCount() {
    return lastMeshCount;
  }

  public float[] lastFrustumPlanes() {
    return lastFrustumPlanes == null ? null : Arrays.copyOf(lastFrustumPlanes, lastFrustumPlanes.length);
  }

  private void ensureAlive() {
    if (destroyed) {
      throw new IllegalStateException("Dispatch has been destroyed");
    }
  }
}
