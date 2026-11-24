package org.dynamisengine.gpu.test.mock;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.dynamisengine.gpu.api.gpu.ComputeCullingDispatch;
import org.dynamisengine.gpu.api.gpu.VkCommandBuffer;

public final class MockComputeCullingDispatch implements ComputeCullingDispatch {
  private final List<DispatchCall> calls = new ArrayList<>();
  private boolean destroyed;

  @Override
  public void dispatch(VkCommandBuffer cmd, int meshCount, float[] frustumPlanes) {
    if (frustumPlanes.length != FRUSTUM_PLANE_FLOATS) {
      throw new IllegalArgumentException("frustumPlanes must contain exactly " + FRUSTUM_PLANE_FLOATS + " values");
    }
    calls.add(new DispatchCall(cmd.handle(), meshCount, Arrays.copyOf(frustumPlanes, frustumPlanes.length)));
  }

  @Override
  public void destroy() {
    destroyed = true;
  }

  public List<DispatchCall> calls() {
    return List.copyOf(calls);
  }

  public boolean destroyed() {
    return destroyed;
  }

  public record DispatchCall(long commandBufferHandle, int meshCount, float[] frustumPlanes) {}
}
