package org.dynamisgpu.test.mock;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.dynamisgpu.api.gpu.ComputeCullingDispatch;
import org.dynamisgpu.api.gpu.VkCommandBuffer;

public final class MockComputeCullingDispatch implements ComputeCullingDispatch {
  private final List<DispatchCall> calls = new ArrayList<>();
  private boolean destroyed;

  @Override
  public void dispatch(VkCommandBuffer cmd, int meshCount, float[] frustumPlanes6) {
    calls.add(new DispatchCall(cmd.handle(), meshCount, Arrays.copyOf(frustumPlanes6, frustumPlanes6.length)));
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

  public record DispatchCall(long commandBufferHandle, int meshCount, float[] frustumPlanes6) {}
}
