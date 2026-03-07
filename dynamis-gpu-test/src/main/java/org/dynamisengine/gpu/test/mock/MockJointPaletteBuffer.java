package org.dynamisengine.gpu.test.mock;

import java.util.ArrayList;
import java.util.List;
import org.dynamisengine.gpu.api.gpu.JointPaletteBuffer;

public final class MockJointPaletteBuffer implements JointPaletteBuffer {
  private final long bufferHandle;
  private final List<float[]> uploads = new ArrayList<>();
  private int jointCount;
  private boolean destroyed;

  public MockJointPaletteBuffer(long bufferHandle) {
    this.bufferHandle = bufferHandle;
  }

  @Override
  public void upload(float[] jointMatrices) {
    uploads.add(jointMatrices.clone());
    jointCount = jointMatrices.length / 16;
  }

  @Override
  public long bufferHandle() {
    return bufferHandle;
  }

  @Override
  public int jointCount() {
    return jointCount;
  }

  @Override
  public void destroy() {
    destroyed = true;
  }

  public List<float[]> uploads() {
    return List.copyOf(uploads);
  }

  public boolean destroyed() {
    return destroyed;
  }
}
