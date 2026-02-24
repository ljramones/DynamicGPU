package org.dynamisgpu.test.mock;

import java.util.ArrayList;
import java.util.List;
import org.dynamisgpu.api.gpu.InstanceDataBuffer;

public final class MockInstanceDataBuffer implements InstanceDataBuffer {
  private final long bufferHandle;
  private final List<UploadCall> uploads = new ArrayList<>();
  private int instanceCount;
  private boolean destroyed;

  public MockInstanceDataBuffer(long bufferHandle) {
    this.bufferHandle = bufferHandle;
  }

  @Override
  public void upload(float[][] modelMatrices, int[] materialIndices, int[] flags) {
    uploads.add(new UploadCall(modelMatrices.length, materialIndices.length, flags.length));
    instanceCount = modelMatrices.length;
  }

  @Override
  public long bufferHandle() {
    return bufferHandle;
  }

  @Override
  public int instanceCount() {
    return instanceCount;
  }

  @Override
  public void destroy() {
    destroyed = true;
  }

  public List<UploadCall> uploads() {
    return List.copyOf(uploads);
  }

  public boolean destroyed() {
    return destroyed;
  }

  public record UploadCall(int modelMatricesCount, int materialIndicesCount, int flagsCount) {}
}
