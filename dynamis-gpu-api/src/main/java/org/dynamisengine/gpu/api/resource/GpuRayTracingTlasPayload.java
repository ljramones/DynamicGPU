package org.dynamisengine.gpu.api.resource;

/**
 * Immutable TLAS output payload metadata.
 */
public final class GpuRayTracingTlasPayload {
  private final int instanceCount;
  private final int instanceByteSize;

  private GpuRayTracingTlasPayload(int instanceCount, int instanceByteSize) {
    this.instanceCount = instanceCount;
    this.instanceByteSize = instanceByteSize;
  }

  public static GpuRayTracingTlasPayload of(int instanceCount, int instanceByteSize) {
    if (instanceCount <= 0) {
      throw new IllegalArgumentException("instanceCount must be > 0");
    }
    if (instanceByteSize <= 0) {
      throw new IllegalArgumentException("instanceByteSize must be > 0");
    }
    return new GpuRayTracingTlasPayload(instanceCount, instanceByteSize);
  }

  public int instanceCount() {
    return instanceCount;
  }

  public int instanceByteSize() {
    return instanceByteSize;
  }
}

