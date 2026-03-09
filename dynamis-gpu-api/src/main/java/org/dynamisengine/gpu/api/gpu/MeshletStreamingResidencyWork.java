package org.dynamisengine.gpu.api.gpu;

import java.util.Objects;
import org.dynamisengine.gpu.api.resource.GpuMeshletStreamingResource;

/**
 * Describes one meshlet streaming residency work unit.
 */
public final class MeshletStreamingResidencyWork {
  private final GpuMeshletStreamingResource streamingResource;
  private final int targetStreamUnitId;

  public MeshletStreamingResidencyWork(GpuMeshletStreamingResource streamingResource, int targetStreamUnitId) {
    this.streamingResource = Objects.requireNonNull(streamingResource, "streamingResource");
    if (targetStreamUnitId < 0) {
      throw new IllegalArgumentException("targetStreamUnitId must be >= 0");
    }
    this.targetStreamUnitId = targetStreamUnitId;
  }

  public static MeshletStreamingResidencyWork forTargetStreamUnit(
      GpuMeshletStreamingResource streamingResource, int targetStreamUnitId) {
    return new MeshletStreamingResidencyWork(streamingResource, targetStreamUnitId);
  }

  public GpuMeshletStreamingResource streamingResource() {
    return streamingResource;
  }

  public int targetStreamUnitId() {
    return targetStreamUnitId;
  }
}

