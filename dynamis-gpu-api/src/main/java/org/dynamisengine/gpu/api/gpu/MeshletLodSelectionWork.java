package org.dynamisengine.gpu.api.gpu;

import java.util.Objects;
import org.dynamisengine.gpu.api.resource.GpuMeshletLodResource;

/**
 * Describes one meshlet LOD selection work unit.
 */
public final class MeshletLodSelectionWork {
  private final GpuMeshletLodResource lodResource;
  private final int targetLodLevel;

  public MeshletLodSelectionWork(GpuMeshletLodResource lodResource, int targetLodLevel) {
    this.lodResource = Objects.requireNonNull(lodResource, "lodResource");
    if (targetLodLevel < 0) {
      throw new IllegalArgumentException("targetLodLevel must be >= 0");
    }
    this.targetLodLevel = targetLodLevel;
  }

  public static MeshletLodSelectionWork forTargetLevel(GpuMeshletLodResource lodResource, int targetLodLevel) {
    return new MeshletLodSelectionWork(lodResource, targetLodLevel);
  }

  public GpuMeshletLodResource lodResource() {
    return lodResource;
  }

  public int targetLodLevel() {
    return targetLodLevel;
  }
}
