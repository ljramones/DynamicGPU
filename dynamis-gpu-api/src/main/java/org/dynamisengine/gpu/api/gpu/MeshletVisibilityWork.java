package org.dynamisengine.gpu.api.gpu;

import java.util.Objects;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsResource;

/**
 * Describes one meshlet visibility compute work unit.
 */
public final class MeshletVisibilityWork {
  private final GpuMeshletBoundsResource boundsResource;
  private final int meshletCount;
  private final MeshletVisibilityFrustum frustum;

  public MeshletVisibilityWork(
      GpuMeshletBoundsResource boundsResource, int meshletCount, MeshletVisibilityFrustum frustum) {
    this.boundsResource = Objects.requireNonNull(boundsResource, "boundsResource");
    this.frustum = Objects.requireNonNull(frustum, "frustum");
    if (meshletCount < 0) {
      throw new IllegalArgumentException("meshletCount must be >= 0");
    }
    if (meshletCount > boundsResource.meshletCount()) {
      throw new IllegalArgumentException(
          "meshletCount exceeds bounds resource meshlet count: requested="
              + meshletCount
              + " resource="
              + boundsResource.meshletCount());
    }
    this.meshletCount = meshletCount;
  }

  public static MeshletVisibilityWork forAllMeshlets(
      GpuMeshletBoundsResource boundsResource, MeshletVisibilityFrustum frustum) {
    Objects.requireNonNull(boundsResource, "boundsResource");
    return new MeshletVisibilityWork(boundsResource, boundsResource.meshletCount(), frustum);
  }

  public GpuMeshletBoundsResource boundsResource() {
    return boundsResource;
  }

  public int meshletCount() {
    return meshletCount;
  }

  public MeshletVisibilityFrustum frustum() {
    return frustum;
  }
}

