package org.dynamisengine.gpu.api.gpu;

import java.util.Objects;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsResource;

/**
 * Describes one meshlet visibility compaction work unit.
 */
public final class MeshletVisibilityCompactionWork {
  private final GpuMeshletVisibilityFlagsResource flagsResource;
  private final int meshletCount;

  public MeshletVisibilityCompactionWork(GpuMeshletVisibilityFlagsResource flagsResource, int meshletCount) {
    this.flagsResource = Objects.requireNonNull(flagsResource, "flagsResource");
    if (meshletCount < 0) {
      throw new IllegalArgumentException("meshletCount must be >= 0");
    }
    if (meshletCount > flagsResource.meshletCount()) {
      throw new IllegalArgumentException(
          "meshletCount exceeds flags resource meshlet count: requested="
              + meshletCount
              + " resource="
              + flagsResource.meshletCount());
    }
    this.meshletCount = meshletCount;
  }

  public static MeshletVisibilityCompactionWork forAllMeshlets(GpuMeshletVisibilityFlagsResource flagsResource) {
    Objects.requireNonNull(flagsResource, "flagsResource");
    return new MeshletVisibilityCompactionWork(flagsResource, flagsResource.meshletCount());
  }

  public GpuMeshletVisibilityFlagsResource flagsResource() {
    return flagsResource;
  }

  public int meshletCount() {
    return meshletCount;
  }
}

