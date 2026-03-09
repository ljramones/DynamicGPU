package org.dynamisengine.gpu.api.gpu;

import java.util.Objects;
import org.dynamisengine.gpu.api.resource.GpuMeshletDrawMetadataPayload;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListResource;

/**
 * Describes one indirect draw generation work unit from compact visible meshlet indices.
 */
public final class MeshletIndirectDrawGenerationWork {
  private final GpuVisibleMeshletListResource visibleMeshlets;
  private final GpuMeshletDrawMetadataPayload drawMetadata;
  private final int commandCount;

  public MeshletIndirectDrawGenerationWork(
      GpuVisibleMeshletListResource visibleMeshlets,
      GpuMeshletDrawMetadataPayload drawMetadata,
      int commandCount) {
    this.visibleMeshlets = Objects.requireNonNull(visibleMeshlets, "visibleMeshlets");
    this.drawMetadata = Objects.requireNonNull(drawMetadata, "drawMetadata");
    if (commandCount < 0) {
      throw new IllegalArgumentException("commandCount must be >= 0");
    }
    if (commandCount > visibleMeshlets.visibleMeshletCount()) {
      throw new IllegalArgumentException(
          "commandCount exceeds visibleMeshletCount: requested="
              + commandCount
              + " visible="
              + visibleMeshlets.visibleMeshletCount());
    }
    if (visibleMeshlets.sourceMeshletCount() > drawMetadata.meshletCount()) {
      throw new IllegalArgumentException(
          "draw metadata meshlet count is smaller than source meshlet count: metadata="
              + drawMetadata.meshletCount()
              + " source="
              + visibleMeshlets.sourceMeshletCount());
    }
    this.commandCount = commandCount;
  }

  public static MeshletIndirectDrawGenerationWork forAllVisibleMeshlets(
      GpuVisibleMeshletListResource visibleMeshlets, GpuMeshletDrawMetadataPayload drawMetadata) {
    Objects.requireNonNull(visibleMeshlets, "visibleMeshlets");
    return new MeshletIndirectDrawGenerationWork(
        visibleMeshlets, drawMetadata, visibleMeshlets.visibleMeshletCount());
  }

  public GpuVisibleMeshletListResource visibleMeshlets() {
    return visibleMeshlets;
  }

  public GpuMeshletDrawMetadataPayload drawMetadata() {
    return drawMetadata;
  }

  public int commandCount() {
    return commandCount;
  }
}

