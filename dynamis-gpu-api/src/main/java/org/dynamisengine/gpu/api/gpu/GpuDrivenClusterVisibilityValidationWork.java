package org.dynamisengine.gpu.api.gpu;

import java.util.Objects;
import org.dynamisengine.gpu.api.resource.GpuMeshletBoundsResource;
import org.dynamisengine.gpu.api.resource.GpuMeshletDrawMetadataPayload;

/**
 * End-to-end validation work input for the GPU-driven cluster visibility chain.
 */
public final class GpuDrivenClusterVisibilityValidationWork {
  private final GpuMeshletBoundsResource boundsResource;
  private final MeshletVisibilityFrustum frustum;
  private final GpuMeshletDrawMetadataPayload drawMetadata;
  private final int meshletCount;

  public GpuDrivenClusterVisibilityValidationWork(
      GpuMeshletBoundsResource boundsResource,
      MeshletVisibilityFrustum frustum,
      GpuMeshletDrawMetadataPayload drawMetadata,
      int meshletCount) {
    this.boundsResource = Objects.requireNonNull(boundsResource, "boundsResource");
    this.frustum = Objects.requireNonNull(frustum, "frustum");
    this.drawMetadata = Objects.requireNonNull(drawMetadata, "drawMetadata");
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
    if (drawMetadata.meshletCount() < meshletCount) {
      throw new IllegalArgumentException(
          "draw metadata meshlet count is smaller than meshletCount: metadata="
              + drawMetadata.meshletCount()
              + " meshletCount="
              + meshletCount);
    }
    this.meshletCount = meshletCount;
  }

  public static GpuDrivenClusterVisibilityValidationWork forAllMeshlets(
      GpuMeshletBoundsResource boundsResource,
      MeshletVisibilityFrustum frustum,
      GpuMeshletDrawMetadataPayload drawMetadata) {
    Objects.requireNonNull(boundsResource, "boundsResource");
    return new GpuDrivenClusterVisibilityValidationWork(
        boundsResource, frustum, drawMetadata, boundsResource.meshletCount());
  }

  public GpuMeshletBoundsResource boundsResource() {
    return boundsResource;
  }

  public MeshletVisibilityFrustum frustum() {
    return frustum;
  }

  public GpuMeshletDrawMetadataPayload drawMetadata() {
    return drawMetadata;
  }

  public int meshletCount() {
    return meshletCount;
  }
}

