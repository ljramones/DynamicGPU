package org.dynamisengine.gpu.api.resource;

import java.util.Arrays;

/**
 * Per-meshlet draw metadata used to generate indexed indirect commands.
 */
public final class GpuMeshletDrawMetadataPayload {
  private final int meshletCount;
  private final int[] indexCounts;
  private final int[] firstIndices;
  private final int[] vertexOffsets;

  private GpuMeshletDrawMetadataPayload(
      int meshletCount, int[] indexCounts, int[] firstIndices, int[] vertexOffsets) {
    this.meshletCount = meshletCount;
    this.indexCounts = indexCounts;
    this.firstIndices = firstIndices;
    this.vertexOffsets = vertexOffsets;
  }

  public static GpuMeshletDrawMetadataPayload of(
      int[] indexCounts, int[] firstIndices, int[] vertexOffsets) {
    if (indexCounts == null) {
      throw new NullPointerException("indexCounts");
    }
    if (firstIndices == null) {
      throw new NullPointerException("firstIndices");
    }
    if (vertexOffsets == null) {
      throw new NullPointerException("vertexOffsets");
    }
    int meshletCount = indexCounts.length;
    if (firstIndices.length != meshletCount || vertexOffsets.length != meshletCount) {
      throw new IllegalArgumentException("all metadata arrays must have identical length");
    }
    int[] indexCountsCopy = Arrays.copyOf(indexCounts, meshletCount);
    int[] firstIndicesCopy = Arrays.copyOf(firstIndices, meshletCount);
    int[] vertexOffsetsCopy = Arrays.copyOf(vertexOffsets, meshletCount);
    for (int i = 0; i < meshletCount; i++) {
      if (indexCountsCopy[i] < 0) {
        throw new IllegalArgumentException("indexCounts[" + i + "] must be >= 0");
      }
      if (firstIndicesCopy[i] < 0) {
        throw new IllegalArgumentException("firstIndices[" + i + "] must be >= 0");
      }
    }
    return new GpuMeshletDrawMetadataPayload(
        meshletCount, indexCountsCopy, firstIndicesCopy, vertexOffsetsCopy);
  }

  public int meshletCount() {
    return meshletCount;
  }

  public int indexCount(int meshletIndex) {
    return indexCounts[meshletIndex];
  }

  public int firstIndex(int meshletIndex) {
    return firstIndices[meshletIndex];
  }

  public int vertexOffset(int meshletIndex) {
    return vertexOffsets[meshletIndex];
  }
}

