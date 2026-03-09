package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * Immutable RT geometry build-input payload contract for BLAS-ready geometry linkage.
 *
 * <p>This contract extends metadata-only RT regions with explicit vertex/index buffer linkage and
 * basic build-range assumptions required by backend BLAS setup.
 */
public final class GpuRayTracingBuildInputPayload {
  private final GpuRayTracingGeometryResource geometryResource;
  private final GpuBufferHandle vertexBufferHandle;
  private final GpuBufferHandle indexBufferHandle;
  private final int vertexStrideBytes;
  private final int maxVertexIndex;
  private final long vertexDataOffsetBytes;
  private final long indexDataOffsetBytes;

  private GpuRayTracingBuildInputPayload(
      GpuRayTracingGeometryResource geometryResource,
      GpuBufferHandle vertexBufferHandle,
      GpuBufferHandle indexBufferHandle,
      int vertexStrideBytes,
      int maxVertexIndex,
      long vertexDataOffsetBytes,
      long indexDataOffsetBytes) {
    this.geometryResource = geometryResource;
    this.vertexBufferHandle = vertexBufferHandle;
    this.indexBufferHandle = indexBufferHandle;
    this.vertexStrideBytes = vertexStrideBytes;
    this.maxVertexIndex = maxVertexIndex;
    this.vertexDataOffsetBytes = vertexDataOffsetBytes;
    this.indexDataOffsetBytes = indexDataOffsetBytes;
  }

  public static GpuRayTracingBuildInputPayload of(
      GpuRayTracingGeometryResource geometryResource,
      GpuBufferHandle vertexBufferHandle,
      GpuBufferHandle indexBufferHandle,
      int vertexStrideBytes,
      int maxVertexIndex,
      long vertexDataOffsetBytes,
      long indexDataOffsetBytes) {
    Objects.requireNonNull(geometryResource, "geometryResource");
    Objects.requireNonNull(vertexBufferHandle, "vertexBufferHandle");
    Objects.requireNonNull(indexBufferHandle, "indexBufferHandle");
    if (geometryResource.isClosed()) {
      throw new IllegalArgumentException("geometryResource is already closed");
    }
    if (geometryResource.regionCount() <= 0) {
      throw new IllegalArgumentException("geometryResource must contain at least one region");
    }
    if (vertexStrideBytes <= 0) {
      throw new IllegalArgumentException("vertexStrideBytes must be > 0");
    }
    if ((vertexStrideBytes % Integer.BYTES) != 0) {
      throw new IllegalArgumentException("vertexStrideBytes must be int-aligned");
    }
    if (maxVertexIndex < 0) {
      throw new IllegalArgumentException("maxVertexIndex must be >= 0");
    }
    if (vertexDataOffsetBytes < 0) {
      throw new IllegalArgumentException("vertexDataOffsetBytes must be >= 0");
    }
    if (indexDataOffsetBytes < 0) {
      throw new IllegalArgumentException("indexDataOffsetBytes must be >= 0");
    }
    return new GpuRayTracingBuildInputPayload(
        geometryResource,
        vertexBufferHandle,
        indexBufferHandle,
        vertexStrideBytes,
        maxVertexIndex,
        vertexDataOffsetBytes,
        indexDataOffsetBytes);
  }

  public GpuRayTracingGeometryResource geometryResource() {
    return geometryResource;
  }

  public GpuBufferHandle vertexBufferHandle() {
    return vertexBufferHandle;
  }

  public GpuBufferHandle indexBufferHandle() {
    return indexBufferHandle;
  }

  public int vertexStrideBytes() {
    return vertexStrideBytes;
  }

  public int maxVertexIndex() {
    return maxVertexIndex;
  }

  public long vertexDataOffsetBytes() {
    return vertexDataOffsetBytes;
  }

  public long indexDataOffsetBytes() {
    return indexDataOffsetBytes;
  }

  public int regionCount() {
    return geometryResource.regionCount();
  }
}

