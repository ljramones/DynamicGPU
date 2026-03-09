package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * Ingestion seam for BLAS build-input-ready RT geometry payload contracts.
 */
public final class RayTracingBuildInputIngestion {
  private RayTracingBuildInputIngestion() {}

  public static GpuRayTracingBuildInputPayload ingest(
      GpuRayTracingGeometryResource geometryResource,
      GpuBufferHandle vertexBufferHandle,
      GpuBufferHandle indexBufferHandle,
      int vertexStrideBytes,
      int maxVertexIndex,
      long vertexDataOffsetBytes,
      long indexDataOffsetBytes) {
    Objects.requireNonNull(geometryResource, "geometryResource");
    validateRegionIndexRanges(geometryResource.payload());
    return GpuRayTracingBuildInputPayload.of(
        geometryResource,
        vertexBufferHandle,
        indexBufferHandle,
        vertexStrideBytes,
        maxVertexIndex,
        vertexDataOffsetBytes,
        indexDataOffsetBytes);
  }

  public static GpuRayTracingBuildInputResource toResolvedResource(
      GpuRayTracingBuildInputPayload payload,
      long vertexBufferDeviceAddress,
      long indexBufferDeviceAddress) {
    return new GpuRayTracingBuildInputResource(
        Objects.requireNonNull(payload, "payload"),
        vertexBufferDeviceAddress,
        indexBufferDeviceAddress);
  }

  private static void validateRegionIndexRanges(GpuRayTracingGeometryPayload payload) {
    ByteBuffer regions = payload.regionsBytes().order(ByteOrder.LITTLE_ENDIAN);
    int offsetInts = payload.regionsOffsetInts();
    int strideInts = payload.regionsStrideInts();
    for (int i = 0; i < payload.regionCount(); i++) {
      int baseInt = offsetInts + (i * strideInts);
      int baseByte = baseInt * Integer.BYTES;
      int firstIndex = regions.getInt(baseByte + Integer.BYTES);
      int indexCount = regions.getInt(baseByte + (2 * Integer.BYTES));
      if (firstIndex < 0) {
        throw new IllegalArgumentException("region firstIndex must be >= 0: region=" + i);
      }
      if (indexCount <= 0) {
        throw new IllegalArgumentException("region indexCount must be > 0: region=" + i);
      }
    }
  }
}

