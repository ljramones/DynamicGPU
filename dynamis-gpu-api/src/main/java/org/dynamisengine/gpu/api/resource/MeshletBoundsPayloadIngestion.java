package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.util.Objects;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * Ingestion seam for finalized upstream meshlet-bounds payload contracts.
 *
 * <p>This utility consumes upstream metadata + bytes as authoritative and creates
 * DynamisGPU payload/resource models without redefining layout.
 */
public final class MeshletBoundsPayloadIngestion {
  private MeshletBoundsPayloadIngestion() {}

  public static GpuMeshletBoundsPayload ingest(
      int meshletCount,
      int boundsOffsetFloats,
      int boundsStrideFloats,
      int boundsFloatCount,
      int expectedBoundsFloatCount,
      ByteBuffer boundsBytes) {
    if (boundsFloatCount != expectedBoundsFloatCount) {
      throw new IllegalArgumentException(
          "upstream bounds float count mismatch: count="
              + boundsFloatCount
              + " expected="
              + expectedBoundsFloatCount);
    }
    GpuMeshletBoundsPayload payload =
        GpuMeshletBoundsPayload.fromLittleEndianBytes(
            meshletCount, boundsOffsetFloats, boundsStrideFloats, boundsBytes);
    if (payload.boundsFloatCount() != boundsFloatCount) {
      throw new IllegalArgumentException(
          "payload float count mismatch after ingestion: expected="
              + boundsFloatCount
              + " actual="
              + payload.boundsFloatCount());
    }
    return payload;
  }

  public static GpuMeshletBoundsResource toResource(
      GpuBufferHandle bufferHandle, GpuMeshletBoundsPayload payload) {
    Objects.requireNonNull(bufferHandle, "bufferHandle");
    Objects.requireNonNull(payload, "payload");
    return new GpuMeshletBoundsResource(bufferHandle, payload);
  }
}

